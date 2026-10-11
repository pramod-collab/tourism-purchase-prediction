import json
import platform
import hashlib
import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    fbeta_score, roc_auc_score, average_precision_score, confusion_matrix,
    classification_report)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from tourism_project.model_building.common import (
    ROOT, DATA, SPLITS, REPORTS, DEPLOY, NUMERIC, CATEGORICAL, SEED, profile_groups)

def evaluate(y, probability, threshold):
    pred = (probability >= threshold).astype(int)
    return {'accuracy': float(accuracy_score(y, pred)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred)),
            'f1': float(f1_score(y, pred, zero_division=0)),
            'f2': float(fbeta_score(y, pred, beta=2, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, probability)),
            'average_precision': float(average_precision_score(y, probability))}

def train():
    Xtrain = pd.read_csv(SPLITS / 'Xtrain.csv')
    ytrain = pd.read_csv(SPLITS / 'ytrain.csv').iloc[:, 0]
    Xtest = pd.read_csv(SPLITS / 'Xtest.csv')
    ytest = pd.read_csv(SPLITS / 'ytest.csv').iloc[:, 0]
    preprocessor = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), NUMERIC),
        ('category', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), CATEGORICAL)
    ])
    # Fit imputation and encoding inside each fold, not before cross-validation.
    from sklearn.tree import DecisionTreeClassifier
    candidates = {
        'Decision Tree': (DecisionTreeClassifier(random_state=SEED), {
            'model__max_depth': [4, 8, None],
            'model__min_samples_leaf': [2, 10],
            'model__class_weight': [None, 'balanced']}),
        'Random Forest': (RandomForestClassifier(random_state=SEED, n_jobs=1), {
            'model__n_estimators': [150], 'model__max_depth': [10, None],
            'model__min_samples_leaf': [1, 3],
            'model__class_weight': [None, 'balanced']}),
        'Gradient Boosting': (GradientBoostingClassifier(random_state=SEED), {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.05, 0.1], 'model__max_depth': [2, 3]})
    }
    groups = profile_groups(Xtrain)
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True,
                 random_state=SEED).split(Xtrain, ytrain, groups))
    tracking = ROOT / 'mlruns'
    mlflow.set_tracking_uri(tracking.as_uri())
    mlflow.set_experiment('Tourism pre-contact purchase prediction')
    REPORTS.mkdir(parents=True, exist_ok=True)
    DEPLOY.mkdir(parents=True, exist_ok=True)
    rows, best_score, best_search, best_name = [], -1, None, None
    with mlflow.start_run(run_name='model_comparison') as parent:
        mlflow.log_params({'seed': SEED, 'cv_folds': 3, 'selection_metric': 'average_precision',
                           'feature_count': Xtrain.shape[1], 'grouped_cv': True})
        for name, (estimator, grid) in candidates.items():
            pipeline = Pipeline([('preprocess', clone(preprocessor)), ('model', estimator)])
            search = GridSearchCV(pipeline, grid, scoring='average_precision',
                                  cv=folds, n_jobs=1, return_train_score=True, error_score='raise')
            search.fit(Xtrain, ytrain)
            results = search.cv_results_
            for i, params in enumerate(results['params']):
                row = {'model': name, 'parameters': json.dumps(params, sort_keys=True),
                       'cv_average_precision': float(results['mean_test_score'][i]),
                       'cv_std': float(results['std_test_score'][i]),
                       'train_average_precision': float(results['mean_train_score'][i])}
                rows.append(row)
                with mlflow.start_run(run_name=f'{name}_{i+1}', nested=True):
                    mlflow.log_param('model_family', name)
                    mlflow.log_params(params)
                    mlflow.log_metrics({k: row[k] for k in
                        ['cv_average_precision', 'cv_std', 'train_average_precision']})
            print(f'{name}: best cross-validation AP = {search.best_score_:.4f}', flush=True)
            if search.best_score_ > best_score:
                best_score, best_search, best_name = search.best_score_, search, name
        comparison = pd.DataFrame(rows).sort_values('cv_average_precision', ascending=False)
        comparison.to_csv(REPORTS / 'experiments.csv', index=False)
        best = best_search.best_estimator_
        # These predictions come only from training folds. The test set is not used.
        oof = cross_val_predict(clone(best), Xtrain, ytrain, cv=folds,
                               method='predict_proba', n_jobs=1)[:, 1]
        thresholds = np.round(np.arange(0.10, 0.81, 0.05), 2)
        scores = [evaluate(ytrain, oof, t) for t in thresholds]
        threshold_table = pd.DataFrame(scores)
        threshold_table.insert(0, 'threshold', thresholds)
        # F2 weights recall more than precision. Ties use the higher threshold.
        choice = threshold_table.sort_values(['f2', 'threshold'], ascending=False).iloc[0]
        threshold = float(choice['threshold'])
        threshold_table.to_csv(REPORTS / 'thresholds.csv', index=False)
        # The final model already fits all training rows. Holdout remains untouched.
        probability = best.predict_proba(Xtest)[:, 1]
        pred = (probability >= threshold).astype(int)
        metrics = evaluate(ytest, probability, threshold)
        default_metrics = evaluate(ytest, probability, 0.5)
        dummy = DummyClassifier(strategy='prior').fit(Xtrain, ytrain)
        baseline = evaluate(ytest, dummy.predict_proba(Xtest)[:, 1], 0.5)
        joblib.dump(best, DEPLOY / 'model.joblib', compress=3)
        metadata = {'model_family': best_name, 'threshold': threshold,
            'features': list(Xtrain.columns), 'best_parameters': best_search.best_params_,
            'best_cv_average_precision': best_score,
            'test_metrics': metrics, 'test_metrics_at_0_5': default_metrics,
            'baseline_metrics': baseline, 'confusion_matrix': confusion_matrix(ytest, pred).tolist(),
            'train_metrics': evaluate(ytrain, best.predict_proba(Xtrain)[:, 1], threshold),
            'training_rows': len(Xtrain), 'test_rows': len(Xtest),
            'python': platform.python_version(), 'sklearn': sklearn.__version__,
            'data_sha256': hashlib.sha256((DATA / 'tourism.csv').read_bytes()).hexdigest(),
            'mlflow_parent_run_id': parent.info.run_id}
        (DEPLOY / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        (REPORTS / 'metrics.json').write_text(json.dumps(metadata, indent=2))
        pd.DataFrame({'actual': ytest, 'probability': probability, 'prediction': pred}).to_csv(
            REPORTS / 'test_predictions.csv', index=False)
        (REPORTS / 'classification_report.txt').write_text(
            classification_report(ytest, pred, target_names=['Not purchased', 'Purchased'], zero_division=0))
        mlflow.log_params({'selected_model': best_name, 'selected_threshold': threshold,
                          **best_search.best_params_})
        mlflow.log_metrics({f'test_{k}': v for k, v in metrics.items()})
        mlflow.log_artifact(str(DEPLOY / 'model.joblib'), artifact_path='model')
        mlflow.log_artifact(str(DEPLOY / 'metadata.json'), artifact_path='model')
        mlflow.log_artifact(str(REPORTS / 'experiments.csv'))
        mlflow.log_artifact(str(REPORTS / 'thresholds.csv'))
    print('Selected model:', best_name)
    print('Threshold selected on training folds:', threshold)
    print(pd.DataFrame({'Baseline': baseline, 'Model at 0.5': default_metrics,
                        'Selected threshold': metrics}).round(4).to_string())
    print('Confusion matrix:', metadata['confusion_matrix'])
    print('MLflow parent run:', metadata['mlflow_parent_run_id'])
    return metadata

if __name__ == '__main__':
    train()
