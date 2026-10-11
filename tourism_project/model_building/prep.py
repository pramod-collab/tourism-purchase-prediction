import json
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tourism_project.model_building.common import (
    DATA, SPLITS, REPORTS, TARGET, SEED, clean_data, profile_groups)

def prepare():
    raw = pd.read_csv(DATA / 'tourism.csv')
    data = clean_data(raw)
    X, y = data.drop(columns=TARGET), data[TARGET].astype(int)
    groups = profile_groups(X)
    # One of five folds is held out. Grouping prevents identical profiles leaking.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    SPLITS.mkdir(parents=True, exist_ok=True)
    for name, frame in [('Xtrain', X.iloc[train_idx]), ('Xtest', X.iloc[test_idx]),
                        ('ytrain', y.iloc[train_idx]), ('ytest', y.iloc[test_idx])]:
        frame.to_csv(SPLITS / f'{name}.csv', index=False)
    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    assert not train_groups.intersection(test_groups)
    summary = {'train_rows': len(train_idx), 'test_rows': len(test_idx),
               'features': list(X.columns), 'train_buyer_rate': float(y.iloc[train_idx].mean()),
               'test_buyer_rate': float(y.iloc[test_idx].mean()),
               'repeated_profile_rows': int(X.duplicated().sum()),
               'shared_profiles': 0, 'random_state': SEED}
    (REPORTS / 'split_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary

if __name__ == '__main__':
    prepare()
