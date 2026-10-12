import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from tourism_project.model_building.common import ROOT, SPLITS, DEPLOY, profile_groups

def test_no_profile_leakage():
    train = pd.read_csv(SPLITS / 'Xtrain.csv')
    test = pd.read_csv(SPLITS / 'Xtest.csv')
    assert not set(profile_groups(train)) & set(profile_groups(test))
    assert 'CustomerID' not in train
    assert 'DurationOfPitch' not in train

def test_saved_model_handles_missing_and_unseen_values():
    model = joblib.load(DEPLOY / 'model.joblib')
    frame = pd.read_csv(SPLITS / 'Xtest.csv').head(2)
    expected = model.predict_proba(frame)
    assert np.allclose(expected.sum(axis=1), 1)
    frame.loc[frame.index[0], 'Age'] = np.nan
    frame.loc[frame.index[1], 'Occupation'] = 'New occupation'
    result = model.predict_proba(frame)
    assert np.isfinite(result).all()
    metadata = json.loads((DEPLOY / 'metadata.json').read_text())
    assert list(model.feature_names_in_) == metadata['features']

def test_app_prediction_and_invalid_party():
    app = AppTest.from_file(str(DEPLOY / 'app.py')).run(timeout=60)
    assert not app.exception
    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert len(app.metric) == 1
    app.number_input[3].set_value(10)
    app.button[0].click().run(timeout=60)
    assert len(app.error) == 1
