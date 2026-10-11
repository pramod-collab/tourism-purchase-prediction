from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / 'tourism_project'
DATA = PROJECT / 'data'
SPLITS = PROJECT / 'splits'
REPORTS = PROJECT / 'reports'
DEPLOY = PROJECT / 'deployment'
TARGET = 'ProdTaken'
SEED = 42
EXPECTED = ['CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier',
            'DurationOfPitch', 'Occupation', 'Gender', 'NumberOfPersonVisiting',
            'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar',
            'MaritalStatus', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
            'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome']
# These fields describe the current contact or events after contact.
AFTER_CONTACT = ['TypeofContact', 'DurationOfPitch', 'NumberOfFollowups',
                 'ProductPitched', 'PitchSatisfactionScore']
CATEGORICAL = ['CityTier', 'Occupation', 'Gender', 'MaritalStatus', 'Designation']
NUMERIC = ['Age', 'NumberOfPersonVisiting', 'PreferredPropertyStar',
           'NumberOfTrips', 'Passport', 'OwnCar', 'NumberOfChildrenVisiting',
           'MonthlyIncome']
FEATURES = NUMERIC + CATEGORICAL

def clean_data(raw):
    data = raw.copy()
    for col in data.select_dtypes('object'):
        data[col] = data[col].str.strip()
    data['Gender'] = data['Gender'].replace({'Fe Male': 'Female'})
    # Keep Single and Unmarried separate: the dictionary does not say they are identical.
    return data[FEATURES + [TARGET]]

def profile_groups(features):
    # Identical available profiles stay in the same partition, even with different IDs.
    return pd.util.hash_pandas_object(features, index=False).astype(str)
