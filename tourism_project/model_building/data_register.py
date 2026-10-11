import hashlib
import json
import pandas as pd
from tourism_project.model_building.common import DATA, REPORTS, EXPECTED

def register():
    path = DATA / 'tourism.csv'
    df = pd.read_csv(path)
    missing = sorted(set(EXPECTED) - set(df.columns))
    extra = [c for c in df.columns if c not in EXPECTED and not c.startswith('Unnamed:')]
    if missing or extra:
        raise ValueError(f'Invalid schema. Missing: {missing}; unexpected: {extra}')
    if df.empty or df.CustomerID.isna().any() or df.CustomerID.duplicated().any():
        raise ValueError('The file must contain rows and unique, nonmissing customer IDs.')
    if df.ProdTaken.isna().any() or set(df.ProdTaken.unique()) != {0, 1}:
        raise ValueError('ProdTaken must contain both 0 and 1, without missing labels.')
    numeric = [c for c in EXPECTED if c not in
               ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched',
                'MaritalStatus', 'Designation']]
    for col in numeric:
        pd.to_numeric(df[col], errors='raise')
    for col in ['Passport', 'OwnCar']:
        if not df[col].dropna().isin([0, 1]).all():
            raise ValueError(f'{col} must be binary.')
    summary = {'rows': len(df), 'columns': len(df.columns),
               'buyers': int(df.ProdTaken.sum()),
               'buyer_rate': float(df.ProdTaken.mean()),
               'missing_values': int(df.isna().sum().sum()),
               'duplicate_rows': int(df.duplicated().sum()),
               'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / 'data_registration.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary

if __name__ == '__main__':
    register()
