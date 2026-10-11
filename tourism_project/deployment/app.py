from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
st.set_page_config(page_title='Tourism purchase prediction', layout='centered')

@st.cache_resource
def load_model():
    return joblib.load(HERE / 'model.joblib'), json.loads((HERE / 'metadata.json').read_text())

model, metadata = load_model()
st.title('Tourism purchase prediction')
st.write('Enter customer details available before the sales call.')
st.caption('Course project for Visit with Us. The estimate is based on past package purchases.')

with st.form('customer'):
    left, right = st.columns(2)
    with left:
        age = st.number_input('Age', 18, 100, 35)
        income = st.number_input('Monthly income', 0, 1000000, 23000, step=500)
        occupation = st.selectbox('Occupation', ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
        designation = st.selectbox('Designation', ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])
        gender = st.selectbox('Gender', ['Female', 'Male'])
        marital = st.selectbox('Marital status', ['Married', 'Single', 'Unmarried', 'Divorced'])
    with right:
        tier = st.selectbox('City tier', [1, 2, 3])
        people = st.number_input('People travelling', 1, 20, 3)
        children = st.number_input('Children under 5', 0, 20, 1)
        stars = st.selectbox('Preferred hotel stars', [3, 4, 5])
        trips = st.number_input('Trips per year', 0, 50, 3)
        passport = st.selectbox('Valid passport', ['No', 'Yes'])
        car = st.selectbox('Owns a car', ['No', 'Yes'])
    submitted = st.form_submit_button('Predict purchase')

if submitted:
    if children > people:
        st.error('Children cannot exceed the total number of people travelling.')
    else:
        customer = pd.DataFrame([{
            'Age': age, 'NumberOfPersonVisiting': people, 'PreferredPropertyStar': stars,
            'NumberOfTrips': trips, 'Passport': int(passport == 'Yes'),
            'OwnCar': int(car == 'Yes'), 'NumberOfChildrenVisiting': children,
            'MonthlyIncome': income, 'CityTier': tier, 'Occupation': occupation,
            'Gender': gender, 'MaritalStatus': marital, 'Designation': designation
        }])[metadata['features']]
        probability = float(model.predict_proba(customer)[0, 1])
        st.metric('Estimated purchase likelihood', f'{probability:.1%}')
        if probability >= metadata['threshold']:
            st.success('Suggested action: include this customer in the contact list.')
        else:
            st.info('Suggested action: lower priority for this campaign.')
        st.caption('This is a model estimate, not a confirmed purchase or a calibrated probability.')
        if age > 61 or people > 5 or trips > 22 or income < 1000 or income > 98678:
            st.warning('Some inputs are outside the ranges seen in the training data. Interpret carefully.')
        with st.expander('Entered customer details'):
            st.dataframe(customer, hide_index=True)
st.caption('Use only information already held before contact. Do not use the score to deny service.')
