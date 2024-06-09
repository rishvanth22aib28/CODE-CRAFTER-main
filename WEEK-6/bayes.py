import pandas as pd
import numpy as np
import streamlit as st
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Sample dataset creation by CODE-CRAFTER team
def create_sample_data():
    sample_data = {
        'Fever': [1, 1, 1, 0, 0, 0],  # 1: Yes, 0: No
        'Cough': [1, 1, 0, 1, 0, 0],  # 1: Yes, 0: No
        'Fatigue': [1, 0, 1, 1, 0, 1],  # 1: Yes, 0: No
        'COVID': [1, 1, 1, 0, 0, 1]  # 1: Positive, 0: Negative
    }
    return pd.DataFrame(sample_data)

# Bayesian Network model training function
def train_bayesian_network(data):
    bayesian_model = BayesianNetwork([('Fever', 'COVID'),
                                      ('Cough', 'COVID'),
                                      ('Fatigue', 'COVID')])
    bayesian_model.fit(data, estimator=MaximumLikelihoodEstimator)
    return bayesian_model

# Main function to run Streamlit app
def main():
    st.title('CODE-CRAFTER')
    st.title('COVID-19 Diagnosis with Bayesian Network')

    st.write('### Step 1: Enter Symptoms')
    
    # User input for symptoms
    fever_input = st.selectbox('Fever', ['Yes', 'No'])
    cough_input = st.selectbox('Cough', ['Yes', 'No'])
    fatigue_input = st.selectbox('Fatigue', ['Yes', 'No'])

    # Encode the input symptoms
    fever_encoded = 1 if fever_input == 'Yes' else 0
    cough_encoded = 1 if cough_input == 'Yes' else 0
    fatigue_encoded = 1 if fatigue_input == 'Yes' else 0

    # Create sample data and train the Bayesian Network
    st.write("### Step 2: Training Bayesian Network")
    data = create_sample_data()
    model = train_bayesian_network(data)

    # Bayesian Network Inference
    inference_engine = VariableElimination(model)
    
    # Perform diagnosis
    if st.button('Diagnose'):
        evidence = {'Fever': fever_encoded, 'Cough': cough_encoded, 'Fatigue': fatigue_encoded}
        diagnosis_result = inference_engine.map_query(variables=['COVID'], evidence=evidence)
        diagnosis = 'Positive' if diagnosis_result['COVID'] == 1 else 'Negative'
        st.write(f'The diagnosis for COVID-19 is: **{diagnosis}**')

if __name__ == '__main__':
    main()
