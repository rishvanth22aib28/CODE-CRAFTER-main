import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Load data from CSV
@st.cache_data
def load_data():
    return pd.read_csv('trainingdata.csv')

# Function to implement the learning method of the Candidate elimination algorithm
def learn(concepts, target):
    '''
    learn() function implements the learning method of the Candidate elimination algorithm.
    Arguments:
        concepts - a data frame with all the features
        target - an array with corresponding output values
    '''
    # Initialize specific_h with the first instance from concepts
    specific_h = concepts[0].copy()
    st.write("\nInitialization of specific_h and general_h")
    st.write(f"Specific_h: {specific_h}")

    # Initialize general_h with '?'
    general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]
    st.write(f"General_h: {general_h}")

    # The learning iterations
    for i, h in enumerate(concepts):
        # Checking if the hypothesis has a positive target
        if target[i] == "Yes":
            for x in range(len(specific_h)):
                # Change values in specific_h & general_h only if values change
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
        # Checking if the hypothesis has a negative target
        if target[i] == "No":
            for x in range(len(specific_h)):
                # For negative hypothesis, change values only in general_h
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

        st.write(f"\nStep {i+1} of Candidate Elimination Algorithm")
        st.write(f"Specific_h: {specific_h}")
        st.write(f"General_h: {general_h}")

    # Find indices where we have empty rows, meaning those that are unchanged
    indices = [i for i, val in enumerate(general_h) if val == ['?' for _ in range(len(specific_h))]]
    for i in indices:
        # Remove those rows from general_h
        general_h.remove(['?' for _ in range(len(specific_h))])

    # Return final values
    return specific_h, general_h

# Streamlit app
def main():
    st.title('CODE-CRAFTER')
    st.title('Candidate Elimination Algorithm')

    # Load data
    data = load_data()

    # Display the loaded data
    st.subheader('Loaded Data')
    st.write(data)

    # Add user input controls
    st.sidebar.title('User Inputs')
    # Add sliders for interactive control
    threshold = st.sidebar.slider('Threshold:', min_value=0.1, max_value=1.0, value=0.5, step=0.1, format='%.1f')
    learning_rate = st.sidebar.slider('Learning Rate:', min_value=0.01, max_value=0.1, value=0.05, step=0.01, format='%.2f')

    # Separate concept features from target
    concepts = data.iloc[:, :-1].values
    target = data.iloc[:, -1].values

    # Learn
    s_final, g_final = learn(concepts, target)

    # Display final hypotheses
    st.subheader('Final Specific_h:')
    st.write(s_final)
    st.subheader('Final General_h:')
    st.write(g_final)

    # Visualization
    st.subheader('Visualization')
    # Plot histogram of target variable
    fig = px.histogram(data, x=data.columns[-1], title='Target Variable Distribution')
    st.plotly_chart(fig)

    # Show correlation heatmap
    numerical_data = data.select_dtypes(include=np.number)  # Select only numerical columns
    corr = numerical_data.corr()
    fig = px.imshow(corr, color_continuous_scale='Viridis', title='Correlation Heatmap')
    st.plotly_chart(fig)

if __name__ == '__main__':
    main()