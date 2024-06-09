import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import streamlit as st

# Data preparation by CODE-CRAFTER team
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
})

# Converting categorical data into numerical format
for feature in ['Outlook', 'Temperature', 'Humidity']:
    data[feature] = data[feature].astype('category').cat.codes

# Features and target variable separation
features = data.drop('PlayTennis', axis=1)
target = data['PlayTennis']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Training the Gaussian Naive Bayes model
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train, y_train)

# Streamlit application starts here
st.title(' CODE-CRAFTER')

st.title('Play Tennis Predictor')

st.write('Input the current weather conditions to predict if tennis can be played:')

# Collecting user input
input_outlook = st.selectbox('Outlook', ['Sunny', 'Overcast', 'Rainy'])
input_temperature = st.selectbox('Temperature', ['Hot', 'Mild', 'Cool'])
input_humidity = st.selectbox('Humidity', ['High', 'Normal'])
input_windy = st.selectbox('Windy', ['False', 'True'])

# Encoding the user input
encoded_outlook = ['Sunny', 'Overcast', 'Rainy'].index(input_outlook)
encoded_temperature = ['Hot', 'Mild', 'Cool'].index(input_temperature)
encoded_humidity = ['High', 'Normal'].index(input_humidity)
encoded_windy = ['False', 'True'].index(input_windy)

# Making the prediction
if st.button('Predict'):
    user_input = [[encoded_outlook, encoded_temperature, encoded_humidity, encoded_windy]]
    result = naive_bayes_model.predict(user_input)
    st.write(f"Based on the input, you can {'play' if result[0] == 'Yes' else 'not play'} tennis.")
