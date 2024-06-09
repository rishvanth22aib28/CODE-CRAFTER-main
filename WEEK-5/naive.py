import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load sample data function by CODE-CRAFTER team
def load_sample_data():
    # Sample dataset - customize as needed
    sample_data = {
        'text': [
            'I love this movie, it is amazing!',
            'The movie was terrible and boring.',
            'Fantastic film with a great plot.',
            'Not my cup of tea, quite dull.',
            'An outstanding performance by the lead actor.',
            'I did not like the movie at all.'
        ],
        'label': [1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative
    }
    return pd.DataFrame(sample_data)

# Model training function by CODE-CRAFTER team
def train_naive_bayes_classifier(data):
    texts = data['text']
    labels = data['label']

    # Text vectorization
    vectorizer = CountVectorizer()
    texts_vectorized = vectorizer.fit_transform(texts)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(texts_vectorized, labels, test_size=0.3, random_state=42)

    # Initialize and train Naive Bayes model
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, y_train)

    # Model evaluation
    predictions = nb_classifier.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)

    return nb_classifier, vectorizer, acc, prec, rec

# Main function for Streamlit app
def main():
    st.title('CODE-CRAFTER')
    st.title('Document Classification with Naive Bayes')

    st.write("### Step 1: Load Dataset")
    dataset = load_sample_data()
    st.write(dataset)

    st.write("### Step 2: Train the Model")
    model, vectorizer, accuracy, precision, recall = train_naive_bayes_classifier(dataset)

    st.write(f"**Accuracy:** {accuracy:.2f}")
    st.write(f"**Precision:** {precision:.2f}")
    st.write(f"**Recall:** {recall:.2f}")

    st.write("### Step 3: Predict Document Sentiment")
    user_input = st.text_area("Enter a document to classify", "")
    if user_input:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        prediction_label = "Positive" if prediction == 1 else "Negative"
        st.write(f"Prediction: {prediction_label}")

if __name__ == '__main__':
    main()
