import streamlit as st
import numpy as np
import pandas as pd

class Node:
    def __init__(self, feature=None, value=None, result=None):
        self.feature = feature
        self.value = value
        self.result = result
        self.children = {}

def entropy_calc(labels):
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def gain_info(data, feature_name, labels):
    total_entropy = entropy_calc(labels)
    values, counts = np.unique(data[feature_name], return_counts=True)
    weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * entropy_calc(labels[data[feature_name] == values[i]]) for i in range(len(values))])
    info_gain = total_entropy - weighted_entropy
    return info_gain

def build_decision_tree(data, labels, features):
    if len(np.unique(labels)) == 1:
        return Node(result=labels.iloc[0])
   
    if len(features) == 0:
        return Node(result=labels.mode()[0])
   
    max_gain = -1
    best_feature = None
    for feature in features:
        gain = gain_info(data, feature, labels)
        if gain > max_gain:
            max_gain = gain
            best_feature = feature
   
    root = Node(feature=best_feature)
    values = np.unique(data[best_feature])
    for value in values:
        sub_data = data[data[best_feature] == value]
        sub_labels = labels[data[best_feature] == value]
        if len(sub_data) == 0:
            root.children[value] = Node(result=labels.mode()[0])
        else:
            root.children[value] = build_decision_tree(sub_data, sub_labels, [f for f in features if f != best_feature])
    return root

def predict(root, sample):
    if root.result is not None:
        return root.result
    value = sample[root.feature]
    if value not in root.children:
        return None
    return predict(root.children[value], sample)

def main():    
    st.title("CODE-CRAFTER")
    st.title("Decision Tree Classifier with ID3 Algorithm ")
   
    # Sample data
    data = pd.DataFrame({
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal'],
        'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak'],
        'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
    })

    labels = data['PlayTennis']
    features = data.columns[:-1]

    # Build the decision tree
    root = build_decision_tree(data, labels, features)

    # Collect input from user
    st.sidebar.header("Input Features")
    sample = {}
    for feature in features:
        sample[feature] = st.sidebar.selectbox(f"Select {feature}", data[feature].unique())

    # Classify the input sample
    prediction = predict(root, sample)

    # Display the result
    st.write(f"Predicted Class: {prediction}")

if __name__ == "__main__":
    main()