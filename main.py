import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from langchain_community.llms import Ollama

# Streamlit UI setup
st.set_page_config(page_title="ML + LLM Explainer", layout="centered")
st.title("🔍 Credit Card Fraud Detection + LLM Explainer (TinyLlama)")

# File upload
uploaded_file = st.file_uploader("📤 Upload your credit card CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    if 'Class' not in data.columns:
        st.error("The dataset must contain a 'Class' column as the target.")
    else:
        st.write("📊 Data Preview", data.head())

        # Class Distribution
        class_counts = data['Class'].value_counts()
        st.write("✅ Legit: ", int(class_counts[0]))
        st.write("⚠️ Fraud: ", int(class_counts[1]))

        # Balance the dataset
        legit = data[data.Class == 0]
        fraud = data[data.Class == 1]
        legit_sample = legit.sample(n=len(fraud), random_state=42)
        balanced_data = pd.concat([legit_sample, fraud], axis=0)

        # Train-test split
        X = balanced_data.drop(columns='Class', axis=1)
        Y = balanced_data['Class']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

        # Logistic Regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, Y_train)

        # Accuracy
        Y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_pred)
        st.success(f"📈 Model trained. Accuracy: {accuracy*100:.2f}%")

        # Extract features for LLM
        feature_names = list(X.columns)
        top_features = feature_names[:3] if len(feature_names) >= 3 else feature_names

        # Use TinyLlama via Ollama
        llm = Ollama(model="tinyllama")

        prompt = f"""
        I trained a logistic regression model to detect credit card fraud.
        The dataset has columns: {', '.join(feature_names)}.
        It reached {accuracy*100:.2f}% accuracy.
        The most important features were: {', '.join(top_features)}.
        Can you explain this to a non-technical person in simple terms?
        """

        with st.spinner("🤖 Thinking with TinyLlama..."):
            explanation = llm.invoke(prompt.strip())

        st.markdown("### 🧠 LLM Explanation")
        st.write(explanation)

