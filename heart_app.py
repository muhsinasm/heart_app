import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# Title
st.title("❤️ Heart Disease Prediction App")

# Introduction
st.markdown("This app uses a **Logistic Regression** model to predict the risk of heart disease based on user health inputs.")

# Load and cache dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/muhsinasm/heart_app/main/heart.csv"
    return pd.read_csv(url)

df = load_data()

# Show Dataset Preview
if st.checkbox("📊 Show Raw Dataset"):
    st.subheader("Dataset Preview")
    st.write(df.head())

# Data Info
X = df.drop(columns='num')
y = df['num']

# Handle numeric conversion and missing values
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
y = pd.to_numeric(y, errors='coerce').fillna(0)

# Train Model
model = LogisticRegression()
model.fit(X, y)

# User Inputs
st.header("🩺 Enter Your Health Details:")
def user_input_features():
    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex", [0, 1])
    dataset = st.selectbox("Dataset (source)", [0, 1])
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG results", [0, 1, 2])
    thalch = st.number_input("Max Heart Rate Achieved (thalch)", 60, 250, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of ST segment", [0, 1, 2])
    ca = st.selectbox("Number of major vessels (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

    data = {
        'age': age, 'sex': sex, 'dataset': dataset, 'cp': cp,
        'trestbps': trestbps, 'chol': chol, 'fbs': fbs, 'restecg': restecg,
        'thalch': thalch, 'exang': exang, 'oldpeak': oldpeak,
        'slope': slope, 'ca': ca, 'thal': thal
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()
input_df = input_df[X.columns]  # Ensure column order matches

# Prediction
if st.button("🔍 Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100

    if prediction == 1:
        st.error(f"❗ Risk Detected: {probability:.2f}% chance of heart disease. Please consult a doctor.")
    else:
        st.success(f"✅ Good News: Only {100 - probability:.2f}% chance of heart disease. Stay healthy!")

# Accuracy and Confusion Matrix
if st.checkbox("📈 Show Model Accuracy & Confusion Matrix"):
    st.subheader("Model Accuracy")
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    st.write(f"Accuracy: **{acc * 100:.2f}%**")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.title("Confusion Matrix")
    st.pyplot(fig)

# Feature Correlation Heatmap
if st.checkbox("📊 Show Correlation Heatmap"):
    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(fig)

# Distribution of Heart Disease
if st.checkbox("🧬 Show Target Distribution"):
    st.subheader("Heart Disease Target Class Count")
    fig, ax = plt.subplots()
    sns.countplot(x='num', data=df, palette='Set2')
    plt.xticks([0, 1], ['No Disease', 'Disease'])
    st.pyplot(fig)

# Pie Chart of Chest Pain Types
if st.checkbox("🫀 Show Chest Pain Type Distribution"):
    st.subheader("Chest Pain Type (cp) Distribution")
    cp_counts = df['cp'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(cp_counts, labels=cp_counts.index, autopct='%1.1f%%', startangle=90)
    st.pyplot(fig)

# Histogram of Cholesterol
if st.checkbox("🍔 Show Cholesterol Distribution"):
    st.subheader("Cholesterol Levels")
    fig, ax = plt.subplots()
    sns.histplot(df['chol'], bins=30, kde=True, color='orange')
    st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Made with ❤️ by Muhsina S M | Powered by Logistic Regression and Streamlit")
