import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("❤️ Heart Disease Prediction App")

# Introduction
st.markdown("This app uses a machine learning model to predict the risk of heart disease based on health inputs.")

# Load and prepare dataset
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/muhsinasm/heart_app/main/heart.csv") # You can replace with your own hosted dataset
    return df

df = load_data()

# Train the model
X = df.drop(columns='num')
y = df['num']

# Convert to numeric in case any values are strings
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

# Handle missing values (if any)
X = X.fillna(0)
y = y.fillna(0)

# Now fit the model
model = LogisticRegression()
model.fit(X, y)

# User Input
st.header("🩺 Enter Your Health Details:")
def user_input_features():
    age = st.number_input("Enter your age:", min_value=1, max_value=120, step=1)
    sex = st.radio("Sex:", ["Female", "Male"])
    sex = 1 if sex == "Male" else 0
    cp = st.slider("Chest pain type (0–3):", 0, 3)
    trestbps = st.number_input("Resting blood pressure (in mm Hg):", 80, 200)
    chol = st.number_input("Serum cholesterol (in mg/dl):", 100, 600)
    fbs = st.radio("Fasting blood sugar > 120 mg/dl?", ["False", "True"])
    fbs = 1 if fbs == "True" else 0
    restecg = st.slider("Resting ECG results (0–2):", 0, 2)
    thalach = st.number_input("Max heart rate achieved:", 60, 250)
    exang = st.radio("Exercise-induced angina?", ["No", "Yes"])
    exang = 1 if exang == "Yes" else 0
    oldpeak = st.slider("ST depression induced by exercise:", 0.0, 10.0, step=0.1)
    slope = st.slider("Slope of ST segment (0–2):", 0, 2)
    ca = st.slider("Number of major vessels colored by fluoroscopy (0–3):", 0, 3)
    thal = st.slider("Thalassemia (1=Normal, 2=Fixed defect, 3=Reversible defect):", 1, 3)

    data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])
    return data

input_data = user_input_features()
input_df = input_data[X.columns]  # Rearranged to match training

# Prediction
if st.button("🔍 Predict"):
    # Ensure input_data is in the same format and order as training features
    input_df = pd.DataFrame([input_data], columns=X.columns)

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100

    if prediction == 1:
        st.error(f"❗ Risk Detected: {probability:.2f}% chance of heart disease. Please consult a doctor.")
    else:
        st.success(f"✅ Good News: Only {100 - probability:.2f}% chance of heart disease. Keep maintaining your health!")

# Optional: Confusion matrix display
if st.checkbox("Show Model Accuracy"):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    st.write(f"Model Accuracy: {acc * 100:.2f}%")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    st.pyplot(fig)
