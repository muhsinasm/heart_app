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
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", [0, 1])
    dataset = st.selectbox("Dataset (source)", [0, 1])
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)", [0, 1])
    restecg = st.selectbox("Resting ECG results", [0, 1, 2])
    thalch = st.number_input("Max Heart Rate Achieved (thalch)", min_value=60, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina (1 = yes; 0 = no)", [0, 1])
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of ST segment", [0, 1, 2])
    ca = st.selectbox("Number of major vessels colored by fluoroscopy (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

    data = {
        'age': age,
        'sex': sex,
        'dataset': dataset,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalch': thalch,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    return pd.DataFrame(data, index=[0])

input_data = user_input_features()  # Should return a dictionary or DataFrame
input_df = pd.DataFrame([input_data])  # Convert to single-row DataFrame

# Rearrange columns to match training data
input_df = input_df[X.columns]  # This ensures correct order and count

# Prediction
input_data = user_input_features()

if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100
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
