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
st.markdown("This app uses a Logistic Regression ML model to predict the risk of heart disease based on your health inputs.")

# Load and prepare dataset
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/muhsinasm/heart_app/main/heart.csv")
    return df

df = load_data()

# Remove the 'dataset' column if it's not there
if 'dataset' in df.columns:
    df = df.drop(columns='dataset')

# Train the model
X = df.drop(columns='num')
y = df['num']

# Handle missing values
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
y = pd.to_numeric(y, errors='coerce').fillna(0)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# 🩺 Input from user
st.header("🩺 Enter Your Health Details Below")

def user_input_features():
    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex (0 = female, 1 = male)", [0, 1])
    cp = st.selectbox("Chest Pain Type (0 = typical angina, 1 = atypical, 2 = non-anginal, 3 = asymptomatic)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (in mm Hg)", 80, 200, 120)
    chol = st.number_input("Cholesterol (in mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = yes, 0 = no)", [0, 1])
    restecg = st.selectbox("Resting ECG results (0 = normal, 1 = ST-T wave abnormality, 2 = probable LVH)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
    exang = st.selectbox("Exercise Induced Angina (1 = yes, 0 = no)", [0, 1])
    oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of ST segment (0 = up, 1 = flat, 2 = down)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)", [1, 2, 3])

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Reorder to match training data
input_df = input_df[X.columns]

# Predict on button click
if st.button("🔍 Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100
    if prediction == 1:
        st.error(f"❗ Risk Detected: {probability:.2f}% chance of heart disease. Please consult a doctor.")
    else:
        st.success(f"✅ Good News: Only {100 - probability:.2f}% chance of heart disease. Keep maintaining your health!")

# Accuracy and confusion matrix
if st.checkbox("📊 Show Model Accuracy & Confusion Matrix"):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    st.write(f"✅ Model Accuracy: **{acc * 100:.2f}%**")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
    st.pyplot(fig)

# Extra Visualization (optional)
if st.checkbox("📈 Show Some Health Data Visualizations"):
    st.subheader("🔸 Age Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['age'], bins=20, kde=True, ax=ax1, color='green')
    st.pyplot(fig1)

    st.subheader("🔸 Chest Pain Types Count")
    fig2, ax2 = plt.subplots()
    sns.countplot(x='cp', data=df, palette='mako', ax=ax2)
    ax2.set_xlabel("Chest Pain Type")
    ax2.set_ylabel("Count")
    st.pyplot(fig2)
