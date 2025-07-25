import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load and preprocess dataset
@st.cache_data
def load_model():
    df = pd.read_csv("heart.csv")

    if 'dataset' in df.columns:
        df = df.drop(['dataset'], axis=1)
    if 'id' in df.columns:
        df = df.drop(['id'], axis=1)

    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df)

    X = df.drop('num', axis=1)
    y = df['num'].apply(lambda x: 1 if x > 0 else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

model, acc = load_model()

# Streamlit UI
st.title("💓 Heart Disease Risk Prediction App")
st.markdown("Please enter the following health parameters to predict your risk level.")

# Input fields with clear labels
age = st.number_input("Age", min_value=1, max_value=120, value=40)
sex = st.selectbox("Sex", [("Male", 1), ("Female", 0)])
cp = st.selectbox("Chest Pain Type (0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", [("Yes", 1), ("No", 0)])
restecg = st.selectbox("Resting ECG (0: Normal, 1: ST-T, 2: LVH)", [0, 1, 2])
thalch = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise-induced Angina?", [("Yes", 1), ("No", 0)])
oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of the ST segment", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1: Normal, 2: Fixed, 3: Reversible)", [1, 2, 3])

# Prepare input
user_input = np.array([[age, sex[1], cp, trestbps, chol, fbs[1], restecg, thalch,
                        exang[1], oldpeak, slope, ca, thal]])

# Predict
if st.button("🔍 Predict"):
    prediction = model.predict(user_input)[0]
    if prediction == 1:
        st.error("⚠️ You may have a risk of heart disease. Please consult a doctor.")
    else:
        st.success("✅ You are not showing signs of heart disease. Stay healthy!")

    st.info(f"Model Accuracy: {acc * 100:.2f}%")
