import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.markdown("This app uses Logistic Regression to predict heart disease risk based on your health details.")

# 🔽 Load dataset from GitHub
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/muhsinasm/heart_app/main/heart.csv"
    df = pd.read_csv(url)

    # Drop unwanted columns
    df.drop(columns=[col for col in ['dataset', 'id'] if col in df.columns], inplace=True)

    # Encode categorical columns
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col].astype(str))

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df)

    return df

df = load_data()

# 🧠 Prepare features and target
X = df.drop(columns='num')
y = df['num'].apply(lambda x: 1 if x > 0 else 0)  # Binary class: 0 or 1

# Split not necessary for online app; just train on all data
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# 🩺 User input form
st.header("🩺 Enter Your Health Details Below")

def user_input_features():
    age = st.slider("Age", 20, 100, 50)
    sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    restecg = st.selectbox("Resting ECG (0=Normal, 1=ST-T Abnormality, 2=LVH)", [0, 1, 2])
    thalch = st.slider("Max Heart Rate Achieved", 60, 250, 150)
    exang = st.selectbox("Exercise Induced Angina", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
    ca = st.selectbox("Major Vessels Colored (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (1=Normal, 2=Fixed, 3=Reversible)", [1, 2, 3])

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalch': thalch,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()
st.write("Model Input:", input_data)

# 🧠 Make prediction
if st.button("🔍 Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100
    if prediction == 1:
        st.error(f"❗ Risk Detected: {probability:.2f}% chance of heart disease. Please consult a doctor.")
    else:
        st.success(f"✅ Good News: Only {100 - probability:.2f}% chance of heart disease. Stay healthy!")

# 📊 Show model accuracy and confusion matrix
if st.checkbox("📈 Show Model Accuracy & Confusion Matrix"):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    st.write(f"📌 Accuracy: **{acc * 100:.2f}%**")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
    st.pyplot(fig)

# 📉 Extra visualizations
if st.checkbox("📊 Show Data Visualizations"):
    st.subheader("🔸 Age Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['age'], bins=20, kde=True, ax=ax1, color='green')
    st.pyplot(fig1)

    st.subheader("🔸 Chest Pain Types")
    fig2, ax2 = plt.subplots()
    sns.countplot(x='cp', data=df, palette='mako', ax=ax2)
    st.pyplot(fig2)

