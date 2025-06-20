
import streamlit as st
import joblib
import numpy as np

# Load trained pipeline (preprocessor + model)
model = joblib.load("final_xgb_pipeline.pkl")

# Page title
st.set_page_config(page_title="Heart Attack Risk Prediction", layout="centered")
st.title("💓 Heart Attack Risk Prediction App")
st.markdown("Enter the patient information below:")

# Input fields
age = st.slider("Age", 20, 100, 45)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure (trestbps)", 80, 200, 120)
chol = st.slider("Cholesterol (chol)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", ["Yes", "No"])
restecg = st.selectbox("Resting ECG Results (restecg)", [0, 1, 2])
thalach = st.slider("Maximum Heart Rate Achieved (thalach)", 70, 210, 150)
exang = st.selectbox("Exercise-Induced Angina (exang)", ["Yes", "No"])
oldpeak = st.slider("ST Depression Induced by Exercise (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
slope = st.selectbox("Slope of the ST Segment (slope)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Convert to numerical format
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# Final feature array
features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])

# Predict
if st.button("Predict Risk"):
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"🚨 High Risk of Heart Attack ({prob*100:.1f}%)")
    else:
        st.success(f"✅ Low Risk of Heart Attack ({(1 - prob)*100:.1f}%)")
