
import streamlit as st
import numpy as np
import joblib

# Load KNN model (with full preprocessor inside if pipeline was saved)
model = joblib.load("final_knn_model.pkl")

st.title("💓 KNN-Based Heart Attack Prediction App")

# Show only selected features to the user
age = st.slider("Age", 18, 100, 45)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain_type = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
thalach = st.slider("Maximum Heart Rate", 60, 220, 150)
oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
ca = st.selectbox("Major Vessels Colored (0–3)", [0, 1, 2, 3])

# Convert 'sex' to numeric
sex_val = 1 if sex == "Male" else 0

# Provide default values for other features
resting_blood_pressure = 120
serum_cholesterol = 200
fasting_blood_sugar = 0
restecg = 1
exang = 0
slope = 1
thal = 2

# Final input array (full feature order)
input_data = np.array([[age, sex_val, chest_pain_type, resting_blood_pressure, serum_cholesterol,
                        fasting_blood_sugar, restecg, thalach, exang, oldpeak, slope, ca, thal]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease! Probability: {prob:.2f}")
    else:
        st.success(f"✅ Low Risk of Heart Disease. Probability: {prob:.2f}")
