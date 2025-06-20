
import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Heart Attack Risk Predictor", layout="centered")
st.title("💓 Heart Attack Risk Predictor")

st.markdown("Enter patient information below:")

# Essential features for user input (keep form short and user-friendly)
age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
thalach = st.slider("Maximum Heart Rate Achieved (thalach)", 70, 210, 150)
exang = st.selectbox("Exercise-Induced Angina (exang)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.slider("ST Depression Induced by Exercise (oldpeak)", 0.0, 6.0, 1.0, step=0.1)

if st.button("Predict Risk"):
    try:
        # Load model with preprocessing
        model = joblib.load("final_xgb_pipeline.pkl")
        
        # Prepare input
        input_data = np.array([[age, sex, cp, 0, 0, 0, 0, thalach, exang, oldpeak, 0, 0, 0]])  # Fill missing features with 0s
        prediction = model.predict(input_data)[0]
        
        if prediction == 1:
            st.error("⚠️ High Risk of Heart Attack")
        else:
            st.success("✅ Low Risk of Heart Attack")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
