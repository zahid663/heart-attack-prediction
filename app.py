import streamlit as st
import joblib
import numpy as np

model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💓 Heart Attack Risk Predictor")
st.markdown("Fill in the details below to check your heart attack risk.")

age = st.slider("Age", 25, 85, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
thalach = st.slider("Max Heart Rate Achieved", 70, 200, 150)
exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])

default_input = [
    age,
    1 if sex == "Male" else 0,
    cp,
    130,
    245,
    0,
    1,
    thalach,
    1 if exang == "Yes" else 0,
    oldpeak,
    slope,
    0,
    2
]

input_array = np.array([default_input])
scaled_input = scaler.transform(input_array)

if st.button("Predict"):
    prediction = model.predict(scaled_input)[0]
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error("💔 High Risk of Heart Attack")
    else:
        st.success("❤️ Low Risk of Heart Attack")
