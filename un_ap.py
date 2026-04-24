import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import requests

# ================= DOWNLOAD MODEL (GOOGLE DRIVE) =================
MODEL_URL = "https://drive.google.com/uc?id=1FZsyrKfuo500dYa5psB4xHZKSHvXxHJx"

if not os.path.exists("fair_model.pkl"):
    with open("fair_model.pkl", "wb") as f:
        response = requests.get(MODEL_URL)
        f.write(response.content)

# ================= LOAD FILES =================
BASE_DIR = os.path.dirname(__file__)

model = joblib.load("fair_model.pkl")
scaler = joblib.load(os.path.join(BASE_DIR, "fair_scaler.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "fair_features.pkl"))

# ================= GEMINI SETUP =================
USE_GEMINI = True
try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
except:
    USE_GEMINI = False

# ================= UI =================
st.set_page_config(page_title="Fair AI Income Predictor", layout="centered")

st.title("💰 Fair Income Prediction System")
st.write("⚖️ Fairness-aware model (bias reduced)")

st.header("Enter User Details")

age = st.slider("Age", 18, 65, 30)
education_num = st.slider("Education Level (1=Low, 16=PhD)", 1, 16, 10)
hours_per_week = st.slider("Hours per Week", 1, 100, 40)

# ================= INPUT =================
input_data = pd.DataFrame({
    "age": [age],
    "education-num": [education_num],
    "hours-per-week": [hours_per_week]
})

for col in features:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[features]
input_scaled = scaler.transform(input_data)

# ================= MAIN BUTTON =================
if st.button("Predict & Explain 🔥"):

    prob = model._pmf_predict(input_scaled)[:,1][0]
    pred = 1 if prob > 0.5 else 0
    prob = round(prob, 3)

    st.subheader("Prediction")

    if pred == 1:
        st.success("Predicted Income: >50K 💰")
    else:
        st.warning("Predicted Income: <=50K")

    st.write(f"📊 Confidence: {prob*100:.2f}%")

    # ================= EXPLANATION =================
    st.subheader("🧠 Explanation")

    explanation = None

    if USE_GEMINI:
        try:
            prompt = f"""
            A fairness-aware ML model predicted income.

            Age: {age}
            Education Level: {education_num}
            Working Hours: {hours_per_week}
            Probability: {prob}

            Explain in simple 2 lines why this prediction was made.
            """

            response = gemini_model.generate_content(prompt)
            explanation = response.text

        except Exception:
            explanation = None

    # ================= FALLBACK =================
    if not explanation:
        if education_num > 12 and hours_per_week > 40:
            explanation = "Higher education and longer working hours increase chances of high income."
        elif education_num < 8:
            explanation = "Lower education level reduces chances of high income."
        elif hours_per_week < 30:
            explanation = "Working fewer hours leads to lower income prediction."
        else:
            explanation = "Income depends on a mix of education and work hours."

    st.write(explanation)

# ================= FOOTER =================
st.markdown("---")
st.write("Built with ❤️ using Fairlearn + Streamlit + Gemini")
