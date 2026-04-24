import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ================= LOAD MODEL =================
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))

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
education_num = st.slider("Education Number", 1, 16, 10)
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
        st.success(f">50K 💰")
    else:
        st.warning(f"<=50K")

    st.write(f"📊 Confidence: {prob*100:.2f}%")

    # ================= EXPLANATION =================
    st.subheader("🧠 Explanation")

    explanation = ""

    if USE_GEMINI:
        try:
            prompt = f"""
            Age: {age}, Education: {education_num}, Hours: {hours_per_week}.
            Probability: {prob}

            Explain simply why this prediction was made.
            """

            response = gemini_model.generate_content(prompt)
            explanation = response.text

        except Exception as e:
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
