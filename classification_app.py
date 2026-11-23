from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Cleveland Heart Disease Predictor",
    page_icon="❤️",
    layout="centered",
)

MODEL_PATH = Path(__file__).parent / "classification_model.joblib"
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# ---------------------------------------------------------------
# Header + instructions
# ---------------------------------------------------------------
st.title("❤️ Heart Disease Risk Prediction App")
st.markdown(
    "This app uses a **Random Forest classifier** trained on the "
    "**UCI Heart Disease (Cleveland cohort)** dataset to estimate whether a "
    "patient's clinical measurements are consistent with heart disease."
)

with st.expander("How to use this app", expanded=True):
    st.markdown(
        "1. Enter the patient's clinical parameters below.\n"
        "2. Click **Predict Heart Disease Risk**.\n"
        "3. Review the predicted class and the model's probability."
    )

with st.sidebar:
    st.header("About")
    st.markdown(
        "- **Dataset:** UCI Cleveland Heart Disease (303 patients)\n"
        "- **Target:** disease present (1) vs. absent (0)\n"
        "- **Model:** StandardScaler + Random Forest pipeline\n"
        "- **Course:** BDA700 Health Analytics"
    )
    st.warning(
        "Educational use only. This tool is **not** a medical device and "
        "must not be used for diagnosis or treatment decisions."
    )

st.markdown("---")

# ---------------------------------------------------------------
# User inputs
# ---------------------------------------------------------------
st.subheader("Patient Clinical Parameters")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (years)", 29, 77, 54)
    sex = st.selectbox(
        "Sex", options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male",
    )
    cp = st.selectbox(
        "Chest Pain Type", options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Typical Angina", 2: "Atypical Angina",
            3: "Non-anginal Pain", 4: "Asymptomatic",
        }[x],
    )
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.slider("Serum Cholesterol (mg/dl)", 120, 564, 240)
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl", options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
    )
    restecg = st.selectbox(
        "Resting ECG Results", options=[0, 1, 2],
        format_func=lambda x: {
            0: "Normal", 1: "ST-T Wave Abnormality",
            2: "Left Ventricular Hypertrophy",
        }[x],
    )

with col2:
    thalach = st.slider("Maximum Heart Rate Achieved (bpm)", 71, 202, 150)
    exang = st.selectbox(
        "Exercise-Induced Angina", options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
    )
    oldpeak = st.slider(
        "ST Depression Induced by Exercise", 0.0, 6.2, 1.0, step=0.1
    )
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment", options=[1, 2, 3],
        format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x],
    )
    ca = st.selectbox(
        "Major Vessels Colored by Fluoroscopy (0–3)", options=[0, 1, 2, 3]
    )
    thal = st.selectbox(
        "Thalassemia", options=[3, 6, 7],
        format_func=lambda x: {
            3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect",
        }[x],
    )

input_data = pd.DataFrame(
    [[age, sex, cp, trestbps, chol, fbs, restecg,
      thalach, exang, oldpeak, slope, ca, thal]],
    columns=FEATURES,
)

with st.expander("Review entered values"):
    st.dataframe(input_data, hide_index=True)

st.markdown("---")

# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
if st.button("Predict Heart Disease Risk", type="primary"):
    prediction = int(model.predict(input_data)[0])
    proba = model.predict_proba(input_data)[0]

    st.subheader("Prediction Results")
    if prediction == 1:
        st.error(
            f"⚠️ **High Risk Detected** — estimated probability of heart "
            f"disease: **{proba[1] * 100:.1f}%**\n\n"
            "The model predicts a pattern consistent with heart disease. "
            "Please consult a qualified clinician for evaluation."
        )
    else:
        st.success(
            f"✅ **Low Risk Detected** — estimated probability of no heart "
            f"disease: **{proba[0] * 100:.1f}%**\n\n"
            "The model indicates a lower likelihood of heart disease."
        )

    st.progress(float(proba[1]), text=f"Heart disease probability: {proba[1] * 100:.1f}%")
