

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Mental Health Risk Prediction",
    page_icon="🧠",
    layout="centered"
)

# Chargement du modèle
MODEL_PATH = Path(__file__).parent / "models" / "logistic.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Interface
st.title("🧠 Mental Health Risk Prediction")
st.write(
    "Cette application permet de prédire le niveau de risque "
    "en santé mentale chez les étudiants."
)


st.subheader("Informations de l'étudiant")

with st.form("prediction_form"):

    Age = st.number_input(
    "Age",
    min_value=15,
    max_value=100,
    value=22
    )

    year_of_study = st.number_input(
        "Year of Study",
        min_value=1,
        max_value=10,
        value=2
    )

    sleep_hours = st.number_input(
        "Sleep Hours",
        min_value=0.0,
        max_value=24.0,
        value=7.0
    )

    study_hours = st.number_input(
        "Study Hours Per Week",
        min_value=0.0,
        max_value=168.0,
        value=20.0
    )

    gpa = st.number_input(
        "GPA",
        min_value=0.0,
        max_value=4.0,
        value=3.0
    )

    social_media_hours = st.number_input(
        "Social Media Hours",
        min_value=0.0,
        max_value=24.0,
        value=3.0
    )

    screen_time_hours = st.number_input(
        "Screen Time Hours",
        min_value=0.0,
        max_value=24.0,
        value=5.0
    )

    financial_stress = st.number_input(
        "Financial Stress",
        min_value=1,
        max_value=5,
        value=3
    )

    family_support = st.number_input(
        "Family Support",
        min_value=1,
        max_value=5,
        value=3
    )

    attendance_rate = st.number_input(
        "Attendance Rate",
        min_value=0.0,
        max_value=100.0,
        value=80.0
    )

    physical_activity = st.selectbox(
        "Physical Activity",
        ["Low", "Moderate", "High"]
    )

    diet_quality = st.selectbox(
        "Diet Quality",
        ["Poor", "Average", "Good"]
    )
    Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
    )
    Previous_Mental_Health_History = st.selectbox(
    "Previous Mental Health History",
    ["Yes", "No"]
    )
    Part_Time_Job = st.selectbox(
    "Part Time Job",
    ["Yes", "No"]
    )

    submit = st.form_submit_button("Predict")

# Prédiction
if submit:

    input_data = pd.DataFrame([{
        "Age": Age,
        "Year_of_Study": year_of_study,
        "Sleep_Hours": sleep_hours,
        "Study_Hours_Per_Week": study_hours,
        "GPA": gpa,
        "Social_Media_Hours": social_media_hours,
        "Screen_Time_Hours": screen_time_hours,
        "Financial_Stress": financial_stress,
        "Family_Support": family_support,
        "Attendance_Rate": attendance_rate,
        "Physical_Activity": physical_activity,
        "Diet_Quality": diet_quality,
        "Gender": Gender,
        "Previous_Mental_Health_History": Previous_Mental_Health_History,
        "Part_Time_Job": Part_Time_Job
    }])

    # Prédiction
    prediction = model.predict(input_data)[0]

    st.subheader("Résultat de la prédiction")
    st.success(f"Niveau de risque prédit : {prediction}")

    # Probabilités
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_data)[0]

        probability_df = pd.DataFrame({
            "Classe": model.classes_,
            "Probabilité (%)": probabilities * 100
        })

        st.subheader("Probabilités par classe")
        st.dataframe(probability_df, use_container_width=True)

        st.bar_chart(
            probability_df.set_index("Classe")
        )
