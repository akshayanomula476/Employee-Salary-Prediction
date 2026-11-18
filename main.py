
import streamlit as st
from model import train_model

# Page configuration
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="centered"
)

# Train model
model, mae, r2, data = train_model()

# Title
st.title("💼 Employee Salary Prediction")

st.write(
    "Predict an employee's estimated salary using "
    "Machine Learning."
)

st.divider()

# Employee details
st.subheader("Enter Employee Details")

experience = st.slider(
    "Experience (Years)",
    min_value=0,
    max_value=30,
    value=5
)

education = st.slider(
    "Education Level (Years)",
    min_value=10,
    max_value=25,
    value=16
)

job_role = st.selectbox(
    "Job Role",
    data["job_role"].unique()
)

# Prediction button
if st.button("Predict Salary"):

    employee = [[experience, education, job_role]]

    import pandas as pd

    employee_df = pd.DataFrame(
        employee,
        columns=[
            "experience_years",
            "education_level",
            "job_role"
        ]
    )

    prediction = model.predict(employee_df)

    st.success("Prediction Completed!")

    st.subheader("Estimated Salary")

    st.write(f"₹ {prediction[0]:,.2f}")

st.divider()

# Model information
st.subheader("Model Performance")

st.write(f"Mean Absolute Error: {mae:.2f}")
st.write(f"R² Score: {r2:.2f}")

st.caption(
    "This project uses a small sample dataset for educational demonstration. "
    "Predictions are not reliable real-world salary estimates."
)
