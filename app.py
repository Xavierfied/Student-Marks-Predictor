import streamlit as st
import pandas as pd
import joblib

# --- Load the saved model components from the .joblib files ---
try:
    preprocessor = joblib.load('preprocessor.joblib')
    poly_features_max = joblib.load('poly_features_max.joblib')
    poly_model_max = joblib.load('poly_model_max.joblib')
except FileNotFoundError:
    st.error(
        "Error: Model files not found. Please make sure 'preprocessor.joblib', 'poly_features_max.joblib', and 'poly_model_max.joblib' are in the same directory.")

# --- Streamlit UI and Logic ---

st.title('Student Score Predictor')
st.markdown("---")
st.write("Adjust the sliders and select the options to get a predicted score.")

st.subheader("Numerical Features")
col1, col2 = st.columns(2)
with col1:
    hours_studied = st.slider("Hours Studied (Per Week)", min_value=0, max_value=100, value=30)
    sleep_hours = st.slider("Sleep Hours (Per Day)", min_value=0, max_value=12, value=8)
    tutoring_sessions = st.slider("Tutoring Sessions (per week)", min_value=0, max_value=10, value=3)

with col2:
    attendance = st.slider("Attendance (%)", min_value=0, max_value=100, value=78)
    physical_activity = st.slider("Physical Activity (hours)", min_value=0, max_value=10, value=3)
st.markdown("---")

st.subheader("Categorical Features")
col3, col4 = st.columns(2)

with col3:
    parental_involvement = st.selectbox("Parental Involvement", options=['High', 'Medium', 'Low'])
    school_type = st.selectbox("School Type", options=['Public', 'Private'])
    family_income = st.selectbox("Family Income", options=['High', 'Medium', 'Low'])
with col4:
    access_to_resources = st.selectbox("Access to Resources", options=['High', 'Medium', 'Low'])
    internet_access = st.selectbox("Internet Access", options=['Yes', 'No'])

if st.button("Predict Score"):
    new_student_data = pd.DataFrame({
        'Hours_Studied': [hours_studied],
        'Attendance': [attendance],
        'Parental_Involvement': [parental_involvement],
        'Access_to_Resources': [access_to_resources],
        'Sleep_Hours': [sleep_hours],
        'School_Type': [school_type],
        'Tutoring_Sessions': [tutoring_sessions],
        'Internet_Access': [internet_access],
        'Family_Income': [family_income],
        'Physical_Activity': [physical_activity]
    })

    # Transform the data using the loaded preprocessor
    new_student_encoded = preprocessor.transform(new_student_data)

    # Transform the data using the loaded polynomial features
    new_student_final = poly_features_max.transform(new_student_encoded)

    # Make the prediction
    predicted_score = poly_model_max.predict(new_student_final)[0]

    st.markdown("---")
    st.subheader("Predicted Exam Score:")
    st.success(f"# {predicted_score:.1f}")