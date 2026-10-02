# We are creating the project interface here. i.e. the ui
import streamlit as st
import pandas as pd
import joblib

#  4. Loading the model
model = joblib.load("random_forest_model.pkl")

# 1. For providing title to the web app
st.title("Student Performance Predictor")
st.write("Enter the student's academic details to predict their expected final marks.")

# for adding small heading section
st.subheader("Student Details")

# adding divider to look standard
st.divider()

# 2. For labelling the input fields
study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    step=0.5
    )
attendance = st.number_input(
    "Attendance",
    min_value=0.0,
    max_value=100.0,
    step=1.0)
previous_marks = st.number_input(
    "Previous Marks",
    min_value=0.0,
    max_value=100.0,
    step=1.0)
assignment_score = st.number_input(
    "Assignment Score",
    min_value=0.0,
    max_value=100.0,
    step=1.0)
test_score = st.number_input(
    "Test Score",
    min_value=0.0,
    max_value=100.0,
    step=1.0)


st.divider()
# 3. For creating a button to trigger the prediction
if st.button("Predict Performance"):
    new_student = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks,
        "assignment_score": assignment_score,
        "test_score": test_score
    }])

    prediction = model.predict(new_student)
    
    st.success(f"Predicted Performance: {prediction[0] : .2f}/100")