import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from config.config import MODEL_CLF_PATH, MODEL_REG_PATH
from src.utils.io import load_model

# LOAD MODEL
@st.cache_resource
def load_models():
    clf = load_model(MODEL_CLF_PATH)
    reg = load_model(MODEL_REG_PATH)
    return clf, reg

clf_model, reg_model = load_models()

# PAGE CONFIG 
st.set_page_config(page_title="Student Prediction", layout="wide")

# TITLE 
st.title("Student Placement & Salary Prediction")
st.markdown("Predict placement status and salary based on student profile")

# SIDEBAR 
st.sidebar.title("Input Data")

# Basic
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
branch = st.sidebar.selectbox("Branch", ["CSE", "ECE", "ME", "CE", "IT"])

# Academic
st.sidebar.subheader("Academic")
cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, 7.0)
tenth = st.sidebar.slider("10th %", 0, 100, 75)
twelfth = st.sidebar.slider("12th %", 0, 100, 75)
backlogs = st.sidebar.slider("Backlogs", 0, 10, 0)
attendance = st.sidebar.slider("Attendance %", 0, 100, 80)

# Study & Lifestyle
st.sidebar.subheader("Study & Lifestyle")
study_hours = st.sidebar.slider("Study Hours", 0, 12, 4)
sleep = st.sidebar.slider("Sleep Hours", 0, 12, 7)
stress = st.sidebar.slider("Stress Level", 1, 10, 5)

# Skills
st.sidebar.subheader("Skills")
coding_skill = st.sidebar.slider("Coding Skill", 1, 10, 5)
communication = st.sidebar.slider("Communication", 1, 10, 6)
aptitude = st.sidebar.slider("Aptitude", 1, 10, 6)

# Experience
st.sidebar.subheader("Experience")
internships = st.sidebar.slider("Internships", 0, 5, 1)
projects = st.sidebar.slider("Projects", 0, 10, 2)
hackathons = st.sidebar.slider("Hackathons", 0, 10, 1)
certifications = st.sidebar.slider("Certifications", 0, 10, 1)

# Others
st.sidebar.subheader("Others")
part_time = st.sidebar.selectbox("Part Time Job", ["Yes", "No"])
income = st.sidebar.selectbox("Family Income", ["Low", "Medium", "High"])
city = st.sidebar.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
internet = st.sidebar.selectbox("Internet Access", ["Yes", "No"])
extra = st.sidebar.selectbox("Extracurricular", ["Low", "Medium", "High"])

# Button
predict_btn = st.sidebar.button("Predict")

# PREDICTION 
if predict_btn:

    data = {
        "gender": gender,
        "branch": branch,
        "cgpa": cgpa,
        "tenth_percentage": tenth,
        "twelfth_percentage": twelfth,
        "backlogs": backlogs,
        "study_hours_per_day": study_hours,
        "attendance_percentage": attendance,
        "projects_completed": projects,
        "internships_completed": internships,
        "coding_skill_rating": coding_skill,
        "communication_skill_rating": communication,
        "aptitude_skill_rating": aptitude,
        "hackathons_participated": hackathons,
        "certifications_count": certifications,
        "sleep_hours": sleep,
        "stress_level": stress,
        "part_time_job": part_time,
        "family_income_level": income,
        "city_tier": city,
        "internet_access": internet,
        "extracurricular_involvement": extra
    }

    df = pd.DataFrame([data])

    # MODEL PREDICTION
    placement = clf_model.predict(df)[0]
    salary = reg_model.predict(df)[0]

    # RESULT
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if placement == 1:
            st.success("Placed")
        else:
            st.error("Not Placed")

    with col2:
        st.metric("Estimated Salary (LPA)", f"{salary:.2f}")

    st.divider()

    # VISUALIZATION
    st.subheader("Visualization Dashboard")

    col_viz1, col_viz2, col_viz3 = st.columns(3)

    # 1. Skill Overview
    with col_viz1:
        fig1, ax1 = plt.subplots()
        ax1.bar(["CGPA", "Study", "Coding", "Comm", "Apt"],
                [cgpa, study_hours, coding_skill, communication, aptitude])
        ax1.set_title("Skills")
        st.pyplot(fig1)

    # 2. Academic Performance
    with col_viz2:
        fig2, ax2 = plt.subplots()
        ax2.plot(["10th", "12th", "CGPA", "Attend"],
                 [tenth, twelfth, cgpa*10, attendance], marker='o')
        ax2.set_title("Academic")
        st.pyplot(fig2)

    # 3. Experience
    with col_viz3:
        fig3, ax3 = plt.subplots()
        ax3.bar(["Proj", "Intern", "Hack", "Cert"],
                [projects, internships, hackathons, certifications])
        ax3.set_title("Experience")
        st.pyplot(fig3)

    st.divider()

    # DATA
    st.subheader("Input Data")
    st.dataframe(df)