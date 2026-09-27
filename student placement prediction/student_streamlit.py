import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Student Placement & Salary Prediction")

API_URL = "http://127.0.0.1:8000/predict"

# SIDEBAR INPUT 
st.sidebar.header("Input Data")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
branch = st.sidebar.selectbox("Branch", ["CSE", "ECE", "ME", "CE", "IT"])

cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, 7.0)
tenth = st.sidebar.slider("10th %", 0, 100, 75)
twelfth = st.sidebar.slider("12th %", 0, 100, 75)
backlogs = st.sidebar.slider("Backlogs", 0, 10, 0)

study = st.sidebar.slider("Study Hours", 0, 12, 4)
attendance = st.sidebar.slider("Attendance %", 0, 100, 80)

coding = st.sidebar.slider("Coding Skill", 1, 10, 5)
communication = st.sidebar.slider("Communication", 1, 10, 6)
aptitude = st.sidebar.slider("Aptitude", 1, 10, 6)

projects = st.sidebar.slider("Projects", 0, 10, 2)
internships = st.sidebar.slider("Internships", 0, 5, 1)

hackathons = st.sidebar.slider("Hackathons", 0, 10, 1)
certifications = st.sidebar.slider("Certifications", 0, 10, 1)

sleep = st.sidebar.slider("Sleep Hours", 0, 12, 7)
stress = st.sidebar.slider("Stress Level", 1, 10, 5)

part_time = st.sidebar.selectbox("Part Time Job", ["Yes", "No"])
income = st.sidebar.selectbox("Family Income", ["Low", "Medium", "High"])
city = st.sidebar.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
internet = st.sidebar.selectbox("Internet Access", ["Yes", "No"])
extra = st.sidebar.selectbox("Extracurricular", ["Low", "Medium", "High"])

# BUTTON 
if st.sidebar.button("Predict"):

    payload = {
        "gender": gender,
        "branch": branch,
        "cgpa": cgpa,
        "tenth_percentage": tenth,
        "twelfth_percentage": twelfth,
        "backlogs": backlogs,
        "study_hours_per_day": study,
        "attendance_percentage": attendance,
        "projects_completed": projects,
        "internships_completed": internships,
        "coding_skill_rating": coding,
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

    response = requests.post(API_URL, json=payload)

    # RESULT
    if response.status_code == 200:
        result = response.json()

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            if result["placement"] == 1:
                st.success("Placed")
            else:
                st.error("Not Placed")

        with col2:
            st.metric("Estimated Salary (LPA)", f"{result['salary']:.2f}")

        st.divider()

        # VISUALIZATION 
        st.subheader("Visualization Dashboard")

        col1, col2, col3 = st.columns(3)

        # Skill Chart
        with col1:
            fig1, ax1 = plt.subplots()
            ax1.bar(
                ["CGPA", "Study", "Coding", "Comm", "Apt"],
                [cgpa, study, coding, communication, aptitude]
            )
            ax1.set_title("Skills Overview")
            st.pyplot(fig1)

        # Academic Chart
        with col2:
            fig2, ax2 = plt.subplots()
            ax2.plot(
                ["10th", "12th", "CGPA", "Attend"],
                [tenth, twelfth, cgpa * 10, attendance],
                marker='o'
            )
            ax2.set_title("Academic Performance")
            st.pyplot(fig2)

        # Experience Chart
        with col3:
            fig3, ax3 = plt.subplots()
            ax3.bar(
                ["Projects", "Internships", "Hackathons", "Certs"],
                [projects, internships, hackathons, certifications]
            )
            ax3.set_title("Experience")
            st.pyplot(fig3)

        st.divider()

        # INPUT DATA TABLE 
        st.subheader("Input Data")

        df = pd.DataFrame([payload])
        st.dataframe(df)

    else:
        st.error("API Error")