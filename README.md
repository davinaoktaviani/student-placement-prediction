# Student Placement & Salary Prediction

A machine learning project that predicts **student placement status** and **estimated salary (LPA)** based on academic performance, skills, experience, study habits, and other student-related factors.

The project implements an end-to-end machine learning workflow covering data preprocessing, model training, evaluation, experiment tracking with MLflow, model serialization, and prediction through a Streamlit application.

---

## Project Overview

Getting placed after graduation can be influenced by many factors, including academic performance, technical skills, communication skills, internships, projects, attendance, and other student characteristics.

This project uses student profile data to build two predictive models:

1. **Classification** — predicts whether a student is **Placed** or **Not Placed**.
2. **Regression** — estimates the student's expected salary in **LPA (Lakhs Per Annum)**.

The application allows users to enter a student's profile and view the prediction together with simple visualizations of academic performance, skills, and experience.

---

### Main Features

- Gender
- Branch
- CGPA
- 10th percentage
- 12th percentage
- Backlogs
- Study hours per day
- Attendance percentage
- Projects completed
- Internships completed
- Coding skill rating
- Communication skill rating
- Aptitude skill rating
- Hackathons participated
- Certifications count
- Sleep hours
- Stress level
- Part-time job
- Family income level
- City tier
- Internet access
- Extracurricular involvement

### Target Variables

- `placement_status` — classification target
- `salary_lpa` — regression target

The dataset contains an imbalanced placement target, with **4,303 Placed students (86%)** and **697 Not Placed students (14%)**.

The `extracurricular_involvement` feature contains missing values. These categorical missing values are handled using the **mode** during preprocessing.

---

## Machine Learning Approach

### Preprocessing

The project uses a reusable Scikit-learn preprocessing pipeline:

- Numerical missing values are handled using median imputation.
- Numerical features are standardized using `StandardScaler`.
- Categorical missing values are handled using most-frequent imputation.
- Categorical features are encoded using `OneHotEncoder`.
- Unknown categorical values are handled safely with `handle_unknown="ignore"`.

### Classification

The classification models evaluated in the notebook are:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

**Evaluation metric:** Accuracy

| Model | Accuracy |
|---|---:|
| Logistic Regression | 0.890 |
| SVM | **0.892** |
| Random Forest | 0.887 |

The SVM achieved the highest accuracy among the three classification models evaluated, with an accuracy of **0.892**.

### Regression

The regression models evaluated are:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

**Evaluation metrics:** RMSE and R²

| Model | RMSE | R² |
|---|---:|---:|
| Linear Regression | **4.0874** | **0.5712** |
| Decision Tree | 5.9655 | 0.0866 |
| Random Forest | 4.1372 | 0.5607 |

Linear Regression achieved the lowest RMSE and the highest R² among the evaluated regression models.

---

## Deployment

The project provides a **Streamlit** application for interactive prediction.

Users can enter:

- Academic information
- Study and lifestyle information
- Technical and soft skills
- Projects, internships, hackathons, and certifications
- Family income and city tier
- Other student-related information

The application provides:

- Placement prediction
- Estimated salary in LPA
- Skills visualization
- Academic performance visualization
- Experience visualization
- Input data summary

The repository also contains a **FastAPI** application that exposes a `/predict` endpoint for prediction through an API.

---

## Future Improvements

- Address class imbalance in the placement target.
- Add more comprehensive classification metrics such as precision, recall, and F1-score.
- Perform hyperparameter tuning.
- Improve model explainability.
- Enhance the Streamlit dashboard.
- Deploy the application and API to a cloud platform.
