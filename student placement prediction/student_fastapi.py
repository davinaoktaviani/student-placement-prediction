from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load pipeline models
clf_model = joblib.load("artifacts/svm_pipeline.pkl")
reg_model = joblib.load("artifacts/linear_pipeline.pkl")

app = FastAPI(title="Student Placement API")

# Schema input
class StudentFeatures(BaseModel):
    gender: str
    branch: str
    cgpa: float
    tenth_percentage: float
    twelfth_percentage: float
    backlogs: int
    study_hours_per_day: float
    attendance_percentage: float
    projects_completed: int
    internships_completed: int
    coding_skill_rating: int
    communication_skill_rating: int
    aptitude_skill_rating: int
    hackathons_participated: int
    certifications_count: int
    sleep_hours: float
    stress_level: int
    part_time_job: str
    family_income_level: str
    city_tier: str
    internet_access: str
    extracurricular_involvement: str


@app.get("/")
def root():
    return {"message": "Student Placement API is running"}


@app.post("/predict")
def predict(data: StudentFeatures):

    df = pd.DataFrame([data.model_dump()])

    placement = clf_model.predict(df)[0]
    salary = reg_model.predict(df)[0]

    return {
    "placement": placement,
    "salary": float(salary)
}