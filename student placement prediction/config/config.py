from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Targets
TARGET_CLASS = "placement_status"
TARGET_REG   = "salary_lpa"

DROP_COLS = ["Student_ID"]

NUM_FEATURES = [
    "cgpa", "tenth_percentage", "twelfth_percentage", "backlogs",
    "study_hours_per_day", "attendance_percentage",
    "projects_completed", "internships_completed",
    "coding_skill_rating", "communication_skill_rating",
    "aptitude_skill_rating", "hackathons_participated",
    "certifications_count", "sleep_hours", "stress_level"
]

CAT_FEATURES = [
    "gender", "branch", "part_time_job",
    "family_income_level", "city_tier",
    "internet_access", "extracurricular_involvement"
]

# MLflow
MLFLOW_TRACKING_URI = "file:./mlruns"
MLFLOW_EXPERIMENT = "student-pipeline"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Save model
MODEL_CLF_PATH = ARTIFACTS_DIR / "svm_pipeline.pkl"
MODEL_REG_PATH = ARTIFACTS_DIR / "linear_pipeline.pkl"