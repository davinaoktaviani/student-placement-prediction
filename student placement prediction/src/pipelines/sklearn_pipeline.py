from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from src.features.pipeline_preprocessor import build_preprocessor

def build_pipeline_clf(num_features, cat_features):
    return Pipeline([
        ("preprocessing", build_preprocessor(num_features, cat_features)),
        ("model", SVC())
    ])

def build_pipeline_reg(num_features, cat_features):
    return Pipeline([
        ("preprocessing", build_preprocessor(num_features, cat_features)),
        ("model", LinearRegression())
    ])