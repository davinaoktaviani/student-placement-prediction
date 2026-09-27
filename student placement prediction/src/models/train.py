import mlflow
import mlflow.sklearn
import joblib
from sklearn.metrics import accuracy_score, r2_score
from config.config import *

def train_pipeline(clf_pipeline, reg_pipeline,
                   X_train, X_test,
                   y_train_c, y_test_c,
                   y_train_r, y_test_r):

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():

        # Train
        clf_pipeline.fit(X_train, y_train_c)
        reg_pipeline.fit(X_train, y_train_r)

        # Predict
        pred_c = clf_pipeline.predict(X_test)
        pred_r = reg_pipeline.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test_c, pred_c)
        r2  = r2_score(y_test_r, pred_r)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("r2", r2)

        # Log params
        mlflow.log_param("model_clf", "SVM")
        mlflow.log_param("model_reg", "LinearRegression")

        # Log models
        mlflow.sklearn.log_model(clf_pipeline, "placement_model")
        mlflow.sklearn.log_model(reg_pipeline, "salary_model")

        # Save .pkl
        MODEL_CLF_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf_pipeline, MODEL_CLF_PATH)
        joblib.dump(reg_pipeline, MODEL_REG_PATH)

        print("Accuracy:", acc)
        print("R2:", r2)