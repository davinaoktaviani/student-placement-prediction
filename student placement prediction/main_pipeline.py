from src.data.loader import load_data, split_data
from src.pipelines.sklearn_pipeline import build_pipeline_clf, build_pipeline_reg
from src.models.train import train_pipeline
from config.config import NUM_FEATURES, CAT_FEATURES

def main():
    print("=== Student Placement Pipeline ===")

    df = load_data()

    X_train, X_test, y_train_c, y_test_c, y_train_r, y_test_r = split_data(df)

    clf_pipeline = build_pipeline_clf(NUM_FEATURES, CAT_FEATURES)
    reg_pipeline = build_pipeline_reg(NUM_FEATURES, CAT_FEATURES)

    train_pipeline(
        clf_pipeline, reg_pipeline,
        X_train, X_test,
        y_train_c, y_test_c,
        y_train_r, y_test_r
    )

if __name__ == "__main__":
    main()