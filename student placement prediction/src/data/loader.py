import pandas as pd
from sklearn.model_selection import train_test_split
from config.config import *

def load_data():
    df_features = pd.read_csv(DATA_RAW_DIR / "A.csv")
    df_target   = pd.read_csv(DATA_RAW_DIR / "A_targets.csv")

    df = pd.merge(df_features, df_target, on="Student_ID")
    return df

def split_data(df):
    X = df.drop(DROP_COLS + [TARGET_CLASS, TARGET_REG], axis=1)

    y_class = df[TARGET_CLASS]
    y_reg   = df[TARGET_REG]

    X_train, X_test, y_train_c, y_test_c = train_test_split(
        X, y_class, test_size=TEST_SIZE, stratify=y_class, random_state=RANDOM_STATE
    )

    _, _, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train_c, y_test_c, y_train_r, y_test_r