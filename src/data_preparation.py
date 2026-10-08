from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "heart+disease" / "processed.cleveland.data"
TARGET_COLUMN = "target"
RAW_TARGET_COLUMN = "num"
FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
RAW_COLUMNS = FEATURE_COLUMNS + [RAW_TARGET_COLUMN]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the headerless UCI Cleveland file and binarize its disease label."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    frame = pd.read_csv(
        path,
        header=None,
        names=RAW_COLUMNS,
        na_values=["?", ""],
        skipinitialspace=True,
    )
    if frame.empty or frame[RAW_TARGET_COLUMN].isna().any():
        raise ValueError("Dataset must contain rows with a non-missing disease label")
    if not set(frame[RAW_TARGET_COLUMN].unique()).issubset({0, 1, 2, 3, 4}):
        raise ValueError("The Cleveland disease label 'num' must contain values from 0 to 4")

    frame[TARGET_COLUMN] = frame.pop(RAW_TARGET_COLUMN).gt(0).astype("int64")
    return frame


def build_preprocessor() -> ColumnTransformer:
    """Impute missing features, scale continuous values, and encode categories."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )


if __name__ == "__main__":
    dataset = load_data()
    print(f"Loaded {len(dataset)} rows and {len(FEATURE_COLUMNS)} features from {DATA_PATH}")
    print(f"Missing feature values: {int(dataset[FEATURE_COLUMNS].isna().sum().sum())}")
    print(f"Binary target counts: {dataset[TARGET_COLUMN].value_counts().sort_index().to_dict()}")
