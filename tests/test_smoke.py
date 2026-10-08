import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_preparation import (
    DATA_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_preprocessor,
    load_data,
)


def test_cleveland_data_is_loaded_and_target_is_binary():
    dataset = load_data(DATA_PATH)

    assert dataset.shape == (303, 14)
    assert TARGET_COLUMN == "target"
    assert set(dataset[TARGET_COLUMN].unique()) == {0, 1}
    assert int(dataset[FEATURE_COLUMNS].isna().sum().sum()) == 6


def test_preprocessor_handles_missing_values_and_categories():
    dataset = load_data()
    features = dataset[FEATURE_COLUMNS]
    transformed = build_preprocessor().fit_transform(features)
    values = transformed.toarray() if hasattr(transformed, "toarray") else transformed

    assert values.shape[0] == len(dataset)
    assert np.isfinite(values).all()


def test_full_pipeline_can_be_saved_and_reloaded(tmp_path):
    dataset = load_data()
    features = dataset[FEATURE_COLUMNS]
    target = dataset[TARGET_COLUMN]
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipeline.fit(features, target)
    expected = pipeline.predict(features.iloc[:8])
    model_path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, model_path)
    restored = joblib.load(model_path)

    assert np.array_equal(restored.predict(features.iloc[:8]), expected)
