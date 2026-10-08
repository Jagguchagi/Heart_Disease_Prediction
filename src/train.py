import json
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from .data_preparation import (
    DATA_PATH,
    FEATURE_COLUMNS,
    ROOT,
    TARGET_COLUMN,
    build_preprocessor,
    load_data,
)

matplotlib.use("Agg")

MODEL_PATH = ROOT / "models" / "heart_disease_pipeline.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
ARTIFACTS_DIR = ROOT / "artifacts"
MLFLOW_DATABASE = ROOT / "mlflow.db"
EXPERIMENT_NAME = "heart-disease-cleveland"
SCORING = ["accuracy", "precision", "recall", "roc_auc"]
SEED = 42


def model_searches() -> dict:
    """Return small, documented parameter searches for the required classifiers."""
    return {
        "logistic_regression": (
            Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor()),
                    ("classifier", LogisticRegression(max_iter=2000, random_state=SEED)),
                ]
            ),
            {"classifier__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor()),
                    ("classifier", RandomForestClassifier(random_state=SEED, class_weight="balanced")),
                ]
            ),
            {
                "classifier__n_estimators": [100, 250],
                "classifier__max_depth": [None, 5],
                "classifier__min_samples_leaf": [1, 3],
            },
        ),
    }


def classification_metrics(y_true, predictions, probabilities) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def save_evaluation_plots(name: str, y_true, predictions, probabilities) -> list[Path]:
    plot_dir = ARTIFACTS_DIR / "evaluation"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        confusion_matrix(y_true, predictions),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No disease", "Disease"],
        yticklabels=["No disease", "Disease"],
        ax=axis,
    )
    axis.set_title(f"{name.replace('_', ' ').title()} Test Confusion Matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    figure.tight_layout()
    confusion_path = plot_dir / f"{name}_confusion_matrix.png"
    figure.savefig(confusion_path, dpi=160)
    plt.close(figure)
    paths.append(confusion_path)

    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(false_positive_rate, true_positive_rate, label=f"AUC {roc_auc_score(y_true, probabilities):.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="No-skill baseline")
    axis.set_title(f"{name.replace('_', ' ').title()} Test ROC Curve")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.legend(loc="lower right")
    figure.tight_layout()
    roc_path = plot_dir / f"{name}_roc_curve.png"
    figure.savefig(roc_path, dpi=160)
    plt.close(figure)
    paths.append(roc_path)
    return paths


def train() -> Path:
    dataset = load_data(DATA_PATH)
    features = dataset[FEATURE_COLUMNS]
    target = dataset[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=SEED,
        stratify=target,
    )

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DATABASE.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tuning_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    comparison = []
    fitted_models = {}

    for name, (pipeline, parameter_grid) in model_searches().items():
        search = GridSearchCV(
            pipeline,
            parameter_grid,
            scoring="roc_auc",
            cv=tuning_cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(x_train, y_train)
        best_pipeline = search.best_estimator_
        cv_results = cross_validate(
            best_pipeline,
            x_train,
            y_train,
            scoring=SCORING,
            cv=cross_validation,
            n_jobs=1,
            return_train_score=False,
        )
        predictions = best_pipeline.predict(x_test)
        probabilities = best_pipeline.predict_proba(x_test)[:, 1]
        test_metrics = classification_metrics(y_test, predictions, probabilities)
        cv_metrics = {
            metric: float(np.mean(cv_results[f"test_{metric}"])) for metric in SCORING
        }
        row = {
            "model": name,
            "best_cv_roc_auc_during_tuning": float(search.best_score_),
            **{f"cv_{key}": value for key, value in cv_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
            "best_parameters": json.dumps(search.best_params_, sort_keys=True),
        }
        comparison.append(row)
        fitted_models[name] = (best_pipeline, search.best_params_, row)

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_param("selection_metric", "mean 5-fold training ROC-AUC")
            mlflow.log_param("tuning_cv_folds", tuning_cv.n_splits)
            mlflow.log_param("evaluation_cv_folds", cross_validation.n_splits)
            mlflow.log_param("test_size", 0.2)
            for parameter, value in search.best_params_.items():
                mlflow.log_param(parameter, value if value is not None else "None")
            mlflow.log_metric("tuning_best_roc_auc", float(search.best_score_))
            for metric, value in cv_metrics.items():
                mlflow.log_metric(f"cv_{metric}", value)
            for metric, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric}", value)
            for plot_path in save_evaluation_plots(name, y_test, predictions, probabilities):
                mlflow.log_artifact(str(plot_path), artifact_path="evaluation")
            eda_summary = ARTIFACTS_DIR / "eda_summary.json"
            if eda_summary.exists():
                mlflow.log_artifact(str(eda_summary), artifact_path="eda")
            for plot_name in ("class_balance.png", "feature_distributions.png", "correlation_heatmap.png"):
                plot_path = ARTIFACTS_DIR / plot_name
                if plot_path.exists():
                    mlflow.log_artifact(str(plot_path), artifact_path="eda")
            run_model_path = ARTIFACTS_DIR / "model_runs" / f"{name}.joblib"
            run_model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(best_pipeline, run_model_path)
            mlflow.log_artifact(str(run_model_path), artifact_path="model")

    comparison_frame = pd.DataFrame(comparison).sort_values(
        by="cv_roc_auc", ascending=False
    )
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_frame.to_csv(ARTIFACTS_DIR / "model_comparison.csv", index=False)
    selected_name = str(comparison_frame.iloc[0]["model"])
    selected_pipeline, selected_parameters, selected_metrics = fitted_models[selected_name]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_pipeline, MODEL_PATH)
    metadata = {
        "selected_model": selected_name,
        "selection_criterion": "highest mean 5-fold training ROC-AUC after tuning",
        "target_mapping": "num=0 -> 0; num=1,2,3,4 -> 1",
        "feature_columns": FEATURE_COLUMNS,
        "best_parameters": selected_parameters,
        "metrics": selected_metrics,
        "warning": "Educational dataset/model; not for clinical diagnosis or treatment decisions.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(comparison_frame.to_string(index=False))
    print(f"Selected {selected_name} using mean cross-validation ROC-AUC.")
    print(f"Saved complete preprocessing/model pipeline to {MODEL_PATH}")
    return MODEL_PATH


if __name__ == "__main__":
    train()
