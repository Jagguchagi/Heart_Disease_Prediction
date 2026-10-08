import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from .data_preparation import (
    DATA_PATH,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    ROOT,
    TARGET_COLUMN,
    load_data,
)

ARTIFACTS_DIR = ROOT / "artifacts"


def run_eda() -> dict:
    """Create reproducible summary data and EDA charts from the source dataset."""
    dataset = load_data()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    target_counts = dataset[TARGET_COLUMN].value_counts().sort_index()
    missing_counts = dataset[FEATURE_COLUMNS].isna().sum()
    summary = {
        "source_file": str(DATA_PATH.relative_to(ROOT)),
        "rows": int(dataset.shape[0]),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "data_types": {name: str(dataset[name].dtype) for name in FEATURE_COLUMNS},
        "missing_values": {
            name: int(count) for name, count in missing_counts.items() if count > 0
        },
        "duplicate_rows": int(dataset.duplicated().sum()),
        "target_mapping": "num=0 -> target=0; num=1,2,3,4 -> target=1",
        "target_counts": {str(name): int(count) for name, count in target_counts.items()},
        "target_percentages": {
            str(name): round(float(count / len(dataset) * 100), 2)
            for name, count in target_counts.items()
        },
    }

    figure, axis = plt.subplots(figsize=(7, 5))
    labels = ["No disease (0)", "Disease present (1)"]
    axis.bar(labels, [target_counts.get(0, 0), target_counts.get(1, 0)], color=["#247ba0", "#f25f5c"])
    axis.set_title("Cleveland Heart Disease: Target Class Balance")
    axis.set_ylabel("Number of records")
    axis.set_xlabel("Binarized diagnosis")
    axis.bar_label(axis.containers[0])
    figure.tight_layout()
    figure.savefig(ARTIFACTS_DIR / "class_balance.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    for axis, column in zip(axes.flat, NUMERIC_COLUMNS):
        sns.histplot(data=dataset, x=column, hue=TARGET_COLUMN, bins=18, element="step", stat="count", common_norm=False, ax=axis)
        axis.set_title(f"{column} distribution")
        axis.set_xlabel(column)
        axis.set_ylabel("Records")
    axes.flat[-1].axis("off")
    figure.suptitle("Continuous Feature Distributions by Disease Class", y=1.02)
    figure.tight_layout()
    figure.savefig(ARTIFACTS_DIR / "feature_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    correlation = dataset[NUMERIC_COLUMNS].corr()
    sns.heatmap(correlation, annot=True, cmap="vlag", center=0, fmt=".2f", square=True, ax=axis)
    axis.set_title("Correlation Among Continuous Measurements")
    figure.tight_layout()
    figure.savefig(ARTIFACTS_DIR / "correlation_heatmap.png", dpi=160)
    plt.close(figure)

    summary_path = ARTIFACTS_DIR / "eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved EDA plots and summary under {ARTIFACTS_DIR}")
    return summary


if __name__ == "__main__":
    run_eda()