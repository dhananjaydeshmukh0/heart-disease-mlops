
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.preprocessing import create_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart_disease.csv"
MODEL_DIR = PROJECT_ROOT / "models"
EDA_DIR = PROJECT_ROOT / "screenshots" / "eda"
ARTIFACT_DIR = PROJECT_ROOT / "mlflow_artifacts"

MLFLOW_TRACKING_URI = (
    f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
)

EDA_PLOTS = [
    "target_distribution.png",
    "feature_distributions.png",
    "correlation_heatmap.png",
]


def log_eda_plots():
    """Log existing EDA visualizations to the active MLflow run."""
    logged_count = 0

    for filename in EDA_PLOTS:
        plot_path = EDA_DIR / filename

        if plot_path.exists():
            mlflow.log_artifact(
                str(plot_path),
                artifact_path="eda_plots",
            )
            print(f"Logged EDA plot: {filename}")
            logged_count += 1
        else:
            print(f"WARNING: EDA plot not found: {plot_path}")

    if logged_count == 0:
        print("WARNING: No EDA plots were logged.")


def create_model_comparison_plot(results_df, output_path):
    """Create a comparison of mean cross-validation metrics."""
    metrics = [
        "cv_accuracy_mean",
        "cv_precision_mean",
        "cv_recall_mean",
        "cv_roc_auc_mean",
    ]

    plot_df = results_df.set_index("model")[metrics].T
    plot_df.index = [
        "Accuracy",
        "Precision",
        "Recall",
        "ROC-AUC",
    ]

    ax = plot_df.plot(
        kind="bar",
        figsize=(10, 6),
        ylim=(0, 1),
        rot=0,
    )
    ax.set_title("Heart Disease Model Comparison (5-Fold CV)")
    ax.set_ylabel("Mean cross-validation score")
    ax.set_xlabel("Metric")
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"])
    y = df["target"]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42,
        ),
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Heart-Disease-Classification")

    results = []
    run_ids = []

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                ("classifier", estimator),
            ]
        )

        with mlflow.start_run(run_name=model_name) as run:
            run_ids.append(run.info.run_id)

            cv_results = cross_validate(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=scoring,
                return_train_score=False,
                n_jobs=-1,
            )

            metrics = {}

            for metric_name in scoring:
                scores = cv_results[f"test_{metric_name}"]
                metrics[f"cv_{metric_name}_mean"] = float(scores.mean())
                metrics[f"cv_{metric_name}_std"] = float(scores.std())

            mlflow.log_params(
                {
                    "model": model_name,
                    "cv_folds": 5,
                    "random_state": 42,
                    **estimator.get_params(),
                }
            )
            mlflow.log_metrics(metrics)

            # Attach EDA plots to this model's MLflow run.
            log_eda_plots()

            # Fit the complete pipeline on the available dataset.
            pipeline.fit(X, y)

            model_path = MODEL_DIR / f"{model_name}.joblib"
            joblib.dump(pipeline, model_path)

            mlflow.log_artifact(
                str(model_path),
                artifact_path="models",
            )

            results.append(
                {
                    "model": model_name,
                    **metrics,
                }
            )

            print(f"\n{model_name}")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

    results_df = pd.DataFrame(results)

    results_path = MODEL_DIR / "cross_validation_results.csv"
    results_df.to_csv(results_path, index=False)

    # Generate the comparison after both models have completed.
    comparison_path = ARTIFACT_DIR / "model_comparison.png"
    create_model_comparison_plot(results_df, comparison_path)

    # Log the comparison plot and results CSV to BOTH model runs.
    for run_id in run_ids:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(
                str(comparison_path),
                artifact_path="evaluation_plots",
            )
            mlflow.log_artifact(
                str(results_path),
                artifact_path="evaluation_results",
            )

    print("\nCross-validation comparison:")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {results_path}")
    print(f"Model comparison plot: {comparison_path}")
    print(f"Models saved in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
