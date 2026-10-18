
from pathlib import Path

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

MLFLOW_TRACKING_URI = (
    f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

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

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                ("classifier", estimator),
            ]
        )

        with mlflow.start_run(run_name=model_name):
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
                mean_score = cv_results[
                    f"test_{metric_name}"
                ].mean()

                std_score = cv_results[
                    f"test_{metric_name}"
                ].std()

                metrics[f"cv_{metric_name}_mean"] = float(mean_score)
                metrics[f"cv_{metric_name}_std"] = float(std_score)

            mlflow.log_params(
                {
                    "model": model_name,
                    "cv_folds": 5,
                    "random_state": 42,
                    **estimator.get_params(),
                }
            )
            mlflow.log_metrics(metrics)

            # Fit the pipeline on all available data after CV.
            pipeline.fit(X, y)

            model_path = MODEL_DIR / f"{model_name}.joblib"

            import joblib

            joblib.dump(pipeline, model_path)
            mlflow.log_artifact(str(model_path))

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

    print("\nCross-validation comparison:")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {results_path}")
    print(f"Models saved in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
