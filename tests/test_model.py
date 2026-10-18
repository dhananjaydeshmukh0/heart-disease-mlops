
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "LogisticRegression.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart_disease.csv"


def test_saved_model_loads_and_predicts():
    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(DATA_PATH)

    X = data.drop(columns=["target"]).iloc[:5]
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    assert len(predictions) == 5
    assert probabilities.shape == (5, 2)
    assert set(predictions).issubset({0, 1})
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
