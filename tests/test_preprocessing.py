
import pandas as pd

from src.preprocessing import create_preprocessor


def test_preprocessor_handles_missing_values():
    data = pd.DataFrame(
        {
            "age": [63, 45, 55],
            "sex": [1, 0, 1],
            "cp": [3, 1, 2],
            "trestbps": [145, 120, 130],
            "chol": [233, 200, 250],
            "fbs": [1, 0, 0],
            "restecg": [0, 1, 0],
            "thalach": [150, 170, 160],
            "exang": [0, 0, 1],
            "oldpeak": [2.3, 1.0, 1.5],
            "slope": [0, 1, 1],
            "ca": [0.0, float("nan"), 1.0],
            "thal": [1.0, 2.0, float("nan")],
        }
    )

    preprocessor = create_preprocessor()
    transformed = preprocessor.fit_transform(data)

    assert transformed.shape[0] == 3
    assert transformed.shape[1] > 0
    assert not pd.isna(transformed).any()
