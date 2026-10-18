
from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app)

SAMPLE_PATIENT = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1,
}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction_endpoint():
    response = client.post("/predict", json=SAMPLE_PATIENT)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in (0, 1)
    assert result["label"] in (
        "heart_disease_present",
        "heart_disease_absent",
    )
    assert 0 <= result["confidence"] <= 1
    assert abs(
        result["probabilities"]["absence"]
        + result["probabilities"]["presence"]
        - 1
    ) < 1e-6


def test_prediction_rejects_invalid_input():
    invalid_patient = {**SAMPLE_PATIENT, "age": -5}

    response = client.post("/predict", json=invalid_patient)

    assert response.status_code == 422


def test_prediction_handles_missing_optional_features():
    patient = {**SAMPLE_PATIENT, "ca": None, "thal": None}

    response = client.post("/predict", json=patient)

    assert response.status_code == 200
