
import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "LogisticRegression.joblib"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "Educational ML prediction service. "
        "Not intended for clinical diagnosis."
    ),
    version="1.0.0",
)

model = joblib.load(MODEL_PATH)

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal",
]


class PatientData(BaseModel):
    age: int = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=0, le=3)
    trestbps: float = Field(gt=0)
    chol: float = Field(gt=0)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0)
    exang: int = Field(ge=0, le=1)
    oldpeak: float
    slope: int = Field(ge=0, le=2)
    ca: float | None = Field(default=None, ge=0, le=3)
    thal: float | None = Field(default=None, ge=0, le=3)


@app.get("/health")
def health():
    return {"status": "healthy", "model": "LogisticRegression"}


@app.post("/predict")
def predict(patient: PatientData):
    try:
        values = patient.model_dump()

        # Missing ca/thal values are handled by the fitted imputer.
        for feature in ("ca", "thal"):
            if values[feature] is None:
                values[feature] = float("nan")

        features = pd.DataFrame(
            [[values[name] for name in FEATURE_NAMES]],
            columns=FEATURE_NAMES,
        )

        probabilities = model.predict_proba(features)[0]
        prediction = int(model.predict(features)[0])

        logger.info("Prediction request processed successfully")

        return {
            "prediction": prediction,
            "label": (
                "heart_disease_present"
                if prediction == 1
                else "heart_disease_absent"
            ),
            "confidence": round(float(max(probabilities)), 4),
            "probabilities": {
                "absence": round(float(probabilities[0]), 4),
                "presence": round(float(probabilities[1]), 4),
            },
            "disclaimer": (
                "Educational prediction only; not a medical diagnosis."
            ),
        }

    except Exception:
        logger.exception("Prediction request failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from None
Instrumentator().instrument(app).expose(
    app,
    endpoint="/metrics",
)