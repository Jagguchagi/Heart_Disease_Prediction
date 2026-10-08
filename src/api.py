import logging
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "heart_disease_pipeline.joblib"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Heart Disease Risk API",
    version="0.1.0",
    description="Educational prediction demo; not for clinical decisions.",
)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float = Field(gt=0, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(gt=0)
    chol: float = Field(ge=0)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0)
    exang: int = Field(ge=0, le=1)
    oldpeak: float
    slope: int = Field(ge=1, le=3)
    ca: float | None = Field(default=None, ge=0, le=3)
    thal: int | None = Field(default=None)


class PredictionResponse(BaseModel):
    prediction: int = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    disease_probability: float = Field(ge=0, le=1)


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError("Model is not trained. Run `python -m src.train` first.")
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PatientFeatures) -> PredictionResponse:
    try:
        model = load_model()
        features = pd.DataFrame([payload.model_dump()])
        prediction = int(model.predict(features)[0])
        class_probabilities = model.predict_proba(features)[0]
        positive_class_index = list(model.classes_).index(1)
        disease_probability = float(class_probabilities[positive_class_index])
        confidence = float(class_probabilities[list(model.classes_).index(prediction)])
        logger.info("Prediction completed: class=%s", prediction)
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            disease_probability=disease_probability,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction could not be completed") from error
