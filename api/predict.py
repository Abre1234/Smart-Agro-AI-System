from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"
ENCODER_PATH = BASE_DIR / "artifacts" / "label_encoder.joblib"

app = FastAPI(title="Smart Agro AI API")

class PredictRequest(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


def load_artifacts():
    if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
        raise FileNotFoundError("Model artifacts not found. Run `python train.py` first.")

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder


model, encoder = load_artifacts()


@app.get("/")
def root():
    return {
        "name": "Smart Agro AI API",
        "predict_url": "/api/predict",
        "documentation": "Use POST /api/predict with soil and weather features."
    }


@app.get("/predict")
def predict_info():
    return {
        "name": "Smart Agro AI API",
        "documentation": "POST /api/predict with soil and weather features.",
        "fields": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    }


@app.post("/")
@app.post("/predict")
def predict(request: PredictRequest):
    feature_data = {
        "N": [request.N],
        "P": [request.P],
        "K": [request.K],
        "temperature": [request.temperature],
        "humidity": [request.humidity],
        "ph": [request.ph],
        "rainfall": [request.rainfall],
    }
    features = pd.DataFrame(feature_data)

    prediction_index = model.predict(features)[0]
    prediction_label = encoder.inverse_transform([prediction_index])[0]
    probabilities = model.predict_proba(features)[0]
    class_labels = encoder.inverse_transform(np.arange(len(probabilities)))

    return {
        "prediction": prediction_label,
        "probabilities": {
            label: float(prob)
            for label, prob in zip(class_labels, probabilities)
        }
    }
