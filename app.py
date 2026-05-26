"""FastAPI app for crop recommendation (local + Vercel via GitHub)."""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

_model = None
_encoder = None
_scaler = None
_feature_names = None


def _resolve_artifacts_dir() -> Path:
    """Find artifacts on local disk or Vercel serverless (/var/task)."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "artifacts",
        here / "api" / "artifacts",
        here.parent / "artifacts",
        Path("/var/task") / "artifacts",
        Path("/var/task") / "api" / "artifacts",
        Path(os.environ.get("LAMBDA_TASK_ROOT", "")) / "artifacts",
        Path(os.environ.get("LAMBDA_TASK_ROOT", "")) / "api" / "artifacts",
    ]
    for path in candidates:
        if path and (path / "model.joblib").exists():
            return path
    return here / "artifacts"


ARTIFACTS_DIR = _resolve_artifacts_dir()


class PredictRequest(BaseModel):
    N: float = Field(..., ge=0, le=200)
    P: float = Field(..., ge=0, le=200)
    K: float = Field(..., ge=0, le=250)
    temperature: float = Field(..., ge=0, le=50)
    humidity: float = Field(..., ge=0, le=100)
    ph: float = Field(..., ge=3, le=10)
    rainfall: float = Field(..., ge=0, le=400)


def create_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df_enhanced = df.copy()
    df_enhanced["NPK_sum"] = df_enhanced["N"] + df_enhanced["P"] + df_enhanced["K"]
    df_enhanced["NP_ratio"] = df_enhanced["N"] / (df_enhanced["P"] + 1)
    df_enhanced["NK_ratio"] = df_enhanced["N"] / (df_enhanced["K"] + 1)
    df_enhanced["PK_ratio"] = df_enhanced["P"] / (df_enhanced["K"] + 1)
    df_enhanced["temp_humidity_interaction"] = (
        df_enhanced["temperature"] * df_enhanced["humidity"]
    )
    df_enhanced["temp_rainfall_interaction"] = (
        df_enhanced["temperature"] * df_enhanced["rainfall"]
    )
    df_enhanced["humidity_rainfall_interaction"] = (
        df_enhanced["humidity"] * df_enhanced["rainfall"]
    )
    return df_enhanced


def load_artifacts():
    global _model, _encoder, _scaler, _feature_names

    if _model is not None:
        return

    artifacts_dir = _resolve_artifacts_dir()
    model_path = artifacts_dir / "model.joblib"
    encoder_path = artifacts_dir / "label_encoder.joblib"
    scaler_path = artifacts_dir / "scaler.joblib"
    feature_names_path = artifacts_dir / "feature_names.joblib"

    missing = [p.name for p in (model_path, encoder_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts in {artifacts_dir}: {', '.join(missing)}. "
            "Run: python train.py"
        )

    _model = joblib.load(model_path)
    _encoder = joblib.load(encoder_path)
    _scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    _feature_names = (
        joblib.load(feature_names_path) if feature_names_path.exists() else None
    )


app = FastAPI(title="Smart Agro AI API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def run_prediction(request: PredictRequest) -> dict:
    load_artifacts()

    features = pd.DataFrame(
        {
            "N": [request.N],
            "P": [request.P],
            "K": [request.K],
            "temperature": [request.temperature],
            "humidity": [request.humidity],
            "ph": [request.ph],
            "rainfall": [request.rainfall],
        }
    )
    features_enhanced = create_enhanced_features(features)

    if _feature_names is not None:
        features_enhanced = features_enhanced.reindex(columns=_feature_names, fill_value=0)

    if _scaler is not None:
        features_scaled = _scaler.transform(features_enhanced)
    else:
        features_scaled = features_enhanced.values

    prediction_index = int(_model.predict(features_scaled)[0])
    prediction_label = _encoder.inverse_transform([prediction_index])[0]
    probabilities = _model.predict_proba(features_scaled)[0]
    class_labels = _encoder.inverse_transform(np.arange(len(probabilities)))

    return {
        "prediction": str(prediction_label),
        "probabilities": {
            str(label): float(prob)
            for label, prob in zip(class_labels, probabilities)
        },
        "model_version": "2.0",
    }


@app.get("/api/health")
def health():
    artifacts_dir = _resolve_artifacts_dir()
    return {
        "status": "ok",
        "service": "Smart Agro AI",
        "artifacts_dir": str(artifacts_dir),
        "model_exists": (artifacts_dir / "model.joblib").exists(),
    }


@app.get("/api")
def api_root():
    return {
        "name": "Smart Agro AI API",
        "version": "2.0",
        "predict_url": "/api/predict",
        "health_url": "/api/health",
    }


@app.get("/api/predict")
def predict_info():
    return {
        "documentation": "POST /api/predict with JSON body",
        "fields": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
    }


@app.post("/api/predict")
def predict(request: PredictRequest):
    try:
        return run_prediction(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# Serve UI when running locally (Vercel serves public/ via CDN)
_PUBLIC_DIR = Path(__file__).resolve().parent / "public"
if _PUBLIC_DIR.exists() and not os.environ.get("VERCEL"):
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_PUBLIC_DIR, html=True), name="static")
