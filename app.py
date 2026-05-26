"""FastAPI app for crop recommendation (Vercel + local)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

_model: Any = None
_encoder: Any = None
_scaler: Any = None
_feature_names: Any = None


class PredictRequest(BaseModel):
    N: float = Field(..., ge=0, le=200)
    P: float = Field(..., ge=0, le=200)
    K: float = Field(..., ge=0, le=250)
    temperature: float = Field(..., ge=0, le=50)
    humidity: float = Field(..., ge=0, le=100)
    ph: float = Field(..., ge=3, le=10)
    rainfall: float = Field(..., ge=0, le=400)


def _create_enhanced_features(df):
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


def load_artifacts() -> None:
    global _model, _encoder, _scaler, _feature_names

    if _model is not None:
        return

    import joblib

    model_path = ARTIFACTS_DIR / "model.joblib"
    encoder_path = ARTIFACTS_DIR / "label_encoder.joblib"
    scaler_path = ARTIFACTS_DIR / "scaler.joblib"
    feature_names_path = ARTIFACTS_DIR / "feature_names.joblib"

    missing = [p.name for p in (model_path, encoder_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing in {ARTIFACTS_DIR}: {', '.join(missing)}. Run: python train.py"
        )

    _model = joblib.load(model_path)
    _encoder = joblib.load(encoder_path)
    _scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    _feature_names = (
        joblib.load(feature_names_path) if feature_names_path.exists() else None
    )


app = FastAPI(title="Smart Agro AI API", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "artifacts": str(ARTIFACTS_DIR),
        "model_exists": (ARTIFACTS_DIR / "model.joblib").exists(),
    }


@app.get("/api")
def api_root():
    return {"predict_url": "/api/predict", "health_url": "/api/health"}


@app.get("/api/predict")
def predict_info():
    return {
        "documentation": "POST /api/predict",
        "fields": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
    }


@app.post("/api/predict")
def predict(request: PredictRequest):
    try:
        import numpy as np
        import pandas as pd

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
        features_enhanced = _create_enhanced_features(features)

        if _feature_names is not None:
            features_enhanced = features_enhanced.reindex(
                columns=_feature_names, fill_value=0
            )

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
            "model_version": "2.1",
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


_PUBLIC = BASE_DIR / "public"
if _PUBLIC.exists() and not os.environ.get("VERCEL"):
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_PUBLIC, html=True), name="static")
