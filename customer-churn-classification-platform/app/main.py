from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.schemas import CustomerFeatures, PredictionResponse
from ml.train import train_model

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FRONTEND = ROOT / "frontend"
MODEL_PATH = ART / "churn_model.joblib"
if not MODEL_PATH.exists():
    train_model()
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Customer Churn Intelligence API", version="1.0.0")
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

@app.get("/")
def home():
    return FileResponse(FRONTEND / "index.html")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

@app.get("/model-info")
def model_info():
    p = ART / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerFeatures):
    row = pd.DataFrame([payload.model_dump()])
    prob = float(model.predict_proba(row)[0, 1])
    pred = prob >= 0.5
    if prob >= 0.75:
        band, rec = "High", "Prioritize proactive retention outreach and personalized incentive."
    elif prob >= 0.45:
        band, rec = "Medium", "Monitor engagement and consider a targeted value-based offer."
    else:
        band, rec = "Low", "Maintain regular engagement; no immediate retention intervention required."
    return PredictionResponse(
        churn_probability=round(prob, 4), churn_prediction=pred,
        risk_band=band, recommendation=rec
    )
