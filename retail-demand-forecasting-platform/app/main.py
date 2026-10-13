from __future__ import annotations
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from forecasting.train import train_model
from forecasting.service import ForecastService

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts'; FRONTEND=ROOT/'frontend'
if not (ART/'demand_model.joblib').exists(): train_model()
service=ForecastService(ART)
app=FastAPI(title='Retail Demand Forecasting API',version='1.0.0'); app.mount('/static',StaticFiles(directory=FRONTEND),name='static')
class ForecastRequest(BaseModel):
    store_id:int=Field(ge=1,le=3); product_id:int=Field(ge=1,le=4); horizon:int=Field(default=14,ge=1,le=30); promo:int=Field(default=0,ge=0,le=1); price:float|None=Field(default=None,gt=0)
@app.get('/')
def home(): return FileResponse(FRONTEND/'index.html')
@app.get('/health')
def health(): return {'status':'ok'}
@app.get('/metrics')
def metrics(): return json.loads((ART/'metrics.json').read_text())
@app.post('/forecast')
def forecast(req:ForecastRequest):
    try: return service.forecast(**req.model_dump())
    except ValueError as e: raise HTTPException(404,str(e))
