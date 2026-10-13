from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app
from forecasting.data import generate_retail_data
from forecasting.features import add_features

def test_feature_pipeline_creates_lags():
    df=add_features(generate_retail_data(days=30,stores=1,products=1))
    assert {'lag_1','lag_7','rolling_7'} <= set(df.columns)

def test_forecast_endpoint():
    c=TestClient(app)
    out=c.post('/forecast',json={'store_id':1,'product_id':1,'horizon':7,'promo':0}).json()
    assert len(out['forecast'])==7
    assert all(x['forecast_units']>=0 for x in out['forecast'])
