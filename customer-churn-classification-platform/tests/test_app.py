from fastapi.testclient import TestClient
from app.main import app

def test_health():
    assert TestClient(app).get('/health').status_code == 200

def test_predict():
    payload={"age":42,"tenure_months":8,"monthly_charges":105,"total_charges":840,"support_tickets":4,"contract_type":"Month-to-month","internet_service":"Fiber optic","paperless_billing":"Yes","payment_method":"Electronic check","senior_citizen":0}
    r=TestClient(app).post('/predict',json=payload)
    assert r.status_code==200
    assert 0 <= r.json()['churn_probability'] <= 1
