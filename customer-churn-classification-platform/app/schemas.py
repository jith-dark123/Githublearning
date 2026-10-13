from pydantic import BaseModel, Field

class CustomerFeatures(BaseModel):
    age: int = Field(ge=18, le=100)
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0)
    total_charges: float = Field(ge=0)
    support_tickets: int = Field(ge=0, le=50)
    contract_type: str
    internet_service: str
    paperless_billing: str
    payment_method: str
    senior_citizen: int = Field(ge=0, le=1)

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_band: str
    recommendation: str
