from __future__ import annotations
import numpy as np
import pandas as pd

def generate_customer_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, n)
    tenure = rng.integers(0, 73, n)
    monthly = np.clip(rng.normal(72, 25, n), 18, 150).round(2)
    support = rng.poisson(1.5, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55,.25,.20])
    internet = rng.choice(["Fiber optic", "DSL", "No"], n, p=[.5,.4,.1])
    paperless = rng.choice(["Yes", "No"], n, p=[.7,.3])
    payment = rng.choice(["Electronic check", "Credit card", "Bank transfer", "Mailed check"], n)
    senior = (age >= 65).astype(int)
    total = np.maximum(0, monthly * tenure + rng.normal(0, 200, n)).round(2)
    logit = (-2.6 + 1.2*(contract=="Month-to-month") + .65*(internet=="Fiber optic")
             + .55*(payment=="Electronic check") + .20*support - .035*tenure + .008*(monthly-70)
             + .25*senior)
    prob = 1/(1+np.exp(-logit))
    churn = rng.binomial(1, prob)
    return pd.DataFrame({"age":age,"tenure_months":tenure,"monthly_charges":monthly,
        "total_charges":total,"support_tickets":support,"contract_type":contract,
        "internet_service":internet,"paperless_billing":paperless,"payment_method":payment,
        "senior_citizen":senior,"churn":churn})
