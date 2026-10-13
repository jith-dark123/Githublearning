from __future__ import annotations
import json
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from forecasting.data import generate_retail_data
from forecasting.features import add_features, FEATURES

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts'; ART.mkdir(exist_ok=True)

def train_model(seed=42):
    raw=generate_retail_data(seed=seed)
    feat=add_features(raw)
    cutoff=feat['date'].quantile(.8)
    train=feat[feat.date<=cutoff]; test=feat[feat.date>cutoff]
    model=RandomForestRegressor(n_estimators=220,max_depth=14,min_samples_leaf=2,n_jobs=-1,random_state=seed)
    model.fit(train[FEATURES],train['units'])
    pred=model.predict(test[FEATURES])
    metrics={'mae':round(float(mean_absolute_error(test.units,pred)),3),'rmse':round(float(mean_squared_error(test.units,pred)**.5),3),'train_rows':len(train),'test_rows':len(test)}
    joblib.dump(model,ART/'demand_model.joblib')
    raw.to_csv(ART/'history.csv',index=False)
    (ART/'metrics.json').write_text(json.dumps(metrics,indent=2))
    return metrics
if __name__=='__main__': print(json.dumps(train_model(),indent=2))
