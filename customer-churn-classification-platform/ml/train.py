from __future__ import annotations
import json
from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ml.data import generate_customer_data

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"; ART.mkdir(exist_ok=True)
NUM = ["age","tenure_months","monthly_charges","total_charges","support_tickets","senior_citizen"]
CAT = ["contract_type","internet_service","paperless_billing","payment_method"]

def train_model(seed: int = 42):
    df = generate_customer_data(seed=seed); X=df.drop(columns="churn"); y=df.churn
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=seed)
    prep=ColumnTransformer([("num",StandardScaler(),NUM),("cat",OneHotEncoder(handle_unknown="ignore"),CAT)])
    pipe=Pipeline([("prep",prep),("model",RandomForestClassifier(n_estimators=280,max_depth=12,min_samples_leaf=3,class_weight="balanced",random_state=seed,n_jobs=-1))])
    pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); proba=pipe.predict_proba(Xte)[:,1]
    metrics={"accuracy":accuracy_score(yte,pred),"precision":precision_score(yte,pred),"recall":recall_score(yte,pred),"f1":f1_score(yte,pred),"roc_auc":roc_auc_score(yte,proba),"train_rows":len(Xtr),"test_rows":len(Xte)}
    metrics={k:(round(float(v),4) if isinstance(v,float) else int(v)) for k,v in metrics.items()}
    joblib.dump(pipe,ART/"churn_model.joblib"); (ART/"metrics.json").write_text(json.dumps(metrics,indent=2))
    return metrics
if __name__=="__main__": print(json.dumps(train_model(),indent=2))
