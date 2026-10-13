from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd
from forecasting.features import FEATURES

class ForecastService:
    def __init__(self, artifact_dir: Path):
        self.model=joblib.load(artifact_dir/'demand_model.joblib')
        self.history=pd.read_csv(artifact_dir/'history.csv',parse_dates=['date'])

    def forecast(self, store_id:int, product_id:int, horizon:int=14, promo:int=0, price:float|None=None):
        hist=self.history[(self.history.store_id==store_id)&(self.history.product_id==product_id)].sort_values('date').copy()
        if hist.empty: raise ValueError('Unknown store/product combination')
        if price is None: price=float(hist.price.tail(30).median())
        units=list(hist.units.astype(float))
        last=hist.date.max(); out=[]
        for step in range(1,horizon+1):
            d=last+pd.Timedelta(days=step)
            row={'store_id':store_id,'product_id':product_id,'price':price,'promo':promo,'dow':d.dayofweek,'month':d.month,'dayofyear':d.dayofyear,'lag_1':units[-1],'lag_7':units[-7],'rolling_7':sum(units[-7:])/7}
            y=max(0,float(self.model.predict(pd.DataFrame([row])[FEATURES])[0]))
            units.append(y); out.append({'date':d.date().isoformat(),'forecast_units':round(y,1)})
        history=[{'date':r.date.date().isoformat(),'units':int(r.units)} for r in hist.tail(30).itertuples()]
        return {'history':history,'forecast':out,'store_id':store_id,'product_id':product_id,'assumed_price':price,'promo':promo}
