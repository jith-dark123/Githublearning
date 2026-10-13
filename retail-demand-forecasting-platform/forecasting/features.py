from __future__ import annotations
import pandas as pd

FEATURES = ['store_id','product_id','price','promo','dow','month','dayofyear','lag_1','lag_7','rolling_7']

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.sort_values(['store_id','product_id','date']).copy()
    x['dow'] = x['date'].dt.dayofweek
    x['month'] = x['date'].dt.month
    x['dayofyear'] = x['date'].dt.dayofyear
    g = x.groupby(['store_id','product_id'])['units']
    x['lag_1'] = g.shift(1)
    x['lag_7'] = g.shift(7)
    x['rolling_7'] = g.transform(lambda s: s.shift(1).rolling(7).mean())
    return x.dropna().reset_index(drop=True)
