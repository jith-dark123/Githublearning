from __future__ import annotations
import numpy as np
import pandas as pd

def generate_retail_data(days: int = 540, stores: int = 3, products: int = 4, seed: int = 42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2025-01-01', periods=days, freq='D')
    rows = []
    for s in range(1, stores+1):
        for p in range(1, products+1):
            base = 28 + 6*p + 3*s
            trend = rng.uniform(0.002, 0.02)
            for i, d in enumerate(dates):
                weekend = int(d.dayofweek >= 5)
                promo = int(rng.random() < 0.12)
                price = round((7 + p*1.8) * (0.88 if promo else 1.0) + rng.normal(0, .15), 2)
                seasonal = 7*np.sin(2*np.pi*d.dayofyear/365) + 4*np.sin(2*np.pi*d.dayofweek/7)
                units = base + seasonal + 10*promo + 4*weekend + trend*i + rng.normal(0, 4)
                rows.append([d, s, p, price, promo, max(0, round(units))])
    return pd.DataFrame(rows, columns=['date','store_id','product_id','price','promo','units'])
