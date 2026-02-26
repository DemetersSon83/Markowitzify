import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def price_df():
    rng = np.random.default_rng(42)
    n_days = 200
    n_assets = 4
    returns = rng.normal(loc=0.0005, scale=0.01, size=(n_days, n_assets))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    idx = pd.date_range('2022-01-01', periods=n_days, freq='D')
    cols = ['AAA', 'BBB', 'CCC', 'DDD']
    return pd.DataFrame(prices, index=idx, columns=cols)


@pytest.fixture
def ohlc_df(price_df):
    base = price_df['AAA']
    out = pd.DataFrame(index=price_df.index)
    out['Adj Close'] = base
    out['Open'] = base * 0.998
    out['high'] = base * 1.01
    out['low'] = base * 0.99
    return out
