import numpy as np
import pandas as pd

import helper_monkey as hm
import markowitzify


def test_import_module():
    assert markowitzify is not None


def test_markowitz_optimizer(price_df):
    w = hm.markowitz(price_df)
    assert w.shape == (price_df.shape[1],)
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert np.all(w >= -1e-8)
    assert np.all(w <= 1 + 1e-8)


def test_nco_optimizer(price_df):
    cov = hm.cov_matrix(price_df)
    w = hm.optPort_nco(cov)
    assert w.shape[0] == price_df.shape[1]
    assert np.isclose(w.sum(), 1.0, atol=1e-6)


def test_portfolio_smoke(price_df):
    p = markowitzify.portfolio()
    p.portfolio = price_df
    p.cov = hm.cov_matrix(price_df)
    p.markowitz()
    assert isinstance(p.optimal, pd.DataFrame)
    assert p.optimal.shape == (1, price_df.shape[1])

    p.NCO()
    assert isinstance(p.nco, pd.DataFrame)
    assert p.nco.shape == (1, price_df.shape[1])


def test_stonks_helpers_smoke(ohlc_df):
    rsi_df = hm.RSI(ohlc_df, initial_lookback=14, lookback=14)
    assert 'RSI' in rsi_df.columns
    assert np.isfinite(rsi_df['RSI']).all()

    tp, up, lo = hm.bollinger(ohlc_df, n=20, m=2, log_=False)
    assert len(tp) == len(ohlc_df)
    assert len(up) == len(ohlc_df)
    assert len(lo) == len(ohlc_df)

    fract = hm.fractal_indicator(ohlc_df, n=20, min_max_lookback=14)
    assert 'Fractal_Indicator' in fract.columns


def test_monte_carlo_helpers(price_df):
    sim = hm.simulate(price_df['AAA'], days=30, iterations=50)
    roi = hm.ROI(sim)
    assert isinstance(roi, float)
    assert np.isfinite(roi)
