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


def test_nco_cluster_upper_bound_and_small_universe():
    cov4 = np.array(
        [
            [0.04, 0.01, 0.0, 0.0],
            [0.01, 0.03, 0.0, 0.0],
            [0.0, 0.0, 0.05, 0.02],
            [0.0, 0.0, 0.02, 0.06],
        ]
    )
    w4 = hm.optPort_nco(cov4, maxNumClusters=10)
    assert w4.shape == (4, 1)
    assert np.isclose(w4.sum(), 1.0, atol=1e-6)

    cov2 = np.array([[0.04, 0.01], [0.01, 0.03]])
    w2 = hm.optPort_nco(cov2, maxNumClusters=10)
    assert w2.shape == (2, 1)
    assert np.isclose(w2.sum(), 1.0, atol=1e-6)

    mu2 = np.array([0.01, 0.02])
    w2_mu = hm.optPort_nco(cov2, mu=mu2, maxNumClusters=10)
    assert w2_mu.shape == (2, 1)
    assert np.isclose(w2_mu.sum(), 1.0, atol=1e-6)


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


def test_datareader_provider_errors_when_unavailable(monkeypatch):
    monkeypatch.setattr(hm, 'wb', None)
    with np.testing.assert_raises(ImportError):
        hm.import_stock_data_DataReader(start='2020-01-01', tickers=['AAA'])

    with np.testing.assert_raises(ImportError):
        hm.import_high_low(start='2020-01-01', ticker='AAA', provider='datareader')
