# Markowitzify

Markowitzify is a lightweight Python library for portfolio optimization (`portfolio`) and technical-analysis helpers (`stonks`).

It includes:
- **Portfolio analytics**: Markowitz optimization, NCO optimization, Sharpe ratio, Hurst exponent, Monte Carlo simulation, trend scan.
- **Stonks analytics**: Fractal indicator, Bollinger bands, RSI, signal generation, and basic strategy backtesting.

## Installation

```bash
pip install markowitzify
```

For local development:

```bash
pip install -e ".[dev]"
```

Optional market-data provider extras:

```bash
pip install -e ".[data]"
```

## Quickstart (offline-safe)

```python
import numpy as np
import pandas as pd
import markowitzify
import helper_monkey as hm

rng = np.random.default_rng(42)
returns = rng.normal(0.0005, 0.01, size=(200, 4))
prices = 100 * np.exp(np.cumsum(returns, axis=0))
df = pd.DataFrame(prices, columns=["AAA", "BBB", "CCC", "DDD"])

p = markowitzify.portfolio()
p.portfolio = df
p.cov = hm.cov_matrix(df)

p.markowitz()
print(p.optimal)

p.NCO()
print(p.nco)
```

## API Overview

### `portfolio`

```python
p = markowitzify.portfolio(API_KEY=None, verbose=False)
```

Key methods/attributes:
- `build_portfolio(TKR_list, time_delta, end_date=None, datareader=True, provider="auto")`
  - `provider="auto"` prefers `yfinance` if installed, otherwise `pandas_datareader`.
- `build_TSP()` (depends on external endpoint availability).
- `import_portfolio(input_path, filename="portfolio.csv", dates_kw="date")`
- `markowitz()` → sets `p.optimal`
- `NCO()` / `optimize_nco()` → sets `p.nco`
- `hurst()`, `sharpe_ratio()`, `simulate()`, `trend()`

### `stonks`

```python
s = markowitzify.stonks("AAPL", provider="auto")
```

Key methods/attributes:
- `fractal()`
- `bollinger()`
- `RSI()`
- `signal()`
- `strategize()`

## Testing / Contributing

Run checks locally:

```bash
ruff check .
pytest -q
```

## Notes and limitations

- External market APIs/providers may change or break over time.
- Tests are intentionally offline and do not rely on Yahoo/MarketStack/TSP network calls.
- MarketStack-based portfolio building requires a valid API key.
