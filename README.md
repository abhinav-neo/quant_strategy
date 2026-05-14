# Quant Strategy ? Kalman + OU + HMM + Kelly

Renaissance-grade statistical arbitrage on BTC/ETH/SOL using
Kalman filter cointegration, Ornstein-Uhlenbeck MLE, Hidden Markov
Model regime detection, and Kelly position sizing.

## Architecture

```
modules/
    math_guards.py      Financial safety layer (no NaN/Inf reaches output)
    data_fetcher.py     Alpaca 1-min OHLCV fetcher + OHLC validation
    stationarity.py     ADF + Hurst exponent + Ljung-Box entry gate
    kalman_filter.py    Dynamic hedge ratio (Kalman state-space)
    ou_estimator.py     OU MLE parameters + Bertram optimal threshold
    hmm_regime.py       3-state HMM regime detection (Baum-Welch)
    kelly_sizer.py      Kelly fraction x HMM state x ATR risk target
    backtest.py         Full backtest engine on real OHLCV data
    walk_forward.py     Rolling out-of-sample validator
```

## Signal flow

```
Real 1-min OHLCV (Alpaca)
    -> Stationarity gate (ADF + Hurst + Ljung-Box)
    -> Kalman filter  -> spread X(t), beta(t)
    -> OU estimator   -> theta, mu, sigma, half_life, a*
    -> HMM inference  -> P(state | observations), kelly_scale
    -> Kelly sizer    -> position = min(Kelly x HMM, ATR risk target)
    -> Trade engine   -> entry/exit/stop/time-stop + PnL
    -> Walk-forward   -> rolling out-of-sample verdict
```

## Setup

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/quant_strategy.git
cd quant_strategy
pip install -r requirements.txt
```

### 2. Set Alpaca API keys

Paper trading keys work fine (no real money needed for backtesting).
Get them at: https://alpaca.markets -> Paper Trading -> API Keys

**Option A ? environment variables (recommended):**
```bash
# Windows
set ALPACA_KEY=your_paper_key
set ALPACA_SECRET=your_paper_secret

# Linux/Mac
export ALPACA_KEY=your_paper_key
export ALPACA_SECRET=your_paper_secret
```

**Option B ? edit run_backtest.py directly:**
```python
ALPACA_KEY    = "your_paper_key"
ALPACA_SECRET = "your_paper_secret"
```

### 3. Run tests

```bash
# Windows
set PYTHONPATH=E:\MyDevelopment\GitHub\quant_strategy
run_tests.bat

# Linux/Mac
export PYTHONPATH=$(pwd)
python -m modules.math_guards && python -m modules.backtest
```

All 9 modules must show `X/X tests passed` before running live.

### 4. Run backtest

```bash
set PYTHONPATH=E:\MyDevelopment\GitHub\quant_strategy
python run_backtest.py
```

Results saved to `results/` as `.txt` summary and `.csv` trade log.

## Deployment

### GitHub Actions (weekly automated backtest)

1. Push this repo to GitHub
2. Go to Settings -> Secrets -> Actions
3. Add secrets: `ALPACA_KEY` and `ALPACA_SECRET`
4. Go to Actions tab -> "Quant Strategy Backtest" -> Run workflow

The workflow runs every Sunday 6am UTC automatically.
Results appear as downloadable artifacts in the Actions tab.

### VM deployment (live paper trading)

For continuous live paper trading (requires always-on process):

**Recommended: Oracle Cloud Free Tier (permanently free)**
- VM.Standard.E2.1.Micro ? 1 OCPU, 1GB RAM
- Sufficient for this strategy
- Setup: Ubuntu 22.04, Python 3.12, clone repo, set env vars

**Paper trading script:** (coming in next phase)
```bash
python run_live.py  # watches 1-min bars, fires signals in real time
```

## Walk-forward verdicts

```
STRONG_EDGE   -> Median Sharpe >= 1.0, profitable >= 60% of windows
DEPLOYABLE    -> Median Sharpe >= 0.5, profitable >= 60% of windows
WEAK_EDGE     -> Profitable but Sharpe < 0.5
NOT_ROBUST    -> Profitable < 60% of windows
INSUFFICIENT  -> Too few valid windows to conclude
```

**Deploy only if verdict is STRONG_EDGE or DEPLOYABLE.**
Paper trade for minimum 4 weeks before using real capital.

## Test results

```
math_guards     43/43
data_fetcher    22/22
stationarity    24/24
kalman_filter   26/26
ou_estimator    33/33
hmm_regime      28/28
kelly_sizer     29/29
backtest        24/24
walk_forward    17/17
TOTAL          246/246
```

## Important notes

- HMM log-probabilities (e.g. -3400) are mathematically correct ?
  they are not errors. The system handles them correctly.
- safeN() is for financial ratios only (Sharpe, Kelly fraction, etc.)
  Never for log-probabilities or dollar amounts.
- Walk-forward on real data: expect 15-30 min for 9 months of 1-min bars.
  Most warnings during a run are expected (non-MR windows, few trades).
  Only ERROR level messages indicate real problems.

## Capital allocation

Recommended: $10,000-$50,000 allocation to this strategy.
Target: 25-60% annual return with Sharpe > 1.0.
This is one income stream ? not a retirement plan on its own.
