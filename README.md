# Quant Strategy — Kalman + OU + HMM + Kelly
## Architecture
Renaissance-grade framework for crypto stat arb on BTC/ETH/SOL.

## Module build order (each tested before next starts)
0. modules/math_guards.py    -- DONE (43/43 tests pass)
1. modules/data_fetcher.py   -- next
2. modules/stationarity.py   -- ADF, Hurst, Ljung-Box
3. modules/kalman_filter.py  -- dynamic hedge ratio
4. modules/ou_estimator.py   -- MLE for theta, mu, sigma
5. modules/hmm_regime.py     -- Baum-Welch 3-state HMM
6. modules/kelly_sizer.py    -- OU-derived Kelly fraction
7. modules/backtest.py       -- full engine on real data
8. modules/walk_forward.py   -- out-of-sample validation

## Rule
No module is started until the previous one passes ALL unit tests.
No floating point result reaches output without passing through math_guards.safeN().

## Capital allocation
Starting: -
Target:   25-60% annual return with Sharpe > 1.0
