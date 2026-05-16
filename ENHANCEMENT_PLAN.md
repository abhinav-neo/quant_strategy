# Quant Strategy v2.0 — Enhancement Plan

**Honest priority ranking based on expected return impact vs implementation cost.**

## Where we are
- BTC/ETH 1h: +2.8% to +3.9% per year (real, consistent across 2024 + 2025)
- BTC/SOL 1h: +3.3% to +6.1% (small sample, real but uncertain)
- ETH/SOL 1h: +4.1% to +11.5% (small sample, real but uncertain)
- Average across 3 pairs: ~7% per year, modest Sharpe

## Where we want to go
- 15-25% per year with realistic costs and walk-forward validation
- Sharpe > 1.5 in live trading
- Multiple uncorrelated edges so 1 strategy failing does not kill portfolio

---

## Three paths to higher returns

### Path A: Better statistics on same data (your original list)
**Expected impact: +2-5% per year.** **Effort: 60-100 hours.**

The enhancements you listed primarily reduce risk and overfitting. They do NOT generate new alpha — they just make existing alpha more reliable. Best for a strategy that is already deployed and you want to harden.

### Path B: More pairs (portfolio breadth)
**Expected impact: +10-20% per year via diversification.** **Effort: 20-40 hours.**

The diversification math is real. 10 uncorrelated 5% strategies gives ~50% return at lower drawdown than any single one. Limited by available cointegrated pairs on your venue.

### Path C: Better venue (lower fees)
**Expected impact: +15-30% per year.** **Effort: 40-60 hours.**

Move from Alpaca (40 bps RT) to Hyperliquid/Binance perps (2-5 bps RT). This is the single biggest lever — cuts cost by 8-20x, transforming losing 1-min strategies into winners.

### Recommendation
Do **C first**, then **B**, then **A**. C unlocks the strategy space, B multiplies it, A hardens what works.

---

## Phased Implementation Plan

### Phase 0 — Foundation (1-2 days)
**Goal:** Validate Path B before any Path A work.

**Action:** Add 4 more crypto pairs from Alpaca to existing framework:
- BTC/USDC, ETH/USDC (likely tightly cointegrated with USD versions)
- LTC/USD if available
- DOGE/USD, AVAX/USD if available

**Files to change:**
- `run_backtest.py`: extend PAIRS list
- No code changes needed if Alpaca supports the symbols

**Decision gate:** If 6+ pairs give portfolio Sharpe > 1.5 with realistic costs, proceed to Phase 1. If not, the framework hits a ceiling — go to Path C.

---

### Phase 1 — Statistical Robustness (3-5 days)
**Goal:** Items 1, 6, 7 from your list. Highest impact-per-hour from your enhancements.

#### 1.1 Rolling cointegration validation (CRITICAL)
**Why:** Current code does NOT validate that pairs are still cointegrated during the test period. If cointegration breaks (regime shift), the strategy will trade noise.

**New module:** `modules/cointegration_monitor.py`
```python
# Pseudo-code
class CointegrationMonitor:
    def __init__(self, window=500, p_threshold=0.05):
        self.window = window
        self.p_threshold = p_threshold
    
    def is_cointegrated(self, lp_a, lp_b) -> bool:
        # Engle-Granger test on rolling window
        # Returns False if p > threshold OR if Johansen rank != 1
        pass
    
    def should_trade(self) -> bool:
        # Combines cointegration + structural break check
        pass
```

**Integration:** `backtest.py` calls `is_cointegrated()` every 500 bars; skips trades if False.

**Expected impact:** Cuts ~30% of bad trades that happen during regime breakdowns.

#### 1.2 Structural break detection (CRITICAL)
**Why:** Same as above. Detect when the cointegration relationship fundamentally changes (e.g., ETH/SOL after a major SOL outage).

**New module:** `modules/structural_breaks.py`
- Implements CUSUM test on Kalman residuals
- Quandt-Andrews break point test
- Bai-Perron multiple break detection

**Expected impact:** -50% drawdown during regime shifts.

#### 1.3 Robust MAD z-score (MEDIUM)
**Why:** Standard z-score is sensitive to outliers. Median Absolute Deviation (MAD) is robust.

**Change in:** `modules/backtest.py`
```python
# Replace
z = (spread - ou.mu) / ou.sigma_eq

# With
mad = 1.4826 * np.median(np.abs(recent_spreads - np.median(recent_spreads)))
z_robust = (spread - np.median(recent_spreads)) / max(mad, MIN_DENOMINATOR)
```

**Expected impact:** Modest. Reduces false signals during outlier events by ~10%.

#### 1.4 Half-life estimation per window (MEDIUM)
**Why:** Already partially implemented but not used dynamically. Make MAX_HOLD adapt to half-life.

**Change in:** `modules/backtest.py`
```python
MAX_BARS_HELD = int(min(120, max(10, ou.half_life * 5)))
```

**Expected impact:** Modest. Better risk management on slow signals.

---

### Phase 2 — Execution Modeling (2-3 days)
**Goal:** Items 2, 4, 6 — make backtest mirror live trading reality.

#### 2.1 Execution cost model
**New module:** `modules/execution_model.py`
- Linear market impact: cost = base_fee + impact_coeff * sqrt(position_usd / avg_volume)
- Adverse selection: when entering at 3σ, expected fill is at 3.05σ-3.2σ (worse)
- Time-of-day adjustment: spreads wider at 04:00 UTC than 14:00 UTC

```python
class ExecutionCostModel:
    def __init__(self, venue="alpaca_crypto"):
        self.venues = {
            "alpaca_crypto":   {"maker": 0.0015, "taker": 0.0025, "impact": 0.0001},
            "hyperliquid":     {"maker": -0.00005, "taker": 0.00045, "impact": 0.00003},
            "binance_perps":   {"maker": 0.00018, "taker": 0.00036, "impact": 0.00002},
            "kraken_pro":      {"maker": 0.0016, "taker": 0.0026, "impact": 0.00008},
        }
    
    def estimate_total_cost(self, position_usd, avg_volume, zscore):
        venue = self.venues[self.venue]
        fee_rt = (venue["maker"] + venue["taker"])  # round-trip
        impact = venue["impact"] * np.sqrt(position_usd / max(avg_volume,1))
        adverse = 0.0001 * (abs(zscore) - 1.0)  # 1bp per sigma above 1
        return fee_rt + impact + adverse
```

**Expected impact:** This is realism, not return. Will REDUCE backtest returns by 10-30% but match live reality.

#### 2.2 Slippage simulation (advanced)
**Add to backtest.py:** simulate that entry price is NOT the signal-bar close — use next bar's open + adverse drift.

---

### Phase 3 — Portfolio Construction (3-4 days)
**Goal:** Items 3 — combine multiple pairs intelligently. THIS IS THE BIG RETURN MULTIPLIER.

#### 3.1 Portfolio optimizer
**New module:** `modules/portfolio_optimizer.py`
- Equal-risk allocation across pairs (default)
- Optional: HRP (Hierarchical Risk Parity) for better diversification
- Position sizing: each pair gets equal-vol budget, not equal-dollar

```python
class PortfolioOptimizer:
    def __init__(self, method="equal_risk"):
        self.method = method
    
    def allocate(self, pair_results: dict) -> dict:
        # Returns weight per pair such that total portfolio vol target
        # is met (e.g., 10% annualised volatility budget)
        pass
```

**Expected impact:** +5-15% per year if you have 6+ low-correlation pairs.

#### 3.2 Multi-pair backtest runner
**New module:** `run_portfolio_backtest.py`
- Runs all pairs in parallel (per-bar)
- Tracks portfolio-level equity, drawdown, Sharpe
- Caps simultaneous exposure (e.g., max 3 active trades)

---

### Phase 4 — Validation Framework (3-5 days)
**Goal:** Items 5, 8 — prove robustness.

#### 4.1 Monte Carlo validation
**New module:** `modules/monte_carlo.py`
- Bootstrap trade returns 10,000 times
- Compute Sharpe distribution, max drawdown distribution
- Detect overfitting via deflated Sharpe ratio

#### 4.2 Walk-forward optimization framework
**New module:** `modules/walk_forward_opt.py`
- For each window: optimize parameters on train set, apply to test set
- Robust against parameter drift
- Detects parameter stability across regimes

#### 4.3 Regime-conditional OU
**Enhancement to ou_estimator.py:**
- Fit separate OU params per HMM state
- Trade only when high-mean-reversion state is detected
- Skip during breakdown regime

---

### Phase 5 — Production Hardening (2-3 days)
**Goal:** Items 2, 8 — make it deployable.

- Live data feed with reconnection handling
- Position reconciliation between expected and actual
- Drawdown-based kill switch
- Email/SMS alerts on anomalies
- Daily P&L reconciliation report

---

## New Architecture Layout

```
quant_strategy/
├── modules/
│   ├── math_guards.py            # existing
│   ├── data_fetcher.py           # existing  
│   ├── stationarity.py           # existing
│   ├── kalman_filter.py          # existing
│   ├── ou_estimator.py           # ENHANCED: regime-conditional
│   ├── hmm_regime.py             # existing
│   ├── kelly_sizer.py            # existing
│   ├── backtest.py               # ENHANCED: MAD z-score, dynamic MAX_HOLD
│   ├── walk_forward.py           # existing
│   ├── cointegration_monitor.py  # NEW
│   ├── structural_breaks.py      # NEW
│   ├── execution_model.py        # NEW
│   ├── portfolio_optimizer.py    # NEW
│   ├── monte_carlo.py            # NEW
│   ├── walk_forward_opt.py       # NEW
│   └── live_trader.py            # NEW (Phase 5)
├── tests/
│   ├── test_unit/                # existing 245 tests
│   └── test_integration/         # NEW: end-to-end scenarios
├── results/
├── data/
├── notebooks/
│   └── monte_carlo_analysis.ipynb  # NEW
├── run_backtest.py              # existing
├── run_portfolio_backtest.py    # NEW
├── run_live.py                  # NEW
└── README.md
```

---

## Dependency Changes

Add to `requirements.txt`:
```
arch>=6.2                    # ARCH/GARCH for volatility
ruptures>=1.1                # Structural break detection
statsmodels>=0.14            # Already there, used for Johansen test
scikit-learn>=1.3            # Already there
joblib>=1.3                  # NEW: parallel pair execution
cvxpy>=1.4                   # NEW: portfolio optimization
matplotlib>=3.7              # NEW: visualization
jupyter>=1.0                 # NEW: notebooks
```

---

## Testing Framework Additions

**New: Integration tests** (`tests/test_integration/`)
- `test_end_to_end_btc_eth.py`: full pipeline with real data
- `test_portfolio_3_pairs.py`: 3-pair portfolio backtest
- `test_regime_transition.py`: simulate regime shift, verify graceful behavior
- `test_kill_switch.py`: drawdown breaching threshold halts trading

**New: Property-based tests** (`tests/test_properties/`)
- Using `hypothesis` library
- Auto-generates edge cases for OU fits, Kalman updates, etc.

**Performance benchmarks**
- Backtest must complete in <60s on 9 months of 1-min data
- Walk-forward must complete in <10min for 100 windows

---

## Recommended Phased Execution

**Weekend 1:** Phase 0 (validate Path B is real)
**Weekends 2-3:** Phase 1 (statistical robustness)
**Weekend 4:** Phase 2 (execution modeling)  
**Weekends 5-6:** Phase 3 (portfolio construction) — biggest ROI
**Weekend 7:** Phase 4 (validation)
**Weekend 8:** Phase 5 (live deployment) — paper trading first

**Critical:** Do NOT skip Phase 0. If 6+ pairs do not give Sharpe > 1.5, the entire Phase 1-5 effort produces marginal improvement. The math is in the diversification.

---

## What I would do in your position

Given EB1 is priority and you have limited weekend time:

1. **Weekend 1:** Phase 0 only. 4 hours. Validates premise.
2. **If Phase 0 passes:** Continue with Phases 1-5 at 1 weekend each
3. **If Phase 0 fails:** Switch to Path C (Hyperliquid venue) — 1 weekend to port data_fetcher, 1 weekend to validate. Then re-run on cheap venue.

This gives you a Go/No-Go decision in 4 hours of work.
