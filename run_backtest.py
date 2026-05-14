#!/usr/bin/env python3
"""
run_backtest.py
================
Entry point for the quant strategy pipeline.

Steps
-----
1. Fetch real 1-minute OHLCV bars from Alpaca
2. Run stationarity tests to confirm edge exists
3. Run walk-forward validation across rolling windows
4. Print full results report

Usage
-----
    python run_backtest.py

Before running:
    Set ALPACA_KEY and ALPACA_SECRET below (paper trading keys are fine).
    Or set them as environment variables and leave the defaults below.

Output
------
    Console: full results summary with verdict
    File:    results/walk_forward_YYYYMMDD.txt
    File:    results/trades_YYYYMMDD.csv
"""

import logging
import os
import sys
import math
from datetime import datetime, timezone

import numpy  as np
import pandas as pd

# ?? Setup logging ??????????????????????????????????????????????????????????
logging.basicConfig(
    level  = logging.WARNING,
    format = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S",
)
# Keep backtest progress visible at INFO, suppress math_guards noise
logging.getLogger("backtest").setLevel(logging.INFO)
logging.getLogger("walk_forward").setLevel(logging.INFO)
logging.getLogger("run_backtest").setLevel(logging.INFO)
# These fire thousands of times on real data - suppress to ERROR only
logging.getLogger("math_guards").setLevel(logging.ERROR)
logging.getLogger("ou_estimator").setLevel(logging.ERROR)
logging.getLogger("kalman_filter").setLevel(logging.ERROR)
logging.getLogger("hmm_regime").setLevel(logging.ERROR)
logging.getLogger("kelly_sizer").setLevel(logging.ERROR)
logging.getLogger("stationarity").setLevel(logging.ERROR)
log = logging.getLogger("run_backtest")

# ?? Configuration ??????????????????????????????????????????????????????????
# Replace with your Alpaca paper trading keys
# Or set environment variables ALPACA_KEY and ALPACA_SECRET
ALPACA_KEY    = os.environ.get("ALPACA_KEY",    "PKHWEQLLRWG52N7SUYH3EKTV3Y")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "EF9t8Hx2Hc7h5eqfio6Ena6HP7enC9PopFyMR8a4Bbit")

# Date range -- override via START_DATE env var in CI
_start_env = os.environ.get("START_DATE", "")
if _start_env:
    try:
        _sd = datetime.strptime(_start_env, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        START_DATE = _sd
        END_DATE   = datetime.now(timezone.utc).replace(hour=0,minute=0,second=0,microsecond=0)
    except ValueError:
        START_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)
        END_DATE   = datetime(2024, 10, 1, tzinfo=timezone.utc)
else:
    START_DATE = datetime(2024, 1, 1,  tzinfo=timezone.utc)
    END_DATE   = datetime(2024, 10, 1, tzinfo=timezone.utc)

# Pairs -- override via PAIRS env var: "btc_eth", "btc_sol", "eth_sol", "all"
_pairs_env = os.environ.get("PAIRS", "all").lower()
_ALL_PAIRS = [
    ("BTC/USD", "ETH/USD"),
    ("BTC/USD", "SOL/USD"),
    ("ETH/USD", "SOL/USD"),
]
PAIRS = {
    "btc_eth": [("BTC/USD","ETH/USD")],
    "btc_sol": [("BTC/USD","SOL/USD")],
    "eth_sol": [("ETH/USD","SOL/USD")],
    "all":     _ALL_PAIRS,
}.get(_pairs_env, _ALL_PAIRS)

# Strategy parameters
INITIAL_CAPITAL = 10_000.0   # dollars per pair
RISK_PCT        = 0.01        # 1% capital at risk per trade
TRAIN_BARS      = 2000        # ~33 hours for training window
TEST_BARS       = 500         # ~8 hours per test window
STEP_BARS       = 500         # advance by one test window

# Output directory
RESULTS_DIR = r"E:\MyDevelopment\GitHub\quant_strategy\results"


def check_keys():
    """Validate API keys are set before making any API calls."""
    if "YOUR_PAPER_KEY" in ALPACA_KEY or "YOUR_PAPER_SECRET" in ALPACA_SECRET:
        log.error("="*60)
        log.error("API KEYS NOT SET")
        log.error("Edit run_backtest.py and set ALPACA_KEY and ALPACA_SECRET")
        log.error("Or set environment variables:")
        log.error("  set ALPACA_KEY=your_key")
        log.error("  set ALPACA_SECRET=your_secret")
        log.error("="*60)
        return False
    return True


def fetch_data(fetcher, symbol, start, end):
    """Fetch and validate bars for one symbol. Returns DataFrame or None."""
    log.info(f"Fetching {symbol} from {start.date()} to {end.date()}...")
    result = fetcher.fetch_bars(symbol, start, end)
    log.info(result.summary())
    if not result.ok:
        log.error(f"Failed to fetch {symbol}: {result.error_message}")
        return None
    df = result.bars[result.bars["valid_bar"]].copy()
    log.info(f"{symbol}: {len(df):,} valid bars ready")
    return df


def save_results(wfr, symbol_a, symbol_b, results_dir):
    """Save walk-forward summary and trade log to files."""
    os.makedirs(results_dir, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d_%H%M%S")
    pair  = f"{symbol_a.replace('/','_')}_{symbol_b.replace('/','_')}"

    # Summary text
    summary_path = os.path.join(results_dir, f"wf_{pair}_{today}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(wfr.summary())
        f.write("\n\nPer-window detail:\n")
        f.write("-" * 60 + "\n")
        for w in wfr.windows:
            if w.is_valid:
                bt = w.backtest
                f.write(
                    f"Window {w.window_idx:2d}: "
                    f"return={bt.total_return*100:+.2f}%  "
                    f"Sharpe={bt.sharpe:.3f}  "
                    f"trades={bt.total_trades}  "
                    f"{'PROFIT' if w.is_profitable else 'LOSS'}\n"
                )
    log.info(f"Summary saved: {summary_path}")

    # Trade log CSV (best window only)
    best_w = max(
        (w for w in wfr.windows if w.is_valid),
        key=lambda w: w.backtest.sharpe,
        default=None,
    )
    if best_w and best_w.backtest.trades:
        trades_path = os.path.join(results_dir, f"trades_{pair}_{today}.csv")
        rows = []
        for t in best_w.backtest.trades:
            rows.append({
                "bar_entry":     t.bar_entry,
                "bar_exit":      t.bar_exit,
                "direction":     "long" if t.direction == 1 else "short",
                "entry_spread":  round(t.entry_spread, 8),
                "exit_spread":   round(t.exit_spread, 8),
                "position_usd":  round(t.position_usd, 2),
                "raw_pnl":       round(t.raw_pnl, 2),
                "net_pnl":       round(t.net_pnl, 2),
                "trade_return":  round(t.trade_return * 100, 4),
                "exit_reason":   t.exit_reason,
                "regime_state":  t.regime_state,
                "half_life_bars": round(t.half_life, 2),
                "capital_after": round(t.capital_after, 2),
            })
        pd.DataFrame(rows).to_csv(trades_path, index=False)
        log.info(f"Trade log saved: {trades_path}")

    return summary_path


def run_pair(fetcher, symbol_a, symbol_b, df_cache):
    """Run full pipeline for one pair. Returns WalkForwardResult."""
    from modules.stationarity import run_all_tests
    from modules.walk_forward import run_walk_forward

    log.info("=" * 60)
    log.info(f"PAIR: {symbol_a} / {symbol_b}")
    log.info("=" * 60)

    # Get data (fetch or use cache)
    if symbol_a not in df_cache:
        df_cache[symbol_a] = fetch_data(fetcher, symbol_a, START_DATE, END_DATE)
    if symbol_b not in df_cache:
        df_cache[symbol_b] = fetch_data(fetcher, symbol_b, START_DATE, END_DATE)

    df_a = df_cache.get(symbol_a)
    df_b = df_cache.get(symbol_b)

    if df_a is None or df_b is None:
        log.error(f"Skipping {symbol_a}/{symbol_b} ? data fetch failed")
        return None

    # Cache to CSV so re-runs do not need to re-fetch
    try:
        import os
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(cache_dir, exist_ok=True)
        for sym, df in [(symbol_a, df_a), (symbol_b, df_b)]:
            fname = sym.replace("/", "_") + "_1min.csv"
            fpath = os.path.join(cache_dir, fname)
            if not os.path.exists(fpath):
                df.to_csv(fpath)
                log.info(f"Cached {len(df):,} bars -> {fpath}")
    except Exception as e:
        log.warning(f"Cache save failed (non-fatal): {e}")

    # Stationarity check on log-price spread
    log.info(f"Running stationarity tests on {symbol_a}/{symbol_b} spread...")
    try:
        common = df_a.index.intersection(df_b.index)
        lp_a   = np.log(df_a.loc[common, "close"].values)
        lp_b   = np.log(df_b.loc[common, "close"].values)
        # Simple static beta estimate for stationarity test
        from numpy.linalg import lstsq
        X     = np.column_stack([lp_b, np.ones(len(lp_b))])
        coeff, _, _, _ = lstsq(X, lp_a, rcond=None)
        spread = lp_a - coeff[0] * lp_b - coeff[1]
        stat_report = run_all_tests(spread, symbol=f"{symbol_a}/{symbol_b}")
        log.info(stat_report.summary())
        if not stat_report.edge_confirmed:
            log.warning(
                f"Stationarity tests did NOT confirm edge for "
                f"{symbol_a}/{symbol_b}. "
                "Walk-forward will still run but results may be unreliable."
            )
    except Exception as e:
        log.warning(f"Stationarity test failed: {e} ? continuing anyway")

    # Walk-forward validation
    log.info(f"Running walk-forward validation...")
    wfr = run_walk_forward(
        df_a        = df_a,
        df_b        = df_b,
        symbol_a    = symbol_a,
        symbol_b    = symbol_b,
        train_bars  = TRAIN_BARS,
        test_bars   = TEST_BARS,
        step_bars   = STEP_BARS,
        capital     = INITIAL_CAPITAL,
        risk_pct    = RISK_PCT,
    )

    print()
    print(wfr.summary())
    print()

    # Save results
    save_results(wfr, symbol_a, symbol_b, RESULTS_DIR)

    return wfr


def main():
    log.info("=" * 60)
    log.info("Quant Strategy ? Walk-Forward Backtest")
    log.info(f"Period : {START_DATE.date()} to {END_DATE.date()}")
    log.info(f"Capital: ${INITIAL_CAPITAL:,.0f} per pair")
    log.info(f"Pairs  : {len(PAIRS)}")
    log.info("=" * 60)

    # Check keys
    if not check_keys():
        sys.exit(1)

    # Import data fetcher
    from modules.data_fetcher import AlpacaDataFetcher
    fetcher  = AlpacaDataFetcher(ALPACA_KEY, ALPACA_SECRET)
    df_cache = {}   # cache fetched DataFrames to avoid re-fetching

    # Run each pair
    results = {}
    for symbol_a, symbol_b in PAIRS:
        try:
            wfr = run_pair(fetcher, symbol_a, symbol_b, df_cache)
            results[f"{symbol_a}/{symbol_b}"] = wfr
        except Exception as e:
            log.error(f"Pair {symbol_a}/{symbol_b} failed: {e}")
            results[f"{symbol_a}/{symbol_b}"] = None

    # Final summary across all pairs
    print()
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for pair_key, wfr in results.items():
        if wfr is None:
            print(f"{pair_key:20s}  FAILED")
        elif not wfr.ok:
            print(f"{pair_key:20s}  ERROR: {wfr.error}")
        else:
            deploy = "YES" if wfr.deployable else "NO"
            print(
                f"{pair_key:20s}  "
                f"verdict={wfr.verdict:15s}  "
                f"Sharpe={wfr.median_sharpe:.3f}  "
                f"profitable={wfr.pct_profitable:.0f}%  "
                f"deployable={deploy}"
            )
    print("=" * 60)
    print()

    # Recommend best pair
    deployable = [(k, v) for k, v in results.items()
                  if v and v.ok and v.deployable]
    if deployable:
        best = max(deployable, key=lambda x: x[1].median_sharpe)
        print(f"RECOMMENDATION: Deploy on {best[0]}")
        print(f"  Median Sharpe : {best[1].median_sharpe:.3f}")
        print(f"  Profitable    : {best[1].pct_profitable:.0f}% of windows")
        print(f"  Verdict       : {best[1].verdict}")
        print()
        print("Next step: paper trade this pair for minimum 4 weeks.")
        print("Only move to live capital after paper trading confirms edge.")
    else:
        print("No pair met the deployability threshold on this data period.")
        print("Options:")
        print("  1. Try a different date range (different market regime)")
        print("  2. Adjust TRAIN_BARS / TEST_BARS parameters")
        print("  3. Wait ? cointegration relationships shift over time")
    print()


if __name__ == "__main__":
    main()
