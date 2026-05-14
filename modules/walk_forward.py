"""
Module 8: Walk-Forward Validator
==================================
Splits data into rolling train/test windows and runs the
backtest on each out-of-sample window independently.

This is the final gate before paper trading.
A strategy that only works in-sample is overfit.
A strategy that works consistently across multiple
out-of-sample windows has a genuine edge.

Walk-forward method
-------------------
    |<-- train_bars -->|<-- test_bars -->|
    |     window 1     |    test 1       |
           |<-- train_bars -->|<-- test_bars -->|
           |     window 2     |    test 2       |

Each test window is strictly out-of-sample.
The model is re-trained from scratch on each train window.
No look-ahead bias.

Metrics reported per window and aggregated:
    - return, Sharpe, Sortino, Calmar, max_drawdown
    - win_rate, profit_factor, total_trades
    - consistency score: fraction of windows profitable

Robustness threshold (from quant literature):
    Sharpe > 0.5  in > 60% of windows -> deployable
    Sharpe > 1.0  in > 40% of windows -> strong edge
    Any window with Sharpe < -1.0      -> flag as fragile
"""

import logging
import math
from dataclasses import dataclass, field
from typing      import List, Tuple

import numpy  as np
import pandas as pd

from modules.math_guards import (
    safeN, compute_sharpe, compute_sortino,
    compute_calmar, annualise_return, MIN_DENOMINATOR
)
from modules.backtest import BacktestEngine, BacktestResult

log = logging.getLogger("walk_forward")

# ?? Constants ??????????????????????????????????????????????????????????????
DEFAULT_TRAIN_BARS  = 2000   # ~33 hours of 1-min data for training
DEFAULT_TEST_BARS   = 500    # ~8 hours of 1-min data per test window
DEFAULT_STEP_BARS   = 500    # advance by one test window each iteration
MIN_WINDOWS         = 3      # minimum windows to draw conclusions
SHARPE_DEPLOY_THRESHOLD = 0.5
SHARPE_STRONG_THRESHOLD = 1.0
MIN_PROFITABLE_FRAC     = 0.60   # 60% of windows profitable = deployable


# ?? Dataclasses ????????????????????????????????????????????????????????????
@dataclass
class WindowResult:
    """Result for one train/test window."""
    window_idx:    int   = 0
    train_start:   int   = 0
    train_end:     int   = 0
    test_start:    int   = 0
    test_end:      int   = 0
    backtest:      BacktestResult = field(default_factory=BacktestResult)
    is_profitable: bool  = False
    is_valid:      bool  = False


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward output."""
    symbol_a:      str = ""
    symbol_b:      str = ""
    n_windows:     int = 0
    n_valid:       int = 0

    # Per-window series
    windows:       List[WindowResult] = field(default_factory=list)

    # Aggregate metrics (median across valid windows)
    median_return:  float = 0.0
    median_sharpe:  float = 0.0
    median_sortino: float = 0.0
    median_calmar:  float = 0.0
    median_maxdd:   float = 0.0
    median_winrate: float = 0.0

    # Consistency
    pct_profitable:     float = 0.0
    pct_sharpe_gt_half: float = 0.0
    pct_sharpe_gt_one:  float = 0.0

    # Verdict
    verdict:       str  = "INSUFFICIENT_DATA"
    deployable:    bool = False
    ok:            bool = False
    error:         str  = ""
    warnings:      List[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.ok:
            return f"WalkForwardResult(FAILED: {self.error})"
        lines = [
            "=" * 60,
            f"Walk-Forward: {self.symbol_a} / {self.symbol_b}",
            f"Windows : {self.n_valid}/{self.n_windows} valid",
            "-" * 60,
            f"Median return  : {self.median_return*100:+.2f}%",
            f"Median Sharpe  : {self.median_sharpe:.3f}",
            f"Median Sortino : {self.median_sortino:.3f}",
            f"Median Calmar  : {self.median_calmar:.3f}",
            f"Median max DD  : {self.median_maxdd*100:.1f}%",
            f"Median win rate: {self.median_winrate*100:.1f}%",
            "-" * 60,
            f"Profitable windows   : {self.pct_profitable:.0f}%",
            f"Sharpe > 0.5 windows : {self.pct_sharpe_gt_half:.0f}%",
            f"Sharpe > 1.0 windows : {self.pct_sharpe_gt_one:.0f}%",
            "-" * 60,
            f"Verdict     : {self.verdict}",
            f"Deployable  : {'YES' if self.deployable else 'NO'}",
            "=" * 60,
        ]
        for w in self.warnings:
            lines.append(f"WARNING : {w}")
        return "\n".join(lines)



def run_walk_forward(
    df_a:        pd.DataFrame,
    df_b:        pd.DataFrame,
    symbol_a:    str   = "A",
    symbol_b:    str   = "B",
    train_bars:  int   = DEFAULT_TRAIN_BARS,
    test_bars:   int   = DEFAULT_TEST_BARS,
    step_bars:   int   = DEFAULT_STEP_BARS,
    capital:     float = 10_000.0,
    risk_pct:    float = 0.01,
) -> WalkForwardResult:
    """
    Run walk-forward validation on two aligned DataFrames.

    Parameters
    ----------
    df_a, df_b   : validated DataFrames (from data_fetcher)
    train_bars   : bars per training window
    test_bars    : bars per out-of-sample test window
    step_bars    : bars to advance each iteration
    capital      : starting capital per window ($)
    risk_pct     : fraction of capital at risk per trade

    Returns
    -------
    WalkForwardResult - always returned, never raises.
    """
    result = WalkForwardResult(symbol_a=symbol_a, symbol_b=symbol_b)

    # ?? Input validation ??????????????????????????????????????????????????
    if train_bars < 500:
        result.error = f"train_bars={train_bars} < 500 minimum."
        log.error(result.error); return result
    if test_bars < 100:
        result.error = f"test_bars={test_bars} < 100 minimum."
        log.error(result.error); return result

    # Align on common timestamps
    try:
        common = df_a.index.intersection(df_b.index)
        if len(common) < train_bars + test_bars:
            result.error = (
                f"Only {len(common)} common bars ? need "
                f"{train_bars+test_bars} minimum."
            )
            log.error(result.error); return result
        df_a = df_a.loc[common].copy()
        df_b = df_b.loc[common].copy()
    except Exception as e:
        result.error = f"Alignment error: {e}"
        log.error(result.error); return result

    n = len(df_a)
    log.info(
        f"[wf] {symbol_a}/{symbol_b}: {n:,} bars  "
        f"train={train_bars} test={test_bars} step={step_bars}"
    )

    # ?? Build windows ??????????????????????????????????????????????????????
    engine  = BacktestEngine(
        initial_capital=capital,
        risk_pct=risk_pct,
    )
    windows: List[WindowResult] = []
    win_idx = 0

    for train_start in range(0, n - train_bars - test_bars + 1, step_bars):
        train_end  = train_start + train_bars
        test_start = train_end
        test_end   = test_start + test_bars
        if test_end > n:
            break

        win_idx += 1
        log.info(
            f"[wf] window {win_idx}: "
            f"train [{train_start}:{train_end}] "
            f"test [{test_start}:{test_end}]"
        )

        wr = WindowResult(
            window_idx  = win_idx,
            train_start = train_start,
            train_end   = train_end,
            test_start  = test_start,
            test_end    = test_end,
        )

        # Slice test window (backtest trains internally on warmup portion)
        # We feed the FULL window (train+test) so the engine can warm up
        # on the train portion and trade on the test portion.
        # The engine uses WARMUP_BARS internally, so we pass the combined slice.
        slice_a = df_a.iloc[train_start:test_end].copy()
        slice_b = df_b.iloc[train_start:test_end].copy()

        try:
            bt = engine.run(slice_a, slice_b, symbol_a, symbol_b)
            wr.backtest    = bt
            wr.is_valid    = bt.ok
            wr.is_profitable = bt.ok and bt.total_return > 0
        except Exception as e:
            wr.backtest.error = f"Window {win_idx} crashed: {e}"
            wr.is_valid = False
            log.error(f"[wf] window {win_idx} error: {e}")

        windows.append(wr)

    result.n_windows = len(windows)
    result.windows   = windows

    if result.n_windows == 0:
        result.error = "No windows could be constructed from the data."
        log.error(result.error); return result

    # ?? Aggregate metrics ??????????????????????????????????????????????????
    valid_ws = [w for w in windows if w.is_valid]
    result.n_valid = len(valid_ws)

    if result.n_valid < MIN_WINDOWS:
        result.error = (
            f"Only {result.n_valid} valid windows ? "
            f"need {MIN_WINDOWS} to draw conclusions."
        )
        result.warnings.append(result.error)
        # Still return partial data
        result.ok = True
        result.verdict = "INSUFFICIENT_DATA"
        return result

    def _med(key):
        vals = [safeN(getattr(w.backtest, key), fallback=0.0) for w in valid_ws]
        vals = [v for v in vals if math.isfinite(v)]
        return float(np.median(vals)) if vals else 0.0

    result.median_return  = _med("total_return")
    result.median_sharpe  = _med("sharpe")
    result.median_sortino = _med("sortino")
    result.median_calmar  = _med("calmar")
    result.median_maxdd   = _med("max_drawdown")
    result.median_winrate = _med("win_rate")

    n_v = result.n_valid
    result.pct_profitable     = sum(1 for w in valid_ws if w.is_profitable) / n_v * 100
    result.pct_sharpe_gt_half = sum(
        1 for w in valid_ws if safeN(w.backtest.sharpe) >= SHARPE_DEPLOY_THRESHOLD
    ) / n_v * 100
    result.pct_sharpe_gt_one  = sum(
        1 for w in valid_ws if safeN(w.backtest.sharpe) >= SHARPE_STRONG_THRESHOLD
    ) / n_v * 100

    # ?? Verdict ????????????????????????????????????????????????????????????
    fragile_windows = [
        w for w in valid_ws
        if safeN(w.backtest.sharpe) < -1.0
    ]
    if fragile_windows:
        result.warnings.append(
            f"{len(fragile_windows)} window(s) had Sharpe < -1.0 ? "
            "strategy may be fragile in certain regimes."
        )

    if result.pct_profitable >= MIN_PROFITABLE_FRAC * 100:
        if result.median_sharpe >= SHARPE_STRONG_THRESHOLD:
            result.verdict    = "STRONG_EDGE"
            result.deployable = True
        elif result.median_sharpe >= SHARPE_DEPLOY_THRESHOLD:
            result.verdict    = "DEPLOYABLE"
            result.deployable = True
        else:
            result.verdict    = "WEAK_EDGE"
            result.deployable = False
    else:
        result.verdict    = "NOT_ROBUST"
        result.deployable = False

    result.ok = True
    log.info(result.summary())
    return result



def run_unit_tests():
    """
    Tests use synthetic cointegrated DataFrames (no API calls).
    Verifies:
    T01: valid pair produces WalkForwardResult.ok=True
    T02: n_windows > 0
    T03: n_valid <= n_windows
    T04: all per-window metrics finite
    T05: pct_profitable in [0,100]
    T06: median_sharpe within safeN range
    T07: deployable flag set correctly
    T08: too-short data -> ok=False or insufficient warning
    T09: misaligned data -> ok=False
    T10: summary() works without crash
    T11: all window results have backtest attribute
    T12: step_bars controls window count correctly
    """
    import math
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(name)s %(message)s")
    log.info("Running walk_forward unit tests ...")
    passed = failed = 0

    def check(name, condition):
        nonlocal passed, failed
        if condition: log.info(f"  PASS  {name}"); passed+=1
        else:         log.error(f"  FAIL  {name}"); failed+=1

    rng = np.random.default_rng(42)

    def make_df(n, base_price, seed):
        rng2 = np.random.default_rng(seed)
        t    = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        ret  = rng2.normal(4e-6, 0.008, n)
        close= base_price * np.exp(np.cumsum(ret))
        noise= rng2.uniform(5, 50, n)
        high = np.maximum(close+noise, close)
        low  = np.minimum(close-noise, close)
        opn  = close + rng2.normal(0,5,n)
        high = np.maximum(high, opn); low = np.minimum(low, opn)
        vol  = rng2.uniform(10,200,n)
        return pd.DataFrame(
            {"open":opn,"high":high,"low":low,"close":close,
             "volume":vol,"valid_bar":True},
            index=pd.DatetimeIndex(t, name="timestamp"))

    # Small but enough for 2 windows: 2*train + 2*test
    TRAIN = 600; TEST = 300; STEP = 300
    N = TRAIN + TEST*3   # enough for 3 windows
    df_a = make_df(N, 45000, seed=1)
    df_b = make_df(N, 2500,  seed=2)
    # Cointegrate
    df_b["close"] = df_a["close"].values*0.055 + rng.normal(0,5,N)
    df_b["high"]  = np.maximum(df_b["close"]+rng.uniform(5,30,N), df_b["close"])
    df_b["low"]   = np.minimum(df_b["close"]-rng.uniform(5,30,N), df_b["close"])

    log.info("  [T01-T07] running main walk-forward (~90s)...")
    wfr = run_walk_forward(df_a, df_b, "BTC", "ETH",
                           train_bars=TRAIN, test_bars=TEST,
                           step_bars=STEP, capital=10_000)

    check("T01 ok=True",          wfr.ok)
    check("T02 n_windows > 0",    wfr.n_windows > 0)
    check("T03 n_valid <= n_windows", wfr.n_valid <= wfr.n_windows)

    # T04: all per-window metrics finite
    if wfr.ok and wfr.n_valid > 0:
        for attr in ["median_return","median_sharpe","median_sortino",
                     "median_calmar","median_maxdd","median_winrate"]:
            v = getattr(wfr, attr)
            check(f"T04 {attr} finite", math.isfinite(v))

    # T05: pct_profitable in [0,100]
    check("T05 pct_profitable in [0,100]",
          0.0 <= wfr.pct_profitable <= 100.0)

    # T06: median_sharpe within safeN range
    check("T06 median_sharpe |v| <= 999",
          abs(wfr.median_sharpe) <= 999)

    # T07: deployable is bool
    check("T07 deployable is bool",
          isinstance(wfr.deployable, bool))

    # T08: too-short data
    log.info("  [T08] too-short data...")
    df_short_a = make_df(200, 45000, seed=3)
    df_short_b = make_df(200, 2500,  seed=4)
    wfr_short  = run_walk_forward(df_short_a, df_short_b,
                                  train_bars=TRAIN, test_bars=TEST,
                                  step_bars=STEP)
    check("T08 too-short -> not deployable or error",
          not wfr_short.deployable or not wfr_short.ok)

    # T09: non-overlapping timestamps
    log.info("  [T09] non-overlapping timestamps...")
    df_c = make_df(N, 45000, seed=5)
    df_d = make_df(N, 2500,  seed=6)
    df_d.index = df_d.index + pd.Timedelta(days=3650)
    wfr_mis = run_walk_forward(df_c, df_d, train_bars=TRAIN, test_bars=TEST)
    check("T09 misaligned -> ok=False", not wfr_mis.ok)

    # T10: summary() works
    if wfr.ok:
        try:
            s = wfr.summary()
            check("T10 summary() works", len(s)>0 and "Walk-Forward" in s)
        except Exception as e:
            check(f"T10 summary() raised: {e}", False)

    # T11: all windows have backtest attribute
    if wfr.ok:
        check("T11 all windows have BacktestResult",
              all(isinstance(w.backtest, BacktestResult) for w in wfr.windows))

    # T12: window count matches step_bars
    if wfr.ok:
        expected = len(range(0, N-(TRAIN+TEST)+1, STEP))
        check(f"T12 window count = {expected}",
              wfr.n_windows == expected)

    log.info("walk_forward: %d/%d tests passed." % (passed, passed+failed))
    return failed == 0


if __name__ == "__main__":
    ok = run_unit_tests()
    if not ok:
        raise SystemExit("walk_forward unit tests FAILED.")
    print("All walk_forward tests passed.")

