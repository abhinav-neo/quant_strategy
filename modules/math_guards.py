"""
Module 0: Mathematical Guards
==============================
Single source of truth for all numerical safety in the quant framework.
Every module imports from here. Nothing bypasses these guards.

Contract
--------
- safeN(v)       : rejects NaN, Inf, values outside [-MAX_RATIO, MAX_RATIO]
- safe_divide    : never divides by zero
- clamp          : hard bounds on any value
- safe_log       : ln(x) only for x > 0
- safe_sqrt      : sqrt(x) only for x >= 0
- clamp_pnl      : single trade PnL <= +/-50% of position
- validate_capital: guards capital after every trade
- compute_sharpe  : correct annualisation via sqrt(trades_per_year)
- compute_sortino : downside deviation only
- compute_calmar  : compound annualised return / max_drawdown
- annualise_return: compound formula, not simple multiplication
"""

import logging
import math
import numpy as np
from typing import Union

log = logging.getLogger("math_guards")

MAX_RATIO         = 9999.0
MAX_POSITION_MULT = 0.50
MIN_PRICE         = 1e-8
MIN_DENOMINATOR   = 1e-10
MAX_LOG_RETURN    = 0.20


def safeN(v, fallback=0.0, label=""):
    """Return v if finite and within [-MAX_RATIO, MAX_RATIO], else fallback."""
    try:
        fv = float(v)
    except (TypeError, ValueError):
        log.warning(f"safeN[{label}]: cannot convert {type(v).__name__} -> {fallback}")
        return fallback
    if math.isnan(fv):
        log.warning(f"safeN[{label}]: NaN -> {fallback}")
        return fallback
    if math.isinf(fv):
        log.warning(f"safeN[{label}]: Inf -> {fallback}")
        return fallback
    if abs(fv) > MAX_RATIO:
        log.warning(f"safeN[{label}]: |{fv:.4g}| > {MAX_RATIO} -> {fallback}. Check formula.")
        return fallback
    return fv


def safe_divide(numerator, denominator, fallback=0.0, label=""):
    """Divide with zero-denominator guard. Both inputs pass through safeN."""
    n = safeN(numerator,   fallback=fallback, label=f"{label}.num")
    d = safeN(denominator, fallback=0.0,      label=f"{label}.den")
    if abs(d) < MIN_DENOMINATOR:
        log.warning(f"safe_divide[{label}]: denominator ~0 -> {fallback}")
        return fallback
    return safeN(n / d, fallback=fallback, label=f"{label}.result")


def clamp(v, lo, hi):
    """Hard clamp to [lo, hi]. Non-finite v returns lo."""
    if not math.isfinite(float(v)):
        return lo
    return max(lo, min(hi, float(v)))


def safe_log(x, label=""):
    """ln(x). Returns NaN for x <= 0 so callers detect invalid inputs."""
    if not math.isfinite(x) or x <= MIN_PRICE:
        log.warning(f"safe_log[{label}]: x={x} <= 0 -> NaN")
        return float("nan")
    return math.log(x)


def safe_sqrt(x, label=""):
    """sqrt(x). Returns 0.0 for x < 0 with warning."""
    if not math.isfinite(x):
        log.warning(f"safe_sqrt[{label}]: non-finite {x} -> 0.0")
        return 0.0
    if x < 0:
        log.warning(f"safe_sqrt[{label}]: negative {x} -> 0.0")
        return 0.0
    return math.sqrt(x)


def safe_exp(x, label=""):
    """exp(x) with input clamped to [-700, 700] to prevent overflow."""
    x_c = clamp(x, -700.0, 700.0)
    if x_c != x:
        log.warning(f"safe_exp[{label}]: {x} clamped to {x_c}")
    return math.exp(x_c)


def log_return(price_t, price_t1, label=""):
    """
    Compute ln(price_t / price_t1).
    Returns NaN if either price <= 0.
    Logs a warning if |result| > MAX_LOG_RETURN (spike detection).
    """
    if price_t <= MIN_PRICE or price_t1 <= MIN_PRICE:
        log.warning(f"log_return[{label}]: non-positive price p={price_t}, p1={price_t1} -> NaN")
        return float("nan")
    r = safe_log(price_t / price_t1, label=label)
    if math.isfinite(r) and abs(r) > MAX_LOG_RETURN:
        log.warning(f"log_return[{label}]: spike |{r:.4f}| > {MAX_LOG_RETURN}. Verify data.")
    return r


def clamp_pnl(raw_pnl, position_size):
    """
    Clamp trade PnL to +/-50% of position_size.
    Prevents data errors from blowing up capital curve.
    """
    if position_size <= 0 or not math.isfinite(position_size):
        log.warning(f"clamp_pnl: invalid position_size={position_size} -> 0.0")
        return 0.0
    limit   = position_size * MAX_POSITION_MULT
    clamped = clamp(raw_pnl, -limit, limit)
    if clamped != raw_pnl:
        log.warning(f"clamp_pnl: {raw_pnl:.4f} clamped to {clamped:.4f} (limit={limit:.4f})")
    return clamped


def validate_capital(capital_new, capital_old, label=""):
    """
    Validate capital after a trade.
    Rules: finite, >= $1 floor, not > 10x old capital in one trade.
    """
    if not math.isfinite(capital_new):
        log.error(f"validate_capital[{label}]: {capital_new} non-finite -> reset to {capital_old:.2f}")
        return max(capital_old, 1.0)
    if capital_new < 1.0:
        return 1.0
    if capital_new > capital_old * 10:
        log.error(f"validate_capital[{label}]: {capital_old:.2f} -> {capital_new:.2f} (>10x) -> reset")
        return capital_old
    return capital_new


def compute_sharpe(trade_returns, trades_per_year):
    """
    Sharpe = (mean(R) / std(R)) * sqrt(trades_per_year)

    CRITICAL: annualise by sqrt(trades_per_year), NOT sqrt(bars_per_year).
    Using bars_per_year on a sparse strategy overstates Sharpe by ~30-50x.

    trade_returns   : list of per-trade net returns (net_pnl / capital_before)
    trades_per_year : n_trades * (365 / observation_days)
    """
    n = len(trade_returns)
    if n < 2:
        return 0.0
    # Fewer than 5 trades: std dev unreliable -> ratio blows up -> return 0
    if n < 5:
        return 0.0
    r   = np.array(trade_returns, dtype=np.float64)
    mu  = float(np.mean(r))
    sig = float(np.std(r, ddof=1))
    if sig < MIN_DENOMINATOR:
        return 0.0
    annF   = safe_sqrt(max(trades_per_year, 1.0), label="sharpe.annF")
    sharpe = safe_divide(mu, sig, label="sharpe") * annF
    return safeN(sharpe, fallback=0.0, label="sharpe.final")


def compute_sortino(trade_returns, trades_per_year):
    """
    Sortino = (mean(R) / sigma_down) * sqrt(trades_per_year)
    sigma_down = sqrt(mean(r^2 for r < 0))  -- downside deviation only
    """
    n = len(trade_returns)
    if n < 2:
        return 0.0
    if n < 5:
        return 0.0
    r    = np.array(trade_returns, dtype=np.float64)
    mu   = float(np.mean(r))
    down = r[r < 0]
    if len(down) < 2:
        return 0.0
    sig_down = safe_sqrt(float(np.mean(down**2)), label="sortino.sig_down")
    if sig_down < MIN_DENOMINATOR:
        return 0.0
    annF    = safe_sqrt(max(trades_per_year, 1.0), label="sortino.annF")
    sortino = safe_divide(mu, sig_down, label="sortino") * annF
    return safeN(sortino, fallback=0.0, label="sortino.final")


def compute_calmar(annualised_return, max_drawdown):
    """Calmar = annualised_return / max_drawdown. max_drawdown as fraction [0,1]."""
    if max_drawdown < MIN_DENOMINATOR:
        return 0.0
    return safeN(safe_divide(annualised_return, max_drawdown, label="calmar"),
                 fallback=0.0, label="calmar.final")


def annualise_return(total_return, observation_days):
    """
    Compound annualisation: (1 + R)^(365/days) - 1

    Uses compound formula, NOT simple R*(365/days).
    Simple formula overstates for returns > ~10%.
    """
    if observation_days < 1:
        log.warning("annualise_return: observation_days < 1 -> 0.0")
        return 0.0
    base = 1.0 + total_return
    if base <= 0:
        log.warning(f"annualise_return: 1+R={base} <= 0 (total ruin) -> -1.0")
        return -1.0
    exponent = 365.0 / observation_days
    result   = safe_exp(safe_log(base) * exponent) - 1.0
    return safeN(result, fallback=0.0, label="annualise_return")


def run_unit_tests():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    log.info("Running math_guards unit tests ...")
    passed = failed = 0

    def check(name, condition):
        nonlocal passed, failed
        if condition:
            log.info(f"  PASS  {name}")
            passed += 1
        else:
            log.error(f"  FAIL  {name}")
            failed += 1

    # safeN
    check("safeN: normal value",          safeN(1.5) == 1.5)
    check("safeN: NaN -> 0",              safeN(float("nan")) == 0.0)
    check("safeN: Inf -> 0",              safeN(float("inf")) == 0.0)
    check("safeN: -Inf -> 0",             safeN(float("-inf")) == 0.0)
    check("safeN: >MAX_RATIO -> 0",       safeN(10000.0) == 0.0)
    check("safeN: exactly MAX_RATIO ok",  safeN(9999.0) == 9999.0)
    check("safeN: negative ok",           safeN(-5.0) == -5.0)
    # safe_divide
    check("safe_divide: normal",          safe_divide(10.0, 4.0) == 2.5)
    check("safe_divide: by zero -> 0",    safe_divide(1.0, 0.0) == 0.0)
    check("safe_divide: tiny denom -> 0", safe_divide(1.0, 1e-12) == 0.0)
    check("safe_divide: NaN num -> 0",    safe_divide(float("nan"), 2.0) == 0.0)
    # clamp
    check("clamp: within range",          clamp(5.0, 0.0, 10.0) == 5.0)
    check("clamp: below lo",              clamp(-5.0, 0.0, 10.0) == 0.0)
    check("clamp: above hi",              clamp(15.0, 0.0, 10.0) == 10.0)
    check("clamp: NaN -> lo",             clamp(float("nan"), 0.0, 10.0) == 0.0)
    # safe_log
    check("safe_log: valid",              abs(safe_log(math.e) - 1.0) < 1e-10)
    check("safe_log: zero -> NaN",        math.isnan(safe_log(0.0)))
    check("safe_log: negative -> NaN",    math.isnan(safe_log(-1.0)))
    # safe_sqrt
    check("safe_sqrt: valid",             abs(safe_sqrt(4.0) - 2.0) < 1e-10)
    check("safe_sqrt: negative -> 0",     safe_sqrt(-1.0) == 0.0)
    check("safe_sqrt: zero ok",           safe_sqrt(0.0) == 0.0)
    # safe_exp
    check("safe_exp: normal",             abs(safe_exp(1.0) - math.e) < 1e-10)
    check("safe_exp: overflow clamped",   math.isfinite(safe_exp(1e6)))
    # log_return
    check("log_return: equal prices = 0", abs(log_return(100.0, 100.0)) < 1e-10)
    check("log_return: +10% correct",     abs(log_return(110.0, 100.0) - math.log(1.1)) < 1e-10)
    check("log_return: zero -> NaN",      math.isnan(log_return(0.0, 100.0)))
    # clamp_pnl
    check("clamp_pnl: within +/-50%",    clamp_pnl(40.0, 100.0) == 40.0)
    check("clamp_pnl: >50% clamped",     clamp_pnl(60.0, 100.0) == 50.0)
    check("clamp_pnl: <-50% clamped",    clamp_pnl(-80.0, 100.0) == -50.0)
    # validate_capital
    check("validate_capital: normal",    validate_capital(1100.0, 1000.0) == 1100.0)
    check("validate_capital: floor $1",  validate_capital(-50.0, 1000.0) == 1.0)
    check("validate_capital: 10x reset", validate_capital(15000.0, 1000.0) == 1000.0)
    check("validate_capital: NaN reset", validate_capital(float("nan"), 1000.0) == 1000.0)
    # compute_sharpe -- verify formula directly
    rets = [0.01, 0.02, -0.005, 0.015, 0.008, -0.003, 0.012, 0.006, -0.001, 0.009]
    mu   = np.mean(rets)
    sig  = np.std(rets, ddof=1)
    annF = math.sqrt(252)
    expected_sh = (mu / sig) * annF
    got_sh = compute_sharpe(rets, trades_per_year=252.0)
    check("compute_sharpe: known value",  abs(got_sh - expected_sh) < 1e-6)
    check("compute_sharpe: empty -> 0",   compute_sharpe([], 252) == 0.0)
    check("compute_sharpe: 1 trade -> 0", compute_sharpe([0.01], 252) == 0.0)
    # compute_sortino
    check("compute_sortino: > 0",         compute_sortino(rets, 252.0) > 0)
    check("compute_sortino: empty -> 0",  compute_sortino([], 252) == 0.0)
    # annualise_return
    check("annualise_return: 1yr = total",  abs(annualise_return(0.40, 365) - 0.40) < 1e-6)
    check("annualise_return: 90d > 1yr",    annualise_return(0.40, 90) > 0.40)
    check("annualise_return: bad days->0",  annualise_return(0.10, 0) == 0.0)
    # compute_calmar
    check("compute_calmar: 60%/15%=4.0",   abs(compute_calmar(0.60, 0.15) - 4.0) < 1e-6)
    check("compute_calmar: zero DD -> 0",   compute_calmar(0.60, 0.0) == 0.0)

    log.info(f"\nmath_guards: {passed}/{passed+failed} tests passed.")
    return failed == 0


if __name__ == "__main__":
    ok = run_unit_tests()
    if not ok:
        raise SystemExit("math_guards unit tests FAILED.")
    print("\nAll math_guards tests passed.")
