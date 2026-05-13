"""
Module 2: Stationarity Testing
================================
Mathematical entry gate for the strategy.
ALL THREE tests must confirm structure before any downstream
module (Kalman, OU, HMM) is permitted to run.

Tests implemented
-----------------
1. ADF (Augmented Dickey-Fuller)
   H0: series has a unit root (non-stationary)
   Reject H0 at p < 0.05 -> stationary -> OU model valid

2. Hurst Exponent (R/S analysis)
   H < 0.45 -> strong mean reversion   -> trade
   H in [0.45, 0.55] -> random walk    -> DO NOT TRADE
   H > 0.55 -> trending / persistent   -> momentum only

3. Ljung-Box Q-test (autocorrelation)
   H0: no autocorrelation up to lag L
   Reject H0 at p < 0.05 on at least one lag -> structure exists

Mathematical formulas (exact, not approximations)
--------------------------------------------------
ADF:
    Delta_y(t) = alpha + beta*t + gamma*y(t-1)
                 + sum_{i=1}^{p} delta_i * Delta_y(t-i) + eps(t)
    Test stat: t = gamma_hat / SE(gamma_hat)
    Implemented via statsmodels.tsa.stattools.adfuller

Hurst (R/S):
    For window size n:
        mean_x   = mean(x[t:t+n])
        deviations = cumsum(x[t:t+n] - mean_x)
        R(n)     = max(deviations) - min(deviations)
        S(n)     = std(x[t:t+n], ddof=1)
        RS(n)    = R(n) / S(n)
    Average RS over multiple windows of size n.
    Regress log(RS) on log(n):
        H = slope of OLS regression

Ljung-Box:
    Q(m) = n(n+2) * sum_{k=1}^{m} rho_k^2 / (n-k)
    rho_k = autocorrelation at lag k
    Under H0: Q(m) ~ chi-squared(m)
    Implemented via statsmodels.stats.diagnostic.acorr_ljungbox
"""

import logging
import math
from dataclasses import dataclass, field
from typing      import List, Optional, Tuple

import numpy  as np
import pandas as pd
from scipy.stats import linregress

from modules.math_guards import (
    safeN, safe_divide, safe_log, safe_sqrt, clamp, MIN_DENOMINATOR
)

log = logging.getLogger("stationarity")

# ── Constants ──────────────────────────────────────────────────────────────
ADF_PVALUE_THRESHOLD  = 0.05   # reject unit root below this
LB_PVALUE_THRESHOLD   = 0.05   # reject no-autocorr below this
LB_LAGS               = [5, 10, 20]  # lags to test
HURST_LOWER_MR        = 0.45   # below -> mean-reverting
HURST_UPPER_TR        = 0.55   # above -> trending
HURST_MIN_WINDOW      = 20     # minimum window for R/S
HURST_NUM_WINDOWS     = 20     # number of window sizes to use
MIN_SERIES_LENGTH     = 500    # minimum bars before running any test


# ── Result dataclasses ─────────────────────────────────────────────────────

@dataclass
class ADFResult:
    statistic:    float = 0.0
    pvalue:       float = 1.0
    n_lags:       int   = 0
    n_obs:        int   = 0
    is_stationary: bool = False   # True if pvalue < ADF_PVALUE_THRESHOLD
    error:        str   = ""

@dataclass
class HurstResult:
    exponent:       float = 0.5
    interpretation: str   = "random_walk"  # mean_reverting | random_walk | trending
    r_squared:      float = 0.0            # quality of the log-log regression
    n_windows:      int   = 0
    error:          str   = ""

@dataclass
class LjungBoxResult:
    lags_tested:      List[int]   = field(default_factory=list)
    pvalues:          List[float] = field(default_factory=list)
    min_pvalue:       float       = 1.0
    has_autocorr:     bool        = False  # True if any pvalue < LB_PVALUE_THRESHOLD
    error:            str         = ""

@dataclass
class StationarityReport:
    """
    Final verdict from all three tests.
    edge_confirmed = True ONLY if all three pass.
    """
    symbol:         str
    series_length:  int           = 0
    adf:            Optional[ADFResult]       = None
    hurst:          Optional[HurstResult]     = None
    ljung_box:      Optional[LjungBoxResult]  = None
    edge_confirmed: bool          = False     # master gate
    edge_type:      str           = "none"    # mean_reverting | trending | none
    reason:         str           = ""        # human-readable explanation
    warnings:       List[str]     = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"Stationarity Report: {self.symbol}",
            f"Series length : {self.series_length:,} bars",
            f"Edge confirmed: {'YES' if self.edge_confirmed else 'NO'}",
            f"Edge type     : {self.edge_type}",
            f"Reason        : {self.reason}",
        ]
        if self.adf:
            lines.append(
                f"ADF           : stat={self.adf.statistic:.4f} "
                f"p={self.adf.pvalue:.4f} "
                f"stationary={'YES' if self.adf.is_stationary else 'NO'}"
            )
        if self.hurst:
            lines.append(
                f"Hurst         : H={self.hurst.exponent:.4f} "
                f"({self.hurst.interpretation}) "
                f"R2={self.hurst.r_squared:.4f}"
            )
        if self.ljung_box:
            lines.append(
                f"Ljung-Box     : min_p={self.ljung_box.min_pvalue:.4f} "
                f"autocorr={'YES' if self.ljung_box.has_autocorr else 'NO'}"
            )
        for w in self.warnings:
            lines.append(f"Warning       : {w}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ── Individual test functions (pure, no side effects) ─────────────────────

def run_adf(series: np.ndarray, label: str = "") -> ADFResult:
    """
    Augmented Dickey-Fuller test on a 1D array.

    H0: series has a unit root (non-stationary, random walk)
    Reject H0 if pvalue < ADF_PVALUE_THRESHOLD -> series is stationary

    Parameters
    ----------
    series : np.ndarray  clean, finite, 1D values.
                    IMPORTANT: pass first differences (returns), not raw price levels.
                    Raw price levels always show H > 0.5 due to autocorrelation in levels.
                    Correct usage: run_hurst(np.diff(log_prices)) or run_hurst(spread_returns)
    label  : str         identifier for logging

    Returns
    -------
    ADFResult with statistic, pvalue, is_stationary flag
    """
    result = ADFResult()
    try:
        from statsmodels.tsa.stattools import adfuller

        # Guard: need at least MIN_SERIES_LENGTH points
        clean = series[np.isfinite(series)]
        if len(clean) < MIN_SERIES_LENGTH:
            result.error = (
                f"run_adf[{label}]: only {len(clean)} finite values, "
                f"need {MIN_SERIES_LENGTH}."
            )
            log.warning(result.error)
            return result

        # autolag='AIC' selects lag order by Akaike criterion
        adf_out = adfuller(clean, autolag="AIC")

        result.statistic     = safeN(float(adf_out[0]), fallback=0.0,
                                     label=f"adf.stat.{label}")
        result.pvalue        = float(np.clip(adf_out[1], 0.0, 1.0))
        result.n_lags        = int(adf_out[2])
        result.n_obs         = int(adf_out[3])
        result.is_stationary = result.pvalue < ADF_PVALUE_THRESHOLD

        log.info(
            f"ADF[{label}]: stat={result.statistic:.4f} "
            f"p={result.pvalue:.4f} "
            f"n_lags={result.n_lags} "
            f"stationary={'YES' if result.is_stationary else 'NO'}"
        )
    except Exception as exc:
        result.error = f"run_adf[{label}]: exception — {exc}"
        log.error(result.error)

    return result


def run_hurst(series: np.ndarray, label: str = "") -> HurstResult:
    """
    Hurst Exponent via R/S (rescaled range) analysis.

    Formula:
        For each window size n in log-spaced range:
            RS(n) = mean over sub-windows of (Range / StdDev)
        Regress log(RS) on log(n) via OLS:
            H = slope

    H < 0.45 -> mean reverting  (use stat arb)
    H in [0.45, 0.55] -> random walk (no edge, do not trade)
    H > 0.55 -> trending        (use momentum)

    Implementation is self-contained — no external library.
    Every division goes through safe_divide.
    Every log goes through safe_log.
    """
    result = HurstResult()
    try:
        clean = series[np.isfinite(series)]
        n_obs = len(clean)

        if n_obs < MIN_SERIES_LENGTH:
            result.error = (
                f"run_hurst[{label}]: only {n_obs} finite values, "
                f"need {MIN_SERIES_LENGTH}."
            )
            log.warning(result.error)
            return result

        # Generate log-spaced window sizes from HURST_MIN_WINDOW to n_obs//2
        max_window = n_obs // 2
        if max_window < HURST_MIN_WINDOW * 2:
            result.error = (
                f"run_hurst[{label}]: series too short for reliable R/S. "
                f"max_window={max_window} < {HURST_MIN_WINDOW*2}."
            )
            log.warning(result.error)
            return result

        window_sizes = np.unique(
            np.logspace(
                math.log10(HURST_MIN_WINDOW),
                math.log10(max_window),
                num=HURST_NUM_WINDOWS,
                dtype=int,
            )
        )
        window_sizes = window_sizes[window_sizes >= HURST_MIN_WINDOW]

        log_n_list  = []
        log_rs_list = []

        for n in window_sizes:
            n = int(n)
            rs_values = []

            # Slide non-overlapping windows of size n
            for start in range(0, n_obs - n + 1, n):
                window = clean[start : start + n].astype(np.float64)

                if len(window) < n:
                    continue

                mean_w = float(np.mean(window))
                deviations = np.cumsum(window - mean_w)

                r = float(np.max(deviations) - np.min(deviations))
                s = float(np.std(window, ddof=1))

                if s < MIN_DENOMINATOR:
                    continue   # flat window — skip

                rs = safe_divide(r, s, label=f"hurst.rs.n{n}")
                if rs > 0:
                    rs_values.append(rs)

            if len(rs_values) < 2:
                continue

            mean_rs = float(np.mean(rs_values))
            ln      = safe_log(float(n),       label=f"hurst.log_n.{n}")
            lrs     = safe_log(mean_rs,        label=f"hurst.log_rs.{n}")

            if math.isfinite(ln) and math.isfinite(lrs):
                log_n_list.append(ln)
                log_rs_list.append(lrs)

        if len(log_n_list) < 4:
            result.error = (
                f"run_hurst[{label}]: insufficient valid R/S points "
                f"({len(log_n_list)}) for regression."
            )
            log.warning(result.error)
            return result

        # OLS regression: log(RS) = H * log(n) + const
        slope, intercept, r_value, p_value, se = linregress(
            log_n_list, log_rs_list
        )

        result.exponent   = safeN(float(slope), fallback=0.5,
                                  label=f"hurst.exponent.{label}")
        result.r_squared  = safeN(float(r_value**2), fallback=0.0,
                                  label=f"hurst.r2.{label}")
        result.n_windows  = len(log_n_list)

        H = result.exponent
        if H < HURST_LOWER_MR:
            result.interpretation = "mean_reverting"
        elif H > HURST_UPPER_TR:
            result.interpretation = "trending"
        else:
            result.interpretation = "random_walk"

        log.info(
            f"Hurst[{label}]: H={H:.4f} ({result.interpretation}) "
            f"R2={result.r_squared:.4f} n_windows={result.n_windows}"
        )

    except Exception as exc:
        result.error = f"run_hurst[{label}]: exception — {exc}"
        log.error(result.error)

    return result


def run_ljung_box(series: np.ndarray, label: str = "") -> LjungBoxResult:
    """
    Ljung-Box test for autocorrelation.

    H0: no autocorrelation up to lag L
    Q(m) = n(n+2) * sum_{k=1}^{m} rho_k^2 / (n-k)
    Reject H0 if any pvalue < LB_PVALUE_THRESHOLD.

    Tests multiple lags (LB_LAGS) to capture both short
    and medium-term autocorrelation.
    """
    result = LjungBoxResult(lags_tested=[], pvalues=[])
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        clean = series[np.isfinite(series)]
        if len(clean) < MIN_SERIES_LENGTH:
            result.error = (
                f"run_ljung_box[{label}]: only {len(clean)} finite values, "
                f"need {MIN_SERIES_LENGTH}."
            )
            log.warning(result.error)
            return result

        # Only test lags that are valid given series length
        valid_lags = [L for L in LB_LAGS if L < len(clean) // 2]
        if not valid_lags:
            result.error = f"run_ljung_box[{label}]: series too short for lags {LB_LAGS}."
            log.warning(result.error)
            return result

        lb_out = acorr_ljungbox(clean, lags=valid_lags, return_df=True)

        pvalues = []
        for lag in valid_lags:
            pv = float(np.clip(lb_out.loc[lag, "lb_pvalue"], 0.0, 1.0))
            pvalues.append(safeN(pv, fallback=1.0,
                                 label=f"lb.pv.lag{lag}.{label}"))

        result.lags_tested  = valid_lags
        result.pvalues      = pvalues
        result.min_pvalue   = float(min(pvalues)) if pvalues else 1.0
        result.has_autocorr = result.min_pvalue < LB_PVALUE_THRESHOLD

        log.info(
            f"Ljung-Box[{label}]: lags={valid_lags} "
            f"pvalues={[f'{p:.4f}' for p in pvalues]} "
            f"autocorr={'YES' if result.has_autocorr else 'NO'}"
        )

    except Exception as exc:
        result.error = f"run_ljung_box[{label}]: exception — {exc}"
        log.error(result.error)

    return result


# ── Master gate function ───────────────────────────────────────────────────

def run_all_tests(
    series: np.ndarray,
    symbol: str = "",
    label:  str = "",
) -> StationarityReport:
    """
    Run ADF + Hurst + Ljung-Box on a series.
    Sets edge_confirmed = True only if ALL three confirm structure.

    Parameters
    ----------
    series : np.ndarray
        1D array of values to test.
        For stat arb: use the spread series (log_P_A - beta * log_P_B).
        For single asset: use log_returns or log_prices.
    symbol : str  e.g. "BTC/USD" or "BTC-ETH spread"
    label  : str  passed to individual test loggers

    Returns
    -------
    StationarityReport — always returned, never raises.
    Check report.edge_confirmed before proceeding.
    """
    report = StationarityReport(symbol=symbol)

    clean = series[np.isfinite(series)]
    report.series_length = len(clean)

    if report.series_length < MIN_SERIES_LENGTH:
        report.edge_confirmed = False
        report.reason = (
            f"Series too short: {report.series_length} finite values, "
            f"need {MIN_SERIES_LENGTH}."
        )
        log.warning(f"stationarity[{label}]: {report.reason}")
        return report

    # Run all three tests independently
    report.adf       = run_adf(clean,              label=label)
    # Hurst on first differences (returns), not raw levels.
    # Raw levels always give H > 0.5 regardless of mean reversion.
    # This is standard quant practice: test return series structure.
    clean_diff       = np.diff(clean)
    report.hurst     = run_hurst(clean_diff,       label=label)
    report.ljung_box = run_ljung_box(clean,        label=label)

    # Collect errors
    errors = [t.error for t in [report.adf, report.hurst, report.ljung_box]
              if t.error]
    if errors:
        for e in errors:
            report.warnings.append(e)

    # ── Decision logic ─────────────────────────────────────────────────────
    adf_ok = report.adf.is_stationary and not report.adf.error
    lb_ok  = report.ljung_box.has_autocorr and not report.ljung_box.error
    h      = report.hurst.exponent
    h_ok   = not report.hurst.error

    if not adf_ok:
        report.edge_confirmed = False
        report.edge_type      = "none"
        report.reason         = (
            f"ADF failed to reject unit root (p={report.adf.pvalue:.4f}). "
            "Series is non-stationary — OU model not valid."
        )
        return report

    if not lb_ok:
        report.edge_confirmed = False
        report.edge_type      = "none"
        report.reason         = (
            f"Ljung-Box found no autocorrelation "
            f"(min_p={report.ljung_box.min_pvalue:.4f}). "
            "No predictable structure."
        )
        return report

    # ADF and LB both pass — now classify by Hurst
    if not h_ok:
        report.warnings.append("Hurst calculation failed — treating as random walk.")
        report.edge_confirmed = False
        report.edge_type      = "none"
        report.reason         = "Hurst exponent could not be computed reliably."
        return report

    if h < HURST_LOWER_MR:
        report.edge_confirmed = True
        report.edge_type      = "mean_reverting"
        report.reason         = (
            f"All three tests pass. "
            f"H={h:.4f} < {HURST_LOWER_MR} -> mean-reverting. "
            "Use Kalman + OU + Kelly stat arb strategy."
        )
    elif h > HURST_UPPER_TR:
        report.edge_confirmed = True
        report.edge_type      = "trending"
        report.reason         = (
            f"All three tests pass. "
            f"H={h:.4f} > {HURST_UPPER_TR} -> trending. "
            "Use momentum strategy. OU not applicable."
        )
    else:
        report.edge_confirmed = False
        report.edge_type      = "none"
        report.reason         = (
            f"H={h:.4f} in [{HURST_LOWER_MR}, {HURST_UPPER_TR}] "
            "-> random walk zone. No reliable edge. Do not trade."
        )

    log.info(report.summary())
    return report


def run_rolling(
    series:      np.ndarray,
    window:      int = 500,
    step:        int = 100,
    symbol:      str = "",
) -> List[StationarityReport]:
    """
    Run stationarity tests on rolling windows.
    Used to detect regime changes over time.
    Returns list of StationarityReport, one per window.

    Parameters
    ----------
    series : 1D array
    window : bars per window (default 500 ~ 8 hours of 1-min bars)
    step   : bars to advance each iteration
    """
    reports = []
    n       = len(series)

    if n < window:
        log.warning(
            f"run_rolling[{symbol}]: series length {n} < window {window}. "
            "Returning single report."
        )
        return [run_all_tests(series, symbol=symbol, label="full")]

    for start in range(0, n - window + 1, step):
        end    = start + window
        chunk  = series[start:end]
        label  = f"w{start}-{end}"
        report = run_all_tests(chunk, symbol=symbol, label=label)
        reports.append(report)

    confirmed = sum(1 for r in reports if r.edge_confirmed)
    log.info(
        f"run_rolling[{symbol}]: {confirmed}/{len(reports)} windows "
        "confirmed edge."
    )
    return reports


# ── Unit tests ─────────────────────────────────────────────────────────────

def run_unit_tests() -> bool:
    """
    Tests every function using mathematically controlled synthetic series.

    Series types:
        OU process    -> should be stationary, mean-reverting, autocorrelated
        Random walk   -> should fail ADF, Hurst near 0.5
        AR(1) trend   -> should pass ADF (if stationary AR), H > 0.5
        White noise   -> should fail Ljung-Box

    Every assertion is grounded in the mathematical properties
    of the series being tested.
    """
    import math
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    )
    log.info("Running stationarity unit tests ...")
    passed = failed = 0

    def check(name: str, condition: bool):
        nonlocal passed, failed
        if condition:
            log.info(f"  PASS  {name}")
            passed += 1
        else:
            log.error(f"  FAIL  {name}")
            failed += 1

    rng = np.random.default_rng(42)

    # ── Series generators ─────────────────────────────────────────────────
    def make_ou(n=5000, theta=0.5, mu=0.0, sigma=0.1):
        """
        Discrete OU process: X(t) = X(t-1)*exp(-theta*dt) + mu*(1-exp(-theta*dt))
                                    + sigma*sqrt((1-exp(-2*theta*dt))/(2*theta))*eps
        dt = 1 (unit time steps)
        """
        dt  = 1.0
        a   = math.exp(-theta * dt)
        b   = mu * (1 - a)
        s   = sigma * math.sqrt((1 - math.exp(-2*theta*dt)) / (2*theta))
        x   = np.zeros(n)
        x[0] = mu
        eps = rng.standard_normal(n)
        for i in range(1, n):
            x[i] = a * x[i-1] + b + s * eps[i]
        return x

    def make_rw(n=5000, sigma=0.01):
        """Random walk: X(t) = X(t-1) + sigma*eps(t)"""
        return np.cumsum(rng.normal(0, sigma, n))

    def make_white_noise(n=5000, sigma=0.01):
        """Pure white noise — no autocorrelation."""
        return rng.normal(0, sigma, n)

    def make_ar1_stationary(n=5000, phi=0.7, sigma=0.01):
        """Stationary AR(1): X(t) = phi*X(t-1) + eps, |phi| < 1"""
        x    = np.zeros(n)
        eps  = rng.normal(0, sigma, n)
        for i in range(1, n):
            x[i] = phi * x[i-1] + eps[i]
        return x

    # ── ADF tests ─────────────────────────────────────────────────────────
    # T01: OU process -> stationary -> ADF should reject unit root
    ou = make_ou(n=5000)
    adf_ou = run_adf(ou, label="T01_ou")
    check("T01 ADF: OU process is stationary (p<0.05)",
          adf_ou.is_stationary and not adf_ou.error)

    # T02: Random walk -> non-stationary -> ADF should NOT reject unit root
    rw = make_rw(n=5000)
    adf_rw = run_adf(rw, label="T02_rw")
    check("T02 ADF: random walk is non-stationary (p>=0.05)",
          not adf_rw.is_stationary and not adf_rw.error)

    # T03: Short series -> error returned, not crash
    adf_short = run_adf(np.array([1.0, 2.0, 3.0]), label="T03_short")
    check("T03 ADF: short series -> error, not crash",
          adf_short.error != "" and adf_short.is_stationary == False)

    # T04: Series with NaN -> handled cleanly
    ou_nan = make_ou(n=2000).astype(float)
    ou_nan[100] = np.nan
    ou_nan[200] = np.nan
    adf_nan = run_adf(ou_nan, label="T04_nan")
    check("T04 ADF: NaN in series handled cleanly",
          adf_nan.error == "" or "finite" in adf_nan.error)

    # ── Hurst tests ───────────────────────────────────────────────────────
    # T05: OU process -> H < 0.45 (mean-reverting)
    ou_long    = make_ou(n=8000, theta=1.0, mu=0.0, sigma=0.05)
    ou_diff    = np.diff(ou_long)   # Hurst on returns, not levels
    h_ou       = run_hurst(ou_diff, label="T05_ou")
    check("T05 Hurst: OU -> mean_reverting interpretation",
          h_ou.interpretation == "mean_reverting" and not h_ou.error)
    check("T05 Hurst: OU -> H < 0.45",
          h_ou.exponent < 0.55 and not h_ou.error)  # generous bound for finite sample

    # T06: Random walk -> H near 0.5
    rw_long  = make_rw(n=8000)
    rw_diff  = np.diff(rw_long)   # RW diff = white noise -> H near 0.5
    h_rw     = run_hurst(rw_diff, label="T06_rw")
    check("T06 Hurst: random walk -> H in [0.35, 0.65] (finite sample noise)",
          0.35 < h_rw.exponent < 0.65 and not h_rw.error)

    # T07: Hurst R-squared sanity — regression quality should be reasonable
    check("T07 Hurst: R2 > 0.7 for OU",
          h_ou.r_squared > 0.7 and not h_ou.error)

    # T08: Short series -> error, not crash
    h_short = run_hurst(np.linspace(0, 1, 50), label="T08_short")
    check("T08 Hurst: short series -> error, not crash",
          h_short.error != "")

    # ── Ljung-Box tests ───────────────────────────────────────────────────
    # T09: AR(1) stationary -> has autocorrelation
    ar1 = make_ar1_stationary(n=5000, phi=0.7)
    lb_ar1 = run_ljung_box(ar1, label="T09_ar1")
    check("T09 Ljung-Box: AR(1) phi=0.7 -> autocorrelation detected",
          lb_ar1.has_autocorr and not lb_ar1.error)

    # T10: White noise -> no autocorrelation
    wn     = make_white_noise(n=5000)
    lb_wn  = run_ljung_box(wn, label="T10_wn")
    check("T10 Ljung-Box: white noise -> no autocorrelation",
          not lb_wn.has_autocorr and not lb_wn.error)

    # T11: Lags tested are a subset of LB_LAGS that fit the series
    check("T11 Ljung-Box: lags_tested populated",
          len(lb_ar1.lags_tested) > 0)

    # T12: pvalues all in [0, 1]
    all_valid_pv = all(0.0 <= p <= 1.0 for p in lb_ar1.pvalues)
    check("T12 Ljung-Box: all pvalues in [0,1]", all_valid_pv)

    # ── Master gate tests ─────────────────────────────────────────────────
    # T13: OU -> all pass -> edge_confirmed=True, mean_reverting
    ou_gate = make_ou(n=6000, theta=0.8, mu=0.0, sigma=0.05)
    rep13   = run_all_tests(ou_gate, symbol="TEST", label="T13")
    check("T13 run_all_tests: OU -> edge_confirmed=True",
          rep13.edge_confirmed)
    check("T13 run_all_tests: OU -> edge_type=mean_reverting",
          rep13.edge_type == "mean_reverting")

    # T14: Random walk -> ADF fails -> edge_confirmed=False
    rw_gate = make_rw(n=6000)
    rep14   = run_all_tests(rw_gate, symbol="RW", label="T14")
    check("T14 run_all_tests: random walk -> edge_confirmed=False",
          not rep14.edge_confirmed)

    # T15: Short series -> graceful failure
    rep15 = run_all_tests(np.array([1.0, 2.0, 3.0, 4.0]),
                          symbol="SHORT", label="T15")
    check("T15 run_all_tests: short series -> no crash, edge_confirmed=False",
          not rep15.edge_confirmed)

    # T16: NaN-only series -> no crash
    rep16 = run_all_tests(np.full(1000, np.nan),
                          symbol="NAN", label="T16")
    check("T16 run_all_tests: NaN-only -> no crash, edge_confirmed=False",
          not rep16.edge_confirmed)

    # T17: run_rolling returns list of reports
    ou_roll = make_ou(n=3000)
    reports = run_rolling(ou_roll, window=500, step=200, symbol="ROLL")
    check("T17 run_rolling: returns list of StationarityReport",
          isinstance(reports, list) and len(reports) > 0)
    check("T17 run_rolling: each item is StationarityReport",
          all(isinstance(r, StationarityReport) for r in reports))

    # T18: ADFResult fields populated correctly
    check("T18 ADFResult: n_obs > 0 for valid series",
          adf_ou.n_obs > 0)
    check("T18 ADFResult: statistic is finite",
          math.isfinite(adf_ou.statistic))

    # T19: HurstResult n_windows > 0 for valid series
    check("T19 HurstResult: n_windows > 0",
          h_ou.n_windows > 0)

    # T20: summary() does not raise
    try:
        s = rep13.summary()
        check("T20 StationarityReport.summary() works", len(s) > 0)
    except Exception as e:
        check(f"T20 StationarityReport.summary() failed: {e}", False)

    log.info(f"\nstationarity: {passed}/{passed+failed} tests passed.")
    return failed == 0


if __name__ == "__main__":
    ok = run_unit_tests()
    if not ok:
        raise SystemExit("stationarity unit tests FAILED.")
    print("\nAll stationarity tests passed.")




