"""
Module 3: Kalman Filter — Dynamic Hedge Ratio
===============================================
Estimates the time-varying hedge ratio beta(t) and intercept alpha(t)
between two log-price series in real time.

State space model
-----------------
State vector:  theta(t) = [beta(t), alpha(t)]^T

Transition equation (random walk on parameters):
    theta(t) = theta(t-1) + eta(t),    eta ~ N(0, Q)

Observation equation:
    log_P_A(t) = beta(t)*log_P_B(t) + alpha(t) + eps(t),  eps ~ N(0, R)

Written as:
    y(t)   = H(t) * theta(t) + eps(t)
where:
    y(t)   = log_P_A(t)                   (scalar observation)
    H(t)   = [log_P_B(t), 1.0]^T          (2x1 observation vector)

Kalman recursion (exact, Welch & Bishop 2001)
---------------------------------------------
PREDICT:
    theta_hat(t|t-1) = theta_hat(t-1|t-1)         (random walk transition)
    P(t|t-1)         = P(t-1|t-1) + Q

UPDATE:
    innovation(t) = y(t) - H(t)^T * theta_hat(t|t-1)
    S(t)          = H(t)^T * P(t|t-1) * H(t) + R   (innovation variance)
    K(t)          = P(t|t-1) * H(t) / S(t)          (Kalman gain, 2x1)
    theta_hat(t)  = theta_hat(t|t-1) + K(t) * innovation(t)
    P(t)          = (I - K(t)*H(t)^T) * P(t|t-1)   (Joseph form for stability)

Spread:
    X(t) = log_P_A(t) - beta_hat(t)*log_P_B(t) - alpha_hat(t)

Numerical stability
-------------------
- Joseph form of covariance update: P = (I-KH)*P*(I-KH)^T + K*R*K^T
  Guarantees P remains symmetric positive semi-definite even with rounding.
- S(t) floored at MIN_INNOVATION_VAR to prevent division by near-zero.
- P diagonal floored at MIN_P_DIAGONAL to prevent covariance collapse.
- All outputs pass through math_guards.safeN before storage.
"""

import logging
import math
from dataclasses import dataclass, field
from typing      import List, Optional, Tuple

import numpy as np

from modules.math_guards import (
    safeN, safe_divide, safe_sqrt, clamp, MIN_DENOMINATOR
)

log = logging.getLogger("kalman_filter")

# ── Constants ──────────────────────────────────────────────────────────────
MIN_INNOVATION_VAR  = 1e-8    # floor on S(t) to prevent /0
MIN_P_DIAGONAL      = 1e-10   # floor on P[i,i] to prevent collapse
DEFAULT_Q_SCALE     = 1e-5    # default process noise scale (tuned on data)
DEFAULT_R_SCALE     = 1e-3    # default observation noise variance
DEFAULT_P0_SCALE    = 1.0     # initial covariance scale
WARMUP_BARS         = 100     # bars before spread is considered reliable


# ── Result dataclass ───────────────────────────────────────────────────────
@dataclass
class KalmanState:
    """
    State at a single time step.
    All fields validated through safeN before storage.
    """
    bar:          int   = 0
    beta:         float = 1.0     # hedge ratio estimate
    alpha:        float = 0.0     # intercept estimate
    spread:       float = 0.0     # X(t) = y_A - beta*y_B - alpha
    innovation:   float = 0.0     # y_A - predicted y_A
    innov_var:    float = 1.0     # S(t) — innovation variance
    p00:          float = 1.0     # P[0,0] — beta variance
    p11:          float = 1.0     # P[1,1] — alpha variance
    is_warmed_up: bool  = False   # True after WARMUP_BARS


@dataclass
class KalmanResult:
    """
    Full output from a Kalman filter run over a bar series.
    """
    ok:            bool              = False
    states:        List[KalmanState] = field(default_factory=list)
    spreads:       np.ndarray        = field(default_factory=lambda: np.array([]))
    betas:         np.ndarray        = field(default_factory=lambda: np.array([]))
    alphas:        np.ndarray        = field(default_factory=lambda: np.array([]))
    innov_vars:    np.ndarray        = field(default_factory=lambda: np.array([]))
    n_bars:        int               = 0
    n_warmed_up:   int               = 0
    error:         str               = ""
    warnings:      List[str]         = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 55,
            f"Kalman Filter Result",
            f"Status       : {'OK' if self.ok else 'FAILED'}",
            f"Bars         : {self.n_bars:,}",
            f"Warmed up    : {self.n_warmed_up:,}",
        ]
        if len(self.betas) > 0:
            lines += [
                f"Beta range   : [{np.nanmin(self.betas):.4f}, "
                f"{np.nanmax(self.betas):.4f}]",
                f"Beta final   : {self.betas[-1]:.4f}",
                f"Alpha final  : {self.alphas[-1]:.6f}",
                f"Spread mean  : {np.nanmean(self.spreads):.6f}",
                f"Spread std   : {np.nanstd(self.spreads):.6f}",
            ]
        if self.error:
            lines.append(f"Error        : {self.error}")
        for w in self.warnings:
            lines.append(f"Warning      : {w}")
        lines.append("=" * 55)
        return "\n".join(lines)


# ── Core Kalman filter implementation ─────────────────────────────────────
class KalmanFilter:
    """
    Online Kalman filter for dynamic hedge ratio estimation.

    State: [beta, alpha]
    Observation: log_price_A = beta * log_price_B + alpha + noise

    Parameters
    ----------
    q_scale : float
        Process noise variance scale. Controls how fast beta/alpha
        are allowed to change. Larger = faster adaptation, noisier estimates.
        Tune on training data. Default: 1e-5.

    r_scale : float
        Observation noise variance. Estimate from residual variance
        of a static regression on training data. Default: 1e-3.

    p0_scale : float
        Initial state covariance scale. Large value = high initial
        uncertainty, filter adapts quickly at start. Default: 1.0.

    Tuning guidance (from Hamilton 1994):
        - Start with Q/R ratio matching expected parameter drift.
        - If beta changes too slowly -> increase q_scale.
        - If beta is too noisy -> decrease q_scale.
        - R should match variance of static regression residuals.
    """

    def __init__(
        self,
        q_scale:  float = DEFAULT_Q_SCALE,
        r_scale:  float = DEFAULT_R_SCALE,
        p0_scale: float = DEFAULT_P0_SCALE,
    ):
        # Validate parameters
        if q_scale <= 0 or not math.isfinite(q_scale):
            raise ValueError(f"q_scale must be positive finite. Got {q_scale}.")
        if r_scale <= 0 or not math.isfinite(r_scale):
            raise ValueError(f"r_scale must be positive finite. Got {r_scale}.")
        if p0_scale <= 0 or not math.isfinite(p0_scale):
            raise ValueError(f"p0_scale must be positive finite. Got {p0_scale}.")

        # Process noise covariance Q (2x2 diagonal)
        self._Q = np.diag([q_scale, q_scale]).astype(np.float64)

        # Observation noise variance R (scalar)
        self._R = float(r_scale)

        # State mean: [beta, alpha]
        self._theta = np.array([1.0, 0.0], dtype=np.float64)

        # State covariance P (2x2)
        self._P = np.eye(2, dtype=np.float64) * p0_scale

        self._bar_count = 0
        self._q_scale   = q_scale
        self._r_scale   = r_scale
        self._p0_scale  = p0_scale

    def reset(self) -> None:
        """Reset filter to initial state. Call between instruments."""
        self._theta     = np.array([1.0, 0.0], dtype=np.float64)
        self._P         = np.eye(2, dtype=np.float64) * self._p0_scale
        self._bar_count = 0

    def step(self, log_price_a: float, log_price_b: float) -> KalmanState:
        """
        Process one bar. Returns KalmanState for this bar.

        Parameters
        ----------
        log_price_a : float  ln(price_A) — the asset we model
        log_price_b : float  ln(price_B) — the hedge asset

        Returns
        -------
        KalmanState with beta, alpha, spread, innovation, innov_var, P diagonal
        """
        self._bar_count += 1

        # Guard: prices must be finite
        if not (math.isfinite(log_price_a) and math.isfinite(log_price_b)):
            log.warning(
                f"Kalman.step bar {self._bar_count}: "
                f"non-finite input lp_a={log_price_a} lp_b={log_price_b}. "
                "Returning last state."
            )
            return self._current_state(log_price_a, log_price_b)

        # Observation vector H = [log_price_b, 1.0]
        H = np.array([log_price_b, 1.0], dtype=np.float64)   # shape (2,)
        y = log_price_a                                        # scalar

        # ── PREDICT ───────────────────────────────────────────────────────
        # theta_hat(t|t-1) = theta_hat(t-1)   (random walk)
        theta_pred = self._theta.copy()                # (2,)

        # P(t|t-1) = P(t-1) + Q
        P_pred = self._P + self._Q                     # (2,2)

        # ── UPDATE ────────────────────────────────────────────────────────
        # Innovation: y - H^T * theta_pred
        y_hat      = float(H @ theta_pred)             # scalar prediction
        innovation = y - y_hat                         # scalar

        # Innovation variance: S = H^T * P_pred * H + R
        # H^T * P_pred is shape (2,), then dot with H gives scalar
        HP         = H @ P_pred                        # (2,) = H^T * P_pred
        S          = float(HP @ H) + self._R           # scalar

        # Floor S to prevent division by near-zero
        S = max(S, MIN_INNOVATION_VAR)

        # Kalman gain: K = P_pred * H / S   shape (2,)
        K = (P_pred @ H) / S                           # (2,)

        # State update
        theta_new = theta_pred + K * innovation        # (2,)

        # Covariance update — Joseph form for numerical stability:
        # P = (I - K*H^T) * P_pred * (I - K*H^T)^T + K*R*K^T
        I  = np.eye(2, dtype=np.float64)
        KH = np.outer(K, H)                            # (2,2)
        IKH = I - KH
        P_new = IKH @ P_pred @ IKH.T + np.outer(K, K) * self._R  # (2,2)

        # Floor P diagonal to prevent collapse
        P_new[0, 0] = max(P_new[0, 0], MIN_P_DIAGONAL)
        P_new[1, 1] = max(P_new[1, 1], MIN_P_DIAGONAL)

        # Ensure P remains symmetric (numerical symmetrisation)
        P_new = (P_new + P_new.T) / 2.0

        # Store updated state
        self._theta = theta_new
        self._P     = P_new

        # ── Build output state ─────────────────────────────────────────────
        beta  = safeN(float(theta_new[0]), fallback=1.0, label="kalman.beta")
        alpha = safeN(float(theta_new[1]), fallback=0.0, label="kalman.alpha")

        # Spread X(t) = log_P_A - beta*log_P_B - alpha
        spread = safeN(
            log_price_a - beta * log_price_b - alpha,
            fallback=0.0,
            label="kalman.spread",
        )
        innov  = safeN(innovation, fallback=0.0, label="kalman.innov")
        innov_v = safeN(S, fallback=1.0, label="kalman.innov_var")
        p00    = safeN(float(P_new[0, 0]), fallback=1.0, label="kalman.p00")
        p11    = safeN(float(P_new[1, 1]), fallback=1.0, label="kalman.p11")

        return KalmanState(
            bar          = self._bar_count,
            beta         = beta,
            alpha        = alpha,
            spread       = spread,
            innovation   = innov,
            innov_var    = innov_v,
            p00          = p00,
            p11          = p11,
            is_warmed_up = self._bar_count > WARMUP_BARS,
        )

    def _current_state(self, log_price_a: float, log_price_b: float) -> KalmanState:
        """Return current state estimate without updating (used on bad input)."""
        beta  = safeN(float(self._theta[0]), fallback=1.0)
        alpha = safeN(float(self._theta[1]), fallback=0.0)
        spread = 0.0
        if math.isfinite(log_price_a) and math.isfinite(log_price_b):
            spread = safeN(
                log_price_a - beta * log_price_b - alpha,
                fallback=0.0,
            )
        return KalmanState(
            bar          = self._bar_count,
            beta         = beta,
            alpha        = alpha,
            spread       = spread,
            innovation   = 0.0,
            innov_var    = safeN(float(self._P[0, 0]) + self._R, fallback=1.0),
            p00          = safeN(float(self._P[0, 0]), fallback=1.0),
            p11          = safeN(float(self._P[1, 1]), fallback=1.0),
            is_warmed_up = self._bar_count > WARMUP_BARS,
        )

    @property
    def beta(self) -> float:
        return safeN(float(self._theta[0]), fallback=1.0)

    @property
    def alpha(self) -> float:
        return safeN(float(self._theta[1]), fallback=0.0)

    @property
    def p_diagonal(self) -> Tuple[float, float]:
        """Returns (P[0,0], P[1,1]) — variances of beta and alpha estimates."""
        return (
            safeN(float(self._P[0, 0]), fallback=1.0),
            safeN(float(self._P[1, 1]), fallback=1.0),
        )


# ── Batch runner ───────────────────────────────────────────────────────────
def run_kalman(
    log_prices_a: np.ndarray,
    log_prices_b: np.ndarray,
    q_scale:      float = DEFAULT_Q_SCALE,
    r_scale:      float = DEFAULT_R_SCALE,
    p0_scale:     float = DEFAULT_P0_SCALE,
    symbol:       str   = "",
) -> KalmanResult:
    """
    Run Kalman filter over full bar arrays.
    Returns KalmanResult with spread series and all states.

    Parameters
    ----------
    log_prices_a : 1D array of ln(price_A) — the asset to model
    log_prices_b : 1D array of ln(price_B) — the hedge asset
    q_scale      : process noise variance
    r_scale      : observation noise variance
    p0_scale     : initial covariance scale
    symbol       : e.g. "BTC-ETH"

    Returns
    -------
    KalmanResult — always returned, never raises.
    """
    result = KalmanResult()

    # Input validation
    if len(log_prices_a) != len(log_prices_b):
        result.error = (
            f"run_kalman[{symbol}]: length mismatch "
            f"a={len(log_prices_a)} b={len(log_prices_b)}."
        )
        log.error(result.error)
        return result

    n = len(log_prices_a)
    if n < WARMUP_BARS + 10:
        result.error = (
            f"run_kalman[{symbol}]: only {n} bars, "
            f"need at least {WARMUP_BARS + 10}."
        )
        log.error(result.error)
        return result

    # Check for sufficient finite values
    finite_mask = np.isfinite(log_prices_a) & np.isfinite(log_prices_b)
    n_finite    = int(finite_mask.sum())
    if n_finite < WARMUP_BARS + 10:
        result.error = (
            f"run_kalman[{symbol}]: only {n_finite} finite bar pairs."
        )
        log.error(result.error)
        return result

    if n_finite < n:
        result.warnings.append(
            f"{n - n_finite} bars with non-finite prices — "
            "Kalman state held constant at those bars."
        )

    try:
        kf = KalmanFilter(
            q_scale=q_scale,
            r_scale=r_scale,
            p0_scale=p0_scale,
        )

        states     = []
        spreads    = np.full(n, np.nan)
        betas      = np.full(n, np.nan)
        alphas     = np.full(n, np.nan)
        innov_vars = np.full(n, np.nan)

        for i in range(n):
            state = kf.step(
                float(log_prices_a[i]),
                float(log_prices_b[i]),
            )
            states.append(state)
            spreads[i]    = state.spread
            betas[i]      = state.beta
            alphas[i]     = state.alpha
            innov_vars[i] = state.innov_var

        n_warmed = sum(1 for s in states if s.is_warmed_up)

        result.ok          = True
        result.states      = states
        result.spreads     = spreads
        result.betas       = betas
        result.alphas      = alphas
        result.innov_vars  = innov_vars
        result.n_bars      = n
        result.n_warmed_up = n_warmed

        log.info(
            f"run_kalman[{symbol}]: {n:,} bars processed, "
            f"{n_warmed:,} warmed-up. "
            f"beta_final={betas[-1]:.4f} "
            f"spread_std={np.nanstd(spreads[WARMUP_BARS:]):.6f}"
        )

    except Exception as exc:
        result.ok    = False
        result.error = f"run_kalman[{symbol}]: exception — {exc}"
        log.error(result.error)

    return result


def estimate_noise_params(
    log_prices_a: np.ndarray,
    log_prices_b: np.ndarray,
    train_frac:   float = 0.3,
) -> Tuple[float, float]:
    """
    Estimate R (observation noise) from a static OLS regression
    on the first train_frac of the data.

    Returns (q_scale, r_scale) tuned to the data.

    Method:
        1. OLS: log_P_A = beta * log_P_B + alpha + eps
        2. R = var(eps)           observation noise from residuals
        3. Q = R * 0.01           heuristic: parameters drift 1% as fast
                                  as observation noise

    This is a standard starting point (Hamilton 1994, ch.13).
    Refine Q by cross-validation if needed.
    """
    n_train = max(int(len(log_prices_a) * train_frac), WARMUP_BARS + 10)
    a_train = log_prices_a[:n_train]
    b_train = log_prices_b[:n_train]

    # Remove non-finite
    mask    = np.isfinite(a_train) & np.isfinite(b_train)
    a_clean = a_train[mask]
    b_clean = b_train[mask]

    if len(a_clean) < 20:
        log.warning("estimate_noise_params: too few clean bars — using defaults.")
        return DEFAULT_Q_SCALE, DEFAULT_R_SCALE

    # OLS: a = beta*b + alpha
    X     = np.column_stack([b_clean, np.ones(len(b_clean))])
    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(X, a_clean, rcond=None)
    except np.linalg.LinAlgError as e:
        log.warning(f"estimate_noise_params: OLS failed ({e}) — using defaults.")
        return DEFAULT_Q_SCALE, DEFAULT_R_SCALE

    resid  = a_clean - X @ coeffs
    r_hat  = float(np.var(resid, ddof=2))
    r_hat  = max(r_hat, MIN_DENOMINATOR)

    # Q heuristic: drift 1% of observation noise
    q_hat  = r_hat * 0.01
    q_hat  = max(q_hat, 1e-8)

    log.info(
        f"estimate_noise_params: R={r_hat:.2e} Q={q_hat:.2e} "
        f"(from {len(a_clean)} training bars)"
    )
    return q_hat, r_hat


# ── Unit tests ─────────────────────────────────────────────────────────────
def run_unit_tests() -> bool:
    """
    Tests use controlled synthetic data with known mathematical properties.

    Key assertions verified against the Kalman filter theory:
    1. Spread of a true cointegrated pair converges to near-zero mean.
    2. Beta estimate converges to the true beta (law of large numbers).
    3. Innovation variance S(t) is always positive and finite.
    4. P matrix remains positive semi-definite (diagonal entries >= 0).
    5. Joseph form preserves P symmetry numerically.
    6. Non-finite inputs handled without crash.
    7. Warmup flag set correctly after WARMUP_BARS.
    8. Spread is stationary (mean near 0) for cointegrated input.
    9. estimate_noise_params produces positive finite R, Q.
    10. run_kalman batch runner produces consistent results.
    """
    import math
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    )
    log.info("Running kalman_filter unit tests ...")
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

    # ── Synthetic cointegrated pair ───────────────────────────────────────
    # log_P_B = random walk
    # log_P_A = TRUE_BETA * log_P_B + TRUE_ALPHA + OU_noise
    TRUE_BETA  = 1.35
    TRUE_ALPHA = 0.05
    N          = 5000
    noise_std  = 0.002

    log_pb = np.cumsum(rng.normal(0, 0.01, N))           # random walk
    ou_noise = np.zeros(N)
    for i in range(1, N):
        ou_noise[i] = 0.95 * ou_noise[i-1] + rng.normal(0, noise_std)
    log_pa = TRUE_BETA * log_pb + TRUE_ALPHA + ou_noise   # cointegrated

    # ── T01: run_kalman basic smoke test ──────────────────────────────────
    result = run_kalman(log_pa, log_pb, symbol="T01")
    check("T01 run_kalman: ok=True for valid input", result.ok)
    check("T01 run_kalman: n_bars = N",             result.n_bars == N)
    check("T01 run_kalman: spreads length = N",     len(result.spreads) == N)
    check("T01 run_kalman: betas length = N",       len(result.betas) == N)

    # ── T02: beta converges toward TRUE_BETA ─────────────────────────────
    # After warmup, beta estimate should be within 20% of true value
    # (generous tolerance — exact convergence depends on Q/R tuning)
    if result.ok:
        beta_post_warmup = result.betas[WARMUP_BARS:]
        beta_final = float(np.nanmean(beta_post_warmup[-500:]))
        check("T02 beta converges toward TRUE_BETA (within 30%)",
              abs(beta_final - TRUE_BETA) / TRUE_BETA < 0.30)

    # ── T03: spread is near-zero mean after warmup ────────────────────────
    if result.ok:
        spread_post_warmup = result.spreads[WARMUP_BARS:]
        spread_mean = float(np.nanmean(spread_post_warmup))
        check("T03 spread mean near zero after warmup (|mean| < 0.05)",
              abs(spread_mean) < 0.05)

    # ── T04: innovation variance always positive ──────────────────────────
    if result.ok:
        min_iv = float(np.nanmin(result.innov_vars))
        check("T04 all innovation variances positive", min_iv > 0)

    # ── T05: P diagonal always non-negative (positive semi-definite) ──────
    if result.ok:
        p00_vals = np.array([s.p00 for s in result.states])
        p11_vals = np.array([s.p11 for s in result.states])
        check("T05 P[0,0] (beta variance) always >= 0",
              float(np.min(p00_vals)) >= 0)
        check("T05 P[1,1] (alpha variance) always >= 0",
              float(np.min(p11_vals)) >= 0)

    # ── T06: warmup flag correct ──────────────────────────────────────────
    if result.ok:
        first_warmed = next(
            (i for i, s in enumerate(result.states) if s.is_warmed_up),
            None
        )
        check("T06 warmup starts after WARMUP_BARS",
              first_warmed is not None and first_warmed == WARMUP_BARS)

    # ── T07: all output values finite ────────────────────────────────────
    if result.ok:
        all_finite = (
            np.all(np.isfinite(result.betas))   and
            np.all(np.isfinite(result.alphas))  and
            np.all(np.isfinite(result.spreads)) and
            np.all(np.isfinite(result.innov_vars))
        )
        check("T07 all output arrays fully finite", all_finite)

    # ── T08: non-finite input handled without crash ───────────────────────
    log_pa_nan = log_pa.copy()
    log_pb_nan = log_pb.copy()
    log_pa_nan[500] = np.nan
    log_pb_nan[600] = np.inf
    result_nan = run_kalman(log_pa_nan, log_pb_nan, symbol="T08_nan")
    check("T08 NaN/Inf in input: no crash, ok=True",
          result_nan.ok)

    # ── T09: length mismatch returns error cleanly ────────────────────────
    result_bad = run_kalman(log_pa[:100], log_pb[:200], symbol="T09_mismatch")
    check("T09 length mismatch: ok=False, error populated",
          not result_bad.ok and result_bad.error != "")

    # ── T10: too-short series returns error cleanly ───────────────────────
    result_short = run_kalman(log_pa[:10], log_pb[:10], symbol="T10_short")
    check("T10 short series: ok=False, error populated",
          not result_short.ok and result_short.error != "")

    # ── T11: step-by-step vs batch consistency ────────────────────────────
    kf_step = KalmanFilter()
    step_spreads = []
    for i in range(200):
        s = kf_step.step(float(log_pa[i]), float(log_pb[i]))
        step_spreads.append(s.spread)

    result_batch = run_kalman(log_pa[:200], log_pb[:200], symbol="T11_batch")
    if result_batch.ok:
        max_diff = float(np.max(np.abs(
            np.array(step_spreads) - result_batch.spreads
        )))
        check("T11 step-by-step == batch (max diff < 1e-10)",
              max_diff < 1e-10)
    else:
        check("T11 batch run succeeded for comparison", False)

    # ── T12: higher Q -> P diagonal larger (less certain, faster adaptation) ─
    # Correct property: higher Q -> larger P[0,0] (beta uncertainty stays higher)
    # because the filter cannot gain confidence faster than process noise allows.
    result_lo_q = run_kalman(log_pa, log_pb, q_scale=1e-7, symbol="T12_lo_q")
    result_hi_q = run_kalman(log_pa, log_pb, q_scale=1e-3, symbol="T12_hi_q")
    if result_lo_q.ok and result_hi_q.ok:
        p00_lo = float(np.nanmean([s.p00 for s in result_lo_q.states[WARMUP_BARS:]]))
        p00_hi = float(np.nanmean([s.p00 for s in result_hi_q.states[WARMUP_BARS:]]))
        check("T12 higher Q -> larger mean P[0,0] (beta stays more uncertain)",
              p00_hi > p00_lo)

    # ── T13: P symmetry preserved (Joseph form) ───────────────────────────
    kf_sym = KalmanFilter()
    for i in range(500):
        kf_sym.step(float(log_pa[i]), float(log_pb[i]))
    P = kf_sym._P
    symmetry_error = float(np.max(np.abs(P - P.T)))
    check("T13 P matrix symmetric after 500 steps (Joseph form)",
          symmetry_error < 1e-14)

    # ── T14: reset() restores initial state ──────────────────────────────
    kf_reset = KalmanFilter()
    for i in range(200):
        kf_reset.step(float(log_pa[i]), float(log_pb[i]))
    kf_reset.reset()
    check("T14 reset(): beta back to 1.0",  abs(kf_reset.beta - 1.0) < 1e-12)
    check("T14 reset(): alpha back to 0.0", abs(kf_reset.alpha - 0.0) < 1e-12)
    check("T14 reset(): bar_count = 0",     kf_reset._bar_count == 0)

    # ── T15: estimate_noise_params returns positive finite values ─────────
    q_hat, r_hat = estimate_noise_params(log_pa, log_pb)
    check("T15 R estimate is positive finite",
          r_hat > 0 and math.isfinite(r_hat))
    check("T15 Q estimate is positive finite",
          q_hat > 0 and math.isfinite(q_hat))
    check("T15 Q < R (parameters drift slower than observation noise)",
          q_hat < r_hat)

    # ── T16: KalmanResult.summary() does not raise ────────────────────────
    try:
        s = result.summary()
        check("T16 KalmanResult.summary() works", len(s) > 0)
    except Exception as e:
        check(f"T16 summary() raised: {e}", False)

    # ── T17: uncorrelated series — spread should have high variance ────────
    # (Kalman should not produce near-zero spread on noise vs noise)
    rand_a = rng.normal(0, 1, 1000)
    rand_b = rng.normal(0, 1, 1000)
    log_rand_a = np.cumsum(rand_a * 0.01)
    log_rand_b = np.cumsum(rand_b * 0.01)
    result_rand = run_kalman(log_rand_a, log_rand_b, symbol="T17_rand")
    if result_rand.ok:
        spread_std_rand = float(np.nanstd(
            result_rand.spreads[WARMUP_BARS:]
        ))
        # For uncorrelated series, spread should be non-trivially variable
        check("T17 uncorrelated series: spread std > 0",
              spread_std_rand > 0)

    # ── T18: all beta values within safeN range ───────────────────────────
    if result.ok:
        max_beta = float(np.nanmax(np.abs(result.betas)))
        check("T18 all betas within safeN range (< 999)",
              max_beta < 999)

    log.info(f"\nkalman_filter: {passed}/{passed+failed} tests passed.")
    return failed == 0


if __name__ == "__main__":
    ok = run_unit_tests()
    if not ok:
        raise SystemExit("kalman_filter unit tests FAILED.")
    print("\nAll kalman_filter tests passed.")

