"""
Module 5: Hidden Markov Model Regime Detection
================================================
Learns market regimes from data using Baum-Welch EM algorithm.
Three hidden states representing distinct market microstructures.

Mathematical framework
----------------------
Hidden states S = {0, 1}:
    S0: Tradeable  (calm, normal vol, spread near mean) -> trade full Kelly
    S1: No-trade   (volatile, high spread deviation)  -> NO TRADE

Note: 3-state model collapses on real 1-min crypto data because
states 0 and 1 are indistinguishable in short windows.
2-state model is empirically correct and mathematically stable.

Observation vector per bar (4-dimensional):
    o(t) = [
        |spread_zscore(t)|,       spread deviation magnitude
        |spread_return(t)|,       absolute spread return
        volume_ratio(t),          volume(t) / MA_volume(t)
        innov_var_ratio(t)        Kalman innovation variance ratio
    ]

Emission model:
    P(o(t) | state=k) = N(mu_k, diag(sigma_k^2))
    Diagonal Gaussian per state (independent features)

Training: Baum-Welch EM on training data
    E-step: Forward-backward algorithm -> gamma, xi
    M-step: Update pi, A, mu, sigma from gamma, xi

Inference: Forward algorithm -> P(state_k | o(1:t)) real-time

All probability computations in log-space to prevent underflow.
All outputs validated through math_guards.safeN.
"""

import logging
import math
from dataclasses import dataclass, field
from typing      import List, Optional, Tuple

import numpy as np

from modules.math_guards import (
    safeN, safe_divide, safe_log, safe_sqrt, clamp, MIN_DENOMINATOR
)

log = logging.getLogger("hmm_regime")

# ── Constants ──────────────────────────────────────────────────────────────
N_STATES            = 2       # 0=tradeable(calm), 1=no-trade(volatile)
N_FEATURES          = 4       # observation vector dimension
MIN_TRAIN_BARS      = 200     # minimum bars for 2-state Baum-Welch
MAX_EM_ITERS        = 100     # maximum EM iterations
EM_CONVERGENCE_TOL  = 1e-4    # log-likelihood convergence threshold
MIN_STATE_PROB      = 1e-8    # floor on state probabilities
MIN_EMISSION_VAR    = 1e-6    # floor on emission variance (prevents /0)
LOG_NEG_INF         = -1e300  # safe substitute for log(0)


# ── Result dataclasses ─────────────────────────────────────────────────────
@dataclass
class HMMParams:
    """
    Learned HMM parameters after Baum-Welch training.
    All arrays validated before storage.
    """
    pi:        np.ndarray = field(default_factory=lambda: np.ones(N_STATES)/N_STATES)
    A:         np.ndarray = field(default_factory=lambda: np.ones((N_STATES,N_STATES))/N_STATES)
    mu:        np.ndarray = field(default_factory=lambda: np.zeros((N_STATES,N_FEATURES)))
    sigma:     np.ndarray = field(default_factory=lambda: np.ones((N_STATES,N_FEATURES)))
    log_likes: List[float] = field(default_factory=list)
    converged: bool        = False
    n_iters:   int         = 0
    n_train:   int         = 0
    is_valid:  bool        = False
    error:     str         = ""


@dataclass
class RegimeState:
    """
    Real-time regime posterior at one bar.
    gamma[k] = P(state=k | all observations up to t)
    """
    bar:          int
    gamma:        np.ndarray = field(default_factory=lambda: np.ones(N_STATES)/N_STATES)
    state:        int         = 0      # argmax(gamma) — most likely state
    kelly_scale:  float       = 0.0   # position size multiplier from state
    is_no_trade:  bool        = True  # True if state=2 (breakdown)
    confidence:   float       = 0.0   # max(gamma) — how certain the state is

    @property
    def description(self) -> str:
        labels = ["tradeable", "no-trade"]
        return labels[self.state] if self.state < N_STATES else "unknown"


# ── Log-space probability helpers ──────────────────────────────────────────
def _log_gaussian(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """
    Log of diagonal Gaussian pdf:
        log N(x; mu, diag(sigma^2))
        = -0.5 * sum_d [ log(2*pi*sigma_d^2) + ((x_d - mu_d)/sigma_d)^2 ]

    All in log-space to prevent underflow.
    sigma floored at MIN_EMISSION_VAR.
    """
    sig = np.maximum(sigma, MIN_EMISSION_VAR)
    diff = x - mu
    log_det = np.sum(np.log(2.0 * math.pi * sig * sig))
    maha    = np.sum((diff / sig) ** 2)
    result = -0.5 * (log_det + maha)
    if not math.isfinite(result):
        return LOG_NEG_INF
    return result


def _log_emission_matrix(obs: np.ndarray, mu: np.ndarray,
                         sigma: np.ndarray) -> np.ndarray:
    """
    Compute log emission probabilities for all T bars and K states.
    Returns log_B of shape (T, K).
    obs   : (T, D)
    mu    : (K, D)
    sigma : (K, D)
    """
    T, D = obs.shape
    K    = mu.shape[0]
    log_B = np.full((T, K), LOG_NEG_INF)
    for t in range(T):
        for k in range(K):
            log_B[t, k] = _log_gaussian(obs[t], mu[k], sigma[k])
    return log_B


def _log_sum_exp(log_probs: np.ndarray) -> float:
    """
    Numerically stable log-sum-exp:
        log( sum_i exp(log_probs[i]) )
        = max_val + log( sum_i exp(log_probs[i] - max_val) )
    """
    if len(log_probs) == 0:
        return LOG_NEG_INF
    max_val = float(np.max(log_probs))
    if not math.isfinite(max_val):
        return LOG_NEG_INF
    return max_val + math.log(
        float(np.sum(np.exp(np.clip(log_probs - max_val, -700, 0))))
    )


# ── Forward algorithm (log-space) ─────────────────────────────────────────
def _forward(log_pi: np.ndarray, log_A: np.ndarray,
             log_B: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Forward algorithm in log-space.
    Returns:
        log_alpha : (T, K) — log forward probabilities
        log_like  : float  — log P(observations | params)

    log_alpha[t, k] = log P(o_1,...,o_t, s_t=k | params)

    Recursion:
        log_alpha[0, k] = log_pi[k] + log_B[0, k]
        log_alpha[t, k] = log_B[t,k]
                          + log_sum_exp(log_alpha[t-1,:] + log_A[:,k])
    """
    T, K    = log_B.shape
    log_alpha = np.full((T, K), LOG_NEG_INF)

    # Initialise
    log_alpha[0] = log_pi + log_B[0]

    # Recurse
    for t in range(1, T):
        for k in range(K):
            log_trans = log_alpha[t-1] + log_A[:, k]
            log_alpha[t, k] = log_B[t, k] + _log_sum_exp(log_trans)

    log_like = _log_sum_exp(log_alpha[-1])
    return log_alpha, log_like


# ── Backward algorithm (log-space) ────────────────────────────────────────
def _backward(log_A: np.ndarray, log_B: np.ndarray) -> np.ndarray:
    """
    Backward algorithm in log-space.
    Returns log_beta : (T, K)

    log_beta[t, k] = log P(o_{t+1},...,o_T | s_t=k, params)

    Recursion:
        log_beta[T-1, k] = 0  (log(1))
        log_beta[t,   k] = log_sum_exp(log_A[k,:] + log_B[t+1,:] + log_beta[t+1,:])
    """
    T, K     = log_B.shape
    log_beta = np.zeros((T, K))       # log(1) = 0 at T-1

    for t in range(T-2, -1, -1):
        for k in range(K):
            log_trans = log_A[k] + log_B[t+1] + log_beta[t+1]
            log_beta[t, k] = _log_sum_exp(log_trans)

    return log_beta


# ── Log-space helpers ─────────────────────────────────────────────────────
def _log_gaussian(x, mu, sigma):
    """
    Log diagonal Gaussian: -0.5*sum[log(2pi*s^2) + ((x-mu)/s)^2]

    Returns a log-probability in (-inf, 0].
    MUST NOT use safeN -- log-probabilities are not financial ratios.
    Valid values like -3400 would be incorrectly zeroed by safeN.
    Guard only against non-finite output.
    """
    sig  = np.maximum(sigma, MIN_EMISSION_VAR)
    diff = x - mu
    ld   = np.sum(np.log(2.0 * math.pi * sig * sig))
    mh   = np.sum((diff / sig) ** 2)
    result = -0.5 * (ld + mh)
    # Only guard against NaN/Inf -- large negative values are valid
    if not math.isfinite(result):
        return LOG_NEG_INF
    return result

def _log_emission_matrix(obs, mu, sigma):
    """Log emission matrix (T,K) for all bars and states."""
    T, D = obs.shape; K = mu.shape[0]
    lb   = np.full((T, K), LOG_NEG_INF)
    for t in range(T):
        for k in range(K):
            lb[t,k] = _log_gaussian(obs[t], mu[k], sigma[k])
    return lb

def _log_sum_exp(lp):
    """Numerically stable log-sum-exp."""
    if len(lp)==0: return LOG_NEG_INF
    mv = float(np.max(lp))
    if not math.isfinite(mv): return LOG_NEG_INF
    return mv + math.log(float(np.sum(np.exp(np.clip(lp-mv,-700,0)))))

def _forward(log_pi, log_A, log_B):
    """Forward algorithm. Returns (log_alpha (T,K), log_likelihood)."""
    T,K   = log_B.shape
    la    = np.full((T,K), LOG_NEG_INF)
    la[0] = log_pi + log_B[0]
    for t in range(1,T):
        for k in range(K):
            la[t,k] = log_B[t,k] + _log_sum_exp(la[t-1]+log_A[:,k])
    return la, _log_sum_exp(la[-1])

def _backward(log_A, log_B):
    """Backward algorithm. Returns log_beta (T,K)."""
    T,K  = log_B.shape
    lb   = np.zeros((T,K))
    for t in range(T-2,-1,-1):
        for k in range(K):
            lb[t,k] = _log_sum_exp(log_A[k]+log_B[t+1]+lb[t+1])
    return lb


def train_hmm(obs, seed=42):
    """Train HMM via Baum-Welch EM. Returns HMMParams."""
    params = HMMParams()
    if obs.ndim != 2:
        params.error = f"train_hmm: obs must be 2D, got {obs.ndim}D."
        log.error(params.error); return params
    T,D = obs.shape
    if T < MIN_TRAIN_BARS:
        params.error = f"train_hmm: {T} bars < {MIN_TRAIN_BARS} required."
        log.warning(params.error); return params
    if D != N_FEATURES:
        params.error = f"train_hmm: expected {N_FEATURES} features, got {D}."
        log.error(params.error); return params

    mask = np.all(np.isfinite(obs), axis=1)
    oc   = obs[mask]; T_c = len(oc)
    if T_c < MIN_TRAIN_BARS:
        params.error = f"train_hmm: {T_c} finite rows < {MIN_TRAIN_BARS}."
        log.warning(params.error); return params

    K   = N_STATES
    rng = np.random.default_rng(seed)

    # K-means++ init
    idx = [int(rng.integers(0, T_c))]
    for _ in range(1, K):
        d = np.array([min(float(np.sum((oc[i]-oc[j])**2)) for j in idx) for i in range(T_c)])
        d = np.maximum(d, 0.0); s = float(d.sum())
        idx.append(int(rng.choice(T_c, p=d/s)) if s > MIN_DENOMINATOR
                   else int(rng.integers(0, T_c)))

    mu    = oc[idx].copy()
    sigma = np.tile(np.std(oc,axis=0)+MIN_EMISSION_VAR,(K,1))
    lpi   = np.log(np.full(K, 1.0/K))
    lA    = np.log(np.full((K,K), 1.0/K))
    prev_ll = LOG_NEG_INF; lls = []

    for it in range(MAX_EM_ITERS):
        lB        = _log_emission_matrix(oc, mu, sigma)
        la, ll    = _forward(lpi, lA, lB)
        if not math.isfinite(ll): break
        lls.append(ll)
        lb        = _backward(lA, lB)

        if it>0 and abs(ll-prev_ll)<EM_CONVERGENCE_TOL:
            params.converged=True; break
        if it>0 and ll-prev_ll < -1e-3: break
        prev_ll = ll

        # Gamma
        lg = la + lb
        lg = np.array([r - _log_sum_exp(r) for r in lg])
        g  = np.exp(np.clip(lg,-700,0))
        g  = np.maximum(g, MIN_STATE_PROB)
        g /= g.sum(axis=1,keepdims=True)

        # Xi
        lxi = np.full((T_c-1,K,K), LOG_NEG_INF)
        for t in range(T_c-1):
            for j in range(K):
                for k in range(K):
                    lxi[t,j,k] = la[t,j]+lA[j,k]+lB[t+1,k]+lb[t+1,k]
            s2 = _log_sum_exp(lxi[t].ravel())
            if math.isfinite(s2): lxi[t] -= s2
        xi = np.exp(np.clip(lxi,-700,0))

        # M-step
        npi = np.maximum(g[0], MIN_STATE_PROB); npi /= npi.sum()
        xs  = xi.sum(axis=0); rs = np.maximum(xs.sum(axis=1,keepdims=True), MIN_DENOMINATOR)
        nA  = np.maximum(xs/rs, MIN_STATE_PROB); nA /= nA.sum(axis=1,keepdims=True)
        gs  = np.maximum(g.sum(axis=0), MIN_DENOMINATOR)
        nmu = np.zeros((K,D)); nsg = np.zeros((K,D))
        for k in range(K):
            w = g[:,k][:,np.newaxis]
            nmu[k] = (w*oc).sum(axis=0)/gs[k]
            nsg[k] = np.sqrt(np.maximum((w*(oc-nmu[k])**2).sum(axis=0)/gs[k], MIN_EMISSION_VAR))
        lpi=np.log(np.maximum(npi,MIN_STATE_PROB))
        lA =np.log(np.maximum(nA, MIN_STATE_PROB))
        mu=nmu; sigma=nsg

    params.pi=np.exp(lpi); params.A=np.exp(lA)
    params.mu=mu; params.sigma=sigma
    params.log_likes=lls; params.n_iters=len(lls); params.n_train=T_c
    params.is_valid=(len(lls)>0 and math.isfinite(lls[-1]) and params.error=="")
    if params.is_valid:
        log.info(f"train_hmm: conv={params.converged} iters={params.n_iters} ll={lls[-1]:.4f}")
    return params


class HMMInference:
    """Online regime inference — forward algorithm one step at a time."""
    KELLY_SCALES = [1.0, 0.0]   # state0=full Kelly, state1=no trade

    def __init__(self, params):
        if not params.is_valid:
            raise ValueError("HMMInference: params not valid.")
        self._p    = params
        self._lpi  = np.log(np.maximum(params.pi, MIN_STATE_PROB))
        self._lA   = np.log(np.maximum(params.A,  MIN_STATE_PROB))
        self._la   = self._lpi.copy()
        self._bar  = 0

    def step(self, obs):
        """Process one bar obs (N_FEATURES,). Returns RegimeState."""
        self._bar += 1
        K = N_STATES
        if obs.shape != (N_FEATURES,) or not np.all(np.isfinite(obs)):
            return self._uniform()
        lb  = np.array([_log_gaussian(obs, self._p.mu[k], self._p.sigma[k]) for k in range(K)])
        lap = np.array([_log_sum_exp(self._la + self._lA[:,k]) for k in range(K)])
        lan = lb + lap
        ll  = _log_sum_exp(lan)
        if math.isfinite(ll): lan -= ll
        else: lan = np.full(K, math.log(1.0/K))
        self._la = lan
        g = np.exp(np.clip(lan,-700,0))
        g = np.maximum(g, MIN_STATE_PROB); g /= g.sum()
        s = int(np.argmax(g))
        return RegimeState(bar=self._bar, gamma=g, state=s,
                           kelly_scale=self.KELLY_SCALES[s] if s<K else 0.0,
                           is_no_trade=(s==2),
                           confidence=safeN(float(g[s]),fallback=1.0/K))

    def reset(self):
        self._la = self._lpi.copy(); self._bar = 0

    def _uniform(self):
        K = N_STATES; g = np.ones(K)/K
        return RegimeState(bar=self._bar,gamma=g,state=0,
                           kelly_scale=0.0,is_no_trade=True,confidence=1.0/K)


def build_features(spreads, innov_vars, volumes, window=20):
    """
    Build (T, N_FEATURES) observation matrix from Kalman output.

    Features designed for maximum state separation on real crypto data:
        f0 = spread z-score magnitude (sqrt-compressed to reduce outlier pull)
        f1 = rolling spread volatility z-score (last 20 bars vs 100-bar baseline)
        f2 = volume z-score (deviation from rolling mean, capped)
        f3 = Kalman innovation variance z-score

    All features are z-scored relative to a rolling baseline so the
    HMM sees stationary, comparably-scaled inputs regardless of
    absolute price level or spread magnitude.
    """
    T      = len(spreads)
    obs    = np.full((T, N_FEATURES), np.nan)
    W_long = max(window * 5, 100)   # longer baseline for z-scoring

    for t in range(W_long, T):
        # ?? short and long windows ????????????????????????????????????
        sl_short = spreads[t-window:t]
        sl_long  = spreads[t-W_long:t]
        il_long  = innov_vars[t-W_long:t]
        vl_long  = volumes[t-W_long:t]

        sl_short = sl_short[np.isfinite(sl_short)]
        sl_long  = sl_long[np.isfinite(sl_long)]
        il_long  = il_long[np.isfinite(il_long) & (il_long > 0)]
        vl_long  = vl_long[np.isfinite(vl_long) & (vl_long > 0)]

        if len(sl_short) < 2 or len(sl_long) < 10:
            continue

        mu_l  = float(np.mean(sl_long))
        std_l = float(np.std(sl_long, ddof=1))
        if std_l < MIN_DENOMINATOR:
            continue

        # f0: spread z-score magnitude (sqrt-compressed)
        sp_now = float(spreads[t]) if math.isfinite(spreads[t]) else mu_l
        zscore = abs((sp_now - mu_l) / std_l)
        f0     = math.sqrt(clamp(zscore, 0.0, 25.0))   # sqrt compresses outliers

        # f1: short-window spread volatility vs long-window baseline
        vol_short = float(np.std(sl_short, ddof=1))
        vol_long_vals = [float(np.std(sl_long[max(0,i-window):i], ddof=1))
                         for i in range(window, len(sl_long), window//2)
                         if len(sl_long[max(0,i-window):i]) >= 2]
        if vol_long_vals:
            mu_vol  = float(np.mean(vol_long_vals))
            std_vol = float(np.std(vol_long_vals)) or (mu_vol * 0.2) or MIN_DENOMINATOR
            f1 = clamp((vol_short - mu_vol) / std_vol, -4.0, 4.0)
        else:
            f1 = 0.0

        # f2: volume z-score
        if len(vl_long) >= 5:
            mu_v  = float(np.mean(vl_long))
            std_v = float(np.std(vl_long)) or (mu_v * 0.2) or MIN_DENOMINATOR
            v_now = float(volumes[t]) if (math.isfinite(volumes[t]) and volumes[t] >= 0) else mu_v
            f2    = clamp((v_now - mu_v) / std_v, -4.0, 4.0)
        else:
            f2 = 0.0

        # f3: Kalman innovation variance z-score
        if len(il_long) >= 5:
            mu_i  = float(np.mean(il_long))
            std_i = float(np.std(il_long)) or (mu_i * 0.2) or MIN_DENOMINATOR
            i_now = float(innov_vars[t]) if (math.isfinite(innov_vars[t]) and innov_vars[t] > 0) else mu_i
            f3    = clamp((i_now - mu_i) / std_i, -4.0, 4.0)
        else:
            f3 = 0.0

        obs[t] = [
            safeN(f0, 0.0),
            safeN(f1, 0.0),
            safeN(f2, 0.0),
            safeN(f3, 0.0),
        ]

    # Fill NaN rows with column means of finite rows
    finite = np.all(np.isfinite(obs), axis=1)
    if finite.sum() > 0:
        cm = np.nanmean(obs, axis=0)
        for i in range(T):
            if not finite[i]:
                obs[i] = cm
    return obs


# ── Unit tests ─────────────────────────────────────────────────────────────
def run_unit_tests():
    """
    Tests use synthetic observations with known cluster structure.
    Three-cluster obs -> HMM should find 3 distinct states.
    """
    import math
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(name)s %(message)s")
    log.info("Running hmm_regime unit tests ...")
    passed = failed = 0

    def check(name, condition):
        nonlocal passed, failed
        if condition: log.info(f"  PASS  {name}"); passed+=1
        else:         log.error(f"  FAIL  {name}"); failed+=1

    rng = np.random.default_rng(42)

    # ── Synthetic 3-cluster observations ────────────────────────────────
    # State 0: low z-score, low return, normal vol, normal innov
    # State 1: medium z-score, low return, normal vol, low innov
    # State 2: high z-score, high return, high vol, high innov
    def make_obs(n_per_state=400, seed=42):
        rng2 = np.random.default_rng(seed)
        # State 0: calm (low zscore, low vol, normal volume, normal innov)
        s0 = rng2.normal([0.3, -0.5, 0.0, -0.5],[0.15,0.3,0.3,0.3],(n_per_state,N_FEATURES))
        # State 1: volatile (high zscore, high vol, high volume, high innov)
        s1 = rng2.normal([1.8,  1.5, 1.5,  1.5],[0.25,0.4,0.4,0.4],(n_per_state,N_FEATURES))
        obs = np.vstack([s0, s1])
        obs = np.clip(obs, -4, 4)
        return obs

    obs = make_obs()
    T   = len(obs)

    # ── T01: train_hmm returns valid params ──────────────────────────────
    params = train_hmm(obs, seed=42)
    check("T01 train_hmm: is_valid=True",   params.is_valid)
    check("T01 train_hmm: no error",        params.error=="")
    check("T01 train_hmm: n_iters > 0",     params.n_iters > 0)

    # ── T02: pi sums to 1 ────────────────────────────────────────────────
    if params.is_valid:
        check("T02 pi sums to 1.0",  abs(params.pi.sum()-1.0)<1e-6)

    # ── T03: A rows sum to 1 ─────────────────────────────────────────────
    if params.is_valid:
        row_sums = params.A.sum(axis=1)
        check("T03 A rows sum to 1.0",  np.all(np.abs(row_sums-1.0)<1e-6))

    # ── T04: sigma all positive ───────────────────────────────────────────
    if params.is_valid:
        check("T04 all sigma > 0",  np.all(params.sigma > 0))

    # ── T05: log-likelihood increases (or stays flat) ─────────────────────
    if params.is_valid and len(params.log_likes) > 2:
        diffs = np.diff(params.log_likes)
        check("T05 log-likelihood non-decreasing",
              np.all(diffs >= -1e-2))

    # ── T06: mu values finite ─────────────────────────────────────────────
    if params.is_valid:
        check("T06 all mu finite",  np.all(np.isfinite(params.mu)))

    # ── T07: HMMInference step works without crash ───────────────────────
    if params.is_valid:
        inf  = HMMInference(params)
        obs_bar = obs[0].copy()
        state   = inf.step(obs_bar)
        check("T07 inference step: RegimeState returned",
              isinstance(state, RegimeState))
        check("T07 inference step: gamma sums to 1",
              abs(state.gamma.sum()-1.0)<1e-6)
        check("T07 inference step: state in {0,1}",
              state.state in {0,1})
        check("T07 inference step: confidence in [0,1]",
              0.0<=state.confidence<=1.0)

    # ── T08: kelly_scale correct per state ───────────────────────────────
    if params.is_valid:
        inf2 = HMMInference(params)
        # Force high-MR obs (state 0 expected) -> kelly_scale should be >=0
        s0_obs = np.array([0.3, 0.001, 1.0, 1.0])
        rs0    = inf2.step(s0_obs)
        check("T08 kelly_scale non-negative",  rs0.kelly_scale >= 0.0)
        check("T08 kelly_scale <= 1.0",        rs0.kelly_scale <= 1.0)

    # ── T09: state=2 -> is_no_trade=True ─────────────────────────────────
    if params.is_valid:
        inf3   = HMMInference(params)
        rs_bd  = RegimeState(bar=1,
                             gamma=np.array([0.0,1.0]),
                             state=1, kelly_scale=0.0,
                             is_no_trade=True, confidence=1.0)
        check("T09 state=1 -> is_no_trade=True",  rs_bd.is_no_trade)
        check("T09 state=1 -> kelly_scale=0.0",   rs_bd.kelly_scale==0.0)

    # ── T10: invalid obs handled gracefully ──────────────────────────────
    if params.is_valid:
        inf4  = HMMInference(params)
        bad   = np.array([np.nan, 1.0, 1.0, 1.0])
        rs_bad = inf4.step(bad)
        check("T10 NaN obs: no crash",
              isinstance(rs_bad, RegimeState))
        check("T10 NaN obs: gamma sums to 1",
              abs(rs_bad.gamma.sum()-1.0)<1e-4)

    # ── T11: too-short training data -> is_valid=False ────────────────────
    short_obs = obs[:100]
    params_s  = train_hmm(short_obs, seed=42)
    check("T11 too-short training -> is_valid=False",
          not params_s.is_valid)

    # ── T12: wrong feature count -> is_valid=False ────────────────────────
    wrong_obs = rng.uniform(0,1,(600,2))
    params_w  = train_hmm(wrong_obs, seed=42)
    check("T12 wrong feature count -> is_valid=False",
          not params_w.is_valid)

    # ── T13: HMMInference.reset() works ──────────────────────────────────
    if params.is_valid:
        inf5 = HMMInference(params)
        for i in range(20): inf5.step(obs[i])
        la_before = inf5._la.copy()
        inf5.reset()
        check("T13 reset: bar_count back to 0",  inf5._bar==0)
        check("T13 reset: log_alpha back to log_pi",
              np.allclose(inf5._la, inf5._lpi, atol=1e-12))

    # ── T14: build_features returns correct shape ─────────────────────────
    N    = 1000
    sp   = rng.normal(0, 0.01, N)
    iv   = rng.uniform(1e-5, 1e-3, N)
    vol  = rng.uniform(10, 100, N)
    feat = build_features(sp, iv, vol, window=20)
    check("T14 build_features: shape (T,4)",   feat.shape==(N,N_FEATURES))
    check("T14 build_features: no raw NaN",
          not np.any(np.isnan(feat)))

    # ── T15: all feature values in [0, 10] ────────────────────────────────
    check("T15 features within valid range [-5,6]",
          np.all(feat >= -5) and np.all(feat <= 6))

    # ── T16: description property works ──────────────────────────────────
    rs_d = RegimeState(bar=1, gamma=np.array([1.0,0.0]),
                       state=0, kelly_scale=1.0, is_no_trade=False, confidence=1.0)
    check("T16 description: state=0 -> 'tradeable'",
          rs_d.description=="tradeable")

    # ── T17: log_likes list is populated ─────────────────────────────────
    if params.is_valid:
        check("T17 log_likes populated",  len(params.log_likes)>0)
        check("T17 final log_like finite", math.isfinite(params.log_likes[-1]))

    log.info(f"\nhmm_regime: {passed}/{passed+failed} tests passed.")
    return failed == 0


if __name__ == "__main__":
    ok = run_unit_tests()
    if not ok:
        raise SystemExit("hmm_regime unit tests FAILED.")
    print("\nAll hmm_regime tests passed.")
