"""
Module 6: Kelly Position Sizer
================================
Combines three independent sizing constraints into one final
position size. All three must agree - the most conservative wins.

Constraint 1: Kelly fraction from OU geometry
    p_win from OU first-passage time distribution
    b     = reward / risk ratio from Bertram threshold
    f*    = (b*p - (1-p)) / b  capped at MAX_KELLY

Constraint 2: HMM state weighting
    f_hmm = f* * kelly_scale(state)
    state 0 -> 1.0x, state 1 -> 0.5x, state 2 -> 0.0 (no trade)

Constraint 3: ATR risk target
    pos_atr = (capital * risk_pct) / (stop_mult * ATR)
    capped at capital * MAX_POS_FRAC

Final:
    position = min(f_hmm * capital, pos_atr)
    position = clamp(position, 0, capital * MAX_POS_FRAC)

All division via safe_divide. All output via safeN.
"""

import logging
import math
from dataclasses import dataclass

from modules.math_guards import (
    safeN, safe_divide, safe_exp, safe_log,
    clamp, MIN_DENOMINATOR
)
from modules.ou_estimator import OUParams
from modules.hmm_regime   import RegimeState

log = logging.getLogger("kelly_sizer")

MAX_KELLY        = 0.25
MAX_POS_FRAC     = 0.30
MIN_POS_FRAC     = 0.005
DEFAULT_RISK_PCT = 0.01
COST_FRAC        = 0.002


@dataclass
class SizeResult:
    """Output of compute_size(). Check is_valid before placing any order."""
    position_usd:    float = 0.0
    kelly_fraction:  float = 0.0
    kelly_hmm:       float = 0.0
    pos_kelly_usd:   float = 0.0
    pos_atr_usd:     float = 0.0
    win_prob:        float = 0.0
    reward_risk:     float = 0.0
    hmm_scale:       float = 0.0
    atr:             float = 0.0
    stop_dist:       float = 0.0
    risk_dollars:    float = 0.0
    is_valid:        bool  = False
    no_trade_reason: str   = ""

    def summary(self) -> str:
        if not self.is_valid:
            return f"SizeResult(NO TRADE: {self.no_trade_reason})"
        return (
            f"SizeResult(pos=${self.position_usd:.2f} "
            f"kelly={self.kelly_fraction:.4f} "
            f"hmm_scale={self.hmm_scale:.2f} "
            f"p_win={self.win_prob:.4f} "
            f"b={self.reward_risk:.4f} "
            f"risk=${self.risk_dollars:.2f})"
        )


def _ou_win_prob(theta, sigma, a, cost=COST_FRAC):
    """
    Win probability from OU first-passage time.

    Entry at spread = -a (below mean).
    Win:  spread returns to mu (0 in z-score space).
    Loss: spread hits -2*a  (stop at 2x entry distance).

    Using scale function of OU process (Karatzas & Shreve 1991):
        s(x) = integral_0^x exp(theta*u^2/sigma^2) du

    P(hit 0 before -2a | start=-a)
        = [s(-a) - s(-2a)] / [s(0) - s(-2a)]

    We approximate s via numerical integration over a fine grid.
    All operations guarded against overflow.

    Parameters
    ----------
    theta : OU mean-reversion speed (must be > 0)
    sigma : OU diffusion (must be > 0)
    a     : entry threshold in spread units (must be > 0)
    cost  : round-trip cost as fraction of spread

    Returns
    -------
    p_win in [0, 1]
    """
    if theta <= MIN_DENOMINATOR or sigma <= MIN_DENOMINATOR or a <= 0:
        return 0.5   # neutral fallback

    # Scale the problem: work in units of sigma_eq = sigma/sqrt(2*theta)
    sigma_eq = sigma / math.sqrt(max(2.0 * theta, MIN_DENOMINATOR))
    if sigma_eq < MIN_DENOMINATOR:
        return 0.5

    # Barriers in normalised units
    x_start = -a / sigma_eq
    x_win   =  0.0
    x_loss  = -2.0 * a / sigma_eq

    # Scale function integrand: exp(u^2) for normalised OU
    # s(x) = integral_0^x exp(u^2) du  (Dawson-like integral)
    # Numerical integration via Simpson's rule on 200 points
    def scale_integral(lo, hi, n=200):
        if abs(hi - lo) < 1e-12:
            return 0.0
        u   = [lo + (hi - lo) * i / n for i in range(n + 1)]
        # exp(u^2) clamped to prevent overflow - u values near 0 are safe
        vals = [math.exp(clamp(ui * ui, 0, 500)) for ui in u]
        h    = (hi - lo) / n
        # Simpson's rule
        total = vals[0] + vals[-1]
        for i in range(1, n):
            total += (4 if i % 2 == 1 else 2) * vals[i]
        return total * h / 3.0

    s_start_to_win  = scale_integral(x_start, x_win)
    s_loss_to_win   = scale_integral(x_loss,  x_win)

    if abs(s_loss_to_win) < MIN_DENOMINATOR:
        return 0.5

    p_win = safe_divide(
        s_start_to_win,
        s_loss_to_win,
        fallback=0.5,
        label="ou_win_prob",
    )
    return clamp(safeN(p_win, fallback=0.5, label="p_win"), 0.0, 1.0)


def _kelly_fraction(p_win, reward_risk):
    """
    Kelly fraction: f* = (b*p - (1-p)) / b
    where b = reward_risk ratio.
    Capped at MAX_KELLY. Returns 0 if edge is negative.
    """
    if reward_risk < MIN_DENOMINATOR:
        return 0.0
    raw = safe_divide(
        reward_risk * p_win - (1.0 - p_win),
        reward_risk,
        label="kelly_f",
    )
    f = safeN(raw, fallback=0.0, label="kelly_f_safe")
    return clamp(f, 0.0, MAX_KELLY)


def compute_size(
    capital:      float,
    ou_params:    OUParams,
    regime:       RegimeState,
    atr:          float,
    entry_price:  float,
    stop_mult:    float = 1.5,
    target_mult:  float = 2.5,
    risk_pct:     float = DEFAULT_RISK_PCT,
) -> SizeResult:
    """
    Compute final position size combining Kelly + HMM + ATR.

    Parameters
    ----------
    capital      : current portfolio value in dollars
    ou_params    : fitted OU parameters (from ou_estimator)
    regime       : current HMM regime state (from hmm_regime)
    atr          : current ATR value in price units
    entry_price  : current bar close price
    stop_mult    : ATR multiplier for stop loss
    target_mult  : ATR multiplier for take profit
    risk_pct     : fraction of capital to risk per trade

    Returns
    -------
    SizeResult - always returned, never raises.
    Check .is_valid before using .position_usd.
    """
    result = SizeResult()

    # ── Guard 1: HMM no-trade state ──────────────────────────────────────
    if regime.is_no_trade:
        result.no_trade_reason = f"HMM state={regime.state} (breakdown)"
        log.debug(f"compute_size: NO TRADE - {result.no_trade_reason}")
        return result

    # ── Guard 2: OU params must be valid ─────────────────────────────────
    if not ou_params.is_valid:
        result.no_trade_reason = "OU params invalid"
        log.debug(f"compute_size: NO TRADE - {result.no_trade_reason}")
        return result

    # ── Guard 3: capital must be positive finite ──────────────────────────
    if not math.isfinite(capital) or capital < 1.0:
        result.no_trade_reason = f"capital invalid: {capital}"
        log.warning(f"compute_size: NO TRADE - {result.no_trade_reason}")
        return result

    # ── Guard 4: ATR must be positive finite ──────────────────────────────
    if not math.isfinite(atr) or atr <= 0:
        result.no_trade_reason = f"ATR invalid: {atr}"
        log.warning(f"compute_size: NO TRADE - {result.no_trade_reason}")
        return result

    # ── Guard 5: entry price must be positive ────────────────────────────
    if not math.isfinite(entry_price) or entry_price <= 0:
        result.no_trade_reason = f"entry_price invalid: {entry_price}"
        log.warning(f"compute_size: NO TRADE - {result.no_trade_reason}")
        return result

    # ── Step 1: ATR-based stop distance and reward/risk ──────────────────
    stop_dist  = stop_mult  * atr      # price distance to stop
    tgt_dist   = target_mult * atr     # price distance to target

    if stop_dist < MIN_DENOMINATOR:
        result.no_trade_reason = "stop_dist near zero"
        return result

    reward_risk = safe_divide(tgt_dist, stop_dist, label="rr_ratio")
    reward_risk = safeN(reward_risk, fallback=0.0, label="rr_safe")
    if reward_risk < 0.1:
        result.no_trade_reason = f"reward/risk too low: {reward_risk:.4f}"
        return result

    # ── Step 2: Win probability from OU geometry ─────────────────────────
    # Use Bertram threshold (a*) as entry distance in spread units
    # Map to price units: stop_dist_spread ~ stop_dist / entry_price
    a_spread = safe_divide(stop_dist, entry_price, label="a_spread")
    a_spread = safeN(a_spread, fallback=0.01, label="a_spread_safe")

    p_win = _ou_win_prob(
        theta = ou_params.theta,
        sigma = ou_params.sigma,
        a     = a_spread,
        cost  = COST_FRAC,
    )

    # ── Step 3: Kelly fraction ────────────────────────────────────────────
    f_kelly = _kelly_fraction(p_win, reward_risk)

    # ── Step 4: HMM state weighting ──────────────────────────────────────
    hmm_scale  = safeN(regime.kelly_scale, fallback=0.0, label="hmm_scale")
    f_hmm      = f_kelly * hmm_scale
    f_hmm      = clamp(f_hmm, 0.0, MAX_KELLY)

    # ── Step 5: Kelly-sized position ─────────────────────────────────────
    pos_kelly = f_hmm * capital
    pos_kelly = clamp(pos_kelly if math.isfinite(pos_kelly) else 0.0, 0.0, capital * MAX_POS_FRAC)

    # ── Step 6: ATR risk-target position ─────────────────────────────────
    # Units = risk_dollars / stop_dist_price
    # Position_value = units * entry_price
    risk_dollars = capital * risk_pct
    units        = safe_divide(risk_dollars, stop_dist, label="units")
    pos_atr      = units * entry_price
    pos_atr      = clamp(
        pos_atr if math.isfinite(pos_atr) else 0.0,
        0.0,
        capital * MAX_POS_FRAC,
    )

    # ── Step 7: Final position = most conservative of Kelly and ATR ──────
    position = min(pos_kelly, pos_atr)

    # ── Step 8: Minimum size check ────────────────────────────────────────
    min_pos = capital * MIN_POS_FRAC
    if position < min_pos:
        result.no_trade_reason = (
            f"position ${position:.2f} < minimum ${min_pos:.2f}"
        )
        return result

    # Step 9: Populate result
    # IMPORTANT: dollar amounts do NOT use safeN (which caps at 999)
    # safeN is for ratios only. Dollar values use finite check + clamp.
    def _vd(v):
        return float(v) if math.isfinite(v) else 0.0
    result.position_usd   = _vd(position)
    result.pos_kelly_usd  = _vd(pos_kelly)
    result.pos_atr_usd    = _vd(pos_atr)
    result.risk_dollars   = _vd(risk_dollars)
    result.atr            = _vd(atr)
    result.stop_dist      = _vd(stop_dist)
    result.kelly_fraction = safeN(f_kelly,    fallback=0.0)
    result.kelly_hmm      = safeN(f_hmm,      fallback=0.0)
    result.win_prob       = safeN(p_win,       fallback=0.0)
    result.reward_risk    = safeN(reward_risk, fallback=0.0)
    result.hmm_scale      = safeN(hmm_scale,   fallback=0.0)
    result.is_valid       = True

    log.debug(f"compute_size: {result.summary()}")
    return result


def run_unit_tests():
    import math
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s"
    )
    log.info("Running kelly_sizer unit tests ...")
    passed = failed = 0

    def check(name, condition):
        nonlocal passed, failed
        if condition: log.info(f"  PASS  {name}"); passed += 1
        else:         log.error(f"  FAIL  {name}"); failed += 1

    import numpy as np

    # ── Helpers to build valid inputs ────────────────────────────────────
    def valid_ou():
        p = OUParams(
            theta=0.5, mu=0.0, sigma=0.05,
            half_life=math.log(2)/0.5,
            a_hat=math.exp(-0.5),
            sigma_eq=0.05/math.sqrt(2*0.5),
            n_obs=500,
        )
        p.is_valid = True
        return p

    def valid_regime(state=0):
        scales = [1.0, 0.5, 0.0]
        return RegimeState(
            bar=1,
            gamma=np.eye(3)[state],
            state=state,
            kelly_scale=scales[state],
            is_no_trade=(state==2),
            confidence=1.0,
        )

    ou   = valid_ou()
    reg0 = valid_regime(0)   # high-MR, full Kelly
    reg1 = valid_regime(1)   # low-MR,  half Kelly
    reg2 = valid_regime(2)   # breakdown, no trade
    CAP  = 10_000.0
    ATR  = 50.0             # $50 ATR on BTC
    PRICE = 45_000.0

    # ── T01: valid inputs produce is_valid=True ───────────────────────────
    log.info("  [T01] valid inputs ...")
    r = compute_size(CAP, ou, reg0, ATR, PRICE)
    check("T01 valid inputs -> is_valid=True",        r.is_valid)
    check("T01 position_usd > 0",                     r.position_usd > 0)
    check("T01 position_usd <= capital*MAX_POS_FRAC", r.position_usd <= CAP*MAX_POS_FRAC)

    # ── T02: HMM state=2 -> no trade ─────────────────────────────────────
    log.info("  [T02] HMM breakdown state ...")
    r2 = compute_size(CAP, ou, reg2, ATR, PRICE)
    check("T02 HMM state=2 -> is_valid=False", not r2.is_valid)
    check("T02 position_usd = 0",              r2.position_usd == 0.0)

    # ── T03: HMM state=1 -> half Kelly vs state=0 ────────────────────────
    log.info("  [T03] HMM state weighting ...")
    r0 = compute_size(CAP, ou, reg0, ATR, PRICE)
    r1 = compute_size(CAP, ou, reg1, ATR, PRICE)
    if r0.is_valid and r1.is_valid:
        check("T03 state=1 kelly_hmm <= state=0 kelly_hmm",
              r1.kelly_hmm <= r0.kelly_hmm + 1e-10)
        check("T03 state=1 position <= state=0 position",
              r1.position_usd <= r0.position_usd + 1e-6)

    # ── T04: invalid OU params -> no trade ───────────────────────────────
    log.info("  [T04] invalid OU params ...")
    bad_ou = OUParams()   # is_valid=False by default
    r4 = compute_size(CAP, bad_ou, reg0, ATR, PRICE)
    check("T04 invalid OU -> is_valid=False", not r4.is_valid)

    # ── T05: ATR=0 -> no trade ────────────────────────────────────────────
    log.info("  [T05] ATR=0 ...")
    r5 = compute_size(CAP, ou, reg0, 0.0, PRICE)
    check("T05 ATR=0 -> is_valid=False", not r5.is_valid)

    # ── T06: negative ATR -> no trade ─────────────────────────────────────
    log.info("  [T06] negative ATR ...")
    r6 = compute_size(CAP, ou, reg0, -10.0, PRICE)
    check("T06 negative ATR -> is_valid=False", not r6.is_valid)

    # ── T07: capital < $1 -> no trade ─────────────────────────────────────
    log.info("  [T07] zero capital ...")
    r7 = compute_size(0.0, ou, reg0, ATR, PRICE)
    check("T07 zero capital -> is_valid=False", not r7.is_valid)

    # ── T08: NaN capital -> no trade ──────────────────────────────────────
    log.info("  [T08] NaN capital ...")
    r8 = compute_size(float("nan"), ou, reg0, ATR, PRICE)
    check("T08 NaN capital -> is_valid=False", not r8.is_valid)

    # ── T09: NaN ATR -> no trade ──────────────────────────────────────────
    log.info("  [T09] NaN ATR ...")
    r9 = compute_size(CAP, ou, reg0, float("nan"), PRICE)
    check("T09 NaN ATR -> is_valid=False", not r9.is_valid)

    # ── T10: position never exceeds MAX_POS_FRAC * capital ───────────────
    log.info("  [T10] position cap ...")
    # Use very high risk_pct to try to exceed cap
    r10 = compute_size(CAP, ou, reg0, ATR, PRICE, risk_pct=0.99)
    check("T10 position <= MAX_POS_FRAC * capital",
          r10.position_usd <= CAP * MAX_POS_FRAC + 1e-6)

    # ── T11: win probability in [0, 1] ────────────────────────────────────
    log.info("  [T11] win probability bounds ...")
    check("T11 win_prob in [0,1]",
          0.0 <= r.win_prob <= 1.0)

    # ── T12: reward/risk > 0 for valid result ─────────────────────────────
    log.info("  [T12] reward/risk ...")
    check("T12 reward_risk > 0 for valid result", r.reward_risk > 0)

    # ── T13: kelly_fraction <= MAX_KELLY ─────────────────────────────────
    log.info("  [T13] Kelly cap ...")
    check("T13 kelly_fraction <= MAX_KELLY",
          r.kelly_fraction <= MAX_KELLY + 1e-10)

    # ── T14: risk_dollars = capital * risk_pct ───────────────────────────
    log.info("  [T14] risk dollars ...")
    expected_risk = CAP * DEFAULT_RISK_PCT
    check("T14 risk_dollars = capital * risk_pct",
          abs(r.risk_dollars - expected_risk) < 1e-6)

    # ── T15: _ou_win_prob bounds ──────────────────────────────────────────
    log.info("  [T15] win prob function ...")
    for th, sg, a in [(0.5,0.05,0.01),(1.0,0.1,0.02),(0.1,0.02,0.005)]:
        p = _ou_win_prob(th, sg, a)
        check(f"T15 p_win({th},{sg},{a}) in [0,1]", 0.0<=p<=1.0)

    # ── T16: _ou_win_prob invalid inputs return 0.5 ───────────────────────
    log.info("  [T16] win prob invalid inputs ...")
    check("T16 theta=0 -> p_win=0.5",  _ou_win_prob(0.0, 0.05, 0.01)==0.5)
    check("T16 sigma=0 -> p_win=0.5",  _ou_win_prob(0.5, 0.0,  0.01)==0.5)
    check("T16 a=0     -> p_win=0.5",  _ou_win_prob(0.5, 0.05, 0.0 )==0.5)

    # ── T17: higher reward/risk -> higher Kelly fraction ──────────────────
    log.info("  [T17] Kelly increases with reward/risk ...")
    f_lo = _kelly_fraction(0.6, 1.0)
    f_hi = _kelly_fraction(0.6, 3.0)
    check("T17 higher b -> higher Kelly fraction", f_hi >= f_lo)

    # ── T18: negative edge -> Kelly = 0 ──────────────────────────────────
    log.info("  [T18] negative edge ...")
    f_neg = _kelly_fraction(0.3, 1.0)   # p=0.3, b=1 -> edge < 0
    check("T18 negative edge -> Kelly = 0", f_neg == 0.0)

    # ── T19: zero entry price -> no trade ─────────────────────────────────
    log.info("  [T19] zero entry price ...")
    r19 = compute_size(CAP, ou, reg0, ATR, 0.0)
    check("T19 zero entry price -> is_valid=False", not r19.is_valid)

    # ── T20: summary() works for both valid and invalid ───────────────────
    log.info("  [T20] summary() ...")
    try:
        sv = r.summary();  si = r2.summary()
        check("T20 summary() works for valid",   len(sv)>0)
        check("T20 summary() works for invalid", "NO TRADE" in si)
    except Exception as e:
        check(f"T20 summary() raised: {e}", False)

    log.info(f"\nkelly_sizer: {passed}/{passed+failed} tests passed.")
    return failed == 0


if __name__ == "__main__":
    ok = run_unit_tests()
    if not ok:
        raise SystemExit("kelly_sizer unit tests FAILED.")
    print("\nAll kelly_sizer tests passed.")
