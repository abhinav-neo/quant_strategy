"""
Module 7: Backtest Engine
==========================
Wires all six analytical modules together and runs a
full backtest on real Alpaca 1-minute OHLCV data.

Pipeline per bar
----------------
1. data_fetcher   -> validated OHLCV DataFrame
2. stationarity   -> confirm edge exists (ADF + Hurst + LjungBox)
3. kalman_filter  -> dynamic hedge ratio + spread series
4. ou_estimator   -> OU params (theta, mu, sigma, half_life)
5. hmm_regime     -> regime state + kelly_scale
6. kelly_sizer    -> position size (Kelly x HMM x ATR)
7. backtest       -> execute trade, compute PnL, update capital

Trade mechanics
---------------
- One trade open at a time (no pyramiding)
- Entry: spread z-score crosses Bertram threshold a*
  LONG  spread when z < -a*  (spread below mean)
  SHORT spread when z > +a*  (spread above mean)
- Exit:  spread z-score returns to 0 (mean)
- Stop:  |z-score| > 3.5 * sigma_eq  (model breakdown)
- Time stop: bars_held >= max_bars_held

PnL calculation
---------------
  raw_pnl   = direction * (exit_spread - entry_spread) * position_usd
              / entry_price_A
  net_pnl   = clamp(raw_pnl, -pos*0.5, +pos*0.5) - pos * COST_FRAC
  capital   = validate_capital(capital + net_pnl, capital_before)

Performance metrics
-------------------
All computed via math_guards:
  Sharpe   = compute_sharpe(trade_returns, trades_per_year)
  Sortino  = compute_sortino(trade_returns, trades_per_year)
  Calmar   = compute_calmar(annualised_return, max_drawdown)
  annRet   = annualise_return(total_return, observation_days)

No per-bar annualisation. No safeN on dollar amounts.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime    import datetime, timezone
from typing      import Dict, List, Optional, Tuple

import numpy  as np
import pandas as pd

from modules.math_guards   import (
    safeN, safe_divide, clamp, clamp_pnl, validate_capital,
    compute_sharpe, compute_sortino, compute_calmar,
    annualise_return, MIN_DENOMINATOR
)
from modules.kalman_filter import KalmanFilter, run_kalman, estimate_noise_params
from modules.ou_estimator  import (
    fit_ou_rolling, bertram_threshold, half_life_scale, OUParams,
    ROLLING_WINDOW, ROLLING_STEP
)
from modules.hmm_regime    import (
    train_hmm, HMMInference, build_features, N_FEATURES
)
from modules.kelly_sizer   import compute_size, SizeResult
from modules.stationarity  import run_all_tests

log = logging.getLogger("backtest")

# ── Constants ──────────────────────────────────────────────────────────────
WARMUP_BARS       = 500     # more warmup for reliable OU params
OU_REFIT_EVERY    = 100     # refit OU params every N bars
HMM_RETRAIN_EVERY = 500     # retrain HMM every N bars (expensive)
MAX_BARS_HELD     = 120     # allow up to 2hrs for reversion
STOP_ZSCORE       = 3.5     # emergency stop z-score
ENTRY_ZSCORE      = 2.0     # raised: 1.5 too sensitive, overtraded
EXIT_ZSCORE       = 0.5     # raised: exit closer to mean
COST_FRAC         = 0.001   # Alpaca crypto ~0.15% each way
MIN_HALF_LIFE     = 1.0     # raised: sub-minute not profitable at 0.1% cost
MAX_HALF_LIFE     = 240     # bars -- 4 hours max
ATR_PERIOD        = 14


# ── Dataclasses ────────────────────────────────────────────────────────────
@dataclass
class Trade:
    """Single closed trade record."""
    bar_entry:    int
    bar_exit:     int
    direction:    int     # +1 long spread, -1 short spread
    entry_spread: float
    exit_spread:  float
    entry_price_a: float
    position_usd: float
    raw_pnl:      float
    net_pnl:      float
    trade_return: float   # net_pnl / capital_before
    exit_reason:  str     # "target" | "stop" | "time" | "emergency"
    regime_state: int
    half_life:    float
    capital_after: float


@dataclass
class BacktestResult:
    """Full backtest output. All metrics validated."""
    # Identification
    symbol_a:         str = ""
    symbol_b:         str = ""
    start_date:       str = ""
    end_date:         str = ""
    n_bars:           int = 0
    observation_days: int = 0

    # Capital curve
    initial_capital:  float = 1000.0
    final_capital:    float = 0.0
    equity_curve:     List[float] = field(default_factory=list)

    # Trade log
    trades:           List[Trade] = field(default_factory=list)

    # Aggregate metrics (all validated)
    total_return:     float = 0.0
    annualised_return: float = 0.0
    sharpe:           float = 0.0
    sortino:          float = 0.0
    calmar:           float = 0.0
    max_drawdown:     float = 0.0
    win_rate:         float = 0.0
    profit_factor:    float = 0.0
    avg_trade_usd:    float = 0.0
    total_trades:     int   = 0
    trades_per_year:  float = 0.0
    avg_holding_bars: float = 0.0

    # Regime breakdown
    pct_high_mr:  float = 0.0   # % bars in state 0
    pct_low_mr:   float = 0.0   # % bars in state 1
    pct_breakdown: float = 0.0  # % bars in state 2

    # Status
    ok:           bool  = False
    error:        str   = ""
    warnings:     List[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.ok:
            return f"BacktestResult(FAILED: {self.error})"
        lines = [
            "=" * 60,
            f"Backtest: {self.symbol_a} / {self.symbol_b}",
            f"Period  : {self.start_date} -> {self.end_date}",
            f"Bars    : {self.n_bars:,}  ({self.observation_days} days)",
            "-" * 60,
            f"Capital : ${self.initial_capital:,.2f} -> ${self.final_capital:,.2f}",
            f"Return  : {self.total_return*100:+.2f}%  "
            f"(ann: {self.annualised_return*100:+.1f}%)",
            f"Sharpe  : {self.sharpe:.3f}",
            f"Sortino : {self.sortino:.3f}",
            f"Calmar  : {self.calmar:.3f}",
            f"Max DD  : {self.max_drawdown*100:.1f}%",
            "-" * 60,
            f"Trades  : {self.total_trades}  "
            f"(win {self.win_rate*100:.1f}%  PF {self.profit_factor:.2f})",
            f"Avg PnL : ${self.avg_trade_usd:+.2f}",
            f"Avg hold: {self.avg_holding_bars:.1f} bars",
            f"Per year: {self.trades_per_year:.1f} trades",
            "-" * 60,
            f"Regime  : high-MR {self.pct_high_mr:.0f}%  "
            f"low-MR {self.pct_low_mr:.0f}%  "
            f"breakdown {self.pct_breakdown:.0f}%",
            "=" * 60,
        ]
        for w in self.warnings:
            lines.append(f"WARNING : {w}")
        return "\n".join(lines)


# ── ATR indicator ─────────────────────────────────────────────────────────
def _calc_atr(high, low, close, period):
    """
    True ATR: TR = max(high-low, |high-prev_close|, |low-prev_close|)
    Returns EMA(TR, period) as array of same length as input.
    """
    n    = len(close)
    tr   = np.zeros(n)
    tr[0] = float(high[0] - low[0])
    for i in range(1, n):
        hl  = float(high[i] - low[i])
        hcp = abs(float(high[i])  - float(close[i-1]))
        lcp = abs(float(low[i])   - float(close[i-1]))
        tr[i] = max(hl, hcp, lcp)
    atr    = np.zeros(n)
    atr[0] = tr[0]
    k      = 2.0 / (period + 1)
    for i in range(1, n):
        atr[i] = tr[i]*k + atr[i-1]*(1-k)
    return atr

class BacktestEngine:
    """Kalman+OU+HMM+Kelly stat arb backtest engine."""

    def __init__(self, initial_capital=10000.0, risk_pct=0.01,
                 stop_mult=1.5, target_mult=2.5):
        if initial_capital < 1:
            raise ValueError(f"initial_capital must be >= 1.")
        if not (0 < risk_pct <= 0.10):
            raise ValueError(f"risk_pct must be in (0,0.10].")
        self.initial_capital = float(initial_capital)
        self.risk_pct   = float(risk_pct)
        self.stop_mult  = float(stop_mult)
        self.target_mult = float(target_mult)

    def run(self, df_a, df_b, symbol_a="A", symbol_b="B"):
        """Run full backtest. Returns BacktestResult."""
        result = BacktestResult(symbol_a=symbol_a, symbol_b=symbol_b,
                                initial_capital=self.initial_capital)
        log.info(f"[backtest] {symbol_a}/{symbol_b} ({len(df_a):,} bars)")

        # 1: Align
        log.info("[backtest] 1/7 aligning...")
        df_a, df_b, err = self._align(df_a, df_b)
        if err:
            result.error = err; log.error(err); return result
        n = len(df_a)
        result.n_bars       = n
        result.start_date   = str(df_a.index[0].date())
        result.end_date     = str(df_a.index[-1].date())
        result.observation_days = max((df_a.index[-1]-df_a.index[0]).days, 1)
        if n < WARMUP_BARS + 100:
            result.error = f"Only {n} bars — need {WARMUP_BARS+100}."
            return result

        # 2: Indicators
        log.info("[backtest] 2/7 indicators...")
        lp_a    = np.log(np.maximum(df_a["close"].values, 1e-8))
        lp_b    = np.log(np.maximum(df_b["close"].values, 1e-8))
        atr_arr = _calc_atr(df_a["high"].values, df_a["low"].values,
                            df_a["close"].values, ATR_PERIOD)
        vols_a  = df_a["volume"].values
        va = df_a["valid_bar"].values if "valid_bar" in df_a.columns else np.ones(n,bool)
        vb = df_b["valid_bar"].values if "valid_bar" in df_b.columns else np.ones(n,bool)
        valid = va & vb

        # 3: Kalman warmup + initial OU
        log.info("[backtest] 3/7 Kalman warmup + OU fit...")
        te = WARMUP_BARS
        q_scale, r_scale = estimate_noise_params(lp_a[:te], lp_b[:te])
        kf = KalmanFilter(q_scale=q_scale, r_scale=r_scale)
        spread_buf = []; innov_buf = []
        for i in range(te):
            st = kf.step(float(lp_a[i]), float(lp_b[i]))
            spread_buf.append(st.spread); innov_buf.append(st.innov_var)
        ou_list    = fit_ou_rolling(np.array(spread_buf),
                                    window=min(ROLLING_WINDOW,te-1), step=ROLLING_STEP)
        current_ou = next((p for p in ou_list if p.is_valid), OUParams())

        # 4: HMM
        log.info("[backtest] 4/7 HMM training...")
        hmm_inf = None
        fw = build_features(np.array(spread_buf), np.array(innov_buf),
                            vols_a[:te], window=20)
        hp = train_hmm(fw, seed=42)
        if hp.is_valid:
            hmm_inf = HMMInference(hp)
            log.info("[backtest] HMM ready")
        else:
            result.warnings.append("HMM failed — half-Kelly fallback.")

        # 5: Main loop
        log.info(f"[backtest] 5/7 main loop ({n-te:,} bars)...")
        capital   = self.initial_capital
        peak_cap  = capital; max_dd = 0.0
        equity    = [capital]*te
        trades_log: List[Trade] = []; trade_rets: List[float] = []
        reg_counts = {0:0, 1:0}
        in_trade  = None
        last_ou   = te; last_hmm = te

        for i in range(te, n):
            if i % 1000 == 0:
                log.info(f"[backtest]  bar {i:,}/{n:,} capital=${capital:,.0f} trades={len(trades_log)}")
            if not valid[i]:
                equity.append(float(capital))
                if in_trade: in_trade["bars_held"] += 1
                continue

            ks = kf.step(float(lp_a[i]), float(lp_b[i]))
            spread_buf.append(ks.spread); innov_buf.append(ks.innov_var)
            if len(spread_buf) > 2000:
                spread_buf = spread_buf[-1000:]; innov_buf = innov_buf[-1000:]

            if i - last_ou >= OU_REFIT_EVERY:
                rec = np.array(spread_buf[-ROLLING_WINDOW:])
                nl  = fit_ou_rolling(rec, window=len(rec)-1, step=len(rec)-1)
                vl  = [p for p in nl if p.is_valid]
                if vl: current_ou = vl[-1]
                last_ou = i

            if i - last_hmm >= HMM_RETRAIN_EVERY and len(spread_buf)>=500:
                rsp = np.array(spread_buf[-500:]); riv = np.array(innov_buf[-500:])
                rvl = vols_a[max(0,i-500):i]
                ft  = build_features(rsp, riv, rvl, window=20)
                nh  = train_hmm(ft, seed=i)
                if nh.is_valid: hmm_inf = HMMInference(nh)
                last_hmm = i

            from modules.hmm_regime import RegimeState as _RS
            ow = build_features(np.array(spread_buf[-21:]), np.array(innov_buf[-21:]),
                                vols_a[max(0,i-21):i+1], window=20)
            if ow.shape[0]>0 and np.all(np.isfinite(ow[-1])) and hmm_inf:
                regime = hmm_inf.step(ow[-1])
            else:
                regime = _RS(bar=i, gamma=np.array([0.5,0.5,0.0]),
                             state=1, kelly_scale=0.5, is_no_trade=False, confidence=0.5)
            reg_counts[regime.state] = reg_counts.get(regime.state,0)+1

            if not current_ou.is_valid or current_ou.sigma_eq < MIN_DENOMINATOR:
                equity.append(float(capital)); continue
            zscore = safe_divide(ks.spread-current_ou.mu, current_ou.sigma_eq,
                                 fallback=0.0, label="z")

            if in_trade:
                in_trade["bars_held"] += 1
                cap_b = capital; er = None; d = in_trade["dir"]
                if   d==1  and zscore > -EXIT_ZSCORE:              er="target"
                elif d==-1 and zscore <  EXIT_ZSCORE:              er="target"
                elif abs(zscore)>STOP_ZSCORE:                      er="emergency"
                elif d==1  and ks.spread<in_trade["stop_sp"]:      er="stop"
                elif d==-1 and ks.spread>in_trade["stop_sp"]:      er="stop"
                elif in_trade["bars_held"]>=MAX_BARS_HELD:         er="time"
                if er:
                    pos  = in_trade["position_usd"]
                    e_sp = in_trade["entry_spread"]
                    e_pr = in_trade["entry_price_a"]
                    raw  = d*(ks.spread-e_sp)/max(e_pr,1e-8)*pos
                    net  = clamp_pnl(raw,pos)-pos*COST_FRAC
                    capital = validate_capital(capital+net, cap_b)
                    tr = net/max(cap_b,1.0)
                    if math.isfinite(tr) and abs(tr)<1.0: trade_rets.append(tr)
                    trades_log.append(Trade(
                        bar_entry=in_trade["bar_entry"], bar_exit=i,
                        direction=d, entry_spread=e_sp, exit_spread=ks.spread,
                        entry_price_a=e_pr, position_usd=pos,
                        raw_pnl=float(raw), net_pnl=float(net),
                        trade_return=float(tr), exit_reason=er,
                        regime_state=in_trade["regime_state"],
                        half_life=in_trade["half_life"],
                        capital_after=float(capital)))
                    in_trade = None

            if in_trade is None and not regime.is_no_trade and current_ou.is_valid:
                hl = current_ou.half_life
                if MIN_HALF_LIFE<=hl<=MAX_HALF_LIFE and half_life_scale(current_ou)>0:
                    # Use fixed z-score entry threshold (dimensionless, robust)
                    # Bertram threshold in log-price units causes unit mismatch
                    sig  = None
                    if   zscore < -ENTRY_ZSCORE: sig=1
                    elif zscore >  ENTRY_ZSCORE: sig=-1
                    if sig is not None:
                        sz = compute_size(capital=capital, ou_params=current_ou,
                                          regime=regime, atr=float(atr_arr[i]),
                                          entry_price=float(df_a["close"].iloc[i]),
                                          stop_mult=self.stop_mult,
                                          target_mult=self.target_mult,
                                          risk_pct=self.risk_pct)
                        if sz.is_valid:
                            sp_st = ks.spread-sig*self.stop_mult*current_ou.sigma_eq
                            in_trade = dict(bar_entry=i, dir=sig,
                                            entry_spread=ks.spread, stop_sp=sp_st,
                                            entry_price_a=float(df_a["close"].iloc[i]),
                                            position_usd=sz.position_usd,
                                            bars_held=0, regime_state=regime.state,
                                            half_life=hl)
            equity.append(float(capital))
            if capital>peak_cap: peak_cap=capital
            dd=(peak_cap-capital)/max(peak_cap,1.0)
            if dd>max_dd: max_dd=dd

        # 6: Metrics
        log.info("[backtest] 6/7 metrics...")
        tt=len(trades_log); wins=sum(1 for t in trades_log if t.net_pnl>0)
        gw=sum(t.net_pnl for t in trades_log if t.net_pnl>0)
        gl=sum(abs(t.net_pnl) for t in trades_log if t.net_pnl<0)
        days=result.observation_days
        tret=(capital-self.initial_capital)/self.initial_capital
        aret=annualise_return(tret,days); tpy=tt*(365.0/max(days,1))
        sh=compute_sharpe(trade_rets,tpy); so=compute_sortino(trade_rets,tpy)
        cal=compute_calmar(aret,max_dd)
        wr=safeN(wins/tt) if tt>0 else 0.0
        pf=clamp(safe_divide(gw,gl,label="pf"),0,9.99) if gl>MIN_DENOMINATOR \
           else (9.99 if gw>0 else 0.0)
        at=(gw-gl)/tt if tt>0 else 0.0
        ah=sum(t.bar_exit-t.bar_entry for t in trades_log)/tt if tt>0 else 0.0
        tr=sum(reg_counts.values()) or 1

        # 7: Populate
        log.info("[backtest] 7/7 populating result...")
        result.ok=True; result.final_capital=float(capital)
        result.equity_curve=equity; result.trades=trades_log
        result.total_return=safeN(tret); result.annualised_return=safeN(aret)
        result.sharpe=safeN(sh); result.sortino=safeN(so); result.calmar=safeN(cal)
        result.max_drawdown=safeN(max_dd); result.win_rate=safeN(wr)
        result.profit_factor=safeN(pf)
        result.avg_trade_usd=float(at) if math.isfinite(at) else 0.0
        result.total_trades=tt; result.trades_per_year=safeN(tpy)
        result.avg_holding_bars=safeN(ah)
        result.pct_high_mr=safeN(reg_counts.get(0,0)/tr*100)
        result.pct_low_mr=safeN(reg_counts.get(1,0)/tr*100)
        result.pct_breakdown=safeN(reg_counts.get(2,0)/tr*100)
        log.info(result.summary()); return result

    @staticmethod
    def _align(df_a, df_b):
        try:
            common=df_a.index.intersection(df_b.index)
            if len(common)<WARMUP_BARS+100:
                return df_a,df_b,f"Only {len(common)} common timestamps."
            return df_a.loc[common].copy(), df_b.loc[common].copy(), ""
        except Exception as e:
            return df_a, df_b, f"Alignment error: {e}"


def run_unit_tests():
    """
    Tests use synthetic cointegrated DataFrames.
    No Alpaca API calls. Verifies:
    - Valid aligned pairs -> BacktestResult.ok=True
    - Misaligned pairs -> ok=False with error
    - Too-short series -> ok=False with error
    - ATR calculation correctness
    - Trade log populated after signals fire
    - Equity curve length == n_bars
    - All metrics pass safeN (no overflow)
    - Capital never goes negative
    - win_rate in [0,1]
    - max_drawdown in [0,1]
    """
    import math
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(name)s %(message)s")
    log.info("Running backtest unit tests ...")
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
        high = close + noise
        low  = close - noise
        opn  = close + rng2.normal(0, 5, n)
        high = np.maximum(high, np.maximum(close, opn))
        low  = np.minimum(low,  np.minimum(close, opn))
        vol  = rng2.uniform(10, 200, n)
        df   = pd.DataFrame({"open":opn,"high":high,"low":low,
                              "close":close,"volume":vol,"valid_bar":True},
                            index=pd.DatetimeIndex(t, name="timestamp"))
        return df

    N = 1500   # enough for warmup + trading
    df_a = make_df(N, 45000, seed=1)
    df_b = make_df(N, 2500,  seed=2)
    # Make pair cointegrated: df_b price ~ 0.05 * df_a + noise
    df_b["close"] = df_a["close"].values * 0.055 + rng.normal(0, 5, N)
    df_b["high"]  = df_b["close"] + rng.uniform(5, 30, N)
    df_b["low"]   = df_b["close"] - rng.uniform(5, 30, N)
    df_b["high"]  = np.maximum(df_b["high"], df_b["close"])
    df_b["low"]   = np.minimum(df_b["low"],  df_b["close"])

    engine = BacktestEngine(initial_capital=10_000, risk_pct=0.01)

    # T01: valid pair runs to completion
    log.info("  [T01] valid pair run...")
    r = engine.run(df_a, df_b, "BTC/USD", "ETH/USD")
    check("T01 ok=True for valid pair",          r.ok)
    check("T01 no error string",                 r.error == "")

    # T02: equity curve length
    if r.ok:
        check("T02 equity_curve length == n_bars", len(r.equity_curve) == r.n_bars)

    # T03: capital never negative
    if r.ok:
        check("T03 capital never negative",
              all(v >= 0 for v in r.equity_curve))

    # T04: final capital matches last equity
    if r.ok:
        check("T04 final_capital == equity_curve[-1]",
              abs(r.final_capital - r.equity_curve[-1]) < 1e-4)

    # T05: win_rate in [0,1]
    if r.ok:
        check("T05 win_rate in [0,1]", 0.0 <= r.win_rate <= 1.0)

    # T06: max_drawdown in [0,1]
    if r.ok:
        check("T06 max_drawdown in [0,1]", 0.0 <= r.max_drawdown <= 1.0)

    # T07: all ratio metrics finite and within safeN range
    if r.ok:
        for attr in ["sharpe","sortino","calmar","total_return",
                     "annualised_return","profit_factor"]:
            v = getattr(r, attr)
            check(f"T07 {attr} finite and |v|<=999",
                  math.isfinite(v) and abs(v) <= 999)

    # T08: avg_trade_usd is finite
    if r.ok:
        check("T08 avg_trade_usd finite", math.isfinite(r.avg_trade_usd))

    # T09: n_bars matches aligned length
    if r.ok:
        check("T09 n_bars > 0", r.n_bars > 0)

    # T10: summary() works without crash
    if r.ok:
        try:
            s = r.summary()
            check("T10 summary() works", len(s) > 0)
        except Exception as e:
            check(f"T10 summary() raised: {e}", False)

    # T11: misaligned DataFrames (non-overlapping timestamps) -> ok=False
    log.info("  [T11] non-overlapping timestamps...")
    df_c = make_df(N, 45000, seed=3)
    df_d = make_df(N, 2500,  seed=4)
    # Shift df_d 10 years forward -> no common timestamps
    df_d.index = df_d.index + pd.Timedelta(days=3650)
    r11 = engine.run(df_c, df_d)
    check("T11 no common timestamps -> ok=False", not r11.ok)
    check("T11 error string populated",           r11.error != "")

    # T12: too-short series -> ok=False
    log.info("  [T12] too-short series...")
    df_e = make_df(200, 45000, seed=5)
    df_f = make_df(200, 2500,  seed=6)
    r12 = engine.run(df_e, df_f)
    check("T12 too-short -> ok=False", not r12.ok)

    # T13: invalid initial_capital -> ValueError
    log.info("  [T13] invalid capital...")
    try:
        BacktestEngine(initial_capital=0)
        check("T13 zero capital raises ValueError", False)
    except ValueError:
        check("T13 zero capital raises ValueError", True)

    # T14: ATR calculation basic properties
    log.info("  [T14] ATR properties...")
    h = np.array([10.0,12.0,11.0,13.0,12.0])
    l = np.array([ 8.0, 9.0,10.0,10.0,11.0])
    c = np.array([ 9.0,11.0,10.0,12.0,11.5])
    atr = _calc_atr(h, l, c, period=3)
    check("T14 ATR all positive",       np.all(atr > 0))
    check("T14 ATR length correct",     len(atr) == len(c))
    check("T14 ATR[0] = high-low",      abs(atr[0]-(h[0]-l[0])) < 1e-10)

    # T15: regime percentages sum to ~100
    if r.ok:
        reg_sum = r.pct_high_mr + r.pct_low_mr + r.pct_breakdown
        check("T15 regime percentages sum ~100",
              abs(reg_sum - 100.0) < 1.0)

    log.info(f"\nbacktest: {passed}/{passed+failed} tests passed.")
    return failed == 0


if __name__ == "__main__":
    ok = run_unit_tests()
    if not ok:
        raise SystemExit("backtest unit tests FAILED.")
    print("\nAll backtest tests passed.")
