"""
Weekend 1: Test 7 candidate pairs identified in Phase 0.
Uses cached 1-min data, resamples to 1h, runs proper rolling-OU backtest.
"""
import os, sys, math, numpy as np, pandas as pd
sys.path.insert(0, r"E:\MyDevelopment\GitHub\quant_strategy")
import logging; logging.basicConfig(level=logging.WARNING)

from modules.kalman_filter import KalmanFilter, estimate_noise_params
from modules.ou_estimator  import fit_ou_rolling, half_life_scale

# Settings (matching what gave +3-12% on original pairs)
ENTRY_Z = 3.0
EXIT_Z  = 0.2
MAX_HOLD = 24
COST_RT = 0.0040   # realistic Alpaca crypto cost

PAIRS = [
    ("BCH", "DOGE"),    # strongest signal
    ("ETH", "UNI"),
    ("LTC", "UNI"),
    ("BCH", "BTC"),
    ("AVAX", "LINK"),
    ("DOGE", "LTC"),
    ("BTC", "DOGE"),
]

data_dir = r"E:\MyDevelopment\GitHub\quant_strategy\data\phase0"

def load_1h(symbol):
    fpath = os.path.join(data_dir, f"{symbol}_USD_1min.csv")
    if not os.path.exists(fpath):
        return None
    df = pd.read_csv(fpath, parse_dates=[0], index_col=0)
    # Resample to 1h OHLCV
    out = pd.DataFrame()
    out["close"] = df["close"].resample("1h").last()
    return out.dropna()

def run_pair(sym_a, sym_b):
    """Returns (n_trades, win_rate, total_return_pct, max_dd_pct, days)."""
    df_a = load_1h(sym_a)
    df_b = load_1h(sym_b)
    if df_a is None or df_b is None:
        return None
    common = df_a.index.intersection(df_b.index)
    if len(common) < 500:
        return None
    df_a = df_a.loc[common]; df_b = df_b.loc[common]
    lp_a = np.log(df_a["close"].values)
    lp_b = np.log(df_b["close"].values)
    days = len(df_a) / 24.0

    WARMUP, OU_WIN, REFIT = 200, 200, 100
    if len(lp_a) < WARMUP + 200: return None
    q, r = estimate_noise_params(lp_a[:WARMUP], lp_b[:WARMUP])
    kf = KalmanFilter(q_scale=q, r_scale=r)
    spreads = []
    for i in range(WARMUP):
        st = kf.step(float(lp_a[i]), float(lp_b[i]))
        spreads.append(st.spread)
    ou_list = fit_ou_rolling(np.array(spreads), window=len(spreads)-1, step=len(spreads)-1)
    ou = next((p for p in ou_list if p.is_valid), None)
    if ou is None: return None
    
    in_trade=None; trades=[]; cap=10000; peak=cap; max_dd=0; last_refit=WARMUP
    for i in range(WARMUP, len(lp_a)):
        st = kf.step(float(lp_a[i]), float(lp_b[i]))
        spreads.append(st.spread)
        if i-last_refit >= REFIT:
            recent = np.array(spreads[-OU_WIN:])
            nl = fit_ou_rolling(recent, window=len(recent)-1, step=len(recent)-1)
            no = next((p for p in nl if p.is_valid), None)
            if no: ou = no
            last_refit = i
        z = (st.spread - ou.mu) / max(ou.sigma_eq, 1e-12)
        if in_trade:
            in_trade["held"] += 1; d = in_trade["dir"]; ex=False
            if d==1  and z > -EXIT_Z: ex=True
            elif d==-1 and z <  EXIT_Z: ex=True
            elif in_trade["held"] >= MAX_HOLD: ex=True
            if ex:
                pos=1000
                raw = d * (st.spread - in_trade["entry_spread"]) * pos
                net = max(min(raw,pos*0.5),-pos*0.5) - pos*COST_RT
                trades.append(net); cap += net
                if cap > peak: peak = cap
                dd = (peak-cap)/peak*100
                if dd > max_dd: max_dd = dd
                in_trade = None
        if in_trade is None and 0.3 <= ou.half_life <= 240 and half_life_scale(ou) > 0:
            if z < -ENTRY_Z: in_trade=dict(dir=1, entry_spread=st.spread, held=0)
            elif z > ENTRY_Z: in_trade=dict(dir=-1, entry_spread=st.spread, held=0)
    
    if not trades: return (0, 0, 0, 0, days)
    wins = sum(1 for t in trades if t > 0)
    pct = (cap - 10000) / 100
    return (len(trades), wins/len(trades)*100, pct, max_dd, days)

print("="*78)
print(" Weekend 1: Validate 7 new pairs (2024 data, realistic 0.40% cost)")
print("="*78)
print(f"{'Pair':>12} {'Trades':>7} {'Win%':>6} {'Return':>8} {'MaxDD':>7} {'%/yr':>7} {'Verdict':>10}")
print("-"*78)

results = []
for sym_a, sym_b in PAIRS:
    r = run_pair(sym_a, sym_b)
    if r is None:
        print(f"  {sym_a}/{sym_b}: data unavailable")
        continue
    n_trades, win_pct, pct, max_dd, days = r
    if n_trades == 0:
        print(f"{sym_a+'/'+sym_b:>12} {n_trades:>7d} {'-':>6} {'-':>8} {'-':>7} {'-':>7} {'no trades':>10}")
        continue
    ann = pct * (365/days)
    verdict = "PROCEED" if ann > 5 else ("WEAK" if ann > 0 else "REJECT")
    results.append({"pair": f"{sym_a}/{sym_b}", "trades": n_trades,
                    "win_pct": win_pct, "return_pct": pct, "max_dd": max_dd,
                    "ann_pct": ann, "verdict": verdict})
    print(f"{sym_a+'/'+sym_b:>12} {n_trades:>7d} {win_pct:>5.1f}% "
          f"{pct:>+6.2f}% {max_dd:>5.2f}% {ann:>+5.1f}% {verdict:>10}")

print()
print("="*78)
strong = [r for r in results if r["verdict"] == "PROCEED"]
print(f"  RESULT: {len(strong)} of {len(results)} pairs deliver > 5% annualized")
print("="*78)

if len(strong) >= 3:
    print("DECISION: Proceed to Weekend 2 (portfolio backtest)")
    print("Strong pairs:", [r["pair"] for r in strong])
else:
    print("DECISION: Stop here.")
    print("Less than 3 pairs deliver edge. Path B ceiling reached.")
