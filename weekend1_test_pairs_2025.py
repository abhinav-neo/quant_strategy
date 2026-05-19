r"""
weekend1_test_pairs_2025.py
============================
Re-run the same backtest on 2025 data to validate cross-year consistency.

Decision gate (per PARALLEL_TRACK.md):
  - 3+ of 7 pairs deliver > 5%/yr in BOTH 2024 AND 2025  -> proceed to Weekend 2
  - Otherwise -> stop, accept current ceiling
"""
import os, sys, math
import numpy as np
import pandas as pd
sys.path.insert(0, r"E:\MyDevelopment\GitHub\quant_strategy")
import logging; logging.basicConfig(level=logging.WARNING)

from modules.kalman_filter import KalmanFilter, estimate_noise_params
from modules.ou_estimator  import fit_ou_rolling, half_life_scale

ENTRY_Z = 3.0
EXIT_Z  = 0.2
MAX_HOLD = 24
COST_RT = 0.0040

PAIRS = [
    ("BCH", "DOGE"),
    ("ETH", "UNI"),
    ("LTC", "UNI"),
    ("BCH", "BTC"),
    ("AVAX", "LINK"),
    ("DOGE", "LTC"),
    ("BTC", "DOGE"),
]

DATA_2025 = r"E:\MyDevelopment\GitHub\quant_strategy\data\2025"

def load_1h(symbol):
    fpath = os.path.join(DATA_2025, f"{symbol}_USD_1min.csv")
    if not os.path.exists(fpath):
        return None
    df = pd.read_csv(fpath, parse_dates=[0], index_col=0)
    out = pd.DataFrame()
    out["close"] = df["close"].resample("1h").last()
    return out.dropna()

def run_pair(sym_a, sym_b):
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

# Compare to Weekend 1 results
RESULTS_2024 = {
    "BCH/DOGE":  {"trades": 66,  "win_pct": 100.0, "ann": 27.6},
    "ETH/UNI":   {"trades": 56,  "win_pct": 100.0, "ann": 12.8},
    "LTC/UNI":   {"trades": 55,  "win_pct": 98.0,  "ann": 16.9},
    "BCH/BTC":   {"trades": 102, "win_pct": 100.0, "ann": 13.4},
    "AVAX/LINK": {"trades": 63,  "win_pct": 100.0, "ann": 18.8},
    "DOGE/LTC":  {"trades": 92,  "win_pct": 100.0, "ann": 26.8},
    "BTC/DOGE":  {"trades": 68,  "win_pct": 100.0, "ann": 11.3},
}

print("="*90)
print(" Weekend 1: 2025 cross-year validation (realistic 0.40% cost)")
print("="*90)
print(f"{'Pair':>12} {'2024 ann%':>10} {'2025 trades':>12} {'2025 win%':>10} "
      f"{'2025 ret%':>10} {'2025 DD%':>9} {'2025 ann%':>10} {'consistent':>11}")
print("-"*90)

both_strong = []
for sym_a, sym_b in PAIRS:
    label = f"{sym_a}/{sym_b}"
    r25 = run_pair(sym_a, sym_b)
    if r25 is None:
        print(f"  {label}: data unavailable in 2025")
        continue
    n25, win25, pct25, dd25, days25 = r25
    if n25 == 0:
        print(f"{label:>12} {RESULTS_2024[label]['ann']:>+9.1f}% "
              f"{0:>12d} {'-':>10} {'-':>10} {'-':>9} {'no trades':>10} {'STOP':>11}")
        continue
    ann25 = pct25 * (365/days25)
    ann24 = RESULTS_2024[label]['ann']
    consistent = "YES" if (ann24 > 5 and ann25 > 5) else "  weak"
    if ann24 > 5 and ann25 > 5:
        both_strong.append(label)
    print(f"{label:>12} {ann24:>+9.1f}% {n25:>12d} {win25:>9.1f}% "
          f"{pct25:>+9.2f}% {dd25:>7.2f}% {ann25:>+9.1f}% {consistent:>11}")

print()
print("="*90)
print(f"RESULT: {len(both_strong)} of {len(PAIRS)} pairs deliver >5%/yr in BOTH 2024 AND 2025")
print("="*90)
if len(both_strong) >= 3:
    print("DECISION: PROCEED to Weekend 2 (portfolio backtest)")
    print(f"Strong pairs: {both_strong}")
else:
    print("DECISION: STOP. Less than 3 pairs are consistent.")
    print("Path B real ceiling reached. Switch to Path C or accept current.")
