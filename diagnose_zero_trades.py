#!/usr/bin/env python3
"""
diagnose_zero_trades.py
========================
Runs each pipeline stage independently on real Alpaca data
and prints exactly where signal generation breaks down.

Run:
    set ALPACA_KEY=your_key
    set ALPACA_SECRET=your_secret
    set PYTHONPATH=E:\MyDevelopment\GitHub\quant_strategy
    python diagnose_zero_trades.py
"""
import os, sys, math, logging
import numpy as np
from datetime import datetime, timezone

logging.basicConfig(level=logging.WARNING,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s")
log = logging.getLogger("diag")

ALPACA_KEY    = os.environ.get("ALPACA_KEY",    "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "")
SYMBOL_A = "BTC/USD"
SYMBOL_B = "ETH/USD"
START    = datetime(2024, 6, 1, tzinfo=timezone.utc)
END      = datetime(2024, 7, 1, tzinfo=timezone.utc)  # just 1 month
WARMUP   = 300

def sep(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)

def ok(msg):  print(f"  [OK]   {msg}")
def warn(msg):print(f"  [WARN] {msg}")
def fail(msg):print(f"  [FAIL] {msg}")

sep("STEP 1: Fetch data")
if not ALPACA_KEY:
    fail("ALPACA_KEY not set. Run: set ALPACA_KEY=your_key")
    sys.exit(1)

from modules.data_fetcher import AlpacaDataFetcher
fetcher = AlpacaDataFetcher(ALPACA_KEY, ALPACA_SECRET)

r_a = fetcher.fetch_bars(SYMBOL_A, START, END)
r_b = fetcher.fetch_bars(SYMBOL_B, START, END)

if not r_a.ok: fail(f"{SYMBOL_A}: {r_a.error_message}"); sys.exit(1)
if not r_b.ok: fail(f"{SYMBOL_B}: {r_b.error_message}"); sys.exit(1)

df_a = r_a.bars[r_a.bars["valid_bar"]].copy()
df_b = r_b.bars[r_b.bars["valid_bar"]].copy()

# Save cache
import os as _os
cache_dir = r"E:\MyDevelopment\GitHub\quant_strategy\data"
_os.makedirs(cache_dir, exist_ok=True)
for sym, df in [(SYMBOL_A, df_a), (SYMBOL_B, df_b)]:
    fpath = _os.join(cache_dir, sym.replace("/","_")+"_1min.csv")
    df.to_csv(fpath)
    print(f"  Saved cache: {fpath}")

# Align
common = df_a.index.intersection(df_b.index)
df_a = df_a.loc[common]; df_b = df_b.loc[common]
N = len(df_a)
ok(f"{SYMBOL_A}: {N:,} aligned bars  "
   f"close=[{df_a['close'].min():.0f},{df_a['close'].max():.0f}]")
ok(f"{SYMBOL_B}: {N:,} aligned bars  "
   f"close=[{df_b['close'].min():.2f},{df_b['close'].max():.2f}]")

if N < WARMUP + 200:
    fail(f"Too few bars: {N}. Need {WARMUP+200}+."); sys.exit(1)

sep("STEP 2: Kalman filter warmup")
import numpy as np
from modules.kalman_filter import KalmanFilter, estimate_noise_params

lp_a = np.log(df_a["close"].values)
lp_b = np.log(df_b["close"].values)

q, r = estimate_noise_params(lp_a[:WARMUP], lp_b[:WARMUP])
ok(f"Noise params: Q={q:.2e}  R={r:.2e}")

kf = KalmanFilter(q_scale=q, r_scale=r)
spreads = []; innov_v = []
for i in range(WARMUP):
    st = kf.step(float(lp_a[i]), float(lp_b[i]))
    spreads.append(st.spread); innov_v.append(st.innov_var)

sp_arr = np.array(spreads)
ok(f"Spread after warmup: mean={np.mean(sp_arr):.8f}  "
   f"std={np.std(sp_arr):.8f}  "
   f"range=[{sp_arr.min():.8f},{sp_arr.max():.8f}]")

if np.std(sp_arr) < 1e-8:
    fail("Spread std near zero - pairs may not be cointegrated or Q too small")

sep("STEP 3: OU parameter estimation")
from modules.ou_estimator import (
    fit_ou_rolling, bertram_threshold, half_life_scale,
    ROLLING_WINDOW, MIN_HALF_LIFE, MAX_HALF_LIFE
)

ou_list = fit_ou_rolling(sp_arr, window=min(ROLLING_WINDOW, WARMUP-1),
                         step=ROLLING_STEP := min(100, WARMUP//3))
valid_ou = [p for p in ou_list if p.is_valid]
ok(f"OU fits attempted: {len(ou_list)}  valid: {len(valid_ou)}")

if not valid_ou:
    fail("NO valid OU parameters. Root cause of zero trades.")
    fail("Possible reasons:")
    fail("  a) Spread is non-stationary (random walk, no mean reversion)")
    fail("  b) a_hat outside (0,1) - series trending not mean-reverting")
    fail("  c) half_life > MAX_HALF_LIFE - signal too slow")
    fail(f"  d) half_life < MIN_HALF_LIFE={MIN_HALF_LIFE} - signal too fast")
    # Show raw OU fit details
    for i, p in enumerate(ou_list[:3]):
        print(f"  Fit {i}: theta={p.theta:.6f} hl={p.half_life:.2f} "
              f"valid={p.is_valid} err='{p.error[:60]}'")
    sys.exit(1)

ou = valid_ou[-1]
ok(f"OU: theta={ou.theta:.6f}  mu={ou.mu:.8f}  sigma={ou.sigma:.8f}")
ok(f"    half_life={ou.half_life:.2f} bars  sigma_eq={ou.sigma_eq:.8f}")
ok(f"    half_life_scale={half_life_scale(ou):.2f}")

if ou.half_life < MIN_HALF_LIFE:
    fail(f"half_life={ou.half_life:.2f} < MIN={MIN_HALF_LIFE} → no trades")
if ou.half_life > MAX_HALF_LIFE:
    fail(f"half_life={ou.half_life:.2f} > MAX={MAX_HALF_LIFE} → no trades")
if half_life_scale(ou) == 0:
    fail("half_life_scale=0 → entry gate blocked → no trades")

sep("STEP 4: Bertram entry threshold")
a_star = bertram_threshold(ou, cost_frac=0.002)
a_norm = a_star / max(ou.sigma_eq, 1e-12)
ok(f"Bertram a*={a_star:.8f}")
ok(f"Normalized threshold = a*/sigma_eq = {a_norm:.2f} sigma units")

if a_norm > 10:
    fail(f"Threshold {a_norm:.1f} sigma is WAY too large. "
         "Spread will almost never cross it. "
         "Root cause: sigma_eq too small relative to cost.")
    fail(f"  sigma_eq={ou.sigma_eq:.2e}  cost={0.002:.4f}")
    fail(f"  cost/sigma_eq ratio = {0.002/max(ou.sigma_eq,1e-12):.1f}")
    fail("  Fix: cost_frac too high relative to spread volatility, OR")
    fail("       spread needs to be in price units not log-price units")

sep("STEP 5: Z-score range vs threshold")
# Extend spread series past warmup
more_sp = list(spreads)
for i in range(WARMUP, min(N, WARMUP+500)):
    st = kf.step(float(lp_a[i]), float(lp_b[i]))
    more_sp.append(st.spread)

zscores = np.array([(s - ou.mu) / max(ou.sigma_eq, 1e-12)
                    for s in more_sp[-500:]])
ok(f"Z-score stats (500 bars post-warmup):")
ok(f"  mean={zscores.mean():.2f}  std={zscores.std():.2f}")
ok(f"  min={zscores.min():.2f}  max={zscores.max():.2f}")
ok(f"  |z|>{a_norm:.2f}: {(np.abs(zscores)>a_norm).sum()} times out of 500")

if (np.abs(zscores) > a_norm).sum() == 0:
    fail(f"Z-score NEVER crosses threshold {a_norm:.2f}. Zero trades guaranteed.")
    fail(f"  Spread std in z-score units: {zscores.std():.2f}")
    fail(f"  Threshold in z-score units:  {a_norm:.2f}")
    if a_norm > zscores.std() * 3:
        fail("  Threshold is >3x the actual spread volatility.")
        fail("  Either lower cost_frac or the pair lacks mean-reversion.")
else:
    ok(f"Signal fires {(np.abs(zscores)>a_norm).sum()} times → trades should occur")

sep("STEP 6: Summary")
print(f"  Spread std (log units): {np.std(more_sp[-500:]):.8f}")
print(f"  sigma_eq (OU):          {ou.sigma_eq:.8f}")
print(f"  Bertram a* (log units): {a_star:.8f}")
print(f"  Cost frac:              0.002000")
print(f"  a*/sigma_eq ratio:      {a_norm:.2f}")
print()
if a_norm > 5:
    print("ROOT CAUSE: Bertram threshold is unreachably high.")
    print("The spread (in log-price units) is too small relative to cost.")
    print()
    print("FIXES TO TRY:")
    print("  1. Reduce cost_frac from 0.002 to 0.0005 in backtest.py")
    print("     (Alpaca crypto maker fee is ~0.15%, not 0.20%)")
    print("  2. Use a fixed z-score entry (e.g. |z|>1.5) instead of Bertram")
    print("  3. Widen the spread units (scale by entry price)")
else:
    print("Pipeline looks healthy. Check HMM regime blocking trades.")
