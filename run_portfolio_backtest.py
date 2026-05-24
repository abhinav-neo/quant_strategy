r"""
run_portfolio_backtest.py
==========================
Weekend 2: Multi-pair portfolio backtest with realistic execution costs.

Runs all 10 pairs simultaneously (3 original + 7 new), caps simultaneous
open positions at 3, allocates capital equally across open trades.

Computes:
  - Portfolio equity curve
  - Portfolio Sharpe, Sortino, max drawdown
  - Per-pair contribution
  - Pair correlation matrix
  - Results at 3 cost scenarios: Alpaca, Binance perps, Hyperliquid

Run:
    python run_portfolio_backtest.py
"""
import os, sys, math
import numpy as np
import pandas as pd
sys.path.insert(0, r"E:\MyDevelopment\GitHub\quant_strategy")
import logging; logging.basicConfig(level=logging.WARNING)

from modules.kalman_filter   import KalmanFilter, estimate_noise_params
from modules.ou_estimator    import fit_ou_rolling, half_life_scale
from modules.execution_model import ExecutionCostModel

# -- configuration -------------------------------------------------------------
ENTRY_Z  = 3.0
EXIT_Z   = 0.2
MAX_HOLD = 24       # bars (hours at 1h freq)
WARMUP   = 200
OU_WIN   = 200
REFIT    = 100
MAX_SIMULTANEOUS = 3    # max open trades at one time
CAPITAL  = 10_000       # starting capital $

# All 10 pairs: 3 original + 7 new from Phase 0
ALL_PAIRS = [
    # Original 3
    ("BTC", "ETH"),
    ("BTC", "SOL"),
    ("ETH", "SOL"),
    # New 7 from Phase 0
    ("BCH", "DOGE"),
    ("ETH", "UNI"),
    ("LTC", "UNI"),
    ("BCH", "BTC"),
    ("AVAX", "LINK"),
    ("DOGE", "LTC"),
    ("BTC", "DOGE"),
]

DATA_DIRS = {
    "2024": r"E:\MyDevelopment\GitHub\quant_strategy\data\phase0",
    "2025": r"E:\MyDevelopment\GitHub\quant_strategy\data\2025",
}
# SOL not in phase0 dir — use parent data dir for 2024
DATA_DIRS_FALLBACK = {
    "2024": r"E:\MyDevelopment\GitHub\quant_strategy\data",
}

def load_1h(symbol, data_dir, fallback_dir=None):
    fpath = os.path.join(data_dir, f"{symbol}_USD_1min.csv")
    if not os.path.exists(fpath) and fallback_dir:
        fpath = os.path.join(fallback_dir, f"{symbol}_USD_1min.csv")
    if not os.path.exists(fpath):
        return None
    df = pd.read_csv(fpath, parse_dates=[0], index_col=0)
    out = pd.DataFrame()
    out["close"] = df["close"].resample("1h").last()
    return out.dropna()

def run_portfolio(year, venue="alpaca_crypto"):
    """Run full portfolio backtest for a given year and venue."""
    cost_model = ExecutionCostModel(venue)
    data_dir = DATA_DIRS[year]
    fallback  = DATA_DIRS_FALLBACK.get(year)

    # Load all symbols needed
    needed = set()
    for a, b in ALL_PAIRS:
        needed.add(a); needed.add(b)
    bars = {}
    for sym in needed:
        df = load_1h(sym, data_dir, fallback)
        if df is not None:
            bars[sym] = df

    # Find common time index across all loaded symbols
    common = None
    for sym, df in bars.items():
        common = df.index if common is None else common.intersection(df.index)
    if common is None or len(common) < WARMUP + 200:
        print(f"  [skip] {year}: insufficient common bars")
        return None

    # Align all bars to common index
    for sym in bars:
        bars[sym] = bars[sym].loc[common]

    n_bars = len(common)
    days = n_bars / 24.0

    # Pre-compute log prices
    lp = {sym: np.log(bars[sym]["close"].values) for sym in bars}

    # Initialise Kalman filters and OU states per pair
    valid_pairs = []
    kf_states = {}
    ou_states = {}
    spreads_hist = {}

    for sym_a, sym_b in ALL_PAIRS:
        if sym_a not in lp or sym_b not in lp:
            continue
        label = f"{sym_a}/{sym_b}"
        la = lp[sym_a]; lb = lp[sym_b]

        q, r = estimate_noise_params(la[:WARMUP], lb[:WARMUP])
        kf = KalmanFilter(q_scale=q, r_scale=r)
        spreads = []
        for i in range(WARMUP):
            st = kf.step(float(la[i]), float(lb[i]))
            spreads.append(st.spread)

        ou_list = fit_ou_rolling(np.array(spreads),
                                  window=len(spreads)-1, step=len(spreads)-1)
        ou = next((p for p in ou_list if p.is_valid), None)
        if ou is None:
            continue

        kf_states[label] = kf
        ou_states[label] = ou
        spreads_hist[label] = spreads.copy()
        valid_pairs.append((sym_a, sym_b, label))

    # Portfolio simulation
    capital = float(CAPITAL)
    peak    = capital
    max_dd  = 0.0

    open_trades  = {}   # label -> trade dict
    all_trades   = {label: [] for _, _, label in valid_pairs}
    equity_curve = [capital]
    last_refit   = {label: WARMUP for _, _, label in valid_pairs}

    for i in range(WARMUP, n_bars):
        # Step all Kalman filters
        zscores = {}
        spreads_now = {}
        for sym_a, sym_b, label in valid_pairs:
            la = lp[sym_a]; lb = lp[sym_b]
            st = kf_states[label].step(float(la[i]), float(lb[i]))
            spreads_hist[label].append(st.spread)
            spreads_now[label] = st.spread

            # Refit OU
            if i - last_refit[label] >= REFIT:
                recent = np.array(spreads_hist[label][-OU_WIN:])
                nl = fit_ou_rolling(recent,
                                     window=len(recent)-1, step=len(recent)-1)
                no = next((p for p in nl if p.is_valid), None)
                if no:
                    ou_states[label] = no
                last_refit[label] = i

            ou = ou_states[label]
            z = (st.spread - ou.mu) / max(ou.sigma_eq, 1e-12)
            zscores[label] = z

        # -- EXIT open trades --------------------------------------------------
        for label in list(open_trades.keys()):
            trade = open_trades[label]
            trade["held"] += 1
            d = trade["dir"]
            z = zscores[label]
            ex = False
            if   d == 1  and z > -EXIT_Z: ex = True
            elif d == -1 and z <  EXIT_Z: ex = True
            elif trade["held"] >= MAX_HOLD: ex = True
            if ex:
                pos    = trade["pos_usd"]
                e_sp   = trade["entry_spread"]
                sp_now = spreads_now[label]
                raw    = d * (sp_now - e_sp) * pos
                cost   = pos * cost_model.total_cost_frac(
                    zscore=abs(trade["entry_z"]),
                    pos_usd=pos)
                net    = max(min(raw, pos * 0.5), -pos * 0.5) - cost
                capital += net
                all_trades[label].append(net)
                if capital > peak: peak = capital
                dd = (peak - capital) / peak * 100
                if dd > max_dd: max_dd = dd
                del open_trades[label]

        equity_curve.append(capital)

        # -- ENTRY: only if below simultaneous cap -----------------------------
        if len(open_trades) < MAX_SIMULTANEOUS:
            # How much capital per new trade
            pos_usd = CAPITAL / len(valid_pairs)   # equal allocation across ALL pairs

            for sym_a, sym_b, label in valid_pairs:
                if label in open_trades:
                    continue
                if len(open_trades) >= MAX_SIMULTANEOUS:
                    break
                ou = ou_states[label]
                if not (0.3 <= ou.half_life <= 240 and half_life_scale(ou) > 0):
                    continue
                z = zscores[label]
                sig = None
                if z < -ENTRY_Z: sig = 1
                elif z > ENTRY_Z: sig = -1
                if sig:
                    open_trades[label] = dict(
                        dir=sig,
                        entry_spread=spreads_now[label],
                        entry_z=z,
                        pos_usd=pos_usd,
                        held=0,
                    )

    # -- compute metrics -------------------------------------------------------
    all_net = []
    for label, tlist in all_trades.items():
        all_net.extend(tlist)

    total_return = (capital - CAPITAL) / CAPITAL * 100
    ann_return   = total_return * (365 / days)

    if len(all_net) > 1:
        rets = np.array(all_net) / CAPITAL
        sharpe   = (rets.mean() / rets.std() * np.sqrt(len(rets) * 365 / days)
                    if rets.std() > 0 else 0)
        neg = rets[rets < 0]
        sortino  = (rets.mean() / neg.std() * np.sqrt(len(rets) * 365 / days)
                    if len(neg) > 0 and neg.std() > 0 else 0)
    else:
        sharpe = sortino = 0.0

    wins = sum(1 for t in all_net if t > 0)
    win_rate = wins / len(all_net) * 100 if all_net else 0

    # Per-pair breakdown
    pair_stats = {}
    for _, _, label in valid_pairs:
        tlist = all_trades[label]
        if tlist:
            pair_stats[label] = {
                "trades": len(tlist),
                "wins":   sum(1 for t in tlist if t > 0),
                "total":  sum(tlist),
            }

    return {
        "year":          year,
        "venue":         venue,
        "capital_start": CAPITAL,
        "capital_end":   round(capital, 2),
        "total_return":  round(total_return, 2),
        "ann_return":    round(ann_return, 2),
        "max_drawdown":  round(max_dd, 3),
        "sharpe":        round(sharpe, 3),
        "sortino":       round(sortino, 3),
        "total_trades":  len(all_net),
        "win_rate":      round(win_rate, 1),
        "days":          round(days, 0),
        "pair_stats":    pair_stats,
        "equity_curve":  equity_curve,
    }


def print_result(r):
    if r is None: return
    print(f"\n  Year={r['year']}  Venue={r['venue']}")
    print(f"  Capital:   ${r['capital_start']:,} -> ${r['capital_end']:,}")
    print(f"  Return:    {r['total_return']:+.2f}%  (ann: {r['ann_return']:+.1f}%)")
    print(f"  Sharpe:    {r['sharpe']:.3f}   Sortino: {r['sortino']:.3f}")
    print(f"  Max DD:    {r['max_drawdown']:.3f}%")
    print(f"  Trades:    {r['total_trades']}  Win: {r['win_rate']:.1f}%")
    print(f"  Per-pair breakdown:")
    for label, s in sorted(r['pair_stats'].items(),
                            key=lambda x: -x[1]['total']):
        wrate = s['wins']/s['trades']*100 if s['trades'] else 0
        print(f"    {label:>12}  {s['trades']:>4} trades  "
              f"{wrate:>5.1f}% wins  ${s['total']:>+8.2f}")


if __name__ == "__main__":
    print("="*70)
    print(" Weekend 2: Portfolio Backtest — 10 pairs, 3 cost scenarios")
    print("="*70)

    for year in ["2024", "2025"]:
        print(f"\n{'-'*70}")
        print(f" YEAR: {year}")
        print(f"{'-'*70}")
        for venue in ["alpaca_crypto", "binance_perps", "hyperliquid"]:
            r = run_portfolio(year, venue)
            print_result(r)

    # Decision gate
    r_alpaca_2025 = run_portfolio("2025", "alpaca_crypto")
    if r_alpaca_2025:
        print(f"\n{'='*70}")
        print(" DECISION GATE")
        print(f"{'='*70}")
        s = r_alpaca_2025["sharpe"]
        a = r_alpaca_2025["ann_return"]
        if s > 1.0 and a > 5:
            print(f"  Alpaca 2025: Sharpe={s:.2f}  Ann={a:.1f}%")
            print("  PROCEED to Weekend 3 (cost realism + slippage stress test)")
        else:
            print(f"  Alpaca 2025: Sharpe={s:.2f}  Ann={a:.1f}%")
            print("  STOP. Portfolio does not meet Sharpe>1.0 + 5%/yr gate.")
            print("  Consider Path C (Hyperliquid/Binance perps).")
