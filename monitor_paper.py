"""
monitor_paper.py
=================
Daily 5-minute check on paper trading performance.
Run any time to see current P&L, open trades, and health metrics.

Usage:
    python monitor_paper.py
"""
import os, sys, json
from datetime import datetime, timezone
sys.path.insert(0, str(__file__).replace("monitor_paper.py",""))

STATE_FILE  = "data/paper_state.json"
TRADES_FILE = "data/paper_trades.csv"
CAPITAL     = 10_000.0
KILL_DD     = 8.0
WARN_WINRATE = 60.0

def fmt_pnl(v):
    sign = "+" if v >= 0 else ""
    return sign + "$" + str(round(v, 2))

def main():
    if not os.path.exists(STATE_FILE):
        print("No paper_state.json found.")
        print("Start paper trading first: python run_paper_live.py")
        return

    state = json.load(open(STATE_FILE, encoding="utf-8"))
    closed = state.get("closed_trades", [])
    open_t  = state.get("open_trades", {})
    capital = state.get("capital", CAPITAL)
    bars    = state.get("bars_seen", 0)
    started = state.get("started_at", "unknown")

    hours_running = bars / 1.0  # 1 bar per hour (1h polling)
    days_running  = hours_running / 24.0

    # P&L metrics
    n = len(closed)
    wins = sum(1 for t in closed if t["net_pnl"] > 0)
    losses = n - wins
    total_pnl = sum(t["net_pnl"] for t in closed)
    win_rate  = wins / n * 100 if n else 0
    avg_pnl   = total_pnl / n if n else 0

    # Drawdown
    running_cap = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    for t in closed:
        running_cap += t["net_pnl"]
        if running_cap > peak: peak = running_cap
        dd = (peak - running_cap) / peak * 100
        if dd > max_dd: max_dd = dd

    ann_ret = (capital - CAPITAL) / CAPITAL * 100 * (365 / days_running) if days_running > 0 else 0

    # Per-pair breakdown
    pair_pnl = {}
    for t in closed:
        pair_pnl.setdefault(t["pair"], []).append(t["net_pnl"])

    # Health flags
    flags = []
    if max_dd > KILL_DD:
        flags.append("KILL SWITCH: drawdown > " + str(KILL_DD) + "%")
    if n >= 20 and win_rate < WARN_WINRATE:
        flags.append("WARNING: win rate " + str(round(win_rate,1)) + "% < " + str(WARN_WINRATE) + "%")
    if n >= 5 and avg_pnl < 0:
        flags.append("WARNING: avg P&L is negative ($" + str(round(avg_pnl,2)) + ")")

    # Print report
    print("=" * 60)
    print(" Paper Trading Monitor — " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("=" * 60)
    print()
    print("  Session started:  " + str(started)[:19])
    print("  Bars seen:        " + str(bars) + " (" + str(round(days_running, 1)) + " days)")
    print("  Open trades:      " + str(len(open_t)) + " — " + str(list(open_t.keys())))
    print()
    print("  Capital:          $" + str(CAPITAL) + " -> $" + str(round(capital, 2)))
    print("  Total P&L:        " + fmt_pnl(total_pnl))
    print("  Annualised return:" + str(round(ann_ret, 1)) + "%")
    print("  Max drawdown:     " + str(round(max_dd, 3)) + "%")
    print()
    print("  Closed trades:    " + str(n) + "  wins=" + str(wins) + " losses=" + str(losses))
    print("  Win rate:         " + str(round(win_rate, 1)) + "%")
    print("  Avg P&L/trade:    " + fmt_pnl(avg_pnl))
    print()

    if pair_pnl:
        print("  Per-pair P&L:")
        for pair, pnls in sorted(pair_pnl.items(), key=lambda x: -sum(x[1])):
            pw = sum(1 for p in pnls if p > 0)
            print("    " + pair.ljust(14) +
                  str(len(pnls)) + " trades  " +
                  str(round(pw/len(pnls)*100, 0)) + "% wins  " +
                  fmt_pnl(sum(pnls)))
        print()

    if flags:
        print("  *** ALERTS ***")
        for f in flags:
            print("  !! " + f)
        print()
    else:
        print("  Health: OK — no alerts")
        print()

    # Backtest comparison
    print("  Backtest expectation (9-month, 8 pairs, 0.40% cost):")
    print("    Expected win rate:    97%")
    print("    Expected ann return:  ~130% (optimistic) / 10-15% (conservative)")
    print("    Expected max DD:      < 0.1%")
    print()

    if n < 20:
        print("  Note: " + str(n) + "/20 trades for statistical significance.")
        print("  Wait for 20 trades before drawing conclusions.")

    print("=" * 60)

if __name__ == "__main__":
    main()
