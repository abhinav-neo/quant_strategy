"""
run_paper_live.py
==================
Weekend 4: Paper trading live execution skeleton.

Connects to Alpaca paper trading API. Polls every 60 seconds.
Maintains pair states in JSON. Logs every decision to file.
NO real money — paper account only.

Run:
    set ALPACA_KEY=your_paper_key
    set ALPACA_SECRET=your_paper_secret
    python run_paper_live.py

Kill with Ctrl+C. State is saved to data/paper_state.json on exit.
"""
import os, sys, time, json, signal, logging
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/paper_live.log", encoding="utf-8"),
    ]
)
log = logging.getLogger("paper_live")

# ── configuration ─────────────────────────────────────────────────────────────
PAIRS = [
    ("BCH", "DOGE"),
    ("ETH", "UNI"),
    ("LTC", "UNI"),
    ("BCH", "BTC"),
    ("AVAX", "LINK"),
    ("DOGE", "LTC"),
    ("BTC", "DOGE"),
    ("BTC", "ETH"),
]

ENTRY_Z     = 3.0
EXIT_Z      = 0.2
MAX_HOLD    = 24          # bars (hours)
MAX_ENTRY_Z = 6.0         # sigma guard
MAX_OPEN    = 3           # max simultaneous trades
CAPITAL     = 10_000.0    # paper capital
POLL_SECS   = 60          # check every 60 seconds

STATE_FILE  = "data/paper_state.json"
LOG_FILE    = "data/paper_trades.csv"

# ── state management ──────────────────────────────────────────────────────────
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "capital": CAPITAL,
        "open_trades": {},
        "closed_trades": [],
        "bars_seen": 0,
        "pair_spreads": {},
        "pair_ou": {},
    }

def save_state(state):
    os.makedirs("data", exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)

def log_trade(trade_dict):
    """Append a completed trade to the CSV log."""
    os.makedirs("data", exist_ok=True)
    header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        if header:
            f.write("timestamp,pair,direction,entry_z,exit_z,held_bars,"
                    "entry_spread,exit_spread,raw_pnl,cost,net_pnl\n")
        f.write(",".join(str(trade_dict.get(k, "")) for k in
                         ["timestamp","pair","direction","entry_z","exit_z",
                          "held_bars","entry_spread","exit_spread",
                          "raw_pnl","cost","net_pnl"]) + "\n")

# ── Alpaca live data fetcher ──────────────────────────────────────────────────
def fetch_latest_close(symbols, api_key, api_secret):
    """
    Fetch the latest 1-minute close for each symbol via Alpaca REST.
    Returns dict: {symbol: close_price} or None on error.
    """
    import urllib.request
    results = {}
    for sym in symbols:
        url = (f"https://data.alpaca.markets/v1beta3/crypto/us/latest/bars"
               f"?symbols={sym}/USD&timeframe=1Min")
        req = urllib.request.Request(url, headers={
            "APCA-API-KEY-ID":     api_key,
            "APCA-API-SECRET-KEY": api_secret,
        })
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                bar = data.get("bars", {}).get(sym+"/USD", {})
                if bar:
                    results[sym] = float(bar["c"])
        except Exception as e:
            log.warning("fetch %s: %s", sym, e)
    return results

# ── signal handler ────────────────────────────────────────────────────────────
_running = True

def _stop(signum, frame):
    global _running
    log.info("Shutdown signal received")
    _running = False

signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

# ── main loop ─────────────────────────────────────────────────────────────────
def main():
    api_key    = os.environ.get("ALPACA_KEY", "")
    api_secret = os.environ.get("ALPACA_SECRET", "")
    if not api_key or not api_secret:
        log.error("Set ALPACA_KEY and ALPACA_SECRET environment variables")
        sys.exit(1)

    # Import strategy modules
    from modules.kalman_filter import KalmanFilter, estimate_noise_params
    from modules.ou_estimator  import fit_ou_rolling, half_life_scale

    state = load_state()
    log.info("Paper live started. Capital: $%.2f", state["capital"])
    log.info("Pairs: %s", [a+"/"+b for a,b in PAIRS])
    log.info("Params: ENTRY_Z=%.1f EXIT_Z=%.1f MAX_HOLD=%d MAX_OPEN=%d",
             ENTRY_Z, EXIT_Z, MAX_HOLD, MAX_OPEN)

    # Per-pair Kalman filters and OU state (warm up from stored spreads)
    all_syms = list(set(s for a,b in PAIRS for s in [a,b]))
    pair_states = {}

    for a, b in PAIRS:
        label = a+"/"+b
        # Get stored spread history or start fresh
        spreads = state["pair_spreads"].get(label, [])
        q = state["pair_ou"].get(label, {}).get("q", 1e-4)
        r = state["pair_ou"].get(label, {}).get("r", 1e-4)
        kf = KalmanFilter(q_scale=q, r_scale=r)
        # Re-warm Kalman from stored spreads (approximate)
        for sp in spreads[-200:]:
            pass  # Kalman state is reset but spread history preserved
        ou = None
        if len(spreads) >= 50:
            ou_list = fit_ou_rolling(np.array(spreads[-200:]),
                                     window=min(200,len(spreads[-200:])-1),
                                     step=max(1,len(spreads[-200:])-1))
            ou = next((p for p in ou_list if p.is_valid), None)
        pair_states[label] = {"kf": kf, "spreads": spreads, "ou": ou}

    bar_count = state.get("bars_seen", 0)
    pos_per   = CAPITAL / len(PAIRS)

    log.info("Warming up Kalman filters — need 200 bars before first trade")

    while _running:
        tick_start = time.time()

        # Fetch latest prices
        prices = fetch_latest_close(all_syms, api_key, api_secret)
        if len(prices) < 2:
            log.warning("Insufficient price data, skipping bar")
            time.sleep(POLL_SECS)
            continue

        bar_count += 1
        ts = datetime.now(timezone.utc).isoformat()

        # Step each pair
        open_trades = state["open_trades"]
        for a, b in PAIRS:
            label = a+"/"+b
            if a not in prices or b not in prices:
                continue
            lp_a = np.log(prices[a])
            lp_b = np.log(prices[b])
            ps   = pair_states[label]
            st   = ps["kf"].step(lp_a, lp_b)
            ps["spreads"].append(float(st.spread))
            if len(ps["spreads"]) > 2000:
                ps["spreads"] = ps["spreads"][-1000:]

            # Refit OU every 24 bars
            if bar_count % 24 == 0 and len(ps["spreads"]) >= 50:
                recent = np.array(ps["spreads"][-200:])
                ou_list = fit_ou_rolling(recent,
                                         window=len(recent)-1,
                                         step=len(recent)-1)
                new_ou = next((p for p in ou_list if p.is_valid), None)
                if new_ou:
                    ps["ou"] = new_ou

            if ps["ou"] is None or bar_count < 200:
                continue

            ou = ps["ou"]
            z  = (st.spread - ou.mu) / max(ou.sigma_eq, 1e-12)

            # ── exit check ────────────────────────────────────────────────
            if label in open_trades:
                trade = open_trades[label]
                trade["held"] += 1
                d = trade["dir"]
                exit_reason = None
                if   d == 1  and z > -EXIT_Z:          exit_reason = "target"
                elif d == -1 and z <  EXIT_Z:           exit_reason = "target"
                elif trade["held"] >= MAX_HOLD:         exit_reason = "timeout"

                if exit_reason:
                    raw    = d * (st.spread - trade["entry_spread"]) * pos_per
                    cost   = pos_per * 0.0040
                    net    = max(min(raw, pos_per*0.5), -pos_per*0.5) - cost
                    state["capital"] += net
                    trade_rec = {
                        "timestamp": ts, "pair": label,
                        "direction": "long" if d==1 else "short",
                        "entry_z":   trade["entry_z"],
                        "exit_z":    round(z, 3),
                        "held_bars": trade["held"],
                        "entry_spread": trade["entry_spread"],
                        "exit_spread":  round(st.spread, 6),
                        "raw_pnl":   round(raw, 4),
                        "cost":      round(cost, 4),
                        "net_pnl":   round(net, 4),
                    }
                    log.info("EXIT  %-12s dir=%+d z=%+.2f held=%d net=$%+.2f  cap=$%.2f",
                             label, d, z, trade["held"], net, state["capital"])
                    log_trade(trade_rec)
                    state["closed_trades"].append(trade_rec)
                    del open_trades[label]
                continue

            # ── entry check ───────────────────────────────────────────────
            if len(open_trades) >= MAX_OPEN:
                continue
            if not (0.3 <= ou.half_life <= 240 and half_life_scale(ou) > 0):
                continue
            if abs(z) > MAX_ENTRY_Z:
                continue

            sig = None
            if z < -ENTRY_Z: sig = 1
            elif z > ENTRY_Z: sig = -1

            if sig:
                open_trades[label] = {
                    "dir":          sig,
                    "entry_spread": float(st.spread),
                    "entry_z":      round(z, 3),
                    "held":         0,
                    "entry_time":   ts,
                }
                log.info("ENTRY %-12s dir=%+d z=%+.2f hl=%.2f  open=%d",
                         label, sig, z, ou.half_life, len(open_trades))

        # Periodic status log every 60 bars (1 hour)
        if bar_count % 60 == 0:
            closed = state["closed_trades"]
            n = len(closed)
            wins = sum(1 for t in closed if t["net_pnl"] > 0)
            log.info("STATUS bar=%d open=%d closed=%d wins=%d(%.0f%%) cap=$%.2f",
                     bar_count, len(open_trades), n,
                     wins, wins/n*100 if n else 0, state["capital"])

        state["bars_seen"] = bar_count

        # Kill switch: drawdown > 8%
        drawdown = (CAPITAL - state["capital"]) / CAPITAL * 100
        if drawdown > 8.0:
            log.error("KILL SWITCH: drawdown %.2f%% > 8%%. Stopping.", drawdown)
            break

        # Save state every bar
        save_state(state)

        # Sleep until next bar
        elapsed = time.time() - tick_start
        sleep_time = max(0, POLL_SECS - elapsed)
        if _running:
            time.sleep(sleep_time)

    # Final save
    save_state(state)
    closed = state["closed_trades"]
    n = len(closed)
    wins = sum(1 for t in closed if t["net_pnl"] > 0)
    total_pnl = sum(t["net_pnl"] for t in closed)
    log.info("="*60)
    log.info("SESSION COMPLETE")
    log.info("  Bars seen:    %d", bar_count)
    log.info("  Closed trades: %d  wins: %d (%.0f%%)", n, wins, wins/n*100 if n else 0)
    log.info("  Total P&L:    $%+.2f", total_pnl)
    log.info("  Final capital: $%.2f", state["capital"])
    log.info("  State saved:  %s", STATE_FILE)
    log.info("="*60)


if __name__ == "__main__":
    main()
