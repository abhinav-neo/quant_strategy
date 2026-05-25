"""
run_live.py
============
Production bot: live signal detection + real order execution via Alpaca.
Works on paper account (default) or live account (set PAPER=false).

Architecture:
  - Every 60 seconds: fetch latest prices for all 16 symbols
  - Step Kalman + OU per pair -> compute z-score
  - On signal: place two-leg market order via AlpacaExecutor
  - On exit: close both legs
  - Persist state to data/live_state.json after every bar
  - Kill switch: stop if portfolio drawdown > 8%

Run:
    set ALPACA_KEY=your_key
    set ALPACA_SECRET=your_secret
    python run_live.py

    # For real money (use with caution):
    set PAPER=false
    python run_live.py
"""
import os, sys, time, json, signal, logging
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)-12s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/live.log", encoding="utf-8"),
    ]
)
log = logging.getLogger("run_live")

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

ENTRY_Z      = 3.0
EXIT_Z       = 0.2
MAX_HOLD     = 24
MAX_ENTRY_Z  = 6.0
WARMUP       = 200
OU_WIN       = 200
REFIT        = 24
MAX_OPEN     = 3
CAPITAL      = 10_000.0
NOTIONAL_PER_LEG = CAPITAL / len(PAIRS)   # $1,250 per leg
POLL_SECS    = 60
KILL_DD_PCT  = 8.0

STATE_FILE   = "data/live_state.json"
TRADE_LOG    = "data/live_trades.csv"

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {
        "started_at":    datetime.now(timezone.utc).isoformat(),
        "capital":       CAPITAL,
        "open_trades":   {},
        "closed_trades": [],
        "bars_seen":     0,
        "pair_spreads":  {},
        "equity_curve":  [CAPITAL],
    }

def save_state(state):
    os.makedirs("data", exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, STATE_FILE)   # atomic write

def log_trade(t: dict):
    os.makedirs("data", exist_ok=True)
    header = not os.path.exists(TRADE_LOG)
    keys = ["ts","pair","dir","entry_z","exit_z","held",
            "entry_spread","exit_spread","raw_pnl","cost","net_pnl","exit_reason"]
    with open(TRADE_LOG, "a", encoding="utf-8") as f:
        if header:
            f.write(",".join(keys) + "\n")
        f.write(",".join(str(t.get(k,"")) for k in keys) + "\n")

def fetch_prices(symbols: list, executor) -> dict:
    """
    Fetch latest 1-min close for each base symbol via Alpaca data API.
    Returns {sym: price} e.g. {"BTC": 65432.10, "ETH": 3210.50}
    """
    import urllib.request
    prices = {}
    slash_syms = [s+"/USD" for s in symbols]
    joined = ",".join(slash_syms)
    url = ("https://data.alpaca.markets/v1beta3/crypto/us/latest/bars"
           "?symbols=" + joined + "&timeframe=1Min")
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID":     executor.api_key,
        "APCA-API-SECRET-KEY": executor.api_secret,
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            for slash_sym in slash_syms:
                bar = data.get("bars", {}).get(slash_sym)
                if bar:
                    base = slash_sym.replace("/USD", "")
                    prices[base] = float(bar["c"])
    except Exception as e:
        log.warning("price fetch failed: %s", e)
    return prices

_running = True
def _stop(sig, frame):
    global _running
    log.info("Shutdown signal — finishing current bar")
    _running = False

signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)


def main():
    api_key    = os.environ.get("ALPACA_KEY", "")
    api_secret = os.environ.get("ALPACA_SECRET", "")
    is_paper   = os.environ.get("PAPER", "true").lower() != "false"

    if not api_key or not api_secret:
        log.error("Set ALPACA_KEY and ALPACA_SECRET environment variables")
        sys.exit(1)

    from modules.alpaca_execution import AlpacaExecutor
    from modules.kalman_filter    import KalmanFilter, estimate_noise_params
    from modules.ou_estimator     import fit_ou_rolling, half_life_scale

    executor = AlpacaExecutor(api_key, api_secret, paper=is_paper)
    acc = executor.get_account()
    buying_power = float(acc.get("buying_power", 0))
    log.info("Account ready. Buying power: $%.2f  Paper=%s", buying_power, is_paper)

    if not is_paper and buying_power < CAPITAL * 0.5:
        log.error("Insufficient buying power. Need $%.0f, have $%.0f", CAPITAL, buying_power)
        sys.exit(1)

    state = load_state()
    log.info("State loaded. Bars seen: %d  Open trades: %d",
             state["bars_seen"], len(state["open_trades"]))

    # Reconcile open positions with Alpaca on startup
    if state["open_trades"]:
        enriched = {}
        for label, t in state["open_trades"].items():
            a, b = label.split("/")
            enriched[label] = {"sym_a": a+"/USD", "sym_b": b+"/USD", "dir": t["dir"]}
        discrepancies = executor.reconcile_positions(enriched)
        if discrepancies:
            log.warning("Position discrepancies found: %s", list(discrepancies.keys()))
            log.warning("Review manually before proceeding. Clearing mismatched trades.")
            for label in discrepancies:
                state["open_trades"].pop(label, None)

    # Init Kalman + OU per pair
    all_syms = list(set(s for a, b in PAIRS for s in [a, b]))
    pair_states = {}
    for a, b in PAIRS:
        label    = a+"/"+b
        spreads  = state["pair_spreads"].get(label, [])
        kf = KalmanFilter(q_scale=1e-4, r_scale=1e-4)
        ou = None
        if len(spreads) >= 50:
            ou_list = fit_ou_rolling(np.array(spreads[-200:]),
                                     window=min(200, len(spreads[-200:]))-1,
                                     step=max(1, len(spreads[-200:])-1))
            ou = next((p for p in ou_list if p.is_valid), None)
        pair_states[label] = {"kf": kf, "spreads": spreads, "ou": ou, "refit_count": 0}

    bar_count    = state.get("bars_seen", 0)
    start_cap    = state.get("equity_curve", [CAPITAL])[0]
    log.info("Warmup needed: %d bars before first trade",
             max(0, WARMUP - bar_count))

    while _running:
        tick_start = time.time()
        ts = datetime.now(timezone.utc).isoformat()

        prices = fetch_prices(all_syms, executor)
        if len(prices) < 4:
            log.warning("Only %d prices fetched — skipping bar", len(prices))
            time.sleep(POLL_SECS)
            continue

        bar_count += 1
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

            # Refit OU every REFIT bars
            ps["refit_count"] += 1
            if ps["refit_count"] >= REFIT and len(ps["spreads"]) >= 50:
                recent  = np.array(ps["spreads"][-OU_WIN:])
                ou_list = fit_ou_rolling(recent,
                                         window=len(recent)-1,
                                         step=len(recent)-1)
                new_ou  = next((p for p in ou_list if p.is_valid), None)
                if new_ou:
                    ps["ou"] = new_ou
                ps["refit_count"] = 0

            if ps["ou"] is None or bar_count < WARMUP:
                continue

            ou = ps["ou"]
            z  = (st.spread - ou.mu) / max(ou.sigma_eq, 1e-12)

            # ── EXIT ──────────────────────────────────────────────────
            if label in open_trades:
                trade = open_trades[label]
                trade["held"] += 1
                d = trade["dir"]
                exit_reason = None

                if   d == 1  and z > -EXIT_Z:      exit_reason = "target"
                elif d == -1 and z <  EXIT_Z:       exit_reason = "target"
                elif trade["held"] >= MAX_HOLD:     exit_reason = "timeout"

                if exit_reason:
                    sym_a_slash = a+"/USD"
                    sym_b_slash = b+"/USD"
                    res_a, res_b = executor.exit_pair(sym_a_slash, sym_b_slash)

                    raw  = d * (st.spread - trade["entry_spread"]) * NOTIONAL_PER_LEG
                    cost = NOTIONAL_PER_LEG * 0.0040
                    net  = max(min(raw, NOTIONAL_PER_LEG*0.5), -NOTIONAL_PER_LEG*0.5) - cost

                    state["capital"] += net
                    state["equity_curve"].append(state["capital"])
                    if len(state["equity_curve"]) > 10000:
                        state["equity_curve"] = state["equity_curve"][-5000:]

                    rec = {
                        "ts": ts, "pair": label, "dir": d,
                        "entry_z":      trade["entry_z"],
                        "exit_z":       round(z, 3),
                        "held":         trade["held"],
                        "entry_spread": trade["entry_spread"],
                        "exit_spread":  round(st.spread, 6),
                        "raw_pnl":      round(raw, 4),
                        "cost":         round(cost, 4),
                        "net_pnl":      round(net, 4),
                        "exit_reason":  exit_reason,
                        "orders_ok":    res_a.ok and res_b.ok,
                    }
                    state["closed_trades"].append(rec)
                    log_trade(rec)
                    log.info("EXIT  %-12s dir=%+d z=%+.2f held=%d pnl=$%+.2f cap=$%.2f orders=%s",
                             label, d, z, trade["held"], net, state["capital"],
                             "OK" if res_a.ok and res_b.ok else "PARTIAL")
                    del open_trades[label]
                continue

            # ── ENTRY ─────────────────────────────────────────────────
            if len(open_trades) >= MAX_OPEN:
                continue
            if not (0.3 <= ou.half_life <= 240 and half_life_scale(ou) > 0):
                continue
            if abs(z) > MAX_ENTRY_Z:
                continue

            sig = None
            if z < -ENTRY_Z:  sig = 1
            elif z > ENTRY_Z: sig = -1

            if sig:
                sym_a_slash = a+"/USD"
                sym_b_slash = b+"/USD"
                res_a, res_b = executor.enter_pair(
                    sym_a_slash, sym_b_slash,
                    direction=sig, notional_usd=NOTIONAL_PER_LEG)

                if res_a.ok and res_b.ok:
                    open_trades[label] = {
                        "dir":          sig,
                        "entry_spread": float(st.spread),
                        "entry_z":      round(z, 3),
                        "held":         0,
                        "entry_ts":     ts,
                        "sym_a":        a,
                        "sym_b":        b,
                    }
                    log.info("ENTRY %-12s dir=%+d z=%+.2f hl=%.1fh open=%d",
                             label, sig, z, ou.half_life, len(open_trades))
                else:
                    log.warning("ENTRY failed %s — orders not placed", label)

        # Hourly status
        if bar_count % 60 == 0:
            closed = state["closed_trades"]
            n    = len(closed)
            wins = sum(1 for t in closed if t["net_pnl"] > 0)
            pnl  = sum(t["net_pnl"] for t in closed)
            log.info("STATUS bars=%d open=%d closed=%d win=%.0f%% pnl=$%+.2f cap=$%.2f",
                     bar_count, len(open_trades), n,
                     wins/n*100 if n else 0, pnl, state["capital"])

        # Kill switch
        peak = max(state.get("equity_curve", [CAPITAL]))
        dd   = (peak - state["capital"]) / peak * 100
        if dd >= KILL_DD_PCT:
            log.error("KILL SWITCH — drawdown %.2f%% >= %.0f%% — closing all positions",
                      dd, KILL_DD_PCT)
            for label in list(open_trades.keys()):
                a2, b2 = label.split("/")
                executor.exit_pair(a2+"/USD", b2+"/USD")
                del open_trades[label]
            break

        # Save spread history for persistence
        state["pair_spreads"] = {lb: ps["spreads"][-500:]
                                  for lb, ps in pair_states.items()}
        state["bars_seen"] = bar_count
        save_state(state)

        elapsed = time.time() - tick_start
        sleep   = max(1, POLL_SECS - elapsed)
        log.debug("bar=%d elapsed=%.1fs sleeping=%.0fs", bar_count, elapsed, sleep)
        if _running:
            time.sleep(sleep)

    # Final
    save_state(state)
    closed = state["closed_trades"]
    n    = len(closed)
    wins = sum(1 for t in closed if t["net_pnl"] > 0)
    pnl  = sum(t["net_pnl"] for t in closed)
    log.info("="*60)
    log.info("SESSION END  bars=%d trades=%d wins=%d(%.0f%%) pnl=$%+.2f cap=$%.2f",
             bar_count, n, wins, wins/n*100 if n else 0, pnl, state["capital"])
    log.info("State: %s", STATE_FILE)
    log.info("Trades: %s", TRADE_LOG)
    log.info("="*60)


if __name__ == "__main__":
    main()
