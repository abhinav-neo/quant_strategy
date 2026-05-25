"""
modules/alpaca_execution.py
============================
Live order execution for Alpaca paper/live trading.
Handles the two-leg stat-arb structure: buy one asset, sell the other.

Supports:
  - Market orders (default, for speed at signal time)
  - Position reconciliation on startup
  - Position queries to verify fills
  - Clean close of both legs on exit signal

Usage:
    from modules.alpaca_execution import AlpacaExecutor
    ex = AlpacaExecutor(api_key, api_secret, paper=True)
    ex.enter_pair("BCH/USD", "DOGE/USD", direction=1, notional_usd=1250)
    ex.exit_pair("BCH/USD", "DOGE/USD")
"""

import json
import logging
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("alpaca_exec")

BASE_PAPER = "https://paper-api.alpaca.markets"
BASE_LIVE  = "https://api.alpaca.markets"
DATA_BASE  = "https://data.alpaca.markets"


@dataclass
class OrderResult:
    ok: bool
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    qty: float = 0.0
    filled_qty: float = 0.0
    filled_avg_price: float = 0.0
    status: str = ""
    error: str = ""


@dataclass
class Position:
    symbol: str
    qty: float          # positive = long, negative = short
    avg_entry: float
    market_value: float
    unrealized_pnl: float


class AlpacaExecutor:
    """
    Thin wrapper around Alpaca REST API for crypto stat-arb execution.

    Alpaca crypto specifics:
    - Symbols use slash format for data API: BTC/USD
    - Symbols use no-slash format for trading API: BTCUSD
    - Fractional shares supported for crypto (min $1 notional)
    - Crypto trades 24/7 — no market-hours check needed
    - Paper account mirrors live API behaviour
    """

    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.base       = BASE_PAPER if paper else BASE_LIVE
        self.paper      = paper
        log.info("AlpacaExecutor ready (paper=%s)", paper)

    def _headers(self) -> dict:
        return {
            "APCA-API-KEY-ID":     self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type":        "application/json",
        }

    def _request(self, method: str, path: str, body: dict = None,
                 base_override: str = None) -> dict:
        url  = (base_override or self.base) + path
        data = json.dumps(body).encode() if body else None
        req  = urllib.request.Request(url, data=data,
                                      headers=self._headers(), method=method)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body_bytes = e.read()
            log.error("HTTP %d %s %s — %s", e.code, method, path, body_bytes[:200])
            return {"error": str(e.code), "detail": body_bytes.decode(errors="replace")}
        except Exception as e:
            log.error("Request failed %s %s — %s", method, path, e)
            return {"error": str(e)}

    @staticmethod
    def to_trading_sym(slash_sym: str) -> str:
        """BTC/USD -> BTCUSD"""
        return slash_sym.replace("/", "")

    @staticmethod
    def to_data_sym(slash_sym: str) -> str:
        """BTC/USD -> BTC/USD (unchanged, already correct for data API)"""
        return slash_sym

    def get_account(self) -> dict:
        return self._request("GET", "/v2/account")

    def get_positions(self) -> list[Position]:
        """Return all open positions."""
        raw = self._request("GET", "/v2/positions")
        if isinstance(raw, list):
            positions = []
            for p in raw:
                try:
                    positions.append(Position(
                        symbol          = p["symbol"],
                        qty             = float(p["qty"]),
                        avg_entry       = float(p.get("avg_entry_price", 0)),
                        market_value    = float(p.get("market_value", 0)),
                        unrealized_pnl  = float(p.get("unrealized_pl", 0)),
                    ))
                except Exception as e:
                    log.warning("parse position: %s", e)
            return positions
        return []

    def get_position(self, slash_sym: str) -> Optional[Position]:
        """Return position for a single symbol, None if flat."""
        sym = self.to_trading_sym(slash_sym)
        raw = self._request("GET", f"/v2/positions/{sym}")
        if "error" in raw or "code" in raw:
            return None
        try:
            return Position(
                symbol         = raw["symbol"],
                qty            = float(raw["qty"]),
                avg_entry      = float(raw.get("avg_entry_price", 0)),
                market_value   = float(raw.get("market_value", 0)),
                unrealized_pnl = float(raw.get("unrealized_pl", 0)),
            )
        except Exception as e:
            log.warning("get_position parse error: %s", e)
            return None

    def place_market_order(self, slash_sym: str, side: str,
                           notional_usd: float) -> OrderResult:
        """
        Place a fractional market order by notional USD amount.
        side: "buy" or "sell"
        """
        sym = self.to_trading_sym(slash_sym)
        body = {
            "symbol":        sym,
            "notional":      str(round(notional_usd, 2)),
            "side":          side,
            "type":          "market",
            "time_in_force": "ioc",   # immediate-or-cancel for crypto
        }
        log.info("ORDER %s %s $%.2f", side.upper(), sym, notional_usd)
        raw = self._request("POST", "/v2/orders", body=body)

        if "error" in raw or raw.get("status") == "rejected":
            return OrderResult(ok=False, symbol=sym, side=side,
                               error=raw.get("detail", str(raw)))

        return OrderResult(
            ok               = True,
            order_id         = raw.get("id", ""),
            symbol           = sym,
            side             = side,
            qty              = float(raw.get("qty") or 0),
            filled_qty       = float(raw.get("filled_qty") or 0),
            filled_avg_price = float(raw.get("filled_avg_price") or 0),
            status           = raw.get("status", ""),
        )

    def close_position(self, slash_sym: str) -> OrderResult:
        """Close entire position for a symbol (market order)."""
        sym = self.to_trading_sym(slash_sym)
        log.info("CLOSE position %s", sym)
        raw = self._request("DELETE", f"/v2/positions/{sym}",
                             body={"percentage": "100"})
        if "error" in raw:
            return OrderResult(ok=False, symbol=sym, error=str(raw))
        return OrderResult(ok=True, symbol=sym,
                           order_id=raw.get("id",""),
                           status=raw.get("status",""))

    def enter_pair(self, sym_a: str, sym_b: str,
                   direction: int, notional_usd: float) -> tuple[OrderResult, OrderResult]:
        """
        Enter a stat-arb pair trade.

        direction = +1: spread too low -> buy A, sell B
        direction = -1: spread too high -> sell A, buy B

        notional_usd: dollar size per leg (each leg gets this amount)
        """
        if direction == 1:
            side_a, side_b = "buy", "sell"
        else:
            side_a, side_b = "sell", "buy"

        log.info("ENTER PAIR %s/%s dir=%+d notional=$%.0f",
                 sym_a, sym_b, direction, notional_usd)

        res_a = self.place_market_order(sym_a, side_a, notional_usd)
        time.sleep(0.2)   # slight delay between legs
        res_b = self.place_market_order(sym_b, side_b, notional_usd)

        if res_a.ok and res_b.ok:
            log.info("ENTER OK  %s %s / %s %s", side_a, sym_a, side_b, sym_b)
        else:
            log.error("ENTER PARTIAL  A=%s B=%s — may need manual reconcile",
                      res_a.ok, res_b.ok)
            # If one leg failed, close the other to avoid naked exposure
            if res_a.ok and not res_b.ok:
                log.warning("Leg B failed — closing leg A to stay flat")
                self.close_position(sym_a)
            elif res_b.ok and not res_a.ok:
                log.warning("Leg A failed — closing leg B to stay flat")
                self.close_position(sym_b)

        return res_a, res_b

    def exit_pair(self, sym_a: str, sym_b: str) -> tuple[OrderResult, OrderResult]:
        """Close both legs of an open pair trade."""
        log.info("EXIT PAIR %s/%s", sym_a, sym_b)
        res_a = self.close_position(sym_a)
        time.sleep(0.2)
        res_b = self.close_position(sym_b)
        if res_a.ok and res_b.ok:
            log.info("EXIT OK  %s / %s", sym_a, sym_b)
        else:
            log.error("EXIT PARTIAL  A=%s B=%s", res_a.ok, res_b.ok)
        return res_a, res_b

    def reconcile_positions(self, expected_pairs: dict) -> dict:
        """
        On startup, compare bot's expected open trades to Alpaca's actual positions.
        Returns dict of discrepancies that need manual review.

        expected_pairs: {label: {sym_a, sym_b, dir}} from state file
        """
        live_positions = {p.symbol: p for p in self.get_positions()}
        discrepancies = {}

        for label, trade in expected_pairs.items():
            sym_a = self.to_trading_sym(trade["sym_a"])
            sym_b = self.to_trading_sym(trade["sym_b"])
            d     = trade["dir"]

            pos_a = live_positions.get(sym_a)
            pos_b = live_positions.get(sym_b)

            expected_side_a = "long" if d == 1 else "short"
            expected_side_b = "short" if d == 1 else "long"

            ok_a = (pos_a is not None and
                    ((d == 1 and pos_a.qty > 0) or (d == -1 and pos_a.qty < 0)))
            ok_b = (pos_b is not None and
                    ((d == 1 and pos_b.qty < 0) or (d == -1 and pos_b.qty > 0)))

            if not ok_a or not ok_b:
                discrepancies[label] = {
                    "expected_a": expected_side_a, "actual_a": pos_a,
                    "expected_b": expected_side_b, "actual_b": pos_b,
                }
                log.warning("RECONCILE mismatch for %s: check manually", label)

        if not discrepancies:
            log.info("RECONCILE OK — %d open pairs match Alpaca positions",
                     len(expected_pairs))
        return discrepancies

    def get_latest_price(self, slash_sym: str) -> Optional[float]:
        """Fetch latest trade price for a symbol from Alpaca data API."""
        sym = slash_sym.replace("/", "%2F")
        path = f"/v1beta3/crypto/us/latest/trades?symbols={slash_sym}"
        raw = self._request("GET", path, base_override=DATA_BASE)
        try:
            return float(raw["trades"][slash_sym]["p"])
        except Exception:
            return None


def _run_tests(api_key: str, api_secret: str) -> int:
    """Smoke tests — call only with paper keys."""
    passed = 0
    ex = AlpacaExecutor(api_key, api_secret, paper=True)

    acc = ex.get_account()
    assert "buying_power" in acc or "error" not in acc, "account fetch failed"
    passed += 1

    pos = ex.get_positions()
    assert isinstance(pos, list), "get_positions should return list"
    passed += 1

    sym_convert = AlpacaExecutor.to_trading_sym("BTC/USD")
    assert sym_convert == "BTCUSD", f"symbol conversion failed: {sym_convert}"
    passed += 1

    return passed


if __name__ == "__main__":
    import os
    key = os.environ.get("ALPACA_KEY", "")
    sec = os.environ.get("ALPACA_SECRET", "")
    if not key:
        print("Set ALPACA_KEY and ALPACA_SECRET to run smoke tests")
    else:
        n = _run_tests(key, sec)
        print(f"alpaca_execution: {n}/{n} smoke tests passed.")
        ex = AlpacaExecutor(key, sec, paper=True)
        acc = ex.get_account()
        print(f"Paper account buying power: ${float(acc.get('buying_power',0)):,.2f}")
        positions = ex.get_positions()
        print(f"Open positions: {len(positions)}")
        for p in positions:
            print(f"  {p.symbol}: qty={p.qty:.4f} entry=${p.avg_entry:.2f} pnl=${p.unrealized_pnl:.2f}")
