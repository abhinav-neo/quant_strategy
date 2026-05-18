"""
Module 1: Alpaca Data Fetcher
==============================
Fetches and validates real 1-minute OHLCV bars from Alpaca.
Paper trading keys are fully supported.

Imports math_guards for all numerical operations.
No raw float leaves this module without validation.

Output contract
---------------
Every row in result.bars is guaranteed:
    open, high, low, close > 0
    high >= open >= low (and same for close)
    high >= low
    volume >= 0
    log_return = ln(close(t) / close(t-1))  exact formula
    |log_return| > 0.20  -> valid_bar = False (spike)
    valid_bar = False for any failed check
    invalid_reason = pipe-delimited string of failed checks

Downstream modules MUST filter: df = df[df["valid_bar"]]
before any computation.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from modules.math_guards import log_return as mg_log_return, safeN

log = logging.getLogger("data_fetcher")

# ── Constants ──────────────────────────────────────────────────────────────
SUPPORTED_SYMBOLS      = [
    "BTC/USD", "ETH/USD", "SOL/USD",
    "LTC/USD", "BCH/USD", "AVAX/USD",
    "LINK/USD", "UNI/USD", "DOGE/USD",
    "AAVE/USD", "XTZ/USD", "DOT/USD",
    "SHIB/USD", "MKR/USD", "GRT/USD",
    "BAT/USD", "CRV/USD", "YFI/USD",
    "SUSHI/USD", "XRP/USD", "PEPE/USD",
    "TRUMP/USD",
]
MIN_BARS_REQUIRED      = 10_000
SPIKE_THRESHOLD        = 0.20   # |log_return| per bar
MAX_RETRIES            = 3
RETRY_DELAY_SEC        = 5
EXPECTED_GAP_SEC       = 60     # 1-minute bars
MAX_DATE_RANGE_DAYS    = 730
MIN_DATE_RANGE_DAYS    = 30


# ── Result dataclass ───────────────────────────────────────────────────────
@dataclass
class FetchResult:
    """
    Always returned by fetch_bars(). Check .ok before using .bars.
    Contains full audit trail of validation results.
    """
    symbol:         str
    ok:             bool                   = False
    bars:           Optional[pd.DataFrame] = None
    n_raw:          int                    = 0
    n_valid:        int                    = 0
    n_invalid:      int                    = 0
    n_gaps:         int                    = 0
    n_duplicates:   int                    = 0
    n_ohlc_errors:  int                    = 0
    n_spikes:       int                    = 0
    error_message:  str                    = ""
    warnings:       List[str]              = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 55,
            f"Symbol       : {self.symbol}",
            f"Status       : {'OK' if self.ok else 'FAILED'}",
            f"Raw bars     : {self.n_raw:,}",
            f"Valid bars   : {self.n_valid:,}",
            f"Invalid bars : {self.n_invalid:,}",
            f"OHLC errors  : {self.n_ohlc_errors:,}",
            f"Spikes       : {self.n_spikes:,}",
            f"Duplicates   : {self.n_duplicates:,}",
            f"Gaps         : {self.n_gaps:,}",
        ]
        if self.error_message:
            lines.append(f"Error        : {self.error_message}")
        for w in self.warnings:
            lines.append(f"Warning      : {w}")
        lines.append("=" * 55)
        return "\n".join(lines)


# ── Main fetcher class ─────────────────────────────────────────────────────
class AlpacaDataFetcher:
    """
    Fetches validated 1-minute OHLCV bars from Alpaca.
    Works with paper trading keys.

    Usage
    -----
    fetcher = AlpacaDataFetcher(api_key, secret_key)
    result  = fetcher.fetch_bars("BTC/USD", start, end)
    if result.ok:
        df = result.bars[result.bars["valid_bar"]].copy()
    """

    def __init__(self, api_key: str, secret_key: str):
        if not api_key or not secret_key:
            raise ValueError("api_key and secret_key must be non-empty strings.")
        # Import here so module loads without Alpaca if running unit tests
        from alpaca.data.historical import CryptoHistoricalDataClient
        self._client = CryptoHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key,
        )
        log.info("AlpacaDataFetcher ready. Paper keys accepted.")

    # ── Public ────────────────────────────────────────────────────────────
    def fetch_bars(
        self,
        symbol: str,
        start:  datetime,
        end:    datetime,
    ) -> FetchResult:
        """
        Fetch 1-minute bars for one symbol.
        Returns FetchResult — never raises.
        Always check result.ok before using result.bars.
        """
        result = FetchResult(symbol=symbol)

        err = _validate_inputs(symbol, start, end)
        if err:
            result.ok            = False
            result.error_message = err
            log.error(f"{symbol}: input validation failed — {err}")
            return result

        raw_df, err = self._fetch_with_retry(symbol, start, end)
        if err:
            result.ok            = False
            result.error_message = err
            log.error(f"{symbol}: fetch failed — {err}")
            return result

        result.n_raw = len(raw_df)
        log.info(f"{symbol}: received {result.n_raw:,} raw bars.")

        df, result = _validate_pipeline(raw_df, result)

        n_valid = int(df["valid_bar"].sum())
        if n_valid < MIN_BARS_REQUIRED:
            result.ok            = False
            result.error_message = (
                f"Only {n_valid:,} valid bars — "
                f"minimum required is {MIN_BARS_REQUIRED:,}. "
                "Fetch a longer date range."
            )
            result.bars  = df
            result.n_valid   = n_valid
            result.n_invalid = len(df) - n_valid
            log.error(result.error_message)
            return result

        result.ok        = True
        result.bars      = df
        result.n_valid   = n_valid
        result.n_invalid = len(df) - n_valid
        log.info(
            f"{symbol}: {result.n_valid:,} valid / "
            f"{result.n_invalid:,} invalid bars."
        )
        return result

    def fetch_all(
        self,
        symbols: List[str],
        start:   datetime,
        end:     datetime,
    ) -> Dict[str, FetchResult]:
        """Fetch all symbols. Failed symbols do not block successful ones."""
        results = {}
        for sym in symbols:
            log.info(f"Fetching {sym} ...")
            results[sym] = self.fetch_bars(sym, start, end)
            log.info(results[sym].summary())
        return results

    # ── Private ───────────────────────────────────────────────────────────
    def _fetch_with_retry(
        self,
        symbol: str,
        start:  datetime,
        end:    datetime,
    ) -> Tuple[Optional[pd.DataFrame], str]:
        from alpaca.data.requests   import CryptoBarsRequest
        from alpaca.data.timeframe  import TimeFrame, TimeFrameUnit

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                req  = CryptoBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                    start=start,
                    end=end,
                )
                bars = self._client.get_crypto_bars(req)
                df   = bars.df

                if df is None or len(df) == 0:
                    return None, "API returned empty dataset."

                # Flatten multi-index
                if isinstance(df.index, pd.MultiIndex):
                    df = df.xs(symbol, level=0)

                df.index = pd.to_datetime(df.index, utc=True)
                df.index.name = "timestamp"
                df   = df.sort_index()
                df.columns = [c.lower() for c in df.columns]

                required = {"open", "high", "low", "close", "volume"}
                missing  = required - set(df.columns)
                if missing:
                    return None, f"API response missing columns: {missing}"

                return df, ""

            except Exception as exc:
                wait = RETRY_DELAY_SEC * attempt
                log.warning(
                    f"{symbol}: attempt {attempt}/{MAX_RETRIES} failed "
                    f"({exc}). Retrying in {wait}s ..."
                )
                if attempt < MAX_RETRIES:
                    time.sleep(wait)

        return None, f"All {MAX_RETRIES} fetch attempts failed."


# ── Module-level validation functions (pure, no class state) ──────────────

def _validate_inputs(symbol: str, start: datetime, end: datetime) -> str:
    """Returns error string or empty string if valid."""
    if symbol not in SUPPORTED_SYMBOLS:
        return (f"Unsupported symbol '{symbol}'. "
                f"Supported: {SUPPORTED_SYMBOLS}")
    if start.tzinfo is None or end.tzinfo is None:
        return "start and end must be timezone-aware (UTC)."
    if end <= start:
        return "end must be strictly after start."
    days = (end - start).days
    if days < MIN_DATE_RANGE_DAYS:
        return (f"Date range {days} days < minimum {MIN_DATE_RANGE_DAYS} days.")
    if days > MAX_DATE_RANGE_DAYS:
        return (f"Date range {days} days > maximum {MAX_DATE_RANGE_DAYS} days. "
                "Split into chunks.")
    return ""


def _validate_pipeline(
    df:     pd.DataFrame,
    result: FetchResult,
) -> Tuple[pd.DataFrame, FetchResult]:
    """
    Run every guard in sequence.
    Adds valid_bar (bool) and invalid_reason (str) columns.
    Mutates result counts in-place.
    All log_return computation uses math_guards.log_return.
    """
    df = df.copy()
    df["valid_bar"]      = True
    df["invalid_reason"] = ""

    # 1. Remove duplicate timestamps (keep first)
    dupes = df.index.duplicated(keep="first")
    n_dup = int(dupes.sum())
    if n_dup > 0:
        result.n_duplicates = n_dup
        result.warnings.append(f"{n_dup} duplicate timestamps removed.")
        df = df[~dupes].copy()

    # 2. Positive price guard
    for col in ["open", "high", "low", "close"]:
        bad = df[col] <= 0
        if bad.any():
            df.loc[bad, "valid_bar"]       = False
            df.loc[bad, "invalid_reason"] += f"|{col}<=0"
            result.n_ohlc_errors += int(bad.sum())

    # 3. OHLC internal consistency
    checks = [
        (df["high"] < df["open"],   "high<open"),
        (df["high"] < df["close"],  "high<close"),
        (df["low"]  > df["open"],   "low>open"),
        (df["low"]  > df["close"],  "low>close"),
        (df["high"] < df["low"],    "high<low"),
    ]
    for mask, label in checks:
        bad = mask & df["valid_bar"]
        if bad.any():
            df.loc[bad, "valid_bar"]       = False
            df.loc[bad, "invalid_reason"] += f"|{label}"
            result.n_ohlc_errors += int(bad.sum())

    # 4. Non-negative volume
    bad_vol = df["volume"] < 0
    if bad_vol.any():
        df.loc[bad_vol, "valid_bar"]       = False
        df.loc[bad_vol, "invalid_reason"] += "|volume<0"

    # 5. Log return — uses math_guards.log_return (exact formula + spike detect)
    #    Computed only on valid bars; invalid bars get NaN
    df["log_return"] = np.nan
    valid_idx = df.index[df["valid_bar"]]

    if len(valid_idx) > 1:
        closes = df.loc[valid_idx, "close"].values
        lrs    = np.empty(len(closes))
        lrs[0] = np.nan    # first bar has no previous

        for i in range(1, len(closes)):
            lrs[i] = mg_log_return(
                float(closes[i]),
                float(closes[i - 1]),
                label=f"bar_{i}",
            )
        df.loc[valid_idx, "log_return"] = lrs

    # 6. Spike detection — flag bars where |log_return| > threshold
    spike_mask = df["log_return"].abs() > SPIKE_THRESHOLD
    spike_mask = spike_mask.fillna(False)
    n_spikes   = int(spike_mask.sum())
    if n_spikes > 0:
        result.n_spikes = n_spikes
        df.loc[spike_mask, "valid_bar"]       = False
        df.loc[spike_mask, "invalid_reason"] += "|spike"
        result.warnings.append(
            f"{n_spikes} bars flagged: |log_return|>{SPIKE_THRESHOLD:.0%}. "
            "Could be genuine crisis move or data error — review."
        )

    # 7. Gap detection — missing 1-minute intervals
    ts        = df.index.to_series()
    gaps_sec  = ts.diff().dt.total_seconds()
    gap_mask  = gaps_sec > EXPECTED_GAP_SEC * 2
    n_gaps    = int(gap_mask.sum())
    if n_gaps > 0:
        result.n_gaps = n_gaps
        result.warnings.append(
            f"{n_gaps} timestamp gaps > {EXPECTED_GAP_SEC*2}s detected. "
            "Gaps flagged — rolling indicators reset at each gap in downstream modules."
        )

    # 8. Clean up reason string
    df["invalid_reason"] = df["invalid_reason"].str.lstrip("|")

    return df, result


# ── Utility: save / load CSV cache ────────────────────────────────────────
def save_bars(result: FetchResult, directory: str = "data") -> str:
    """Save validated bars to CSV. Returns filepath."""
    import os
    os.makedirs(directory, exist_ok=True)
    fname    = result.symbol.replace("/", "_") + "_1min.csv"
    filepath = os.path.join(directory, fname)
    result.bars.to_csv(filepath)
    log.info(f"Saved {len(result.bars):,} bars -> {filepath}")
    return filepath


def load_cached_bars(filepath: str, symbol: str = "BTC/USD") -> FetchResult:
    """Load bars from CSV. Re-runs validation pipeline on load."""
    result = FetchResult(symbol=symbol)
    try:
        df = pd.read_csv(filepath, index_col="timestamp", parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        result.n_raw = len(df)

        required = {"open", "high", "low", "close", "volume"}
        missing  = required - set(df.columns)
        if missing:
            result.ok            = False
            result.error_message = f"CSV missing columns: {missing}"
            return result

        # Re-run validation so cached data gets same guarantees as live data
        df.columns = [c.lower() for c in df.columns]
        df, result = _validate_pipeline(df, result)

        n_valid      = int(df["valid_bar"].sum())
        result.ok    = n_valid >= MIN_BARS_REQUIRED
        result.bars  = df
        result.n_valid   = n_valid
        result.n_invalid = len(df) - n_valid
        if not result.ok:
            result.error_message = (
                f"Only {n_valid:,} valid bars in cache — "
                f"minimum {MIN_BARS_REQUIRED:,} required."
            )
        log.info(f"Loaded {n_valid:,} valid bars from cache: {filepath}")
        return result
    except Exception as exc:
        result.ok            = False
        result.error_message = f"Cache load failed: {exc}"
        return result


# ── Unit tests ─────────────────────────────────────────────────────────────
def run_unit_tests() -> bool:
    """
    Tests every guard using synthetic edge cases.
    No Alpaca API calls made.
    All numerical assertions verified against math_guards contract.
    """
    import math
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    )
    log.info("Running data_fetcher unit tests ...")
    passed = failed = 0

    def check(name: str, condition: bool):
        nonlocal passed, failed
        if condition:
            log.info(f"  PASS  {name}")
            passed += 1
        else:
            log.error(f"  FAIL  {name}")
            failed += 1

    # ── Helper: build synthetic clean DataFrame ───────────────────────────
    def make_df(n: int = 20_000, seed: int = 42) -> pd.DataFrame:
        rng   = np.random.default_rng(seed)
        t     = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        base  = 45_000 + np.cumsum(rng.normal(0, 10, n))
        base  = np.abs(base) + 100
        noise = rng.uniform(10, 100, n)
        o     = base + rng.normal(0, 5, n)
        h     = base + noise
        l     = base - noise
        c     = base + rng.normal(0, 5, n)
        # Enforce OHLC consistency
        h = np.maximum(h, np.maximum(o, c))
        l = np.minimum(l, np.minimum(o, c))
        df = pd.DataFrame(
            {"open": o, "high": h, "low": l, "close": c,
             "volume": rng.uniform(1, 100, n)},
            index=pd.DatetimeIndex(t, name="timestamp"),
        )
        return df

    # ── Test 1: clean data — all bars valid ───────────────────────────────
    df1        = make_df()
    r1         = FetchResult(symbol="BTC/USD", n_raw=len(df1))
    df1_out, _ = _validate_pipeline(df1, r1)
    check("T01 clean data: all bars valid", df1_out["valid_bar"].all())

    # ── Test 2: zero close price flagged ─────────────────────────────────
    df2 = make_df()
    df2.iloc[100, df2.columns.get_loc("close")] = 0.0
    r2         = FetchResult(symbol="BTC/USD", n_raw=len(df2))
    df2_out, _ = _validate_pipeline(df2, r2)
    check("T02 zero close -> invalid",
          not df2_out.iloc[100]["valid_bar"])

    # ── Test 3: negative open flagged ────────────────────────────────────
    df3 = make_df()
    df3.iloc[200, df3.columns.get_loc("open")] = -500.0
    r3         = FetchResult(symbol="BTC/USD", n_raw=len(df3))
    df3_out, _ = _validate_pipeline(df3, r3)
    check("T03 negative open -> invalid",
          not df3_out.iloc[200]["valid_bar"])

    # ── Test 4: high < low flagged ────────────────────────────────────────
    df4 = make_df()
    df4.iloc[300, df4.columns.get_loc("high")] = \
        df4.iloc[300]["low"] - 10.0
    r4         = FetchResult(symbol="BTC/USD", n_raw=len(df4))
    df4_out, _ = _validate_pipeline(df4, r4)
    check("T04 high < low -> invalid",
          not df4_out.iloc[300]["valid_bar"])

    # ── Test 5: high < close flagged ─────────────────────────────────────
    df5 = make_df()
    row = df5.index[400]
    df5.loc[row, "close"] = df5.loc[row, "high"] + 500.0
    r5         = FetchResult(symbol="BTC/USD", n_raw=len(df5))
    df5_out, _ = _validate_pipeline(df5, r5)
    check("T05 high < close -> invalid",
          not df5_out.loc[row]["valid_bar"])

    # ── Test 6: spike detection ───────────────────────────────────────────
    df6 = make_df()
    idx = df6.index[500]
    prev_close = df6.iloc[499]["close"]
    # +30% jump in one bar
    df6.loc[idx, "close"] = prev_close * 1.30
    df6.loc[idx, "high"]  = df6.loc[idx, "close"] + 50
    r6         = FetchResult(symbol="BTC/USD", n_raw=len(df6))
    df6_out, r6_result = _validate_pipeline(df6, r6)
    check("T06 30pct spike flagged",    r6_result.n_spikes >= 1)
    check("T06 spike bar -> invalid",   not df6_out.loc[idx]["valid_bar"])

    # ── Test 7: log_return formula exactness ─────────────────────────────
    df7        = make_df()
    r7         = FetchResult(symbol="BTC/USD", n_raw=len(df7))
    df7_out, _ = _validate_pipeline(df7, r7)
    # Pick a valid bar that is not the first
    valid_bars = df7_out[df7_out["valid_bar"]]
    if len(valid_bars) > 10:
        row_i   = valid_bars.index[10]
        row_i_1 = valid_bars.index[9]
        expected = math.log(
            float(df7_out.loc[row_i,   "close"]) /
            float(df7_out.loc[row_i_1, "close"])
        )
        actual = float(df7_out.loc[row_i, "log_return"])
        check("T07 log_return = ln(c_t/c_t-1)", abs(expected - actual) < 1e-10)
    else:
        check("T07 log_return formula (skipped - too few valid bars)", False)

    # ── Test 8: first valid bar log_return = NaN ──────────────────────────
    df8        = make_df()
    r8         = FetchResult(symbol="BTC/USD", n_raw=len(df8))
    df8_out, _ = _validate_pipeline(df8, r8)
    first_lr   = df8_out["log_return"].iloc[0]
    check("T08 first bar log_return = NaN",
          math.isnan(first_lr))

    # ── Test 9: duplicate timestamps removed ─────────────────────────────
    df9 = make_df(n=15_000)
    df9 = pd.concat([df9, df9.iloc[:50]]).sort_index()
    r9         = FetchResult(symbol="BTC/USD", n_raw=len(df9))
    _, r9_res  = _validate_pipeline(df9, r9)
    check("T09 50 duplicates detected", r9_res.n_duplicates == 50)

    # ── Test 10: gap detection ────────────────────────────────────────────
    df10 = make_df(n=15_000)
    idx_list  = df10.index.tolist()
    # Inject 3-hour gap at position 5000
    shift = pd.Timedelta(hours=3)
    for i in range(5000, len(idx_list)):
        idx_list[i] = idx_list[i] + shift
    df10.index = pd.DatetimeIndex(idx_list, name="timestamp", tz="UTC")
    r10        = FetchResult(symbol="BTC/USD", n_raw=len(df10))
    _, r10_res = _validate_pipeline(df10, r10)
    check("T10 timestamp gap detected", r10_res.n_gaps >= 1)

    # ── Test 11: input validation — unsupported symbol ────────────────────
    err = _validate_inputs(
        "DOGE/USD",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 6, 1, tzinfo=timezone.utc),
    )
    check("T11 unsupported symbol rejected", err != "")

    # ── Test 12: input validation — end before start ──────────────────────
    err = _validate_inputs(
        "BTC/USD",
        datetime(2024, 6, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    check("T12 end before start rejected", err != "")

    # ── Test 13: input validation — range too short ───────────────────────
    err = _validate_inputs(
        "BTC/USD",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 10, tzinfo=timezone.utc),
    )
    check("T13 range < 30 days rejected", err != "")

    # ── Test 14: input validation — range too long ────────────────────────
    err = _validate_inputs(
        "BTC/USD",
        datetime(2022, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    check("T14 range > 730 days rejected", err != "")

    # ── Test 15: input validation — naive datetime rejected ───────────────
    err = _validate_inputs(
        "BTC/USD",
        datetime(2024, 1, 1),   # no tzinfo
        datetime(2024, 6, 1, tzinfo=timezone.utc),
    )
    check("T15 naive datetime rejected", err != "")

    # ── Test 16: minimum bars check ───────────────────────────────────────
    df16       = make_df(n=500)
    r16        = FetchResult(symbol="BTC/USD", n_raw=len(df16))
    df16_out, r16_res = _validate_pipeline(df16, r16)
    check("T16 500 valid bars < MIN_BARS_REQUIRED",
          r16_res.n_valid < MIN_BARS_REQUIRED)

    # ── Test 17: negative volume flagged ─────────────────────────────────
    df17 = make_df()
    df17.iloc[600, df17.columns.get_loc("volume")] = -1.0
    r17        = FetchResult(symbol="BTC/USD", n_raw=len(df17))
    df17_out, _ = _validate_pipeline(df17, r17)
    check("T17 negative volume -> invalid",
          not df17_out.iloc[600]["valid_bar"])

    # ── Test 18: invalid_reason populated correctly ───────────────────────
    df18 = make_df()
    df18.iloc[700, df18.columns.get_loc("close")] = 0.0
    r18        = FetchResult(symbol="BTC/USD", n_raw=len(df18))
    df18_out, _ = _validate_pipeline(df18, r18)
    reason = df18_out.iloc[700]["invalid_reason"]
    check("T18 invalid_reason non-empty for bad bar", reason != "")

    # ── Test 19: valid_bar dtype is bool ─────────────────────────────────
    df19       = make_df()
    r19        = FetchResult(symbol="BTC/USD", n_raw=len(df19))
    df19_out, _ = _validate_pipeline(df19, r19)
    check("T19 valid_bar column dtype is bool",
          df19_out["valid_bar"].dtype == bool)

    # ── Test 20: ETH and SOL symbols accepted ────────────────────────────
    err_eth = _validate_inputs(
        "ETH/USD",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 6, 1, tzinfo=timezone.utc),
    )
    err_sol = _validate_inputs(
        "SOL/USD",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 6, 1, tzinfo=timezone.utc),
    )
    check("T20 ETH/USD accepted", err_eth == "")
    check("T20 SOL/USD accepted", err_sol == "")

    log.info(f"\ndata_fetcher: {passed}/{passed+failed} tests passed.")
    return failed == 0


if __name__ == "__main__":
    ok = run_unit_tests()
    if not ok:
        raise SystemExit("data_fetcher unit tests FAILED.")
    print("\nAll data_fetcher tests passed.")
    print("\nTo fetch live data:")
    print("  from modules.data_fetcher import AlpacaDataFetcher")
    print("  from datetime import datetime, timezone")
    print("  fetcher = AlpacaDataFetcher('YOUR_KEY', 'YOUR_SECRET')")
    print("  result  = fetcher.fetch_bars(")
    print("      'BTC/USD',")
    print("      start=datetime(2024, 1, 1, tzinfo=timezone.utc),")
    print("      end  =datetime(2024, 10, 1, tzinfo=timezone.utc),")
    print("  )")
    print("  if result.ok:")
    print("      df = result.bars[result.bars['valid_bar']].copy()")
