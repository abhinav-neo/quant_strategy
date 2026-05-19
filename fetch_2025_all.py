#!/usr/bin/env python3
r"""
fetch_2025_all.py
==================
Fetch 2025 data for all 9 Alpaca crypto symbols already cached for 2024.
Skips symbols that are already cached. Saves to data/2025/.

Run:
    set PYTHONPATH=E:\MyDevelopment\GitHub\quant_strategy
    set PYTHONUSERBASE=C:\Users\abhin\AppData\Roaming\Python
    set ALPACA_KEY=your_paper_key
    set ALPACA_SECRET=your_paper_secret
    python fetch_2025_all.py
"""
import os, sys
from datetime import datetime, timezone

sys.path.insert(0, r"E:\MyDevelopment\GitHub\quant_strategy")
import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("fetch_2025")

# Symbols already cached from Phase 0 (2024 data)
SYMBOLS = [
    "BTC/USD", "ETH/USD",
    "LTC/USD", "BCH/USD", "AVAX/USD",
    "LINK/USD", "UNI/USD", "DOGE/USD", "AAVE/USD",
]

# Full year of 2025
START = datetime(2025, 1, 1, tzinfo=timezone.utc)
END   = datetime(2025, 12, 31, tzinfo=timezone.utc)

OUT_DIR = r"E:\MyDevelopment\GitHub\quant_strategy\data\2025"

def main():
    key = os.environ.get("ALPACA_KEY", "")
    sec = os.environ.get("ALPACA_SECRET", "")
    if not key or not sec:
        log.error("ALPACA_KEY and ALPACA_SECRET environment variables required")
        log.error("Run from terminal with:")
        log.error("    set ALPACA_KEY=your_key")
        log.error("    set ALPACA_SECRET=your_secret")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Avoid circular import warnings — only import data_fetcher here
    from modules.data_fetcher import AlpacaDataFetcher
    fetcher = AlpacaDataFetcher(key, sec)

    fetched = 0
    skipped = 0
    failed  = 0

    log.info("=" * 60)
    log.info(f"Fetching {len(SYMBOLS)} symbols for {START.date()} to {END.date()}")
    log.info(f"Output:  {OUT_DIR}")
    log.info("=" * 60)

    for i, sym in enumerate(SYMBOLS, 1):
        fname = sym.replace("/", "_") + "_1min.csv"
        fpath = os.path.join(OUT_DIR, fname)

        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            log.info(f"[{i}/{len(SYMBOLS)}] [skip] {sym}: already cached ({size_mb:.1f} MB)")
            skipped += 1
            continue

        log.info(f"[{i}/{len(SYMBOLS)}] [fetch] {sym}: requesting from Alpaca...")
        try:
            r = fetcher.fetch_bars(sym, START, END)
            if r.ok and r.n_valid > 1000:
                df = r.bars[r.bars["valid_bar"]].copy()
                df.to_csv(fpath)
                log.info(f"[{i}/{len(SYMBOLS)}] [ok]   {sym}: "
                         f"{len(df):,} bars  ->  {fname}")
                fetched += 1
            else:
                log.warning(f"[{i}/{len(SYMBOLS)}] [fail] {sym}: "
                            f"ok={r.ok} n_valid={r.n_valid} "
                            f"err='{r.error_message[:80] if r.error_message else ''}'")
                failed += 1
        except Exception as e:
            log.error(f"[{i}/{len(SYMBOLS)}] [error] {sym}: {e}")
            failed += 1

    log.info("=" * 60)
    log.info(f"DONE: {fetched} fetched, {skipped} skipped, {failed} failed")
    log.info("=" * 60)

    if fetched + skipped >= 6:
        print("\nNext step: validate 7 pairs on 2025 data with:")
        print("    python weekend1_test_pairs_2025.py")
    else:
        print("\nNot enough symbols cached. Re-run or check API key.")


if __name__ == "__main__":
    main()
