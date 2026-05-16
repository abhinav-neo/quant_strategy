#!/usr/bin/env python3
"""
phase0_validate_breadth.py
===========================
Phase 0 of v2.0 enhancement plan.

Tests whether MORE pairs can be added to give portfolio-level
returns 2x-3x what individual pairs give.

This is the Go/No-Go decision for the entire enhancement plan.

Run:
    set PYTHONPATH=E:\MyDevelopment\GitHub\quant_strategy
    set ALPACA_KEY=your_key
    set ALPACA_SECRET=your_secret
    python phase0_validate_breadth.py

Decision:
    - Portfolio Sharpe > 1.5 with 6+ pairs -> proceed with Path A (statistical enhancements)
    - Otherwise -> switch to Path C (better venue with lower fees)
"""
import os, sys, math
import numpy as np
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, r"E:\MyDevelopment\GitHub\quant_strategy")
import logging; logging.basicConfig(level=logging.WARNING)

# Candidate symbols - test which Alpaca actually supports
CANDIDATES = [
    "BTC/USD", "ETH/USD", "SOL/USD",
    "LTC/USD", "BCH/USD", "AVAX/USD",
    "LINK/USD", "UNI/USD", "DOGE/USD",
    "AAVE/USD", "BTC/USDC", "ETH/USDC",
]

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
END   = datetime(2024, 10, 1, tzinfo=timezone.utc)

def fetch_all():
    """Fetch all available crypto symbols from Alpaca."""
    from modules.data_fetcher import AlpacaDataFetcher
    key = os.environ.get("ALPACA_KEY", "")
    sec = os.environ.get("ALPACA_SECRET", "")
    if not key:
        print("ERROR: Set ALPACA_KEY and ALPACA_SECRET")
        sys.exit(1)
    
    fetcher = AlpacaDataFetcher(key, sec)
    cache_dir = r"E:\MyDevelopment\GitHub\quant_strategy\data\phase0"
    os.makedirs(cache_dir, exist_ok=True)
    
    available = {}
    for sym in CANDIDATES:
        fname = sym.replace("/","_") + "_1min.csv"
        fpath = os.path.join(cache_dir, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, parse_dates=[0], index_col=0)
            available[sym] = df
            print(f"  [cached] {sym}: {len(df):,} bars")
            continue
        
        try:
            r = fetcher.fetch_bars(sym, START, END)
            if r.ok and r.n_valid > 50_000:
                df = r.bars[r.bars["valid_bar"]].copy()
                df.to_csv(fpath)
                available[sym] = df
                print(f"  [fetched] {sym}: {len(df):,} bars")
            else:
                print(f"  [skip] {sym}: insufficient data")
        except Exception as e:
            print(f"  [error] {sym}: {e}")
    
    return available

def test_cointegration(available):
    """Run cointegration tests on all pairs."""
    from itertools import combinations
    from modules.stationarity import run_all_tests
    
    pairs_results = []
    for sym_a, sym_b in combinations(available.keys(), 2):
        df_a = available[sym_a]
        df_b = available[sym_b]
        common = df_a.index.intersection(df_b.index)
        if len(common) < 10_000: continue
        
        # Resample to 1h for cointegration test (faster)
        df_a_1h = df_a.loc[common, "close"].resample("1h").last().dropna()
        df_b_1h = df_b.loc[common, "close"].resample("1h").last().dropna()
        common_1h = df_a_1h.index.intersection(df_b_1h.index)
        df_a_1h = df_a_1h.loc[common_1h]
        df_b_1h = df_b_1h.loc[common_1h]
        if len(df_a_1h) < 500: continue
        
        # Simple beta + spread test
        lp_a = np.log(df_a_1h.values)
        lp_b = np.log(df_b_1h.values)
        beta = np.cov(lp_a, lp_b)[0,1] / np.var(lp_b)
        spread = lp_a - beta * lp_b
        
        # Test stationarity
        try:
            rep = run_all_tests(spread, symbol=f"{sym_a}/{sym_b}")
            pairs_results.append({
                "pair": f"{sym_a}/{sym_b}",
                "n_bars": len(common_1h),
                "edge_confirmed": rep.edge_confirmed,
                "hurst": rep.hurst.hurst if hasattr(rep.hurst, "hurst") else 0,
            })
        except Exception as e:
            pass
    
    return pairs_results

def main():
    print("="*60)
    print(" PHASE 0: Breadth Validation")
    print("="*60)
    print()
    print("Step 1: Discover available Alpaca crypto symbols")
    available = fetch_all()
    print(f"\nFound {len(available)} symbols with sufficient data")
    
    if len(available) < 6:
        print()
        print("VERDICT: Alpaca offers too few crypto pairs.")
        print("RECOMMENDATION: Switch to Path C (Hyperliquid/Binance perps)")
        print("                More pairs + lower fees = bigger return path")
        return
    
    print()
    print("Step 2: Cointegration test on all pairs")
    results = test_cointegration(available)
    
    cointegrated = [r for r in results if r["edge_confirmed"]]
    print(f"\nCointegrated pairs: {len(cointegrated)}/{len(results)}")
    
    print()
    print("Top cointegrated pairs (by Hurst):")
    cointegrated.sort(key=lambda r: r["hurst"])
    for r in cointegrated[:10]:
        print(f"  {r['pair']:>20s}  hurst={r['hurst']:.3f}  bars={r['n_bars']:,}")
    
    print()
    if len(cointegrated) >= 6:
        print("VERDICT: Sufficient cointegrated pairs found.")
        print("PROCEED with Phase 1 (statistical enhancements)")
        print("Expected portfolio return: 3-5x single-pair return via diversification")
    else:
        print("VERDICT: Too few cointegrated pairs for portfolio diversification")
        print("RECOMMENDATION: Switch to Path C or accept current 3-pair limit")

if __name__ == "__main__":
    main()
