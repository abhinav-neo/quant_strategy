r"""
modules/execution_model.py
===========================
Realistic execution cost model for crypto stat-arb on Alpaca.

Key insight: slippage is NOT constant. It depends on:
  1. Z-score at entry: higher z = faster spread move = worse fill
  2. Position size vs average volume: larger trades = more market impact
  3. Venue: Alpaca crypto has maker/taker fees of 0.15%/0.25%

Usage:
    from modules.execution_model import ExecutionCostModel
    model = ExecutionCostModel(venue="alpaca_crypto")
    cost_frac = model.total_cost_frac(zscore=3.5, pos_usd=1000, avg_vol_usd=500000)
"""

from __future__ import annotations
import numpy as np

# Venue fee structures (round-trip as fractions)
VENUE_FEES = {
    "alpaca_crypto": {
        "maker_rt":  0.0030,   # 0.15% per side x2
        "taker_rt":  0.0050,   # 0.25% per side x2
        "impact_k":  0.0001,   # linear impact coefficient
        "base_slip": 0.0005,   # base adverse fill (5 bps per side)
        "slip_per_sigma": 0.0002,  # additional slip per sigma above 3
    },
    "hyperliquid": {
        "maker_rt":  0.0000,   # -0.005% rebate x2 (effectively free)
        "taker_rt":  0.0009,   # 0.045% per side x2
        "impact_k":  0.00003,
        "base_slip": 0.0001,
        "slip_per_sigma": 0.00005,
    },
    "binance_perps": {
        "maker_rt":  0.0004,   # 0.02% per side x2
        "taker_rt":  0.0010,   # 0.05% per side x2
        "impact_k":  0.00004,
        "base_slip": 0.0001,
        "slip_per_sigma": 0.00005,
    },
}

# Minimum denominator for safe division
_MIN_VOL = 1.0

class ExecutionCostModel:
    r"""
    Estimates realistic round-trip execution cost as a fraction of position.

    Cost components:
    - Maker fee (round-trip): fixed, venue-dependent
    - Adverse fill (slippage): base + z-score-dependent component
      * At z=3.0: minimal extra slip (strategy enters at threshold)
      * At z=6.0: 3 sigma above entry threshold = spread moved fast = worse fill
      * At z=10+: very large moves, significant slippage
    - Market impact: proportional to sqrt(pos_usd / avg_daily_vol_usd)

    All costs are returned as a fraction (e.g. 0.004 = 0.40% of position).
    """

    def __init__(self, venue: str = "alpaca_crypto"):
        if venue not in VENUE_FEES:
            raise ValueError(f"Unknown venue '{venue}'. Supported: {list(VENUE_FEES)}")
        self.venue = venue
        self._fees = VENUE_FEES[venue]

    def adverse_fill_frac(self, zscore: float) -> float:
        """
        Adverse fill increases above z=3 (our entry threshold).
        At z=3: base slip only.
        At z=5: base + 2*slip_per_sigma.
        At z=10: base + 7*slip_per_sigma.
        """
        sigma_above_threshold = max(0.0, abs(zscore) - 3.0)
        return (self._fees["base_slip"]
                + sigma_above_threshold * self._fees["slip_per_sigma"])

    def market_impact_frac(self, pos_usd: float,
                           avg_daily_vol_usd: float = 5_000_000) -> float:
        """
        Linear-in-sqrt market impact.
        Default avg_daily_vol: $5M (conservative for mid-cap crypto on Alpaca).
        """
        ratio = pos_usd / max(avg_daily_vol_usd, _MIN_VOL)
        return self._fees["impact_k"] * np.sqrt(ratio)

    def total_cost_frac(self, zscore: float, pos_usd: float = 1000.0,
                        avg_daily_vol_usd: float = 5_000_000,
                        use_maker: bool = True) -> float:
        """
        Total round-trip cost as fraction of position.

        Args:
            zscore: absolute z-score at entry (positive)
            pos_usd: position size in USD
            avg_daily_vol_usd: average daily volume of asset in USD
            use_maker: True = assume maker fills (limit orders)
        """
        fee = (self._fees["maker_rt"] if use_maker
               else self._fees["taker_rt"])
        slip  = self.adverse_fill_frac(zscore)
        impact = self.market_impact_frac(pos_usd, avg_daily_vol_usd)
        return fee + 2 * slip + impact  # factor 2: slip applies to both entry and exit

    def cost_table(self, zscores=(3.0, 3.5, 4.0, 5.0, 7.0, 10.0),
                   pos_usd: float = 1000.0) -> str:
        """Print a human-readable cost table for this venue."""
        lines = [f"\n  Execution cost table — {self.venue} — pos=${pos_usd:,.0f}"]
        lines.append(f"  {'Entry z':>8} {'Fee RT':>8} {'Slip RT':>8} "
                     f"{'Impact':>8} {'Total RT':>9} {'Net/trade':>10}")
        lines.append("  " + "-"*60)
        for z in zscores:
            fee    = self._fees["maker_rt"]
            slip   = self.adverse_fill_frac(z)
            impact = self.market_impact_frac(pos_usd)
            total  = fee + 2*slip + impact
            net    = pos_usd * total
            lines.append(f"  {z:>8.1f} {fee*100:>7.3f}% {2*slip*100:>7.3f}% "
                         f"{impact*100:>7.4f}% {total*100:>8.3f}% "
                         f"${net:>9.4f}")
        return "\n".join(lines)


# ── self-tests ────────────────────────────────────────────────────────────────
def _run_tests() -> int:
    passed = 0

    model = ExecutionCostModel("alpaca_crypto")

    # 1. Base cost at z=3 (no extra slippage)
    c = model.total_cost_frac(zscore=3.0, pos_usd=1000)
    assert 0.003 < c < 0.006, f"Expected 0.003-0.006 at z=3, got {c:.5f}"
    passed += 1

    # 2. Cost strictly increases with z-score
    c3 = model.total_cost_frac(zscore=3.0)
    c5 = model.total_cost_frac(zscore=5.0)
    c10 = model.total_cost_frac(zscore=10.0)
    assert c3 < c5 < c10, "Cost should increase with z-score"
    passed += 1

    # 3. Hyperliquid cheaper than Alpaca
    alp = ExecutionCostModel("alpaca_crypto").total_cost_frac(3.0)
    hyp = ExecutionCostModel("hyperliquid").total_cost_frac(3.0)
    assert hyp < alp, f"Hyperliquid should be cheaper: hyp={hyp:.5f} alp={alp:.5f}"
    passed += 1

    # 4. Larger positions have higher impact
    c_small = model.market_impact_frac(1_000)
    c_large = model.market_impact_frac(100_000)
    assert c_small < c_large, "Larger positions have higher impact"
    passed += 1

    # 5. Cost table prints without error
    table = model.cost_table()
    assert "alpaca_crypto" in table
    passed += 1

    return passed


if __name__ == "__main__":
    n = _run_tests()
    print(f"execution_model: {n}/{n} tests passed. All execution_model tests passed.")

    # Print cost tables for all venues
    for venue in VENUE_FEES:
        m = ExecutionCostModel(venue)
        print(m.cost_table())
