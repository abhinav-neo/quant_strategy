# Parallel Track v2 — Time-Boxed Plan
## Goal: 10-15% annual portfolio return while EB1 is primary focus

## Time budget: MAX 6 weekends, 8 hours each = 48 hours total
## If exceeded, stop and revisit priority

---

## Weekend 1: Validate 7 new pairs (8 hours)
**Goal:** Run existing backtest on all 7 loose-cointegration pairs.

**Pairs to test:**
1. BCH/DOGE   (strongest signal, EG p=0.0007)
2. ETH/UNI    (EG p=0.084)
3. LTC/UNI    (EG p=0.129)
4. BCH/BTC    (EG p=0.197)
5. AVAX/LINK  (EG p=0.210)
6. DOGE/LTC   (EG p=0.225)
7. BTC/DOGE   (EG p=0.244)

**Tasks:**
- Update `run_backtest.py` PAIRS list (10 minutes)
- Run on cached 2024 data (~30 min compute)
- Run on cached 2025 data if you fetched it (~30 min compute)
- Document each pair's standalone return + max drawdown

**Decision gate:** If at least 3 of 7 show > 5% annual return in BOTH years
   → continue to Weekend 2
   → otherwise stop, this ceiling is real

---

## Weekend 2: Portfolio backtest (8 hours)
**Goal:** Combine all profitable pairs into one portfolio.

**New module: `run_portfolio_backtest.py`**
- Run all pairs simultaneously
- Cap simultaneous open trades at 3 (capital constraint)
- Compute portfolio Sharpe, max drawdown, correlation matrix
- Output: actual portfolio return, not sum of individual returns

**Decision gate:** Portfolio Sharpe > 1.0 with realistic costs
   → continue to Weekend 3
   → otherwise stop

---

## Weekend 3: Cost realism + slippage validation (8 hours)
**Goal:** Stress test against worst case.

**Tasks:**
- Verify current 0.40% cost assumption against Alpaca docs
- Add 0.05% slippage per side (5 bps adverse fill at 3-sigma entries)
- Run portfolio backtest at 0.50% RT cost
- If still profitable, you have margin of safety

**Decision gate:** Portfolio still > 5% annualized at 0.50% cost
   → continue to Weekend 4
   → otherwise stop

---

## Weekend 4: Paper trading setup (8 hours)
**Goal:** Build live execution skeleton — paper only.

**New module: `run_paper_live.py`**
- Connects to Alpaca paper trading API
- Polls every 60 seconds during crypto hours
- Maintains pair states (entry, exit, P&L) in JSON file
- Logs every decision to file (not just trades)

**No money at risk. Paper account only.**

---

## Weekend 5: 4-week paper trading observation (passive — 0 hours)
**Goal:** Let it run. Check daily P&L vs backtest expectation.

**During this weekend:** EB1 only. Don't touch trading code.

**Daily check (5 min each):**
- Did the strategy fire today?
- Was the trade in line with backtest expectation?
- Any errors in the log?

**Kill switch criteria:**
- Drawdown > 8% → stop and investigate
- 3 consecutive days of unexpected behavior → stop
- Win rate < 60% over first 20 trades → stop

---

## Weekend 6: Decision point (8 hours)
**Goal:** Based on paper trading data, decide go/no-go.

**Go criteria (ALL must be met):**
- Win rate matches backtest within 10%
- Live Sharpe > 1.0
- No catastrophic single trades (>2% capital loss)
- You feel comfortable deploying real $5k for next 4 weeks

**If GO:** Move 10% of trading capital to real money. Continue weekly review.
**If NO-GO:** Stop. Write Medium article on the journey. Move on.

---

## What this is NOT
- This is NOT a path to retirement money
- This is NOT worth >48 hours total
- This is NOT the v2.0 enhancement plan (skip that)
- This is NOT a substitute for index funds and 401k

## What this IS
- A bounded experiment with a clean exit at each weekend
- A way to actually deploy your existing infrastructure
- A source of Medium content on real quant work
- A learning vehicle that doesn't compromise EB1

---

## Stopping rules — be honest
Stop if ANY of these happen:
- Phase results contradict the plan (below threshold)
- EB1 work is impacted
- You spend 50+ hours and aren't paper trading yet
- You feel anxious about it — that means it has too much priority
