import os, sys, numpy as np, pandas as pd
sys.path.insert(0, r"E:\MyDevelopment\GitHub\quant_strategy")
import logging; logging.basicConfig(level=logging.WARNING)
from modules.kalman_filter import KalmanFilter, estimate_noise_params
from modules.ou_estimator  import fit_ou_rolling, half_life_scale

ENTRY_Z=3.0; EXIT_Z=0.2; MAX_HOLD=24; MAX_ENTRY_Z=6.0
WARMUP=200; OU_WIN=200; REFIT=100; MAX_OPEN=3; CAPITAL=10_000; N_MONTHS=9

PAIRS=[("BCH","DOGE"),("ETH","UNI"),("LTC","UNI"),("BCH","BTC"),
       ("AVAX","LINK"),("DOGE","LTC"),("BTC","DOGE"),("BTC","ETH")]

DATA={"2024": r"E:\MyDevelopment\GitHub\quant_strategy\data\phase0",
      "2025": r"E:\MyDevelopment\GitHub\quant_strategy\data\2025"}

def load_1h(sym, data_dir):
    p = os.path.join(data_dir, sym+"_USD_1min.csv")
    if not os.path.exists(p): return None
    df = pd.read_csv(p, parse_dates=[0], index_col=0)
    return pd.DataFrame({"close": df["close"].resample("1h").last()}).dropna()


def run_portfolio(year, cost_rt):
    data_dir = DATA[year]
    bars = {}
    for a, b in PAIRS:
        for s in [a, b]:
            if s not in bars:
                df = load_1h(s, data_dir)
                if df is not None: bars[s] = df

    common = None
    for s, df in bars.items():
        common = df.index if common is None else common.intersection(df.index)
    if common is None or len(common) < WARMUP+500: return None

    common = common[:N_MONTHS*30*24]
    for s in bars: bars[s] = bars[s].loc[bars[s].index.intersection(common)]
    n_bars = min(len(df) for df in bars.values())
    for s in bars: bars[s] = bars[s].iloc[:n_bars]
    lp = {s: np.log(bars[s]["close"].values) for s in bars}
    days = n_bars / 24.0

    valid_pairs=[]; kf_s={}; ou_s={}; sp_h={}
    for a, b in PAIRS:
        if a not in lp or b not in lp: continue
        label = a+"/"+b
        q, r = estimate_noise_params(lp[a][:WARMUP], lp[b][:WARMUP])
        kf = KalmanFilter(q_scale=q, r_scale=r)
        spreads=[]
        for i in range(WARMUP):
            st=kf.step(float(lp[a][i]), float(lp[b][i]))
            spreads.append(st.spread)
        ou_list=fit_ou_rolling(np.array(spreads),window=len(spreads)-1,step=len(spreads)-1)
        ou=next((p for p in ou_list if p.is_valid), None)
        if ou is None: continue
        kf_s[label]=kf; ou_s[label]=ou; sp_h[label]=spreads[:]
        valid_pairs.append((a,b,label))

    pos_per=CAPITAL/len(valid_pairs)
    cap=float(CAPITAL); peak=cap; max_dd=0.0
    open_trades={}
    all_trades={lb:[] for _,_,lb in valid_pairs}
    last_refit={lb:WARMUP for _,_,lb in valid_pairs}

    for i in range(WARMUP, n_bars):
        zs={}; sp_now={}
        for a,b,label in valid_pairs:
            st=kf_s[label].step(float(lp[a][i]),float(lp[b][i]))
            sp_h[label].append(st.spread); sp_now[label]=st.spread
            if i-last_refit[label]>=REFIT:
                recent=np.array(sp_h[label][-OU_WIN:])
                nl=fit_ou_rolling(recent,window=len(recent)-1,step=len(recent)-1)
                no=next((p for p in nl if p.is_valid),None)
                if no: ou_s[label]=no
                last_refit[label]=i
            ou=ou_s[label]
            zs[label]=(st.spread-ou.mu)/max(ou.sigma_eq,1e-12)

        for label in list(open_trades.keys()):
            t=open_trades[label]; t["held"]+=1; d=t["dir"]; z=zs[label]
            if (d==1 and z>-EXIT_Z) or (d==-1 and z<EXIT_Z) or t["held"]>=MAX_HOLD:
                raw=d*(sp_now[label]-t["entry_spread"])*pos_per
                net=max(min(raw,pos_per*0.5),-pos_per*0.5)-pos_per*cost_rt
                all_trades[label].append(net); cap+=net
                if cap>peak: peak=cap
                dd=(peak-cap)/peak*100
                if dd>max_dd: max_dd=dd
                del open_trades[label]

        if len(open_trades)<MAX_OPEN:
            for _,_,label in valid_pairs:
                if label in open_trades: continue
                if len(open_trades)>=MAX_OPEN: break
                ou=ou_s[label]
                if not (0.3<=ou.half_life<=240 and half_life_scale(ou)>0): continue
                z=zs[label]
                if abs(z)>MAX_ENTRY_Z: continue
                sig=None
                if z<-ENTRY_Z: sig=1
                elif z>ENTRY_Z: sig=-1
                if sig: open_trades[label]=dict(dir=sig,entry_spread=sp_now[label],held=0)

    all_net=[t for tl in all_trades.values() for t in tl]
    total_ret=(cap-CAPITAL)/CAPITAL*100
    ann_ret=total_ret*(365/days)
    if len(all_net)>1 and np.std(all_net)>0:
        rets=np.array(all_net)/CAPITAL
        sharpe=rets.mean()/rets.std()*np.sqrt(len(rets)*365/days)
    else:
        sharpe=0.0
    wins=sum(1 for t in all_net if t>0)
    return {"year":year,"cost_rt":cost_rt,"days":days,"cap_end":round(cap,2),
            "total_ret":round(total_ret,2),"ann_ret":round(ann_ret,2),
            "max_dd":round(max_dd,3),"sharpe":round(sharpe,3),
            "n_trades":len(all_net),"win_rate":round(wins/len(all_net)*100,1) if all_net else 0,
            "pairs":{lb:{"n":len(tl),"wins":sum(1 for t in tl if t>0),"pnl":round(sum(tl),2)}
                     for lb,tl in all_trades.items() if tl}}


if __name__=="__main__":
    COSTS=[0.0040, 0.0050, 0.0060]
    print("="*72)
    print(" Weekend 3: Stress Test  (9-month window, z<=6 sigma guard)")
    print("="*72)
    print("  Year   Cost  Trades  Win%    Ann%   MaxDD  Sharpe")
    print("  "+"-"*58)

    results={}
    for year in ["2024","2025"]:
        results[year]={}
        for cost in COSTS:
            r=run_portfolio(year,cost)
            results[year][cost]=r
            if r:
                print("  "+year+" "+str(round(cost*100,2))+"% "+
                      str(r["n_trades"])+" trades  "+str(r["win_rate"])+"% wins  "+
                      str(r["ann_ret"])+"% ann  "+str(r["max_dd"])+"% dd  "+
                      str(r["sharpe"])+" sharpe")
        print()

    for year in ["2024","2025"]:
        r=results[year][0.0040]
        if r:
            print("  Per-pair breakdown -- "+year+" @ 0.40% cost:")
            for lb,s in sorted(r["pairs"].items(),key=lambda x:-x[1]["pnl"]):
                wr=round(s["wins"]/s["n"]*100,1) if s["n"] else 0
                print("    "+lb+"  "+str(s["n"])+" trades  "+str(wr)+"% wins  $"+str(s["pnl"]))
            print()

    r_base=results["2025"][0.0040]
    r_s1=results["2025"][0.0050]
    r_s2=results["2025"][0.0060]
    print("="*72)
    print(" DECISION GATE")
    print("="*72)
    for lbl,r in [("0.40% baseline",r_base),("0.50% stress-1",r_s1),("0.60% stress-2",r_s2)]:
        if r:
            flag="PASS" if r["ann_ret"]>5 else "FAIL"
            print("  2025 "+lbl+": Ann="+str(r["ann_ret"])+"% Sharpe="+str(r["sharpe"])+" ["+flag+"]")

    if r_s1 and r_s1["ann_ret"]>5:
        print("\n  PROCEED to Weekend 4 (paper trading setup)")
    else:
        print("\n  STOP or consider lower-cost venue (Binance/Hyperliquid)")
