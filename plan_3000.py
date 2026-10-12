#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plan_3000.py  —— 选股/选策略的落地准备（3000 元总资金）
======================================================
做三件事:
  1. 抓宽基 ETF 实时报价（100 股/手，3000 元能买得起的关键）
  2. **方差拖累诊断**: 为什么"集中持有"在复利口径下会大幅落后于宽基等权
     （随机 K 只组合 → 算术均值 vs 复利收益 vs 波动率）
  3. 用 3000 元生成一份**可执行清单**（整数手 + 波动率阀门档位）
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

import quant_data as qd
from auto_train import build_extended_panel, OUT, BENCH
import vol_agents as va
from vol_target import forecasts as vol_forecasts

CACHE = Path(__file__).resolve().parent / "cache"

ETFS = [("sh510300", "沪深300ETF"), ("sh510500", "中证500ETF"),
        ("sh512100", "中证1000ETF"), ("sz159915", "创业板ETF"),
        ("sh510880", "红利ETF"), ("sh518880", "黄金ETF")]

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


def fetch_etf_quotes() -> dict:
    f = CACHE / "etf_quotes.csv"
    codes = ",".join(c for c, _ in ETFS)
    for _ in range(3):
        p = subprocess.run(["curl", "-s", "-m", "20", "--compressed",
                            "-H", f"User-Agent: {UA}",
                            f"https://qt.gtimg.cn/q={codes}"],
                           capture_output=True)
        if p.returncode == 0 and p.stdout:
            s = p.stdout.decode("gbk", "replace")
            out = {}
            for line in s.split(";"):
                line = line.strip()
                if not line.startswith("v_"):
                    continue
                sym, payload = line.split("=", 1)
                fld = payload.strip('"').split("~")
                if len(fld) < 10:
                    continue
                try:
                    out[sym[2:]] = {"name": fld[1], "price": float(fld[3]),
                                    "prev": float(fld[4]), "amount": float(fld[37]) * 1e4
                                    if len(fld) > 37 and fld[37] else None}
                except (ValueError, IndexError):
                    continue
            if out:
                pd.DataFrame(out).T.rename_axis("code").to_csv(f, encoding="utf-8")
                return out
        time.sleep(2)
    if f.exists():
        d = pd.read_csv(f, dtype={"code": str}).set_index("code")
        print("   [warn] 实时抓取失败, 用缓存报价")
        return {c: {"name": r["name"], "price": float(r["price"]),
                    "prev": float(r.get("prev", r["price"])),
                    "amount": None} for c, r in d.iterrows()}
    return {}


def variance_drag(df, all_dates, h=10, ks=(10, 20, 50, 100, 300, 9999), draws=30, seed=0):
    """随机选 K 只等权持有 → 算术均值 / 复利收益 / 波动率。隔离出'集中持有的复利代价'。"""
    rng = np.random.default_rng(seed)
    sub = df[["date", "code", f"fwd{h}"]].dropna().rename(columns={f"fwd{h}": "fwd"})
    g = {d: v["fwd"].values for d, v in sub.groupby("date")}
    dates = sorted(g.keys())
    rows = []
    for k in ks:
        rets_all = []
        for _ in range(draws):
            rs = []
            for d in dates[::h]:
                v = g[d]
                if len(v) < 5:
                    continue
                pick = v if k >= len(v) else rng.choice(v, k, replace=False)
                rs.append(float(np.mean(pick)))
            rs = np.array(rs)
            if len(rs) < 20:
                continue
            yrs = len(rs) * h / 250.0
            rets_all.append((float(np.mean(rs)), float((1 + rs).prod() ** (1 / yrs) - 1),
                             float(np.std(rs, ddof=1))))
        if not rets_all:
            continue
        a = np.array(rets_all)
        rows.append({"k": "全池" if k >= 9999 else k,
                     "arith_ann": round(float(a[:, 0].mean()) * 250 / h, 4),
                     "geo_ann": round(float(a[:, 1].mean()), 4),
                     "ann_vol": round(float(a[:, 2].mean()) * np.sqrt(250 / h), 4)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=3000.0)
    ap.add_argument("--target-vol", type=float, default=0.12)
    args = ap.parse_args()
    cap = args.capital
    print("=" * 96)
    print(f"3000 元落地准备  (总资金 {cap:.0f} 元, 目标波动率 {args.target_vol:.0%})")
    print("=" * 96)

    # ---------- 1) 方差拖累诊断 ----------
    df = build_extended_panel()
    all_dates = np.sort(df["date"].unique())
    rows = variance_drag(df, all_dates)
    print("\n【方差拖累诊断】随机等权持有 K 只, 每 10 个交易日调仓 (30 次随机抽样平均)")
    print(f"  {'持仓数':<8}{'算术年化':>10}{'复利年化':>10}{'年化波动':>10}{'复利代价':>10}")
    for r in rows:
        drag = r["arith_ann"] - r["geo_ann"]
        print(f"  {str(r['k']):<8}{r['arith_ann']:>10.2%}{r['geo_ann']:>10.2%}"
              f"{r['ann_vol']:>10.2%}{drag:>10.2%}")
    print("  -> 集中持有会抬高波动, 复利收益被 '算术均值 − ½σ²' 吃掉一大块。")
    print("     **这就是'选股越集中, 复利越吃亏'的数学原因。**")

    # ---------- 2) 当前波动率预测 → 阀门档位 ----------
    bench = qd.index_hist_bs(BENCH)
    dev_v, hold_v, _ = vol_forecasts(bench, list(va.FEATS), 10, va.Cfg())
    last = pd.concat([dev_v, hold_v]).sort_values("date").iloc[-1]
    cur_vol = float(last["p"])
    print(f"\n【当前波动率预测】沪深300 未来 10 日预测年化波动率 = {cur_vol:.1%} "
          f"(截至 {pd.Timestamp(last['date']).date()})")
    if cur_vol < 0.15:
        pos = 1.0
    elif cur_vol < 0.22:
        pos = 0.70
    else:
        pos = 0.40
    print(f"  → 建议总仓位档位: **{pos:.0%}**(档位规则: <15%→100%, 15~22%→70%, >22%→40%)")

    # ---------- 3) ETF 报价 + 可行清单 ----------
    q = fetch_etf_quotes()
    print("\n【宽基 ETF 报价】")
    usable = []
    for code, nm in ETFS:
        r = q.get(code)
        if not r:
            continue
        lot_cost = r["price"] * 100
        ok = lot_cost <= cap * 0.5
        print(f"  {nm:<12}{code:<10} 现价 {r['price']:>7.3f}  1手(100份)={lot_cost:>7.0f}元 "
              f"{'可买' if ok else '占比过大'}")
        if ok:
            usable.append((code, nm, r["price"], lot_cost))

    print(f"\n【{cap:.0f} 元可执行清单】总仓位 {pos:.0%} → 可投 {cap*pos:.0f} 元")
    if usable:
        # 目标权重: 宽基分散(小盘偏多, 因为等权小市值是历史上最强的基准)
        W = {"sh510300": 0.30, "sh510500": 0.30, "sh512100": 0.30, "sh510880": 0.10}
        budget = cap * pos
        px = {c: (nm, p, lot) for c, nm, p, lot in usable}
        picks, spent = [], 0.0
        for code, w in W.items():
            if code not in px or spent >= budget * 0.97:
                continue
            nm, p, lot = px[code]
            n = int(min(budget * w, budget - spent) // lot)
            if n >= 1:
                picks.append({"code": code, "name": nm, "price": p, "lots": n,
                              "shares": n * 100, "cost": round(n * lot, 2)})
                spent += n * lot
        # 剩余预算优先补给最便宜且已持有的标的
        for code in sorted([p["code"] for p in picks],
                           key=lambda c: px[c][2]):
            nm, p, lot = px[code]
            while spent + lot <= budget * 0.99:
                for it in picks:
                    if it["code"] == code:
                        it["lots"] += 1
                        it["shares"] += 100
                        it["cost"] = round(it["cost"] + lot, 2)
                        spent += lot
                        break
                else:
                    break
        for p in picks:
            print(f"  {p['name']:<12}{p['code']:<10} {p['lots']}手({p['shares']}份) "
                  f"≈ {p['cost']:.0f} 元  权重 {p['cost']/max(1,spent):.0%}")
        # 成本测算
        rt = len(picks) * 5.0 * 2
        print(f"  合计投入 {spent:.0f} 元, 留现金 {cap - spent:.0f} 元")
        print(f"  成本测算: {len(picks)} 只 × 一买一卖佣金最低 5 元 = **{rt:.0f} 元 "
              f"≈ 总资金 {rt/cap:.2%}**(ETF 无印花税; 若券商 ETF 佣金下限低于 5 元则更低)")
        out = {"capital": cap, "target_vol": args.target_vol, "cur_vol": cur_vol,
               "position": pos, "picks": picks, "cash": round(cap - spent, 2),
               "round_trip_cost": rt, "variance_drag": rows, "etf_quotes": q}
    else:
        out = {"capital": cap, "target_vol": args.target_vol, "cur_vol": cur_vol,
               "position": pos, "picks": [], "variance_drag": rows, "etf_quotes": q}
        print("  未能获取 ETF 报价, 请手动确认价格")

    (OUT / "plan_3000.json").write_text(json.dumps(out, ensure_ascii=False, indent=2,
                                                   default=str), encoding="utf-8")
    print(f"\n已写 {OUT/'plan_3000.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
