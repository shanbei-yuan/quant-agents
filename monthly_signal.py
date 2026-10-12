#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monthly_signal.py —— 每月一键出清单（实盘用）
=============================================
只读缓存, 绝不修改数据。流程:
  1. 用"预测横截面排名"模型在**全部历史**上做一次滚动验证, 报告最近一段的真实样本外表现
     （让你每月都能看到模型是在衰减还是变强 —— 不是黑箱）
  2. 用截至最新可得数据训练最终模型, 给**最新截面的每只股票打分**
  3. 结合波动率阀门算出本月总仓位
  4. 输出两份清单:
       A. 主仓: 宽基 ETF 组合（按当前档位）
       B. 实验仓(可选): 打分最高的 N 只个股, 各 1 手
  5. 结果写入 out/下单清单_YYYYMMDD.md 与 .json

用法:
    python -u monthly_signal.py --capital 3000 --topn 3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from auto_train import build_extended_panel, BENCH, OUT, COST_STOCK
from tail_ltr import make_reg, CAP
import quant_data as qd
import train_agents as ta
import vol_agents as va
from vol_target import forecasts as vol_forecasts

CACHE = Path(__file__).resolve().parent / "cache"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

ETFS = [("sh510300", "沪深300ETF"), ("sh510500", "中证500ETF"),
        ("sh512100", "中证1000ETF"), ("sh510880", "红利ETF")]
WEIGHTS = {"sh510300": 0.30, "sh510500": 0.30, "sh512100": 0.30, "sh510880": 0.10}


def quotes(codes: list[str]) -> dict:
    out = {}
    for i in range(0, len(codes), 150):
        chunk = codes[i:i + 150]
        for _ in range(3):
            p = subprocess.run(["curl", "-s", "-m", "20", "--compressed",
                               "-H", f"User-Agent: {UA}",
                               "https://qt.gtimg.cn/q=" + ",".join(chunk)],
                              capture_output=True)
            if p.returncode == 0 and p.stdout:
                s = p.stdout.decode("gbk", "replace")
                for line in s.split(";"):
                    line = line.strip()
                    if not line.startswith("v_"):
                        continue
                    sym, payload = line.split("=", 1)
                    f = payload.strip('"').split("~")
                    if len(f) < 6:
                        continue
                    try:
                        out[sym[2:]] = {"name": f[1], "price": float(f[3]),
                                        "prev": float(f[4]), "open": float(f[5]),
                                        "high": float(f[33]) if len(f) > 33 else np.nan,
                                        "low": float(f[34]) if len(f) > 34 else np.nan,
                                        "volume": float(f[6]) if f[6] else 0.0}
                    except (ValueError, IndexError):
                        continue
                break
            time.sleep(1.5)
        time.sleep(0.3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=3000.0)
    ap.add_argument("--h", type=int, default=10)
    ap.add_argument("--topn", type=int, default=3)
    ap.add_argument("--val-seg", type=int, default=250)
    a = ap.parse_args()
    t0 = time.time()
    print("=" * 96)
    print(f"本月清单生成  (资金 {a.capital:.0f} 元, 周期 h={a.h})")
    print("=" * 96, flush=True)

    df = build_extended_panel()
    feats = [f for f in ta.ALL_FEATS if f in df.columns]
    last_date = df["date"].max()
    print(f"[数据] 截至 {pd.Timestamp(last_date).date()}, {len(df):,} 行 "
          f"(如需更新请先运行 refresh_data.py)", flush=True)

    # ---- 1) 滚动验证: 最近一段的真实样本外表现 ----
    df[f"yrank{a.h}"] = df.groupby("date")[f"fwd{a.h}"].rank(pct=True) - 0.5
    w = df[["date", "code", f"fwd{a.h}", f"yrank{a.h}"] + feats].dropna()
    D = w["date"].values
    C = w["code"].values
    Y = w[f"yrank{a.h}"].values.astype(np.float32)
    F = w[f"fwd{a.h}"].values.astype(np.float32)
    A = w[feats].to_numpy(dtype=np.float32)
    dates = np.sort(np.unique(D))
    tr_end = len(dates) - a.val_seg
    tr = np.where(D < dates[tr_end])[0]
    te = np.where(D >= dates[tr_end])[0]
    if len(tr) > CAP:
        tr = np.random.default_rng(0).choice(tr, CAP, replace=False)
    mval = make_reg("hgbr", 0)
    mval.fit(A[tr], Y[tr])
    pv = mval.predict(A[te])
    sv = pd.DataFrame({"date": D[te], "p": pv, "fwd": F[te]})
    ic = sv.groupby("date").apply(lambda g: g["p"].corr(g["fwd"], method="spearman"))
    ic = ic.dropna()
    dd = np.sort(sv["date"].unique())[::max(1, a.h)]
    tr_r, po_r = [], []
    for d in dd:
        g = sv[sv["date"] == d]
        if len(g) < 40:
            continue
        tr_r.append(float(g.sort_values("p", ascending=False).head(10)["fwd"].mean()))
        po_r.append(float(g["fwd"].mean()))
    val = {"seg_start": str(pd.Timestamp(dates[tr_end]).date()),
           "seg_end": str(pd.Timestamp(dates[-1]).date()),
           "rank_ic": round(float(ic.mean()), 4),
           "ic_pos": round(float((ic > 0).mean()), 3),
           "top10_mean": round(float(np.mean(tr_r)) * 100, 4),
           "pool_mean": round(float(np.mean(po_r)) * 100, 4),
           "excess_mean": round(float(np.mean(np.array(tr_r) - np.array(po_r))) * 100, 4)}
    print(f"\n【最近一段真实样本外表现】{val['seg_start']} ~ {val['seg_end']}")
    print(f"  rank IC {val['rank_ic']:+.4f} (正比例 {val['ic_pos']:.0%}) | "
          f"top10 平均 {val['top10_mean']}% vs 全池 {val['pool_mean']}% "
          f"(差 {val['excess_mean']:+.4f}pp/期)")
    if val["rank_ic"] <= 0 or val["excess_mean"] <= 0:
        print("  ⚠️ 最近一段模型**没有跑赢全池** —— 本月实验仓建议空仓或减半")
    print(flush=True)

    # ---- 2) 最终模型 + 最新截面打分 ----
    tr_all = np.where(D <= np.datetime64(last_date))[0]
    known = ~np.isnan(Y[tr_all])
    tr_all = tr_all[known]
    if len(tr_all) > CAP:
        tr_all = np.random.default_rng(0).choice(tr_all, CAP, replace=False)
    mfin = make_reg("hgbr", 0)
    mfin.fit(A[tr_all], Y[tr_all])
    sc = df[df["date"] == last_date].copy()
    okf = sc[feats].notna().all(axis=1)
    sc = sc[okf].copy()
    sc["score"] = mfin.predict(sc[feats].to_numpy(dtype=np.float32))
    sc = sc.sort_values("score", ascending=False)
    print(f"[打分] 最新截面 {len(sc)} 只, 分数范围 {sc['score'].min():+.3f} ~ "
          f"{sc['score'].max():+.3f}", flush=True)

    # 股票池元信息(名称/市场)
    snap = qd.snapshot()
    meta = snap.set_index("code")[["name", "mkt"]].to_dict("index")

    # ---- 3) 波动率阀门 ----
    bench = qd.index_hist_bs(BENCH)
    dev_v, hold_v, _ = vol_forecasts(bench, list(va.FEATS), 10, va.Cfg())
    cur_vol = float(pd.concat([dev_v, hold_v]).sort_values("date").iloc[-1]["p"])
    pos = 1.0 if cur_vol < 0.15 else (0.70 if cur_vol < 0.22 else 0.40)
    print(f"[阀门] 沪深300 预测年化波动率 {cur_vol:.1%} → 目标总仓位 **{pos:.0%}**",
          flush=True)

    # ---- 4) 报价 ----
    etf_codes = [c for c, _ in ETFS]
    cand_codes = []
    for code in sc["code"].head(a.topn * 5):
        m = meta.get(code)
        if m and m.get("name") and "ST" not in str(m["name"]) and "退" not in str(m["name"]):
            cand_codes.append(("sh" if int(m["mkt"]) == 1 else "sz") + str(code))
    q = quotes(etf_codes + cand_codes)
    print(f"[报价] 取得 {len(q)} 条实时报价", flush=True)

    # ---- 5) ETF 主仓清单（贪心: 每次买“当前权重低于目标最多”的那只）----
    budget = a.capital * pos
    etf_pick = []
    while True:
        cand = None
        total = sum(p["cost"] for p in etf_pick)
        for code, w_ in WEIGHTS.items():
            r = q.get(code)
            if not r or r["price"] <= 0:
                continue
            lot = r["price"] * 100
            if lot > budget - total + 1e-9:
                continue
            cur = sum(p["cost"] for p in etf_pick if p["code"] == code)
            deficit = w_ * budget - cur
            if deficit <= 0:
                continue
            if cand is None or deficit > cand[1]:
                cand = (code, deficit, r, lot)
        if cand is None:
            break
        code, _, r, lot = cand
        hit = next((p for p in etf_pick if p["code"] == code), None)
        if hit:
            hit["lots"] += 1
            hit["shares"] += 100
            hit["cost"] = round(hit["cost"] + lot, 2)
        else:
            etf_pick.append({"code": code, "name": r["name"], "price": r["price"],
                             "lots": 1, "shares": 100, "cost": round(lot, 2)})
    spent = sum(p["cost"] for p in etf_pick)
    etf_pick.sort(key=lambda p: -p["cost"])
    # ---- 6) 实验仓股票清单 ----
    exp_budget = min(a.capital * 0.30, 1000.0)
    stock_pick, exp_spent = [], 0.0
    if val["rank_ic"] > 0 and val["excess_mean"] > 0:
        for code in sc["code"]:
            if len(stock_pick) >= a.topn:
                break
            m = meta.get(code)
            if not m or not m.get("name"):
                continue
            if "ST" in str(m["name"]) or "退" in str(m["name"]):
                continue
            sym = ("sh" if int(m["mkt"]) == 1 else "sz") + str(code)
            r = q.get(sym)
            if not r or r["price"] <= 0 or r["price"] > 25:
                continue
            lot = r["price"] * 100
            if exp_spent + lot > exp_budget:
                continue
            stock_pick.append({"code": code, "name": r["name"], "price": r["price"],
                               "lots": 1, "shares": 100, "cost": round(lot, 2),
                               "score": round(float(sc.loc[sc["code"] == code, "score"].iloc[0]), 4)})
            exp_spent += lot

    # ---- 7) 输出 ----
    L = []
    L.append(f"# 本月下单清单 ({pd.Timestamp.now():%Y-%m-%d})")
    L.append("")
    L.append(f"- 资金 {a.capital:.0f} 元 | 数据截止 **{pd.Timestamp(last_date).date()}**"
             f" | 预测周期 {a.h} 个交易日")
    L.append(f"- 沪深300 预测年化波动率 **{cur_vol:.1%}** → 目标总仓位 **{pos:.0%}**"
             f"（档位: <15%→100% / 15~22%→70% / >22%→40%）")
    L.append("")
    L.append(f"## 最近一段真实样本外表现（{val['seg_start']} ~ {val['seg_end']}）")
    L.append("")
    L.append(f"- rank IC **{val['rank_ic']:+.4f}**（正比例 {val['ic_pos']:.0%}）")
    L.append(f"- top10 平均 {val['top10_mean']}% vs 全池 {val['pool_mean']}% "
             f"→ 差 **{val['excess_mean']:+.4f}pp/期**")
    L.append(f"- 结论: {'模型仍有效 ✓' if val['rank_ic']>0 and val['excess_mean']>0 else '**模型已失效 ✗（实验仓空仓）**'}")
    L.append("")
    L.append(f"## A. 主仓：宽基 ETF（目标投入 {budget:.0f} 元）")
    L.append("")
    L.append("| 代码 | 名称 | 现价 | 数量 | 金额 |")
    L.append("|---|---|---|---|---|")
    for p in etf_pick:
        L.append(f"| {p['code']} | {p['name']} | {p['price']:.3f} | {p['lots']}手"
                 f"({p['shares']}份) | {p['cost']:.0f} |")
    L.append(f"| | **合计** | | | **{spent:.0f}** |")
    L.append(f"| | 现金(货币基金) | | | {a.capital - spent:.0f} |")
    L.append("")
    L.append(f"## B. 实验仓：选股（预算 {exp_budget:.0f} 元，可选）")
    L.append("")
    if stock_pick:
        L.append("| 代码 | 名称 | 现价 | 数量 | 金额 | 模型分 |")
        L.append("|---|---|---|---|---|---|")
        for p in stock_pick:
            L.append(f"| {p['code']} | {p['name']} | {p['price']:.2f} | 1手(100股) | "
                     f"{p['cost']:.0f} | {p['score']:+.4f} |")
        L.append(f"| | **合计** | | | **{exp_spent:.0f}** | |")
    else:
        L.append("**本月不建仓**（最近一段模型未跑赢全池，或没有可用标的）")
    L.append("")
    L.append("## C. 执行提醒")
    L.append("")
    L.append(f"1. 下单前**重新核对实时价格**（本清单价格取自生成时刻）")
    L.append(f"2. 合计投入 {spent + exp_spent:.0f} 元，占总资金 "
             f"{(spent+exp_spent)/a.capital:.0%}，现金保留 "
             f"{a.capital - spent - exp_spent:.0f} 元")
    L.append(f"3. 佣金最低 5 元/笔 → 本清单共 {len(etf_pick)+len(stock_pick)} 笔，"
             f"一买一卖合计约 {(len(etf_pick)+len(stock_pick))*10} 元")
    L.append(f"4. 实验仓规则：持有 4 周或 −8% 止损；**绝不超过 {exp_budget:.0f} 元**")
    L.append(f"5. 下月同日再跑一次本脚本，对比 rank IC 是升是降")
    L.append("")
    L.append("> 免责声明：本清单是本项目的研究输出，不构成投资建议。"
             "本项目自身的结论是「没有可靠选股能力」，主仓设计为宽基 ETF + 波动率阀门。")
    txt = "\n".join(L)
    stamp = f"{pd.Timestamp.now():%Y%m%d}"
    (OUT / f"下单清单_{stamp}.md").write_text(txt, encoding="utf-8")
    (OUT / f"下单清单_{stamp}.json").write_text(json.dumps(
        {"capital": a.capital, "data_date": str(pd.Timestamp(last_date).date()),
         "cur_vol": cur_vol, "position": pos, "validation": val,
         "etf": etf_pick, "stock": stock_pick,
         "cash": a.capital - spent - exp_spent}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("\n" + txt)
    print(f"\n已写 {OUT/f'下单清单_{stamp}.md'} | 用时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
