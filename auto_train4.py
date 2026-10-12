#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自主训练 第四阶段 auto_train4.py —— 【负向筛选】
================================================
第三阶段发现: 预测"后 25%(差生)"的 AUC 高达 0.62~0.63, 远高于预测"前 25%"(0.57)
和"跑赢中位数"(0.55)。说明模型最强的能力是**识别差生**, 不是挑牛股。

那么最该做的用法是"负向筛选":
    不要选股, 而是把预测最差的 X% 剔掉, 剩下的**等权持有**。
    基准仍然是"全池等权" —— 优势就来自"少亏"。

同时测试组合版: 先剔掉差生, 再在前 25% 里挑。
纪律: 开发期选 X 与模型, 留出集只评估一次。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from auto_train import build_extended_panel, HS, HOLDOUT, COST_STOCK, BENCH, OUT
from auto_train2 import wf, make_model, FEATS
import quant_data as qd
import train_agents as ta
import vol_agents as va
from train_agents import setup_font

XS = (0.10, 0.20, 0.30, 0.40)


def screen_bt(pooled, all_dates, h, x, cost=COST_STOCK):
    """把 p(差生概率) 最高的 x 比例剔掉, 其余等权持有。返回 (指标, 曲线, 等权曲线)"""
    if pooled is None or pooled.empty:
        return None, None, None
    ix = {pd.Timestamp(d): i for i, d in enumerate(all_dates)}
    prev, eq, eqw = set(), 1.0, 1.0
    curve, ew = [], []
    rets = []
    for d, g in pooled.sort_values("date").groupby("date"):
        i = ix.get(pd.Timestamp(d))
        if i is None or (i % h) != 0 or len(g) < 40:
            continue
        n_excl = int(len(g) * x)
        keep = g.sort_values("p").head(len(g) - n_excl)
        if len(keep) < 10:
            continue
        news = set(keep["code"])
        turn = (len(news - prev) + len(prev - news)) / max(1, 2 * len(news))
        r = float(keep["fwd"].mean()) - cost * turn
        rw = float(g["fwd"].mean())
        eq *= (1 + r)
        eqw *= (1 + rw)
        curve.append(eq)
        ew.append(eqw)
        rets.append(r)
        prev = news
    if len(curve) < 20:
        return None, None, None
    curve, ew = np.array(curve), np.array(ew)
    rets = np.array(rets)
    yrs = len(curve) * h / 250.0
    ann = float(curve[-1] ** (1 / yrs) - 1)
    ew_ann = float(ew[-1] ** (1 / yrs) - 1)
    mdd = float((curve / np.maximum.accumulate(curve) - 1).min())
    sd = float(np.std(rets, ddof=1))
    return {"ann": round(ann, 4), "ew_ann": round(ew_ann, 4),
            "excess_ann": round(ann - ew_ann, 4), "mdd": round(mdd, 4),
            "sharpe": round(float(np.mean(rets) / sd * np.sqrt(250 / h)), 3) if sd else None,
            "calmar": round(ann / abs(mdd), 3) if mdd else None,
            "n": len(curve), "x": x, "years": round(yrs, 2)}, curve, ew


def combo_bt(pooled_top, pooled_bot, all_dates, h, k, x, cost=COST_STOCK):
    """先剔掉最差的 x, 再取 p_top 最高的 k 只。"""
    if pooled_top is None or pooled_bot is None:
        return None
    m = pooled_top.merge(pooled_bot[["date", "code", "p"]],
                         on=["date", "code"], suffixes=("_top", "_bot"))
    ix = {pd.Timestamp(d): i for i, d in enumerate(all_dates)}
    prev, eq, eqw = set(), 1.0, 1.0
    rets, curve, ew = [], [], []
    for d, g in m.groupby("date"):
        i = ix.get(pd.Timestamp(d))
        if i is None or (i % h) != 0 or len(g) < 40:
            continue
        n_excl = int(len(g) * x)
        g2 = g.sort_values("p_bot").head(len(g) - n_excl)
        if len(g2) < k:
            continue
        sel = g2.sort_values("p_top", ascending=False).head(k)
        news = set(sel["code"])
        turn = len(news - prev) / max(1, k)
        r = float(sel["fwd"].mean()) - cost * turn
        rw = float(g["fwd"].mean())
        eq *= (1 + r)
        eqw *= (1 + rw)
        curve.append(eq)
        ew.append(eqw)
        rets.append(r)
        prev = news
    if len(curve) < 20:
        return None
    curve, ew = np.array(curve), np.array(ew)
    rets = np.array(rets)
    yrs = len(curve) * h / 250.0
    ann = float(curve[-1] ** (1 / yrs) - 1)
    ew_ann = float(ew[-1] ** (1 / yrs) - 1)
    mdd = float((curve / np.maximum.accumulate(curve) - 1).min())
    sd = float(np.std(rets, ddof=1))
    return {"ann": round(ann, 4), "ew_ann": round(ew_ann, 4),
            "excess_ann": round(ann - ew_ann, 4), "mdd": round(mdd, 4),
            "sharpe": round(float(np.mean(rets) / sd * np.sqrt(250 / h)), 3) if sd else None,
            "n": len(curve), "k": k, "x": x, "years": round(yrs, 2)}


def main():
    t0 = time.time()
    df = build_extended_panel()
    all_dates = np.sort(df["date"].unique())
    dev_end = all_dates[int(len(all_dates) * (1 - HOLDOUT))]
    feats_all = [f for f in ta.ALL_FEATS if f in df.columns]
    FS = {"all": feats_all,
          "nomkt": [f for f in feats_all if f not in ta.FEATS_MARKET]}
    for h in HS:
        rk = df.groupby("date")[f"fwd{h}"].rank(pct=True)
        df[f"ybot{h}"] = np.where(df[f"fwd{h}"].isna(), np.nan,
                                  (rk <= 0.25).astype(float))
        df[f"ytop{h}"] = np.where(df[f"fwd{h}"].isna(), np.nan,
                                  (rk > 0.75).astype(float))

    print("=" * 100)
    print("【第四阶段】负向筛选: 剔掉预测的差生, 其余等权持有")
    print("=" * 100, flush=True)
    out = []
    for h in HS:
        for kind in ("hgb", "hgbdeep", "logit"):
            for fname, feats in FS.items():
                p_bot, h_bot = wf(df, feats, f"ybot{h}", h, kind, dev_end)
                if p_bot is None:
                    continue
                p_top, h_top = wf(df, feats, f"ytop{h}", h, kind, dev_end)
                rows, rows_h = [], []
                for x in XS:
                    met, cur, ew = screen_bt(p_bot, all_dates, h, x)
                    if met:
                        rows.append(met)
                    m2, _, _ = screen_bt(h_bot, all_dates, h, x)
                    if m2:
                        rows_h.append(m2)
                bs = max(rows, key=lambda z: z["excess_ann"]) if rows else {}
                # 留出期用开发期最优的 x (不看留出期)
                bx = bs.get("x")
                bh = next((r for r in rows_h if abs(r["x"] - bx) < 1e-9), None) if bx else None
                # 组合版
                cb = None
                for k in (10, 20):
                    c = combo_bt(p_top, p_bot, all_dates, h, k, 0.20)
                    if c and (cb is None or c["excess_ann"] > cb["excess_ann"]):
                        cb = c
                rec = {"h": h, "kind": kind, "feats": fname, "x_dev_best": bx,
                       "dev_best": bs, "hold_same_x": bh, "combo_dev": cb,
                       "dev_all": rows, "hold_all": rows_h}
                out.append(rec)
                print(f"  h={h:<3} {kind:<8} {fname:<6} 开发期最优 x={bx} "
                      f"超额 {bs.get('excess_ann')} 夏普 {bs.get('sharpe')} | "
                      f"留出期(同x) 超额 {None if not bh else bh['excess_ann']} | "
                      f"组合版 超额 {None if not cb else cb['excess_ann']} "
                      f"| {time.time()-t0:.0f}s", flush=True)

    (OUT / "auto_train4.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # 汇总: 按 (h, 模型) 平均
    print("\n" + "=" * 100)
    print("负向筛选 汇总(开发期最优 x 下)")
    print("=" * 100)
    agg = {}
    for r in out:
        agg.setdefault((r["h"], r["kind"]), []).append(
            (r["dev_best"].get("excess_ann"), (r["hold_same_x"] or {}).get("excess_ann")))
    print(f"  {'h':<5}{'模型':<10}{'开发期平均超额':>16}{'留出期平均超额':>16}{'n':>4}")
    for k, v in sorted(agg.items()):
        dv = [a for a, b in v if a is not None]
        hv = [b for a, b in v if b is not None]
        print(f"  {k[0]:<5}{k[1]:<10}{np.mean(dv):>16.4f}"
              f"{(np.mean(hv) if hv else float('nan')):>16.4f}{len(v):>4}")

    setup_font()
    fig, ax = plt.subplots(figsize=(13, 5.5))
    labs, dv, hv = [], [], []
    for k, v in sorted(agg.items()):
        d = [a for a, b in v if a is not None]
        hh = [b for a, b in v if b is not None]
        labs.append(f"h={k[0]}\n{k[1]}")
        dv.append(np.mean(d))
        hv.append(np.mean(hh) if hh else 0)
    x = np.arange(len(labs))
    ax.bar(x - 0.2, dv, width=0.4, color="#4c72b0", label="开发期平均超额")
    ax.bar(x + 0.2, hv, width=0.4, color="#c44e52", label="留出期平均超额")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("相对全池等权的超额年化收益")
    ax.set_title("负向筛选: 剔掉预测最差的 X%, 其余等权持有")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "auto_train4.png", dpi=130)
    plt.close(fig)
    print(f"\n总用时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
