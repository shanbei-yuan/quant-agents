#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monitor.py  —— 持续监控: 模型是在衰减还是在变强？
=================================================
思路: 把历史切成许多**连续的小段**(默认每 120 个交易日一段), 每段只用该段之前的数据训练,
然后预测该段。因为每段都会重新训练, 所以这就是"如果当时上线, 会得到什么"的真实序列。

输出:
  * 每段的 rank IC、top-K 相对全池的收益差、累计复利差
  * 一张趋势图(out/monitor.png): 看到底是衰减还是变强
  * 每次运行的汇总追加到 out/monitor_log.jsonl（可以月月对比）

用法:
    python -u monitor.py --h 10 --k 10 --seg 120
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from auto_train import build_extended_panel, COST_STOCK, OUT
from tail_ltr import make_reg, CAP, WINSOR
import train_agents as ta
from train_agents import setup_font

LOG = OUT / "monitor_log.jsonl"


def monitor(df, feats, h, k, seg_len, min_train=700, seed=0):
    df = df.copy()
    df[f"yrank{h}"] = df.groupby("date")[f"fwd{h}"].rank(pct=True) - 0.5
    w = df[["date", "code", f"fwd{h}", f"yrank{h}"] + feats].dropna()
    D = w["date"].values
    C = w["code"].values
    Y = w[f"yrank{h}"].values.astype(np.float32)
    F = w[f"fwd{h}"].values.astype(np.float32)
    A = w[feats].to_numpy(dtype=np.float32)
    dates = np.sort(np.unique(D))
    rng = np.random.default_rng(seed)
    rows = []
    for i0 in range(min_train, len(dates), seg_len):
        te_dates = dates[i0:i0 + seg_len]
        if len(te_dates) < 40:
            break
        d0, d1 = te_dates[0], te_dates[-1]
        tr = np.where(D < d0)[0]
        tei = np.where((D >= d0) & (D <= d1))[0]
        if len(tr) < 3000 or len(tei) < 300:
            continue
        if len(tr) > CAP:
            tr = rng.choice(tr, CAP, replace=False)
        m = make_reg("hgbr", seed)
        m.fit(A[tr], Y[tr])
        p = m.predict(A[tei])
        s = pd.DataFrame({"date": D[tei], "code": C[tei], "p": p, "fwd": F[tei]})
        ic = s.groupby("date").apply(
            lambda g: g["p"].corr(g["fwd"], method="spearman") if len(g) > 30 else np.nan)
        ic = ic.dropna()
        # 非重叠调仓: 每 h 天一次
        dd = np.sort(s["date"].unique())[::max(1, h)]
        top_r, pool_r = [], []
        for d in dd:
            g = s[s["date"] == d]
            if len(g) < 40:
                continue
            top_r.append(float(g.sort_values("p", ascending=False).head(k)["fwd"].mean()))
            pool_r.append(float(g["fwd"].mean()))
        if len(top_r) < 5:
            continue
        top_r, pool_r = np.array(top_r), np.array(pool_r)
        rows.append({"seg_end": str(pd.Timestamp(d1).date()),
                     "n_dates": int(len(ic)),
                     "rank_ic": round(float(ic.mean()), 4),
                     "ic_pos": round(float((ic > 0).mean()), 3),
                     "topk_mean": round(float(top_r.mean()) * 100, 4),
                     "pool_mean": round(float(pool_r.mean()) * 100, 4),
                     "excess_mean": round(float((top_r - pool_r).mean()) * 100, 4),
                     "topk_geo": round(float((1 + top_r - COST_STOCK).prod() ** (250 / (h * len(top_r))) - 1), 4),
                     "pool_geo": round(float((1 + pool_r).prod() ** (250 / (h * len(pool_r))) - 1), 4),
                     "n_reb": int(len(top_r))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=10)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seg", type=int, default=120)
    a = ap.parse_args()
    t0 = time.time()
    df = build_extended_panel()
    feats = [f for f in ta.ALL_FEATS if f in df.columns]
    print("=" * 100)
    print(f"持续监控: 预测横截面排名, h={a.h}, top-{a.k}, 每段 {a.seg} 个交易日")
    print("=" * 100, flush=True)
    rows = monitor(df, feats, a.h, a.k, a.seg)
    if not rows:
        print("无有效段落")
        return 1
    tab = pd.DataFrame(rows)
    tab["topk_geo"] = (tab["topk_geo"] * 100).round(2)
    tab["pool_geo"] = (tab["pool_geo"] * 100).round(2)
    tab["excess_geo"] = (tab["topk_geo"] - tab["pool_geo"]).round(2)
    print(tab[["seg_end", "n_reb", "rank_ic", "ic_pos", "topk_mean", "pool_mean",
               "excess_mean", "topk_geo", "pool_geo", "excess_geo"]].to_string(index=False),
          flush=True)
    ic = tab["rank_ic"].values
    ex = tab["excess_mean"].values
    print(f"\n汇总: 段数 {len(tab)} | rankIC 均值 {ic.mean():+.4f} "
          f"(前半 {ic[:len(ic)//2].mean():+.4f} → 后半 {ic[len(ic)//2:].mean():+.4f})")
    print(f"      top{a.k} 相对全池 的平均差 {ex.mean():+.4f}pp/期 "
          f"(前半 {ex[:len(ex)//2].mean():+.4f} → 后半 {ex[len(ex)//2:].mean():+.4f})")
    trend = "变强" if ic[len(ic)//2:].mean() > ic[:len(ic)//2].mean() else "衰减"
    print(f"      → rankIC 趋势: **{trend}**")
    print(f"      正超额段比例: {(ex > 0).mean():.0%}")

    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": str(pd.Timestamp.now())[:19], "h": a.h, "k": a.k,
                            "seg": a.seg, "n_seg": len(tab),
                            "ic_mean": round(float(ic.mean()), 4),
                            "ic_first_half": round(float(ic[:len(ic)//2].mean()), 4),
                            "ic_second_half": round(float(ic[len(ic)//2:].mean()), 4),
                            "excess_mean": round(float(ex.mean()), 4),
                            "trend": trend, "rows": rows}, ensure_ascii=False) + "\n")

    setup_font()
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    x = np.arange(len(tab))
    axes[0].bar(x, tab["rank_ic"], color=np.where(tab["rank_ic"] > 0, "#4c72b0", "#c44e52"))
    axes[0].axhline(0, color="k", lw=1)
    axes[0].axhline(ic.mean(), color="#ff7f0e", ls="--", lw=1.2,
                    label=f"均值 {ic.mean():+.4f}")
    axes[0].set_ylabel("rank IC")
    axes[0].set_title(f"持续监控: 每段重新训练的样本外 rank IC (h={a.h}, 每段 {a.seg} 日)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].bar(x - 0.2, tab["topk_geo"], width=0.4, color="#c44e52",
                label=f"top{a.k} 复利年化%")
    axes[1].bar(x + 0.2, tab["pool_geo"], width=0.4, color="#4c72b0", label="全池复利年化%")
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tab["seg_end"], rotation=60, fontsize=7)
    axes[1].set_ylabel("复利年化 %")
    axes[1].set_title("同段落: 选股组合 vs 全池等权")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "monitor.png", dpi=130)
    plt.close(fig)
    print(f"\n图 {OUT/'monitor.png'} | 日志 {LOG} | 用时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
