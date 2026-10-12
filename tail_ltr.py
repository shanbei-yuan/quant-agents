#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按建议②③推进 tail_ltr.py
=========================
建议②：换掉目标函数。前三小时的根因是"模型学的是排序信息（是否高于分位），
        而组合收益由右尾决定"。本脚本直接**回归预测收益本身**，让优化目标与收益对齐，
        并和前 25% / 中位数标签做**同口径对照**。

建议③：用**多段滚动留出**代替单一留出集。把测试期切成 N 段，每段只用该段之前的
        数据训练，逐段报告 OOS 表现 —— 得到的是"表现分布"，而不是一个可能靠运气的数字。

评估指标：
  * 排序信息系数 rank IC（每日 Spearman(预测, 实际收益) 的均值）—— 比 AUC 更贴近收益
  * 组合超额收益（相对全池等权），逐段 + 全期
  * top-K 组合平均收益 vs 全池平均
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

from auto_train import build_extended_panel, portfolio, acc_stats, HS, BENCH, OUT, COST_STOCK
import quant_data as qd
import train_agents as ta
import vol_agents as va
from vol_target import forecasts as vol_forecasts
from train_agents import setup_font

N_SEG = 4
MIN_TRAIN = 700
CAP = 120000
WINSOR = (-0.35, 0.60)      # 收益目标截尾, 降低极端值影响


def make_reg(kind, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "hgbr":
        return HistGradientBoostingRegressor(
            max_iter=120, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=60,
            l2_regularization=1.0, loss="absolute_error", early_stopping=True,
            n_iter_no_change=10, validation_fraction=0.15, random_state=seed)
    if kind == "hgbr2":
        return HistGradientBoostingRegressor(
            max_iter=120, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=60,
            l2_regularization=1.0, loss="squared_error", early_stopping=True,
            n_iter_no_change=10, validation_fraction=0.15, random_state=seed)
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    if kind == "rfr":
        return RandomForestRegressor(n_estimators=40, max_depth=6, min_samples_leaf=150,
                                     n_jobs=-1, random_state=seed)
    raise ValueError(kind)


def make_cls(kind, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=80, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=60,
        l2_regularization=1.0, early_stopping=True, n_iter_no_change=10,
        validation_fraction=0.15, random_state=seed)


def wf_multi(df, feats, target, h, kind, task="reg", n_seg=N_SEG, seed=0):
    """多段滚动留出：训练集只用测试段之前的数据。返回 (逐段结果, 全期 pooled)。"""
    w = df[["date", "code", f"fwd{h}", target] + feats].dropna()
    if len(w) < 8000:
        return [], None
    D = w["date"].values
    C = w["code"].values
    Y = w[target].values.astype(np.float32)
    F = w[f"fwd{h}"].values.astype(np.float32)
    A = w[feats].to_numpy(dtype=np.float32)
    dates = np.sort(np.unique(D))
    if len(dates) < MIN_TRAIN + 200 * n_seg:
        return [], None
    seg_len = (len(dates) - MIN_TRAIN) // n_seg
    rng = np.random.default_rng(seed)
    segs, preds_all = [], []
    for k in range(n_seg):
        i0 = MIN_TRAIN + k * seg_len
        i1 = len(dates) if k == n_seg - 1 else i0 + seg_len
        te_dates = dates[i0:i1]
        if len(te_dates) < 60:
            continue
        d0, d1 = te_dates[0], te_dates[-1]
        tr = np.where(D < d0)[0]
        tei = np.where((D >= d0) & (D <= d1))[0]
        if len(tr) < 3000 or len(tei) < 300:
            continue
        if len(tr) > CAP:
            tr = rng.choice(tr, CAP, replace=False)
        m = make_reg(kind, seed) if task == "reg" else make_cls(kind, seed)
        m.fit(A[tr], Y[tr])
        p = (m.predict(A[tei]) if task == "reg"
             else m.predict_proba(A[tei])[:, 1])
        seg = pd.DataFrame({"date": D[tei], "code": C[tei], "p": p, "fwd": F[tei]})
        seg["seg"] = k
        segs.append(seg)
        preds_all.append(seg)
    pooled = pd.concat(preds_all, ignore_index=True) if preds_all else None
    out = []
    for k, s in enumerate(segs):
        ic = s.groupby("date").apply(
            lambda g: g["p"].corr(g["fwd"], method="spearman") if len(g) > 30 else np.nan)
        ic = ic.dropna()
        top = s.sort_values("p", ascending=False).groupby("date").head(10)
        out.append({"seg": k, "n": int(len(s)),
                    "start": str(pd.Timestamp(s["date"].min()).date()),
                    "end": str(pd.Timestamp(s["date"].max()).date()),
                    "ic_mean": round(float(ic.mean()), 4),
                    "ic_pos": round(float((ic > 0).mean()), 3),
                    "top10_fwd": round(float(top["fwd"].mean()) * 100, 4),
                    "all_fwd": round(float(s["fwd"].mean()) * 100, 4)})
    return out, pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=10)
    a = ap.parse_args()
    t0 = time.time()
    df = build_extended_panel()
    all_dates = np.sort(df["date"].unique())
    bench = qd.index_hist_bs(BENCH)
    dev_v, hold_v, _ = vol_forecasts(bench, list(va.FEATS), 10, va.Cfg())
    volmap = {pd.Timestamp(x): v for x, v in zip(dev_v["date"], dev_v["p"])}
    volmap.update({pd.Timestamp(x): v for x, v in zip(hold_v["date"], hold_v["p"])})

    feats_all = [f for f in ta.ALL_FEATS if f in df.columns]
    feats_nomkt = [f for f in feats_all if f not in ta.FEATS_MARKET]
    h = a.h
    print("=" * 104)
    print(f"建议②: 换目标函数 —— 直接预测收益 vs 预测“是否高于分位” (h={h}, 多段滚动留出 {N_SEG} 段)")
    print("=" * 104)

    # 连续收益目标(截尾)
    lo, hi = WINSOR
    df[f"yret{h}"] = df[f"fwd{h}"].clip(lo, hi)
    df[f"yrank{h}"] = df.groupby("date")[f"fwd{h}"].rank(pct=True) - 0.5

    cases = [
        ("回归-截尾收益 (绝对误差)", f"yret{h}", "reg", "hgbr"),
        ("回归-截尾收益 (平方误差)", f"yret{h}", "reg", "hgbr2"),
        ("回归-截尾收益 (岭回归)", f"yret{h}", "reg", "ridge"),
        ("回归-横截面排名 (0~1)", f"yrank{h}", "reg", "hgbr"),
        ("分类-是否进前25%", f"ytop{h}" if f"ytop{h}" in df.columns else "yrel" + str(h),
         "cls", "hgb"),
        ("分类-是否跑赢中位数", f"yrel{h}", "cls", "hgb"),
    ]
    # 补上 ytop
    if f"ytop{h}" not in df.columns:
        rk = df.groupby("date")[f"fwd{h}"].rank(pct=True)
        df[f"ytop{h}"] = np.where(df[f"fwd{h}"].isna(), np.nan, (rk > 0.75).astype(float))
    cases[4] = ("分类-是否进前25%", f"ytop{h}", "cls", "hgb")

    rows, pooled_store = [], {}
    for name, tgt, task, kind in cases:
        for fname, feats in (("all", feats_all), ("nomkt", feats_nomkt)):
            segs, pooled = wf_multi(df, feats, tgt, h, kind, task=task)
            if not segs:
                print(f"  [skip] {name} / {fname}")
                continue
            ic_all = np.mean([s["ic_mean"] for s in segs])
            ic_pos = np.mean([s["ic_pos"] for s in segs])
            # 组合: K=10, 波动率仓位
            met, cur, ew = portfolio(pooled, 10, all_dates, h, volmap, True)
            met_f, _, _ = portfolio(pooled, 10, all_dates, h, volmap, False)
            rec = {"target": name, "feats": fname, "ic_mean": round(float(ic_all), 4),
                   "ic_pos_rate": round(float(ic_pos), 3), "segs": segs,
                   "excess_vol": None if not met else met["excess_ann"],
                   "excess_full": None if not met_f else met_f["excess_ann"],
                   "sharpe": None if not met else met["sharpe"],
                   "top10_fwd": round(float(np.mean([s["top10_fwd"] for s in segs])), 4),
                   "all_fwd": round(float(np.mean([s["all_fwd"] for s in segs])), 4)}
            rows.append(rec)
            pooled_store[(name, fname)] = pooled
            print(f"  {name:<24} {fname:<6} rankIC {ic_all:+.4f} (正比例 {ic_pos:.0%}) | "
                  f"组合超额 波动率仓位 {rec['excess_vol']} / 满仓 {rec['excess_full']} | "
                  f"{time.time()-t0:.0f}s", flush=True)

    print("\n" + "=" * 104)
    print("逐段明细（看稳定性，而不是看平均）")
    print("=" * 104)
    for r in rows:
        if r["feats"] != "nomkt":
            continue
        print(f"\n{r['target']}:")
        for s in r["segs"]:
            print(f"  段{s['seg']} {s['start']}~{s['end']} n={s['n']:<7} "
                  f"rankIC {s['ic_mean']:+.4f} 正比例 {s['ic_pos']:.0%} | "
                  f"top10 {s['top10_fwd']}% vs 全池 {s['all_fwd']}%")

    (OUT / "tail_ltr.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2,
                                                 default=str), encoding="utf-8")
    setup_font()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    labs = [f"{r['target']}\n{r['feats']}" for r in rows]
    x = np.arange(len(rows))
    axes[0].bar(x, [r["ic_mean"] or 0 for r in rows], color="#4c72b0")
    axes[0].axhline(0, color="k", lw=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labs, fontsize=7, rotation=25)
    axes[0].set_ylabel("rank IC (每日 Spearman 均值)")
    axes[0].set_title("排序信息系数: 换目标函数后是否更有信息量")
    axes[0].grid(alpha=0.25)
    axes[1].bar(x - 0.2, [r["excess_vol"] or 0 for r in rows], width=0.4,
                color="#c44e52", label="超额(波动率仓位)")
    axes[1].bar(x + 0.2, [r["excess_full"] or 0 for r in rows], width=0.4,
                color="#55a868", label="超额(满仓)")
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labs, fontsize=7, rotation=25)
    axes[1].set_ylabel("相对全池等权的超额年化")
    axes[1].set_title("同口径组合表现")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "tail_ltr.png", dpi=130)
    plt.close(fig)
    print(f"\n总用时 {time.time()-t0:.0f}s; 图 {OUT/'tail_ltr.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
