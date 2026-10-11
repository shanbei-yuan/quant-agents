#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第3条建议: 补齐与 HAR-RV 古典模型的差距 vol_har.py
==================================================
诊断: 在 vol_robust 中, 模型相对"朴素基准"14/15 为正, 但相对 **HAR-RV 古典模型**
只有 9/15 为正 —— 说明模型还没把古典模型的信息榨干。

做法: 把 HAR 家族的特征显式补进特征集, 再跑同一个稳健性矩阵, 看能否把
"击败 HAR 的格子数"从 9/15 提上去。新增特征:
  * 多尺度 rv1(日) —— HAR 的日频项
  * 半方差 rsv_pos / rsv_neg (HAR-SV, Patton & Sheppard 2015) 及符号跳跃 sjv
  * 跳跃成分 jump = max(RV - BPV, 0)/RV (HAR-J, Barndorff-Nielsen & Shephard)
  * 已实现四次变差 rq 及其比值 (HARQ, Bollerslev 等)
  * 杠杆效应: 负收益日占比、r 与 |r| 的滚动相关
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import quant_data as qd
import vol_agents as va
from vol_robust import CODES, HS, HOLDOUT, fit_ridge, fit_har, evaluate
from train_agents import Blackboard, setup_font

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"

HAR_FEATS = ["rv1", "rsv_pos5", "rsv_neg5", "rsv_pos20", "rsv_neg20", "sjv",
             "jump", "rq", "rq_ratio", "neg_share20", "corr_lev", "rv250"]


def har_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    r = c.pct_change()
    av = np.sqrt(250)
    f = pd.DataFrame(index=df.index)
    f["rv1"] = r.abs() * av
    pos = r.clip(lower=0)
    neg = (-r.clip(upper=0))
    for n in (5, 20):
        f[f"rsv_pos{n}"] = ((pos ** 2).rolling(n).mean() ** 0.5) * av
        f[f"rsv_neg{n}"] = ((neg ** 2).rolling(n).mean() ** 0.5) * av
    p2, n2 = f["rsv_pos20"] ** 2, f["rsv_neg20"] ** 2
    f["sjv"] = (p2 - n2) / (p2 + n2).replace(0, np.nan)
    rv20 = (r ** 2).rolling(20).mean() * 250.0
    bpv = (np.pi / 2) * (r.abs() * r.abs().shift()).rolling(20).mean() * 250.0
    f["jump"] = np.maximum(rv20 - bpv, 0) / np.maximum(rv20, 1e-12)
    f["rq"] = (r ** 4).rolling(20).mean() * (250.0 ** 2)
    f["rq_ratio"] = f["rq"] / np.maximum(rv20, 1e-12) ** 2
    f["neg_share20"] = (r < 0).rolling(20).mean()
    f["corr_lev"] = r.rolling(20).corr(r.abs())
    f["rv250"] = r.rolling(250).std() * av
    return f.reset_index()


def fit_ridge_a(tr, te, feats, ycol, alpha):
    """带正则强度参数的岭回归(带 Duan smearing)。"""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    m.fit(tr[feats], tr[ycol])
    k = float(np.mean(np.exp(np.asarray(tr[ycol]) - m.predict(tr[feats]))))
    return np.exp(m.predict(te[feats])) * k


VARIANTS = (("基础", None, 10.0), ("HAR增强", "HAR", 10.0),
            ("HAR增强·强正则", "HAR", 100.0))


def main():
    cfg = va.Cfg()
    bb = Blackboard()
    rows = []
    print("=" * 100)
    print("第3条建议: 补齐 HAR 差距 —— 基础特征集 vs HAR 增强特征集")
    print("=" * 100)
    for code in CODES:
        nm = qd.BS_INDICES.get(code, code)
        df = qd.index_hist_bs(code)
        if df is None:
            continue
        f = va.VolFeatureAgent(bb, cfg).build(df)
        f = f.merge(har_features(df), on="date", how="left")
        f = va.VolLabelAgent(bb, cfg).run(f)
        dates = np.sort(f["date"].unique())
        dev_end = dates[int(len(dates) * (1 - HOLDOUT))]
        for h in HS:
            ycol = f"logyvol{h}"
            te = f[f["date"] > dev_end].dropna(
                subset=va.FEATS + HAR_FEATS + [ycol, f"yvol{h}"]).copy()
            tr = f[f["date"] <= dev_end].dropna(
                subset=va.FEATS + HAR_FEATS + [ycol, f"yvol{h}"])
            if len(tr) < 400 or len(te) < 120:
                continue
            harp, _ = fit_har(tr, te, h)
            for tag, use_har, alpha in VARIANTS:
                feats = list(va.FEATS) + (HAR_FEATS if use_har else [])
                pm = fit_ridge_a(tr, te, feats, ycol, alpha)
                t2 = te.copy()
                t2["pm"], t2["ph"] = pm, harp
                r = evaluate(t2, h, "pm", "rv20", "ph")
                r.update({"index": nm, "h": h, "feat": tag, "n_feats": len(feats),
                          "alpha": alpha})
                rows.append(r)
            print(f"  {nm:<8} h={h:<3} " + " | ".join(
                f"{rows[-len(VARIANTS)+i]['feat']} vsHAR {rows[-len(VARIANTS)+i]['gain_har']:+.4f}"
                for i in range(len(VARIANTS))), flush=True)

    tab = pd.DataFrame(rows)
    (OUT / "vol_har.json").write_text(
        json.dumps(tab.to_dict("records"), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print("\n" + "=" * 100)
    for tag, _, alpha in VARIANTS:
        g = tab[tab["feat"] == tag]
        if g.empty:
            continue
        pn = int((g["gain_naive"] > 0).sum())
        ph = int((g["gain_har"] > 0).sum())
        sn = int((g["p_naive"] < 0.10).sum())
        sh = int((g["p_har"] < 0.10).sum())
        print(f"{tag:<14} 特征数 {int(g['n_feats'].iloc[0]):>3} alpha={alpha:<6.0f} | "
              f"vs朴素 {pn}/{len(g)} 为正({sn} 显著) | "
              f"vsHAR {ph}/{len(g)} 为正({sh} 显著) | "
              f"平均 vsHAR 改进 {g['gain_har'].mean():+.5f}")
    print("=" * 100)

    # 出图: 两种特征集 vs HAR 的改进对比
    setup_font()
    piv = {}
    for tag, _, _alpha in VARIANTS:
        g = tab[tab["feat"] == tag].copy()
        if g.empty:
            continue
        g["格子"] = g["index"] + " h=" + g["h"].astype(str)
        piv[tag] = g.set_index("格子")["gain_har"]
    piv = pd.DataFrame(piv)
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(piv))
    w = 0.8 / max(1, len(piv.columns))
    cols = ["#4c72b0", "#55a868", "#c44e52"]
    for i, col in enumerate(piv.columns):
        ax.bar(x + (i - (len(piv.columns) - 1) / 2) * w, piv[col], width=w,
               color=cols[i % 3], label=col)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, fontsize=8)
    ax.set_ylabel("相对 HAR-RV 古典模型的 QLIKE 改进")
    ax.set_title("补齐 HAR 差距: 加入半方差/跳跃/四次变差/杠杆特征后的改进\n"
                 "(正值=比 HAR 好; 柱子越高越好)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "vol_har.png", dpi=130)
    plt.close(fig)
    print(f"图: {OUT / 'vol_har.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
