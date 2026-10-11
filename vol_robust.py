#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方向 C 稳健性检验 vol_robust.py
===============================
把"波动率预测到底有没有比基准强"这件事放到 5 个指数 × 3 个预测周期 上同时检验:
  模型: 全 40 个波动率特征 + 岭回归(对数波动率) + Duan smearing 修正
  基准1: 朴素 —— 用过去 20 日已实现波动率外推
  基准2: HAR-RV 古典模型 —— 用 rv5 / rv20 / rv60 做 OLS(文献标准基准, 比朴素强得多)
  损失: QLIKE; 检验: Diebold-Mariano (非重叠 h 日期间)
开发期训练, 最终留出集(最后 20%)只测一次。
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
from train_agents import Blackboard, setup_font

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
CACHE = ROOT / "cache"

CODES = ["sh.000300", "sh.000001", "sh.000905", "sz.399006", "sz.399001"]
HS = (5, 10, 20)
HOLDOUT = 0.20


def fit_ridge(tr, te, feats, ycol):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    m.fit(tr[feats], tr[ycol])
    yhat_tr = m.predict(tr[feats])
    k = float(np.mean(np.exp(np.asarray(tr[ycol]) - yhat_tr)))      # Duan smearing
    return np.exp(m.predict(te[feats])) * k, k


def fit_har(tr, te, h):
    from sklearn.linear_model import LinearRegression
    cols = ["rv5", "rv20", "rv60"]
    m = LinearRegression()
    m.fit(tr[cols], tr[f"logyvol{h}"])
    yhat_tr = m.predict(tr[cols])
    k = float(np.mean(np.exp(np.asarray(tr[f"logyvol{h}"]) - yhat_tr)))
    return np.exp(m.predict(te[cols])) * k, k


def evaluate(te, h, col_model, col_naive, col_har):
    """非重叠 h 日期间上的 QLIKE 与 DM 检验。"""
    po = te.sort_values("date").iloc[::h]
    tv = po[f"yvol{h}"].values
    def L(col):
        p = po[col].values
        r = (tv ** 2) / np.maximum(p ** 2, 1e-12)
        return r - np.log(r) - 1.0
    lm, ln, lh = L(col_model), L(col_naive), L(col_har)
    pm, pn = po[col_model].values, po[col_naive].values
    out = {"n": int(len(po)),
           "qlike_model": round(float(np.mean(lm)), 5),
           "qlike_naive": round(float(np.mean(ln)), 5),
           "qlike_har": round(float(np.mean(lh)), 5),
           "gain_naive": round(float(np.mean(ln - lm)), 5),
           "gain_har": round(float(np.mean(lh - lm)), 5),
           "p_naive": va.dm_test(lm - ln)[1],
           "p_har": va.dm_test(lm - lh)[1],
           "corr_model": round(float(np.corrcoef(tv, pm)[0, 1]), 3),
           "corr_naive": round(float(np.corrcoef(tv, pn)[0, 1]), 3)}
    return out


def run_index(code, name, cfg, result):
    df = qd.index_hist_bs(code)
    if df is None or len(df) < 800:
        print(f"  {name}: 数据不足")
        return
    bb = Blackboard()
    f = va.VolFeatureAgent(bb, cfg).build(df)
    f = va.VolLabelAgent(bb, cfg).run(f)
    dates = np.sort(f["date"].unique())
    dev_end = dates[int(len(dates) * (1 - HOLDOUT))]
    for h in HS:
        feats = list(va.FEATS)
        ycol = f"logyvol{h}"
        tr = f[f["date"] <= dev_end].dropna(subset=feats + [ycol, f"yvol{h}"])
        te = f[f["date"] > dev_end].dropna(subset=feats + [ycol, f"yvol{h}"]).copy()
        if len(tr) < 400 or len(te) < 120:
            continue
        pm, _ = fit_ridge(tr, te, feats, ycol)
        ph, _ = fit_har(tr, te, h)
        te_ = te.copy()
        te_["pm"], te_["ph"] = pm, ph
        ho = evaluate(te_, h, "pm", "rv20", "ph")
        # 开发期也跑一遍做对比(用最后一个折之前的训练? 简化: 用前 80% 内的时间切分)
        d2 = dates[:int(len(dates) * (1 - HOLDOUT))]
        split = d2[int(len(d2) * 0.7)]
        tr2 = f[f["date"] <= split].dropna(subset=feats + [ycol, f"yvol{h}"])
        te2 = f[(f["date"] > split) & (f["date"] <= dev_end)].dropna(
            subset=feats + [ycol, f"yvol{h}"]).copy()
        dev = None
        if len(tr2) > 400 and len(te2) > 120:
            pm2, _ = fit_ridge(tr2, te2, feats, ycol)
            ph2, _ = fit_har(tr2, te2, h)
            te2["pm"], te2["ph"] = pm2, ph2
            dev = evaluate(te2, h, "pm", "rv20", "ph")
        result.append({"index": name, "code": code, "h": h, "dev": dev, "holdout": ho})
        print(f"  {name:<8} h={h:<3} 留出集 QLIKE改进: vs朴素 {ho['gain_naive']:+.4f} "
              f"(p={ho['p_naive']}) vsHAR {ho['gain_har']:+.4f} (p={ho['p_har']}) | "
              f"相关性 {ho['corr_model']} vs {ho['corr_naive']}", flush=True)


def main():
    cfg = va.Cfg()
    print("=" * 96)
    print("方向 C 稳健性检验: 5 个指数 × 3 个预测周期, 开发期训练 / 最终留出集只测一次")
    print("=" * 96)
    result = []
    for code in CODES:
        run_index(code, qd.BS_INDICES.get(code, code), cfg, result)

    (OUT / "vol_robust.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    rows = []
    for r in result:
        if r["holdout"]:
            rows.append({"指数": r["index"], "周期": f"h={r['h']}",
                         "留出集QLIKE改进(vs朴素)": r["holdout"]["gain_naive"],
                         "p值": r["holdout"]["p_naive"],
                         "留出集QLIKE改进(vsHAR)": r["holdout"]["gain_har"],
                         "p值(HAR)": r["holdout"]["p_har"],
                         "相关性(模型)": r["holdout"]["corr_model"],
                         "相关性(朴素)": r["holdout"]["corr_naive"]})
    tab = pd.DataFrame(rows)
    print("\n" + "=" * 96)
    print("留出集结果汇总")
    print("=" * 96)
    print(tab.to_string(index=False))
    if len(tab):
        pos_n = int((tab["留出集QLIKE改进(vs朴素)"] > 0).sum())
        pos_h = int((tab["留出集QLIKE改进(vsHAR)"] > 0).sum())
        sig_n = int((tab["p值"] < 0.10).sum())
        sig_h = int((tab["p值(HAR)"] < 0.10).sum())
        print(f"\n相对朴素基准: {pos_n}/{len(tab)} 个 (指数,周期) 组合为正, 其中 {sig_n} 个 p<0.10")
        print(f"相对 HAR 基准: {pos_h}/{len(tab)} 个为正, 其中 {sig_h} 个 p<0.10")

    setup_font()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    piv_n = tab.pivot(index="指数", columns="周期", values="留出集QLIKE改进(vs朴素)")
    piv_h = tab.pivot(index="指数", columns="周期", values="留出集QLIKE改进(vsHAR)")
    for ax, piv, ttl in ((axes[0], piv_n, "相对朴素基准(过去20日波动率外推)"),
                         (axes[1], piv_h, "相对 HAR-RV 古典模型")):
        im = ax.imshow(piv.values, cmap="RdYlGn", vmin=-np.nanmax(np.abs(piv.values)),
                       vmax=np.nanmax(np.abs(piv.values)))
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=10)
        ax.set_title(f"留出集 QLIKE 改进 {ttl}\n(绿=比基准好, 红=比基准差)")
        plt.colorbar(im, ax=ax)
    fig.suptitle("方向C 稳健性: 5 指数 × 3 周期, 留出集(2022-06 起)表现", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "vol_robust.png", dpi=130)
    plt.close(fig)
    print(f"\n图: {OUT / 'vol_robust.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
