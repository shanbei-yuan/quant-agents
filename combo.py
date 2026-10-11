#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第2条建议: 合体策略 combo.py
===========================
把两个**各自被验证有效**的零件拼起来:
  选股零件 = 方向A 的横截面相对模型(预测"是否跑赢当天全场中位数", 4/4 折稳定)
  仓位零件 = 方向C 的波动率目标(回撤砍掉约 40%)

规则(每 h 个交易日调仓一次):
  1. 用横截面模型给当天所有股票打分, 取分数最高的 K 只
  2. 总仓位 = min(1, 目标波动率 / 沪深300 预测波动率)
  3. 组合内等权; 成交按换手计费(往返 0.6%)
  4. 基准: ① 沪深300 买入持有 ② 全池等权(市场代理)

对比 4 个方案: 满仓+全池等权 / 满仓+选股 / 波动率仓位+全池 / 波动率仓位+选股

严谨性: 选股模型与波动率模型都严格 walk-forward(开发期) 或开发期训练后只测一次(留出期)。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import quant_data as qd
import train_agents as ta
import vol_agents as va
from vol_target import forecasts as vol_forecasts
from train_agents import Blackboard, setup_font

ROOT = Path(__file__).resolve().parent
CACHE, OUT = ROOT / "cache", ROOT / "out"

COST_STOCK = 0.006      # 个股往返成本(佣金+印花税+滑点)
H = 5                   # 调仓与预测周期
TARGET_VOL = 0.12       # 目标波动率
HOLDOUT = 0.20
BENCH = "sh.000300"


def load_stocks(max_n: int | None = None):
    data = {}
    for f in sorted(CACHE.glob("kline_*.csv")):
        code = f.stem.replace("kline_", "")
        try:
            df = pd.read_csv(f, parse_dates=["date"]).set_index("date").sort_index()
        except Exception:                          # noqa: BLE001
            continue
        cols = [c for c in ["open", "close", "high", "low", "volume", "amount"] if c in df.columns]
        if len(df) < 400 or len(cols) < 5:
            continue
        data[code] = df[cols]
        if max_n and len(data) >= max_n:
            break
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-stocks", type=int, default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()

    print("=" * 96)
    print("第2条建议: 横截面选股 + 波动率仓位 合体策略")
    print("=" * 96)
    data = load_stocks(a.max_stocks)
    print(f"载入个股 {len(data)} 只", flush=True)

    cfg = ta.Config(universe_n=len(data), fast=a.fast, cached_only=True)
    bb = Blackboard()
    bench = qd.index_hist_bs(BENCH)
    panel = ta.FeatureAgent(bb, cfg).run(data, bench)
    df = ta.LabelAgent(bb, cfg).run(panel)
    feats = [f for f in ta.ALL_FEATS if f in df.columns]
    dates = np.sort(df["date"].unique())
    dev_end = dates[int(len(dates) * (1 - HOLDOUT))]
    print(f"[面板] {len(df):,} 行 × {len(feats)} 特征; 开发期截止 "
          f"{dev_end.astype('datetime64[D]')}", flush=True)

    # 1) 选股模型: 横截面相对标签, walk-forward(开发期) + 留出期
    ycol, fwdcol = f"yrel{H}", f"fwd{H}"
    ck = CACHE / f"combo_preds_{len(data)}.pkl"
    if ck.exists():
        o = pd.read_pickle(ck)
        wf_pooled, te, dates, dev_end = o["wf"], o["te"], o["dates"], o["dev_end"]
        print(f"[选股] 读缓存预测: 开发期 {len(wf_pooled):,} 留出期 {len(te):,}", flush=True)
    else:
        wf = ta.ModelAgent(bb, cfg).walk_forward(df, feats, H, "hgb", tag="COMBO",
                                                 label="rel", end_date=dev_end)
        tr = df[df["date"] <= dev_end].dropna(subset=feats + [ycol])
        te = df[df["date"] > dev_end].dropna(subset=feats + [fwdcol]).copy()
        m = ta.ModelAgent(bb, cfg).make_model("hgb")
        m.fit(tr[feats], tr[ycol])
        te["p"] = m.predict_proba(te[feats])[:, 1]
        wf_pooled = wf["pooled"]
        pd.to_pickle({"wf": wf_pooled, "te": te, "dates": dates, "dev_end": dev_end}, ck)
        print(f"[选股] 开发期样本外 {len(wf_pooled):,} 条, 留出期 {len(te):,} 条", flush=True)

    # 2) 仓位模型: 沪深300 波动率预测
    dev_v, hold_v, _ = vol_forecasts(bench, list(va.FEATS), 10, va.Cfg())
    volmap = dict(zip(dev_v["date"], dev_v["p"]))
    volmap.update(dict(zip(hold_v["date"], hold_v["p"])))
    print(f"[仓位] 波动率预测 {len(volmap)} 条", flush=True)

    # ---- 选股模型的样本外统计 (即"池子扩大后 方向A 的表现") ----
    from sklearn.metrics import roc_auc_score
    print("=" * 96)
    print(f"横截面选股模型 样本外统计 (池子 {len(data)} 只, 标签=是否跑赢当天中位数, h={H})")
    print("=" * 96)
    mg_all = df[["date", "code", fwdcol]].dropna().rename(columns={fwdcol: "fwd"})
    for phase, pr in (("开发期", wf_pooled), ("留出期", te)):
        if ycol not in pr.columns:
            continue
        x = pr.dropna(subset=["p", ycol]).merge(mg_all, on=["date", "code"], how="left")
        y = x[ycol].values
        pp = x["p"].values
        acc = float(((pp > 0.5).astype(int) == y).mean())
        base = float(max(y.mean(), 1 - y.mean()))
        auc = float(roc_auc_score(y, pp)) if len(set(y)) > 1 else float("nan")
        topk = x.sort_values("p", ascending=False).groupby("date").head(a.k)
        print(f"{phase}: 样本 {len(x):,} | 命中率 {acc:.4f} vs 基线 {base:.4f} "
              f"(超额 {acc-base:+.4f}) | AUC {auc:.4f} | top{a.k} 平均 5 日收益 "
              f"{topk['fwd'].mean()*100:.3f}% vs 全池 {x['fwd'].mean()*100:.3f}%", flush=True)
    print("=" * 96)

    # 3) 回测
    def backtest(preds, use_vol, k):
        """preds: DataFrame(date, code, fwd, p)。返回净值曲线与指标。"""
        preds = preds.dropna(subset=["p", "fwd"])
        date_ix = {pd.Timestamp(x): i for i, x in enumerate(dates)}
        curve, prev, eq = [], set(), 1.0
        wins, rets, expos, used = [], [], [], []
        for d, g in preds.groupby("date"):
            i = date_ix.get(pd.Timestamp(d))
            if i is None or (i % H) != 0:
                continue
            g = g.sort_values("p", ascending=False)
            sel = g.head(k)
            if len(sel) < 3:
                continue
            expo = 1.0
            if use_vol:
                v = volmap.get(d)
                if v is None or not np.isfinite(v) or v <= 0:
                    continue
                expo = float(np.clip(TARGET_VOL / v, 0.0, 1.0))
            newset = set(sel["code"])
            turn = len(newset - prev) / max(1, k)
            r_gross = float(sel["fwd"].mean())
            r = expo * r_gross - COST_STOCK * turn * expo
            eq *= (1 + r)
            curve.append(eq)
            rets.append(r)
            expos.append(expo)
            wins.append(1 if r > 0 else 0)
            prev = newset
            used.append(pd.Timestamp(d))
        if len(curve) < 20:
            return None
        curve = np.array(curve)
        rets = np.array(rets)
        yrs = len(curve) * H / 250.0
        ann = float(curve[-1] ** (1 / yrs) - 1)
        mdd = float((curve / np.maximum.accumulate(curve) - 1).min())
        sd = float(np.std(rets, ddof=1))
        return {"ann_ret": round(ann, 4), "total_ret": round(float(curve[-1] - 1), 4),
                "max_dd": round(mdd, 4), "sharpe": round(float(np.mean(rets) / sd * np.sqrt(250 / H)), 3) if sd else None,
                "calmar": round(ann / abs(mdd), 3) if mdd else None,
                "win_rate": round(float(np.mean(wins)), 4), "avg_expo": round(float(np.mean(expos)), 3),
                "n_period": len(curve), "years": round(yrs, 2), "curve": curve,
                "dates_used": used}

    rows = []
    results = {}
    used_dates = {}
    for phase, preds in (("开发期", wf_pooled), ("留出期", te)):
        p = preds[["date", "code", "p"]].copy()
        mg = df[["date", "code", fwdcol]].dropna().rename(columns={fwdcol: "fwd"})
        p = p.merge(mg, on=["date", "code"], how="inner")
        for k in (5, 10, 20):
            for use_vol in (False, True):
                r = backtest(p, use_vol, k)
                if r:
                    r.update({"phase": phase, "k": k,
                              "expo": "波动率仓位" if use_vol else "满仓"})
                    rows.append(r)
                    results[(phase, k, use_vol)] = r["curve"]
                    used_dates[(phase, k, use_vol)] = r["dates_used"]
        # 全池等权(市场代理)
        eq_all = p.groupby("date")["fwd"].mean()
        eq_all = eq_all.iloc[::H]
        c = (1 + eq_all).cumprod().values
        yrs = len(c) * H / 250.0
        ann = c[-1] ** (1 / yrs) - 1
        mdd = (c / np.maximum.accumulate(c) - 1).min()
        sd = np.std(eq_all.values, ddof=1)
        rows.append({"phase": phase, "k": "全池等权", "expo": "满仓",
                     "ann_ret": round(ann, 4), "total_ret": round(c[-1] - 1, 4),
                     "max_dd": round(mdd, 4),
                     "sharpe": round(float(np.mean(eq_all.values) / sd * np.sqrt(250 / H)), 3),
                     "calmar": round(ann / abs(mdd), 3) if mdd else None,
                     "win_rate": round(float((eq_all > 0).mean()), 4), "avg_expo": 1.0,
                     "n_period": len(c), "years": round(yrs, 2), "curve": c})

        # 沪深300 买入持有(同期, 对比基准)
        r0 = used_dates.get((phase, 10, False))
        if r0:
            du = r0
            i0 = bench.index.get_indexer([du[0]])[0]
            i1 = bench.index.get_indexer([du[-1]])[0]
            if i0 >= 0 and i1 > i0:
                c300 = (bench["close"].iloc[i0:i1 + 1] / bench["close"].iloc[i0]).values
                yy = len(du) * H / 250.0
                a3 = c300[-1] ** (1 / yy) - 1
                m3 = (c300 / np.maximum.accumulate(c300) - 1).min()
                d4 = np.diff(bench["close"].iloc[i0:i1 + 1].pct_change().dropna())
                rows.append({"phase": phase, "k": "沪深300", "expo": "买入持有",
                             "ann_ret": round(float(a3), 4), "total_ret": round(float(c300[-1] - 1), 4),
                             "max_dd": round(float(m3), 4), "sharpe": None,
                             "calmar": None, "win_rate": None, "avg_expo": 1.0,
                             "n_period": len(c300), "years": round(yy, 2), "curve": c300})

    tab = pd.DataFrame(rows)
    cols = ["phase", "k", "expo", "ann_ret", "total_ret", "max_dd", "sharpe", "calmar",
            "win_rate", "avg_expo", "n_period", "years"]
    print("\n" + "=" * 96)
    print(tab[cols].to_string(index=False))
    print("=" * 96)

    (OUT / "combo.json").write_text(json.dumps(
        tab.drop(columns=["curve"]).to_dict("records"), ensure_ascii=False, indent=2,
        default=str), encoding="utf-8")

    # 出图: 留出期净值
    setup_font()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    for ax, phase in ((axes[0], "开发期"), (axes[1], "留出期")):
        for k in (5, 10, 20):
            for uv, col, ls in ((False, "#c44e52", "-"), (True, "#55a868", "--")):
                key = (phase, k, uv)
                if key in results:
                    lab = f"K={k} {'波动率仓位' if uv else '满仓'}"
                    ax.plot(results[key], color=col, ls=ls, lw=1.5, label=lab)
        allkey = (phase, "全池", "满仓")
        ax.axhline(1.0, color="k", lw=0.7)
        ax.set_yscale("log")
        ax.set_title(f"{phase}: 横截面选股 × 仓位方案 净值")
        ax.set_xlabel(f"调仓次数(每 {H} 个交易日一次)")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "combo.png", dpi=130)
    plt.close(fig)
    print(f"图: {OUT / 'combo.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
