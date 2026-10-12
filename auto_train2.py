#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自主训练 第二阶段 auto_train2.py
================================
第一阶段(192 个配置)只在开发期搜索。本阶段:
  1. 读排行榜, 输出所有配置的分布 —— 量化"多重比较"带来的选择偏差
  2. 按 (标签, 周期) 分组做**概率集合**(同组内多个模型求平均), 通常比单体更稳
  3. 围绕第一阶段最优配置做**局部精调**(超参 + 训练样本上限 + K)
  4. 最后**只对开发期最优者与集合模型在留出集上各评估一次**

纪律: 留出集在整个搜索+精调期间一次都不用。
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

from auto_train import (build_extended_panel, portfolio, acc_stats, load_stocks,
                        vol_forecasts, HS, HOLDOUT, MIN_TRAIN, TEST_LEN, TARGET_VOL,
                        COST_STOCK, BENCH, CACHE, OUT)
import quant_data as qd
import train_agents as ta
import vol_agents as va
from train_agents import setup_font

FEATS = {"all": None, "nomkt": None, "mom": None, "rank": None}


def make_model(kind, seed, params):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    p = params or {}
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=p.get("max_iter", 60), learning_rate=p.get("lr", 0.06),
            max_leaf_nodes=p.get("leaves", 15),
            min_samples_leaf=p.get("min_leaf", 40),
            l2_regularization=p.get("l2", 1.0), early_stopping=True,
            n_iter_no_change=10, validation_fraction=0.15, random_state=seed)
    if kind == "hgbdeep":
        return HistGradientBoostingClassifier(
            max_iter=p.get("max_iter", 120), learning_rate=p.get("lr", 0.04),
            max_leaf_nodes=p.get("leaves", 31),
            min_samples_leaf=p.get("min_leaf", 60),
            l2_regularization=p.get("l2", 2.0), early_stopping=True,
            n_iter_no_change=12, validation_fraction=0.15, random_state=seed)
    if kind == "logit":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=300, C=p.get("C", 0.1), n_jobs=-1))
    if kind == "rf":
        return RandomForestClassifier(n_estimators=p.get("n_est", 40),
                                      max_depth=p.get("depth", 6),
                                      min_samples_leaf=p.get("min_leaf", 150),
                                      n_jobs=-1, random_state=seed)
    raise ValueError(kind)


def wf(df, feats, ycol, h, kind, dev_end, params=None, cap=120000, seed=0):
    w = df[["date", "code", f"fwd{h}", ycol] + feats].dropna()
    if len(w) < 5000:
        return None, None
    D = w["date"].values
    C = w["code"].values
    Y = w[ycol].values.astype(np.float32)
    F = w[f"fwd{h}"].values.astype(np.float32)
    A = w[feats].to_numpy(dtype=np.float32)
    dates = np.sort(np.unique(D))
    dates = dates[dates <= np.datetime64(dev_end)]
    rng = np.random.default_rng(seed)
    out = []
    for si in range(MIN_TRAIN, len(dates), TEST_LEN):
        te_dates = dates[si:si + TEST_LEN]
        if len(te_dates) < 60:
            break
        d0, d1 = te_dates[0], te_dates[-1]
        tr = np.where(D < d0)[0]
        tei = np.where((D >= d0) & (D <= d1))[0]
        if len(tr) < 3000 or len(tei) < 300:
            continue
        if len(tr) > cap:
            tr = rng.choice(tr, cap, replace=False)
        m = make_model(kind, seed, params)
        m.fit(A[tr], Y[tr])
        out.append(pd.DataFrame({"date": D[tei], "code": C[tei],
                                 "p": m.predict_proba(A[tei])[:, 1],
                                 "y": Y[tei], "fwd": F[tei]}))
    if not out:
        return None, None
    pooled = pd.concat(out, ignore_index=True)
    tr = np.where(D <= np.datetime64(dev_end))[0]
    tei = np.where(D > np.datetime64(dev_end))[0]
    if len(tr) > cap:
        tr = rng.choice(tr, cap, replace=False)
    mh = make_model(kind, seed, params)
    mh.fit(A[tr], Y[tr])
    hold = pd.DataFrame({"date": D[tei], "code": C[tei],
                         "p": mh.predict_proba(A[tei])[:, 1], "y": Y[tei], "fwd": F[tei]})
    return pooled, hold


def ensemble(preds_list):
    big = pd.concat([p[["date", "code", "p", "y", "fwd"]] for p in preds_list],
                    ignore_index=True)
    return big.groupby(["date", "code"], as_index=False).agg(
        p=("p", "mean"), y=("y", "first"), fwd=("fwd", "first"))


def eval_all(pooled, all_dates, h, volmap, tag, rows):
    for k in (3, 5, 8, 10, 15, 20, 30):
        for uv in (False, True):
            met, cur, ew = portfolio(pooled, k, all_dates, h, volmap, uv)
            if met is None:
                continue
            met.update({"tag": tag, "k": k, "use_vol": uv})
            rows.append(met)
    return rows


def yearly(pooled, all_dates, h, k, volmap, use_vol=True, cost=COST_STOCK):
    """逐年拆解: 组合 vs 全池等权, 用来看优势是否集中在少数几年。"""
    ix = {pd.Timestamp(d): i for i, d in enumerate(all_dates)}
    rows, prev = [], set()
    for d, g in pooled.sort_values("date").groupby("date"):
        i = ix.get(pd.Timestamp(d))
        if i is None or (i % h) != 0 or len(g) < max(20, 3 * k):
            continue
        g = g.sort_values("p", ascending=False)
        sel = g.head(k)
        expo = 1.0
        if use_vol:
            v = volmap.get(pd.Timestamp(d))
            if v is None or not np.isfinite(v) or v <= 0:
                continue
            expo = float(np.clip(TARGET_VOL / v, 0.0, 1.0))
        newset = set(sel["code"])
        turn = len(newset - prev) / max(1, k)
        rows.append({"date": pd.Timestamp(d),
                     "r": expo * float(sel["fwd"].mean()) - cost * turn * expo,
                     "rw": float(g["fwd"].mean())})
        prev = newset
    d0 = pd.DataFrame(rows)
    if d0.empty:
        return {}
    d0["year"] = d0["date"].dt.year
    out = {}
    for y, g in d0.groupby("year"):
        st = float((1 + g["r"]).prod() - 1)
        ew = float((1 + g["rw"]).prod() - 1)
        out[int(y)] = {"n": int(len(g)), "strat": round(st, 4), "ew": round(ew, 4),
                       "excess": round(st - ew, 4)}
    return out


def main():
    t0 = time.time()
    df = build_extended_panel()
    all_dates = np.sort(df["date"].unique())
    dev_end = all_dates[int(len(all_dates) * (1 - HOLDOUT))]
    bench = qd.index_hist_bs(BENCH)
    dev_v, hold_v, _ = vol_forecasts(bench, list(va.FEATS), 10, va.Cfg())
    volmap = {pd.Timestamp(x): v for x, v in zip(dev_v["date"], dev_v["p"])}
    volmap.update({pd.Timestamp(x): v for x, v in zip(hold_v["date"], hold_v["p"])})

    feats_all = [f for f in ta.ALL_FEATS if f in df.columns]
    feats_nomkt = [f for f in feats_all if f not in ta.FEATS_MARKET]
    feats_mom = [f for f in feats_all if f in (
        "ret1", "ret5", "ret10", "ret20", "ret60", "ma5_gap", "ma10_gap",
        "ma20_gap", "ma60_gap", "ma5_ma20", "ma20_ma60", "slope20_5",
        "slope60_20", "rsi14", "macd_hist", "vr_5_20", "vr_5_60")]
    feats_rank = [f for f in feats_all if f.startswith("rank_") or f in (
        "rs20", "pos60", "pos120", "dd20", "above_ma20", "breadth", "vol20", "vol60")]
    FS = {"all": feats_all, "nomkt": feats_nomkt, "mom": feats_mom, "rank": feats_rank}

    lb = [json.loads(l) for l in (OUT / "search_leaderboard.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    ex = [r["best"]["excess_ann"] for r in lb if r.get("best")]
    print("=" * 100)
    print(f"第一阶段共 {len(lb)} 个配置; 开发期 超额收益(对全池等权) 分布:")
    print(f"  均值 {np.mean(ex):+.4f}  中位 {np.median(ex):+.4f}  "
          f"最小 {np.min(ex):+.4f}  最大 {np.max(ex):+.4f}  "
          f"正比例 {np.mean(np.array(ex) > 0):.1%}")
    print("  -> 最大值几乎必然是选择偏差: 在 190 个配置里挑最好的, 噪声也会显得不错")
    print("=" * 100)

    # 按 (标签, 周期) 聚合
    agg = {}
    for r in lb:
        if not r.get("best"):
            continue
        key = (r["label"], r["h"])
        agg.setdefault(key, []).append(r["best"]["excess_ann"])
    rank = sorted(((k, float(np.mean(v)), len(v)) for k, v in agg.items()),
                  key=lambda x: -x[1])
    print("按 (标签, 周期) 平均超额排序:")
    for k, v, n in rank[:10]:
        print(f"  {k[0]:<6} h={k[1]:<3} 平均超额 {v:+.4f} (n={n})")
    print("=" * 100)

    best_key = rank[0][0]
    lab, h = best_key
    ycol = f"y{lab}{h}"
    print(f"选定 (标签, 周期) = {best_key}  → 开始集合 + 精调", flush=True)

    # 同 (标签,周期) 下的所有变体 -> 集合
    variants = [(r["kind"], r["feats"]) for r in lb if r["label"] == lab and r["h"] == h]
    print(f"集合成员 {len(variants)} 个: {variants}", flush=True)
    pool_dev, pool_hold = [], []
    for kind, fname in variants:
        p, hd = wf(df, FS[fname], ycol, h, kind, dev_end)
        if p is not None:
            pool_dev.append(p)
            pool_hold.append(hd)
    ens_dev = ensemble(pool_dev)
    ens_hold = ensemble(pool_hold)
    print(f"集合完成: 开发期 {len(ens_dev):,} 条, 留出期 {len(ens_hold):,} 条; "
          f"用时 {time.time()-t0:.0f}s", flush=True)

    # 局部精调
    grid = []
    for ml in (15, 40, 100):
        for l2 in (0.5, 1.0, 3.0):
            grid.append({"kind": "hgb", "params": {"min_leaf": ml, "l2": l2,
                                                   "max_iter": 150, "lr": 0.05}})
    grid.append({"kind": "hgbdeep", "params": {"min_leaf": 80, "l2": 3.0,
                                               "max_iter": 200, "lr": 0.03}})
    grid.append({"kind": "hgb", "params": {"min_leaf": 40, "l2": 1.0,
                                           "max_iter": 150, "lr": 0.05}})
    fine = []
    print(f"精调网格 {len(grid)} 个", flush=True)
    for i, g in enumerate(grid):
        p, hd = wf(df, feats_nomkt, ycol, h, g["kind"], dev_end,
                   params=g["params"], cap=250000)
        if p is None:
            continue
        st = acc_stats(p)
        rows_k = eval_all(p, all_dates, h, volmap, f"fine{i}", [])
        bs = max(rows_k, key=lambda x: x["excess_ann"]) if rows_k else {}
        fine.append({"i": i, "kind": g["kind"], "params": g["params"],
                     "acc": st.get("acc"), "edge": st.get("edge"), "auc": st.get("auc"),
                     "best": bs})
        print(f"  [fine{i}] {g['kind']} {g['params']} 命中 {st['acc']:.4f}"
              f"({st['edge']:+.4f}) AUC {st['auc']} | 最优 K={bs.get('k')} "
              f"超额 {bs.get('excess_ann')} 夏普 {bs.get('sharpe')}", flush=True)

    # 开发期总评
    dev_rows = eval_all(ens_dev, all_dates, h, volmap, "ensemble_dev", [])
    met_dev, cur_dev, ew_dev = portfolio(ens_dev, 10, all_dates, h, volmap, True)
    st_ens = acc_stats(ens_dev)
    print("\n" + "=" * 100)
    print(f"【集合模型】开发期: 命中 {st_ens['acc']:.4f} ({st_ens['edge']:+.4f}) "
          f"AUC {st_ens['auc']} | top10 平均 5 日 {st_ens['top10_fwd']}% vs 全池 "
          f"{st_ens['all_fwd']}%", flush=True)
    for r in sorted(dev_rows, key=lambda x: -x["excess_ann"])[:5]:
        print(f"  K={r['k']:<3} {'波动率' if r['use_vol'] else '满仓  '} "
              f"年化 {r['ann']} 超额 {r['excess_ann']:+.4f} 回撤 {r['mdd']} "
              f"夏普 {r['sharpe']} Calmar {r['calmar']}", flush=True)

    # ---- 最终留出集: 只评估一次 ----
    print("\n" + "=" * 100)
    print("【最终留出集】集合模型 (搜索期间从未使用这段数据)")
    print("=" * 100, flush=True)
    st_h = acc_stats(ens_hold)
    hold_rows = eval_all(ens_hold, all_dates, h, volmap, "ensemble_hold", [])
    print(f"命中 {st_h['acc']:.4f} ({st_h['edge']:+.4f}) AUC {st_h['auc']} | "
          f"top10 平均 5 日 {st_h['top10_fwd']}% vs 全池 {st_h['all_fwd']}%", flush=True)
    hold_tab = pd.DataFrame(hold_rows).sort_values("excess_ann", ascending=False)
    print(hold_tab[["k", "use_vol", "ann", "ew_ann", "excess_ann", "mdd", "sharpe",
                    "calmar", "win", "expo", "n", "years"]].to_string(index=False),
          flush=True)
    yd = yearly(ens_dev, all_dates, h, 10, volmap, True)
    yh = yearly(ens_hold, all_dates, h, 10, volmap, True)
    print("\n逐年拆解 (集合模型, K=10, 波动率仓位) 组合 vs 全池等权:")
    print(f"  {'年份':<6} {'期数':>4} {'组合':>9} {'全池':>9} {'超额':>9}")
    for tag, yy in (("开发期", yd), ("留出期", yh)):
        for y, v in yy.items():
            print(f"  {y:<6} {v['n']:>4} {v['strat']:>9.2%} {v['ew']:>9.2%} "
                  f"{v['excess']:>+9.2%}")
        print(f"  {'':<6} {'':>4} {'':>9} {'':>9} {'':>9}  ({tag})")

    out = {"ensemble_acc": st_ens, "hold_acc": st_h,
           "dev": dev_rows, "hold": hold_rows, "fine": fine,
           "yearly_dev": yearly(ens_dev, all_dates, h, 10, volmap, True),
           "yearly_hold": yearly(ens_hold, all_dates, h, 10, volmap, True),
           "dist": {"n": len(lb), "mean": float(np.mean(ex)), "max": float(np.max(ex)),
                    "pos_rate": float(np.mean(np.array(ex) > 0))},
           "rank": [{"label": k[0], "h": k[1], "mean_excess": v, "n": n}
                    for k, v, n in rank]}
    (OUT / "auto_train2.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    setup_font()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    axes[0].hist(ex, bins=25, color="#4c72b0", alpha=0.85)
    axes[0].axvline(0, color="k", lw=1.2)
    axes[0].axvline(float(np.mean(ex)), color="#c44e52", ls="--",
                    label=f"均值 {np.mean(ex):+.4f}")
    axes[0].set_title(f"第一阶段 {len(lb)} 个配置的开发期超额收益分布\n"
                      f"(正比例 {np.mean(np.array(ex)>0):.0%})")
    axes[0].set_xlabel("超额年化收益(组合 − 全池等权)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    if met_dev is not None and cur_dev is not None:
        axes[1].plot(cur_dev, color="#c44e52", lw=1.8, label="集合模型(波动率仓位)")
        axes[1].plot(ew_dev, color="#4c72b0", lw=1.4, label="全池等权")
        axes[1].axhline(1, color="k", lw=0.8)
        axes[1].set_yscale("log")
        axes[1].set_title("集合模型 vs 全池等权 (开发期)")
        axes[1].legend()
        axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "auto_train2.png", dpi=130)
    plt.close(fig)
    print(f"\n总用时 {time.time()-t0:.0f}s; 结果已写 {OUT/'auto_train2.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
