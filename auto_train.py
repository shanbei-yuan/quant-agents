#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自主训练 auto_train.py
======================
在给定时间预算内, 自主搜索"最能提高组合收益可能性"的配置。

【防自欺欺人的核心协议】
  * 全部搜索只在"开发期"上进行(每个配置都用 walk-forward 得到真样本外预测)
  * 最终留出集(最后 20% 交易日)在整个搜索过程中一次都不用
  * 搜索结束后, 只对开发期最优配置在留出集上评估**一次**
  * 同时输出所有配置的分布 —— 用来量化多重比较带来的选择偏差

【三条建议的落地】
  1) 把"全池等权"作为正式基准(选股模型必须先打败它, 而不是沪深300)
  2) 标签改为风格中性: 全体中位数 / 同行业中位数 / 同(行业×波动率分组)中位数
  3) 波动率仓位作为标准零件(目标波动率可调, 含杠杆与融资成本)

搜索维度: 标签 × 周期 × 模型 × 特征集 × 持仓数 K × 仓位模式
结果: out/search_leaderboard.jsonl (逐条追加, 可续跑) + out/auto_train_report.md
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

import quant_data as qd
import train_agents as ta
import vol_agents as va
from vol_target import forecasts as vol_forecasts
from train_agents import Blackboard, setup_font

ROOT = Path(__file__).resolve().parent
CACHE, OUT = ROOT / "cache", ROOT / "out"
BENCH = "sh.000300"
HOLDOUT = 0.20
COST_STOCK = 0.006
MIN_TRAIN = 500
TEST_LEN = 400
CAP = 120000          # 每折训练样本上限(加速)
TARGET_VOL = 0.12
HS = (5, 10, 20)


# ------------------------------------------------------------------ 数据
def industry_map() -> dict:
    f = CACHE / "industry.csv"
    if f.exists():
        d = pd.read_csv(f, dtype=str)
        return dict(zip(d["code"], d["industry"]))
    import baostock as bs
    bs.login()
    rs = bs.query_stock_industry()
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    fields = rs.fields
    bs.logout()
    d = pd.DataFrame(rows, columns=fields)
    d["code6"] = d["code"].astype(str).str.split(".").str[-1]
    d["industry"] = d["industry"].replace("", "未知").fillna("未知")
    d[["code6", "industry"]].rename(columns={"code6": "code"}).to_csv(
        f, index=False, encoding="utf-8")
    return dict(zip(d["code6"], d["industry"]))


def load_stocks(max_n=None):
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


def build_extended_panel(max_stocks=None, force=False) -> pd.DataFrame:
    ck = CACHE / "ext_panel_v2.pkl"
    if ck.exists() and not force:
        return pd.read_pickle(ck)
    data = load_stocks(max_stocks)
    print(f"[面板] 载入个股 {len(data)} 只", flush=True)
    cfg = ta.Config(universe_n=len(data), fast=True, cached_only=True, horizons=HS)
    bb = Blackboard()
    bench = qd.index_hist_bs(BENCH)
    panel = ta.FeatureAgent(bb, cfg).run(data, bench)
    df = ta.LabelAgent(bb, cfg).run(panel)
    print(f"[面板] {len(df):,} 行, 开始加风格分组标签", flush=True)
    ind = industry_map()
    df["industry"] = df["code"].astype(str).map(ind).fillna("未知")
    vq = df.groupby("date")["vol20"].rank(pct=True)
    df["volq"] = np.minimum((vq.fillna(0.5) * 5).astype(int), 4)
    df["grp"] = df["industry"].astype(str) + "#" + df["volq"].astype(str)
    for h in HS:
        fw = df[f"fwd{h}"]
        med_all = df.groupby("date")[f"fwd{h}"].transform("median")
        med_ind = df.groupby(["date", "industry"])[f"fwd{h}"].transform("median")
        med_grp = df.groupby(["date", "grp"])[f"fwd{h}"].transform("median")
        n_ind = df.groupby(["date", "industry"])[f"fwd{h}"].transform("size")
        n_grp = df.groupby(["date", "grp"])[f"fwd{h}"].transform("size")
        df[f"yrel{h}"] = (fw > med_all).astype(float)
        df[f"yind{h}"] = (fw > med_ind).astype(float)
        df[f"ygrp{h}"] = (fw > med_grp).astype(float)
        # 组内样本太少的, 标签不可靠 -> 置空
        df.loc[(n_ind < 5) | fw.isna(), f"yind{h}"] = np.nan
        df.loc[(n_grp < 5) | fw.isna(), f"ygrp{h}"] = np.nan
        df.loc[fw.isna(), f"yrel{h}"] = np.nan
    df.to_pickle(ck)
    print(f"[面板] 已缓存 {ck}", flush=True)
    return df


# ------------------------------------------------------------------ 走查预测
def wf_predict(df, feats, ycol, h, kind, dev_end, seed=0):
    """研发期 walk-forward。返回 pooled DataFrame(date, code, p, y, fwd)。"""
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
        te = np.where((D >= d0) & (D <= d1))[0]
        if len(tr) < 3000 or len(te) < 300:
            continue
        if len(tr) > CAP:
            tr = rng.choice(tr, CAP, replace=False)
        m = _make(kind, seed)
        m.fit(A[tr], Y[tr])
        p = m.predict_proba(A[te])[:, 1]
        out.append(pd.DataFrame({"date": D[te], "code": C[te], "p": p,
                                 "y": Y[te], "fwd": F[te]}))
    if not out:
        return None, None
    pooled = pd.concat(out, ignore_index=True)
    # 留出期: 用开发期全部数据训练一次
    tr = np.where(D <= np.datetime64(dev_end))[0]
    te = np.where(D > np.datetime64(dev_end))[0]
    if len(tr) > CAP:
        tr = rng.choice(tr, CAP, replace=False)
    mh = _make(kind, seed)
    mh.fit(A[tr], Y[tr])
    hold = pd.DataFrame({"date": D[te], "code": C[te],
                         "p": mh.predict_proba(A[te])[:, 1],
                         "y": Y[te], "fwd": F[te]})
    return pooled, hold


def _make(kind, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=60, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=True, n_iter_no_change=10,
            validation_fraction=0.15, random_state=seed)
    if kind == "hgbdeep":
        return HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=60,
            l2_regularization=2.0, early_stopping=True, n_iter_no_change=12,
            validation_fraction=0.15, random_state=seed)
    if kind == "logit":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=300, C=0.1, n_jobs=-1))
    if kind == "rf":
        return RandomForestClassifier(n_estimators=40, max_depth=6, min_samples_leaf=150,
                                      n_jobs=-1, random_state=seed)
    raise ValueError(kind)


# ------------------------------------------------------------------ 组合
def portfolio(pooled, k, all_dates, h, volmap=None, use_vol=False, cost=COST_STOCK):
    """非重叠调仓组合。返回 (指标, 曲线, 全池等权曲线)。"""
    if pooled is None or pooled.empty:
        return None, None, None
    ix = {pd.Timestamp(d): i for i, d in enumerate(all_dates)}
    pooled = pooled.sort_values("date")
    curve, ew, prev, eq, eqw = [], [], set(), 1.0, 1.0
    rets, expos = [], []
    for d, g in pooled.groupby("date"):
        i = ix.get(pd.Timestamp(d))
        if i is None or (i % h) != 0 or len(g) < max(20, 3 * k):
            continue
        g = g.sort_values("p", ascending=False)
        sel = g.head(k)
        expo = 1.0
        if use_vol and volmap is not None:
            v = volmap.get(pd.Timestamp(d))
            if v is None or not np.isfinite(v) or v <= 0:
                continue
            expo = float(np.clip(TARGET_VOL / v, 0.0, 1.0))
        newset = set(sel["code"])
        turn = len(newset - prev) / max(1, k)
        r = expo * float(sel["fwd"].mean()) - cost * turn * expo
        rw = float(g["fwd"].mean())
        eq *= (1 + r)
        eqw *= (1 + rw)
        curve.append(eq)
        ew.append(eqw)
        rets.append(r)
        expos.append(expo)
        prev = newset
    if len(curve) < 20:
        return None, None, None
    curve, ew = np.array(curve), np.array(ew)
    rets = np.array(rets)
    yrs = len(curve) * h / 250.0
    ann = float(curve[-1] ** (1 / yrs) - 1)
    mdd = float((curve / np.maximum.accumulate(curve) - 1).min())
    sd = float(np.std(rets, ddof=1))
    ew_ann = float(ew[-1] ** (1 / yrs) - 1)
    ew_mdd = float((ew / np.maximum.accumulate(ew) - 1).min())
    met = {"ann": round(ann, 4), "ew_ann": round(ew_ann, 4),
           "excess_ann": round(ann - ew_ann, 4), "mdd": round(mdd, 4),
           "ew_mdd": round(ew_mdd, 4),
           "sharpe": round(float(np.mean(rets) / sd * np.sqrt(250 / h)), 3) if sd else None,
           "calmar": round(ann / abs(mdd), 3) if mdd else None,
           "win": round(float((rets > 0).mean()), 4),
           "expo": round(float(np.mean(expos)), 3),
           "n": len(curve), "years": round(yrs, 2)}
    return met, curve, ew


def acc_stats(pooled):
    from sklearn.metrics import roc_auc_score
    if pooled is None or pooled.empty:
        return {}
    y, p = pooled["y"].values, pooled["p"].values
    acc = float(((p > 0.5).astype(int) == y).mean())
    base = float(max(y.mean(), 1 - y.mean()))
    auc = float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")
    top = pooled.sort_values("p", ascending=False).groupby("date").head(10)
    return {"acc": round(acc, 4), "base": round(base, 4),
            "edge": round(acc - base, 4),
            "auc": None if np.isnan(auc) else round(auc, 4),
            "top10_fwd": round(float(top["fwd"].mean()) * 100, 4),
            "all_fwd": round(float(pooled["fwd"].mean()) * 100, 4)}


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=150)
    ap.add_argument("--max-stocks", type=int, default=None)
    args = ap.parse_args()
    t_start = time.time()
    budget = args.minutes * 60

    print("=" * 100)
    print(f"自主训练 v1.0   时间预算 {args.minutes:.0f} 分钟")
    print("=" * 100, flush=True)
    df = build_extended_panel(args.max_stocks)
    all_dates = np.sort(df["date"].unique())
    dev_end = all_dates[int(len(all_dates) * (1 - HOLDOUT))]
    print(f"[协议] 开发期 ~ {dev_end.astype('datetime64[D]')}; "
          f"留出期 {len(all_dates) - int(len(all_dates) * (1 - HOLDOUT))} 天 (搜索期间全程不用)",
          flush=True)

    bench = qd.index_hist_bs(BENCH)
    dev_v, hold_v, _ = vol_forecasts(bench, list(va.FEATS), 10, va.Cfg())
    volmap = {pd.Timestamp(k): v for k, v in zip(dev_v["date"], dev_v["p"])}
    volmap.update({pd.Timestamp(k): v for k, v in zip(hold_v["date"], hold_v["p"])})
    print(f"[零件] 波动率预测 {len(volmap)} 条就绪", flush=True)

    feats_all = [f for f in ta.ALL_FEATS if f in df.columns]
    feats_mom = [f for f in feats_all if f in (
        "ret1", "ret5", "ret10", "ret20", "ret60", "ma5_gap", "ma10_gap",
        "ma20_gap", "ma60_gap", "ma5_ma20", "ma20_ma60", "slope20_5",
        "slope60_20", "rsi14", "macd_hist", "vr_5_20", "vr_5_60")]
    feats_rank = [f for f in feats_all if f.startswith("rank_") or f in (
        "rs20", "pos60", "pos120", "dd20", "above_ma20", "breadth", "vol20", "vol60")]
    featsets = [("all", feats_all),
                ("nomkt", [f for f in feats_all if f not in ta.FEATS_MARKET]),
                ("mom", feats_mom), ("rank", feats_rank)]
    # 标签: 全体中位数 / 同行业中位数 / 同(行业×波动率)分组中位数 / 跑赢沪深300
    label_sets = [("rel", "yrel"), ("ind", "yind"), ("grp", "ygrp"),
                  ("alpha", "yalpha")]
    kinds = ["hgb", "logit", "hgbdeep", "rf"]

    lb_path = OUT / "search_leaderboard.jsonl"
    done = set()
    if lb_path.exists():
        for line in lb_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    done.add(r["key"])
                except Exception:                  # noqa: BLE001
                    pass
    print(f"[续跑] 已有 {len(done)} 条记录", flush=True)

    cfg_list = []
    for lname, _lp in label_sets:
        for h in HS:
            for kind in kinds:
                for fname, _fl in featsets:
                    cfg_list.append({"label": lname, "h": h, "kind": kind,
                                     "feats": fname})
    print(f"[搜索空间] {len(cfg_list)} 个(标签×周期×模型×特征集) 配置, "
          f"每个再评估 K∈{{5,10,20}} × 仓位∈{{满仓,波动率}}", flush=True)

    n_done_this_run = 0
    for cfg in cfg_list:
        key = f"{cfg['label']}|h{cfg['h']}|{cfg['kind']}|{cfg['feats']}"
        if key in done:
            continue
        if time.time() - t_start > budget:
            print(f"[预算] 时间到, 已跑 {n_done_this_run} 个配置, 停止搜索", flush=True)
            break
        t0 = time.time()
        ycol = f"y{cfg['label']}{cfg['h']}"
        feats = dict(featsets)[cfg["feats"]]
        pooled, hold = wf_predict(df, feats, ycol, cfg["h"], cfg["kind"], dev_end)
        if pooled is None:
            print(f"  [skip] {key}: 无有效样本", flush=True)
            continue
        st = acc_stats(pooled)
        rec = {"key": key, **cfg, "n_feats": len(feats), **st,
               "secs": round(time.time() - t0, 1)}
        best = None
        for k in (5, 10, 20):
            for uv in (False, True):
                met, cur, ew = portfolio(pooled, k, all_dates, cfg["h"],
                                         volmap, uv)
                if met is None:
                    continue
                met.update({"k": k, "use_vol": uv})
                rec[f"k{k}_{'vol' if uv else 'full'}"] = met
                if best is None or met["excess_ann"] > best["excess_ann"]:
                    best = met
        if best:
            rec["best"] = best
        with lb_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        done.add(key)
        n_done_this_run += 1
        bs_ = rec.get("best", {})
        print(f"  [{n_done_this_run}] {key:<28} 命中 {rec['acc']:.4f}({rec['edge']:+.4f}) "
              f"AUC {rec['auc']} | 最优组合 K={bs_.get('k')} "
              f"{'波动率' if bs_.get('use_vol') else '满仓'} 年化 {bs_.get('ann')} "
              f"超额(对全池等权) {bs_.get('excess_ann')} 夏普 {bs_.get('sharpe')} "
              f"| {rec['secs']}s", flush=True)

    print("=" * 100)
    print(f"搜索结束, 本轮完成 {n_done_this_run} 个配置, 累计 {len(done)} 条",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
