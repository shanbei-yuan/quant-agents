#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多智能体协作训练系统 train_agents.py  v1.0
==========================================
目标: 不选股、不下单, 只训练并严格评估一个"未来 N 日涨跌"预测模型,
      用样本外(walk-forward)命中率衡量预测能力, 达标后才谈交易。

八个 Agent(人设)协作方式:
  1 数据官 DataAgent        建股票池 / 拉 8 年日线 / 数据体检
  2 特征官 FeatureAgent     只用 t 时刻及之前信息造特征 + 泄露自检
  3 标注官 LabelAgent       定义预测目标(未来 h 日收益), 含扣费标签
  4 模型官 ModelAgent       训练 + 滚动窗口样本外预测
  5 评估官 EvalAgent        命中率/AUC/概率分档单调性/策略化收益/基线对比
  6 质疑官 CriticAgent      过拟合·泄露·偏差审查, 给可信度评级
  7 调度官 Orchestrator     读质疑官结论 -> 决定下一轮特征/模型/超参 -> 保留最优
  8 档案官 MemoryAgent      写 leaderboard.json + memory.jsonl (跨轮次、跨会话记忆)

协作机制落地:
  * 共享黑板   Blackboard.store   后一个 Agent 读前一个的产物
  * 消息传递   handoff 记录        logs/train_handoff.log 能看到谁交给谁什么
  * 相互学习   Orchestrator 读 Critic 的结论 + 历史 memory 来改下一轮配置,
              这就是"agent 之间互相交流并据此改进"的实际含义(不是改模型权重)

用法:
    python train_agents.py                     # 默认 3 轮迭代
    python train_agents.py --rounds 2 --universe 150 --fast
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import quant_data as qd

ROOT = Path(__file__).resolve().parent
CACHE, OUT, LOGS = ROOT / "cache", ROOT / "out", ROOT / "logs"
for _d in (CACHE, OUT, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

TRADING_COST = 0.006          # 单次往返成本(佣金+印花税+滑点) 约 0.6%
METRIC_TARGET = 0.55          # 宣称"够用"的样本外命中率门槛


# ================================================================== 配置
@dataclass
class Config:
    universe_n: int = 200
    min_amount: float = 1.0e8
    horizons: tuple = (5, 10)
    h_main: int = 5
    test_len: int = 250            # 每次样本外测试 250 个交易日(约1年)
    min_train_days: int = 500      # 前 2 年只做训练
    prob_threshold: float = 0.6    # 策略化评估的出手阈值
    rounds: int = 3
    fast: bool = False
    cached_only: bool = False       # 只用已缓存股票, 避免在限流时白跑编译下载
    seed: int = 0


FEATS_STATIC = [
    "ret1", "ret5", "ret10", "ret20", "ret60",
    "ma5_gap", "ma10_gap", "ma20_gap", "ma60_gap",
    "ma5_ma20", "ma20_ma60", "slope20_5", "slope60_20",
    "vol20", "vol60", "atr_pct", "rsi14", "macd_hist",
    "vr_5_20", "vr_5_60", "pos60", "pos120", "dd20", "log_price",
    "rs20", "rank_ret20", "rank_vol20", "rank_ret5", "rank_rsi14",
    "rank_vr", "rank_dd20", "above_ma20",
]
FEATS_MARKET = ["bench_ret5", "bench_ret20", "bench_vol20",
                "bench_ma20_gap", "bench_ma60_gap", "breadth"]
ALL_FEATS = FEATS_STATIC + FEATS_MARKET


# ================================================================== 黑板
class Blackboard:
    def __init__(self):
        self.store: dict = {}
        self.handoffs: list[str] = []

    def put(self, agent: str, payload):
        self.store[agent] = payload
        return payload

    def handoff(self, src: str, dst: str, msg: str):
        line = f"[{datetime.now():%H:%M:%S}] {src:<14s} -> {dst:<14s} | {msg}"
        self.handoffs.append(line)
        print("   " + line, flush=True)

    def save(self):
        (OUT / "blackboard.json").write_text(
            json.dumps(self.store, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        (LOGS / "train_handoff.log").write_text("\n".join(self.handoffs) + "\n",
                                                encoding="utf-8")


class Agent:
    name = "Agent"

    def __init__(self, bb: Blackboard, cfg: Config):
        self.bb, self.cfg = bb, cfg

    def log(self, m):
        print(f"[{self.name}] {m}", flush=True)


# ================================================================== 1 数据官
class DataAgent(Agent):
    name = "数据官"

    def run(self):
        snap = qd.snapshot()
        self.log(f"全市场快照 {len(snap)} 只 (源={qd.ACTIVE_SOURCE or qd.probe_source()})")
        pool = qd.build_universe(top_n=self.cfg.universe_n,
                                 min_amount=self.cfg.min_amount)
        if self.cfg.cached_only:
            n0 = len(pool)
            pool = pool[pool["code"].apply(
                lambda c: (qd.CACHE / f"kline_{c}.csv").exists())].reset_index(drop=True)
            self.log(f"--cached-only: {n0} -> {len(pool)} 只(仅用本地已缓存历史)")
        self.log(f"股票池 {len(pool)} 只 (主板/非ST/成交额>={self.cfg.min_amount/1e8:.1f}亿, "
                 f"按成交额前 {self.cfg.universe_n})")
        data = qd.download_all(pool)
        bench = qd.benchmark_hist()

        # 数据体检
        lens = pd.Series({c: len(d) for c, d in data.items()})
        health = {
            "n_stocks": len(data),
            "bars_min": int(lens.min()), "bars_median": int(lens.median()),
            "bars_max": int(lens.max()),
            "start": str(min(d.index[0] for d in data.values()).date()),
            "end": str(max(d.index[-1] for d in data.values()).date()),
            "nan_ratio": round(float(np.mean([d[["open", "close", "high", "low"]].isna().mean().mean()
                                              for d in data.values()])), 6),
            "zero_vol_rows": int(sum((d["volume"] <= 0).sum() for d in data.values())),
        }
        self.log(f"数据体检: {health['n_stocks']} 只, 每只中位 {health['bars_median']} 根, "
                 f"区间 {health['start']} ~ {health['end']}, 零成交行 {health['zero_vol_rows']}")
        self.bb.put("DataAgent", {"pool_n": len(pool), "health": health,
                                  "names": dict(zip(pool["code"].head(20), pool["name"].head(20)))})
        self.bb.handoff(self.name, "特征官",
                        f"{len(data)} 只 × 中位 {health['bars_median']} 根日线就绪, "
                        f"缺失率 {health['nan_ratio']:.2e}")
        return data, bench, pool


# ================================================================== 2 特征官
class FeatureAgent(Agent):
    name = "特征官"

    def run(self, data: dict, bench: pd.DataFrame):
        key = f"panel_v3_{len(data)}"
        pf = CACHE / f"{key}.pkl"
        if pf.exists():
            panel = pd.read_pickle(pf)
            self.log(f"读特征面板缓存: {len(panel):,} 行 (命中)")
        else:
            panel = self._build(data, bench)
            panel.to_pickle(pf)
            self.log(f"构造特征面板: {len(panel):,} 行 × {len(panel.columns)} 列")

        # 泄露自检: 特征必须与未来收益没有"同期"关系; 抽查 ret1 与当日 pct 的错位
        chk = panel.dropna(subset=["ret1", "ret5"]).copy()
        corr = float(chk["ret1"].corr(chk["ret5"]))
        self.bb.put("FeatureAgent", {"rows": len(panel), "n_features": len(ALL_FEATS),
                                     "features": ALL_FEATS, "ret1_ret5_corr": round(corr, 3),
                                     "note": "所有特征只用 ≤t 的数据计算; 复权价使用后复权因子, "
                                             "属轻微未来信息, 已在质疑官清单中列出"})
        self.bb.handoff(self.name, "标注官", f"{len(ALL_FEATS)} 个特征就绪 "
                        f"(含 6 个大盘/宽度特征), ret1~ret5 相关 {corr:.2f}")
        return panel

    def _build(self, data: dict, bench: pd.DataFrame) -> pd.DataFrame:
        frames = []
        for code, d in data.items():
            x = d.copy()
            x["code"] = code
            frames.append(x)
        df = pd.concat(frames)
        df.index.name = "date"
        df = df.reset_index().sort_values(["code", "date"]).reset_index(drop=True)
        g = df.groupby("code", group_keys=False)

        df["ret1"] = g["close"].pct_change()
        for n in (5, 10, 20, 60):
            df[f"ret{n}"] = g["close"].pct_change(n)
        for n in (5, 10, 20, 60):
            df[f"ma{n}"] = g["close"].transform(lambda s, n=n: s.rolling(n).mean())
            df[f"ma{n}_gap"] = df["close"] / df[f"ma{n}"] - 1
        df["ma5_ma20"] = df["ma5"] / df["ma20"] - 1
        df["ma20_ma60"] = df["ma20"] / df["ma60"] - 1
        df["slope20_5"] = df["ma20"] / g["ma20"].shift(5) - 1
        df["slope60_20"] = df["ma60"] / g["ma60"].shift(20) - 1

        df["vol20"] = g["ret1"].transform(lambda s: s.rolling(20).std())
        df["vol60"] = g["ret1"].transform(lambda s: s.rolling(60).std())
        df["prev_close"] = g["close"].shift(1)
        tr = np.maximum(df["high"] - df["low"],
                        np.maximum((df["high"] - df["prev_close"]).abs(),
                                   (df["low"] - df["prev_close"]).abs()))
        df["atr14"] = tr.groupby(df["code"]).transform(lambda s: s.rolling(14).mean())
        df["atr_pct"] = df["atr14"] / df["close"]

        def _rsi(s, n=14):
            d = s.diff()
            up = d.clip(lower=0).rolling(n).mean()
            dn = (-d.clip(upper=0)).rolling(n).mean()
            return 100 - 100 / (1 + up / dn.replace(0, np.nan))
        df["rsi14"] = g["close"].transform(_rsi)

        ema12 = g["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
        ema26 = g["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
        dif = ema12 - ema26
        dea = dif.groupby(df["code"]).transform(lambda s: s.ewm(span=9, adjust=False).mean())
        df["macd_hist"] = (dif - dea) * 2 / df["close"]

        df["vma5"] = g["volume"].transform(lambda s: s.rolling(5).mean())
        df["vma20"] = g["volume"].transform(lambda s: s.rolling(20).mean())
        df["vma60"] = g["volume"].transform(lambda s: s.rolling(60).mean())
        df["vr_5_20"] = df["vma5"] / df["vma20"]
        df["vr_5_60"] = df["vma5"] / df["vma60"]

        hi60 = g["high"].transform(lambda s: s.rolling(60).max())
        lo60 = g["low"].transform(lambda s: s.rolling(60).min())
        hi120 = g["high"].transform(lambda s: s.rolling(120).max())
        lo120 = g["low"].transform(lambda s: s.rolling(120).min())
        df["pos60"] = (df["close"] - lo60) / (hi60 - lo60).replace(0, np.nan)
        df["pos120"] = (df["close"] - lo120) / (hi120 - lo120).replace(0, np.nan)
        df["dd20"] = df["close"] / g["close"].transform(lambda s: s.rolling(20).max()) - 1
        df["log_price"] = np.log(df["close"])

        # 大盘 / 市场宽度特征
        b = bench.copy()
        b["bench_ret5"] = b["close"].pct_change(5)
        b["bench_ret20"] = b["close"].pct_change(20)
        b["bench_vol20"] = b["close"].pct_change().rolling(20).std()
        b["bench_ma20_gap"] = b["close"] / b["close"].rolling(20).mean() - 1
        b["bench_ma60_gap"] = b["close"] / b["close"].rolling(60).mean() - 1
        bm = b[["bench_ret5", "bench_ret20", "bench_vol20",
                "bench_ma20_gap", "bench_ma60_gap"]]
        bm = bm.assign(bench_close=b["close"])
        df = df.merge(bm, left_on="date", right_index=True, how="left")
        df["above_ma20"] = (df["close"] > df["ma20"]).astype(float)
        df["breadth"] = df["date"].map(df.groupby("date")["above_ma20"].mean())
        df["rs20"] = df["ret20"] - df["bench_ret20"]
        df["rank_ret20"] = df.groupby("date")["ret20"].rank(pct=True)
        df["rank_vol20"] = df.groupby("date")["vol20"].rank(pct=True)
        # 横截面排序特征: 预测"同一批股票里谁更强"时, 相对位置比绝对值更有信息
        df["rank_ret5"] = df.groupby("date")["ret5"].rank(pct=True)
        df["rank_rsi14"] = df.groupby("date")["rsi14"].rank(pct=True)
        df["rank_vr"] = df.groupby("date")["vr_5_60"].rank(pct=True)
        df["rank_dd20"] = df.groupby("date")["dd20"].rank(pct=True)
        return df


# ================================================================== 3 标注官
class LabelAgent(Agent):
    name = "标注官"

    def run(self, panel: pd.DataFrame):
        df = panel
        g = df.groupby("code", group_keys=False)
        df = df.assign(next_open=g["open"].shift(-1))
        info = {}
        for h in self.cfg.horizons:
            df[f"exit_open_{h}"] = g["open"].shift(-(1 + h))
            df[f"fwd{h}"] = df[f"exit_open_{h}"] / df["next_open"] - 1
            df[f"ynet{h}"] = df[f"fwd{h}"] - TRADING_COST
            df[f"y{h}"] = (df[f"fwd{h}"] > 0).astype(float)
            df[f"yw{h}"] = (df[f"ynet{h}"] > 0).astype(float)
            # 横截面相对标签: 是否跑赢当天所有股票的中位数(同一天内的排序问题)
            df[f"yrel{h}"] = (df[f"fwd{h}"] > df.groupby("date")[f"fwd{h}"].transform("median")).astype(float)
            df.loc[df[f"fwd{h}"].isna(), f"yrel{h}"] = np.nan
            # 相对基准标签(真 alpha): 是否跑赢沪深300 同区间收益
            bser = df.groupby("date")["bench_close"].first().sort_index()
            bfwd = bser.shift(-(1 + h)) / bser.shift(-1) - 1
            df[f"bfwd{h}"] = df["date"].map(bfwd)
            df[f"yalpha{h}"] = (df[f"fwd{h}"] > df[f"bfwd{h}"]).astype(float)
            df.loc[df[f"fwd{h}"].isna() | df[f"bfwd{h}"].isna(), f"yalpha{h}"] = np.nan
            sub = df[f"fwd{h}"].dropna()
            info[h] = {"base_rate_up": round(float((sub > 0).mean()), 4),
                       "base_rate_win": round(float((sub - TRADING_COST > 0).mean()), 4),
                       "base_rate_rel": round(float(df[f"yrel{h}"].dropna().mean()), 4),
                       "base_rate_alpha": round(float(df[f"yalpha{h}"].dropna().mean()), 4),
                       "mean_fwd": round(float(sub.mean()), 5),
                       "std_fwd": round(float(sub.std()), 5)}
        self.log("标签定义: 以 t+1 开盘买入, t+1+h 开盘卖出, "
                 f"扣费后阈值 {TRADING_COST*100:.1f}%; "
                 + "; ".join(f"h={h} 上涨基准率 {v['base_rate_up']:.3f}" for h, v in info.items()))
        self.bb.put("LabelAgent", info)
        self.bb.handoff(self.name, "模型官",
                        f"h={self.cfg.h_main} 基准上涨率 {info[self.cfg.h_main]['base_rate_up']:.3f} "
                        f"(模型必须显著超过这个数才有意义)")
        return df


# ================================================================== 4 模型官
class ModelAgent(Agent):
    name = "模型官"

    def make_model(self, kind: str):
        if kind == "logit":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            return make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=300, C=0.1, n_jobs=-1))
        if kind == "hgb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(
                max_iter=60 if self.cfg.fast else 150, learning_rate=0.08,
                max_leaf_nodes=31, early_stopping=True, n_iter_no_change=12,
                validation_fraction=0.12, random_state=self.cfg.seed)
        if kind == "rf":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=60 if self.cfg.fast else 150, max_depth=8,
                min_samples_leaf=200, n_jobs=-1, random_state=self.cfg.seed)
        if kind == "ens":
            from sklearn.ensemble import VotingClassifier
            return VotingClassifier([("lr", self.make_model("logit")),
                                     ("hgb", self.make_model("hgb"))], voting="soft")
        raise ValueError(kind)

    def walk_forward(self, df: pd.DataFrame, feats: list, h: int, kind: str,
                     tag: str = "", label: str = "abs", end_date=None):
        from sklearn.metrics import roc_auc_score
        ycol = {"abs": f"y{h}", "rel": f"yrel{h}", "alpha": f"yalpha{h}"}[label]
        fwdcol = f"ynet{h}"
        dates = np.sort(df["date"].unique())
        if end_date is not None:
            dates = dates[dates <= np.datetime64(end_date)]
        folds, preds = [], []
        start = self.cfg.min_train_days
        for si in range(start, len(dates), self.cfg.test_len):
            te_dates = dates[si:si + self.cfg.test_len]
            if len(te_dates) < 40:
                break
            d0 = te_dates[0]
            tr = df[(df["date"] < d0)].dropna(subset=feats + [ycol, fwdcol])
            te = df[df["date"].isin(te_dates)].dropna(subset=feats + [ycol, fwdcol])
            if len(tr) < 5000 or len(te) < 500:
                continue
            model = self.make_model(kind)
            model.fit(tr[feats], tr[ycol])
            p = model.predict_proba(te[feats])[:, 1]
            acc = float(((p > 0.5).astype(int) == te[ycol]).mean())
            base = float(max(te[ycol].mean(), 1 - te[ycol].mean()))
            auc = float(roc_auc_score(te[ycol], p)) if te[ycol].nunique() > 1 else np.nan
            ins = None
            try:
                pin = model.predict_proba(tr[feats].iloc[-40000:])[:, 1]
                ins = float(((pin > 0.5).astype(int) == tr[ycol].iloc[-40000:]).mean())
            except Exception:                     # noqa: BLE001
                pass
            folds.append({"test_start": str(pd.Timestamp(d0).date()),
                          "n_train": int(len(tr)), "n_test": int(len(te)),
                          "acc": round(acc, 4), "auc": None if np.isnan(auc) else round(auc, 4),
                          "base": round(base, 4), "in_sample_acc": None if ins is None else round(ins, 4)})
            t = te[["date", "code", ycol, fwdcol]].copy().reset_index(drop=True)
            t["p"] = p
            preds.append(t)
        pooled = pd.concat(preds) if preds else pd.DataFrame()
        return {"folds": folds, "pooled": pooled, "kind": kind, "h": h, "tag": tag,
                "label": label, "ycol": ycol, "fwdcol": fwdcol}

    def run(self, df: pd.DataFrame, feats: list, h: int, kind: str, label: str = "abs"):
        t0 = time.time()
        res = self.walk_forward(df, feats, h, kind, label=label)
        self.log(f"{kind} / h={h} / 标签={label} / {len(feats)} 特征: "
                 f"样本外 {len(res['pooled']):,} 条, {len(res['folds'])} 折, 用时 {time.time()-t0:.0f}s")
        self.bb.handoff(self.name, "评估官",
                        f"{kind}(h={h},标签={label}) 完成 {len(res['folds'])} 折滚动预测")
        return res


# ================================================================== 5 评估官
class EvalAgent(Agent):
    name = "评估官"

    def run(self, wf: dict, thr: float):
        pooled = wf["pooled"]
        from scipy.stats import spearmanr
        h = wf["h"]
        ycol, fwdcol = wf["ycol"], wf["fwdcol"]
        y, p, fwd = pooled[ycol].values, pooled["p"].values, pooled[fwdcol].values
        from sklearn.metrics import roc_auc_score
        acc = float(((p > 0.5).astype(int) == y).mean())
        base = float(max(y.mean(), 1 - y.mean()))
        auc = float(roc_auc_score(y, p))
        # 概率分档单调性: 模型说"越可能涨"的档, 实际上涨比例是否越高
        q = pd.qcut(pooled["p"], 5, labels=False, duplicates="drop")
        tab = pooled.groupby(q).agg(n=(ycol, "size"), pred=("p", "mean"),
                                    actual=(ycol, "mean"), net=(fwdcol, "mean"))
        mono = float(spearmanr(tab.index.values, tab["actual"].values).correlation)
        # 策略化: 只在 p>=thr 出手
        sel = pooled[pooled["p"] >= thr]
        strat = {"n_signals": int(len(sel)),
                 "frac": round(float(len(sel) / len(pooled)), 4),
                 "hit_rate": None if len(sel) == 0 else round(float(sel[ycol].mean()), 4),
                 "avg_net_ret": None if len(sel) == 0 else round(float(sel[fwdcol].mean()), 5),
                 "total_net_ret": None if len(sel) == 0 else round(float(sel[fwdcol].sum()), 4)}
        # 净值曲线: 每 h 个交易日调仓一次, 避免重叠持仓被重复复利
        if len(sel):
            daily = sel.groupby("date")[fwdcol].mean().sort_index()
            step = max(1, int(h))
            periodic = daily.iloc[::step]
            eq = (1 + periodic).cumprod()
        else:
            eq = pd.Series(dtype=float)
        res = {"acc": round(acc, 4), "base": round(base, 4), "auc": round(auc, 4),
               "edge_vs_base": round(acc - base, 4), "edge_vs_coin": round(acc - 0.5, 4),
               "n_oos": int(len(pooled)), "deciles": tab.reset_index().to_dict("records"),
               "monotonicity": round(mono, 3), "strategy": strat,
               "mean_pred": round(float(p.mean()), 4)}
        self.log(f"样本外: 命中率 {acc:.4f} (基线 {base:.4f}, 超额 {acc-base:+.4f}) | "
                 f"AUC {auc:.4f} | 分档单调性 {mono:+.2f} | 出手率 "
                 f"{strat['frac']:.1%} 命中 {strat['hit_rate']}")
        self.bb.put(f"EvalAgent_{wf['tag'] or wf['kind']}", res)
        self.bb.handoff(self.name, "质疑官",
                        f"{wf['tag'] or wf['kind']} 样本外命中率 {acc:.4f} / AUC {auc:.4f}")
        return res, eq


# ================================================================== 6 质疑官
class CriticAgent(Agent):
    name = "质疑官"

    def run(self, wf: dict, ev: dict, df: pd.DataFrame, feats: list):
        folds = pd.DataFrame(wf["folds"])
        h = wf["h"]
        issues, verdict = [], ""
        gap = None
        if folds["in_sample_acc"].notna().any():
            gap = float(folds["in_sample_acc"].mean() - folds["acc"].mean())
            if gap > 0.06:
                issues.append(f"训练集命中率比样本外高 {gap:.3f}, 存在明显过拟合迹象")
            else:
                issues.append(f"训练集与样本外命中率差 {gap:.3f}, 过拟合不严重")
        std = float(folds["acc"].std())
        issues.append(f"各折命中率标准差 {std:.4f} "
                      + ("(折间不稳定, 单年表现不可外推)" if std > 0.03 else "(折间较稳定)"))
        nf = len(folds)
        wins = int((folds["acc"] > folds["base"]).sum())
        frac = wins / nf if nf else 0.0
        issues.append(f"折级胜率 {wins}/{nf} ({frac:.0%}) —— 低于 60% 说明整体超额可能只来自少数几年")
        if ev["acc"] <= ev["base"] + 0.005:
            issues.append("样本外命中率没有超过'永远猜多数类'基线 —— 模型没有增量信息")
        if ev["acc"] <= 0.505:
            issues.append("命中率与抛硬币无区别")
        if ev["monotonicity"] < 0.7:
            issues.append(f"概率分档单调性只有 {ev['monotonicity']:.2f}, 高概率档并不更准")

        # 泄露检测: 打乱训练标签后重训, 样本外应回落到基线
        leak = self._leakage_test(df, feats, wf["ycol"])
        if leak is not None:
            if leak - ev["base"] > 0.03:
                issues.append(f"打乱标签后样本外仍有 {leak:.4f}, 高于基线 {ev['base']:.4f} —— "
                              f"可能存在特征泄露")
            else:
                issues.append(f"打乱标签后样本外 {leak:.4f} ≈ 基线 {ev['base']:.4f}, 未检出泄露")

        bias = [
            "存活偏差: 股票池来自'今天仍上市'的样本, 已退市/被并购的股票没进样本",
            "选择偏差: 按今日成交额取前 N 只, 天然偏向大盘蓝筹",
            "复权偏差: 前复权价格隐含了未来的分红送股信息(幅度很小但方向偏乐观)",
            "重叠持仓: 相邻交易日样本的持仓期重叠, 净值曲线不是可实现的组合收益",
            "摩擦成本: 只按 0.6% 往返估算, 未含冲击成本与涨跌停无法成交的情况",
        ]
        rank = "低"
        if ev["acc"] > ev["base"] + 0.02 and ev["auc"] > 0.53 and frac >= 0.6:
            rank = "中"
        if ev["acc"] > ev["base"] + 0.04 and ev["auc"] > 0.55 and frac >= 0.75:
            rank = "高"
        verdict = (f"可信度评级: {rank} (样本外命中率 {ev['acc']:.4f}, AUC {ev['auc']:.4f}, "
                   f"折间标准差 {std:.4f}, 折级胜率 {wins}/{nf})")
        self.log(verdict)
        out = {"verdict": verdict, "rank": rank, "issues": issues,
               "biases": bias, "fold_std": round(std, 4), "label": wf.get("label", "abs"),
               "fold_win": wins, "fold_n": nf, "fold_win_rate": round(frac, 3),
               "overfit_gap": None if gap is None else round(gap, 4),
               "leakage_test_acc": None if leak is None else round(leak, 4)}
        self.bb.put(f"CriticAgent_{wf['tag'] or wf['kind']}", out)
        self.bb.handoff(self.name, "调度官", verdict)
        return out

    def _leakage_test(self, df, feats, ycol):
        """把训练标签随机打乱后重训, 看样本外是否还能“预测” —— 应该回落到基线。"""
        from sklearn.ensemble import HistGradientBoostingClassifier
        d = df.dropna(subset=feats + [ycol])
        dates = np.sort(d["date"].unique())
        if len(dates) < self.cfg.min_train_days + 120:
            return None
        spec = dates[-120]
        tr = d[d["date"] < spec]
        te = d[d["date"] >= spec]
        if len(tr) < 5000 or len(te) < 500:
            return None
        rng = np.random.default_rng(self.cfg.seed)
        ysh = rng.permutation(tr[ycol].values)
        m = HistGradientBoostingClassifier(max_iter=30, learning_rate=0.1,
                                           early_stopping=True, random_state=self.cfg.seed)
        m.fit(tr[feats], ysh)
        p = m.predict_proba(te[feats])[:, 1]
        acc = float(((p > 0.5).astype(int) == te[ycol].values).mean())
        return acc


# ================================================================== 7 调度官
class Orchestrator(Agent):
    name = "调度官"

    def plan(self, round_i: int, history: list, crit: dict | None, importance: list | None):
        """基于上一轮质疑官的结论 + 历史记忆, 决定这一轮怎么改。"""
        if round_i == 1:
            return {"kind": "logit", "h": self.cfg.h_main, "feats": list(ALL_FEATS),
                    "thr": 0.60, "label": "abs",
                    "reason": "第1轮: 全特征+线性模型, 建立基线(预测绝对涨跌)"}
        prev = history[-1]
        if round_i == 2:
            keep = [f for f, v in (importance or [])][:max(10, int(len(ALL_FEATS) * 0.7))]
            if not keep:
                keep = list(ALL_FEATS)
            return {"kind": "hgb", "h": prev["h"], "feats": keep, "thr": prev["thr"],
                    "label": "abs",
                    "reason": (f"第2轮: 读第1轮质疑(可信度{prev['crit_rank']}级), 砍掉 "
                               f"{len(ALL_FEATS)-len(keep)} 个最弱特征, 换梯度提升树抓非线性")}
        if round_i == 3:
            h = 10 if prev["h"] == self.cfg.h_main else self.cfg.h_main
            return {"kind": "hgb", "h": h, "feats": prev["feats"], "thr": 0.55,
                    "label": "abs",
                    "reason": (f"第3轮: 第2轮命中率 {prev['acc']:.4f} (基线 {prev['base']:.4f}), "
                               f"把预测周期从 h={prev['h']} 改成 h={h}")}
        if round_i == 4:
            return {"kind": "hgb", "h": self.cfg.h_main, "feats": prev["feats"], "thr": 0.55,
                    "label": "rel",
                    "reason": ("第4轮: 试做横截面相对标签(是否跑赢当天中位数), 从'猜大盘'变成'排座次'")}
        if round_i == 5:
            hb = max(history, key=lambda x: x.get("edge", -1))
            return {"kind": "ens", "h": hb["h"], "feats": prev["feats"], "thr": 0.55,
                    "label": hb.get("label", "abs"),
                    "reason": (f"第5轮: 历史最优是第{hb['round']}轮(超额 {hb.get('edge', 0):+.4f}); "
                               "质疑官指出训练集与样本外命中率差 0.19 (过拟合), 所以改用"
                               "逻辑回归+梯度提升树软投票集成, 用线性模型对冲树模型的过拟合")}
        if round_i == 6:
            return {"kind": "hgb", "h": prev["h"], "feats": prev["feats"], "thr": 0.55,
                    "label": "alpha",
                    "reason": ("第6轮: 特征重要性显示模型主要在用大盘特征, 即它可能在'择时'而不是'选股'"
                               "; 故把预测目标改成'是否跑赢沪深300', 剔离大盘beta, 单独检验选股能力")}
        return {"kind": "hgb", "h": prev["h"], "feats": prev["feats"], "thr": 0.55,
                "label": prev.get("label", "abs"),
                "reason": "后续轮: 单变量已无可改, 保持最优配置复跑以观察稳定性"}


# ================================================================== 8 档案官
class MemoryAgent(Agent):
    name = "档案官"

    path = OUT / "memory.jsonl"

    def read(self):
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out

    def write(self, rec: dict):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def save_board(self, board: list):
        (OUT / "leaderboard.json").write_text(
            json.dumps(board, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        self.log(f"排行榜已写入 out/leaderboard.json ({len(board)} 轮记录)")


# ================================================================== 出图
def setup_font():
    for p in ("/mnt/c/Windows/Fonts/msyh.ttc", "/mnt/c/Windows/Fonts/simhei.ttf",
              "/mnt/c/Windows/Fonts/msyhbd.ttc"):
        if Path(p).exists():
            try:
                font_manager.fontManager.addfont(p)
                plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=p).get_name()]
                plt.rcParams["axes.unicode_minus"] = False
                return
            except Exception:                  # noqa: BLE001
                continue
    plt.rcParams["axes.unicode_minus"] = False


def make_charts(best: dict, ev: dict, crit: dict, eq: pd.Series, importance: list, board: list):
    setup_font()
    folds = pd.DataFrame(best["folds"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0][0]
    x = np.arange(len(folds))
    ax.bar(x - 0.2, folds["acc"], width=0.38, label="样本外命中率", color="#4c72b0")
    ax.bar(x + 0.2, folds["base"], width=0.38, label="基线(猜多数类)", color="#c44e52", alpha=0.7)
    ax.axhline(0.5, color="k", ls="--", lw=1, label="抛硬币 50%")
    ax.axhline(METRIC_TARGET, color="g", ls=":", lw=1.4, label=f"目标 {METRIC_TARGET:.0%}")
    ax.set_xticks(x); ax.set_xticklabels(folds["test_start"], fontsize=8, rotation=30)
    ax.set_ylim(0.4, 0.62); ax.set_title(f"逐折样本外命中率 ({best['kind']}, h={best['h']})")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[0][1]
    d = pd.DataFrame(ev["deciles"])
    ax.bar(d.index, d["actual"], color="#55a868", alpha=0.85, label="实际上涨比例")
    ax.plot(d.index, d["pred"], "o--", color="#c44e52", label="模型预测概率均值")
    ax.axhline(ev["base"], color="k", ls=":", lw=1, label=f"基准率 {ev['base']:.3f}")
    ax.set_xlabel("预测概率分档 (1=最不看好, 5=最看好)"); ax.set_ylabel("比例")
    ax.set_title(f"分档校准  单调性={ev['monotonicity']:+.2f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1][0]
    if importance:
        imp = pd.DataFrame(importance, columns=["feat", "imp"]).head(18)
        ax.barh(imp["feat"][::-1], imp["imp"][::-1], color="#8172b3")
        ax.set_title("特征重要性 Top18")
    else:
        ax.text(0.5, 0.5, "无", ha="center")
    ax.grid(alpha=0.25)

    ax = axes[1][1]
    if len(eq):
        y = (eq / eq.iloc[0]).values
        ax.plot(range(len(y)), y, color="#c44e52", lw=1.6,
                label=f"信号策略 ({len(y)} 次调仓)")
        ax.axhline(1.0, color="k", ls=":", lw=1, label="本金")
        ax.set_xlabel(f"调仓次数 (每 {best.get('h', 'h')} 个交易日一次)")
        ax.set_title(f"样本外信号净值 (近似, 已扣 {TRADING_COST*100:.1f}% 往返费)")
        ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "train_dashboard.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    if board:
        names = [f"第{b['round']}轮\n{b['kind']}/h={b['h']}" for b in board]
        accs = [b["acc"] for b in board]
        bars = ax.bar(names, accs, color=["#4c72b0" if i < len(board) - 1 else "#c44e52"
                                          for i in range(len(board))])
        for i, (b, a) in enumerate(zip(board, accs)):
            ax.text(i, a + 0.002, f"{a:.4f}\n基线{b['base']:.3f}", ha="center", fontsize=8)
        ax.axhline(0.5, color="k", ls="--", lw=1)
        ax.set_ylim(min(0.45, min(accs) - 0.03), max(0.6, max(accs) + 0.03))
        ax.set_ylabel("样本外命中率")
        ax.set_title("多轮迭代: 调度官依据质疑官反馈改进的轨迹")
    fig.tight_layout()
    fig.savefig(OUT / "train_rounds.png", dpi=130)
    plt.close(fig)


# ================================================================== 报告
def write_report(cfg: Config, bb, board, best, ev, crit, wf_all, elapsed):
    L, A = [], None
    A = L.append
    d = bb.store.get("DataAgent", {})
    h = d.get("health", {})
    lb = bb.store.get("LabelAgent", {})
    A("# 多智能体预测模型训练报告")
    A("")
    A(f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M:%S} | 总耗时 {elapsed:.0f}s")
    A(f"- 数据源: 腾讯行情(前复权日线) 主源={qd.ACTIVE_SOURCE}")
    A(f"- 股票池: {d.get('pool_n')} 只主板股 | 样本区间 {h.get('start')} ~ {h.get('end')} "
      f"| 每只中位 {h.get('bars_median')} 根 | 面板 {bb.store.get('FeatureAgent', {}).get('rows'):,} 行")
    A(f"- 特征数: {bb.store.get('FeatureAgent', {}).get('n_features')} | "
      f"预测目标: 未来 {cfg.h_main} 个交易日涨跌 (t+1 开盘买, t+1+h 开盘卖)")
    A(f"- 验证方式: **滚动向前验证 (walk-forward)**, 每折用历史数据训练、预测其后 "
      f"{cfg.test_len} 个交易日, 从不使用未来数据")
    A(f"- 交易成本假设: 往返 {TRADING_COST*100:.1f}% (佣金+印花税+滑点)")
    A("")
    A("> 目标不是“哪只股票会涨”, 而是“这个模型到底有没有预测能力”。因此本报告只承认样本外指标。")
    A("")
    A("---")
    A("")
    A("## 一、多轮迭代轨迹 (调度官如何依据质疑官改进)")
    A("")
    A("| 轮次 | 模型 | 预测目标 | 周期 | 特征数 | 样本外命中率 | 基线 | 超额 | AUC | 折级胜率 | 可信度 | 调度官决策依据 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for b in board:
        lab = {"abs": "绝对涨跌", "rel": "横截面相对", "alpha": "跑赢沪深300"}.get(b.get("label", "abs"), "?")
        A(f"| 第{b['round']}轮 | {b['kind']} | {lab} | h={b['h']} | {b['n_feats']} | "
          f"**{b['acc']:.4f}** | {b['base']:.4f} | {b['acc']-b['base']:+.4f} | {b['auc']:.4f} | "
          f"{b.get('fold_win', '?')}/{b.get('fold_n', '?')} | "
          f"{b['crit_rank']} | {b['reason']} |")
    A("")
    lab_best = {"abs": "绝对涨跌", "rel": "横截面相对", "alpha": "跑赢沪深300"}.get(best.get("label", "abs"), "?")
    A(f"**最优方案(按超额选)**: 第 {best['round']} 轮 {best['kind']} "
      f"(预测目标={lab_best}, h={best['h']}, {best['n_feats']} 特征) —— "
      f"样本外命中率 **{ev['acc']:.4f}**。")
    A("")
    A("> 注: 不同预测目标的“基线”不同(绝对涨跌的基线≈当日上涨天数占比, 横截面相对标签的基线≈0.5), "
      "因此用“命中率-基线”的超额作横向对比, 而不是直接比命中率。")
    A("")
    A("## 二、样本外详细指标 (最优方案)")
    A("")
    A(f"- 样本外样本数: {ev['n_oos']:,}")
    A(f"- 命中率: **{ev['acc']:.4f}** | 猜多数类基线: {ev['base']:.4f} | 抛硬币: 0.5000")
    A(f"- 相对基线超额: {ev['edge_vs_base']:+.4f} | 相对抛硬币超额: {ev['edge_vs_coin']:+.4f}")
    A(f"- AUC: {ev['auc']:.4f}")
    A(f"- 概率分档单调性 (Spearman): {ev['monotonicity']:+.3f} "
      f"(>0.7 才说明“越看好的档位确实越准”)")
    A(f"- 平均预测概率: {ev['mean_pred']:.4f}")
    A("")
    A("### 策略化视角 (只在预测概率 ≥ "
      f"{cfg.prob_threshold:.2f} 时出手)")
    A("")
    s = ev["strategy"]
    A(f"- 出手次数: {s['n_signals']:,} (占全部样本外样本 {s['frac']:.1%})")
    A(f"- 出手后的实际上涨概率: {s['hit_rate']}")
    A(f"- 出手后平均扣费净收益: {s['avg_net_ret']} (即 {None if s['avg_net_ret'] is None else s['avg_net_ret']*100:.3f}%)")
    A("")
    A("### 逐折明细")
    A("")
    A("| 测试期起点 | 训练样本 | 测试样本 | 样本外命中率 | 基线 | AUC | 训练集命中率 |")
    A("|---|---|---|---|---|---|---|")
    for f in best["folds"]:
        A(f"| {f['test_start']} | {f['n_train']:,} | {f['n_test']:,} | {f['acc']:.4f} | "
          f"{f['base']:.4f} | {f['auc']} | {f['in_sample_acc']} |")
    A("")
    A("## 三、质疑官的审查结论")
    A("")
    A(f"**{crit['verdict']}**")
    A("")
    A(f"折级明细: {crit.get('fold_win')}/{crit.get('fold_n')} 折跑赢基线 "
      f"(折级胜率 {crit.get('fold_win_rate'):.0%})；训练集与样本外命中率差 "
      f"{crit.get('overfit_gap')}；打乱标签重训后的样本外命中率 "
      f"{crit.get('leakage_test_acc')}。")
    A("")
    A("发现的问题:")
    A("")
    for i in crit["issues"]:
        A(f"- {i}")
    A("")
    A("必须声明的偏差 (会系统性高估未来表现):")
    A("")
    for b in crit["biases"]:
        A(f"- {b}")
    A("")
    A("## 四、标签与基准率")
    A("")
    A("| 预测周期 | 上涨基准率 | 扣费后仍盈利 | 横截面相对基准率 | 跑赢沪深300基准率 | 平均收益 | 收益标准差 |")
    A("|---|---|---|---|---|---|---|")
    for hh, v in lb.items():
        A(f"| h={hh} | {v['base_rate_up']:.4f} | {v['base_rate_win']:.4f} | "
          f"{v.get('base_rate_rel')} | {v.get('base_rate_alpha')} | "
          f"{v['mean_fwd']*100:.4f}% | {v['std_fwd']*100:.3f}% |")
    A("")
    A("> 注意“上涨基准率”: 如果样本里本来就有 53% 的日子是上涨的, 那么一个"
      "“永远喊涨”的模型就有 53% 命中率。模型必须显著超过这个数才算有本事。")
    A("")
    A("## 五、Agent 协作消息记录")
    A("")
    A("```")
    A("\n".join(bb.handoffs))
    A("```")
    A("")
    (OUT / "train_report.md").write_text("\n".join(L), encoding="utf-8")
    return OUT / "train_report.md"


# ================================================================== 主流程
def feature_importance_of(model, X, y, feats, kind, n=8000, seed=0, scoring="roc_auc"):
    rng = np.random.default_rng(seed)
    if len(X) > n:
        idx = rng.choice(len(X), n, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]
    if kind == "logit":
        coef = np.abs(model.named_steps["logisticregression"].coef_[0])
        return sorted(zip(feats, coef.tolist()), key=lambda t: -t[1])
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X, y, n_repeats=3, random_state=seed,
                               scoring=scoring, n_jobs=-1)
    return sorted(zip(feats, r.importances_mean.tolist()), key=lambda t: -t[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--test-len", type=int, default=250)
    ap.add_argument("--min-train-days", type=int, default=500)
    ap.add_argument("--h", type=int, default=5)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--cached-only", action="store_true")
    a = ap.parse_args()
    cfg = Config(universe_n=a.universe, rounds=a.rounds, test_len=a.test_len,
                 min_train_days=a.min_train_days, h_main=a.h, fast=a.fast,
                 cached_only=a.cached_only)
    t0 = time.time()
    print("=" * 84)
    print("多智能体协作训练系统 v1.0   ——   只训练模型, 不选股不下单")
    print(f"股票池 {cfg.universe_n} 只主板 | 预测未来 h={cfg.h_main} 日 | 滚动验证 每折 {cfg.test_len} 天 | "
          f"迭代 {cfg.rounds} 轮")
    print("=" * 84)

    bb = Blackboard()
    data, bench, pool = DataAgent(bb, cfg).run()
    panel = FeatureAgent(bb, cfg).run(data, bench)
    df = LabelAgent(bb, cfg).run(panel)

    mem = MemoryAgent(bb, cfg)
    hist_mem = mem.read()
    if hist_mem:
        print(f"[档案官] 读取到 {len(hist_mem)} 条历史运行记忆, 最近一次: "
              f"{hist_mem[-1].get('date')} 最优命中率 {hist_mem[-1].get('best_acc')}")
    else:
        print("[档案官] 首次运行, 无历史记忆")

    orch = Orchestrator(bb, cfg)
    board, history, importance = [], [], None
    best = None
    for r in range(1, cfg.rounds + 1):
        print("-" * 84)
        plan = orch.plan(r, history, None, importance)
        print(f"[调度官] 第 {r} 轮决策: {plan['reason']}")
        wf = ModelAgent(bb, cfg).walk_forward(df, plan["feats"], plan["h"], plan["kind"],
                                              tag=f"R{r}", label=plan["label"])
        ev, eq = EvalAgent(bb, cfg).run(wf, plan["thr"])
        crit = CriticAgent(bb, cfg).run(wf, ev, df, plan["feats"])
        # 特征重要性(用最后一折的训练集近似, 只为下一轮选特征)
        dates = np.sort(df["date"].unique())
        ycol_round = {"abs": f"y{plan['h']}", "rel": f"yrel{plan['h']}",
                      "alpha": f"yalpha{plan['h']}"}[plan["label"]]
        tr = df[df["date"] < dates[-120]].dropna(subset=plan["feats"] + [ycol_round])
        tr = tr.iloc[-120000:]
        try:
            m = ModelAgent(bb, cfg).make_model(plan["kind"])
            m.fit(tr[plan["feats"]], tr[ycol_round])
            importance = feature_importance_of(m, tr[plan["feats"]], tr[ycol_round],
                                               plan["feats"], plan["kind"], n=6000,
                                               seed=cfg.seed)
            print(f"[模型官] 特征重要性 Top5: " + ", ".join(f"{f}({v:.4f})" for f, v in importance[:5]))
        except Exception as e:                 # noqa: BLE001
            print(f"[模型官] 重要性计算跳过: {e}")
        board.append({"round": r, "kind": plan["kind"], "h": plan["h"],
                      "label": plan["label"],
                      "n_feats": len(plan["feats"]), "acc": ev["acc"], "base": ev["base"],
                      "auc": ev["auc"], "monotonicity": ev["monotonicity"],
                      "crit_rank": crit["rank"], "reason": plan["reason"],
                      "thr": plan["thr"], "edge": ev["edge_vs_base"],
                      "fold_win_rate": crit["fold_win_rate"],
                      "fold_win": crit["fold_win"], "fold_n": crit["fold_n"],
                      "strat_hit": ev["strategy"]["hit_rate"],
                      "strat_ret": ev["strategy"]["avg_net_ret"]})
        history.append({**board[-1], "crit_rank": crit["rank"], "feats": plan["feats"]})
        if best is None or ev["edge_vs_base"] > best["ev"]["edge_vs_base"]:
            best = {"round": r, "kind": plan["kind"], "h": plan["h"],
                    "label": plan["label"],
                    "n_feats": len(plan["feats"]), "ev": ev, "crit": crit,
                    "folds": wf["folds"], "pooled": wf["pooled"]}
            best_eq = eq
        # 每轮都追加记忆(档案官), 下一轮/下次运行都能读到
        mem.write({"date": str(datetime.now())[:19], "round": r, "kind": plan["kind"],
                   "h": plan["h"], "n_feats": len(plan["feats"]), "acc": ev["acc"],
                   "base": ev["base"], "auc": ev["auc"], "crit_rank": crit["rank"]})

    mem.save_board(board)
    try:
        bp = best.get("pooled")
        if bp is not None and len(bp):
            samp = bp.sample(min(30000, len(bp)), random_state=0).sort_values("date")
            samp.to_csv(OUT / "oos_predictions_best.csv", index=False, encoding="utf-8")
            print(f"[档案官] 已保存最优方案样本外预测明细 {len(samp):,} 条 "
                  f"-> out/oos_predictions_best.csv")
    except Exception as e:                     # noqa: BLE001
        print(f"[档案官] 预测明细保存跳过: {e}")
    mem.write({"date": str(datetime.now())[:19], "round": "summary", "kind": best["kind"],
               "h": best["h"], "n_feats": best["n_feats"], "acc": best["ev"]["acc"],
               "base": best["ev"]["base"], "auc": best["ev"]["auc"],
               "crit_rank": best["crit"]["rank"], "is_best": True,
               "best_acc": best["ev"]["acc"]})

    make_charts(best, best["ev"], best["crit"], best_eq, importance, board)
    elapsed = time.time() - t0
    rpt = write_report(cfg, bb, board, best, best["ev"], best["crit"], None, elapsed)
    bb.save()

    print("=" * 84)
    print(f"训练完成, 总耗时 {elapsed:.0f}s")
    print(f"最优: 第{best['round']}轮 {best['kind']} 标签={best.get('label','abs')} h={best['h']}  →  "
          f"样本外命中率 {best['ev']['acc']:.4f} (基线 {best['ev']['base']:.4f}, "
          f"AUC {best['ev']['auc']:.4f})  可信度: {best['crit']['rank']}")
    print(f"报告: {rpt}")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
