#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方向 D: 指数择时系统 timing_agents.py  v1.0
=============================================
目标(与方向 A 不同): 不选股, 只回答一个问题 ——
    "未来 N 个交易日, 沪深300 该不该持仓?"
然后严格检验这个判断到底有没有用:
  * 命中率是否显著高于基准率(二项检验 + 置信区间)
  * 择时策略是否打得过"一直满仓"(买入持有)
  * 在从未参与任何选择的最终留出集上是否依然有效

九个 Agent(人设) 与协作:
  1 指数数据官 IndexDataAgent   Baostock 拉 5 个指数 21 年日线 + 体检
  2 特征官 FeatureAgent         指数级特征(动量/均线/波动率状态/量能/回撤/季节性)
  3 标注官 LabelAgent           未来 h 日涨跌 + 扣费后仍盈利
  4 模型官 ModelAgent           walk-forward 滚动样本外预测
  5 时机评估官 TimingEvalAgent  命中率 + 非重叠调仓回测 + 对比买入持有 + 二项检验
  6 质疑官 CriticAgent          小样本/重叠/多重比较/盈亏比审查
  7 调度官 Orchestrator         多轮迭代, 且只在"开发期"上调参
  8 外检官 ExternalValidator    把最终模型套到上证/中证500/创业板, 看跨标的是否同样有效
  9 档案官 MemoryAgent          排行榜 + 跨会话记忆

防过拟合的关键设计: 最后 20% 的交易日被划为"最终留出集", 全程不参与任何
特征/模型/轮次的选择; 只在最后用一次, 用来回答"这套东西真能用吗"。

用法:
    python timing_agents.py
    python timing_agents.py --rounds 6 --h 10
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import quant_data as qd
from train_agents import Blackboard, Agent, MemoryAgent, setup_font, feature_importance_of

ROOT = Path(__file__).resolve().parent
CACHE, OUT, LOGS = ROOT / "cache", ROOT / "out", ROOT / "logs"
for _d in (CACHE, OUT, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

COST = 0.0010        # 指数ETF 往返成本约 0.10%(无印花税, 佣金低, 有滑点)
CASH_RATE = 0.015    # 空仓时资金收益(货币基金年化 1.5%)
TARGET = 0.55        # 命中率门槛


@dataclass
class Cfg:
    code: str = "sh.000300"
    name: str = "沪深300"
    horizons: tuple = (5, 10, 20)
    h_main: int = 10
    test_len: int = 500          # 每次样本外测试约 2 年
    min_train_days: int = 750    # 前 3 年只训练
    holdout_frac: float = 0.20   # 最后 20% 作为最终留出集
    thr: float = 0.55
    rounds: int = 6
    seed: int = 0


FEATS = [
    "ret1", "ret2", "ret3", "ret5", "ret10", "ret20", "ret60", "ret120",
    "ma5_gap", "ma10_gap", "ma20_gap", "ma60_gap", "ma120_gap", "ma250_gap",
    "ma5_slope", "ma20_slope", "ma60_slope", "ma120_slope",
    "ma20_ma60", "ma60_ma120",
    "vol20", "vol60", "vol120", "vol_ratio_20_60", "vol20_vs_250", "atr_pct",
    "rsi14", "rsi28", "macd_hist", "dif_gap",
    "pos60", "pos120", "pos250", "dd60", "dd120", "dd250", "dd_all",
    "v_ratio_5_60", "v_ratio_20_120", "amt_ratio_5_60", "amt_z",
    "skew20", "kurt20", "max1_20", "min1_20", "streak",
    "month", "dayofweek", "month_progress",
]
FEATS_TREND = ["ret5", "ret10", "ret20", "ret60", "ma20_gap", "ma60_gap",
               "ma120_gap", "ma20_slope", "ma60_slope", "ma20_ma60", "ma60_ma120"]
FEATS_VOL = ["vol20", "vol60", "vol120", "vol_ratio_20_60", "vol20_vs_250",
             "atr_pct", "dd60", "dd120", "dd250", "dd_all"]


# ================================================================== 1 指数数据官
class IndexDataAgent(Agent):
    name = "指数数据官"

    def run(self):
        data = {}
        for code, nm in qd.BS_INDICES.items():
            try:
                df = qd.index_hist_bs(code)
            except Exception as e:                    # noqa: BLE001
                self.log(f"{nm} 拉取失败: {e}")
                continue
            if df is not None and len(df) > 500:
                data[code] = df
                self.log(f"{nm} {code}: {len(df)} 根 {df.index[0].date()} ~ {df.index[-1].date()}")
        if self.cfg.code not in data:
            raise RuntimeError("主指数数据缺失")
        main = data[self.cfg.code]
        self.log(f"主标的 = {self.cfg.name} ({self.cfg.code}), {len(main)} 个交易日, "
                 f"缺失率 {main[['open', 'close', 'high', 'low']].isna().mean().mean():.2e}")
        self.bb.put("IndexDataAgent", {"n_indices": len(data),
                                       "start": str(main.index[0].date()),
                                       "end": str(main.index[-1].date()),
                                       "n_bars": len(main),
                                       "zero_vol": int((main["volume"] <= 0).sum())})
        self.bb.handoff(self.name, "特征官",
                        f"{len(data)} 个指数就绪; 主标的 {len(main)} 根 "
                        f"({main.index[0].date()} ~ {main.index[-1].date()})")
        return main, data


# ================================================================== 2 特征官
class FeatureAgent(Agent):
    name = "特征官"

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]
        f = pd.DataFrame(index=df.index)
        r = c.pct_change()
        for n in (1, 2, 3, 5, 10, 20, 60, 120):
            f[f"ret{n}"] = c.pct_change(n)
        for n in (5, 10, 20, 60, 120, 250):
            ma = c.rolling(n).mean()
            f[f"ma{n}_gap"] = c / ma - 1
        for n in (5, 20, 60, 120):
            f[f"ma{n}_slope"] = c.rolling(n).mean().pct_change(5)
        f["ma20_ma60"] = c.rolling(20).mean() / c.rolling(60).mean() - 1
        f["ma60_ma120"] = c.rolling(60).mean() / c.rolling(120).mean() - 1
        av = np.sqrt(250)
        for n in (20, 60, 120):
            f[f"vol{n}"] = r.rolling(n).std() * av
        f["vol_ratio_20_60"] = f["vol20"] / f["vol60"]
        f["vol20_vs_250"] = f["vol20"] / (r.rolling(250).std() * av)
        tr = np.maximum(df["high"] - df["low"],
                        np.maximum((df["high"] - c.shift()).abs(),
                                   (df["low"] - c.shift()).abs()))
        f["atr_pct"] = tr.rolling(14).mean() / c
        for n in (14, 28):
            d = c.diff()
            up = d.clip(lower=0).rolling(n).mean()
            dn = (-d.clip(upper=0)).rolling(n).mean()
            f[f"rsi{n}"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
        e12 = c.ewm(span=12, adjust=False).mean()
        e26 = c.ewm(span=26, adjust=False).mean()
        dif = e12 - e26
        dea = dif.ewm(span=9, adjust=False).mean()
        f["macd_hist"] = (dif - dea) * 2 / c
        f["dif_gap"] = dif / c
        for n in (60, 120, 250):
            hi, lo = df["high"].rolling(n).max(), df["low"].rolling(n).min()
            f[f"pos{n}"] = (c - lo) / (hi - lo).replace(0, np.nan)
            f[f"dd{n}"] = c / hi - 1
        f["dd_all"] = c / c.cummax() - 1
        v = df["volume"].replace(0, np.nan)
        a = df["amount"].replace(0, np.nan)
        f["v_ratio_5_60"] = v.rolling(5).mean() / v.rolling(60).mean()
        f["v_ratio_20_120"] = v.rolling(20).mean() / v.rolling(120).mean()
        f["amt_ratio_5_60"] = a.rolling(5).mean() / a.rolling(60).mean()
        la = np.log(a)
        f["amt_z"] = (la - la.rolling(250).mean()) / la.rolling(250).std()
        f["skew20"] = r.rolling(20).skew()
        f["kurt20"] = r.rolling(20).kurt()
        f["max1_20"] = r.rolling(20).max()
        f["min1_20"] = r.rolling(20).min()
        sgn = np.sign(r.fillna(0).values)
        st = np.zeros(len(sgn))
        for i in range(1, len(sgn)):
            st[i] = st[i - 1] + sgn[i] if (sgn[i] == sgn[i - 1] and sgn[i] != 0) else sgn[i]
        f["streak"] = st
        f["month"] = df.index.month
        f["dayofweek"] = df.index.dayofweek
        f["month_progress"] = df.index.day / 30.0
        f.insert(0, "close", c)
        f.insert(1, "open", df["open"])
        return f.reset_index()

    def run(self, main: pd.DataFrame):
        f = self.build(main)
        bad = f[FEATS].isna().mean()
        self.log(f"构造 {len(FEATS)} 个指数级特征, {len(f)} 行; "
                 f"缺失最多的 3 个: " + ", ".join(f"{k}({v:.1%})" for k, v in bad.nlargest(3).items()))
        self.bb.put("FeatureAgent", {"n_feats": len(FEATS), "rows": len(f),
                                     "note": "全部为指数级、尺度无关的特征, 便于跨指数外部验证"})
        self.bb.handoff(self.name, "标注官", f"{len(FEATS)} 个指数特征就绪 ({len(f)} 行)")
        return f


# ================================================================== 3 标注官
class LabelAgent(Agent):
    name = "标注官"

    def run(self, f: pd.DataFrame):
        df = f
        info = {}
        for h in self.cfg.horizons:
            # t 日收盘决策, t+1 开盘买入, t+1+h 开盘卖出
            df[f"entry{h}"] = df["open"].shift(-1)
            df[f"exit{h}"] = df["open"].shift(-(1 + h))
            df[f"fwd{h}"] = df[f"exit{h}"] / df[f"entry{h}"] - 1
            df[f"ynet{h}"] = df[f"fwd{h}"] - COST
            df[f"y{h}"] = (df[f"fwd{h}"] > 0).astype(float)
            df.loc[df[f"fwd{h}"].isna(), f"y{h}"] = np.nan
            s = df[f"fwd{h}"].dropna()
            info[h] = {"base_up": round(float((s > 0).mean()), 4),
                       "base_net": round(float((s - COST > 0).mean()), 4),
                       "mean": round(float(s.mean()), 5), "std": round(float(s.std()), 5)}
        self.log("基准率: " + "; ".join(
            f"h={h}: 上涨 {v['base_up']:.3f} / 扣费后 {v['base_net']:.3f}" for h, v in info.items()))
        self.bb.put("LabelAgent", info)
        self.bb.handoff(self.name, "模型官",
                        f"h={self.cfg.h_main} 基准上涨率 {info[self.cfg.h_main]['base_up']:.3f} "
                        f"(择时必须显著超过它)")
        return df


# ================================================================== 4 模型官
class ModelAgent(Agent):
    name = "模型官"

    def make(self, kind: str):
        if kind == "logit":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            return make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=400, C=0.3, n_jobs=-1))
        if kind == "hgb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(
                max_iter=120, learning_rate=0.06, max_leaf_nodes=15,
                min_samples_leaf=40, l2_regularization=1.0,
                early_stopping=True, n_iter_no_change=10,
                validation_fraction=0.15, random_state=self.cfg.seed)
        if kind == "ens":
            from sklearn.ensemble import VotingClassifier
            return VotingClassifier([("lr", self.make("logit")), ("hgb", self.make("hgb"))],
                                    voting="soft")
        if kind == "rf":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=200, max_depth=5,
                                          min_samples_leaf=30, n_jobs=-1,
                                          random_state=self.cfg.seed)
        raise ValueError(kind)

    def walk_forward(self, df, feats, h, kind, label="abs", tag="", end_date=None):
        dates = np.sort(df["date"].unique())
        if end_date is not None:
            dates = dates[dates <= np.datetime64(end_date)]
        ycol = {"abs": f"y{h}", "net": f"ynet{h}"}[label] if label != "net" else f"ynet{h}"
        if label == "net":
            df = df.assign(**{f"yn{h}": (df[f"ynet{h}"] > 0).astype(float)})
            ycol = f"yn{h}"
        folds, preds = [], []
        for si in range(self.cfg.min_train_days, len(dates), self.cfg.test_len):
            te_dates = dates[si:si + self.cfg.test_len]
            if len(te_dates) < 60:
                break
            d0 = te_dates[0]
            tr = df[df["date"] < d0].dropna(subset=feats + [ycol])
            te = df[df["date"].isin(te_dates)].dropna(subset=feats + [ycol])
            if len(tr) < 400 or len(te) < 60:
                continue
            m = self.make(kind)
            m.fit(tr[feats], tr[ycol])
            p = m.predict_proba(te[feats])[:, 1]
            from sklearn.metrics import roc_auc_score
            acc = float(((p > 0.5).astype(int) == te[ycol]).mean())
            base = float(max(te[ycol].mean(), 1 - te[ycol].mean()))
            auc = float(roc_auc_score(te[ycol], p)) if te[ycol].nunique() > 1 else np.nan
            folds.append({"test_start": str(pd.Timestamp(d0).date()), "n_train": len(tr),
                          "n_test": len(te), "acc": round(acc, 4), "base": round(base, 4),
                          "auc": None if np.isnan(auc) else round(auc, 4)})
            t = te[["date", f"y{h}", f"ynet{h}", f"fwd{h}"]].copy().reset_index(drop=True)
            t["p"] = p
            preds.append(t)
        pooled = pd.concat(preds) if preds else pd.DataFrame()
        return {"folds": folds, "pooled": pooled, "kind": kind, "h": h, "tag": tag,
                "label": label, "ycol": ycol}

    def run(self, df, feats, h, kind, label="abs", end_date=None):
        t0 = time.time()
        r = self.walk_forward(df, feats, h, kind, label=label, end_date=end_date)
        self.log(f"{kind}/h={h}/标签={label}/{len(feats)}特征 → 样本外 {len(r['pooled'])} 条, "
                 f"{len(r['folds'])} 折, {time.time()-t0:.1f}s")
        self.bb.handoff(self.name, "时机评估官", f"{kind}(h={h}) 完成 {len(r['folds'])} 折")
        return r


# ================================================================== 5 时机评估官
class TimingEvalAgent(Agent):
    name = "时机评估官"

    def backtest(self, pooled: pd.DataFrame, index_df: pd.DataFrame, h: int, thr: float):
        """非重叠调仓回测: 每 h 个交易日做一次决策, 持仓则吃 h 日收益, 空仓则拿货币收益。
        这样每个观测相互独立, 二项检验才成立。"""
        pm = dict(zip(pooled["date"], pooled["p"]))
        o = index_df["open"]
        idx_pos = {d: i for i, d in enumerate(index_df.index)}
        dates = sorted(pm.keys())
        eq, bh = 1.0, 1.0
        curve, bcurve, periods = [], [], []
        n_trade, hits, rets = 0, [], []
        raw, pnl = [], []
        for d in dates[::max(1, h)]:
            i = idx_pos.get(d)
            if i is None or i + 1 + h >= len(index_df):
                continue
            entry, ex = o.iloc[i + 1], o.iloc[i + 1 + h]
            if not np.isfinite(entry) or not np.isfinite(ex) or entry <= 0:
                continue
            rr = ex / entry - 1
            raw.append(rr)
            p = pm[d]
            if p >= thr:
                rn = rr - COST
                eq *= (1 + rn)
                n_trade += 1
                hits.append(1 if rn > 0 else 0)
                rets.append(rn)
            else:
                rn = CASH_RATE * h / 250.0
                eq *= (1 + rn)
            pnl.append(rn)
            bh *= (1 + rr)
            curve.append(eq)
            bcurve.append(bh)
            periods.append(d)
        return {"eq": np.array(curve), "bh": np.array(bcurve), "dates": periods,
                "n_period": len(periods), "n_trade": n_trade, "hits": hits, "rets": rets,
                "raw": raw, "pnl": pnl}

    def _binom(self, hits, base):
        from scipy.stats import binomtest
        if not hits:
            return None, None, None
        k, n = int(sum(hits)), len(hits)
        r = binomtest(k, n, p=base, alternative="greater")
        lo, hi = r.proportion_ci(confidence_level=0.90, method="wilson")
        return round(float(k / n), 4), round(float(r.pvalue), 4), (round(lo, 4), round(hi, 4))

    def run(self, wf: dict, index_df: pd.DataFrame, thr: float):
        pooled, h = wf["pooled"], wf["h"]
        if pooled.empty:
            return None, None
        from sklearn.metrics import roc_auc_score
        ycol = wf["ycol"]
        y, p = pooled[ycol].values, pooled["p"].values
        acc = float(((p > 0.5).astype(int) == y).mean())
        base = float(max(y.mean(), 1 - y.mean()))
        auc = float(roc_auc_score(y, p)) if len(set(y)) > 1 else np.nan
        q = pd.qcut(pooled["p"], 5, labels=False, duplicates="drop")
        tab = pooled.groupby(q).agg(pred=("p", "mean"), actual=(ycol, "mean"))
        from scipy.stats import spearmanr
        mono = float(spearmanr(tab.index.values, tab["actual"].values).correlation)

        bt = self.backtest(pooled, index_df, h, thr)
        # ==== 择时能力的正确度量: 不受指数上行漂移影响 ====
        pm = dict(zip(pooled["date"], pooled["p"]))
        periods, raws = bt["dates"], bt["raw"]
        A = [r for d, r in zip(periods, raws) if pm[d] >= thr]      # 持仓信号期
        B = [r for d, r in zip(periods, raws) if pm[d] < thr]       # 空仓信号期
        skill = spread_p = hit_hold = hit_cash = None
        if len(A) > 15 and len(B) > 15:
            from scipy.stats import ttest_ind
            tt, pv = ttest_ind(A, B, equal_var=False, alternative="greater")
            skill = round(float(np.mean(A) - np.mean(B)), 5)
            spread_p = round(float(pv), 4)
            hit_hold = round(float(np.mean([1 if x > 0 else 0 for x in A])), 4)
            hit_cash = round(float(np.mean([1 if x > 0 else 0 for x in B])), 4)

        def _sharpe(x):
            if len(x) < 10:
                return None
            s = float(np.std(x, ddof=1))
            return None if s == 0 else round(float(np.mean(x) / s * np.sqrt(250.0 / h)), 3)

        sharpe_s, sharpe_b = _sharpe(bt["pnl"]), _sharpe(raws)
        # 命中率(全部非重叠调仓期) + 二项检验
        ym = dict(zip(pooled["date"], pooled[ycol]))
        ph, yh = [], []
        for d in bt["dates"]:
            ph.append(1 if pm[d] > 0.5 else 0)
            yh.append(int(ym[d]))
        hitall = float(np.mean([a == b for a, b in zip(ph, yh)])) if yh else None
        baseall = float(max(np.mean(yh), 1 - np.mean(yh))) if yh else None
        hyp, pval, ci = self._binom([a == b for a, b in zip(ph, yh)], baseall) if yh else (None, None, None)

        # 组合经济学
        eq, bh = bt["eq"], bt["bh"]
        yrs = bt["n_period"] * h / 250.0 if bt["n_period"] else 0
        ann = float(eq[-1] ** (1 / yrs) - 1) if yrs > 0.2 and len(eq) else None
        annb = float(bh[-1] ** (1 / yrs) - 1) if yrs > 0.2 and len(bh) else None
        mdd = float((eq / np.maximum.accumulate(eq) - 1).min()) if len(eq) else None
        mddb = float((bh / np.maximum.accumulate(bh) - 1).min()) if len(bh) else None
        res = {"acc": round(acc, 4), "base": round(base, 4), "edge": round(acc - base, 4),
               "auc": None if np.isnan(auc) else round(auc, 4),
               "monotonicity": round(mono, 3), "n_oos": int(len(pooled)),
               "deciles": tab.reset_index().to_dict("records"),
               "n_period": bt["n_period"], "n_trade": bt["n_trade"],
               "trade_frac": round(bt["n_trade"] / bt["n_period"], 3) if bt["n_period"] else None,
               "period_hit": hitall, "period_base": baseall, "p_value": pval, "ci90": ci,
               "skill": skill, "spread_p": spread_p, "hit_hold": hit_hold,
               "hit_cash": hit_cash, "n_hold": len(A), "n_cash": len(B),
               "avg_ret_hold": None if not A else round(float(np.mean(A)), 5),
               "avg_ret_cash": None if not B else round(float(np.mean(B)), 5),
               "sharpe_strat": sharpe_s, "sharpe_bh": sharpe_b,
               "total_ret": None if not len(eq) else round(float(eq[-1] - 1), 4),
               "bh_total_ret": None if not len(bh) else round(float(bh[-1] - 1), 4),
               "ann_ret": None if ann is None else round(ann, 4),
               "bh_ann_ret": None if annb is None else round(annb, 4),
               "max_dd": None if mdd is None else round(mdd, 4),
               "bh_max_dd": None if mddb is None else round(mddb, 4),
               "years": round(yrs, 2)}
        self.log(f"样本外命中率 {acc:.4f}(基准 {base:.4f}) | AUC "
                 f"{'N/A' if np.isnan(auc) else f'{auc:.4f}'} | 真实择时能力 skill="
                 f"{skill} (p={spread_p}) | 调仓期 {bt['n_period']} 次 | 年化 "
                 f"{ann if ann is None else f'{ann:.2%}'} vs 满仓 "
                 f"{annb if annb is None else f'{annb:.2%}'}")
        self.bb.put(f"Eval_{wf['tag']}", res)
        self.bb.handoff(self.name, "质疑官",
                        f"{wf['tag']} skill={skill}(p={spread_p}) 年化 {ann} vs {annb}")
        return res, bt


# ================================================================== 6 质疑官
class CriticAgent(Agent):
    name = "质疑官"

    def run(self, wf: dict, ev: dict, df: pd.DataFrame, feats: list):
        folds = pd.DataFrame(wf["folds"])
        issues = []
        nf = len(folds)
        wins = int((folds["acc"] > folds["base"]).sum())
        frac = wins / nf if nf else 0
        issues.append(f"折级胜率 {wins}/{nf} ({frac:.0%})")
        std = float(folds["acc"].std())
        issues.append(f"各折命中率标准差 {std:.4f}")
        # 有效独立样本量: 调仓期数才是真正的样本数
        eff = ev.get("n_period") or 0
        issues.append(f"有效独立样本只有 {eff} 个调仓期 —— 命中率的标准误约 "
                      f"{(0.5/np.sqrt(eff) if eff else float('nan')):.3f}, "
                      f"所以小于约 {2*0.5/np.sqrt(eff)*100 if eff else float('nan'):.1f} 个百分点的"
                      f"优势在统计上都不可区分")
        if ev.get("skill") is not None:
            issues.append(
                f"【核心指标】真实择时能力 skill = 持仓信号期上涨概率 {ev['hit_hold']:.3f} − "
                f"空仓信号期 {ev['hit_cash']:.3f} = **{ev['skill']:+.4f}**; 持仓期平均 h 日收益 "
                f"{ev['avg_ret_hold']:+.3%} vs 空仓期 {ev['avg_ret_cash']:+.3%}, "
                f"t 检验 p={ev['spread_p']}")
            if ev["spread_p"] is not None and ev["spread_p"] > 0.10:
                issues.append("该 t 检验不显著(p>0.10): 模型把好时段与差时段分开的能力"
                              "无法与运气区分")
        else:
            issues.append("持仓/空仓期样本不足, 无法计算择时能力")
        issues.append(f"注意: 指数有天然上行漂移, 所以'一直满仓'的猜涨命中率本来就有 "
                      f"{ev['base']:.3f}; 总命中率 {ev['acc']:.4f} 低于它并不等于模型是坏的, "
                      f"真正要看的是上面的 skill")
        if ev.get("ann_ret") is not None and ev.get("bh_ann_ret") is not None:
            if ev["ann_ret"] <= ev["bh_ann_ret"]:
                issues.append(f"择时年化 {ev['ann_ret']:.2%} 没跑赢一直满仓 "
                              f"{ev['bh_ann_ret']:.2%} —— 择时白做")
            else:
                issues.append(f"择时年化 {ev['ann_ret']:.2%} 跑赢满仓 {ev['bh_ann_ret']:.2%}, "
                              f"最大回撤 {ev['max_dd']:.2%} vs {ev['bh_max_dd']:.2%}")
        issues.append("多重比较风险: 本次跑了多轮实验并挑最好的, 即使全是噪声, "
                      "挑出来的那一轮也会显得不错; 所以最终结论只看'最终留出集'")
        biases = [
            "单一标的: 只在沪深300 上验证, 结论未必适用于其他指数(由外检官检验)",
            "历史区间偏差: 2005-2026 含两轮大牛熊, 但未来未必重复同样的波动结构",
            "执行假设: 按开盘价成交、无冲击成本、无涨跌停无法成交的问题",
            "现金收益: 空仓期按年化 1.5% 计, 若实际更低则择时优势缩水",
            "参数选择: 阈值/周期都是在开发期上挑的, 存在选择偏差",
        ]
        rank = "低"
        if (ev.get("spread_p") is not None and ev["spread_p"] < 0.10
                and (ev.get("skill") or 0) > 0.002 and frac >= 0.5):
            rank = "中"
        if (ev.get("spread_p") is not None and ev["spread_p"] < 0.05
                and (ev.get("skill") or 0) > 0.005 and frac >= 0.6
                and (ev.get("sharpe_strat") or -9) > (ev.get("sharpe_bh") or 0)):
            rank = "高"
        verdict = (f"可信度评级: {rank} (skill {ev.get('skill')} p={ev.get('spread_p')}, "
                   f"总命中率 {ev['acc']:.4f} vs 基准 {ev['base']:.4f}, 折级胜率 {wins}/{nf})")
        self.log(verdict)
        out = {"verdict": verdict, "rank": rank, "issues": issues, "biases": biases,
               "fold_win": wins, "fold_n": nf, "fold_win_rate": round(frac, 3),
               "fold_std": round(std, 4)}
        self.bb.put(f"Critic_{wf['tag']}", out)
        self.bb.handoff(self.name, "调度官", verdict)
        return out


# ================================================================== 7 调度官
class Orchestrator(Agent):
    name = "调度官"

    def plan(self, r: int, hist: list, importance=None):
        if r == 1:
            return {"kind": "logit", "h": self.cfg.h_main, "label": "abs",
                    "feats": list(FEATS), "thr": self.cfg.thr,
                    "reason": "第1轮: 全特征 + 逻辑回归, 建立基线(预测绝对涨跌)"}
        prev = hist[-1]
        if r == 2:
            keep = [f for f, v in (importance or [])][:max(12, int(len(FEATS) * 0.7))] \
                or list(FEATS)
            return {"kind": "hgb", "h": prev["h"], "label": "abs", "feats": keep,
                    "thr": prev["thr"],
                    "reason": f"第2轮: 读第1轮质疑(可信度{prev['crit_rank']}), 砍掉 "
                              f"{len(FEATS)-len(keep)} 个最弱特征, 换梯度提升树"}
        if r == 3:
            return {"kind": "hgb", "h": 5, "label": "abs", "feats": prev["feats"],
                    "thr": prev["thr"],
                    "reason": "第3轮: h=10 → h=5 (调仓期数翻倍, 统计上更容易检验出真实优势)"}
        if r == 4:
            return {"kind": "hgb", "h": 20, "label": "abs", "feats": prev["feats"],
                    "thr": prev["thr"],
                    "reason": "第4轮: h=5 → h=20 (更长周期受噪声影响更小, 但样本更少)"}
        if r == 5:
            return {"kind": "ens", "h": prev["h"], "label": "abs", "feats": prev["feats"],
                    "thr": prev["thr"],
                    "reason": "第5轮: 逻辑回归+梯度提升树软投票集成, 用小样本上更稳的线性模型"
                              "对冲树模型过拟合"}
        return {"kind": "hgb", "h": prev["h"], "label": "abs",
                "feats": sorted(set(FEATS_TREND + FEATS_VOL)), "thr": prev["thr"],
                "reason": "第6轮: 只用'趋势+波动率状态'两类特征(11+10个)做极简模型, "
                          "样本量小的时候特征越少越不容易过拟合"}


# ================================================================== 8 外检官
class ExternalValidator(Agent):
    name = "外检官"

    def run(self, df_main: pd.DataFrame, feats, h, kind, others: dict, cfg):
        from sklearn.metrics import roc_auc_score
        end = df_main["date"].max()
        dates = np.sort(df_main["date"].unique())
        dev_end = dates[int(len(dates) * (1 - cfg.holdout_frac))]
        tr = df_main[df_main["date"] <= dev_end].dropna(subset=feats + [f"y{h}"])
        if len(tr) < 400:
            return {}
        m = ModelAgent(self.bb, cfg).make(kind)
        m.fit(tr[feats], tr[f"y{h}"])
        out = {}
        for code, nm in qd.BS_INDICES.items():
            if code == cfg.code or code not in others:
                continue
            o = others[code]
            fo = FeatureAgent(self.bb, cfg).build(o)
            fo = LabelAgent(self.bb, cfg).run(fo)
            te = fo[fo["date"] > dev_end].dropna(subset=feats + [f"y{h}"])
            if len(te) < 120:
                continue
            p = m.predict_proba(te[feats])[:, 1]
            y = te[f"y{h}"].values
            acc = float(((p > 0.5).astype(int) == y).mean())
            base = float(max(y.mean(), 1 - y.mean()))
            auc = float(roc_auc_score(y, p)) if len(set(y)) > 1 else np.nan
            out[nm] = {"n": int(len(te)), "acc": round(acc, 4), "base": round(base, 4),
                       "edge": round(acc - base, 4),
                       "auc": None if np.isnan(auc) else round(auc, 4),
                       "start": str(te["date"].min().date()), "end": str(te["date"].max().date())}
        if out:
            self.log("跨指数外部验证(同一模型, 同一留出期): " + "; ".join(
                f"{k} 超额 {v['edge']:+.3f} (AUC {v['auc']})" for k, v in out.items()))
        else:
            self.log("无可用的其他指数做外部验证")
        self.bb.put("ExternalValidator", out)
        self.bb.handoff(self.name, "报告", f"外检 {len(out)} 个指数")
        return out


# ================================================================== 出图/报告
def make_charts(best, ev, bt, importance, board, label, hold_ev=None, hold_bt=None):
    setup_font()
    folds = pd.DataFrame(best["folds"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0][0]
    x = np.arange(len(folds))
    ax.bar(x - 0.2, folds["acc"], width=0.38, label="样本外命中率", color="#4c72b0")
    ax.bar(x + 0.2, folds["base"], width=0.38, label="基准率", color="#c44e52", alpha=0.7)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.axhline(TARGET, color="g", ls=":", lw=1.2, label=f"门槛 {TARGET:.0%}")
    ax.set_xticks(x)
    ax.set_xticklabels(folds["test_start"], fontsize=7, rotation=40)
    ax.set_ylim(0.3, 0.75)
    ax.set_title(f"逐折命中率 ({best['kind']}, h={best['h']})  {label}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[0][1]
    if bt is not None and len(bt["eq"]):
        ax.plot(bt["eq"], color="#c44e52", lw=1.8, label="择时策略")
        ax.plot(bt["bh"], color="#4c72b0", lw=1.3, label="一直满仓(买入持有)")
        ax.set_yscale("log")
        ax.set_title(f"样本外净值对比 ({ev['n_period']} 次调仓, 每 {best['h']} 日一次)")
        ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1][0]
    if importance:
        imp = pd.DataFrame(importance, columns=["feat", "imp"]).head(16)
        ax.barh(imp["feat"][::-1], imp["imp"][::-1], color="#8172b3")
        ax.set_title("特征重要性 Top16")
    ax.grid(alpha=0.25)

    ax = axes[1][1]
    labs, hp, hc = [], [], []
    for tag, e in (("开发期", ev), ("留出期", hold_ev or {})):
        if not e or e.get("hit_hold") is None:
            continue
        labs.append(tag)
        hp.append(e["hit_hold"])
        hc.append(e["hit_cash"])
    if labs:
        x2 = np.arange(len(labs))
        ax.bar(x2 - 0.2, hp, width=0.38, color="#c44e52", label="持仓信号期上涨概率")
        ax.bar(x2 + 0.2, hc, width=0.38, color="#4c72b0", label="空仓信号期上涨概率")
        for i, (a, b) in enumerate(zip(hp, hc)):
            ax.text(i, max(a, b) + 0.012, f"skill {a-b:+.4f}", ha="center", fontsize=9)
        ax.set_xticks(x2)
        ax.set_xticklabels(labs)
        ax.axhline(0.5, color="k", ls=":", lw=1)
        ax.set_ylim(0, max(hp + hc) + 0.12)
        ax.set_title("核心证据: 持仓信号期 vs 空仓信号期")
        ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "timing_dashboard.png", dpi=130)
    plt.close(fig)

    if hold_bt is not None and len(hold_bt.get("eq", [])):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(hold_bt["eq"], color="#c44e52", lw=2.0, label="择时策略")
        ax.plot(hold_bt["bh"], color="#4c72b0", lw=1.4, label="一直满仓(买入持有)")
        ax.set_title(
            f"【最终留出集】从未参与任何选择的一段: 择时 vs 满仓 | "
            f"skill={(hold_ev or {}).get('skill')} (t检验 p={(hold_ev or {}).get('spread_p')})\n"
            f"年化: 择时 {(hold_ev or {}).get('ann_ret')} vs 满仓 {(hold_ev or {}).get('bh_ann_ret')} | "
            f"最大回撤: {(hold_ev or {}).get('max_dd')} vs {(hold_ev or {}).get('bh_max_dd')}")
        ax.set_xlabel(f"调仓次数 (每 {best['h']} 个交易日一次)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUT / "timing_holdout.png", dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    names = [f"第{b['round']}轮\n{b['kind']}/h={b['h']}" for b in board]
    ax.bar(names, [b["acc"] for b in board], color="#4c72b0")
    for i, b in enumerate(board):
        ax.text(i, b["acc"] + 0.002, f"{b['acc']:.4f}\n基线{b['base']:.3f}\n超额{b['edge']:+.4f}",
                ha="center", fontsize=7)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_ylabel("样本外命中率")
    ax.set_title("方向D 多轮迭代: 调度官依据质疑官反馈改进的轨迹")
    fig.tight_layout()
    fig.savefig(OUT / "timing_rounds.png", dpi=130)
    plt.close(fig)


def write_report(cfg, bb, board, best, ev, crit, hold, ext, bt, label, elapsed):
    L = []
    A = L.append
    d = bb.store.get("IndexDataAgent", {})
    lb = bb.store.get("LabelAgent", {})
    A("# 方向 D: 指数择时模型 训练报告")
    A("")
    A(f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M:%S} | 总耗时 {elapsed:.0f}s")
    A(f"- 标的: **{cfg.name}** ({cfg.code}) | 样本 {d.get('start')} ~ {d.get('end')}, "
      f"{d.get('n_bars')} 个交易日 (约 {d.get('n_bars', 0)/243:.1f} 年)")
    A(f"- 任务: 预测未来 {cfg.h_main} 个交易日涨跌 → 输出\"该不该持仓\"")
    A(f"- 验证: **最终留出集设计** —— 最后 {cfg.holdout_frac:.0%} 的交易日全程不参与任何"
      f"特征/模型/轮次的选择, 最后只用一次")
    A(f"- 成本假设: ETF 往返 {COST*100:.2f}%; 空仓期按年化 {CASH_RATE*100:.1f}% 计")
    A("")
    A("> 与方向 A(个股涨跌)的根本区别: 这里不挑股票, 只判断\"大盘该不该在场\"。"
      "评判标准也从\"命中率\"升级为\"命中率显著性 + 是否打得过一直满仓\"。")
    A("")
    A("---")
    A("")
    A("## 一、多轮迭代轨迹 (只在开发期上调参)")
    A("")
    A("| 轮次 | 模型 | 周期 | 特征数 | 总命中率 | 基准率 | **真实择时能力 skill** | t检验p | AUC | 折级胜率 | 可信度 | 决策依据 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for b in board:
        sk = b.get("skill")
        A(f"| 第{b['round']}轮 | {b['kind']} | h={b['h']} | {b['n_feats']} | {b['acc']:.4f} | "
          f"{b['base']:.4f} | **{'-' if sk is None else f'{sk:+.4f}'}** | {b.get('spread_p')} | "
          f"{b['auc']} | {b.get('fold_win')}/{b.get('fold_n')} | {b['crit_rank']} | {b['reason']} |")
    A("")
    A(f"**开发期最优(按真实择时能力 skill 选)**: 第 {best['round']} 轮 {best['kind']} "
      f"(h={best['h']}, {best['n_feats']} 特征), skill = {ev.get('skill')}, "
      f"t 检验 p = {ev.get('spread_p')}")
    A("")
    A("## 二、样本外详细指标(开发期)")
    A("")
    A("### ★ 核心指标: 真实择时能力 (不受指数上行漂移影响)")
    A("")
    A(f"指数长期向上, 所以闭眼满仓的命中率天然就有 **{ev['base']:.3f}**。"
      f"因此评判择时模型不能看总命中率, 而要看它能否把\"好时段\"和\"差时段\"分开:")
    A("")
    A("| 指标 | 持仓信号期 | 空仓信号期 | 差异 | 显著性 |")
    A("|---|---|---|---|---|")
    _hh, _hc = ev.get("hit_hold"), ev.get("hit_cash")
    _rh, _rc = ev.get("avg_ret_hold"), ev.get("avg_ret_cash")
    _dh = "-" if (_hh is None or _hc is None) else f"{_hh - _hc:+.4f}"
    _dr = "-" if (_rh is None or _rc is None) else f"{_rh - _rc:+.4f}"
    A(f"| 上涨概率 | {_hh} | {_hc} | **{_dh}** | — |")
    A(f"| 平均 h 日收益 | {_rh} | {_rc} | {_dr} | t 检验 p = **{ev.get('spread_p')}** |")
    A(f"| 样本期数 | {ev.get('n_hold')} | {ev.get('n_cash')} | — | — |")
    A("")
    A("若 skill 为正且 t 检验显著, 说明模型真的能识别出更好的持仓时点; "
      "若 p 很大, 那就是在碰运气。")
    A("")
    A(f"- 样本外样本数: {ev['n_oos']:,} 个交易日")
    A(f"- 命中率(逐日口径): **{ev['acc']:.4f}** vs 基准率 {ev['base']:.4f}")
    A(f"- AUC: {ev['auc']} | 概率分档单调性: {ev['monotonicity']:+.3f}")
    A(f"- **非重叠调仓期数: {ev['n_period']}** (每 {best['h']} 个交易日一次, 出手 "
      f"{ev['n_trade']} 次, 持仓占比 {ev['trade_frac']:.1%})")
    A(f"- 调仓期命中率: {ev['period_hit']} vs 基准 {ev['period_base']} | "
      f"**二项检验 p = {ev['p_value']}** | 90% 置信区间 {ev['ci90']}")
    A(f"- 择时策略总收益 {ev['total_ret']:.2%} vs 买入持有 {ev['bh_total_ret']:.2%} "
      f"(区间 {ev['years']} 年)")
    A(f"- 年化: 择时 {ev['ann_ret']:.2%} vs 买入持有 {ev['bh_ann_ret']:.2%}")
    A(f"- 最大回撤: 择时 {ev['max_dd']:.2%} vs 买入持有 {ev['bh_max_dd']:.2%}")
    A(f"- 夏普(年化, 用调仓期收益估): 择时 {ev.get('sharpe_strat')} vs 买入持有 "
      f"{ev.get('sharpe_bh')}")
    A("")
    A("### 逐折明细")
    A("")
    A("| 测试期起点 | 训练样本 | 测试样本 | 命中率 | 基准率 | AUC |")
    A("|---|---|---|---|---|---|")
    for f in best["folds"]:
        A(f"| {f['test_start']} | {f['n_train']} | {f['n_test']} | {f['acc']:.4f} | "
          f"{f['base']:.4f} | {f['auc']} |")
    A("")
    A("## 三、【最重要】最终留出集检验")
    A("")
    if hold and hold.get("acc") is not None:
        A(f"留出期: {hold['start']} ~ {hold['end']}, 共 {hold['n']} 个交易日。"
          f"这段数据在整轮迭代里从未被看过, 用于回答\"这套东西真能用吗\"。")
        A("")
        A("| 指标 | 留出集结果 | 门槛 | 是否达标 |")
        A("|---|---|---|---|")
        A(f"| 真实择时能力 skill | {hold.get('skill')} | > 0 | "
          f"{'是' if (hold.get('skill') or -9) > 0 else '否'} |")
        A(f"| skill 的 t 检验 p | {hold.get('spread_p')} | < 0.10 | "
          f"{'是' if (hold.get('spread_p') or 1) < 0.10 else '否'} |")
        A(f"| 总命中率 | {hold['acc']:.4f} | — | — |")
        A(f"| 基准率(闭眼满仓) | {hold['base']:.4f} | — | — |")
        A(f"| AUC | {hold['auc']} | > 0.53 | {'是' if (hold['auc'] or 0) > 0.53 else '否'} |")
        A(f"| 择时年化 | {hold.get('ann_ret')} | > 满仓 {hold.get('bh_ann_ret')} | "
          f"{'是' if (hold.get('ann_ret') or -9) > (hold.get('bh_ann_ret') or 0) else '否'} |")
        A(f"| 夏普 | {hold.get('sharpe_strat')} | > 满仓 {hold.get('sharpe_bh')} | "
          f"{'是' if (hold.get('sharpe_strat') or -9) > (hold.get('sharpe_bh') or 0) else '否'} |")
        A(f"| 最大回撤 | {hold.get('max_dd')} | 优于满仓 {hold.get('bh_max_dd')} | "
          f"{'是' if (hold.get('max_dd') or -9) > (hold.get('bh_max_dd') or -9) else '否'} |")
    else:
        A("留出集样本不足, 未能检验")
    A("")
    A("## 四、跨指数外部验证")
    A("")
    if ext:
        A("用同一个模型(在沪深300 开发期上训练), 不做任何调整, 直接套到其他指数上, "
          "看信号是否有普遍性:")
        A("")
        A("| 指数 | 区间 | 样本 | 命中率 | 基准率 | 超额 | AUC |")
        A("|---|---|---|---|---|---|---|")
        for k, v in ext.items():
            A(f"| {k} | {v['start']} ~ {v['end']} | {v['n']} | {v['acc']:.4f} | "
              f"{v['base']:.4f} | {v['edge']:+.4f} | {v['auc']} |")
        A("")
        A("> 注意: 各指数高度相关, 所以这不是完全独立的验证, 但能排除\"沪深300 特有噪声\" "
          "这一种解释。")
    else:
        A("无其他指数可用")
    A("")
    A("## 五、质疑官的审查结论")
    A("")
    A(f"**{crit['verdict']}**")
    A("")
    A("发现的问题:")
    A("")
    for i in crit["issues"]:
        A(f"- {i}")
    A("")
    A("必须声明的偏差:")
    A("")
    for b in crit["biases"]:
        A(f"- {b}")
    A("")
    A("## 六、基准率")
    A("")
    A("| 预测周期 | 上涨基准率 | 扣费后仍上涨 | 平均收益 | 收益标准差 |")
    A("|---|---|---|---|---|")
    for hh, v in lb.items():
        A(f"| h={hh} | {v['base_up']:.4f} | {v['base_net']:.4f} | {v['mean']*100:.3f}% | "
          f"{v['std']*100:.2f}% |")
    A("")
    A("## 七、Agent 协作消息记录")
    A("")
    A("```")
    A("\n".join(bb.handoffs))
    A("```")
    A("")
    (OUT / "timing_report.md").write_text("\n".join(L), encoding="utf-8")
    return OUT / "timing_report.md"


# ================================================================== 主流程
def holdout_eval(df, feats, h, kind, cfg, thr, index_df):
    """在最终留出集上跑一次完整评估(只用一次!)。"""
    dates = np.sort(df["date"].unique())
    dev_end = dates[int(len(dates) * (1 - cfg.holdout_frac))]
    tr = df[df["date"] <= dev_end].dropna(subset=feats + [f"y{h}"])
    te = df[df["date"] > dev_end].dropna(subset=feats + [f"y{h}"])
    if len(te) < 120:
        return None, None
    m = ModelAgent(None, cfg).make(kind)
    m.fit(tr[feats], tr[f"y{h}"])
    te = te.copy()
    te["p"] = m.predict_proba(te[feats])[:, 1]
    wf = {"folds": [], "pooled": te, "kind": kind, "h": h, "tag": "HOLDOUT",
          "label": "abs", "ycol": f"y{h}"}
    ev, bt = TimingEvalAgent(Blackboard(), cfg).run(wf, index_df, thr)
    if ev:
        ev["start"] = str(te["date"].min().date())
        ev["end"] = str(te["date"].max().date())
        ev["n"] = int(len(te))
        ev["deciles"] = []
    return ev, bt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--h", type=int, default=10)
    ap.add_argument("--thr", type=float, default=0.55)
    ap.add_argument("--code", type=str, default="sh.000300")
    a = ap.parse_args()
    cfg = Cfg(h_main=a.h, rounds=a.rounds, thr=a.thr, code=a.code,
              name=qd.BS_INDICES.get(a.code, a.code))
    t0 = time.time()
    print("=" * 86)
    print(f"方向 D 指数择时系统 v1.0   —— {cfg.name} | 预测未来 {cfg.h_main} 日 | "
          f"迭代 {cfg.rounds} 轮")
    print(f"最终留出集 = 最后 {cfg.holdout_frac:.0%} 的交易日(全程不参与任何选择)")
    print("=" * 86)

    bb = Blackboard()
    main_df, others = IndexDataAgent(bb, cfg).run()
    feats_df = FeatureAgent(bb, cfg).run(main_df)
    df = LabelAgent(bb, cfg).run(feats_df)

    dates = np.sort(df["date"].unique())
    dev_end = dates[int(len(dates) * (1 - cfg.holdout_frac))]
    print(f"[调度官] 开发期 {dates[0].astype('datetime64[D]')} ~ "
          f"{dev_end.astype('datetime64[D]')} ({int(len(dates)*(1-cfg.holdout_frac))} 天); "
          f"留出期从此之后 ({len(dates)-int(len(dates)*(1-cfg.holdout_frac))} 天)")
    mem = MemoryAgent(bb, cfg)
    hist0 = mem.read()
    print(f"[档案官] 历史记忆 {len(hist0)} 条")

    orch = Orchestrator(bb, cfg)
    board, history, importance = [], [], None
    best, best_ev, best_bt = None, None, None
    for r in range(1, cfg.rounds + 1):
        print("-" * 86)
        plan = orch.plan(r, history, importance)
        print(f"[调度官] 第 {r} 轮: {plan['reason']}")
        wf = ModelAgent(bb, cfg).run(df, plan["feats"], plan["h"], plan["kind"],
                                     label=plan["label"], end_date=dev_end)
        if wf["pooled"].empty:
            print("[时机评估官] 该配置无有效样本外结果, 跳过")
            continue
        ev, bt = TimingEvalAgent(bb, cfg).run(wf, main_df, plan["thr"])
        crit = CriticAgent(bb, cfg).run(wf, ev, df, plan["feats"])
        try:
            tr = df[df["date"] <= dev_end].dropna(subset=plan["feats"] + [f"y{plan['h']}"])
            m = ModelAgent(bb, cfg).make(plan["kind"])
            m.fit(tr[plan["feats"]], tr[f"y{plan['h']}"])
            importance = feature_importance_of(
                m, tr[plan["feats"]], tr[f"y{plan['h']}"], plan["feats"],
                plan["kind"], n=4000, seed=cfg.seed)
            print("[模型官] 特征重要性 Top5: "
                  + ", ".join(f"{f}({v:.4f})" for f, v in importance[:5]))
        except Exception as e:                       # noqa: BLE001
            print(f"[模型官] 重要性跳过: {e}")
        board.append({"round": r, "kind": plan["kind"], "h": plan["h"],
                      "n_feats": len(plan["feats"]), "acc": ev["acc"], "base": ev["base"],
                      "edge": ev["edge"], "auc": ev["auc"], "crit_rank": crit["rank"],
                      "fold_win": crit["fold_win"], "fold_n": crit["fold_n"],
                      "thr": plan["thr"], "skill": ev.get("skill"),
                      "spread_p": ev.get("spread_p"), "reason": plan["reason"]})
        history.append({**board[-1], "crit_rank": crit["rank"], "feats": plan["feats"]})
        mem.write({"date": str(datetime.now())[:19], "round": r, "phase": "D-指数择时",
                   "kind": plan["kind"], "h": plan["h"], "acc": ev["acc"],
                   "base": ev["base"], "edge": ev["edge"], "p_value": ev.get("p_value"),
                   "crit_rank": crit["rank"]})
        if best is None or (ev.get("skill") or -9) > (best_ev.get("skill") or -9):
            best, best_ev, best_bt = {"round": r, "kind": plan["kind"], "h": plan["h"],
                                      "label": plan["label"], "n_feats": len(plan["feats"]),
                                      "feats": plan["feats"], "thr": plan["thr"],
                                      "folds": wf["folds"], "pooled": wf["pooled"],
                                      "crit": crit}, ev, bt

    if best is None:
        print("没有任何一轮产出有效结果")
        return 1
    best_crit = best.get("crit", {"verdict": "见排行榜", "rank": "-",
                                   "issues": [], "biases": []})

    print("=" * 86)
    print("[调度官] 开发期结束, 现在做最终留出集检验(只跑一次)")
    ext = ExternalValidator(bb, cfg).run(df, best["feats"], best["h"], best["kind"],
                                         others, cfg)
    hold, hold_bt = holdout_eval(df, best["feats"], best["h"], best["kind"], cfg,
                                 best["thr"], main_df)
    if hold is None:
        hold, hold_bt = {}, None
    mem.save_board(board)
    make_charts(best, best_ev, best_bt, importance, board, cfg.name,
                hold_ev=hold, hold_bt=hold_bt)
    elapsed = time.time() - t0
    rpt = write_report(cfg, bb, board, best, best_ev, crit=best_crit, hold=hold,
                       ext=ext, bt=best_bt, label=cfg.name, elapsed=elapsed)
    bb.save()
    mem.write({"date": str(datetime.now())[:19], "phase": "D-指数择时", "round": "holdout",
               "kind": best["kind"], "h": best["h"], "acc": hold.get("acc"),
               "base": hold.get("base"), "edge": hold.get("edge"),
               "p_value": hold.get("p_value"), "is_holdout": True})
    print("=" * 86)
    print(f"完成, 耗时 {elapsed:.0f}s")
    print(f"开发期最优: 第{best['round']}轮 {best['kind']} h={best['h']} → 真实择时能力 "
          f"skill={best_ev.get('skill')} (t检验 p={best_ev.get('spread_p')}), "
          f"总命中率 {best_ev['acc']:.4f} vs 基准 {best_ev['base']:.4f}")
    if hold and hold.get("acc") is not None:
        print(f"最终留出集({hold.get('start')}~{hold.get('end')}): skill={hold.get('skill')} "
              f"(p={hold.get('spread_p')}) | 总命中率 {hold.get('acc')} vs 基准 "
              f"{hold.get('base')} | 择时年化 {hold.get('ann_ret')} vs 满仓 "
              f"{hold.get('bh_ann_ret')}")
    print(f"报告: {rpt}")
    print("=" * 86)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
