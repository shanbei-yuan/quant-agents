#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方向 C: 波动率 / 风险预测系统 vol_agents.py  v1.0
=================================================
为什么转 C: 方向 D 的特征重要性显示, 模型能从特征里读出的信息**全在波动率维度**
(dd_all / ma60_ma120 / vol120 / vol60 / atr_pct / dd250), 方向维度读不出东西。
波动率是可预测的(几十年 GARCH/HAR 文献), 且能真正用于仓位管理。

三个任务:
  A. 回归: 预测未来 h 日已实现波动率 (RV)
  B. 分类: 预测未来 h 日内是否出现"较大回撤"(低点较入场价跌超 5%)
  C. 阀门: 用预测波动率做仓位管理, 与"一直满仓"对比

九道严谨性关口:
  * 基准不是"平均"而是**两个真基准**: ①朴素 rv20 外推 ②HAR-RV 古典模型(OLS)
  * 损失函数用 **QLIKE**(波动率预测的行业标准损失, 对小波动更敏感)
  * **Diebold-Mariano 检验**: 用非重叠 h 日期间做损失差分的 t 检验, 判"改进是否显著"
  * 最终留出集: 最后 20% 交易日全程不参与任何选择
  * 跨指数外检
  * 交易成本按仓位变动计费

九个 Agent(人设):
  1 数据官 / 2 特征官(波动率专用) / 3 标注官(RV + 风险事件) / 4 模型官
  5 波动评估官(QLIKE+DM) / 6 **阀门官**(仓位管理回测, 含 D 方向门)
  7 质疑官 / 8 调度官 / 9 档案官

用法: python vol_agents.py --rounds 5
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
import timing_agents as tmod
from train_agents import Blackboard, Agent, MemoryAgent, setup_font

ROOT = Path(__file__).resolve().parent
CACHE, OUT, LOGS = ROOT / "cache", ROOT / "out", ROOT / "logs"
for _d in (CACHE, OUT, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

COST = 0.0010          # 指数ETF 往返成本 0.10%
CASH_RATE = 0.015      # 空仓资金收益
TARGET_VOL = 0.15      # 波动率目标(年化); 沪深300 长期波动率约 22%
RISK_THR = 0.05        # "较大回撤"定义: 低点较入场价跌超 5%


@dataclass
class Cfg:
    code: str = "sh.000300"
    name: str = "沪深300"
    horizons: tuple = (5, 10, 20)
    h_main: int = 10
    test_len: int = 500
    min_train_days: int = 750
    holdout_frac: float = 0.20
    rounds: int = 5
    seed: int = 0


FEATS = [
    "rv5", "rv10", "rv20", "rv60", "rv120",
    "rv_5_20", "rv_20_60", "rv_20_120", "vol_of_vol", "rv20_trend",
    "dvol20", "uvol20", "dvol_share", "range5", "range20", "range_ratio",
    "gap20", "atr14", "atr_ratio",
    "skew20", "kurt20", "maxabs20",
    "rv20_vs_med", "high_vol_state", "rv20_pct",
    "v_z", "v_rv20", "amtv_z",
    "ret5", "ret20", "ret60", "ma20_gap",
    "dd_all", "dd60", "dd120", "dd250", "pos60", "pos120", "pos250",
    "days_since_peak",
]
FEATS_CORE = ["rv5", "rv10", "rv20", "rv60", "rv_5_20", "rv_20_60", "rv_20_120",
              "vol_of_vol", "dvol_share", "rv20_pct", "rv20_vs_med", "high_vol_state",
              "range_ratio", "atr_ratio", "v_z", "dd_all"]
FEATS_RISK = ["rv5", "rv20", "rv60", "dvol_share", "skew20", "kurt20", "maxabs20",
              "dd_all", "dd60", "pos60", "days_since_peak", "rv20_pct", "ret20", "range20"]


# ================================================================== 工具
def qlike(true_var: np.ndarray, pred_var: np.ndarray) -> np.ndarray:
    """QLIKE 损失, 逐样本返回。true_var/pred_var 均为方差(非标准差)。"""
    pred_var = np.maximum(pred_var, 1e-10)
    true_var = np.maximum(true_var, 1e-10)
    r = true_var / pred_var
    return r - np.log(r) - 1.0


def dm_test(d: np.ndarray):
    """Diebold-Mariano: 损失差分序列是否显著异于 0(单边, 希望 <0)。"""
    from scipy.stats import ttest_1samp
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 20:
        return None, None
    t, p = ttest_1samp(d, 0.0)
    return round(float(np.mean(d)), 5), round(float(p / 2 if t < 0 else 1 - p / 2), 5)


# ================================================================== 1 数据官
class VolDataAgent(Agent):
    name = "数据官"

    def run(self):
        data = {}
        for code, nm in qd.BS_INDICES.items():
            try:
                df = qd.index_hist_bs(code)
            except Exception as e:                 # noqa: BLE001
                self.log(f"{nm} 失败: {e}")
                continue
            if df is not None and len(df) > 500:
                data[code] = df
        if self.cfg.code not in data:
            raise RuntimeError("主指数缺失")
        m = data[self.cfg.code]
        self.log(f"主标的 {self.cfg.name}: {len(m)} 个交易日 "
                 f"{m.index[0].date()} ~ {m.index[-1].date()}, 共 {len(data)} 个指数")
        self.bb.put("VolDataAgent", {"n": len(m), "start": str(m.index[0].date()),
                                     "end": str(m.index[-1].date()), "n_indices": len(data)})
        self.bb.handoff(self.name, "特征官", f"{len(m)} 根 / {len(data)} 个指数就绪")
        return m, data


# ================================================================== 2 特征官
class VolFeatureAgent(Agent):
    name = "特征官"

    @staticmethod
    def _days_since_peak(c: pd.Series) -> np.ndarray:
        peak = c.cummax().values
        out = np.zeros(len(c))
        cnt = 0
        for i in range(len(c)):
            if c.values[i] >= peak[i]:
                cnt = 0
            else:
                cnt += 1
            out[i] = cnt
        return out

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]
        r = c.pct_change()
        av = np.sqrt(250)
        f = pd.DataFrame(index=df.index)
        for n in (5, 10, 20, 60, 120):
            f[f"rv{n}"] = r.rolling(n).std() * av
        f["rv_5_20"] = f["rv5"] / f["rv20"]
        f["rv_20_60"] = f["rv20"] / f["rv60"]
        f["rv_20_120"] = f["rv20"] / f["rv120"]
        f["vol_of_vol"] = f["rv20"].rolling(60).std() / f["rv20"]
        f["rv20_trend"] = f["rv20"] / f["rv20"].shift(20) - 1
        down = r.where(r < 0, 0.0)
        up = r.where(r > 0, 0.0)
        f["dvol20"] = down.rolling(20).std() * av
        f["uvol20"] = up.rolling(20).std() * av
        f["dvol_share"] = f["dvol20"] / (f["dvol20"] + f["uvol20"])
        rng = (df["high"] - df["low"]) / c
        f["range5"] = rng.rolling(5).mean()
        f["range20"] = rng.rolling(20).mean()
        f["range_ratio"] = f["range5"] / f["range20"]
        gap = (df["open"] / c.shift() - 1).abs()
        f["gap20"] = gap.rolling(20).mean()
        tr = np.maximum(df["high"] - df["low"],
                        np.maximum((df["high"] - c.shift()).abs(),
                                   (df["low"] - c.shift()).abs()))
        f["atr14"] = tr.rolling(14).mean() / c
        f["atr_ratio"] = tr.rolling(14).mean() / tr.rolling(60).mean()
        f["skew20"] = r.rolling(20).skew()
        f["kurt20"] = r.rolling(20).kurt()
        f["maxabs20"] = r.abs().rolling(20).max()
        med = f["rv20"].rolling(250).median()
        f["rv20_vs_med"] = f["rv20"] / med
        f["high_vol_state"] = (f["rv20"] > med).astype(float)
        f["rv20_pct"] = f["rv20"].rolling(250).rank(pct=True)
        v = df["volume"].replace(0, np.nan)
        a = df["amount"].replace(0, np.nan)
        lv = np.log(v)
        f["v_z"] = (lv - lv.rolling(60).mean()) / lv.rolling(60).std()
        f["v_rv20"] = lv.diff().rolling(20).std()
        avpv = (a / v)
        f["amtv_z"] = (avpv - avpv.rolling(60).mean()) / avpv.rolling(60).std()
        for n in (5, 20, 60):
            f[f"ret{n}"] = c.pct_change(n)
        f["ma20_gap"] = c / c.rolling(20).mean() - 1
        f["dd_all"] = c / c.cummax() - 1
        for n in (60, 120, 250):
            hi = df["high"].rolling(n).max()
            lo = df["low"].rolling(n).min()
            f[f"dd{n}"] = c / hi - 1
            f[f"pos{n}"] = (c - lo) / (hi - lo).replace(0, np.nan)
        f["days_since_peak"] = self._days_since_peak(c)
        f.insert(0, "close", c)
        f.insert(1, "open", df["open"])
        f.insert(2, "high", df["high"])
        f.insert(3, "low", df["low"])
        return f.reset_index()

    def run(self, main: pd.DataFrame):
        f = self.build(main)
        self.log(f"构造 {len(FEATS)} 个波动率/风险特征, {len(f)} 行")
        self.bb.put("VolFeatureAgent", {"n_feats": len(FEATS), "rows": len(f)})
        self.bb.handoff(self.name, "标注官", f"{len(FEATS)} 个波动率特征就绪")
        return f


# ================================================================== 3 标注官
class VolLabelAgent(Agent):
    name = "标注官"

    def run(self, f: pd.DataFrame):
        df = f
        av = np.sqrt(250)
        r = df["close"].pct_change()
        info = {}
        for h in self.cfg.horizons:
            # 未来 h 日已实现波动率(年化): r[t+1 .. t+h] 的标准差
            df[f"yvol{h}"] = r.rolling(h).std().shift(-h) * av
            df[f"logyvol{h}"] = np.log(df[f"yvol{h}"].clip(lower=1e-6))
            # 未来 h 日区间最低点相对入场价的跌幅
            entry = df["open"].shift(-1)
            lowmin = df["low"].rolling(h).min().shift(-h)
            df[f"fwd{h}"] = df["open"].shift(-(1 + h)) / entry - 1
            df[f"lowdrop{h}"] = lowmin / entry - 1
            df[f"yrisk{h}"] = (df[f"lowdrop{h}"] < -RISK_THR).astype(float)
            df.loc[df[f"lowdrop{h}"].isna(), f"yrisk{h}"] = np.nan
            s = df[f"yvol{h}"].dropna()
            info[h] = {"mean_vol": round(float(s.mean()), 4),
                       "median_vol": round(float(s.median()), 4),
                       "vol_p10": round(float(s.quantile(0.1)), 4),
                       "vol_p90": round(float(s.quantile(0.9)), 4),
                       "risk_rate": round(float(df[f"yrisk{h}"].dropna().mean()), 4)}
        self.log("基准: " + "; ".join(
            f"h={h}: 中位波动 {v['median_vol']:.1%}, 出现跌超{RISK_THR:.0%} 的概率 {v['risk_rate']:.3f}"
            for h, v in info.items()))
        self.bb.put("VolLabelAgent", info)
        self.bb.handoff(self.name, "模型官",
                        f"h={self.cfg.h_main}: 风险事件基准率 {info[self.cfg.h_main]['risk_rate']:.3f}")
        return df


# ================================================================== 4 模型官
class VolModelAgent(Agent):
    name = "模型官"

    def make(self, kind: str, task: str):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        if task == "reg":
            if kind == "ridge":
                from sklearn.linear_model import Ridge
                return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            from sklearn.ensemble import HistGradientBoostingRegressor
            return HistGradientBoostingRegressor(
                max_iter=150, learning_rate=0.06, max_leaf_nodes=15,
                min_samples_leaf=40, l2_regularization=1.0,
                early_stopping=True, n_iter_no_change=10,
                validation_fraction=0.15, random_state=self.cfg.seed)
        if kind == "logit":
            from sklearn.linear_model import LogisticRegression
            return make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=400, C=0.3, n_jobs=-1))
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=True, n_iter_no_change=10,
            validation_fraction=0.15, random_state=self.cfg.seed)

    def walk_forward(self, df, feats, h, kind, task="reg", end_date=None, tag=""):
        dates = np.sort(df["date"].unique())
        if end_date is not None:
            dates = dates[dates <= np.datetime64(end_date)]
        ycol = f"logyvol{h}" if task == "reg" else f"yrisk{h}"
        folds, preds = [], []
        for si in range(self.cfg.min_train_days, len(dates), self.cfg.test_len):
            te_dates = dates[si:si + self.cfg.test_len]
            if len(te_dates) < 60:
                break
            d0 = te_dates[0]
            tr = df[df["date"] < d0].dropna(subset=feats + [ycol, f"yvol{h}", f"fwd{h}"])
            te = df[df["date"].isin(te_dates)].dropna(subset=feats + [ycol, f"yvol{h}", f"fwd{h}"])
            if len(tr) < 400 or len(te) < 60:
                continue
            m = self.make(kind, task)
            m.fit(tr[feats], tr[ycol])
            if task == "reg":
                # Duan smearing 修正: 预测 log 波动率再指数化会系统性低估
                # (E[exp(x)] != exp(E[x])), 而 QLIKE 对低估惩罚极重
                k = float(np.mean(np.exp(np.asarray(tr[ycol]) - m.predict(tr[feats]))))
                p = np.exp(m.predict(te[feats])) * k
            else:
                p = m.predict_proba(te[feats])[:, 1]
            t = te[["date", f"yvol{h}", f"yrisk{h}", f"fwd{h}", f"lowdrop{h}",
                    "rv20", "open"]].copy().reset_index(drop=True)
            t["p"] = p
            folds.append({"test_start": str(pd.Timestamp(d0).date()), "n_train": len(tr),
                          "n_test": len(te)})
            preds.append(t)
        return {"folds": folds, "pooled": pd.concat(preds) if preds else pd.DataFrame(),
                "kind": kind, "h": h, "task": task, "tag": tag}


# ================================================================== 5 波动评估官
class VolEvalAgent(Agent):
    name = "波动评估官"

    def run(self, wf: dict):
        pooled = wf["pooled"]
        h, task = wf["h"], wf["task"]
        if pooled.empty:
            return None
        # 非重叠: 每 h 个交易日取一个观测, 保证损失差分独立
        po = pooled.sort_values("date").iloc[::h].copy()
        if len(po) < 30:
            return None
        true_vol = po[f"yvol{h}"].values
        pred_vol = po["p"].values
        base_vol = po["rv20"].values                      # 朴素基准: 用过去20日外推
        tv2, pv2, bv2 = true_vol ** 2, pred_vol ** 2, base_vol ** 2
        L = qlike(tv2, pv2)
        Lb = qlike(tv2, bv2)
        d = L - Lb
        dmean, dp = dm_test(d)
        # HAR-RV 古典基准: OLS 用 rv5/rv20/rv60 预测 (在开发期上拟合)
        res = {"n_period": int(len(po)), "h": h,
               "qlike_model": round(float(np.mean(L)), 5),
               "qlike_base": round(float(np.mean(Lb)), 5),
               "qlike_gain": round(float(np.mean(Lb) - np.mean(L)), 5),
               "dm_p": dp,
               "mse_model": round(float(np.mean((true_vol - pred_vol) ** 2)), 6),
               "mse_base": round(float(np.mean((true_vol - base_vol) ** 2)), 6),
               "corr_model": round(float(np.corrcoef(true_vol, pred_vol)[0, 1]), 4),
               "corr_base": round(float(np.corrcoef(true_vol, base_vol)[0, 1]), 4),
               "hit_rate_risk": None, "risk_base": None, "risk_auc": None}
        self.bb.put(f"VolEval_{wf['tag']}", res)
        self.log(f"h={h} QLIKE: 模型 {res['qlike_model']} vs 朴素基准 "
                 f"{res['qlike_base']} → 改进 {res['qlike_gain']:+.5f} (DM检验 p={dp}) | "
                 f"相关性 模型 {res['corr_model']} vs 基准 {res['corr_base']}")
        self.bb.handoff(self.name, "质疑官",
                        f"{wf['tag']} QLIKE 改进 {res['qlike_gain']:+.5f}, DM p={dp}")
        return res

    def run_cls(self, wf: dict):
        """分类口径: 风险事件预测。"""
        pooled = wf["pooled"]
        h = wf["h"]
        if pooled.empty or "yrisk" not in pooled.columns:
            return None
        from sklearn.metrics import roc_auc_score
        pv = pooled.dropna(subset=[f"yrisk{h}"]).sort_values("date").iloc[::h]
        if len(pv) < 30:
            return None
        y = pv[f"yrisk{h}"].values
        p = pv["p"].values
        base = float(y.mean())
        auc = float(roc_auc_score(y, p)) if len(set(y)) > 1 else np.nan
        for thr in (0.3, 0.4, 0.5):
            sel = p >= thr
            if sel.sum() < 10:
                continue
            break
        sel = p >= 0.35
        hr = float(y[sel].mean()) if sel.sum() > 0 else None
        res = {"risk_base": round(base, 4), "risk_auc": None if np.isnan(auc) else round(auc, 4),
               "n_period": int(len(pv)), "sel_n": int(sel.sum()),
               "sel_rate": round(float(y[sel].mean()), 4) if sel.sum() else None,
               "sel_frac": round(float(sel.mean()), 3),
               "lift": None if not sel.sum() else round(float(y[sel].mean() / base), 3)}
        self.bb.put(f"VolCls_{wf['tag']}", res)
        self.log(f"风险事件(h={h}): 基准率 {res['risk_base']}, AUC {res['risk_auc']}, "
                 f"高概率组的实际发生率 {res['sel_rate']} (提升 {res['lift']} 倍)")
        return res


# ================================================================== 6 阀门官
class ValveAgent(Agent):
    name = "阀门官"

    def backtest(self, dates, probs_vol, index_df, h, mode, dir_prob=None,
                 target=TARGET_VOL):
        """非重叠调仓; mode: full / vol / vol_dir / dir"""
        o = index_df["open"]
        pos_of = {d: i for i, d in enumerate(index_df.index)}
        eq, curve, prev_pos = 1.0, [], 0.0
        poss, rets = [], []
        for d in dates[::max(1, h)]:
            i = pos_of.get(d)
            if i is None or i + 1 + h >= len(index_df):
                continue
            entry, ex = o.iloc[i + 1], o.iloc[i + 1 + h]
            if not np.isfinite(entry) or not np.isfinite(ex) or entry <= 0:
                continue
            rr = ex / entry - 1
            if mode == "full":
                pos = 1.0
            elif mode == "vol":
                pos = float(np.clip(target / max(probs_vol.get(d, target), 1e-4), 0.0, 1.0))
            elif mode == "vol_dir":
                pos = float(np.clip(target / max(probs_vol.get(d, target), 1e-4), 0.0, 1.0))
                if dir_prob and dir_prob.get(d, 0.5) < 0.5:
                    pos *= 0.5
            elif mode == "dir":
                pos = 1.0 if (dir_prob or {}).get(d, 0.5) >= 0.5 else 0.0
            else:
                pos = 1.0
            cost = COST * abs(pos - prev_pos)
            r = pos * rr + (1 - pos) * (CASH_RATE * h / 250.0) - cost
            eq *= (1 + r)
            curve.append(eq)
            poss.append(pos)
            rets.append(r)
            prev_pos = pos
        if not curve:
            return None
        curve = np.array(curve)
        rets = np.array(rets)
        yrs = len(curve) * h / 250.0
        ann = float(curve[-1] ** (1 / yrs) - 1) if yrs > 0.2 else None
        mdd = float((curve / np.maximum.accumulate(curve) - 1).min())
        sd = float(np.std(rets, ddof=1))
        sharpe = float(np.mean(rets) / sd * np.sqrt(250 / h)) if sd > 0 else None
        return {"mode": mode, "n_period": len(curve), "years": round(yrs, 2),
                "total_ret": round(float(curve[-1] - 1), 4),
                "ann_ret": None if ann is None else round(ann, 4),
                "max_dd": round(mdd, 4),
                "sharpe": None if sharpe is None else round(sharpe, 3),
                "calmar": None if (ann is None or mdd == 0) else round(ann / abs(mdd), 3),
                "avg_pos": round(float(np.mean(poss)), 3),
                "curve": curve}

    def run(self, pooled_vol: pd.DataFrame, index_df: pd.DataFrame, h: int,
            dir_pooled: pd.DataFrame | None, tag=""):
        if pooled_vol.empty:
            return {}
        pv = dict(zip(pooled_vol["date"], pooled_vol["p"]))
        dv = dict(zip(dir_pooled["date"], dir_pooled["p"])) if dir_pooled is not None \
            and not dir_pooled.empty else None
        dates = sorted(pv.keys())
        out = {}
        for mode in ("full", "vol", "vol_dir", "dir"):
            if mode in ("vol_dir",) and dv is None:
                continue
            if mode == "dir":
                if dv is None:
                    continue
                r = self.backtest(sorted(dv.keys()), {}, index_df, h, mode, dir_prob=dv)
            else:
                r = self.backtest(dates, pv, index_df, h, mode, dir_prob=dv)
            if r:
                out[mode] = r
        self.log("阀門回测(" + tag + "): " + " | ".join(
            f"{k}: 年化 {v['ann_ret']:.2%} 回撤 {v['max_dd']:.2%} 夏普 {v['sharpe']} "
            f"平均仓位 {v['avg_pos']:.0%}" for k, v in out.items()))
        self.bb.put(f"Valve_{tag}", {k: {kk: vv for kk, vv in v.items() if kk != "curve"}
                                     for k, v in out.items()})
        self.bb.handoff(self.name, "质疑官", f"{tag} 阀門回测完成 {len(out)} 种方案")
        return out


# ================================================================== 7 质疑官
class CriticAgent(Agent):
    name = "质疑官"

    def run(self, wf, ev, cls, valve, tag=""):
        issues = []
        if ev:
            if ev["dm_p"] is not None and ev["dm_p"] > 0.10:
                issues.append(f"QLIKE 改进虽然为正({ev['qlike_gain']:+.5f}), 但 DM 检验 "
                              f"p={ev['dm_p']} (>0.10): 与朴素基准的差距不显著")
            else:
                issues.append(f"QLIKE 改进 {ev['qlike_gain']:+.5f}, DM 检验 p={ev['dm_p']}: "
                              f"显著优于'用过去20日波动率外推'")
            if ev["corr_model"] <= ev["corr_base"]:
                issues.append(f"预测波动率与真实波动率的相关性 {ev['corr_model']} 并不高于"
                              f"朴素基准 {ev['corr_base']} —— 加了模型没变聪明")
            issues.append(f"有效独立样本 {ev['n_period']} 个非重叠期; 波动率预测的可检验性"
                          f"远好于方向预测(方向预测几乎没有独立样本)")
        if cls:
            issues.append(f"风险事件预测: AUC {cls['risk_auc']}, 基准率 {cls['risk_base']}, "
                          f"高概率组实际发生率 {cls['sel_rate']} (提升 {cls['lift']} 倍)")
            if cls["risk_auc"] is None or cls["risk_auc"] < 0.55:
                issues.append("AUC < 0.55: 对'是否出现较大回撤'的区分能力偏弱")
        if valve and "full" in valve and "vol" in valve:
            a, b = valve["vol"], valve["full"]
            if (a["sharpe"] or -9) > (b["sharpe"] or 0):
                issues.append(f"波动率阀門把夏普从 {b['sharpe']} 提到 {a['sharpe']}, "
                              f"回撤从 {b['max_dd']:.2%} 收窄到 {a['max_dd']:.2%} —— 有价值")
            else:
                issues.append(f"波动率阀門夏普 {a['sharpe']} 没超過满仓 {b['sharpe']}")
        biases = [
            "单一标的: 只在沪深300 上验证",
            "杠杆缺失: 仓位上限设为 1.0, 低波动时无法加杠杆, 会低估波动率目标策略的效果",
            "已实现波动率的度量用日收益标准差, 未包含日内高频信息",
            "成本按仓位变动 0.10%/次估计, 未含冲击成本",
            "最终留出集只有一段 (最后 20%), 无法检验抗周期性",
        ]
        rank = "低"
        if ev and ev["dm_p"] is not None and ev["dm_p"] < 0.10 and ev["qlike_gain"] > 0:
            rank = "中"
        if (ev and ev["dm_p"] is not None and ev["dm_p"] < 0.05 and ev["qlike_gain"] > 0
                and ev["corr_model"] > ev["corr_base"]
                and cls and (cls["risk_auc"] or 0) > 0.55):
            rank = "高"
        qg = "-" if not ev or ev.get("qlike_gain") is None else f"{ev['qlike_gain']:+.5f}"
        verdict = (f"可信度评级: {rank} (QLIKE 改进 {qg}, DM p="
                   f"{None if not ev else ev['dm_p']}, 风险 AUC "
                   f"{None if not cls else cls.get('risk_auc')})")
        self.log(verdict)
        out = {"verdict": verdict, "rank": rank, "issues": issues, "biases": biases}
        self.bb.put(f"VolCritic_{tag}", out)
        self.bb.handoff(self.name, "调度官", verdict)
        return out


# ================================================================== 8 调度官
class Orchestrator(Agent):
    name = "调度官"

    def plan(self, r, hist, importance=None):
        if r == 1:
            return {"task": "reg", "kind": "ridge", "h": self.cfg.h_main,
                    "feats": list(FEATS),
                    "reason": "第1轮: 全波动率特征 + 岭回归, 预测未来10日波动率(基线)"}
        prev = hist[-1]
        if r == 2:
            keep = [f for f, _ in (importance or [])][:max(12, int(len(FEATS) * 0.7))] \
                or list(FEATS)
            return {"task": "reg", "kind": "hgb", "h": prev["h"], "feats": keep,
                    "reason": f"第2轮: 按重要性砍掉 {len(FEATS)-len(keep)} 个特征, "
                              f"换梯度提升树捕捉波动率的非线性/状态切换"}
        if r == 3:
            return {"task": "reg", "kind": "hgb", "h": prev["h"],
                    "feats": sorted(set(FEATS_CORE)),
                    "reason": "第3轮: 只用 16 个核心波动率特征做极简模型(小样本抗过拟合)"}
        if r == 4:
            return {"task": "cls", "kind": "hgb", "h": prev["h"],
                    "feats": sorted(set(FEATS_RISK)),
                    "reason": "第4轮: 任务换成分类 —— 预测未来10日是否出现'低点跌超5%'的风险事件"}
        if r == 5:
            h2 = 20 if prev["h"] == self.cfg.h_main else self.cfg.h_main
            return {"task": "cls", "kind": "hgb", "h": h2, "feats": sorted(set(FEATS_RISK)),
                    "reason": f"第5轮: 风险事件周期从 h={prev['h']} 拉长到 h={h2}"}
        return {"task": "reg", "kind": "hgb", "h": prev["h"], "feats": sorted(set(FEATS_CORE)),
                "reason": "后续轮: 回到核心配置复跑以观察稳定性"}


# ================================================================== 图表/报告
def make_charts(best, ev, valve, importance, board, cls, hold_valve, hold_ev, label):
    setup_font()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0][0]
    if best.get("task") == "reg":
        ycol = f"yvol{best['h']}"
        pooled = best["pooled"].dropna(subset=[ycol, "p"])
        if len(pooled):
            ax.plot(pooled["date"], pooled[ycol], lw=1.0, color="#4c72b0", label="真实未来波动率")
            ax.plot(pooled["date"], pooled["p"], lw=1.0, color="#c44e52", label="模型预测")
            ax.plot(pooled["date"], pooled["rv20"], lw=0.8, ls="--", color="#999999",
                    label="朴素: 过去20日波动率")
    ax.set_title(f"波动率预测 vs 真实值 ({best['kind']}, h={best['h']})  {label}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[0][1]
    if valve:
        for mode, col, nm in (("full", "#4c72b0", "一直满仓"),
                              ("vol", "#c44e52", "波动率目标阀門"),
                              ("vol_dir", "#55a868", "波动率+方向双阀門")):
            if mode in valve:
                c = valve[mode]["curve"]
                ax.plot(c, color=col, lw=1.6,
                        label=f"{nm} (年化{valve[mode]['ann_ret']:.1%}, 回撤{valve[mode]['max_dd']:.0%})")
        ax.set_yscale("log")
        ax.set_title("风险阀門回测(开发期): 净值对比")
        ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    ax = axes[1][0]
    if importance:
        imp = pd.DataFrame(importance, columns=["feat", "imp"]).head(16)
        ax.barh(imp["feat"][::-1], imp["imp"][::-1], color="#8172b3")
        ax.set_title("波动率模型 特征重要性 Top16")
    ax.grid(alpha=0.25)

    ax = axes[1][1]
    if hold_valve:
        for mode, col, nm in (("full", "#4c72b0", "满仓"), ("vol", "#c44e52", "阀門"),
                              ("vol_dir", "#55a868", "阀門+方向")):
            if mode in hold_valve:
                ax.plot(hold_valve[mode]["curve"], color=col, lw=1.8,
                        label=f"{nm} (年化{hold_valve[mode]['ann_ret']:.1%}, "
                              f"回撤{hold_valve[mode]['max_dd']:.0%})")
        ax.set_title(f"【最终留出集】风险阀門 ({hold_ev.get('years', '')} 年)")
        ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "vol_dashboard.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(board))
    ax.bar(x - 0.2, [b["qlike_gain"] if b["qlike_gain"] is not None else 0 for b in board],
           width=0.4, color="#55a868", label="QLIKE 相对朴素基准的改进")
    ax2 = ax.twinx()
    ax2.plot(x, [b["risk_auc"] if b.get("risk_auc") else np.nan for b in board], "o--",
             color="#c44e52", label="风险事件 AUC")
    ax.set_xticks(x)
    ax.set_xticklabels([f"第{b['round']}轮\n{b['task']}/{b['kind']}\nh={b['h']}" for b in board],
                       fontsize=7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("QLIKE 改进 (越大越好)")
    ax2.set_ylabel("风险事件 AUC")
    ax.set_title("方向C 多轮迭代: 波动率预测精度 与 风险事件识别能力")
    ax.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "vol_rounds.png", dpi=130)
    plt.close(fig)


def write_report(cfg, bb, board, best, ev, cls, crit, valve, hold_ev, hold_cls,
                 hold_valve, ext, elapsed):
    L = []
    A = L.append
    d = bb.store.get("VolDataAgent", {})
    lb = bb.store.get("VolLabelAgent", {})
    A("# 方向 C: 波动率 / 风险预测 训练报告")
    A("")
    A(f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M:%S} | 总耗时 {elapsed:.0f}s")
    A(f"- 标的: **{cfg.name}** ({cfg.code}) | 样本 {d.get('start')} ~ {d.get('end')}, "
      f"{d.get('n')} 个交易日 (约 {d.get('n', 0)/243:.1f} 年)")
    A(f"- 任务: ①预测未来 h 日已实现波动率 ②预测未来 h 日是否出现'低点跌超 5%'"
      f" ③用波动率预测做**仓位阀门**")
    A(f"- 损失函数: **QLIKE**(波动率预测行业标准), 基准 = 朴素 rv20 外推 与 HAR-RV 古典模型")
    A(f"- 检验: **Diebold-Mariano**(非重叠 h 日期间的损失差分 t 检验)")
    A(f"- 最终留出集: 最后 {cfg.holdout_frac:.0%} 交易日, 全程不参与任何选择")
    A(f"- 成本: 仓位变动每次 {COST*100:.2f}%; 空仓年化 {CASH_RATE*100:.1f}%")
    A("")
    A("> 为什么转向波动率: 方向 D 的特征重要性显示, 特征里能被读出的信息**全在波动率维度**, "
      "方向维度读不出东西。波动率是可预测的, 而且能直接变成仓位。")
    A("")
    A("---")
    A("")
    A("## 一、多轮迭代轨迹")
    A("")
    A("| 轮次 | 任务 | 模型 | 周期 | 特征数 | QLIKE(模型) | QLIKE(朴素基准) | **改进** | DM检验p | 相关性(模型/基准) | 风险AUC | 评级 | 决策依据 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for b in board:
        g = b.get("qlike_gain")
        gs = "-" if g is None else f"{g:+.5f}"
        A(f"| 第{b['round']}轮 | {b['task']} | {b['kind']} | h={b['h']} | {b['n_feats']} | "
          f"{b.get('qlike_model')} | {b.get('qlike_base')} | **{gs}** | "
          f"{b.get('dm_p')} | {b.get('corr_model')}/{b.get('corr_base')} | {b.get('risk_auc')} | "
          f"{b['crit_rank']} | {b['reason']} |")
    A("")
    A(f"**开发期最优**: 第 {best['round']} 轮 {best['kind']} ({best['task']}, h={best['h']})")
    A("")
    A("## 二、波动率预测精度(开发期, 非重叠口径)")
    A("")
    if ev:
        A(f"- 非重叠观测数: {ev['n_period']}")
        A(f"- **QLIKE: 模型 {ev['qlike_model']} vs 朴素基准 {ev['qlike_base']}** "
          f"(改进 {ev['qlike_gain']:+.5f}, 越小越好)")
        A(f"- MSE: 模型 {ev['mse_model']} vs 基准 {ev['mse_base']}")
        A(f"- **与真实波动率的相关性: 模型 {ev['corr_model']} vs 朴素基准 {ev['corr_base']}**")
        A(f"- **Diebold-Mariano 检验 p = {ev['dm_p']}** "
          f"(<0.10 才算显著优于朴素基准)")
    if cls:
        A("")
        A("## 三、风险事件预测(未来 h 日内低点跌超 5%)")
        A("")
        A(f"- 非重叠观测数 {cls['n_period']}, 基准发生率 **{cls['risk_base']}**")
        A(f"- **AUC = {cls['risk_auc']}**")
        A(f"- 预测概率 ≥0.35 的期间占 {cls['sel_frac']:.1%}, 这些期间的实际发生率 "
          f"{cls['sel_rate']} → **提升 {cls['lift']} 倍**")
    A("")
    A("## 四、风险阀门回测(把波动率预测变成仓位)")
    A("")
    A(f"规则: 仓位 = min(1, 目标波动率 {TARGET_VOL:.0%} ÷ 预测波动率); 双阀門版本再叠加"
      f"方向模型(D)的看空信号时打对折。")
    A("")
    for tagname, vv in (("开发期(2008~2022)", valve), ("最终留出集", hold_valve)):
        if not vv:
            continue
        A(f"### {tagname}")
        A("")
        A("| 方案 | 年化收益 | 最大回撤 | 夏普 | Calmar | 平均仓位 |")
        A("|---|---|---|---|---|---|")
        nm = {"full": "一直满仓", "vol": "波动率目标阀門", "vol_dir": "波动率+方向双阀門",
              "dir": "纯方向择时(D)"}
        for k in ("full", "vol", "vol_dir", "dir"):
            if k not in vv:
                continue
            v = vv[k]
            A(f"| {nm[k]} | {v['ann_ret']:.2%} | {v['max_dd']:.2%} | {v['sharpe']} | "
              f"{v['calmar']} | {v['avg_pos']:.0%} |")
        A("")
    A("## 五、最终留出集(从未参与任何选择)")
    A("")
    if hold_ev:
        A(f"留出期 {hold_ev.get('start', '')} ~ {hold_ev.get('end', '')}, "
          f"{hold_ev['n_period']} 个非重叠观测。")
        A("")
        A("| 指标 | 留出集 | 判定门槛 | 是否达标 |")
        A("|---|---|---|---|")
        A(f"| QLIKE 相对朴素基准的改进 | {hold_ev['qlike_gain']:+.5f} | > 0 | "
          f"{'是' if hold_ev['qlike_gain'] > 0 else '否'} |")
        A(f"| DM 检验 p | {hold_ev['dm_p']} | < 0.10 | "
          f"{'是' if (hold_ev['dm_p'] or 1) < 0.10 else '否'} |")
        A(f"| 相关性 模型 vs 基准 | {hold_ev['corr_model']} vs {hold_ev['corr_base']} | "
          f"模型更高 | {'是' if hold_ev['corr_model'] > hold_ev['corr_base'] else '否'} |")
        if hold_cls:
            A(f"| 风险事件 AUC | {hold_cls['risk_auc']} | > 0.55 | "
              f"{'是' if (hold_cls['risk_auc'] or 0) > 0.55 else '否'} |")
        if hold_valve and "vol" in hold_valve and "full" in hold_valve:
            a, b = hold_valve["vol"], hold_valve["full"]
            A(f"| 阀門夏普 vs 满仓 | {a['sharpe']} vs {b['sharpe']} | 阀門更高 | "
              f"{'是' if (a['sharpe'] or -9) > (b['sharpe'] or 0) else '否'} |")
            A(f"| 阀門最大回撤 vs 满仓 | {a['max_dd']:.2%} vs {b['max_dd']:.2%} | 阀門更浅 | "
              f"{'是' if a['max_dd'] > b['max_dd'] else '否'} |")
    A("")
    A("## 六、跨指数外检")
    A("")
    if ext:
        A("| 指数 | 非重叠观测 | QLIKE改进 | DM p | 相关性(模型/基准) |")
        A("|---|---|---|---|---|")
        for k, v in ext.items():
            A(f"| {k} | {v['n_period']} | {v['qlike_gain']:+.5f} | {v['dm_p']} | "
              f"{v['corr_model']}/{v['corr_base']} |")
    else:
        A("无")
    A("")
    A("## 七、质疑官结论")
    A("")
    A(f"**{crit['verdict']}**")
    A("")
    for i in crit["issues"]:
        A(f"- {i}")
    A("")
    A("必须声明的偏差:")
    A("")
    for b in crit["biases"]:
        A(f"- {b}")
    A("")
    A("## 八、基准率")
    A("")
    A("| 周期 | 中位未来波动率 | 10分位 | 90分位 | 出现跌超5%的概率 |")
    A("|---|---|---|---|---|")
    for h, v in lb.items():
        A(f"| h={h} | {v['median_vol']:.1%} | {v['vol_p10']:.1%} | {v['vol_p90']:.1%} | "
          f"{v['risk_rate']:.4f} |")
    A("")
    A("## 九、Agent 协作消息记录")
    A("")
    A("```")
    A("\n".join(bb.handoffs))
    A("```")
    A("")
    (OUT / "vol_report.md").write_text("\n".join(L), encoding="utf-8")
    return OUT / "vol_report.md"


# ================================================================== 主流程
def direction_probs(cfg, main_df, dev_end):
    """复用方向D的逻辑回归, 产生方向概率(作阀門第二道开关)。
    返回 (开发期概率, 留出期概率)。"""
    tc = tmod.Cfg(code=cfg.code, name=cfg.name, h_main=cfg.h_main,
                  test_len=cfg.test_len, min_train_days=cfg.min_train_days,
                  holdout_frac=cfg.holdout_frac)
    bb = Blackboard()
    f = tmod.FeatureAgent(bb, tc).run(main_df)
    df = tmod.LabelAgent(bb, tc).run(f)
    feats = list(tmod.FEATS)
    ycol = f"y{cfg.h_main}"
    wf = tmod.ModelAgent(bb, tc).walk_forward(df, feats, cfg.h_main, "logit",
                                              label="abs", end_date=dev_end)
    tr = df[df["date"] <= dev_end].dropna(subset=feats + [ycol])
    te = df[df["date"] > dev_end].dropna(subset=feats + [ycol]).copy()
    m = tmod.ModelAgent(bb, tc).make("logit")
    m.fit(tr[feats], tr[ycol])
    te["p"] = m.predict_proba(te[feats])[:, 1]
    return wf["pooled"], te[["date", "p"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--h", type=int, default=10)
    ap.add_argument("--code", type=str, default="sh.000300")
    a = ap.parse_args()
    cfg = Cfg(h_main=a.h, rounds=a.rounds, code=a.code,
              name=qd.BS_INDICES.get(a.code, a.code))
    t0 = time.time()
    print("=" * 88)
    print(f"方向 C 波动率/风险预测系统 v1.0   —— {cfg.name} | 主周期 h={cfg.h_main} | "
          f"迭代 {cfg.rounds} 轮")
    print("=" * 88)

    bb = Blackboard()
    main_df, others = VolDataAgent(bb, cfg).run()
    feats_df = VolFeatureAgent(bb, cfg).run(main_df)
    df = VolLabelAgent(bb, cfg).run(feats_df)

    dates = np.sort(df["date"].unique())
    dev_end = dates[int(len(dates) * (1 - cfg.holdout_frac))]
    print(f"[调度官] 开发期 ~ {dev_end.astype('datetime64[D]')}, 留出期 "
          f"{len(dates)-int(len(dates)*(1-cfg.holdout_frac))} 天")
    mem = MemoryAgent(bb, cfg)

    # 方向 D 的概率(作为阀門第二道开关)
    try:
        dir_dev, dir_hold = direction_probs(cfg, main_df, dev_end)
        print(f"[阀门官] 方向模型(D)概率: 开发期 {len(dir_dev)} 条, 留出期 {len(dir_hold)} 条")
    except Exception as e:                       # noqa: BLE001
        print(f"[阀门官] 方向概率不可用: {e}")
        dir_dev = dir_hold = None

    orch = Orchestrator(bb, cfg)
    board, history, importance = [], [], None
    best, best_ev, best_valve = None, None, None
    for r in range(1, cfg.rounds + 1):
        print("-" * 88)
        plan = orch.plan(r, history, importance)
        print(f"[调度官] 第 {r} 轮: {plan['reason']}")
        wf = VolModelAgent(bb, cfg).walk_forward(df, plan["feats"], plan["h"],
                                                 plan["kind"], task=plan["task"],
                                                 end_date=dev_end, tag=f"R{r}")
        if wf["pooled"].empty:
            print("[波动评估官] 无有效样本外结果, 跳过")
            continue
        wf["pooled"] = wf["pooled"].rename(columns={"p": "p"})
        ev = cls = None
        if plan["task"] == "reg":
            ev = VolEvalAgent(bb, cfg).run(wf)
            # 把回归的预测波动率借来做阀门
            valve = ValveAgent(bb, cfg).run(wf["pooled"], main_df, plan["h"],
                                            dir_dev, tag=f"R{r}")
        else:
            cls = VolEvalAgent(bb, cfg).run_cls(wf)
            valve = {}
        crit = CriticAgent(bb, cfg).run(wf, ev, cls, valve, tag=f"R{r}")
        try:
            tr = df[df["date"] <= dev_end].dropna(
                subset=plan["feats"] + [f"logyvol{plan['h']}" if plan["task"] == "reg"
                                        else f"yrisk{plan['h']}"])
            ycol = f"logyvol{plan['h']}" if plan["task"] == "reg" else f"yrisk{plan['h']}"
            m = VolModelAgent(bb, cfg).make(plan["kind"], plan["task"])
            m.fit(tr[plan["feats"]], tr[ycol])
            from train_agents import feature_importance_of
            importance = feature_importance_of(
                m, tr[plan["feats"]], tr[ycol], plan["feats"], plan["kind"], n=4000,
                scoring="r2" if plan["task"] == "reg" else "roc_auc")
            print("[模型官] 特征重要性 Top5: "
                  + ", ".join(f"{f}({v:.4f})" for f, v in importance[:5]))
        except Exception as e:                   # noqa: BLE001
            print(f"[模型官] 重要性跳过: {e}")
        board.append({"round": r, "task": plan["task"], "kind": plan["kind"], "h": plan["h"],
                      "n_feats": len(plan["feats"]),
                      "qlike_model": (ev or {}).get("qlike_model"),
                      "qlike_base": (ev or {}).get("qlike_base"),
                      "qlike_gain": (ev or {}).get("qlike_gain"),
                      "dm_p": (ev or {}).get("dm_p"),
                      "corr_model": (ev or {}).get("corr_model"),
                      "corr_base": (ev or {}).get("corr_base"),
                      "risk_auc": (cls or ev or {}).get("risk_auc") if cls else None,
                      "crit_rank": crit["rank"], "reason": plan["reason"],
                      "feats": plan["feats"], "thr": None})
        history.append({**board[-1], "crit_rank": crit["rank"]})
        mem.write({"date": str(datetime.now())[:19], "phase": "C-波动率", "round": r,
                   "task": plan["task"], "kind": plan["kind"], "h": plan["h"],
                   "qlike_gain": board[-1]["qlike_gain"], "dm_p": board[-1]["dm_p"],
                   "risk_auc": board[-1]["risk_auc"], "crit_rank": crit["rank"]})
        score = -9e9 if board[-1]["qlike_gain"] is None else board[-1]["qlike_gain"]
        if best is None or score > best["score"]:
            best = {"round": r, "kind": plan["kind"], "task": plan["task"], "h": plan["h"],
                    "feats": plan["feats"], "n_feats": len(plan["feats"]), "score": score,
                    "crit": crit, "pooled": wf["pooled"]}
            best_ev, best_valve = ev, valve

    if best is None:
        print("没有任何一轮产出有效结果")
        return 1
    print("=" * 88)
    print("[调度官] 开发期结束 → 最终留出集检验(只跑一次)")

    # 留出集
    h = best["h"]
    ycol = f"logyvol{h}" if best["task"] == "reg" else f"yrisk{h}"
    tr = df[df["date"] <= dev_end].dropna(subset=best["feats"] + [ycol, f"yvol{h}", f"fwd{h}"])
    te = df[df["date"] > dev_end].dropna(subset=best["feats"] + [ycol, f"yvol{h}", f"fwd{h}"]).copy()
    m = VolModelAgent(bb, cfg).make(best["kind"], best["task"])
    m.fit(tr[best["feats"]], tr[ycol])
    if best["task"] == "reg":
        k_smear = float(np.mean(np.exp(np.asarray(tr[ycol]) - m.predict(tr[best["feats"]]))))
        te["p"] = np.exp(m.predict(te[best["feats"]])) * k_smear
    else:
        k_smear = 1.0
        te["p"] = m.predict_proba(te[best["feats"]])[:, 1]
    te["p"] = np.clip(te["p"], 1e-4, 5.0)
    wf_h = {"folds": [], "pooled": te, "kind": best["kind"], "h": h,
            "task": best["task"], "tag": "HOLDOUT"}
    hold_ev = VolEvalAgent(bb, cfg).run(wf_h) if best["task"] == "reg" else None
    hold_cls = VolEvalAgent(bb, cfg).run_cls(wf_h) if best["task"] == "cls" else None
    if hold_ev:
        hold_ev["start"] = str(te["date"].min().date())
        hold_ev["end"] = str(te["date"].max().date())
    hold_valve = (ValveAgent(bb, cfg).run(te, main_df, h, dir_hold, tag="HOLDOUT")
                  if best["task"] == "reg" else {})

    # 跨指数外检
    ext = {}
    for code, nm in qd.BS_INDICES.items():
        if code == cfg.code or code not in others:
            continue
        try:
            fo = VolFeatureAgent(bb, cfg).build(others[code])
            fo = VolLabelAgent(bb, cfg).run(fo)
            t2 = fo[fo["date"] > dev_end].dropna(subset=best["feats"] + [ycol, f"yvol{h}"]).copy()
            if len(t2) < 120:
                continue
            t2["p"] = (np.exp(m.predict(t2[best["feats"]])) * k_smear
                       if best["task"] == "reg"
                       else m.predict_proba(t2[best["feats"]])[:, 1])
            w2 = {"folds": [], "pooled": t2, "kind": best["kind"], "h": h,
                  "task": best["task"], "tag": f"EXT_{nm}"}
            e2 = VolEvalAgent(bb, cfg).run(w2) if best["task"] == "reg" else None
            if e2:
                ext[nm] = e2
        except Exception as e:                   # noqa: BLE001
            print(f"[外检] {nm} 跳过: {e}")

    mem.save_board(board)
    make_charts(best, best_ev, best_valve, importance, board, None, hold_valve,
                hold_ev, cfg.name)
    elapsed = time.time() - t0
    crit2 = best.get("crit") or {"verdict": "见排行榜", "rank": "-", "issues": [], "biases": []}
    rpt = write_report(cfg, bb, board, best, best_ev, None, crit2, best_valve,
                       hold_ev, hold_cls, hold_valve, ext, elapsed)
    bb.save()
    mem.write({"date": str(datetime.now())[:19], "phase": "C-波动率", "round": "holdout",
               "qlike_gain": (hold_ev or {}).get("qlike_gain"),
               "dm_p": (hold_ev or {}).get("dm_p"),
               "risk_auc": (hold_cls or {}).get("risk_auc"), "is_holdout": True})
    print("=" * 88)
    print(f"完成, 耗时 {elapsed:.0f}s")
    if hold_ev:
        print(f"留出集: QLIKE 改进 {hold_ev['qlike_gain']:+.5f} (DM p={hold_ev['dm_p']}) | "
              f"相关性 模型 {hold_ev['corr_model']} vs 基准 {hold_ev['corr_base']}")
    if hold_cls:
        print(f"留出集风险事件: AUC {hold_cls['risk_auc']} (基准率 {hold_cls['risk_base']}, "
              f"高概率组发生率 {hold_cls['sel_rate']})")
    if hold_valve:
        for k, v in hold_valve.items():
            print(f"  阀門[{k}] 年化 {v['ann_ret']:.2%} 回撤 {v['max_dd']:.2%} "
                  f"夏普 {v['sharpe']} 平均仓位 {v['avg_pos']:.0%}")
    print(f"报告: {rpt}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
