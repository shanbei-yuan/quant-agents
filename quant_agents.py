#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多智能体量化选股系统 (Multi-Agent Quant Screener)  v1.0
=======================================================
约束: 本金 3000 元 · A股主板 · 低价股 · 最小 100 股/手 · 计入手续费

流水线 (每个 Agent 是一个独立角色, 通过共享黑板 blackboard 传递消息):

  DataAgent 数据官
      -> TrendAgent 趋势官
      -> MomentumAgent 动量官
      -> RiskAgent 风控官
      -> PortfolioAgent 组合官   (汇总三票 + 资金/手数约束)
      -> DecisionAgent 决策官    (买点/止损/止盈/卖出触发)
      -> CriticAgent 复盘官      (挑毛病 + 写入 memory.jsonl)

多智能体的三个机制在这里都有对应实现:
  1. 消息传递  -> Blackboard.handoff()  写进 logs/handoff.log, 能看到 A 给 B 传了什么
  2. 共享黑板  -> Blackboard.store      后续 Agent 读取前序 Agent 的产物
  3. 跨会话记忆 -> out/memory.jsonl     本次结论落盘, 下次运行被 CriticAgent 读回来对比

数据源: 东方财富公开行情接口 (无需 token)
用法:
    python quant_agents.py
    python quant_agents.py --capital 3000 --price-max 25 --max-positions 3
    LLM_API_KEY=sk-xxx python quant_agents.py --llm      # 额外生成人话研报
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
DATA_DIR, OUT_DIR, LOG_DIR = ROOT / "data", ROOT / "out", ROOT / "logs"
for _d in (DATA_DIR, OUT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

CLIST = "https://push2.eastmoney.com/api/qt/clist/get"
KLINE = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
KL_FIELDS = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
KL_COLS = ["date", "open", "close", "high", "low", "volume",
           "amount", "amplitude", "pct", "change", "turnover"]


# ---------------------------------------------------------------- 配置
@dataclass
class Config:
    capital: float = 3000.0          # 总资金
    price_min: float = 3.0           # 不买仙股
    price_max: float = 25.0          # 不买贵的股 (1手 <= 2500 元)
    lot: int = 100                   # A股最小交易单位
    max_positions: int = 3           # 最多持几只
    min_float_mktcap: float = 2.0e9  # 流通市值 >= 20 亿
    min_amount: float = 6.0e7        # 当日成交额 >= 6000 万
    pe_max: float = 60.0
    pb_max: float = 6.0
    prefilter_n: int = 60            # 进入技术分析的候选数
    min_bars: int = 130              # 至少 130 个交易日历史
    commission_rate: float = 0.00025 # 佣金 万2.5
    commission_min: float = 5.0      # 佣金单笔最低 5 元
    stamp_tax: float = 0.0005        # 印花税 卖出 0.05%
    transfer_fee: float = 0.00001    # 过户费 双向 0.001%
    w_trend: float = 0.40
    w_mom: float = 0.30
    w_risk: float = 0.30
    score_floor: float = 55.0        # 入选最低综合分
    use_llm: bool = False


# ---------------------------------------------------------------- 工具
def _get(url: str, params: dict, retries: int = 4, timeout: int = 20) -> dict:
    last = None
    for i in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:            # noqa: BLE001
            last = e
            time.sleep(0.8 * (i + 1))
    raise RuntimeError(f"请求失败 {url} :: {last}")


def fetch_snapshot() -> pd.DataFrame:
    """全市场 A 股快照 (含科创板/创业板, 后面靠代码前缀剔除)。"""
    fields = "f12,f13,f14,f2,f3,f5,f6,f8,f9,f20,f21,f23"
    rows, pn = [], 1
    while True:
        j = _get(CLIST, {
            "pn": pn, "pz": 1000, "po": 1, "np": 1, "fltt": 2, "invt": 2, "fid": "f6",
            "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
            "fields": fields, "_": int(time.time() * 1000)})
        d = j.get("data") or {}
        diff = d.get("diff") or []
        if not diff:
            break
        rows += diff
        total = int(d.get("total") or 0)
        if len(rows) >= total or len(diff) < 1000:
            break
        pn += 1
        time.sleep(0.2)
    df = pd.DataFrame(rows).rename(columns={
        "f12": "code", "f13": "mkt", "f14": "name", "f2": "price", "f3": "pct",
        "f5": "volume", "f6": "amount", "f8": "turnover", "f9": "pe",
        "f20": "mktcap", "f21": "float_cap", "f23": "pb"})
    for c in ["price", "pct", "amount", "turnover", "pe", "mktcap", "float_cap", "pb", "mkt"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["code"] = df["code"].astype(str)
    return df


def fetch_kline(mkt, code: str, lmt: int = 250):
    j = _get(KLINE, {
        "secid": f"{int(mkt)}.{code}", "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": KL_FIELDS, "klt": 101, "fqt": 1, "end": "20500101",
        "lmt": lmt, "_": int(time.time() * 1000)})
    kl = (j.get("data") or {}).get("klines") or []
    if not kl:
        return None
    df = pd.DataFrame([x.split(",") for x in kl], columns=KL_COLS)
    df["date"] = pd.to_datetime(df["date"])
    for c in KL_COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("date").sort_index()


def fetch_benchmark():
    """沪深300 -> 中证500 -> 上证指数，取第一个能返回数据的。"""
    for sid, nm in (("1.000300", "沪深300"), ("1.000905", "中证500"), ("1.000001", "上证指数")):
        try:
            j = _get(KLINE, {"secid": sid, "fields1": "f1,f2,f3,f4,f5,f6",
                             "fields2": KL_FIELDS, "klt": 101, "fqt": 1,
                             "end": "20500101", "lmt": 250})
            kl = (j.get("data") or {}).get("klines") or []
            if kl:
                df = pd.DataFrame([x.split(",") for x in kl], columns=KL_COLS)
                df["date"] = pd.to_datetime(df["date"])
                for c in KL_COLS[1:]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                return nm, df.set_index("date").sort_index()
        except Exception:                  # noqa: BLE001
            continue
    return None, None


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    for n in (5, 10, 20, 60):
        df[f"ma{n}"] = c.rolling(n).mean()
    dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    df["dif"], df["dea"], df["macd"] = dif, dea, (dif - dea) * 2
    d = c.diff()
    up = d.clip(lower=0).rolling(14).mean()
    dn = (-d.clip(upper=0)).rolling(14).mean()
    df["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - c.shift()).abs(),
                    (df["low"] - c.shift()).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["vol_ma5"] = df["volume"].rolling(5).mean()
    df["vol_ma60"] = df["volume"].rolling(60).mean()
    df["vol20"] = c.pct_change().rolling(20).std()
    return df


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ---------------------------------------------------------------- 黑板
class Blackboard:
    def __init__(self):
        self.store: dict = {}
        self.handoffs: list[str] = []
        self.t0 = time.time()

    def put(self, agent: str, payload: dict):
        self.store[agent] = payload
        return payload

    def handoff(self, src: str, dst: str, msg: str):
        line = f"[{datetime.now():%H:%M:%S}] {src:<16s} -> {dst:<16s} | {msg}"
        self.handoffs.append(line)
        print("   " + line, flush=True)

    def save(self):
        (OUT_DIR / "blackboard.json").write_text(
            json.dumps({"store": self.store, "handoffs": self.handoffs},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        (LOG_DIR / "handoff.log").write_text("\n".join(self.handoffs) + "\n", encoding="utf-8")


class Agent:
    name = "Agent"

    def __init__(self, bb: Blackboard, cfg: Config):
        self.bb, self.cfg = bb, cfg

    def log(self, msg: str):
        print(f"[{self.name}] {msg}", flush=True)

    def run(self):
        raise NotImplementedError


# ---------------------------------------------------------------- 1 数据官
class DataAgent(Agent):
    name = "数据官"

    def run(self):
        snap = fetch_snapshot()
        self.log(f"全市场快照 {len(snap)} 只")
        pool = self.prefilter(snap)
        self.log(f"硬约束过滤后候选池 {len(pool)} 只 "
                 f"(主板/非ST/{self.cfg.price_min:.0f}-{self.cfg.price_max:.0f}元/"
                 f"成交额>={self.cfg.min_amount/1e8:.1f}亿)")

        klines, meta = {}, {}
        for _, r in pool.iterrows():
            try:
                k = fetch_kline(r["mkt"], r["code"])
            except Exception as e:          # noqa: BLE001
                self.log(f"  跳过 {r['code']}: {e}")
                continue
            if k is None or len(k) < self.cfg.min_bars:
                continue
            klines[r["code"]] = add_indicators(k)
            meta[r["code"]] = {"name": r["name"], "mkt": int(r["mkt"]),
                               "pe": float(r["pe"]), "pb": float(r["pb"]),
                               "float_cap": float(r["float_cap"]),
                               "amount": float(r["amount"])}
            time.sleep(0.12)
        self.log(f"成功获取 {len(klines)} 只日线")

        bname, bench = fetch_benchmark()
        self.log(f"基准指数: {bname} ({0 if bench is None else len(bench)} 根)")

        last_date = max(k.index[-1] for k in klines.values())
        self.bb.put("DataAgent", {"n_universe": len(snap), "n_pool": len(klines),
                                  "bench": bname, "last_date": str(last_date.date())})
        self.bb.handoff(self.name, "趋势/动量/风控官",
                        f"{len(klines)} 只候选 × 250日线就绪, 基准={bname}, 最新交易日={last_date.date()}")
        return klines, meta, bench, bname

    def prefilter(self, snap: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg
        d = snap.copy()
        # 只留主板: 沪 600/601/603/605, 深 000/001/002/003
        d = d[d["code"].str.match(r"^(600|601|603|605|000|001|002|003)\d{3}$")]
        d = d[~d["name"].astype(str).str.contains("ST|退", na=False)]
        d = d[(d["price"] >= c.price_min) & (d["price"] <= c.price_max)]
        d = d[d["float_cap"] >= c.min_float_mktcap]
        d = d[d["amount"] >= c.min_amount]
        d = d[(d["pe"] > 0) & (d["pe"] <= c.pe_max)]
        d = d[(d["pb"] > 0) & (d["pb"] <= c.pb_max)]
        # 硬性: 1 手买得起
        d = d[d["price"] * c.lot * (1 + c.commission_rate) <= c.capital]
        return d.sort_values("amount", ascending=False).head(c.prefilter_n).reset_index(drop=True)


# ---------------------------------------------------------------- 2 趋势官
class TrendAgent(Agent):
    name = "趋势官"

    def run(self, klines, meta):
        out = {}
        for code, df in klines.items():
            r = df.iloc[-1]
            s, notes = 0.0, []
            if r.ma5 > r.ma10 > r.ma20:
                s += 22; notes.append("均线多头排列")
            elif r.ma5 > r.ma20:
                s += 10; notes.append("短均线上穿中期")
            else:
                notes.append("均线未走多头")
            if r.close > r.ma20:
                s += 12; notes.append("站上MA20")
            if pd.notna(r.ma20) and r.ma20 > df["ma20"].iloc[-6]:
                s += 10; notes.append("MA20向上")
            if pd.notna(r.ma60) and r.close > r.ma60:
                s += 8; notes.append("站上MA60")
            if pd.notna(r.ma60) and df["ma60"].iloc[-1] >= df["ma60"].iloc[-21]:
                s += 6
            if r.dif > r.dea:
                s += 12; notes.append("MACD多头")
            if r.macd > 0:
                s += 8
            hi60, lo60 = df["high"].iloc[-60:].max(), df["low"].iloc[-60:].min()
            pos = (r.close - lo60) / (hi60 - lo60) if hi60 > lo60 else 0.5
            s += 12 * pos
            out[code] = {"score": clamp(s, 0, 100), "notes": notes, "pos60": round(pos, 3)}
        self.bb.put("TrendAgent", {k: round(v["score"], 1) for k, v in out.items()})
        self.bb.handoff(self.name, "组合官", f"完成 {len(out)} 只趋势打分, 均值 "
                        f"{np.mean([v['score'] for v in out.values()]):.1f}")
        return out


# ---------------------------------------------------------------- 3 动量官
class MomentumAgent(Agent):
    name = "动量官"

    def run(self, klines, meta):
        out = {}
        for code, df in klines.items():
            r = df.iloc[-1]
            notes = []
            rsi = float(r.rsi14) if pd.notna(r.rsi14) else 50.0
            if 50 <= rsi <= 70:
                a = 30
            elif 45 <= rsi < 50 or 70 < rsi <= 78:
                a = 20
            elif 35 <= rsi < 45:
                a = 12
            elif rsi > 78:
                a = 6; notes.append(f"RSI={rsi:.0f} 超买")
            else:
                a = 8; notes.append(f"RSI={rsi:.0f} 超卖")
            ret5 = r.close / df["close"].iloc[-6] - 1
            ret20 = r.close / df["close"].iloc[-21] - 1
            b = clamp(20 + 200 * ret20, 0, 40)
            volr = float(r.vol_ma5 / r.vol_ma60) if r.vol_ma60 and r.vol_ma60 > 0 else 1.0
            cc = clamp(15 * volr, 0, 30)
            if r.close > df["high"].iloc[-21:-1].max() and volr > 1.2:
                cc = clamp(cc + 10, 0, 40); notes.append("放量突破20日高点")
            s = clamp(a + b + cc, 0, 100)
            out[code] = {"score": s, "rsi": round(rsi, 1),
                         "ret5": round(float(ret5) * 100, 2),
                         "ret20": round(float(ret20) * 100, 2),
                         "vol_ratio": round(volr, 2), "notes": notes}
        self.bb.put("MomentumAgent", {k: {"score": round(v["score"], 1), "rsi": v["rsi"],
                                          "ret20": v["ret20"], "vol_ratio": v["vol_ratio"]}
                                      for k, v in out.items()})
        self.bb.handoff(self.name, "组合官", f"完成 {len(out)} 只动量打分")
        return out


# ---------------------------------------------------------------- 4 风控官
class RiskAgent(Agent):
    name = "风控官"

    def run(self, klines, meta, bench):
        bench_ret20 = None
        if bench is not None and len(bench) > 21:
            bench_ret20 = float(bench["close"].iloc[-1] / bench["close"].iloc[-21] - 1)
        out = {}
        for code, df in klines.items():
            r = df.iloc[-1]
            ann_vol = float(r.vol20 * math.sqrt(250)) if pd.notna(r.vol20) else 0.5
            win = df["close"].iloc[-60:]
            dd = float((win / win.cummax() - 1).min())
            atr_pct = float(r.atr14 / r.close) if pd.notna(r.atr14) else 0.04
            ret20 = float(r.close / df["close"].iloc[-21] - 1)
            rs = ret20 - bench_ret20 if bench_ret20 is not None else ret20
            s = (clamp(30 * (1 - ann_vol / 0.60), 0, 30)      # 波动越低越好
                 + clamp(25 * (1 + dd / 0.25), 0, 25)         # 回撤越小越好
                 + clamp(25 + 150 * rs, 0, 25)                # 相对大盘强弱
                 + clamp(20 * (1 - atr_pct / 0.06), 0, 20))   # 单日振幅
            if ann_vol > 0.55:
                level = "高"
            elif ann_vol > 0.40:
                level = "中"
            else:
                level = "低"
            out[code] = {"score": clamp(s, 0, 100), "ann_vol": round(ann_vol, 3),
                         "max_dd": round(dd, 3), "atr_pct": round(atr_pct, 4),
                         "rs20": round(rs * 100, 2), "level": level}
        self.bb.put("RiskAgent", {"bench_ret20": None if bench_ret20 is None else round(bench_ret20 * 100, 2),
                                  "detail": {k: {kk: v[kk] for kk in ("score", "ann_vol", "max_dd", "level")}
                                             for k, v in out.items()}})
        self.bb.handoff(self.name, "组合官",
                        f"完成 {len(out)} 只风控打分; 大盘20日涨幅="
                        f"{'N/A' if bench_ret20 is None else f'{bench_ret20*100:.2f}%'}")
        return out


# ---------------------------------------------------------------- 5 组合官
class PortfolioAgent(Agent):
    name = "组合官"

    def run(self, klines, meta, trend, mom, risk):
        c = self.cfg
        rows = []
        for code in klines:
            t, m, k = trend[code]["score"], mom[code]["score"], risk[code]["score"]
            total = c.w_trend * t + c.w_mom * m + c.w_risk * k
            price = float(klines[code]["close"].iloc[-1])
            gross = price * c.lot
            buy_fee = max(c.commission_min, gross * c.commission_rate) + gross * c.transfer_fee
            rows.append({"code": code, "name": meta[code]["name"], "price": round(price, 2),
                         "trend": round(t, 1), "mom": round(m, 1), "risk": round(k, 1),
                         "score": round(total, 1), "cost": round(gross + buy_fee, 2),
                         "buy_fee": round(buy_fee, 2), "rsi": mom[code]["rsi"],
                         "ann_vol": risk[code]["ann_vol"], "ret20": mom[code]["ret20"],
                         "vol_ratio": mom[code]["vol_ratio"], "rs20": risk[code]["rs20"],
                         "pe": round(meta[code]["pe"], 1), "pb": round(meta[code]["pb"], 2)})
        rows.sort(key=lambda x: -x["score"])
        best = rows[0]["score"] if rows else 0
        floor = max(c.score_floor, best * 0.72)
        cands = [r for r in rows if r["score"] >= floor]
        self.log(f"评分完成 {len(rows)} 只; 入选门槛 {floor:.1f} 分 -> {len(cands)} 只进入组合优化")

        # 枚举组合: 在 3000 元和最多 N 只 的约束下, 最大化资金利用率加权的总分
        best_combo, best_obj = None, -1.0
        for k in range(1, min(c.max_positions, len(cands)) + 1):
            for combo in combinations(range(len(cands)), k):
                cost = sum(cands[i]["cost"] for i in combo)
                if cost > c.capital:
                    continue
                util = cost / c.capital
                score = sum(cands[i]["score"] for i in combo)
                obj = score * (1 - 0.35 * max(0.0, 0.60 - util) / 0.60)
                if obj > best_obj:
                    best_obj, best_combo = obj, combo
        picks = [cands[i] for i in (best_combo or ())]
        invested = sum(p["cost"] for p in picks)
        self.bb.put("PortfolioAgent", {"ranking": rows[:10], "picks": picks,
                                       "invested": round(invested, 2),
                                       "cash_left": round(c.capital - invested, 2),
                                       "utilization": round(invested / c.capital * 100, 1)})
        self.bb.handoff(self.name, "决策官",
                        f"选定 {len(picks)} 只: " + ", ".join(p["name"] for p in picks)
                        + f"; 占用 {invested:.0f}/{c.capital:.0f} 元")
        return rows, picks, invested


# ---------------------------------------------------------------- 6 决策官
class DecisionAgent(Agent):
    name = "决策官"

    def run(self, klines, picks, invested):
        c = self.cfg
        plans = []
        for p in picks:
            df = klines[p["code"]]
            r = df.iloc[-1]
            price, atr = float(r.close), float(r.atr14)
            buy_lo, buy_hi = round(price * 0.975, 2), round(price * 1.015, 2)
            stop = round(min(price - 1.5 * atr, price * 0.94), 2)
            stop = max(stop, price * 0.90)                 # 止损不深于 -10%
            t1, t2 = round(price * 1.08, 2), round(price * 1.15, 2)
            gross = price * c.lot
            buy_fee = max(c.commission_min, gross * c.commission_rate) + gross * c.transfer_fee
            cost = gross + buy_fee
            # 保本价: net(P)=cost, 佣金按最低 5 元算
            be = (cost + c.commission_min) / (c.lot * (1 - c.commission_rate - c.transfer_fee - c.stamp_tax))
            max_loss = (price - stop) * c.lot + buy_fee + c.commission_min + price * c.lot * (c.transfer_fee + c.stamp_tax)
            plans.append({**p, "buy_lo": buy_lo, "buy_hi": buy_hi, "stop": stop,
                          "stop_pct": round((stop / price - 1) * 100, 1),
                          "t1": t1, "t2": t2, "breakeven": round(be, 3),
                          "total_cost": round(cost, 2), "max_loss": round(max_loss, 2),
                          "weight": round(cost / c.capital * 100, 1),
                          "atr": round(atr, 3)})
        sell_rules = [
            "止损: 收盘跌破止损价 → 次日开盘无条件清仓 (不看理由, 不加仓摊平)",
            "止盈一: 触及 +8% 卖出一半, 剩余仓位成本归零",
            "止盈二: 触及 +15% 或 收盘跌破 MA10 → 卖出剩余",
            "移动止盈: 浮盈超过 10% 后, 从最高点回撤 6% 即全部卖出",
            "时间止损: 持有满 30 个交易日仍无盈利 (浮盈<2%) → 换股",
            "异常退出: 公告重大利空/单日跌幅>7%/涨停打开放巨量 → 直接走",
        ]
        self.bb.put("DecisionAgent", {"plans": plans, "sell_rules": sell_rules})
        self.bb.handoff(self.name, "复盘官", f"输出 {len(plans)} 份交易计划 + {len(sell_rules)} 条卖出规则")
        return plans, sell_rules

    def llm_narrative(self, plans, ranking, bench_name):
        if not self.cfg.use_llm:
            return None
        key = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            self.log("未设置 LLM_API_KEY, 跳过 LLM 研报")
            return None
        base = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
        model = os.environ.get("LLM_MODEL", "deepseek-chat")
        payload = {"capital": self.cfg.capital, "benchmark": bench_name,
                   "top10": ranking[:10], "plans": plans}
        prompt = ("你是A股量化研究员。下面是程序化因子模型给出的选股结果(JSON)。"
                  "请写一份 400 字以内的中文简报, 包含: 1)当前大盘环境判断 2)为什么是这几只 "
                  "3)最大风险点 4)给新手的具体执行提醒。不要编造 JSON 里没有的数据。\n"
                  + json.dumps(payload, ensure_ascii=False))
        try:
            r = requests.post(base + "/chat/completions",
                              headers={"Authorization": f"Bearer {key}",
                                       "Content-Type": "application/json"},
                              json={"model": model, "temperature": 0.3,
                                    "messages": [{"role": "user", "content": prompt}]},
                              timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:              # noqa: BLE001
            self.log(f"LLM 调用失败, 改用模板研报: {e}")
            return None


# ---------------------------------------------------------------- 7 复盘官
class CriticAgent(Agent):
    name = "复盘官"

    def run(self, klines, picks, plans, ranking, meta, bb):
        warnings, checks = [], []
        last_run = None
        mem_path = OUT_DIR / "memory.jsonl"
        if mem_path.exists():
            lines = [l for l in mem_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            if lines:
                try:
                    last_run = json.loads(lines[-1])
                except Exception:           # noqa: BLE001
                    last_run = None
        if last_run:
            checks.append(f"读取到上次运行记忆 ({last_run.get('date')}), 上次推荐: "
                          + ", ".join(p.get("name", "?") for p in last_run.get("picks", [])))
        else:
            checks.append("首次运行, 无历史记忆可对比")

        invested = sum(p["cost"] for p in plans)
        if invested / self.cfg.capital > 0.97:
            warnings.append("仓位接近满仓, 没有给手续费和补仓留余地")
        if any(p["rsi"] > 75 for p in picks):
            warnings.append("组合中存在 RSI>75 的标的, 短期超买, 建议等回踩再买而不是追高")
        if len(picks) == 1:
            warnings.append("只押一只, 个股黑天鹅无法分散")
        if any(p["rs20"] < 0 for p in picks):
            warnings.append("存在弱于大盘的标的 (相对强弱为负), 属于逆势操作")
        if any(p["ann_vol"] > 0.55 for p in picks):
            warnings.append("存在年化波动率 >55% 的标的, 单日振幅可能超过 4%")
        worst = max((p["max_loss"] for p in plans), default=0)
        if worst > self.cfg.capital * 0.08:
            warnings.append(f"单只最大潜在亏损 {worst:.0f} 元, 超过总资金 8%")
        if not warnings:
            warnings.append("未发现显著结构性风险, 但请注意模型不预测突发事件")

        critic = {"checks": checks, "warnings": warnings,
                  "samples": len(klines), "sample_note":
                      "候选 60 只来自成交额排序, 非全市场穷举, 存在选择偏差"}
        self.bb.put("CriticAgent", critic)

        with mem_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"date": str(datetime.now())[:19],
                                "capital": self.cfg.capital,
                                "picks": [{"code": p["code"], "name": p["name"],
                                           "score": p["score"], "price": p["price"]}
                                          for p in picks],
                                "warnings": warnings}, ensure_ascii=False) + "\n")
        self.bb.handoff(self.name, "报告", f"提出 {len(warnings)} 条风险警示, 已写入 memory.jsonl")
        return critic


# ---------------------------------------------------------------- 出图
def setup_font():
    for p in ("/mnt/c/Windows/Fonts/msyh.ttc", "/mnt/c/Windows/Fonts/simhei.ttf",
              "/mnt/c/Windows/Fonts/msyhbd.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        if os.path.exists(p):
            try:
                font_manager.fontManager.addfont(p)
                nm = font_manager.FontProperties(fname=p).get_name()
                plt.rcParams["font.sans-serif"] = [nm]
                plt.rcParams["axes.unicode_minus"] = False
                return nm
            except Exception:               # noqa: BLE001
                continue
    plt.rcParams["axes.unicode_minus"] = False
    return None


def make_charts(klines, plans, ranking, bench):
    setup_font()
    top = plans[0]
    df = klines[top["code"]].iloc[-120:]
    x = np.arange(len(df))
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.4, 1.2, 1.2]})
    ax = axes[0]
    ax.plot(x, df["close"].values, lw=1.8, label="收盘价", color="#1f77b4")
    ax.plot(x, df["ma20"].values, lw=1.2, label="MA20", color="#ff7f0e")
    ax.plot(x, df["ma60"].values, lw=1.2, label="MA60", color="#2ca02c")
    if bench is not None:
        b = bench["close"].reindex(df.index)
        if b.notna().any():
            ax.plot(x, (b / b.dropna().iloc[0] * df["close"].iloc[0]).values,
                    lw=1.0, ls="--", color="#999999", label="基准(归一)")
    ax.axhspan(top["buy_lo"], top["buy_hi"], color="#2ca02c", alpha=0.15, label="买入区间")
    ax.axhline(top["stop"], color="#d62728", ls="--", lw=1.2, label=f"止损 {top['stop']}")
    ax.axhline(top["t1"], color="#9467bd", ls=":", lw=1.2, label=f"止盈一 {top['t1']}")
    ax.axhline(top["t2"], color="#8c564b", ls=":", lw=1.2, label=f"止盈二 {top['t2']}")
    ax.set_title(f"{top['name']} ({top['code']})  综合评分 {top['score']}  —  近120个交易日")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.bar(x, df["volume"].values / 1e4, color="#7f7f7f", alpha=0.7)
    ax.plot(x, (df["vol_ma5"] / 1e4).values, lw=1.2, color="#d62728", label="5日均量")
    ax.set_ylabel("成交量\n(万手)"); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[2]
    ax.bar(x, df["macd"].values, color=np.where(df["macd"].values >= 0, "#d62728", "#2ca02c"), alpha=0.7)
    ax.plot(x, df["dif"].values, lw=1.0, color="#1f77b4", label="DIF")
    ax.plot(x, df["dea"].values, lw=1.0, color="#ff7f0e", label="DEA")
    ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("MACD"); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[3]
    ax.plot(x, df["rsi14"].values, lw=1.4, color="#9467bd")
    ax.axhline(70, color="#d62728", ls="--", lw=0.9)
    ax.axhline(30, color="#2ca02c", ls="--", lw=0.9)
    ax.set_ylabel("RSI14"); ax.grid(alpha=0.25)
    ticks = list(range(0, len(df), 15))
    axes[3].set_xticks(ticks)
    axes[3].set_xticklabels([df.index[i].strftime("%m-%d") for i in ticks], rotation=45, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dashboard.png", dpi=130)
    plt.close(fig)

    top10 = ranking[:10]
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [f"{r['name']}\n{r['code']}" for r in top10]
    y = np.arange(len(top10))
    ax.barh(y, [r["score"] for r in top10], color="#4c72b0")
    for i, r in enumerate(top10):
        ax.text(r["score"] + 0.6, i, f"{r['score']:.1f}", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, 105); ax.set_xlabel("综合评分")
    ax.set_title("候选池综合评分 Top10 (趋势40% + 动量30% + 风控30%)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ranking.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- 报告
def write_report(cfg, bb, plans, sell_rules, ranking, critic, bench_name, narrative, elapsed):
    p = bb.store.get("DataAgent", {})
    port = bb.store.get("PortfolioAgent", {})
    L = []
    A = L.append
    A("# 多智能体量化选股报告")
    A("")
    A(f"- 生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    A(f"- 数据最新交易日: **{p.get('last_date')}** (日线, 前复权)")
    A(f"- 本金: **{cfg.capital:.0f} 元** | 单只最多 1 手 = {cfg.lot} 股 | 最多持 {cfg.max_positions} 只")
    A(f"- 基准指数: {bench_name} | 全市场 {p.get('n_universe')} 只 -> 硬约束过滤 -> 技术分析 {p.get('n_pool')} 只")
    A(f"- 运行耗时: {elapsed:.1f}s")
    A("")
    A("> 本报告由 7 个 agent 流水线生成 (数据官/趋势官/动量官/风控官/组合官/决策官/复盘官)，"
      "仅为量化模型输出，不构成投资建议。")
    A("")
    A("---")
    A("")
    A("## 一、结论: 买什么")
    A("")
    A("| # | 代码 | 名称 | 现价 | 综合分 | 趋势 | 动量 | 风控 | 建议买入区间 | 1手总成本 | 资金占比 |")
    A("|---|------|------|------|--------|------|------|------|--------------|-----------|----------|")
    for i, pl in enumerate(plans, 1):
        A(f"| {i} | {pl['code']} | {pl['name']} | {pl['price']} | **{pl['score']}** | "
          f"{pl['trend']} | {pl['mom']} | {pl['risk']} | {pl['buy_lo']} ~ {pl['buy_hi']} | "
          f"{pl['total_cost']} | {pl['weight']}% |")
    A("")
    A(f"合计占用 **{port.get('invested')} 元**, 预留现金 **{port.get('cash_left')} 元** "
      f"(资金利用率 {port.get('utilization')}%)。")
    A("")
    A("## 二、何时买 / 何时抛")
    A("")
    for i, pl in enumerate(plans, 1):
        A(f"### {i}. {pl['name']} ({pl['code']})")
        A("")
        A(f"- **买入**: 在 {pl['buy_lo']} ~ {pl['buy_hi']} 元分 2 笔建仓；若开盘跳空高开超过 3%，"
          f"放弃该笔、等回踩 MA5/MA10 再接")
        A(f"- **止损**: {pl['stop']} 元 ({pl['stop_pct']}%，约 1.5×ATR14={pl['atr']} 的波动带)，"
          f"跌破即次日开盘清仓")
        A(f"- **止盈**: 一档 {pl['t1']} 元 (+8%，卖一半) → 二档 {pl['t2']} 元 (+15% 或跌破 MA10 全出)")
        A(f"- **保本价**: {pl['breakeven']} 元 (买卖两趟手续费共约 "
          f"{pl['total_cost']-pl['price']*cfg.lot:.1f} 元，涨幅不到这个数都是白干)")
        A(f"- **最大潜在亏损**: {pl['max_loss']} 元 (占总资金 {pl['max_loss']/cfg.capital*100:.1f}%)")
        A(f"- 参考指标: RSI14={pl['rsi']} | 20日涨幅={pl['ret20']}% | 量比={pl['vol_ratio']} | "
          f"年化波动={pl['ann_vol']*100:.0f}% | 相对大盘20日={pl['rs20']}% | PE={pl['pe']} PB={pl['pb']}")
        A("")
    A("### 通用卖出纪律 (适用于所有持仓)")
    A("")
    for r in sell_rules:
        A(f"1. {r}")
    A("")
    if narrative:
        A("### AI 研究员简报 (LLM)")
        A("")
        A(narrative)
        A("")
    A("## 三、复盘官的质疑")
    A("")
    for c in critic["checks"]:
        A(f"- {c}")
    A("")
    for w in critic["warnings"]:
        A(f"- ⚠️ {w}")
    A("")
    A(f"- 样本说明: {critic['sample_note']}")
    A("")
    A("## 四、候选评分 Top10")
    A("")
    A("| # | 代码 | 名称 | 现价 | 综合分 | 趋势 | 动量 | 风控 | 1手成本 | RSI |")
    A("|---|------|------|------|--------|------|------|------|---------|-----|")
    for i, r in enumerate(ranking[:10], 1):
        A(f"| {i} | {r['code']} | {r['name']} | {r['price']} | {r['score']} | {r['trend']} | "
          f"{r['mom']} | {r['risk']} | {r['cost']} | {r['rsi']} |")
    A("")
    A("## 五、成本结构 (为什么 3000 元不适合做短线)")
    A("")
    A("| 项目 | 规则 | 3000 元本金下的实际值 |")
    A("|---|---|---|")
    A(f"| 佣金 | 万2.5，单笔最低 5 元 | 一买一卖 = {cfg.commission_min*2:.0f} 元 |")
    A(f"| 印花税 | 卖出 0.05% | 约 1.5 元/笔 |")
    A(f"| 过户费 | 双向 0.001% | 约 0.3 元/笔 |")
    A(f"| **往返合计** | | **约 11.8 元 ≈ 总资金 0.39%** |")
    A("")
    A("也就是说：每次完整的买入卖出，账户先亏 0.39%。若一周做一次，一年光手续费就吃掉 20% 本金。"
      "**3000 元的小资金唯一可行的打法是低频波段（持有 2~8 周），不是日内交易。**")
    A("")
    A("## 六、Agent 之间的消息传递记录")
    A("")
    A("```")
    A("\n".join(bb.handoffs))
    A("```")
    A("")
    (OUT_DIR / "report.md").write_text("\n".join(L), encoding="utf-8")
    return OUT_DIR / "report.md"


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="多智能体量化选股")
    ap.add_argument("--capital", type=float, default=3000.0)
    ap.add_argument("--price-min", type=float, default=3.0)
    ap.add_argument("--price-max", type=float, default=25.0)
    ap.add_argument("--max-positions", type=int, default=3)
    ap.add_argument("--prefilter", type=int, default=60)
    ap.add_argument("--llm", action="store_true", help="额外用 LLM 生成研报 (需 LLM_API_KEY)")
    a = ap.parse_args()

    cfg = Config(capital=a.capital, price_min=a.price_min, price_max=a.price_max,
                 max_positions=a.max_positions, prefilter_n=a.prefilter, use_llm=a.llm)
    t0 = time.time()
    print("=" * 78)
    print("多智能体量化选股系统  v1.0")
    print(f"本金 {cfg.capital:.0f} 元 | 价格 {cfg.price_min:.0f}~{cfg.price_max:.0f} 元 | "
          f"{cfg.lot} 股/手 | 最多 {cfg.max_positions} 只")
    print("=" * 78)

    bb = Blackboard()
    klines, meta, bench, bname = DataAgent(bb, cfg).run()
    if not klines:
        print("候选池为空, 请放宽过滤条件")
        return 1
    trend = TrendAgent(bb, cfg).run(klines, meta)
    mom = MomentumAgent(bb, cfg).run(klines, meta)
    risk = RiskAgent(bb, cfg).run(klines, meta, bench)
    ranking, picks, invested = PortfolioAgent(bb, cfg).run(klines, meta, trend, mom, risk)
    dec = DecisionAgent(bb, cfg)
    plans, sell_rules = dec.run(klines, picks, invested)
    narrative = dec.llm_narrative(plans, ranking, bname)
    critic = CriticAgent(bb, cfg).run(klines, picks, plans, ranking, meta, bb)

    make_charts(klines, plans, ranking, bench)
    elapsed = time.time() - t0
    rpt = write_report(cfg, bb, plans, sell_rules, ranking, critic, bname, narrative, elapsed)
    bb.save()

    print("-" * 78)
    print(f"完成, 耗时 {elapsed:.1f}s")
    for i, pl in enumerate(plans, 1):
        print(f"  {i}. {pl['name']}({pl['code']}) 现价 {pl['price']} | 评分 {pl['score']} | "
              f"买 {pl['buy_lo']}~{pl['buy_hi']} | 止 {pl['stop']} | 盈 {pl['t1']}/{pl['t2']} | "
              f"1手 {pl['total_cost']} 元")
    print(f"  报告: {rpt}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
