#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第1条建议: 波动率仓位管理框架 vol_target.py
==========================================
把方向 C 的波动率预测变成**可配置的仓位管理系统**:
    仓位 = clip( 目标波动率 / 预测波动率 , 0 , 最大杠杆 )
并做 5 个指数 × 3 个目标波动率 × 2 档杠杆 的完整评估。

严谨性:
  * 波动率预测用 walk-forward(开发期) / 开发期训练后只测一次(留出期)
  * 杠杆部分按年化 4% 计融资成本, 空仓按 1.5% 计货币收益 —— 不做"免费杠杆"的假账
  * 成本按仓位变动计费(每次 0.10%)
  * 分别报开发期与最终留出集, 并明确标注"叠加层最容易在留出期失效"
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

CODES = ["sh.000300", "sh.000001", "sh.000905", "sz.399006", "sz.399001"]
H = 10
TARGETS = (0.10, 0.15, 0.20)
MAXLEVS = (1.0, 2.0)
COST = 0.0010
CASH_RATE = 0.015
FIN_RATE = 0.04
HOLDOUT = 0.20


def forecasts(df, feats, h, cfg):
    """返回 (全部日期的预测波动率 dict, 开发期末日期)。"""
    bb = Blackboard()
    f = va.VolFeatureAgent(bb, cfg).build(df)
    f = va.VolLabelAgent(bb, cfg).run(f)
    dates = np.sort(f["date"].unique())
    dev_end = dates[int(len(dates) * (1 - HOLDOUT))]
    ycol = f"logyvol{h}"
    tr = f[f["date"] <= dev_end].dropna(subset=feats + [ycol])
    te = f[f["date"] > dev_end].dropna(subset=feats + [ycol]).copy()
    m = va.VolModelAgent(bb, cfg).make("hgb", "reg")
    m.fit(tr[feats], tr[ycol])
    k = float(np.mean(np.exp(np.asarray(tr[ycol]) - m.predict(tr[feats]))))
    te["p"] = np.exp(m.predict(te[feats])) * k
    # 开发期: 用 walk-forward 保证不偷看
    wf = va.VolModelAgent(bb, cfg).walk_forward(f, feats, h, "hgb", task="reg",
                                                end_date=dev_end, tag="VT")
    dev = wf["pooled"][["date", "p"]]
    return dev, te[["date", "p"]], dev_end


def backtest(date_list, fvol: dict, index_df, h, target, maxlev):
    o = index_df["open"]
    pos_of = {d: i for i, d in enumerate(index_df.index)}
    eq, prev, curve = 1.0, 0.0, []
    poss, turns, rets = [], 0.0, []
    for d in date_list[::max(1, h)]:
        i = pos_of.get(d)
        if i is None or i + 1 + h >= len(index_df):
            continue
        e0, e1 = o.iloc[i + 1], o.iloc[i + 1 + h]
        if not np.isfinite(e0) or not np.isfinite(e1) or e0 <= 0:
            continue
        rr = e1 / e0 - 1
        pos = float(np.clip(target / max(fvol.get(d, target), 1e-4), 0.0, maxlev))
        turn = abs(pos - prev)
        turns += turn
        cost = COST * turn
        fin = max(0.0, pos - 1.0) * FIN_RATE * h / 250.0
        cash = max(0.0, 1.0 - pos) * CASH_RATE * h / 250.0
        r = pos * rr - fin + cash - cost
        eq *= (1 + r)
        curve.append(eq)
        poss.append(pos)
        rets.append(r)
        prev = pos
    if len(curve) < 10:
        return None
    curve = np.array(curve)
    rets = np.array(rets)
    yrs = len(curve) * h / 250.0
    ann = float(curve[-1] ** (1 / yrs) - 1) if yrs > 0.2 else None
    mdd = float((curve / np.maximum.accumulate(curve) - 1).min())
    sd = float(np.std(rets, ddof=1))
    return {"ann_ret": None if ann is None else round(ann, 4),
            "total_ret": round(float(curve[-1] - 1), 4),
            "max_dd": round(mdd, 4),
            "sharpe": None if sd == 0 else round(float(np.mean(rets) / sd * np.sqrt(250 / h)), 3),
            "calmar": None if (ann is None or mdd == 0) else round(ann / abs(mdd), 3),
            "avg_pos": round(float(np.mean(poss)), 3),
            "turnover": round(turns, 1),
            "cost_drag": round(float(turns * COST), 4),
            "n_period": len(curve), "curve": curve}


def run_phase(tag, fmap, dfmap, dev_end):
    """fmap: code -> (dict 预测, date_list)"""
    rows = []
    for code in CODES:
        nm = qd.BS_INDICES.get(code, code)
        idx = dfmap[code]
        fvol, dlist = fmap[code]
        # 基准: 一直满仓 (令 目标/预测 = 1.0 即可得到恒定 1.0 仓位)
        b = backtest(dlist, {d: 1.0 for d in dlist}, idx, H, 1.0, 1.0)
        if b:
            b.update({"index": nm, "target": "满仓基准", "maxlev": 1.0})
            rows.append(b)
        for tgt in TARGETS:
            for lev in MAXLEVS:
                r = backtest(dlist, fvol, idx, H, tgt, lev)
                if r:
                    r.update({"index": nm, "target": f"{tgt:.0%}", "maxlev": lev})
                    rows.append(r)
    return pd.DataFrame(rows)


def main():
    cfg = va.Cfg()
    dfmap, dev_fmap, hold_fmap = {}, {}, {}
    print("=" * 100)
    print("第1条建议: 波动率仓位管理框架 (5 指数 × 3 目标波动率 × 2 档杠杆)")
    print("=" * 100)
    for code in CODES:
        nm = qd.BS_INDICES.get(code, code)
        df = qd.index_hist_bs(code)
        if df is None:
            continue
        dfmap[code] = df
        dev, hold, dev_end = forecasts(df, list(va.FEATS), H, cfg)
        dev_fmap[code] = (dict(zip(dev["date"], dev["p"])), list(dev["date"]))
        hold_fmap[code] = (dict(zip(hold["date"], hold["p"])), list(hold["date"]))
        print(f"  {nm}: 开发期预测 {len(dev)} 条, 留出期 {len(hold)} 条", flush=True)

    dev_tab = run_phase("DEV", dev_fmap, dfmap, None)
    hold_tab = run_phase("HOLD", hold_fmap, dfmap, None)

    cols = ["index", "target", "maxlev", "ann_ret", "max_dd", "sharpe", "calmar",
            "avg_pos", "turnover", "cost_drag"]
    dev_tab["maxlev"] = dev_tab["maxlev"].map(lambda x: f"{x:.1f}x")
    hold_tab["maxlev"] = hold_tab["maxlev"].map(lambda x: f"{x:.1f}x")

    print("\n" + "=" * 100)
    print("开发期 (2008 ~ 2022)")
    print("=" * 100)
    print(dev_tab[cols].to_string(index=False))
    print("\n" + "=" * 100)
    print("最终留出集 (2022-06 ~ 2026-09)")
    print("=" * 100)
    print(hold_tab[cols].to_string(index=False))

    # 汇总: 每种配置在 5 个指数上的平均
    base_dev = dev_tab[dev_tab["target"] == "满仓基准"].set_index("index")["ann_ret"].to_dict()
    base_hold = hold_tab[hold_tab["target"] == "满仓基准"].set_index("index")["ann_ret"].to_dict()

    def summarize(tab, base):
        g = tab[tab["target"] != "满仓基准"].copy()
        g["beat"] = [1 if a > base.get(ix, -9) else 0
                      for a, ix in zip(g["ann_ret"], g["index"])]
        out = g.groupby(["target", "maxlev"]).agg(
            平均年化=("ann_ret", "mean"), 平均回撤=("max_dd", "mean"),
            平均夏普=("sharpe", "mean"), 平均仓位=("avg_pos", "mean"),
            跑赢满仓的指数数=("beat", "sum")).reset_index()
        return out

    s_dev = summarize(dev_tab, base_dev)
    s_hold = summarize(hold_tab, base_hold)
    print("\n" + "=" * 100)
    print("汇总(5 个指数平均) —— 开发期")
    print("=" * 100)
    print(s_dev.to_string(index=False))
    print("\n" + "=" * 100)
    print("汇总(5 个指数平均) —— 最终留出集")
    print("=" * 100)
    print(s_hold.to_string(index=False))

    # 出图
    setup_font()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    for ax, s, ttl in ((axes[0], s_dev, "开发期(2008~2022)"),
                       (axes[1], s_hold, "最终留出集(2022-06 起)")):
        lab = [f"{r.target}\n{r.maxlev}" for r in s.itertuples()]
        x = np.arange(len(s))
        ax.bar(x - 0.2, s["平均年化"] * 100, width=0.38, color="#4c72b0", label="平均年化%")
        ax2 = ax.twinx()
        ax2.plot(x, s["跑赢满仓的指数数"], "o--", color="#c44e52", label="跑赢满仓的指数数(0-5)")
        ax.set_xticks(x)
        ax.set_xticklabels(lab, fontsize=8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"波动率目标仓位: 各配置表现 {ttl}")
        ax.set_ylabel("平均年化收益 %")
        ax2.set_ylabel("跑赢满仓的指数数")
        ax2.set_ylim(0, 5.4)
        ax.legend(fontsize=8, loc="upper left")
        ax2.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "vol_target.png", dpi=130)
    plt.close(fig)

    (OUT / "vol_target.json").write_text(json.dumps(
        {"dev": dev_tab.to_dict("records"), "holdout": hold_tab.to_dict("records"),
         "summary_dev": s_dev.to_dict("records"), "summary_holdout": s_hold.to_dict("records")},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n图表: {OUT / 'vol_target.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
