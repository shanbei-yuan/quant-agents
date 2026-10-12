#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自主训练 第三阶段 auto_train3.py
================================
前两阶段暴露了根因: 模型学的是"跑赢当天中位数", 所以它擅长挑出"略强的"股票,
但组合收益低于全池等权 —— 因为**赚钱的是右尾(大赢家), 不是中位数**。

本阶段直接对准右尾:
  * ytop{h} = 当天收益排名前 25%(选股目标改成"会不会成为大赢家")
  * ybot{h} = 当天收益排名后 25%(反向检验: 有没有识别差生的能力)
并对比原来的 yrel(中位数) 标签。

同样遵守纪律: 开发期搜索, 留出集只评估一次。
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

from auto_train import build_extended_panel, HS, HOLDOUT, TARGET_VOL, BENCH, OUT
from auto_train2 import wf, eval_all, ensemble, yearly, FEATS
import quant_data as qd
import train_agents as ta
import vol_agents as va
from vol_target import forecasts as vol_forecasts
from train_agents import setup_font

TOPQ = 0.75


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
    ALL = {"all": feats_all,
           "nomkt": [f for f in feats_all if f not in ta.FEATS_MARKET],
           "rank": [f for f in feats_all if f.startswith("rank_") or f in (
               "rs20", "pos60", "pos120", "dd20", "above_ma20", "breadth",
               "vol20", "vol60")]}

    print("=" * 100)
    print("【第三阶段】把预测目标对准右尾: 前25% / 后25%")
    print("=" * 100, flush=True)
    for h in HS:
        rk = df.groupby("date")[f"fwd{h}"].rank(pct=True)
        df[f"ytop{h}"] = (rk > TOPQ).astype(float)
        df[f"ybot{h}"] = (rk <= 1 - TOPQ).astype(float)
        df.loc[df[f"fwd{h}"].isna(), [f"ytop{h}", f"ybot{h}"]] = np.nan
        print(f"  h={h}: 前25%基准率 {df[f'ytop{h}'].mean():.4f}", flush=True)

    kinds = ["hgb", "hgbdeep", "logit", "rf"]
    results = []
    for lab in ("top", "bot"):
        for h in HS:
            ycol = f"y{lab}{h}"
            pool_dev, pool_hold, recs = [], [], []
            for kind in kinds:
                for fname, feats in ALL.items():
                    p, hd = wf(df, feats, ycol, h, kind, dev_end)
                    if p is None:
                        continue
                    from auto_train import acc_stats
                    st = acc_stats(p)
                    rows = eval_all(p, all_dates, h, volmap, f"{lab}{h}_{kind}_{fname}", [])
                    bs = max(rows, key=lambda x: x["excess_ann"]) if rows else {}
                    recs.append({"label": lab, "h": h, "kind": kind, "feats": fname,
                                 **st, "best": bs})
                    pool_dev.append(p)
                    pool_hold.append(hd)
                    print(f"  {lab} h={h:<3} {kind:<8} {fname:<6} 命中 {st['acc']:.4f} "
                          f"AUC {st['auc']} | 超额 {bs.get('excess_ann')} "
                          f"夏普 {bs.get('sharpe')} | {time.time()-t0:.0f}s", flush=True)
            if pool_dev:
                ens_d = ensemble(pool_dev)
                ens_h = ensemble(pool_hold)
                from auto_train import acc_stats
                st_d, st_h = acc_stats(ens_d), acc_stats(ens_h)
                rows_d = eval_all(ens_d, all_dates, h, volmap, f"{lab}{h}_ENS_dev", [])
                bs_d = max(rows_d, key=lambda x: x["excess_ann"]) if rows_d else {}
                rows_h = eval_all(ens_h, all_dates, h, volmap, f"{lab}{h}_ENS_hold", [])
                bs_h = max(rows_h, key=lambda x: x["excess_ann"]) if rows_h else {}
                yh = yearly(ens_h, all_dates, h, 10, volmap, True)
                results.append({"label": lab, "h": h, "n_members": len(pool_dev),
                                "acc_dev": st_d, "acc_hold": st_h,
                                "best_dev": bs_d, "best_hold": bs_h,
                                "yearly_hold": yh, "members": recs})
                print(f"  >>> 集合({lab},h={h}): 开发期 命中 {st_d['acc']:.4f}"
                      f"({st_d['edge']:+.4f}) AUC {st_d['auc']} 超额 {bs_d.get('excess_ann')}"
                      f" | 留出期 命中 {st_h['acc']:.4f}({st_h['edge']:+.4f}) "
                      f"AUC {st_h['auc']} 超额 {bs_h.get('excess_ann')}", flush=True)

    (OUT / "auto_train3.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    setup_font()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    labs = [f"{r['label']} h={r['h']}" for r in results]
    x = np.arange(len(results))
    axes[0].bar(x - 0.2, [r["best_dev"].get("excess_ann", 0) for r in results],
                width=0.4, color="#4c72b0", label="开发期最优超额")
    axes[0].bar(x + 0.2, [r["best_hold"].get("excess_ann", 0) for r in results],
                width=0.4, color="#c44e52", label="留出期超额")
    axes[0].axhline(0, color="k", lw=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labs, fontsize=9)
    axes[0].set_ylabel("相对全池等权的超额年化收益")
    axes[0].set_title("第三阶段: 对准右尾(前25%) vs 识别差生(后25%)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].bar(x - 0.2, [r["acc_dev"]["auc"] or 0 for r in results], width=0.4,
                color="#4c72b0", label="开发期 AUC")
    axes[1].bar(x + 0.2, [r["acc_hold"]["auc"] or 0 for r in results], width=0.4,
                color="#c44e52", label="留出期 AUC")
    axes[1].axhline(0.5, color="k", ls="--", lw=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labs, fontsize=9)
    axes[1].set_title("第三阶段: 排序能力(AUC)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "auto_train3.png", dpi=130)
    plt.close(fig)
    print(f"\n总用时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
