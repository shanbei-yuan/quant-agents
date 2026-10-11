#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""查看 vol_target.json 的满仓基准与最优配置对比"""
import json
from pathlib import Path

p = Path(__file__).resolve().parent / "out" / "vol_target.json"
d = json.loads(p.read_text(encoding="utf-8"))

for phase, title in (("dev", "开发期 2008~2022"), ("holdout", "最终留出集 2022-06 起")):
    print("=" * 78)
    print(title)
    print("=" * 78)
    print(f"{'指数':<10} {'满仓年化':>10} {'满仓回撤':>10} {'满仓夏普':>10} | "
          f"{'10%仓位年化':>12} {'10%回撤':>10} {'10%夏普':>10}")
    base = {r["index"]: r for r in d[phase] if r["target"] == "满仓基准"}
    t10 = {r["index"]: r for r in d[phase] if r["target"] == "10%" and r["maxlev"] == "1.0x"}
    for nm, b in base.items():
        a = t10.get(nm)
        if not a:
            continue
        print(f"{nm:<10} {b['ann_ret']:>10.2%} {b['max_dd']:>10.2%} {b['sharpe']:>10} | "
              f"{a['ann_ret']:>12.2%} {a['max_dd']:>10.2%} {a['sharpe']:>10}")
    nb = len(base)
    avg_b_ann = sum(r["ann_ret"] for r in base.values()) / nb
    avg_b_dd = sum(r["max_dd"] for r in base.values()) / nb
    avg_b_sh = sum(r["sharpe"] for r in base.values()) / nb
    avg10 = [r for r in d[phase] if r["target"] == "10%" and r["maxlev"] == "1.0x"]
    avg_a_ann = sum(r["ann_ret"] for r in avg10) / len(avg10)
    avg_a_dd = sum(r["max_dd"] for r in avg10) / len(avg10)
    avg_a_sh = sum(r["sharpe"] for r in avg10) / len(avg10)
    print("-" * 78)
    print(f"{'平均':<10} {avg_b_ann:>10.2%} {avg_b_dd:>10.2%} {avg_b_sh:>10.3f} | "
          f"{avg_a_ann:>12.2%} {avg_a_dd:>10.2%} {avg_a_sh:>10.3f}")
    print()
