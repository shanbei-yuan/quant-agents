#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补齐缺失股票的历史日线。
策略: 优先用东方财富(一次请求拿全 2018 至今), 失败再退回腾讯翻页。
腾讯在连续请求后会明显降速甚至返回空, 所以要错开使用。
"""
import time
from pathlib import Path

import pandas as pd

import quant_data as qd

TOP = 200


def has_data(code: str) -> int:
    f = qd.CACHE / f"kline_{code}.csv"
    if not f.exists():
        return 0
    try:
        d = pd.read_csv(f, parse_dates=["date"])
    except Exception:                          # noqa: BLE001
        return 0
    return len(d)


pool = qd.build_universe(top_n=TOP)
miss = [r for _, r in pool.iterrows() if has_data(r["code"]) < 250]
print(f"股票池 {len(pool)} 只, 缺失 {len(miss)} 只, 开始补齐", flush=True)

ok, t0 = 0, time.time()
for i, r in enumerate(miss, 1):
    mkt, code = int(r["mkt"]), r["code"]
    df = None
    try:
        df = qd._kline_eastmoney(mkt, code, "2018-01-01", "2026-12-31")
    except Exception:                          # noqa: BLE001
        df = None
    src = "em"
    if df is None or len(df) < 250:
        try:
            df = qd._kline_tencent(qd._sym(mkt, code))
        except Exception:                      # noqa: BLE001
            df = None
        src = "tx"
    if df is not None and len(df) >= 250:
        df.to_csv(qd.CACHE / f"kline_{code}.csv", encoding="utf-8")
        ok += 1
    print(f"  [{i}/{len(miss)}] {code} {r['name']}: "
          f"{0 if df is None else len(df)} 根 ({src})  已补齐 {ok}  用时 {time.time()-t0:.0f}s",
          flush=True)
    time.sleep(0.6)

print(f"补齐完成: {ok}/{len(miss)} 只, 总用时 {time.time()-t0:.0f}s", flush=True)
