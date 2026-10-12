#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refresh_data.py —— 增量刷新行情缓存（后复权，口径与历史一致）
=============================================================
只补齐"最近 N 天"的日线并合并进 cache/kline_*.csv，不重抓全历史。
必须用 Baostock adjustflag=1(后复权)：它的后复权基准是累计的，所以新抓的
最近几天与历史数据在同一标尺上，可以安全拼接。

用法(分片并行)：
    python refresh_data.py <SHARD> <NSHARD> [LOOKBACK_DAYS]
或直接跑 run_refresh.sh（8 分片并行）。
"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

import quant_data as qd

CACHE = Path(__file__).resolve().parent / "cache"
INDICES = ["sh.000300", "sh.000001", "sh.000905", "sz.399006", "sz.399001"]


def refresh_stock(bs, code6: str, mkt: int, lookback: int) -> int:
    f = CACHE / f"kline_{code6}.csv"
    old = None
    if f.exists():
        try:
            old = pd.read_csv(f, parse_dates=["date"]).set_index("date").sort_index()
        except Exception:                        # noqa: BLE001
            old = None
    if old is None or len(old) < 100:
        return 0
    start = (old.index[-1] - timedelta(days=10)).date().isoformat()
    sym = ("sh." if mkt == 1 else "sz.") + str(code6)
    new = qd.stock_hist_bs(bs, sym, start=start)
    if new is None or new.empty:
        return 0
    merged = pd.concat([old, new])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    added = len(merged) - len(old)
    if added > 0:
        merged.to_csv(f, encoding="utf-8")
    return added


def main():
    shard = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nshard = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    lookback = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    import baostock as bs
    lg = bs.login()
    if lg.error_code != "0":
        print(f"baostock 登录失败: {lg.error_msg}")
        return 1

    # 指数: 只让 0 号分片做
    if shard == 0:
        for code in INDICES:
            f = CACHE / f"index_{code.replace('.', '_')}.csv"
            try:
                old = pd.read_csv(f, parse_dates=["date"]).set_index("date").sort_index()
            except Exception:                    # noqa: BLE001
                continue
            start = (old.index[-1] - timedelta(days=10)).date().isoformat()
            rs = bs.query_history_k_data_plus(
                code, "date,open,high,low,close,volume,amount",
                start_date=start, end_date=date.today().isoformat(),
                frequency="d", adjustflag="3")
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                continue
            nd = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close",
                                             "volume", "amount"])
            nd["date"] = pd.to_datetime(nd["date"])
            for c in ["open", "high", "low", "close", "volume", "amount"]:
                nd[c] = pd.to_numeric(nd[c], errors="coerce")
            nd = nd.dropna(subset=["close"]).set_index("date").sort_index()
            nd["pct"] = nd["close"].pct_change() * 100
            m = pd.concat([old, nd[old.columns.intersection(nd.columns)]])
            m = m[~m.index.duplicated(keep="last")].sort_index()
            m.to_csv(f, encoding="utf-8")
            print(f"  指数 {code}: +{len(m)-len(old)} 根 → 至 {m.index[-1].date()}",
                  flush=True)

    snap = qd.snapshot()
    pool = snap[snap["code"].astype(str).str.match(
        r"^(600|601|603|605|000|001|002|003)\d{3}$")].reset_index(drop=True)
    sub = pool.iloc[shard::nshard]
    t0, tot = time.time(), 0
    for i, r in sub.iterrows():
        try:
            tot += refresh_stock(bs, str(r["code"]), int(r["mkt"]), lookback)
        except Exception as e:                   # noqa: BLE001
            print(f"   [err] {r['code']}: {e}", flush=True)
        if (i // 1) % 50 == 0 and i:
            print(f"   shard{shard}: {i}/{len(sub)} 新增 {tot} 根 "
                  f"用时 {time.time()-t0:.0f}s", flush=True)
        time.sleep(0.05)
    bs.logout()
    print(f"   shard{shard} 完成: 新增 {tot} 根, 用时 {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
