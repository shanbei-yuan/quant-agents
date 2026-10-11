#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 Baostock 预下载个股历史(后复权, 2015 年起), 覆盖旧的腾讯缓存。
   下游 train_agents.py 读的是同一个 cache/kline_*.csv, 无需改动。"""
import sys
import time

import quant_data as qd

TOP = int(sys.argv[1]) if len(sys.argv) > 1 else 300
START = sys.argv[2] if len(sys.argv) > 2 else "2015-01-01"

print(f"股票池取今日成交额前 {TOP} (主板/非ST)")
pool = qd.build_universe(top_n=TOP)
print(f"股票池 {len(pool)} 只, 开始用 Baostock 下载 (后复权, 自 {START}) ...", flush=True)
t0 = time.time()
ok = qd.download_all_bs(pool, start=START, force=True)
print(f"完成: {ok}/{len(pool)} 只, 总用时 {time.time()-t0:.0f}s")
