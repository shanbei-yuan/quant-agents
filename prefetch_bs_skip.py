#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扩大股票池: 用 Baostock 下载成交额前 800 只主板股的 2015 年至今后复权日线。
   已缓存的(起点早于 2015)会跳过, 所以可以反复运行续传。"""
import sys
import quant_data as qd

N = int(sys.argv[1]) if len(sys.argv) > 1 else 800
pool = qd.build_universe(top_n=N)
print(f"股票池前 {len(pool)} 只, 检查缓存后下载 ...", flush=True)
ok = qd.download_all_bs(pool, start="2015-01-01", sleep=0.03, force=False)
print(f"本轮新下载 {ok} 只")
