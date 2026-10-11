#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baostock 分片并行下载: 用法 python prefetch_bs_par.py <TOP> <SHARD> <NSHARD> [START]"""
import sys
import time

import quant_data as qd

TOP = int(sys.argv[1])
SHARD = int(sys.argv[2])
NSHARD = int(sys.argv[3])
START = sys.argv[4] if len(sys.argv) > 4 else "2015-01-01"

pool = qd.build_universe(top_n=TOP)
sub = pool.iloc[SHARD::NSHARD].reset_index(drop=True)
print(f"[shard {SHARD}/{NSHARD}] {len(sub)} 只, 开始", flush=True)
t0 = time.time()
ok = qd.download_all_bs(sub, start=START, sleep=0.03, force=True)
print(f"[shard {SHARD}/{NSHARD}] 完成 {ok}/{len(sub)}, 用时 {time.time()-t0:.0f}s", flush=True)
