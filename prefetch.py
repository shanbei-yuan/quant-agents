#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预下载: 把训练用的股票池历史日线全部落到 cache/, 之后训练 0 联网。"""
import sys
import quant_data as qd

n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
print(f"源={qd.probe_source()}")
pool = qd.build_universe(top_n=n)
print(f"股票池 {len(pool)} 只, 开始预下载...")
data = qd.download_all(pool)
print(f"完成: {len(data)} 只已缓存")
bad = [c for c in pool["code"] if c not in data]
print(f"未拿到数据: {len(bad)} 只 {bad[:20]}")
