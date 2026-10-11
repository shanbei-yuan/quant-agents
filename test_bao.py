#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试 Baostock 能否拿到长期指数历史（方向 D 的关键）"""
import sys

try:
    import baostock as bs
    print("import baostock OK")
except Exception as e:                       # noqa: BLE001
    print(f"NOT_INSTALLED: {e}")
    sys.exit(1)

lg = bs.login()
print(f"login: code={lg.error_code} msg={lg.error_msg}")

for code, name in [("sh.000300", "沪深300"), ("sh.000001", "上证指数")]:
    rs = bs.query_history_k_data_plus(
        code, "date,open,high,low,close,volume,amount",
        start_date="2005-01-01", end_date="2026-10-11",
        frequency="d", adjustflag="3")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if rows:
        print(f"{code} {name}: {len(rows)} 根  {rows[0][0]} ~ {rows[-1][0]}")
        print(f"    首行={rows[0]}  末行={rows[-1]}")
    else:
        print(f"{code} {name}: 空  err={rs.error_code} {rs.error_msg}")

bs.logout()
