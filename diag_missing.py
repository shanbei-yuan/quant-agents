#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断: 为什么这 81 只股票两个源都拿不到数据"""
import subprocess

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


def curl(url, timeout=25):
    p = subprocess.run(["curl", "-s", "-m", str(timeout), "--compressed",
                        "-H", f"User-Agent: {UA}", url],
                       capture_output=True, text=True, errors="replace")
    return p.returncode, p.stdout


CODES = [("sh", "601166", "兴业银行"), ("sz", "000651", "格力电器"),
         ("sh", "600000", "浦发银行(已知可用的对照)")]

for pref, code, name in CODES:
    sym = pref + code
    mkt = "1" if pref == "sh" else "0"
    print("=" * 70)
    print(f"### {name} {code}")
    u1 = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?"
          f"param={sym},day,2024-01-01,2026-10-10,800,qfq")
    rc, out = curl(u1)
    print(f"[腾讯] rc={rc} len={len(out)}  {out[:180]}")
    u2 = (f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={mkt}.{code}"
          f"&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53&klt=101&fqt=1"
          f"&beg=20180101&end=20261231")
    rc, out = curl(u2)
    print(f"[东财] rc={rc} len={len(out)}  {out[:180]}")
