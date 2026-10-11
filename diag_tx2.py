#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 5: 腾讯 fqkline 单次历史深度 + 批量行情上限"""
import json, subprocess, time

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

def curl(url, binary=False):
    cmd = ["curl", "-s", "-m", "30", "--compressed", "-H", f"User-Agent: {UA}", url]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        return None
    return p.stdout if binary else p.stdout.decode("utf-8", "replace")

def kline_probe(label, code, beg, end, count):
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,{beg},{end},{count},qfq"
    txt = curl(url)
    if txt is None:
        print(f"{label:<42} -> RC!=0"); return
    try:
        j = json.loads(txt)
    except Exception:
        print(f"{label:<42} -> 非JSON: {txt[:80]}"); return
    if j.get("msg"):
        print(f"{label:<42} -> msg={j['msg']}"); return
    d = j.get("data", {}).get(code, {})
    arr = d.get("qfqday") or d.get("day") or []
    if not arr:
        print(f"{label:<42} -> 空"); return
    print(f"{label:<42} -> {len(arr):>5} 根  {arr[0][0]} ~ {arr[-1][0]}  字段数={len(arr[0])}  样例={arr[0]}")

code = "sh600000"
for cnt in (640, 800, 1000, 1200, 2000):
    kline_probe(f"sh600000 2018-2026 count={cnt}", code, "2018-01-01", "2026-10-10", cnt)
    time.sleep(2.5)

kline_probe("sh600000 无起止 count=3000", code, "", "", 3000); time.sleep(2.5)
kline_probe("sz000001 2018-2026 count=1000", "sz000001", "2018-01-01", "2026-10-10", 1000); time.sleep(2.5)

# 批量行情上限
for n in (50, 200, 600):
    codes = []
    for i in range(n):
        if i % 2 == 0:
            codes.append(f"sh60{i//2+1:04d}" if i//2 < 1000 else "")
        else:
            codes.append(f"sz00{i//2+1:04d}")
    codes = [c for c in codes if c]
    url = "https://qt.gtimg.cn/q=" + ",".join(codes)
    txt = curl(url, binary=True)
    if txt is None:
        print(f"批量行情 n={n:<5} -> RC!=0")
    else:
        s = txt.decode("gbk", "replace")
        ok = s.count('v_')  # 返回条数
        print(f"批量行情 n={n:<5} -> 返回 {ok} 条, 字节 {len(txt)}")
    time.sleep(2.5)
