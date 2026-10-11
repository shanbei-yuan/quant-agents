#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断: 东方财富 clist 接口为什么被拒"""
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
FS = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
FIELDS = "f12,f13,f14,f2,f3,f6"
BASE = "https://push2.eastmoney.com/api/qt/clist/get"

def t(name, fn):
    try:
        r = fn()
        print(f"[OK ] {name}: {r[:120]}")
    except Exception as e:
        print(f"[ERR] {name}: {type(e).__name__} {e}")

# 1) params dict (当前写法)
def v1():
    r = requests.get(BASE, headers=UA, timeout=15, params={
        "pn": 1, "pz": 3, "po": 1, "np": 1, "fltt": 2, "invt": 2, "fid": "f6",
        "fs": FS, "fields": FIELDS})
    return r.text

# 2) 手工拼 URL, + 保持字面量
def v2():
    url = (f"{BASE}?pn=1&pz=3&po=1&np=1&fltt=2&invt=2&fid=f6&fs={FS}&fields={FIELDS}")
    r = requests.get(url, headers=UA, timeout=15)
    return r.text

# 3) 手工拼 URL + Referer
def v3():
    h = dict(UA); h["Referer"] = "https://quote.eastmoney.com/"
    url = (f"{BASE}?pn=1&pz=3&po=1&np=1&fltt=2&invt=2&fid=f6&fs={FS}&fields={FIELDS}")
    r = requests.get(url, headers=h, timeout=15)
    return r.text

# 4) 不带 fs 参数
def v4():
    r = requests.get(BASE, headers=UA, timeout=15, params={
        "pn": 1, "pz": 3, "po": 1, "np": 1, "fltt": 2, "invt": 2, "fid": "f6",
        "fields": FIELDS})
    return r.text

for n, f in [("1 params dict", v1), ("2 手工URL", v2), ("3 手工URL+Referer", v3), ("4 无fs", v4)]:
    t(n, f)
