#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, subprocess, urllib.request
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
FS = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
FIELDS = "f12,f13,f14,f2,f3,f6"
CLIST = f"https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=3&po=1&np=1&fltt=2&invt=2&fid=f6&fs={FS}&fields={FIELDS}"
KLINE = "https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.600000&fields1=f1&fields2=f51,f53&klt=101&fqt=1&end=20500101&lmt=3"

def t(name, fn):
    try:
        print(f"[OK ] {name}: {str(fn())[:130]}")
    except Exception as e:
        print(f"[ERR] {name}: {type(e).__name__} {e}")

def urllib_clist():
    req = urllib.request.Request(CLIST, headers=UA)
    return urllib.request.urlopen(req, timeout=15).read().decode()

def requests_kline():
    return requests.get(KLINE, headers=UA, timeout=15).text

def urllib_kline():
    req = urllib.request.Request(KLINE, headers=UA)
    return urllib.request.urlopen(req, timeout=15).read().decode()

def curl_clist():
    return subprocess.run(["curl", "-s", "-m", "20", CLIST],
                          capture_output=True, text=True, check=True).stdout

def curl_kline():
    return subprocess.run(["curl", "-s", "-m", "20", KLINE],
                          capture_output=True, text=True, check=True).stdout

t("urllib -> clist(push2)", urllib_clist)
t("requests -> kline(push2his)", requests_kline)
t("urllib -> kline(push2his)", urllib_kline)
t("curl -> clist(push2)", curl_clist)
t("curl -> kline(push2his)", curl_kline)
