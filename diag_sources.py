#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 2: 找到当前可用的行情数据源"""
import subprocess
import time

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

CASES = [
    ("eastmoney-clist",
     "https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=2&po=1&np=1&fltt=2&invt=2&fid=f6&fs=m:0+t:6,m:1+t:2&fields=f12,f14,f2,f6",
     None),
    ("eastmoney-kline",
     "https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.600000&fields1=f1&fields2=f51,f53&klt=101&fqt=1&end=20500101&lmt=3",
     None),
    ("eastmoney-82kline",
     "https://82.push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.600000&fields1=f1&fields2=f51,f53&klt=101&fqt=1&end=20500101&lmt=3",
     None),
    ("tencent-realtime",
     "https://qt.gtimg.cn/q=sh600000",
     None),
    ("tencent-qfq-kline",
     "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh600000,day,,,30,qfq",
     None),
    ("netease-csv",
     "http://quotes.money.163.com/service/chddata.html?code=0600000&start=20260901&end=20261010&fields=TCLOSE;HIGH;LOW;TOPEN;VOTURNOVER",
     None),
    ("sina-realtime",
     "https://hq.sinajs.cn/list=sh600000",
     "https://finance.sina.com.cn"),
]


def curl(url, referer=None, timeout=20):
    cmd = ["curl", "-s", "-m", str(timeout), "--compressed",
           "-H", f"User-Agent: {UA}"]
    if referer:
        cmd += ["-H", f"Referer: {referer}"]
    cmd.append(url)
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout


for name, url, ref in CASES:
    try:
        rc, out = curl(url, ref)
        if rc != 0:
            print(f"[ERR {rc:>3}] {name}: curl 失败")
        elif not out.strip():
            print(f"[EMPTY  ] {name}: 返回空")
        else:
            head = out.replace("\n", "|")[:150]
            print(f"[OK     ] {name}: {head}")
    except Exception as e:                     # noqa: BLE001
        print(f"[EXC    ] {name}: {e}")
    time.sleep(1.2)
