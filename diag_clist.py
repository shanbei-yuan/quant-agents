#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 3: clist 接口在什么参数组合下可用"""
import subprocess, time

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
FS4 = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
FS2 = "m:0+t:6,m:1+t:2"
F = "f12,f13,f14,f2,f3,f6,f9,f20,f21,f23"

def get(url):
    cmd = ["curl", "-s", "-m", "20", "--compressed", "-H", f"User-Agent: {UA}", url]
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if p.returncode != 0:
        return f"RC={p.returncode}"
    return p.stdout[:110].replace("\n", "|") if p.stdout.strip() else "EMPTY"

HOSTS = ["push2.eastmoney.com", "82.push2.eastmoney.com", "push2delay.eastmoney.com"]
CASES = [
    ("host=push2 pz=20  fs4", f"https://{HOSTS[0]}/api/qt/clist/get?pn=1&pz=20&po=1&np=1&fltt=2&invt=2&fid=f6&fs={FS4}&fields={F}"),
    ("host=push2 pz=100 fs4", f"https://{HOSTS[0]}/api/qt/clist/get?pn=1&pz=100&po=1&np=1&fltt=2&invt=2&fid=f6&fs={FS4}&fields={F}"),
    ("host=push2 pz=1000 fs4", f"https://{HOSTS[0]}/api/qt/clist/get?pn=1&pz=1000&po=1&np=1&fltt=2&invt=2&fid=f6&fs={FS4}&fields={F}"),
    ("host=82   pz=1000 fs4", f"https://{HOSTS[1]}/api/qt/clist/get?pn=1&pz=1000&po=1&np=1&fltt=2&invt=2&fid=f6&fs={FS4}&fields={F}"),
    ("host=delay pz=1000 fs4", f"https://{HOSTS[2]}/api/qt/clist/get?pn=1&pz=1000&po=1&np=1&fltt=2&invt=2&fid=f6&fs={FS4}&fields={F}"),
    ("host=push2 pz=500 fs4+nodash", f"https://{HOSTS[0]}/api/qt/clist/get?pn=1&pz=500&po=1&np=1&fltt=2&invt=2&fid=f6&fs={FS4}&fields={F}"),
    ("host=push2 pz=100 fs2", f"https://{HOSTS[0]}/api/qt/clist/get?pn=1&pz=100&po=1&np=1&fltt=2&invt=2&fid=f6&fs={FS2}&fields={F}"),
]
for name, url in CASES:
    print(f"{name:<28} -> {get(url)}")
    time.sleep(1.5)
