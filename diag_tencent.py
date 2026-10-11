#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 4: 腾讯/网易数据源能力 + 东财冷却后重试"""
import json, subprocess, time

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

def curl(url, referer=None, binary=False):
    cmd = ["curl", "-s", "-m", "25", "--compressed", "-H", f"User-Agent: {UA}"]
    if referer:
        cmd += ["-H", f"Referer: {referer}"]
    cmd.append(url)
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        return None, f"RC={p.returncode} {p.stderr.decode('utf-8','replace')[:80]}"
    raw = p.stdout
    if binary:
        return raw, None
    return raw.decode("utf-8", "replace"), None

def show(name, url, referer=None, binary=False, decode=None):
    txt, err = curl(url, referer, binary)
    if err:
        print(f"[ERR] {name}: {err}"); return
    if binary and txt is not None:
        txt = txt.decode(decode or "gbk", "replace")
    if not txt or not txt.strip():
        print(f"[EMPTY] {name}"); return
    print(f"[OK ] {name}\n      {txt[:230].strip()}")
    print()

# 1 腾讯 复权日线: 大区间 + 大 count 能拿多少根?
show("tencent qfq 2018->now count3000",
     "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh600000,day,2018-01-01,2026-10-10,3000,qfq")
time.sleep(2)

# 2 腾讯 复权日线: 不带日期只给 count
show("tencent qfq count 640",
     "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh600000,day,,,640,qfq")
time.sleep(2)

# 3 腾讯 批量实时(GBK)
show("tencent batch quote (gbk)",
     "https://qt.gtimg.cn/q=sh600000,sz000001,sz000002", binary=True, decode="gbk")
time.sleep(2)

# 4 腾讯 排行榜(能否拿来构建股票池)
show("tencent rank board",
     "https://proxy.finance.qq.com/cgi/cgi-bin/rank/pt/getRank?board_type=hs&sort_type=price&direct=down&offset=0&count=20")
time.sleep(2)

# 5 网易 CSV 全历史
show("netease chddata csv",
     "http://quotes.money.163.com/service/chddata.html?code=0600000&start=20180101&end=20261010&fields=TCLOSE;HIGH;LOW;TOPEN;VOTURNOVER;VATURNOVER")
time.sleep(3)

# 6 东财快照 (冷却后重试)
show("eastmoney snapshot pz=100",
     "https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=100&po=1&np=1&fltt=2&invt=2&fid=f6&fs=m:0+t:6,m:1+t:2&fields=f12,f14,f2,f6")
