#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据层 quant_data.py  v2  (腾讯主源 / 东财备用)
==============================================
设计要点
  1. 不用 python-requests: 其 TLS 指纹会被行情站 WAF 拒绝(RemoteDisconnected / rc=56)。
     统一走 curl 子进程, 并带指数退避重试。
  2. 启动时做一次"源探测": 东财能通就用东财(一次请求拿全历史, 最省),
     不通就全程走腾讯(单次上限 800 根, 自动翻页拼接)。
  3. 全量落盘缓存: 第二次运行 0 联网。
  4. 股票池构建: 腾讯批量行情按 200 只/批扫全主板代码段, 不依赖任何第三方列表文件。

接口
    snapshot(force=False)        -> 全市场主板快照 DataFrame
    build_universe(top_n, ...)   -> 训练股票池
    kline_hist(sym, ...)         -> 单只日线(前复权) DataFrame, 带缓存
    benchmark_hist()             -> 沪深300 日线
    download_all(pool, ...)      -> {code: DataFrame}
    ACTIVE_SOURCE                -> 'eastmoney' | 'tencent'
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

EM_KLINE = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
EM_CLIST = "https://push2.eastmoney.com/api/qt/clist/get"
TX_KLINE = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
TX_QUOTE = "https://qt.gtimg.cn/q="

KL_COLS = ["date", "open", "close", "high", "low", "volume",
           "amount", "amplitude", "pct", "change", "turnover"]
EM_F2 = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"

BEG = "2018-01-01"
TX_MAX = 800                      # 腾讯单次上限
OHLCV = ["open", "close", "high", "low", "volume", "amount"]

ACTIVE_SOURCE = None
_STATS = {"req": 0, "fail": 0}


# ------------------------------------------------------------------ HTTP
def _curl(url: str, binary: bool = False, timeout: int = 30, retries: int = 4):
    for i in range(retries):
        _STATS["req"] += 1
        cmd = ["curl", "-s", "-m", str(timeout), "--compressed",
               "-H", f"User-Agent: {UA}", url]
        try:
            p = subprocess.run(cmd, capture_output=True, timeout=timeout + 10)
        except subprocess.TimeoutExpired:
            time.sleep(1.0 + 1.5 * i)
            continue
        if p.returncode == 0 and p.stdout and p.stdout.strip():
            return p.stdout if binary else p.stdout.decode("utf-8", "replace")
        _STATS["fail"] += 1
        time.sleep(1.0 + 1.5 * i)      # 1s / 2.5s / 4s
    return None


# ------------------------------------------------------------------ 转换
def _sym(mkt: int, code: str) -> str:
    return ("sh" if int(mkt) == 1 else "sz") + str(code)


def _parse_em(txt: str):
    j = json.loads(txt)
    kl = (j.get("data") or {}).get("klines") or []
    if not kl:
        return None
    df = pd.DataFrame([x.split(",") for x in kl], columns=KL_COLS)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in KL_COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    return df[OHLCV]


def _parse_tx(txt: str, sym: str):
    j = json.loads(txt)
    d = (j.get("data") or {}).get(sym) or {}
    arr = d.get("qfqday") or d.get("day") or []
    if not arr:
        return None
    rows = [r[:6] for r in arr if len(r) >= 6]
    df = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    df["amount"] = df["close"] * df["volume"] * 100.0
    return df[OHLCV]


# ------------------------------------------------------------------ 源探测
def probe_source(force: bool = False) -> str:
    global ACTIVE_SOURCE
    if ACTIVE_SOURCE and not force:
        return ACTIVE_SOURCE
    f = CACHE / "source.json"
    if f.exists() and not force:
        try:
            a = json.loads(f.read_text(encoding="utf-8"))
            if time.time() - a.get("ts", 0) < 3600:      # 1 小时内复用
                ACTIVE_SOURCE = a["src"]
                return ACTIVE_SOURCE
        except Exception:                    # noqa: BLE001
            pass
    txt = _curl(f"{EM_KLINE}?secid=1.600000&fields1=f1&fields2=f51,f53"
                f"&klt=101&fqt=1&end=20500101&lmt=3", retries=2, timeout=15)
    src = "eastmoney" if (txt and '"klines"' in txt and '"rc":0' in txt) else "tencent"
    ACTIVE_SOURCE = src
    f.write_text(json.dumps({"src": src, "ts": time.time()}), encoding="utf-8")
    return src


# ------------------------------------------------------------------ 快照
def _snapshot_eastmoney(pz: int = 1000) -> pd.DataFrame:
    fields = "f12,f13,f14,f2,f3,f5,f6,f8,f9,f20,f21,f23"
    rows, pn = [], 1
    while True:
        url = (f"{EM_CLIST}?pn={pn}&pz={pz}&po=1&np=1&fltt=2&invt=2&fid=f6"
               f"&fs=m:0+t:6,m:1+t:2&fields={fields}&_={int(time.time()*1000)}")
        txt = _curl(url, retries=2)
        if not txt:
            break
        j = json.loads(txt)
        d = j.get("data") or {}
        diff = d.get("diff") or []
        if not diff:
            break
        rows += diff
        if len(rows) >= int(d.get("total") or 0) or len(diff) < pz:
            break
        pn += 1
        time.sleep(0.6)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns={
        "f12": "code", "f13": "mkt", "f14": "name", "f2": "price", "f3": "pct",
        "f5": "volume", "f6": "amount", "f8": "turnover", "f9": "pe",
        "f20": "mktcap", "f21": "float_cap", "f23": "pb"})
    return df


def _candidate_syms():
    s = []
    s += [f"sh{i}" for i in range(600000, 604000)]      # 沪主板
    s += [f"sh{i}" for i in range(605000, 606000)]
    s += [f"sz{i:06d}" for i in range(1, 2000)]         # 深主板 000001-001999
    s += [f"sz{i:06d}" for i in range(2000, 4000)]      # 深主板(原中小板)002xxx-003xxx
    return s


def _parse_quote(sym: str, payload: str):
    """解析腾讯行情串。成交额字段位置不固定, 用 价格×成交量 反查定位。"""
    f = payload.split("~")
    if len(f) < 40:
        return None
    try:
        name, code = f[1], f[2]
        price = float(f[3])
        volume = float(f[6]) if f[6] else 0.0
    except (ValueError, IndexError):
        return None
    if price <= 0 or "ST" in name or "退" in name:
        return None
    expect = price * volume * 100.0
    amount = expect
    if expect > 0:
        best, bi = 1e18, None
        for i in range(30, min(len(f), 55)):
            try:
                v = float(f[i])
            except (ValueError, TypeError):
                continue
            if abs(v * 10000 - expect) < best:
                best, bi = abs(v * 10000 - expect), i
        if bi is not None and best / expect < 0.3:
            amount = float(f[bi]) * 10000.0
    return {"mkt": 1 if sym.startswith("sh") else 0, "code": code, "name": name,
            "price": price, "volume": volume, "amount": amount}


def _snapshot_tencent(batch: int = 200) -> pd.DataFrame:
    syms = _candidate_syms()
    rows = []
    t0 = time.time()
    for i in range(0, len(syms), batch):
        chunk = syms[i:i + batch]
        txt = _curl(TX_QUOTE + ",".join(chunk), binary=True, retries=2)
        if txt:
            s = txt.decode("gbk", "replace")
            for line in s.split(";"):
                line = line.strip()
                if not line.startswith("v_"):
                    continue
                try:
                    sym_, payload = line.split("=", 1)
                except ValueError:
                    continue
                rec = _parse_quote(sym_[2:], payload.strip('"'))
                if rec:
                    rows.append(rec)
        if (i // batch + 1) % 10 == 0:
            print(f"   扫描 {i+len(chunk)}/{len(syms)}  命中 {len(rows)}  "
                  f"用时 {time.time()-t0:.0f}s", flush=True)
        time.sleep(0.35)
    return pd.DataFrame(rows)


def snapshot(force: bool = False) -> pd.DataFrame:
    f = CACHE / "snapshot.csv"
    if f.exists() and not force:
        return pd.read_csv(f, dtype={"code": str})
    src = probe_source(force)
    print(f"   拉取全市场快照 (源={src}) ...", flush=True)
    df = _snapshot_eastmoney() if src == "eastmoney" else pd.DataFrame()
    if df.empty:
        df = _snapshot_tencent()
        src = "tencent"
    if df.empty:
        raise RuntimeError("快照拉取失败")
    for c in ["price", "amount", "volume", "pct", "pe", "pb", "mktcap", "float_cap", "turnover"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["code"] = df["code"].astype(str)
    df.to_csv(f, index=False, encoding="utf-8")
    print(f"   快照 {len(df)} 只 (源={src})", flush=True)
    return df


def build_universe(top_n: int = 200, min_amount: float = 1.0e8,
                   max_price: float | None = None) -> pd.DataFrame:
    """主板 + 非ST + 流动性达标; 默认不按价格筛选(价格只作为特征)。"""
    d = snapshot().copy()
    d = d[d["code"].str.match(r"^(600|601|603|605|000|001|002|003)\d{3}$")]
    d = d[~d["name"].astype(str).str.contains("ST|退", na=False)]
    d = d[d["price"].notna() & (d["price"] > 0)]
    if max_price:
        d = d[d["price"] <= max_price]
    amt = d["amount"] if "amount" in d.columns else pd.Series(0.0, index=d.index)
    d = d[amt.fillna(0) >= min_amount]
    d = d.sort_values("amount", ascending=False).head(top_n)
    return d.reset_index(drop=True)


# ------------------------------------------------------------------ 日线
def _kline_eastmoney(mkt: int, code: str, beg: str, end: str):
    b = beg.replace("-", "")
    e = end.replace("-", "")
    txt = _curl(f"{EM_KLINE}?secid={int(mkt)}.{code}&fields1=f1,f2,f3,f4,f5,f6"
                f"&fields2={EM_F2}&klt=101&fqt=1&beg={b}&end={e}"
                f"&_={int(time.time()*1000)}", retries=2)
    if not txt:
        return None
    try:
        return _parse_em(txt)
    except Exception:                        # noqa: BLE001
        return None


def _kline_tencent(sym: str, start: str = BEG, max_pages: int = 5):
    frames, end = [], date.today().isoformat()
    start_ts = pd.Timestamp(start)
    for _ in range(max_pages):
        url = f"{TX_KLINE}?param={sym},day,{start},{end},{TX_MAX},qfq"
        txt = _curl(url, retries=3)
        if not txt:
            break
        try:
            df = _parse_tx(txt, sym)
        except Exception:                    # noqa: BLE001
            df = None
        if df is None or df.empty:
            break
        frames.append(df)
        if df.index[0] <= start_ts or len(df) < TX_MAX:
            break
        end = (df.index[0] - timedelta(days=1)).date().isoformat()
        time.sleep(0.25)
    if not frames:
        return None
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out


def kline_hist(mkt: int, code: str, beg: str = BEG, use_cache: bool = True,
               quiet: bool = True):
    f = CACHE / f"kline_{code}.csv"
    if use_cache and f.exists():
        df = pd.read_csv(f, parse_dates=["date"]).set_index("date").sort_index()
        if len(df) > 100:
            return df[OHLCV]
    src = probe_source()
    df = None
    if src == "eastmoney":
        df = _kline_eastmoney(mkt, code, beg, date.today().isoformat())
    if df is None or df.empty:
        df = _kline_tencent(_sym(mkt, code), beg)
        if not quiet:
            print(f"     [tx] {code}: {0 if df is None else len(df)} 根")
    if df is None or df.empty:
        return None
    df.to_csv(f, encoding="utf-8")
    return df[OHLCV]


def benchmark_hist(sym: str = "sh000300") -> pd.DataFrame:
    f = CACHE / "benchmark_000300.csv"
    if f.exists():
        return pd.read_csv(f, parse_dates=["date"]).set_index("date").sort_index()[OHLCV]
    df = _kline_tencent(sym)
    if df is None or df.empty:
        raise RuntimeError("基准指数获取失败")
    df.to_csv(f, encoding="utf-8")
    print(f"   基准 {sym}: {len(df)} 根 {df.index[0].date()} ~ {df.index[-1].date()}", flush=True)
    return df[OHLCV]


# ------------------------------------------------------------------ 指数长历史(Baostock)
BS_INDICES = {
    "sh.000300": "沪深300", "sh.000001": "上证指数", "sh.000905": "中证500",
    "sz.399006": "创业板指", "sz.399001": "深证成指",
}


BS_FIELDS = "date,open,high,low,close,volume,amount,turn,pctChg"


def stock_hist_bs(bs_mod, code: str, start: str = "2015-01-01"):
    """Baostock 个股日线。adjustflag=1 即**后复权**, 避免前复权隐含未来分红送股信息。
    code 形如 sh.600000 / sz.000001。"""
    rs = bs_mod.query_history_k_data_plus(
        code, BS_FIELDS, start_date=start, end_date=date.today().isoformat(),
        frequency="d", adjustflag="1")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return None
    cols = BS_FIELDS.split(",")
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "close"]).set_index("date").sort_index()
    df = df[df["volume"] > 0]
    return df[["open", "close", "high", "low", "volume", "amount"]]


def download_all_bs(pool, start: str = "2015-01-01", sleep: float = 0.2,
                    force: bool = True) -> int:
    """用 Baostock 批量下载个股(写入与 kline_hist 同一个缓存文件, 下游无感)。"""
    import baostock as bs
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")
    ok, t0 = 0, time.time()
    for i, r in pool.iterrows():
        f = CACHE / f"kline_{r['code']}.csv"
        if f.exists() and not force:
            continue
        code = ("sh." if int(r["mkt"]) == 1 else "sz.") + str(r["code"])
        try:
            df = stock_hist_bs(bs, code, start)
        except Exception as e:                     # noqa: BLE001
            print(f"   [err] {code}: {e}", flush=True)
            df = None
        if df is not None and len(df) > 250:
            df.to_csv(f, encoding="utf-8")
            ok += 1
        if (i + 1) % 50 == 0:
            print(f"   {i+1}/{len(pool)} 成功 {ok} 用时 {time.time()-t0:.0f}s", flush=True)
        time.sleep(sleep)
    bs.logout()
    print(f"   完成 {ok}/{len(pool)}, 用时 {time.time()-t0:.0f}s", flush=True)
    return ok


def index_hist_bs(code: str = "sh.000300", start: str = "2005-01-01",
                  use_cache: bool = True) -> pd.DataFrame | None:
    """Baostock 指数日线(免费/无 token/可回溯到 2005)。不复权, 含成交量成交额。"""
    f = CACHE / f"index_{code.replace('.', '_')}.csv"
    if use_cache and f.exists():
        df = pd.read_csv(f, parse_dates=["date"]).set_index("date").sort_index()
        if len(df) > 200:
            return df
    try:
        import baostock as bs
    except ImportError:
        print("   [warn] baostock 未安装, 无法取指数长历史")
        return None
    lg = bs.login()
    if lg.error_code != "0":
        print(f"   [warn] baostock 登录失败: {lg.error_msg}")
        return None
    rs = bs.query_history_k_data_plus(
        code, "date,open,high,low,close,volume,amount",
        start_date=start, end_date=date.today().isoformat(),
        frequency="d", adjustflag="3")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    bs.logout()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).set_index("date").sort_index()
    df["pct"] = df["close"].pct_change() * 100
    df = df[["open", "close", "high", "low", "volume", "amount", "pct"]]
    df.to_csv(f, encoding="utf-8")
    return df


def download_all(pool: pd.DataFrame, sleep: float = 0.45) -> dict:
    out, hit = {}, 0
    t0 = time.time()
    for i, r in pool.iterrows():
        cached = (CACHE / f"kline_{r['code']}.csv").exists()
        df = kline_hist(int(r["mkt"]), r["code"])
        if df is not None and len(df) >= 250:
            out[r["code"]] = df
            hit += int(cached)
        if not cached:
            time.sleep(sleep)
        if (i + 1) % 50 == 0:
            print(f"   下载 {i+1}/{len(pool)} 成功 {len(out)} 缓存命中 {hit} "
                  f"用时 {time.time()-t0:.0f}s", flush=True)
    print(f"   下载完成 {len(out)}/{len(pool)} 只 (缓存 {hit}), 用时 {time.time()-t0:.0f}s, "
          f"请求 {_STATS['req']} 失败 {_STATS['fail']}", flush=True)
    return out


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    print(f"== 自检 (源={probe_source(force=True)}) ==")
    snap = snapshot(force=not (CACHE / "snapshot.csv").exists())
    print(f"快照 {len(snap)} 只")
    pool = build_universe(top_n=n)
    print(f"股票池 {len(pool)} 只: {list(zip(pool['code'][:n], pool['name'][:n]))}")
    b = benchmark_hist()
    print(f"基准 {len(b)} 根 {b.index[0].date()} ~ {b.index[-1].date()}")
    data = download_all(pool)
    for c, d in list(data.items())[:n]:
        print(f"  {c}: {len(d)} 根 {d.index[0].date()} ~ {d.index[-1].date()}")
