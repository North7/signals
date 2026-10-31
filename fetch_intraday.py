"""從 data.binance.vision 下載 BTCUSDT USDT 永續 5 分 K 線（月檔，含主動買進量）與資金費率，給 ultrashort.py 用。
用法：python fetch_intraday.py → data/btc_5m_perp.pkl、data/funding.csv（若不存在）"""
import io, zipfile, urllib.request, pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fetch_data import months, fetch_funding

BASE = "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/5m"
COLS = ["ts", "open", "high", "low", "close", "volume", "close_time", "qvol", "count", "taker_buy", "taker_buy_q", "ignore"]

def one(ym):
    y, m = ym
    url = f"{BASE}/BTCUSDT-5m-{y}-{m:02d}.zip"
    for _ in range(4):
        try:
            raw = urllib.request.urlopen(url, timeout=60).read(); break
        except Exception:
            raw = None
    if raw is None:
        return None
    z = zipfile.ZipFile(io.BytesIO(raw))
    df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), header=None, names=COLS)
    if not str(df.iloc[0, 0]).isdigit():  # 2022 年起的檔案有表頭
        df = df.iloc[1:]
    return df[["ts", "open", "high", "low", "close", "volume", "taker_buy"]].astype(float)

if __name__ == "__main__":
    yms = list(months((2019, 9), (2026, 9)))
    with ThreadPoolExecutor(8) as ex:
        frames = [f for f in ex.map(one, yms) if f is not None]
    df = pd.concat(frames)
    df["ts"] = df["ts"].where(df["ts"] < 1e14, df["ts"] / 1000)
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop(columns="ts").sort_index()
    df = df[~df.index.duplicated()]
    df.to_pickle("data/btc_5m_perp.pkl")
    print("5m perp", df.index[0], "->", df.index[-1], len(df), "缺根", int(((df.index[-1] - df.index[0]) / pd.Timedelta("5min")) + 1 - len(df)))
    if not Path("data/funding.csv").exists():
        f = fetch_funding(); f.to_csv("data/funding.csv")
        print("funding", f.index[0], "->", f.index[-1], len(f))
