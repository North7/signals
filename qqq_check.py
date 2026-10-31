"""
QQQ 日線簡易規則檢驗＋目前狀態（2026-10-31）

=== 事先寫定（2026-10-31，寫於看到任何報酬結果之前；之後不改）===
寫定前已看過的唯一數字：最近 8 天的價格（10/30 收 690.49，單日 −3.5%、量約平常 2.5 倍）
資料：Yahoo QQQ 日線（含配息調整），1999-03-10～2026-10-30；樣本內 ～2014-12-31、樣本外 2015-01-01～
執行：D 日收盤算訊號 → D+1 開盤成交；報酬用調整後開盤到開盤；成本 0.05%/邊（價差＋滑價，不含券商手續費）；
      空手報酬 0（不計現金利息，對趨勢規則不利）；Sharpe 用日報酬 √252
策略族（每族用樣本內 Sharpe 挑一組參數）：
  T1 趨勢：收盤 > n 日均線才持有，n ∈ {100, 150, 200, 250}
  M1 多頭中買回檔：收盤 > 200 日均線且 RSI(2) < θ → 買進；出場 ∈ {收盤 > 5 日均線, 持有 5 天, 持有 10 天}；θ ∈ {5, 10, 15, 20, 25}
  M2 多頭中買大跌日：收盤 > 200 日均線且單日跌幅 ≥ X% → 持有 H 天；X ∈ {2, 3}、H ∈ {5, 10, 20}
隨機對照：部位序列在評估期間內循環平移，1000 次
通過條件（全部符合才算候選）：
  P1 樣本內、樣本外都勝過循環平移 ≥ 95%
  P2 T1：樣本外 Sharpe ≥ 買進持有；M1、M2：樣本外 Sharpe ≥ 0.5
  P3 樣本內前 5 名參數的樣本外 Sharpe 中位數 > 0
只報告：每族選中規則在 10/30 收盤的訊號；「200 日均線之上、單日跌 ≥3%」之後 5／20／60 天報酬 vs 所有日子
事先知道的限制：只有一個標的、約 3 次大空頭（2000–02、2008、2022）；Yahoo 調整價可能有小誤差；台灣複委託手續費可能遠高於 0.05%
=== 事先寫定到此 ===

執行：python qqq_check.py [隨機次數，預設 1000]（第一次會從 Yahoo 下載到 data/qqq_1d.pkl）
"""
import warnings; warnings.filterwarnings("ignore")
import sys, json, urllib.request
from pathlib import Path
from itertools import product
import numpy as np, pandas as pd

N_SIMS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SPLIT, COST = pd.Timestamp("2015-01-01"), 0.0005
PATH = Path("data/qqq_1d.pkl")
rng = np.random.default_rng(0)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)

def fetch():
    url = "https://query1.finance.yahoo.com/v8/finance/chart/QQQ?period1=0&period2=9999999999&interval=1d&events=div,split&includeAdjustedClose=true"
    j = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read())
    j = j["chart"]["result"][0]; q = j["indicators"]["quote"][0]
    idx = pd.to_datetime(j["timestamp"], unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None)
    df = pd.DataFrame({"open": q["open"], "close": q["close"], "adj": j["indicators"]["adjclose"][0]["adjclose"]}, index=idx)
    df = df.dropna()
    df = df[~df.index.duplicated(keep="last")]
    PATH.parent.mkdir(exist_ok=True); df.to_pickle(PATH)
    return df

k = pd.read_pickle(PATH) if PATH.exists() else fetch()
k = k[k.index <= "2026-10-30"]
fac = k["adj"] / k["close"]
ao, ac = (k["open"] * fac).values, k["adj"]
r = np.append(ao[1:] / ao[:-1] - 1, 0.0)                     # 調整後開盤到下一根開盤
c = k["close"]
split = int(np.searchsorted(k.index, SPLIT))
PER = {"IS": slice(0, split), "OOS": slice(split, len(k))}
sma = {n: c.rolling(n).mean() for n in (5, 100, 150, 200, 250)}
d = c.diff(); g = d.clip(lower=0).ewm(alpha=1 / 2, adjust=False).mean(); l = (-d.clip(upper=0)).ewm(alpha=1 / 2, adjust=False).mean()
rsi2 = 100 - 100 / (1 + g / l)
up200 = (c > sma[200]).values
r1 = c.pct_change().values

def sharpe(p, s):
    p = p[s]; turn = np.abs(np.diff(p, prepend=0.0))
    x = p * r[s] - COST * turn
    return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else np.nan

def stats(p, s):
    p = p[s]; turn = np.abs(np.diff(p, prepend=0.0)); x = p * r[s] - COST * turn
    eq = np.cumprod(1 + x)
    return {"年化%": (eq[-1] ** (252 / len(x)) - 1) * 100, "MaxDD%": (eq / np.maximum.accumulate(eq) - 1).min() * 100,
            "在場%": (p != 0).mean() * 100, "交易/年": ((p != 0) & (np.append(0, p[:-1]) == 0)).sum() / (len(x) / 252)}

def vs_shift(p, s):
    base = sharpe(p, s); seg = p[s]; n = len(seg)
    sims = []
    for _ in range(N_SIMS):
        q = p.copy(); q[s] = np.roll(seg, int(rng.integers(1, n))); sims.append(sharpe(q, s))
    return float(np.mean(np.array(sims) < base) * 100)

def shift1(sig):                                           # D 日收盤的訊號 → D+1 開盤起的部位
    return np.append(0.0, sig[:-1])

def m1(th, ex):
    sig = up200 & (rsi2.values < th); above5 = (c > sma[5]).values
    pos = np.zeros(len(k)); held = 0; inpos = False
    for t in range(len(k) - 1):                             # pos[t+1] 由 t 收盤決定
        if inpos:
            held += 1
            done = above5[t] if ex == "SMA5" else held >= int(ex[:-1])
            if done:
                inpos = False
        if not inpos and sig[t]:
            inpos, held = True, 0
        pos[t + 1] = 1.0 if inpos else 0.0
    return pos

def m2(X, H):
    sig = up200 & (r1 <= -X / 100)
    ev = pd.Series(np.where(sig, 1.0, np.nan)).shift(1)
    return ev.ffill(limit=H - 1).fillna(0).values

fams = {"T1趨勢": [(f"n={n}", shift1((c > sma[n]).values.astype(float)), n) for n in (100, 150, 200, 250)],
        "M1買回檔": [(f"RSI2<{th} 出場={ex}", m1(th, ex), (th, ex)) for th, ex in product([5, 10, 15, 20, 25], ["SMA5", "5天", "10天"])],
        "M2買大跌": [(f"跌≥{X}% 持有{H}天", m2(X, H), (X, H)) for X, H in product([2, 3], [5, 10, 20])]}
bh = np.ones(len(k)); bh[0] = 0
print(f"資料 {k.index[0].date()}～{k.index[-1].date()}，{len(k)} 天")
print("買進持有", {per: round(sharpe(bh, s), 2) for per, s in PER.items()}, {kk: round(v, 1) for kk, v in stats(bh, PER["OOS"]).items()})

rows = []
for fam, items in fams.items():
    tab = pd.DataFrame([{"參數": n, "IS": sharpe(p, PER["IS"]), "OOS": sharpe(p, PER["OOS"]), "_p": p, "_a": a} for n, p, a in items])
    tab = tab.sort_values("IS", ascending=False)
    b = tab.iloc[0]
    row = {"族": fam, "參數": b["參數"], "IS": b["IS"], "OOS": b["OOS"], "前5名OOS中位": tab["OOS"].head(5).median(),
           "勝隨機IS%": vs_shift(b["_p"], PER["IS"]), "勝隨機OOS%": vs_shift(b["_p"], PER["OOS"])}
    row.update({f"OOS {kk}": v for kk, v in stats(b["_p"], PER["OOS"]).items()})
    row["P1"] = row["勝隨機IS%"] >= 95 and row["勝隨機OOS%"] >= 95
    row["P2"] = row["OOS"] >= (sharpe(bh, PER["OOS"]) if fam == "T1趨勢" else 0.5)
    row["P3"] = row["前5名OOS中位"] > 0
    row["通過"] = row["P1"] and row["P2"] and row["P3"]
    # 10/30 收盤的訊號（下週一 11/2 開盤要做什麼）
    a = b["_a"]
    if fam == "T1趨勢":
        now = "持有" if c.iloc[-1] > sma[a].iloc[-1] else "空手"
    elif fam == "M1買回檔":
        now = "買進訊號" if up200[-1] and rsi2.iloc[-1] < a[0] else "無訊號"
    else:
        now = "買進訊號" if up200[-1] and r1[-1] <= -a[0] / 100 else "無訊號"
    row["10/30 訊號"] = now
    rows.append(row)
    print(f"\n{fam} 網格（依樣本內排序）"); print(tab[["參數", "IS", "OOS"]].round(2).to_string(index=False))

out = pd.DataFrame(rows)
print("\n=== 每族樣本內挑參數 ===")
print(out.round(2).to_string(index=False))

print("\n=== 目前狀態（10/30 收盤，只報告）===")
ath = c.cummax()
print({"收盤": round(c.iloc[-1], 2), "單日%": round(r1[-1] * 100, 2), "距 200 日均線%": round((c.iloc[-1] / sma[200].iloc[-1] - 1) * 100, 1),
       "距 50 日均線%": round((c.iloc[-1] / c.rolling(50).mean().iloc[-1] - 1) * 100, 1), "距歷史高點%": round((c.iloc[-1] / ath.iloc[-1] - 1) * 100, 1),
       "歷史高點日": str(c.idxmax().date()), "RSI2": round(rsi2.iloc[-1], 1)})

print("\n=== 「200 日均線之上、單日跌 ≥3%」之後的報酬（調整後收盤，D 日收盤買；只報告）===")
fw = {h: ac.shift(-h) / ac - 1 for h in (5, 20, 60)}
cond = pd.Series(up200 & (r1 <= -0.03), index=k.index)
# 同一波連續大跌只算第一天（20 天內不重複計）
ev, last = [], None
for t in k.index[cond]:
    if last is None or (t - last).days > 28:
        ev.append(t)
    last = t if last is None or (t - last).days > 28 else last
res = []
for name, mask in (("所有日子", pd.Series(True, index=k.index)), ("事件（每波第一天）", pd.Series(k.index.isin(ev), index=k.index))):
    for per, s in (("全期", slice(None)), ("2015 起", slice("2015-01-01", None))):
        m = mask[s]
        row = {"樣本": name, "期間": per, "次數": int(m.sum())}
        for h, f in fw.items():
            x = f[s][m].dropna()
            row[f"{h}天平均%"] = x.mean() * 100; row[f"{h}天中位%"] = x.median() * 100; row[f"{h}天上漲%"] = (x > 0).mean() * 100
        res.append(row)
print(pd.DataFrame(res).round(1).to_string(index=False))
print("2015 起的事件日：", [str(t.date()) for t in ev if t >= SPLIT])
out.drop(columns=[]).to_csv("qqq_results.csv", index=False)
