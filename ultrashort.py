"""
超短線（5 分～1 小時 K 線、持有數分鐘到半天）BTC 策略檢驗（2026-10-31）

=== 事先寫定（2026-10-31，寫於看到任何報酬結果之前；之後不改）===
資料：Binance BTCUSDT 永續 5 分 K 線（含主動買進量，fetch_intraday.py），15 分、1 小時由 5 分合成（標籤＝K 線開始時間）
期間：樣本內 2020-01-01～2023-12-31、樣本外 2024-01-01～2026-09-30
執行：K 線 t 收盤時算訊號 → t+1 開盤進場，固定持有 H 根後在開盤出場；持有中出現新訊號就以新訊號為準並重新計時（不加碼）
      部位 −1／0／+1（多空對稱，除非註明）、不加槓桿；報酬用開盤到開盤
成本：主要評估 0.07%/邊（OKX 永續吃單 0.05%＋滑價 0.02%，與本專案永續設定相同）＋實際資金費（持倉跨過結算時點才付）
      另報 0（毛報酬、不含資金費）、0.02%/邊（全部掛單成交的最樂觀情況，實際上訊號單很難都掛到）、0.10%/邊（壓力測試）
日線趨勢濾網（有標註的參數）：日線收盤 > 200 日均線，shift(1) 天（D 日收盤的結果從 D+1 日 00:00 起才用）；
      做多只在上方、做空只在下方；均線暖身期（2020-07 前）不交易
Sharpe：日報酬 √365

策略族（每族 × 每個週期，用樣本內淨 Sharpe 挑一組參數）：
  F1 急漲急跌反向：z = 過去 k 根報酬 ÷（過去 3 天單根波動 × √k）；z ≤ −θ 做多、z ≥ θ 做空
     k ∈ {1,3,6,12}、θ ∈ {2,3,4}、H ∈ {1,3,6,12}、濾網 ∈ {無, 趨勢}
  F2 急漲急跌順勢：同 F1 但方向相反（z ≥ θ 做多、z ≤ −θ 做空），同一組網格
  F3 短週期通道突破：收盤 > 過去 N 根最高價做多、< 過去 N 根最低價做空；N ∈ {12,24,48,96}、H ∈ {3,6,12,24}、濾網 ∈ {無, 趨勢}
  F4 主動買賣量失衡：失衡 = 2×過去 k 根主動買進量 ÷ 總量 − 1，對過去 3 天標準化成 z；|z| ≥ θ 時順勢或反向
     k ∈ {1,3,12}、θ ∈ {2,3}、H ∈ {1,3,12}、方向 ∈ {順勢, 反向}
  F1–F4 各在 5 分、15 分、1 小時上各做一次（12 組）
  F5 時段效應（只用 1 小時）：用樣本內各 UTC 小時的平均報酬，做多前 m 名小時（只做多）或同時放空後 m 名（多空）
     m ∈ {1,2,3,4,6}、模式 ∈ {只做多, 多空}（小時的挑選本身就是擬合，F5 的樣本內結果不算數）
隨機對照：同一條部位序列在評估期間內循環平移（保留在場比例、持有長度與交易次數，只打亂時機），1000 次

通過條件（全部符合才算候選，成本 0.07%/邊＋資金費）：
  P1 樣本外 Sharpe ≥ 0.5
  P2 樣本外勝過循環平移 ≥ 95%（F1–F4 樣本內也要 ≥ 95%）
  P3 0.10%/邊時樣本外 Sharpe 仍 > 0
  P4 樣本內前 5 名參數的樣本外 Sharpe 中位數 > 0（不是只有挑中的那組剛好好）
  共 13 組，多重比較下偶然通過一兩組並不意外 → 通過的只能列為「候選」，要另外用 ETH 或其他交易所驗證
只報告：年交易次數、每筆平均毛報酬、毛報酬／0.02% 掛單版的 Sharpe、整個網格樣本外 Sharpe 的分佈、
       各年 5 分／15 分／1 小時報酬的一階自相關、單根平均絕對報酬 vs 來回成本；若有通過者，報告與 S1 的日報酬相關
事先知道的限制：5 分 K 線看不到盤口與成交順序，真正的剝頭皮（秒級、掛單賺價差）無法用這份資料檢驗；
  回測假設開盤價成交，急漲急跌時的實際滑價可能大於 0.02%
=== 事先寫定到此 ===

執行：python fetch_intraday.py（下載資料）→ python ultrashort.py [隨機次數，預設 1000]
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from itertools import product
import numpy as np, pandas as pd

N_SIMS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SPLIT, END = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC")
COST, COSTS = 0.0007, {"毛": 0.0, "掛單0.02": 0.0002, "0.07": 0.0007, "0.10": 0.0010}
TFS = {"5m": ("5min", 288), "15m": ("15min", 96), "1h": ("1h", 24)}   # (resample 規則, 每天根數)
rng = np.random.default_rng(0)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)

# ---------------------------------------------------------------- 資料
k5 = pd.read_pickle("data/btc_5m_perp.pkl")
k5 = k5[k5.index < END]
fund = pd.read_csv("data/funding.csv", index_col=0)["funding"]
fund.index = pd.to_datetime(fund.index, utc=True, format="ISO8601").floor("5min")
fund = fund.groupby(level=0).sum()

dc = k5["close"].resample("1D").last()
ma = dc.rolling(200).mean()
up_d = pd.Series(np.where(ma.notna(), (dc > ma).astype(float), np.nan), index=dc.index).shift(1)  # D 日結果 D+1 才用

def bars(tf):
    rule, _ = TFS[tf]
    if tf == "5m":
        k = k5.copy()
    else:
        k = k5.resample(rule, label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "taker_buy": "sum"}).dropna()
    o = k["open"].values
    r = np.append(o[1:] / o[:-1] - 1, 0.0)                     # 開盤到下一根開盤
    step = pd.Timedelta(rule)
    f = pd.Series(0.0, index=k.index)
    fb = fund.copy(); fb.index = fb.index - step               # 持有到結算時點的那一根付資金費
    fb = fb[fb.index.isin(k.index)]
    f.loc[fb.index] = fb.values
    up = up_d.reindex(k.index.floor("D")).values
    day = (k.index.normalize() - k.index[0].normalize()).days.values
    return k, r, f.values, up, day

def hold(sig, H):
    ev = pd.Series(np.where(sig != 0, sig, np.nan)).shift(1)   # t 收盤的訊號 → t+1 開盤進場
    return (ev.ffill(limit=H - 1) if H > 1 else ev).fillna(0).values

def apply_trend(sig, up):
    return np.where(((sig > 0) & (up == 1)) | ((sig < 0) & (up == 0)), sig, 0.0)

# ---------------------------------------------------------------- 評估
class Ctx:
    def __init__(self, tf):
        self.tf = tf
        self.k, self.r, self.f, self.up, self.day = bars(tf)
        self.split = int(np.searchsorted(self.k.index, SPLIT))
        self.per = {"IS": slice(0, self.split), "OOS": slice(self.split, len(self.k))}
        self.bpd = TFS[tf][1]

    def daily(self, p, c, s, fund=True):
        r, f, day = self.r[s], self.f[s], self.day[s]
        turn = np.abs(np.diff(p, prepend=0.0))
        net = p * r - c * turn - (p * f if fund else 0.0)
        d = np.bincount(day - day[0], weights=np.log1p(net))
        return np.expm1(d)

    def sharpe(self, p, c, s, fund=True):
        d = self.daily(p, c, s, fund)
        return float(d.mean() / d.std() * np.sqrt(365)) if d.std() > 0 else np.nan

    def all_costs(self, pos, per):
        s = self.per[per]; p = pos[s]
        return {name: self.sharpe(p, c, s, fund=c > 0) for name, c in COSTS.items()}

    def info(self, pos, per):
        s = self.per[per]; p = pos[s]
        entries = int(((p != 0) & (p != np.append(0.0, p[:-1]))).sum())
        yrs = len(p) / self.bpd / 365
        d = self.daily(p, COST, s)
        eq = np.cumprod(1 + d)
        return {"交易/年": entries / yrs, "每筆毛bp": float((p * self.r[s]).sum() / max(entries, 1) * 1e4),
                "在場%": float((p != 0).mean() * 100), "年化%": float((eq[-1] ** (365 / len(d)) - 1) * 100),
                "MaxDD%": float((eq / np.maximum.accumulate(eq) - 1).min() * 100)}

    def vs_shift(self, pos, per):
        s = self.per[per]; p = pos[s]
        base = self.sharpe(p, COST, s)
        n = len(p)
        sims = [self.sharpe(np.roll(p, int(rng.integers(1, n))), COST, s) for _ in range(N_SIMS)]
        return float(np.mean(np.array(sims) < base) * 100)

# ---------------------------------------------------------------- 策略族
def fam_signals(ctx):
    """產生 (族, 參數字串, 部位) —— F1–F4"""
    k = ctx.k; b3 = 3 * ctx.bpd
    lc = np.log(k["close"])
    vol = lc.diff().rolling(b3).std()
    for kk, th, H, filt in product([1, 3, 6, 12], [2, 3, 4], [1, 3, 6, 12], ["無", "趨勢"]):
        z = ((lc - lc.shift(kk)) / (vol * np.sqrt(kk))).values
        base = np.where(z <= -th, 1.0, np.where(z >= th, -1.0, 0.0))
        for fam, sig in (("F1反向", base), ("F2順勢", -base)):
            if filt == "趨勢":
                sig = apply_trend(sig, ctx.up)
            yield fam, f"k={kk} θ={th} H={H} {filt}", hold(sig, H)
    for N, H, filt in product([12, 24, 48, 96], [3, 6, 12, 24], ["無", "趨勢"]):
        hh = k["high"].rolling(N).max().shift(1).values; ll = k["low"].rolling(N).min().shift(1).values
        c = k["close"].values
        sig = np.where(c > hh, 1.0, np.where(c < ll, -1.0, 0.0))
        if filt == "趨勢":
            sig = apply_trend(sig, ctx.up)
        yield "F3突破", f"N={N} H={H} {filt}", hold(sig, H)
    for kk, th, H, dr in product([1, 3, 12], [2, 3], [1, 3, 12], ["順勢", "反向"]):
        ib = 2 * k["taker_buy"].rolling(kk).sum() / k["volume"].rolling(kk).sum() - 1
        z = ((ib - ib.rolling(b3).mean()) / ib.rolling(b3).std()).values
        sig = np.where(z >= th, 1.0, np.where(z <= -th, -1.0, 0.0))
        if dr == "反向":
            sig = -sig
        yield "F4買賣量", f"k={kk} θ={th} H={H} {dr}", hold(sig, H)

def f5_signals(ctx):
    s = ctx.per["IS"]
    hr = ctx.k.index.hour.values
    mean_by_h = pd.Series(ctx.r[s]).groupby(hr[s]).mean().sort_values(ascending=False)
    for m, mode in product([1, 2, 3, 4, 6], ["只做多", "多空"]):
        top, bot = mean_by_h.index[:m], mean_by_h.index[-m:]
        pos = np.isin(hr, top).astype(float)
        if mode == "多空":
            pos -= np.isin(hr, bot)
        yield "F5時段", f"m={m} {mode} 多={sorted(top.tolist())}" + (f" 空={sorted(bot.tolist())}" if mode == "多空" else ""), pos

# ---------------------------------------------------------------- 執行
print("=== 診斷：單根平均絕對報酬 vs 來回成本（0.07%/邊 → 來回 14bp）、一階自相關 ===")
diag = []
for tf in TFS:
    ctx = Ctx(tf)
    r = pd.Series(ctx.r[:-1], index=ctx.k.index[:-1])
    for y, g in r.groupby(r.index.year):
        diag.append({"週期": tf, "年": y, "平均|r| bp": g.abs().mean() * 1e4, "自相關1": g.autocorr(1)})
diag = pd.DataFrame(diag)
print(diag.pivot(index="年", columns="週期", values="平均|r| bp").round(1).to_string())
print(diag.pivot(index="年", columns="週期", values="自相關1").round(3).to_string())

rows, grid = [], []
for tf in TFS:
    ctx = Ctx(tf)
    gens = list(fam_signals(ctx)) + (list(f5_signals(ctx)) if tf == "1h" else [])
    res = {}
    for fam, par, pos in gens:
        s_is, s_oos = ctx.per["IS"], ctx.per["OOS"]
        g = {"週期": tf, "族": fam, "參數": par,
             "IS": ctx.sharpe(pos[s_is], COST, s_is), "OOS": ctx.sharpe(pos[s_oos], COST, s_oos),
             "OOS毛": ctx.sharpe(pos[s_oos], 0.0, s_oos, fund=False), "IS毛": ctx.sharpe(pos[s_is], 0.0, s_is, fund=False)}
        grid.append(g); res[(fam, par)] = pos
    gdf = pd.DataFrame([g for g in grid if g["週期"] == tf])
    for fam, sub in gdf.groupby("族", sort=False):
        sub = sub.sort_values("IS", ascending=False)
        best = sub.iloc[0]; pos = res[(fam, best["參數"])]
        row = {"週期": tf, "族": fam, "參數": best["參數"]}
        for per in ("IS", "OOS"):
            for name, v in ctx.all_costs(pos, per).items():
                row[f"{per} {name}"] = v
        row["前5名OOS中位"] = float(sub["OOS"].head(5).median())
        row["勝隨機IS%"] = ctx.vs_shift(pos, "IS") if fam != "F5時段" else np.nan
        row["勝隨機OOS%"] = ctx.vs_shift(pos, "OOS")
        for kk, v in ctx.info(pos, "OOS").items():
            row[f"OOS {kk}"] = v
        rows.append(row)
        print(tf, fam, best["參數"], f"IS {row['IS 0.07']:.2f} OOS {row['OOS 0.07']:.2f}", flush=True)

out = pd.DataFrame(rows)
p_is = out["勝隨機IS%"].fillna(100) >= 95
out["P1"] = out["OOS 0.07"] >= 0.5
out["P2"] = (out["勝隨機OOS%"] >= 95) & p_is
out["P3"] = out["OOS 0.10"] > 0
out["P4"] = out["前5名OOS中位"] > 0
out["通過"] = out[["P1", "P2", "P3", "P4"]].all(axis=1)
out.to_csv("ultrashort_results.csv", index=False)
pd.DataFrame(grid).to_csv("ultrashort_grid.csv", index=False)

print("\n=== 每族 × 週期：樣本內挑參數（0.07%/邊＋資金費）===")
cols = ["週期", "族", "參數", "IS 毛", "IS 0.07", "OOS 毛", "OOS 掛單0.02", "OOS 0.07", "OOS 0.10", "前5名OOS中位", "勝隨機IS%", "勝隨機OOS%"]
print(out[cols].round(2).to_string(index=False))
print()
cols = ["週期", "族", "OOS 交易/年", "OOS 每筆毛bp", "OOS 在場%", "OOS 年化%", "OOS MaxDD%", "P1", "P2", "P3", "P4", "通過"]
print(out[cols].round(1).to_string(index=False))

print("\n=== 整個網格的樣本外 Sharpe 分佈（只報告）===")
g = pd.DataFrame(grid)
summ = g.groupby(["週期", "族"], sort=False).agg(組數=("OOS", "size"), 毛中位=("OOS毛", "median"), 毛最高=("OOS毛", "max"),
                                                淨中位=("OOS", "median"), 淨最高=("OOS", "max"),
                                                淨大於0比例=("OOS", lambda x: (x > 0).mean()))
print(summ.round(2).to_string())
print("\n通過：", out.loc[out["通過"], ["週期", "族", "參數"]].to_string(index=False) if out["通過"].any() else "無")
