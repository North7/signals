"""
短期持有者獲利比例（STH percent supply in profit）能不能做成策略、或加進 S1（2026-10-31）

=== 事先寫定（2026-10-31，寫於看到任何報酬結果之前；之後不改）===
指標：STHP = STH 獲利供給 / STH 總供給（CheckOnChain，幣齡 < 155 天，日線，fetch_onchain.py）
價格代理：PR155 = 今天日收盤高於過去 155 天日收盤的比例（只用價格）。
  寫定前已看過的唯一數字：兩者 2015 起相關 0.95（不是報酬，只是「這個指標是什麼」）
時間對齊：標籤 D 的值要 D 日收盤後才算得出，CheckOnChain 每天更新時間不確定 →
  日線策略：D 的訊號決定 D+2 日的報酬（D+1 收盤才進出，保守多等一天）
  4H（S1）：D 的值從 D+1 04:00 開盤的那根 K 線收盤（UTC 08:00）起才可用
成本：日線策略 0.15%/邊（同 D）；S1 照現行實盤（OKX 現貨 0.12%/邊、上限 1 倍、寬度 ≤0.5、波動目標 40%、N20＋N30 各半）
期間：日線 樣本內 2014-01-01～2023-12-31、樣本外 2024-01-01～資料最後一天；S1 樣本內 2019-01-01～2023-12-31、樣本外同上
Sharpe 一律用日報酬 √365

A) 單獨策略（日線、現貨只做多；每族用樣本內 Sharpe 挑一組參數，STHP 與 PR155 各自挑）
  F1 獲利比例高時持有（趨勢）：x ≥ θ，θ ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
  F2 獲利比例極低時買進（投降、均值回歸）：x ≤ θ 後持有 H 天，θ ∈ {0.05, 0.1, 0.2}，H ∈ {10, 30, 60}
  F3 七日急降後買進：x 七日變化 ≤ −X 後持有 H 天，X ∈ {0.3, 0.4, 0.5}，H ∈ {10, 30, 60}
  F4 七日急降當賣出警訊（截圖的「異動」）：平常持有，急降後 H 天空手，X 同上，H ∈ {10, 30}
  隨機對照：把同一條部位序列在評估期間內循環平移（保留在場比例與持有段落長度），1000 次
  通過條件（全部符合才算候選）：
    A1 樣本外 Sharpe > 0，且樣本內、樣本外都勝過循環平移 ≥ 90%
    A2 STHP 版 − PR155 版（各自樣本內挑參數）：樣本內、樣本外 Sharpe 都 ≥ +0.20（否則鏈上資料沒有價格以外的資訊）
  只報告：與 D 的日報酬相關、和 D 各半組合 vs D 單獨（2019 起）

B) 加進 S1（4H、OKX 現貨、現行實盤設定；每族用樣本內 Sharpe 挑一組參數，STHP 與 PR155 各自挑）
  B1 進場濾網「不過熱」：x ≤ θ，θ ∈ {0.8, 0.9, 0.95}
  B2 進場濾網「確認」：x ≥ θ，θ ∈ {0.5, 0.7, 0.9}
  B3 持有條件（低）：在場時 x < θ → 下一根開盤出場，且 x < θ 時不進場，θ ∈ {0.3, 0.5, 0.7}
  B4 持有條件（急降）：在場時 x 七日變化 ≤ −X → 下一根開盤出場，且此時不進場，X ∈ {0.3, 0.4, 0.5}
  通過條件（全部符合才算候選）：
    C1 全期（2019 起）與樣本外 Sharpe 都 ≥ 基準 + 0.20（約 ±0.2 的價格路徑雜訊）
    C2 B1、B2：在「基準交易逐筆做或不做」的版本，全期勝過「隨機略過同樣多筆」≥ 95%（B3、B4 不適用）
    C3 STHP 版不低於 PR155 版（否則是價格資訊，不需要鏈上資料）
  共 8 族 × 2 種資料，多重比較下單一族「勝過隨機 90%」本來就容易出現 —— 這是 A1、C1 要求樣本外也成立的原因
事先知道的限制：CheckOnChain 的歷史是今天下載的版本（方法論若有修改會套用到過去）；BTC 只有約 4 個完整週期
=== 事先寫定到此 ===

執行：python fetch_onchain.py（下載資料）→ python sth_profit.py [隨機次數，預設 1000]
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent / "paper"))
import engine as ENG
from backtest import atr, daily_filter

N_SIMS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
D_FROM, SPLIT, IS_END = pd.Timestamp("2014-01-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2023-12-31")
S1_FROM = pd.Timestamp("2019-01-01")
DCOST, LAG = 0.0015, 2
rng = np.random.default_rng(0)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
assert ENG.S1_COST == 0.0012 and ENG.S1_CAP == 1.0
sh = lambda r: float(r.mean() / r.std() * np.sqrt(365)) if len(r) > 2 and r.std() > 0 else np.nan
mdd = lambda r: float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min())
cg = lambda r: float((1 + r).prod() ** (365 / len(r)) - 1)

# ---------------------------------------------------------------- 資料
oc = pd.read_csv("data/sth_pnl.csv", index_col=0, parse_dates=True)
oc = oc[oc.index >= "2011-01-01"]
px = oc.price
END = oc.index[-1]
def pr_rank(p, w=155):
    return p.rolling(w + 1).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
X = {"STHP": oc.sth_pct, "PR155": pr_rank(px)}
WIN = {"樣本內": (D_FROM, IS_END), "樣本外": (SPLIT, END), "全期": (D_FROM, END)}

print(f"資料：CheckOnChain 日線 {oc.index[0].date()} ～ {END.date()}；最新 STHP {oc.sth_pct.iloc[-1]:.1%}、PR155 {X['PR155'].iloc[-1]:.1%}")

# ---------------------------------------------------------------- 0) 這個指標是什麼
print("\n=== 0) STHP 和價格的關係（2015 起）===")
j = pd.concat({"STHP": X["STHP"], "PR155": X["PR155"], "r1": np.log(px).diff(), "r7": np.log(px).diff(7),
               "d1": X["STHP"].diff(), "d7": X["STHP"].diff(7)}, axis=1).loc["2015":].dropna()
print(f"水準 STHP vs PR155（只用價格）相關 {j.STHP.corr(j.PR155):.3f}")
print(f"當日變化 vs 當日報酬 {j.d1.corr(j.r1):.3f}；七日變化 vs 七日報酬 {j.d7.corr(j.r7):.3f}")
res = j.STHP - np.polyval(np.polyfit(j.PR155, j.STHP, 1), j.PR155)
print(f"STHP 對 PR155 做線性回歸：R² {1 - res.var() / j.STHP.var():.2f}，殘差標準差 {res.std():.3f}")
for nm, x in X.items():
    nxt = np.log(px).shift(-LAG - 29) - np.log(px).shift(-LAG + 1)     # D 的訊號 → D+2 起 30 天報酬
    q = pd.concat({"x": x, "f30": nxt}, axis=1).loc[D_FROM:].dropna()
    q["組"] = pd.cut(q.x, [-.01, .2, .4, .6, .8, 1.01], labels=["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"])
    t = q.groupby("組", observed=True).f30.agg(["count", "mean", "median"])
    t["上漲比例"] = q.groupby("組", observed=True).f30.apply(lambda s: (s > 0).mean())
    print(f"\n{nm} 分組 → 之後 30 天 log 報酬（2014 起，日資料重疊、僅描述）"); print(t.round(3).to_string())

# ---------------------------------------------------------------- A) 單獨策略（日線）
r_d = px.pct_change()
def run_daily(sig, lag=LAG):
    pos = sig.astype(float).shift(lag).fillna(0)
    return pos * r_d - pos.diff().abs().fillna(0) * DCOST, pos

def held(trig, H):
    return trig.astype(float).rolling(H, min_periods=1).max().fillna(0)

def fam_rules(x):
    d7 = x - x.shift(7)
    return {
        "F1 高比例持有": {f"x≥{t:g}": (x >= t) for t in [.3, .4, .5, .6, .7, .8]},
        "F2 極低買進": {f"x≤{t:g} 持{H}天": held(x <= t, H) for t in [.05, .1, .2] for H in [10, 30, 60]},
        "F3 急降買進": {f"7日≤−{X_:g} 持{H}天": held(d7 <= -X_, H) for X_ in [.3, .4, .5] for H in [10, 30, 60]},
        "F4 急降賣出": {f"7日≤−{X_:g} 空手{H}天": 1 - held(d7 <= -X_, H) for X_ in [.3, .4, .5] for H in [10, 30]},
    }

def circ_pct(pos, a, b):
    """評估期間內把部位循環平移 N_SIMS 次，回傳實際 Sharpe 勝過的比例"""
    p, r = pos.loc[a:b].values, r_d.loc[a:b].values
    def s_of(pp):
        x = pp * r - np.abs(np.diff(pp, prepend=pp[0])) * DCOST
        return x.mean() / x.std() * np.sqrt(365) if x.std() > 0 else np.nan
    act = s_of(p)
    sims = np.array([s_of(np.roll(p, rng.integers(30, len(p) - 30))) for _ in range(N_SIMS)])
    return float(np.mean(act > sims))

bh = r_d
print("\n=== A) 單獨策略（日線現貨只做多，D 的訊號 D+2 日生效，0.15%/邊）===")
print("買進持有：" + "；".join(f"{w} Sharpe {sh(bh.loc[a:b]):.2f}" for w, (a, b) in WIN.items() if w != "全期"))
picked = {}
grid_rows = []
for nm, x in X.items():
    for fam, rules in fam_rules(x).items():
        best = None
        for lab, sig in rules.items():
            ret, pos = run_daily(sig)
            row = {"資料": nm, "族": fam, "參數": lab, **{w: sh(ret.loc[a:b]) for w, (a, b) in WIN.items()},
                   "在場": pos.loc[D_FROM:].mean()}
            grid_rows.append(row)
            if best is None or row["樣本內"] > best[0]["樣本內"]:
                best = (row, ret, pos)
        picked[(nm, fam)] = best
G = pd.DataFrame(grid_rows)
print("\n網格（Sharpe）："); print(G.round(2).to_string(index=False))

print("\n每族用樣本內挑的參數：")
A = []
for (nm, fam), (row, ret, pos) in picked.items():
    A.append({"族": fam, "資料": nm, "參數": row["參數"], "IS": row["樣本內"], "OOS": row["樣本外"], "全期": row["全期"],
              "在場": row["在場"], "MaxDD": mdd(ret.loc[D_FROM:]), "CAGR": cg(ret.loc[D_FROM:]),
              "勝隨機IS": circ_pct(pos, D_FROM, IS_END), "勝隨機OOS": circ_pct(pos, SPLIT, END)})
A = pd.DataFrame(A).sort_values(["族", "資料"], ascending=[True, False])
print(A.round(2).to_string(index=False))

print("\n通過條件：")
for fam in A["族"].unique():
    s, p = A[(A.族 == fam) & (A.資料 == "STHP")].iloc[0], A[(A.族 == fam) & (A.資料 == "PR155")].iloc[0]
    a1 = s.OOS > 0 and s.勝隨機IS >= .9 and s.勝隨機OOS >= .9
    a2 = (s.IS - p.IS >= .2) and (s.OOS - p.OOS >= .2)
    print(f"  {fam}：A1 {'✓' if a1 else '✗'}（OOS {s.OOS:.2f}、勝隨機 {s.勝隨機IS:.0%}/{s.勝隨機OOS:.0%}）"
          f"  A2 {'✓' if a2 else '✗'}（STHP−PR155 IS {s.IS - p.IS:+.2f}、OOS {s.OOS - p.OOS:+.2f}）→ {'候選' if a1 and a2 else '不通過'}")

# 與 D 比較（只報告）
k = ENG.load_k()
Dv = ENG.d_run(k)["value"].resample("1D").last()
d_ret = Dv.pct_change()
d_ret.index = d_ret.index.tz_localize(None)
print("\n與 D 比較（2019 起，只報告）：")
for (nm, fam), (row, ret, pos) in picked.items():
    if nm != "STHP": continue
    jj = pd.concat({"x": ret, "D": d_ret}, axis=1).loc[S1_FROM:END].dropna()
    mix = .5 * jj.x + .5 * jj.D
    print(f"  {fam}（{row['參數']}）：與 D 相關 {jj.x.corr(jj.D):.2f}；Sharpe 本身 {sh(jj.x):.2f}、D {sh(jj.D):.2f}、各半 {sh(mix):.2f}"
          f"（樣本外 {sh(jj.x.loc[SPLIT:]):.2f}/{sh(jj.D.loc[SPLIT:]):.2f}/{sh(mix.loc[SPLIT:]):.2f}）")

# 事件研究：截圖的「異動」（七日急降）之後的報酬（只描述）
print("\n事件研究：七日急降 ≥ X 之後（間隔 ≥ 14 天才算新事件；D+2 起算）的報酬，vs 所有日子")
lp = np.log(px)
for nm, x in X.items():
    d7 = x - x.shift(7)
    for X_ in [.3, .4]:
        ev, last = [], None
        for t in d7[d7 <= -X_].loc[D_FROM:].index:
            if last is None or (t - last).days >= 14: ev.append(t)
            last = t
        ev = pd.DatetimeIndex(ev)
        out = {}
        for H in [7, 30, 90]:
            f = (lp.shift(-LAG - H + 1) - lp.shift(-LAG + 1))
            out[f"{H}天"] = f"{f.reindex(ev).mean():+.1%}（全部 {f.loc[D_FROM:].mean():+.1%}）上漲 {(f.reindex(ev) > 0).mean():.0%}"
        print(f"  {nm} 急降 ≥{X_:g}：{len(ev)} 次（樣本外 {(ev >= SPLIT).sum()}）  " + "  ".join(f"{h} {v}" for h, v in out.items()))

# ---------------------------------------------------------------- B) 加進 S1（4H）
def to_4h(s):
    """標籤 D 的日線值 → 從 D+1 04:00 開盤的 K 線起可用"""
    s = s.copy(); s.index = s.index.tz_localize("UTC") + pd.Timedelta(hours=28)
    return s.reindex(k.index, method="ffill")

def s1x(n, entry_ok=None, hold_ok=None, mult=3.0, wth=0.5):
    """ENG.s1_run 的同一套迴圈，另加：entry_ok（額外進場條件）、hold_ok（False 時下一根開盤出場，且不進場）"""
    o, h, l, c = (k[x].values for x in ["open", "high", "low", "close"])
    a = atr(k).values
    hi = k.high.rolling(n).max().shift(1).values
    up_ok = daily_filter(k)[0].values
    ok = (ENG.width_rank(k, n) <= wth).values
    if entry_ok is not None: ok = ok & entry_ok
    if hold_ok is not None: ok = ok & hold_ok
    rv = ENG.realized_vol(k)
    N = len(c)
    ret, size = np.zeros(N), np.zeros(N)
    pos, stop, best, pending, out_pending, sz, trades, ei = 0, np.nan, np.nan, 0, False, 0.0, [], 0
    for t in range(1, N):
        entered = False
        if out_pending and pos:                                  # 持有條件不成立 → 本根開盤出場
            ret[t] += o[t] / c[t - 1] - 1 - ENG.S1_COST; size[t] = sz
            trades.append((ei, t)); pos = 0; stop = np.nan
        out_pending = False
        if pending == 1 and pos == 0:
            pos = 1; ret[t] -= ENG.S1_COST; entered = True; best = o[t]; ei = t
            stop = o[t] - mult * a[t - 1]; sz = ENG.size_from_rv(rv[t - 1])
        pending = 0
        ref = o[t] if entered else c[t - 1]
        if pos:
            if l[t] <= stop:
                fill = stop if (entered and stop < o[t]) else min(o[t], stop)
                ret[t] += fill / ref - 1 - ENG.S1_COST; size[t] = sz
                trades.append((ei, t)); pos = 0; stop = np.nan
            else:
                ret[t] += c[t] / ref - 1
                best = max(best, c[t])
                stop = max(stop, best - mult * a[t])
                size[t] = sz
                if hold_ok is not None and not hold_ok[t]:
                    out_pending = True
        if np.isnan(hi[t]) or np.isnan(a[t]):
            continue
        if pos == 0 and c[t] > hi[t] and up_ok[t] and ok[t]:
            pending = 1
    return ret * size, trades

# 自檢：不加條件時與紙上交易引擎逐根相同
for n in ENG.NS:
    ref = ENG.s1_run(k, n); r0, _ = s1x(n)
    assert np.allclose(r0, (ref["ret"] * ref["size"]).values), f"N{n} 與 engine.s1_run 不一致"

day_lab = k.index.tz_localize(None).normalize()
days, starts = np.unique(day_lab.values, return_index=True)
days = pd.DatetimeIndex(days)
def daily(r4):
    return pd.Series(np.expm1(np.add.reduceat(np.log1p(r4), starts)), days)
def combo(r20, r30):
    return .5 * daily(r20) + .5 * daily(r30)
SW = {"樣本內": (S1_FROM, IS_END), "樣本外": (SPLIT, END), "全期": (S1_FROM, END)}
def s_stats(d):
    return {w: sh(d.loc[a:b]) for w, (a, b) in SW.items()} | {"MaxDD": mdd(d.loc[S1_FROM:END])}

base = {n: s1x(n) for n in ENG.NS}
base_d = combo(base[20][0], base[30][0])
B0 = s_stats(base_d)
print(f"\n=== B) 加進 S1（OKX 現貨 4H，N20＋N30，2019 起）===")
print(f"基準：Sharpe 樣本內 {B0['樣本內']:.2f}、樣本外 {B0['樣本外']:.2f}、全期 {B0['全期']:.2f}、MaxDD {B0['MaxDD']:.1%}")

X4 = {nm: to_4h(x).values for nm, x in X.items()}
D74 = {nm: to_4h(x - x.shift(7)).values for nm, x in X.items()}
for nm in X:
    v = []
    for n in ENG.NS:
        v += [X4[nm][ei - 1] for ei, _ in base[n][1] if k.index[ei] >= S1_FROM.tz_localize("UTC")]
    v = np.array(v)
    print(f"  {nm} 在 S1 進場時的分布（{len(v)} 筆）：中位 {np.nanmedian(v):.2f}，≥0.9 佔 {np.mean(v >= .9):.0%}，≥0.7 佔 {np.mean(v >= .7):.0%}，<0.5 佔 {np.mean(v < .5):.0%}")

def b_rules(nm):
    x, d7 = X4[nm], D74[nm]
    nan = np.isnan(x)
    return {
        "B1 不過熱": {f"x≤{t:g}": ("entry", (x <= t) & ~nan) for t in [.8, .9, .95]},
        "B2 確認": {f"x≥{t:g}": ("entry", (x >= t) & ~nan) for t in [.5, .7, .9]},
        "B3 低於門檻出場": {f"x<{t:g} 出場": ("hold", ~(x < t)) for t in [.3, .5, .7]},
        "B4 急降出場": {f"7日≤−{X_:g} 出場": ("hold", ~(d7 <= -X_)) for X_ in [.3, .4, .5]},
    }

def skip_version(mask_of_trade):
    """基準交易逐筆做或不做（略過的期間空手）"""
    out = {}
    for n in ENG.NS:
        r, tr = base[n]
        m = np.zeros(len(r))
        for i, (ei, xi) in enumerate(tr):
            if mask_of_trade[n][i]: m[ei:xi + 1] = 1
        out[n] = r * m
    return combo(out[20], out[30])

def skip_random_pct(keep, w):
    """在評估期間 w 內，每個 N 隨機保留與濾網同樣多筆基準交易，N_SIMS 次；回傳濾網 Sharpe 勝過的比例"""
    a, b = SW[w]
    in_w = {n: np.array([a <= k.index[ei].tz_localize(None) <= b + pd.Timedelta(days=1) for ei, _ in base[n][1]]) for n in ENG.NS}
    act = sh(skip_version({n: keep[n] & in_w[n] for n in ENG.NS}).loc[a:b])
    sims = []
    for _ in range(N_SIMS):
        m = {}
        for n in ENG.NS:
            idx = np.flatnonzero(in_w[n]); kk = np.zeros(len(in_w[n]), bool)
            kk[rng.choice(idx, int((keep[n] & in_w[n]).sum()), replace=False)] = True
            m[n] = kk
        sims.append(sh(skip_version(m).loc[a:b]))
    return float(np.mean(act > np.array(sims)))

Brows, Bpick = [], {}
for nm in X:
    for fam, rules in b_rules(nm).items():
        best = None
        for lab, (kind, arr) in rules.items():
            res = {n: s1x(n, entry_ok=arr if kind == "entry" else None, hold_ok=arr if kind == "hold" else None) for n in ENG.NS}
            d = combo(res[20][0], res[30][0])
            row = {"資料": nm, "族": fam, "參數": lab, **s_stats(d),
                   "交易數": sum(sum(k.index[ei] >= S1_FROM.tz_localize("UTC") for ei, _ in res[n][1]) for n in ENG.NS)}
            Brows.append(row)
            if best is None or row["樣本內"] > best[0]["樣本內"]:
                best = (row, kind, arr)
        Bpick[(nm, fam)] = best
BG = pd.DataFrame(Brows)
print("\n網格（Sharpe）："); print(BG.round(2).to_string(index=False))

print("\n每族用樣本內挑的參數（Δ = 相對基準）：")
BA = []
for (nm, fam), (row, kind, arr) in Bpick.items():
    rec = {"族": fam, "資料": nm, "參數": row["參數"], "IS": row["樣本內"], "OOS": row["樣本外"], "全期": row["全期"],
           "ΔOOS": row["樣本外"] - B0["樣本外"], "Δ全期": row["全期"] - B0["全期"], "MaxDD": row["MaxDD"], "交易數": row["交易數"]}
    if kind == "entry":
        keep = {n: np.array([bool(arr[ei - 1]) for ei, _ in base[n][1]]) for n in ENG.NS}
        sk = skip_version(keep)
        rec["逐筆版全期"] = sh(sk.loc[S1_FROM:END])
        rec["勝隨機刪單(全期)"] = skip_random_pct(keep, "全期")
        rec["勝隨機刪單(OOS)"] = skip_random_pct(keep, "樣本外")
    BA.append(rec)
BA = pd.DataFrame(BA).sort_values(["族", "資料"], ascending=[True, False])
print(BA.round(2).to_string(index=False))

print("\n通過條件：")
for fam in BA["族"].unique():
    s, p = BA[(BA.族 == fam) & (BA.資料 == "STHP")].iloc[0], BA[(BA.族 == fam) & (BA.資料 == "PR155")].iloc[0]
    c1 = s.ΔOOS >= .2 and s.Δ全期 >= .2
    c2 = s["勝隨機刪單(全期)"] >= .95 if "勝隨機刪單(全期)" in s and pd.notna(s["勝隨機刪單(全期)"]) else None
    c3 = s.OOS >= p.OOS and s.全期 >= p.全期
    ok = c1 and c3 and (c2 is None or c2)
    print(f"  {fam}：C1 {'✓' if c1 else '✗'}（Δ全期 {s.Δ全期:+.2f}、ΔOOS {s.ΔOOS:+.2f}）"
          + (f"  C2 {'✓' if c2 else '✗'}（{s['勝隨機刪單(全期)']:.0%}）" if c2 is not None else "  C2 不適用")
          + f"  C3 {'✓' if c3 else '✗'}（PR155 全期 {p.全期:.2f}、OOS {p.OOS:.2f}）→ {'候選' if ok else '不通過'}")
