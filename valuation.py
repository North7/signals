"""
估值策略 V（賣壓累積＋相對未實現獲利）照本專案標準重新驗證（2026-11-17）

來源：Daybreak 運營 session 的規格（私人 repo daybreak-ops-private/ops/research/valuation-strategy/），
逐日序列複製到 data/valuation_series.csv（date, close, sx, mom, rup, sthp；2011-06-01 起，公開資料算出）。

=== 事先寫定（2026-11-17，寫於執行本腳本任何檢驗之前；之後不改）===
事先已看過的：交接 README 的規則、9 筆逐筆交易、RUP 門檻表與確認天數的敏感度（全部是樣本內、無成本或每邊 0.15%）。
沒有看過的：隨機對照、參數網格、賣壓參數擾動、價格代理、與 D 的關係、4H 開盤成交。

規則（主版本 C）：日線、現貨只做多、全進全出、沒有停損
  買進條件：sx ≥ 99.5 且 RUP ≤ 37.5，連續 2 天成立，且當天賣出條件不成立
  賣出條件：sx ≤ 0 且 mom > 0.85，連續 2 天成立
  sx、mom 只用每日收盤算（自己重算，先確認和 series.csv 一致）；RUP 用 series.csv
時間對齊：D 日的 sx、mom 在 D 收盤就有；RUP(D) 要到 D+1 00:17 UTC 之後才有
  主版本：D+1 收盤成交（2012 起都能算，偏保守）
  實盤版本：D+1 04:00 UTC 那根 4H 開盤成交（台灣 12:00；OKX 現貨 4H，2018 起才有）
  只當參考：D 收盤成交（RUP 其實還沒出來，偏樂觀）
成本：每邊 0.15%（同 D）
期間：全期 2012-06-01 起；切分 樣本內 2012-06-01～2021-12-31（6 筆）、樣本外 2022-01-01 起（3 筆＋未平倉）
  注意：賣壓累積 2026-11 才設計、RUP 門檻看過全部資料，所以這裡的「樣本外」不是真的樣本外，只用來看前後是否一致
Sharpe 一律用日報酬 √365（空手日報酬 0）

檢驗
  T1 重現 README（C 版，D+1 收盤、0.15%）
  T2 隨機對照（各 2000 次）
     a 循環平移：把在場序列在評估期間內循環平移（保留在場比例與段落長度），比 Sharpe；全期、樣本內、樣本外各算
     b 隨機進場：每筆交易隨機換一個進場日（任何一天）、持有天數照原交易，比平均每筆對數報酬；另報「全部賺錢」的機率
     c 估值區內隨機進場：同 b，但進場日只能挑 RUP ≤ 37.5 的日子 → 檢驗賣壓時機與出場規則在「便宜」之外有沒有加分
  T3 參數網格：RUP 門檻 {32.5, 35, 37.5, 40, 42.5} × 確認天數 {1, 2, 3} × 買進 sx {90, 95, 99.5}
     × 出場 mom {0.75, 0.8, 0.85, 0.9} × 出場 sx ≤ {0, 10}，共 360 組；另報用樣本內 Sharpe 挑出的那組在樣本外的表現
  T4 賣壓累積的內部參數擾動：up、dn、th、rn、fw、fup、fdn 各自 ×0.75 與 ×1.25（一次改一個，14 組），重算 sx 後跑 C 版
  T5 價格代理：RUP 換成「收盤 ÷ 1400 日均線（約 200 週）」，門檻取讓樣本內落在區內的天數比例和 RUP ≤ 37.5 相同 → 看鏈上資料有沒有價格以外的資訊
  T6 三種賣法 B（sx ≤ 0）、C、D（STHP 5 日均 ≥ 90）× 三種成交時點
  T7 只報告：與 D 的日報酬相關、V 在場時 D 的平均持倉、V＋D 組合 vs D 單獨（2019 起）
通過條件（全部符合才算「規則本身有支持」，否則上線只能標成觀察）
  P1 T2a：全期、樣本內、樣本外都勝過循環平移 ≥ 90%
  P2 T2b：勝過隨機進場 ≥ 95%
  P3 T2c：勝過估值區內隨機進場 ≥ 90%（否則賣壓與出場規則沒有比「便宜時隨便買、抱同樣久」好）
  P4 T3：採用組合的「鄰居」（只改一個參數一格，最多 9 組）全部累計報酬 > 0，且鄰居 Sharpe 中位數 ≥ 採用組合的 0.7 倍
  P5 T4：14 組全部累計報酬 > 0，且 Sharpe 中位數 ≥ 採用組合的 0.7 倍
  P6 T5 只報告（價格代理若一樣好，代表 RUP 主要是價格資訊）
事先知道的限制：14 年只有 9 筆、四個週期；sx 與 RUP 門檻都是看過全部歷史才定的；Bitview 歷史是今天下載的版本
=== 事先寫定到此 ===

執行：.venv/bin/python valuation.py [隨機次數，預設 2000]
"""
import math
import sys

import numpy as np
import pandas as pd

N_RAND = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
COST = 0.0015
START, IS_END, OOS_START = "2012-06-01", "2021-12-31", "2022-01-01"
RNG = np.random.default_rng(20261117)

# ---------------------------------------------------------------- 資料
S = pd.read_csv("data/valuation_series.csv", parse_dates=["date"]).set_index("date")
DATES = S.index
P = S.close.values.astype(float)
RUP = S.rup.values.astype(float)
STHP5 = S.sthp.rolling(5).mean().values
n = len(S)

# 4H 開盤：D+1 04:00 UTC（OKX 現貨，2018 起）
K4 = pd.read_csv("paper/data/btc_spot_4h.csv", index_col=0, parse_dates=True)
O4 = K4.open[K4.index.hour == 4]
O4.index = O4.index.normalize().tz_localize(None)
OPEN04 = O4.reindex(DATES).values                      # 當天 04:00 UTC 的開盤價

# ---------------------------------------------------------------- 賣壓累積與動能品質（Daybreak stress_viz 2026-11-11 版的演算法，自己重寫）
PRM = dict(n=21, k=0.7, rn=10, rk=1.0, th=0.45, up=20, dn=20, fn=5, fm=30, fup=12, fdn=14, fp=0.7, fw=0.5)

def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def sx_compute(p, prm=PRM):
    m = len(p); lr = np.zeros(m); lr[1:] = np.log(p[1:] / p[:-1])
    S_, R_ = np.full(m, np.nan), np.full(m, np.nan)
    al, em, ev2, r = 2 / (prm["rn"] + 1), 0.0, 0.0, 0.0
    af, u, rf = 2 / (prm["fn"] + 1), 0.0, 0.0
    N, FM = prm["n"], prm["fm"]
    for i in range(1, m):
        em += al * (lr[i] - em); ev2 += al * (lr[i] ** 2 - ev2)
        if i >= FM + 2:
            w = lr[i - FM + 1: i + 1]; sd = w.std(ddof=1)
            u += af * ((lr[i] / sd if sd > 0 else 0.0) - u)
            rf = min(100.0, rf + prm["fup"] * (-u) ** prm["fp"]) if u < 0 else max(0.0, rf - prm["fdn"] * u ** prm["fp"])
        if i < N:
            continue
        w = lr[i - N + 1: i + 1]; mu = w.mean(); sd = w.std(ddof=1)
        s = Phi(prm["k"] * mu / sd * math.sqrt(N)) if sd > 0 else 0.5
        if i > 3 * prm["rn"]:
            f = Phi(prm["rk"] * em / max(1e-12, ev2 - em * em) ** 0.5 * math.sqrt(prm["rn"]))
            r = min(100.0, r + prm["up"] * (prm["th"] - f)) if f < prm["th"] else max(0.0, r - prm["dn"] * (f - prm["th"]))
        S_[i], R_[i] = s, (1 - prm["fw"]) * r + prm["fw"] * rf
    return R_, S_

SX, MOM = sx_compute(P)

# ---------------------------------------------------------------- 回測
def backtest(buy, sell, k=2, timing="C1", cost=COST, start=START):
    """buy、sell：每天的條件（bool 陣列）。timing：D0＝D 收盤、C1＝D+1 收盤、O4＝D+1 04:00 UTC 開盤。
    回傳 每日淨值（收盤）、在場序列（當天收盤到下一天收盤是否持有）、交易清單"""
    i0 = int(np.searchsorted(DATES, pd.Timestamp(start)))
    eq = np.full(n, np.nan); held = np.zeros(n, bool); trades = []
    cash, units, cb, cs, pend, ent = 1.0, 0.0, 0, 0, None, None
    for i in range(i0, n):
        if pend is not None:
            side, px = pend
            if side == "buy":
                units = cash * (1 - cost) / px; ent = (i, px); cash = 0.0
            else:
                cash = units * px * (1 - cost); trades.append(dict(entry=DATES[ent[0]], entry_px=ent[1], exit=DATES[i], exit_px=px,
                                                                   ret=cash_ret(ent[1], px, cost), days=i - ent[0],
                                                                   dip=float(np.nanmin(P[ent[0]:i + 1]) / ent[1] - 1)))
                units = 0.0; ent = None
            pend = None
        cb = cb + 1 if buy[i] else 0
        cs = cs + 1 if sell[i] else 0
        side = None
        if units == 0 and cb >= k and not sell[i]:
            side = "buy"
        elif units > 0 and cs >= k:
            side = "sell"
        if side:
            if timing == "D0":
                px = P[i]
                if side == "buy":
                    units = cash * (1 - cost) / px; ent = (i, px); cash = 0.0
                else:
                    cash = units * px * (1 - cost); trades.append(dict(entry=DATES[ent[0]], entry_px=ent[1], exit=DATES[i], exit_px=px,
                                                                       ret=cash_ret(ent[1], px, cost), days=i - ent[0],
                                                                       dip=float(np.nanmin(P[ent[0]:i + 1]) / ent[1] - 1)))
                    units = 0.0; ent = None
            elif i + 1 < n:
                px = P[i + 1] if timing == "C1" else OPEN04[i + 1]
                if not np.isfinite(px):
                    raise ValueError(f"{DATES[i + 1].date()} 沒有 04:00 開盤價")
                pend = (side, px)
        eq[i] = cash + units * P[i]
        held[i] = units > 0
    opn = None if ent is None else dict(entry=DATES[ent[0]], entry_px=ent[1], ret_now=P[-1] / ent[1] * (1 - cost) - 1)
    return pd.Series(eq, DATES), pd.Series(held, DATES), trades, opn

def cash_ret(a, b, cost):
    return b / a * (1 - cost) ** 2 - 1

def stats(eq, a=START, b=None):
    e = eq.loc[a:b].dropna()
    r = e.pct_change().dropna()
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    sh = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else 0.0           # 整段空手 → 0
    dd = (e / e.cummax() - 1).min()
    return dict(total=e.iloc[-1] / e.iloc[0] - 1, cagr=(e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1, sharpe=sh, maxdd=dd)

def fmt_trades(T):
    return f"{len(T)} 筆 賺 {sum(t['ret'] > 0 for t in T)}" + (f"、最差 {min(t['ret'] for t in T):+.1%}、持有中位 {np.median([t['days'] for t in T]):.0f} 天" if T else "")

def line(name, res, extra=""):
    eq, held, T, opn = res
    a, i_, o = stats(eq), stats(eq, START, IS_END), stats(eq, OOS_START)
    tim = held.loc[START:].mean()
    print(f"  {name:<34} {fmt_trades(T)}｜全期 累計 {a['total']:+,.0%} 年化 {a['cagr']:+.1%} Sharpe {a['sharpe']:.2f} MaxDD {a['maxdd']:.0%}"
          f"｜IS {i_['sharpe']:.2f} OOS {o['sharpe']:.2f}｜在場 {tim:.0%}" + extra
          + (f"｜未平倉 {opn['entry'].date()} @{opn['entry_px']:,.0f} {opn['ret_now']:+.1%}" if opn else ""))

def conds(sx=SX, mom=MOM, rup=RUP, rup_th=37.5, buy_sx=99.5, exit_mom=0.85, exit_sx=0.0, exit="C"):
    buy = (sx >= buy_sx) & (rup <= rup_th)
    if exit == "B":
        sell = sx <= exit_sx
    elif exit == "C":
        sell = (sx <= exit_sx) & (mom > exit_mom)
    else:
        sell = STHP5 >= 90
    return np.nan_to_num(buy).astype(bool), np.nan_to_num(sell).astype(bool)

# ---------------------------------------------------------------- 主程式
if __name__ == "__main__":
    print(f"資料 {DATES[0].date()}～{DATES[-1].date()}，收盤 {P[-1]:,.0f}，sx {SX[-1]:.1f}，mom {MOM[-1]:.3f}，RUP {RUP[-1]}")
    d = np.abs(SX - S.sx.values); dm = np.abs(MOM - S.mom.values)
    print(f"重算 sx 與 series.csv：最大差 {np.nanmax(d):.4f}（>0.01 的 {int(np.nansum(d > 0.01))} 天）；mom 最大差 {np.nanmax(dm):.5f}")

    print("\n== T1 重現 README（C 版）")
    base = backtest(*conds(), timing="C1")
    line("C｜D+1 收盤、0.15%", base)
    line("C｜D 收盤、無成本（README 第一列）", backtest(*conds(), timing="D0", cost=0.0))
    for t in base[2]:
        print(f"    {t['entry'].date()} {t['entry_px']:>9,.1f} → {t['exit'].date()} {t['exit_px']:>9,.1f} {t['ret']:+6.1%}  最深 {t['dip']:+6.1%}  {t['days']} 天")

    print("\n== T6 三種賣法 × 成交時點")
    for ex in "BCD":
        for tm, lab, st in (("D0", "D 收盤（參考）", START), ("C1", "D+1 收盤", START), ("C1", "D+1 收盤，2018 起", "2018-01-12"),
                            ("O4", "D+1 04:00 開盤，2018 起", "2018-01-12")):
            res = backtest(*conds(exit=ex), timing=tm, start=st)
            eq = res[0]; a = stats(eq, st)
            print(f"  {ex}｜{lab:<20} {fmt_trades(res[2])}｜累計 {a['total']:+,.0%} Sharpe {a['sharpe']:.2f} MaxDD {a['maxdd']:.0%}"
                  + (f"｜未平倉 {res[3]['entry'].date()} @{res[3]['entry_px']:,.0f} {res[3]['ret_now']:+.1%}" if res[3] else ""))
    bh = stats(pd.Series(P, DATES)); bh18 = stats(pd.Series(P, DATES), "2018-01-12")
    bi, bo = stats(pd.Series(P, DATES), START, IS_END), stats(pd.Series(P, DATES), OOS_START)
    print(f"  一直抱著：2012-06 起 Sharpe {bh['sharpe']:.2f}（IS {bi['sharpe']:.2f} OOS {bo['sharpe']:.2f}）MaxDD {bh['maxdd']:.0%}；2018 起 累計 {bh18['total']:+.0%} Sharpe {bh18['sharpe']:.2f} MaxDD {bh18['maxdd']:.0%}")

    # ------------------------------------------------ T2 隨機對照
    print(f"\n== T2 隨機對照（{N_RAND} 次，C 版、D+1 收盤）")
    eq0, held0, T0, _ = base
    r_mkt = pd.Series(P, DATES).pct_change().fillna(0).values
    def shift_test(a, b):
        sl = slice(int(np.searchsorted(DATES, pd.Timestamp(a))), int(np.searchsorted(DATES, pd.Timestamp(b), side="right")) if b else n)
        h = held0.values[sl].astype(float); rm = r_mkt[sl][1:]
        def sh(hh):
            x = hh[:-1] * rm - COST * np.abs(np.diff(np.r_[0, hh]))[:-1]
            return x.mean() / x.std() * np.sqrt(365) if x.std() > 0 else 0.0
        real = sh(h); L = len(h)
        sims = np.array([sh(np.roll(h, RNG.integers(60, L - 60))) for _ in range(N_RAND)])
        return real, (sims < real).mean()
    for lab, a, b in (("全期", START, None), ("樣本內", START, IS_END), ("樣本外", OOS_START, None)):
        real, pctl = shift_test(a, b)
        print(f"  a 循環平移｜{lab}：Sharpe {real:.2f}，勝過 {pctl:.0%}")
    lr_trade = np.array([math.log(1 + t["ret"]) for t in T0]); durs = [t["days"] for t in T0]
    i0 = int(np.searchsorted(DATES, pd.Timestamp(START)))
    def rand_entries(pool):
        out_mean, all_win = [], 0
        for _ in range(N_RAND):
            rs = []
            for dd in durs:
                cand = pool[pool + dd + 1 < n]
                e = RNG.choice(cand)
                rs.append(math.log(P[e + 1 + dd] / P[e + 1] * (1 - COST) ** 2))   # 同樣 D+1 收盤進、持有同樣天數
            out_mean.append(np.mean(rs)); all_win += all(x > 0 for x in rs)
        out_mean = np.array(out_mean)
        return (out_mean < lr_trade.mean()).mean(), all_win / N_RAND, np.exp(np.median(out_mean)) - 1
    pool_all = np.arange(i0, n)
    pool_rup = np.array([i for i in range(i0, n) if RUP[i] <= 37.5])
    for lab, pool in (("b 任何一天進場", pool_all), (f"c 只在 RUP ≤ 37.5 的日子進場（{len(pool_rup)} 天）", pool_rup)):
        p_, aw, med = rand_entries(pool)
        print(f"  {lab}：V 每筆平均 {math.exp(lr_trade.mean()) - 1:+.1%}，隨機中位 {med:+.1%}，勝過 {p_:.0%}；隨機 {len(durs)} 筆全賺的機率 {aw:.1%}")

    # ------------------------------------------------ T3 參數網格
    print("\n== T3 參數網格（C 版、D+1 收盤、0.15%）")
    rows = []
    for th in (32.5, 35, 37.5, 40, 42.5):
        for k in (1, 2, 3):
            for bs in (90, 95, 99.5):
                for em in (0.75, 0.8, 0.85, 0.9):
                    for es in (0, 10):
                        res = backtest(*conds(rup_th=th, buy_sx=bs, exit_mom=em, exit_sx=es), k=k, timing="C1")
                        a, i_, o = stats(res[0]), stats(res[0], START, IS_END), stats(res[0], OOS_START)
                        T = res[2]
                        rows.append(dict(th=th, k=k, bs=bs, em=em, es=es, n=len(T), win=sum(t["ret"] > 0 for t in T),
                                         worst=min([t["ret"] for t in T], default=np.nan), total=a["total"], sharpe=a["sharpe"],
                                         maxdd=a["maxdd"], is_sh=i_["sharpe"], oos_sh=o["sharpe"]))
    G = pd.DataFrame(rows); G.to_csv("valuation_grid.csv", index=False)
    me = G[(G.th == 37.5) & (G.k == 2) & (G.bs == 99.5) & (G.em == 0.85) & (G.es == 0)].iloc[0]
    print(f"  採用組合：Sharpe {me.sharpe:.2f}（全部 360 組中第 {int((G.sharpe > me.sharpe).sum()) + 1} 名），IS {me.is_sh:.2f} OOS {me.oos_sh:.2f}")
    print(f"  360 組：Sharpe 中位 {G.sharpe.median():.2f}（25–75% {G.sharpe.quantile(.25):.2f}–{G.sharpe.quantile(.75):.2f}），"
          f"累計 > 0 的 {(G.total > 0).mean():.0%}，有虧損單的 {(G.worst < 0).mean():.0%}")
    grid = dict(th=[32.5, 35, 37.5, 40, 42.5], k=[1, 2, 3], bs=[90, 95, 99.5], em=[0.75, 0.8, 0.85, 0.9], es=[0, 10])
    nb = []
    for key, vals in grid.items():
        j = vals.index(me[key])
        for jj in (j - 1, j + 1):
            if 0 <= jj < len(vals):
                q = G.copy()
                for kk in grid:
                    q = q[q[kk] == (vals[jj] if kk == key else me[kk])]
                nb.append((key, vals[jj], q.iloc[0]))
    for key, v, q in nb:
        print(f"    鄰居 {key}={v}: {q.n} 筆 賺 {q.win}，累計 {q.total:+,.0%}，Sharpe {q.sharpe:.2f}（IS {q.is_sh:.2f} OOS {q.oos_sh:.2f}），最差 {q.worst:+.1%}")
    nb_sh = [q.sharpe for _, _, q in nb]
    print(f"  鄰居 Sharpe 中位 {np.median(nb_sh):.2f}（採用 {me.sharpe:.2f} 的 {np.median(nb_sh) / me.sharpe:.0%}），全部累計 > 0：{all(q.total > 0 for _, _, q in nb)}")
    for kk in grid:
        print(f"    依 {kk} 分組的 Sharpe 中位：" + "、".join(f"{v}→{G[G[kk] == v].sharpe.median():.2f}" for v in grid[kk]))
    best = G.loc[G.is_sh.idxmax()]
    print(f"  用樣本內挑：th={best.th} k={best.k} 買 sx≥{best.bs} 出 mom>{best.em} sx≤{best.es} → IS {best.is_sh:.2f}，OOS {best.oos_sh:.2f}"
          f"（OOS 在 360 組中位 {G.oos_sh.median():.2f}）")

    # ------------------------------------------------ T4 賣壓參數擾動
    print("\n== T4 賣壓累積參數擾動（一次改一個，C 版、D+1 收盤）")
    res4 = []
    for key in ("up", "dn", "th", "rn", "fw", "fup", "fdn"):
        for f in (0.75, 1.25):
            prm = dict(PRM); prm[key] = PRM[key] * f
            if key == "rn":
                prm[key] = int(round(prm[key]))
            sx2, mom2 = sx_compute(P, prm)
            res = backtest(*conds(sx=sx2, mom=mom2), timing="C1")
            a = stats(res[0]); res4.append((key, f, a, res[2]))
            print(f"  {key}×{f}: {fmt_trades(res[2])}｜累計 {a['total']:+,.0%} Sharpe {a['sharpe']:.2f} MaxDD {a['maxdd']:.0%}")
    sh4 = [a["sharpe"] for _, _, a, _ in res4]
    a0 = stats(eq0)
    print(f"  14 組 Sharpe 中位 {np.median(sh4):.2f}（採用 {a0['sharpe']:.2f} 的 {np.median(sh4) / a0['sharpe']:.0%}），全部累計 > 0：{all(a['total'] > 0 for _, _, a, _ in res4)}")

    # ------------------------------------------------ T5 價格代理
    print("\n== T5 價格代理：收盤 ÷ 1400 日均線")
    prox = (pd.Series(P, DATES) / pd.Series(P, DATES).rolling(1400).mean()).values
    isl = slice(int(np.searchsorted(DATES, pd.Timestamp(START))), int(np.searchsorted(DATES, pd.Timestamp(IS_END), side="right")))
    share = np.mean(RUP[isl] <= 37.5)
    th_p = np.nanquantile(prox[isl], share)
    ok = np.isfinite(prox) & np.isfinite(RUP)
    print(f"  樣本內 RUP ≤ 37.5 的天數比例 {share:.1%} → 價格代理門檻 {th_p:.2f}；兩者相關 {np.corrcoef(prox[ok], RUP[ok])[0, 1]:.2f}"
          f"（2012-06 起有代理值：{DATES[np.argmax(np.isfinite(prox))].date()}）")
    line("V（RUP）", base)
    line(f"價格代理 ≤ {th_p:.2f}", backtest(*conds(rup=prox, rup_th=th_p), timing="C1"))
    for q in (0.8, 1.2):
        line(f"價格代理 ≤ {th_p * q:.2f}（門檻 ×{q}）", backtest(*conds(rup=prox, rup_th=th_p * q), timing="C1"))
    b_rup = np.nan_to_num(RUP <= 37.5).astype(bool); s_mom = np.nan_to_num(MOM > 0.85).astype(bool)
    line("只看 RUP ≤ 37.5、賣在強勢動能", backtest(b_rup, s_mom, timing="C1"))
    line("只看賣壓（不看 RUP）", backtest(*conds(rup=np.zeros(n)), timing="C1"))

    # ------------------------------------------------ T7 與 D 的關係
    print("\n== T7 與 D（日線趨勢＋動能）的關係，2019 起（V 用 D+1 04:00 開盤成交）")
    sys.path.insert(0, "paper")
    import engine as E
    k = E.load_k(); dr = E.d_run(k)
    dv = dr["value"].resample("1D").last(); dv.index = dv.index.tz_localize(None)
    dw = dr["weight"].resample("1D").last(); dw.index = dw.index.tz_localize(None)
    v4 = backtest(*conds(), timing="O4", start="2018-01-12")
    vv = v4[0]
    a, b = "2019-01-01", min(dv.index[-1], vv.index[-1])
    rv, rd = vv.loc[a:b].pct_change().dropna(), dv.loc[a:b].pct_change().dropna()
    j = rv.index.intersection(rd.index)
    hv = v4[1].loc[a:b]; dwh = dw.reindex(hv.index)
    print(f"  日報酬相關 {np.corrcoef(rv[j], rd[j])[0, 1]:.2f}；V 在場 {hv.mean():.0%} 的時間，那些日子 D 平均持倉 {dwh[hv.values].mean():.0%}（全期平均 {dwh.mean():.0%}）")
    for w in (0.0, 0.25, 0.5):
        comb = (1 - w) * (dv.loc[a:b] / dv.loc[a]) + w * (vv.loc[a:b] / vv.loc[a])   # 不再平衡
        s = stats(comb.dropna(), a)
        print(f"  D {1 - w:.0%}＋V {w:.0%}（不再平衡）：累計 {s['total']:+.0%} 年化 {s['cagr']:+.1%} Sharpe {s['sharpe']:.2f} MaxDD {s['maxdd']:.0%}")
    sv = stats(vv.loc[a:b].dropna(), a)
    print(f"  V 單獨：累計 {sv['total']:+.0%} 年化 {sv['cagr']:+.1%} Sharpe {sv['sharpe']:.2f} MaxDD {sv['maxdd']:.0%}")
