"""
估值策略 V（觀察組；2026-11-17 起紙上記帳，和 S1、D 的帳戶分開，不影響 RULES.md 的評估）
規格來自 Daybreak 運營；本專案的驗證見 ../valuation.py 與 CLAUDE.md（沒有通過隨機對照，只能當觀察組）。

規則：日線、現貨只做多、全進全出、沒有停損
  買進：賣壓累積 sx ≥ 99.5 且 相對未實現獲利 RUP ≤ 37.5%，連續 2 天成立，且當天賣出條件不成立
  賣出：sx ≤ 0 且 動能品質 mom > 0.85，連續 2 天成立
  sx、mom 只用日收盤算（Daybreak stress_viz 2026-11-11 版的演算法，在這裡重寫）；RUP＝Bitview 全體未實現獲利 ÷ 市值 × 100
時點：D 日的 RUP 要到 D+1 00:17 UTC 之後才有 → 一律在 D+1 04:00 UTC（台灣 12:00）那根 4H 開盤成交，每邊 0.15%
  Bitview 晚到時指令會晚發，模型仍照 12:00 開盤價記帳（網站會標出來）
資料：v_history.csv＝2011-06-01～2026-11-16 的日收盤與 RUP（Daybreak 運營整理的逐日序列；收盤是 CoinMetrics＋Daybreak）
  之後的收盤用 OKX 現貨 4H（只取已收完的日子）；RUP 每次執行從 Bitview 抓，失敗時用 data/v_rup.csv 快取
  用 OKX 收盤算的 sx 和 Daybreak 網站上的數字可能差一兩點（價格來源不同），2018 起的進出場日期相同
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
HIST, CACHE = ROOT / "v_history.csv", ROOT / "data" / "v_rup.csv"
PRM = dict(n=21, k=0.7, rn=10, rk=1.0, th=0.45, up=20, dn=20, fn=5, fm=30, fup=12, fdn=14, fp=0.7, fw=0.5)
SX_BUY, RUP_TH, SX_SELL, MOM_SELL, CONFIRM = 99.5, 37.5, 0.0, 0.85, 2
V_COST = 0.0015
BT_FROM = pd.Timestamp("2018-01-12", tz="UTC")        # OKX 4H 從 2018-01-11 08:00 起，第一個 04:00 開盤在 01-12
BV = "https://bitview.space/api/series/{}/day1?start={}"
EPOCH = pd.Timestamp("2009-01-01")                     # Bitview day1 的第 0 筆
# 自檢：基準資料期間（到 2026-11-16）的進出場日期，和 valuation.py 的「D+1 04:00 開盤」版本一致
EXPECT = [("2018-11-26", "2019-04-09"), ("2019-11-26", "2020-02-13"), ("2020-03-16", "2020-05-03"),
          ("2022-07-02", "2023-01-15"), ("2026-02-13", "2026-04-25"), ("2026-06-04", "2026-08-23"), ("2026-11-12", None)]


def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def sx_compute(p, prm=PRM):
    """每日收盤 → (賣壓累積 0–100, 動能品質 0–1)，和 p 等長；前面要從 2011 年暖身"""
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
        s = _phi(prm["k"] * mu / sd * math.sqrt(N)) if sd > 0 else 0.5
        if i > 3 * prm["rn"]:
            f = _phi(prm["rk"] * em / max(1e-12, ev2 - em * em) ** 0.5 * math.sqrt(prm["rn"]))
            r = min(100.0, r + prm["up"] * (prm["th"] - f)) if f < prm["th"] else max(0.0, r - prm["dn"] * (f - prm["th"]))
        S_[i], R_[i] = s, (1 - prm["fw"]) * r + prm["fw"] * rf
    return R_, S_


def fetch_rup(since, http_json):
    """Bitview：since（含）起每天的 RUP（%）；不含今天（UTC，當天的值是盤中）"""
    i0 = (since - EPOCH).days
    a, b = http_json(BV.format("unrealized_profit", i0)), http_json(BV.format("market_cap", i0))
    if a.get("start") != i0 or b.get("start") != i0 or not isinstance(a.get("data"), list):
        raise RuntimeError(f"Bitview 格式不同：{str(a)[:120]}")
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    out = {}
    for j, (u, mc) in enumerate(zip(a["data"], b["data"])):
        d = EPOCH + pd.Timedelta(days=i0 + j)
        if d < today and u is not None and mc:
            out[d] = u / mc * 100
    return pd.Series(out, dtype=float)


def load_series(k, offline, http_json):
    """日線序列（date 索引，UTC 日期、無時區）：close、rup。回傳 (df, 資料狀態)"""
    h = pd.read_csv(HIST, parse_dates=["date"]).set_index("date")
    end = h.index[-1]
    kc = k.close.copy(); kc.index = kc.index.tz_convert("UTC").tz_localize(None)
    full = kc.index[kc.index.hour == 20].normalize()                  # 有 20:00 那根 = 這天已收完
    dc = kc.resample("1D").last().reindex(full)
    dc = dc[dc.index > end]
    info = {"error": None, "drift": None, "source": "cache"}
    cache = pd.read_csv(CACHE, parse_dates=["date"]).set_index("date").rup if CACHE.exists() else pd.Series(dtype=float)
    if not offline:
        try:
            new = fetch_rup(end - pd.Timedelta(days=7), http_json)
            ov = new.index.intersection(h.index)
            info["drift"] = float((new[ov] - h.rup[ov]).abs().max()) if len(ov) else None   # Bitview 回補或改算法時會變大
            cache = new[new.index > end]
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            cache.rename("rup").to_frame().to_csv(CACHE, index_label="date")
            info["source"] = "bitview"
        except Exception as e:
            info["error"] = str(e)[:200]
    add = pd.DataFrame({"close": dc, "rup": cache.reindex(dc.index)})
    df = pd.concat([h, add])
    info["close_date"] = df.index[-1]
    info["rup_date"] = df.rup.dropna().index[-1]
    return df, info


def v_run(df, k):
    """回傳 dict：交易、4H 淨值（2018-01-12 起，1 起算）、最新狀態"""
    P = df.close.values.astype(float); rup = df.rup.values.astype(float)
    SX, MOM = sx_compute(P)
    buy = (np.nan_to_num(SX) >= SX_BUY) & (np.nan_to_num(rup, nan=1e9) <= RUP_TH)
    sell = (np.nan_to_num(SX, nan=1e9) <= SX_SELL) & (np.nan_to_num(MOM) > MOM_SELL)
    dates = df.index
    last = int(np.where(np.isfinite(rup))[0][-1])                     # 最後一個 RUP 已出來的日子：只判斷到這天
    i0 = int(np.searchsorted(dates, BT_FROM.tz_localize(None)))
    kidx = k.index
    events, trades, cb, cs, pos, pend = [], [], 0, 0, None, None
    for i in range(i0, last + 1):
        cb = cb + 1 if buy[i] else 0
        cs = cs + 1 if sell[i] else 0
        side = "buy" if (pos is None and cb >= CONFIRM and not sell[i]) else "sell" if (pos is not None and cs >= CONFIRM) else None
        if not side:
            continue
        fill_t = pd.Timestamp(dates[i] + pd.Timedelta(days=1, hours=4), tz="UTC")
        if fill_t in kidx:
            px = float(k.open[fill_t])
            events.append((fill_t, side, px, dates[i]))
            if side == "buy":
                pos = (fill_t, px, dates[i])
            else:
                trades.append(_trade(k, pos, (fill_t, px, dates[i])))
                pos = None
        else:                                                          # 已經決定、成交那根還沒開盤（只會發生在最後一天）
            pend = {"side": side, "fill_t": fill_t, "signal_day": dates[i]}
    # 4H 淨值：成交在那根 K 線的開盤
    ev = {kidx.get_loc(t): (s, px) for t, s, px, _ in events}
    close = k.close.values; val = np.full(len(k), np.nan)
    cash, units = 1.0, 0.0
    j0 = int(np.searchsorted(kidx, BT_FROM))
    for j in range(j0, len(k)):
        if j in ev:
            s, px = ev[j]
            if s == "buy":
                units = cash * (1 - V_COST) / px; cash = 0.0
            else:
                cash = units * px * (1 - V_COST); units = 0.0
        val[j] = cash + units * close[j]
    value = pd.Series(val, kidx).dropna()
    opn = None
    if pos is not None:
        opn = _trade(k, pos, None)
    st = {"day": dates[last], "close": float(P[last]), "sx": float(SX[last]), "mom": float(MOM[last]), "rup": float(rup[last]),
          "cb": int(cb), "cs": int(cs), "buy_ok": bool(buy[last]), "sell_ok": bool(sell[last]),
          "sx_now": float(SX[-1]), "mom_now": float(MOM[-1]), "close_day": dates[-1]}       # 收盤比 RUP 多一天時，sx 已經可以先看
    return {"trades": trades, "open": opn, "pending": pend, "value": value, "events": events, "status": st,
            "series": pd.DataFrame({"close": P, "sx": SX, "mom": MOM, "rup": rup}, index=dates)}


def _trade(k, ent, ext):
    et, epx, eday = ent
    xt, xpx = (ext[0], ext[1]) if ext else (k.index[-1], float(k.close.iloc[-1]))
    seg = k.close[(k.index >= et) & (k.index <= xt)]
    return {"entry_t": et, "entry_px": epx, "signal_in": eday, "exit_t": None if ext is None else xt, "exit_px": xpx,
            "signal_out": None if ext is None else ext[2],
            "ret": float(xpx / epx * (1 - V_COST) ** (2 if ext else 1) - 1),
            "dip": float(seg.min() / epx - 1) if len(seg) else 0.0, "days": float((xt - et).total_seconds() / 86400)}


def self_check(res):
    """基準資料期間的進出場日期要和研究版一致（只看訊號日期，不看價格）"""
    got = [(t["entry_t"].strftime("%Y-%m-%d"), t["exit_t"].strftime("%Y-%m-%d")) for t in res["trades"]]
    if res["open"]:
        got.append((res["open"]["entry_t"].strftime("%Y-%m-%d"), None))
    hist_end = pd.read_csv(HIST, usecols=["date"]).date.iloc[-1]
    upto = [g for g in got if g[0] <= hist_end]
    exp = [(a, b if (b is None or b <= hist_end) else None) for a, b in EXPECT]
    upto = [(a, b if (b is None or b <= hist_end) else None) for a, b in upto]
    assert upto == exp, f"V 的進出場和研究版不一致：{upto}"
