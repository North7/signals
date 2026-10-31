"""
下載短期持有者（STH，幣齡 < 155 天）獲利／虧損供給量 → data/sth_pnl.csv
來源：CheckOnChain「LTH/STH supply in profit/loss」圖表頁（嵌在 Plotly 裡的完整日線歷史，2009 起）
  https://charts.checkonchain.com/btconchain/supply/breakdown_lthsth_pnl_0/breakdown_lthsth_pnl_0_light.html
另存 bitcoin-data.com 的 percent-sth-in-profit（免費版只有最近 4 年）→ data/sth_pct_bd.csv，用來交叉核對
欄位：price、sth_profit、sth_loss（BTC 數量）、sth_pct = sth_profit / (sth_profit + sth_loss)
注意：標籤 D 的值是 D 日收盤（UTC 24:00）後才算得出來，使用時要先 shift(1) 天
用法：python fetch_onchain.py
"""
import base64, json, urllib.request
import numpy as np, pandas as pd

UA = {"User-Agent": "Mozilla/5.0"}
COC = "https://charts-cdn.checkonchain.com/btconchain/supply/breakdown_lthsth_pnl_0/breakdown_lthsth_pnl_0_light.html"
BD = "https://bitcoin-data.com/v1/percent-sth-in-profit"

def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read().decode("utf-8", "ignore")

def plotly_traces(html):
    i = html.find("[", html.find("Plotly.newPlot("))
    traces, _ = json.JSONDecoder().raw_decode(html[i:])
    out = {}
    for t in traces:
        y = t["y"]
        if isinstance(y, dict):                                   # Plotly 的 base64 型別陣列
            y = np.frombuffer(base64.b64decode(y["bdata"]), dtype=np.dtype(y["dtype"]))
        out[t["name"]] = pd.Series(np.asarray(y, dtype=float), index=pd.to_datetime(t["x"]))
    return pd.DataFrame(out)

if __name__ == "__main__":
    df = plotly_traces(get(COC))
    out = pd.DataFrame({"price": df["Price"], "sth_profit": df["STH in Profit"], "sth_loss": df["STH in Loss"]})
    out["sth_pct"] = out.sth_profit / (out.sth_profit + out.sth_loss)
    out.index.name = "date"
    out.dropna(subset=["sth_pct"]).to_csv("data/sth_pnl.csv")
    print(f"CheckOnChain：{out.sth_pct.first_valid_index().date()} ～ {out.index[-1].date()}，最後 {out.sth_pct.iloc[-1]:.1%}")
    bd = pd.DataFrame(json.loads(get(BD)))
    bd = pd.Series(bd.percentSthInProfit.values / 100, pd.to_datetime(bd.d), name="sth_pct_bd")
    bd.index.name = "date"
    bd.to_csv("data/sth_pct_bd.csv")
    print(f"bitcoin-data：{bd.index[0].date()} ～ {bd.index[-1].date()}，最後 {bd.iloc[-1]:.1%}")
