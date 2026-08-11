import json
import math
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "rankings.json")

# Broad liquid NSE universe. Add more symbols here as the project grows.
SYMBOLS = """
RELIANCE TCS HDFCBANK ICICIBANK BHARTIARTL INFY SBIN LT ITC AXISBANK KOTAKBANK
M&M MARUTI SUNPHARMA TATAMOTORS TATASTEEL NTPC POWERGRID ONGC COALINDIA
ADANIPORTS ADANIENT JSWSTEEL HINDALCO BAJFINANCE BAJAJFINSV HCLTECH WIPRO
TECHM EICHERMOT HEROMOTOCO TVSMOTOR APOLLOHOSP CIPLA DRREDDY DIVISLAB
GRASIM ULTRACEMCO ASIANPAINT TITAN NESTLEIND BRITANNIA TRENT BEL HAL
BHEL IRCTC RVNL IRFC PFC REC IOC BPCL GAIL PETRONET NHPC SJVN
CANBK BANKBARODA PNB UNIONBANK INDIANBANK IDFCFIRSTB FEDERALBNK
INDUSINDBK BANDHANBNK RBLBANK AUBANK YESBANK
ZOMATO PAYTM NYKAA DELHIVERY DMART DIXON KPIT PERSISTENT
COFORGE LTIM MINDTREE MPHASIS OFSS
VEDL NMDC NATIONALUM SAIL JINDALSTEL JSWENERGY TATAPOWER
TORNTPOWER CESC TATACHEM PIDILITIND SRF DEEPAKNTR APLAPOLLO
MOTHERSON BOSCH TV18BRDC CONCOR GMRINFRA
BIOCON AUROPHARMA LUPIN MANKIND MAXHEALTH FORTIS
INDHOTEL IEX BSE CDSL MCX ANGELONE
KAYNES DIXON POLICYBZR
""".split()

BENCHMARK = "^NSEI"

BANDS = [
    (20, 100, "₹20–₹100"),
    (100, 300, "₹100–₹300"),
    (300, 500, "₹300–₹500"),
    (500, 1000, "₹500–₹1,000"),
    (1000, 1500, "₹1,000–₹1,500"),
    (1500, 2000, "₹1,500–₹2,000"),
]

WEIGHTS = {
    "momentum": 22, "trend": 18, "relativeStrength": 16, "volume": 10,
    "rsiQuality": 8, "breakout": 8, "volatility": 8, "riskReward": 10
}

def clamp(x, lo=0, hi=100):
    return float(max(lo, min(hi, x)))

def score_linear(x, lo, hi):
    if not np.isfinite(x):
        return 50.0
    return clamp((x - lo) / (hi - lo) * 100)

def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def atr(df, n=14):
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def adx(df, n=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([
        high-low,
        (high-close.shift()).abs(),
        (low-close.shift()).abs()
    ], axis=1).max(axis=1)
    atrv = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atrv
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atrv
    dx = (100 * (pdi-mdi).abs() / (pdi+mdi).replace(0, np.nan))
    return dx.ewm(alpha=1/n, adjust=False).mean()

def analyze(symbol, data, bench):
    if data is None or len(data) < 120:
        return None

    close = data["Close"].dropna()
    if len(close) < 120:
        return None

    price = float(close.iloc[-1])
    if not np.isfinite(price):
        return None

    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else sma50

    ret5 = float(close.pct_change(5).iloc[-1] * 100)
    ret20 = float(close.pct_change(20).iloc[-1] * 100)
    ret60 = float(close.pct_change(60).iloc[-1] * 100)

    bclose = bench["Close"].dropna()
    b5 = float(bclose.pct_change(5).iloc[-1] * 100) if len(bclose) >= 5 else 0
    b20 = float(bclose.pct_change(20).iloc[-1] * 100) if len(bclose) >= 20 else 0

    rel5 = ret5 - b5
    rel20 = ret20 - b20

    rv20 = float(data["Volume"].rolling(20).mean().iloc[-1])
    volume = float(data["Volume"].iloc[-1])
    volume_ratio = volume / rv20 if rv20 > 0 else 1

    rsi14 = float(rsi(close).iloc[-1])
    macd_fast = close.ewm(span=12, adjust=False).mean()
    macd_slow = close.ewm(span=26, adjust=False).mean()
    macd_line = macd_fast - macd_slow
    signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_bull = float(macd_line.iloc[-1] - signal.iloc[-1])

    adxv = float(adx(data).iloc[-1])
    atrv = float(atr(data).iloc[-1])
    atr_pct = (atrv / price * 100) if price else 99

    high20 = float(close.rolling(20).max().iloc[-2])
    breakout_gap = ((high20 - price) / price * 100) if price else 99

    momentum = clamp(50 + ret5*4 + ret20*2 + ret60*0.5)
    trend = clamp(
        50
        + (price/sma20 - 1)*500
        + (price/sma50 - 1)*300
        + (price/sma200 - 1)*150
        + (10 if sma20 > sma50 else -5)
        + (10 if sma50 > sma200 else -5)
    )
    relative = clamp(50 + rel5*4 + rel20*2)
    volume_score = clamp(50 + (volume_ratio - 1)*50)
    rsi_score = clamp(100 - abs(rsi14 - 60)*2.2)
    breakout = clamp(100 - max(0, breakout_gap)*12)
    volatility = clamp(100 - atr_pct*8)

    recent_low = float(close.tail(20).min())
    risk_pct = max(atr_pct*1.5, 3.0)
    stop = min(price * (1-risk_pct/100), recent_low*0.985)
    target = price + max(price*0.08, atrv*3)
    rr = (target-price) / max(price-stop, 0.01)
    rr_score = clamp(rr*35)

    parts = {
        "momentum": momentum, "trend": trend, "relativeStrength": relative,
        "volume": volume_score, "rsiQuality": rsi_score,
        "breakout": breakout, "volatility": volatility, "riskReward": rr_score
    }
    total = sum(parts[k] * WEIGHTS[k] for k in WEIGHTS) / 100

    signal = "Strong setup" if total >= 75 else ("Watch" if total >= 60 else "Weak setup")
    horizon = "1–4 weeks"

    return {
        "symbol": symbol,
        "price": round(price, 2),
        "score": round(total, 1),
        "signal": signal,
        "opportunity": round(clamp(total + (ret20 > 0)*5), 1),
        "confidence": round(clamp((trend + relative + volume_score)/3), 1),
        "risk": round(clamp(100-volatility), 1),
        "horizon": horizon,
        "entry": round(price, 2),
        "target": round(target, 2),
        "stop": round(stop, 2),
        "factors": {k: round(v, 1) for k, v in parts.items()},
        "raw": {
            "return5d": round(ret5, 2), "return20d": round(ret20, 2),
            "return60d": round(ret60, 2), "relative5d": round(rel5, 2),
            "relative20d": round(rel20, 2), "rsi14": round(rsi14, 2),
            "macd": round(macd_bull, 4), "adx14": round(adxv, 2),
            "volumeRatio": round(volume_ratio, 2), "atrPercent": round(atr_pct, 2),
            "breakoutGapPercent": round(breakout_gap, 2), "riskReward": round(rr, 2)
        },
        "why": [
            "Price and trend structure passed the screening rules.",
            f"20-day momentum: {ret20:.1f}%.",
            f"Relative strength vs NIFTY over 20 days: {rel20:.1f} percentage points.",
            f"Volume is {volume_ratio:.1f}× its 20-day average.",
            f"RSI(14): {rsi14:.1f}; breakout gap: {breakout_gap:.1f}%.",
        ]
    }

def main():
    symbols = [s.upper() for s in SYMBOLS if s]
    tickers = [s + ".NS" for s in symbols]

    raw = yf.download(
        tickers=tickers,
        period="1y",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False
    )

    bench = yf.download(
        BENCHMARK, period="1y", interval="1d",
        auto_adjust=True, progress=False
    )
    if isinstance(bench.columns, pd.MultiIndex):
        bench = bench.xs(BENCHMARK, axis=1, level=1)

    candidates = []
    for symbol, ticker in zip(symbols, tickers):
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker not in raw.columns.get_level_values(0):
                    continue
                df = raw[ticker].dropna()
            else:
                df = raw.dropna()
            result = analyze(symbol, df, bench)
            if result:
                candidates.append(result)
        except Exception as e:
            print("skip", symbol, e)

    bands = []
    used = set()
    for lo, hi, label in BANDS:
        # Avoid double-counting boundary values by using >= lower and < upper,
        # except the final upper edge.
        pool = [x for x in candidates if lo <= x["price"] < hi and x["symbol"] not in used]
        pool.sort(key=lambda x: x["score"], reverse=True)
        picks = pool[:5]
        for p in picks:
            used.add(p["symbol"])
        bands.append({
            "label": label,
            "min": lo,
            "max": hi,
            "count": len(picks),
            "stocks": picks
        })

    out = {
        "version": "1.1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "market": {"regime": "NIFTY benchmark used", "benchmark": BENCHMARK},
        "methodology": {
            "priceBands": [
                {"min": lo, "max": hi, "label": label, "count": 5}
                for lo, hi, label in BANDS
            ],
            "weights": WEIGHTS,
            "note": "Research signals only. No guaranteed predictions or investment advice."
        },
        "summary": {
            "candidateCount": len(candidates),
            "selectedCount": sum(len(b["stocks"]) for b in bands)
        },
        "bands": bands
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Wrote", OUT, "with", out["summary"]["selectedCount"], "selected stocks.")

if __name__ == "__main__":
    main()
