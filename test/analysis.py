# ═══════════════════════════════════════════════════════════
#   NIFTY F&O TOMORROW PREDICTION — SETUP & RUN GUIDE
# ═══════════════════════════════════════════════════════════
#
# STEP 1 — Install dependencies
# ──────────────────────────────
#   pip install -r requirements_free.txt
#
#
# STEP 2 — IMPORTANT: NSE RULES
# ──────────────────────────────
#   ✅ Run on YOUR LOCAL PC (Windows/Mac/Linux)
#   ✅ Run during market hours or just after close
#   ❌ Does NOT work on cloud servers (AWS, Colab, etc.)
#   ❌ Does NOT work before 9 AM or on weekends
#      (no option chain data available)
#
#
# STEP 3 — DAILY WORKFLOW
# ──────────────────────────────
#
#   🌙 EVENING (after 3:30 PM):
#      python fno_complete.py --evening
#      → Analyzes EOD OI, FII/DII, VIX, Technicals
#      → Saves snapshot to evening_snapshot.json
#      → Gives preliminary tomorrow prediction
#
#   🌅 MORNING (6:30 AM – 9:00 AM):
#      python fno_complete.py --morning
#      → Reads last night's US markets, GIFT Nifty
#      → Combines with evening snapshot
#      → Gives FINAL signal for the day
#
#   ⚡ AUTO-DETECT (runs correct phase by time):
#      python fno_complete.py
#
#
# STEP 4 — UNDERSTANDING THE SIGNAL
# ──────────────────────────────────
#   Score ≥ +4 → STRONG BUY CALL  (high confidence)
#   Score ≥ +2 → BUY CALL         (moderate confidence)
#   Score  0/1 → WAIT / NO TRADE  (unclear market)
#   Score ≤ -2 → BUY PUT          (moderate confidence)
#   Score ≤ -4 → STRONG BUY PUT   (high confidence)
#
#   Signals are based on 8 factors:
#     1. PCR (Put-Call Ratio)
#     2. Technical indicators (RSI, EMA, MACD, Bollinger)
#     3. FII/DII institutional flow
#     4. India VIX (fear index)
#     5. OI buildup direction
#     6. Max Pain level
#     7. IV Skew (CE vs PE implied volatility)
#     8. News sentiment
#
#
# COMMON ERRORS & FIXES
# ──────────────────────────────
#   NSE 403/404 error:
#     → Run on local PC, not server
#     → Try during 10 AM – 5 PM IST
#     → Wait 2-3 min and retry
#
#   yfinance symbol not found:
#     → Already fixed in fno_complete.py
#     → Uses threads=False for M&M.NS and BAJAJ-AUTO.NS
#
#   No evening_snapshot.json:
#     → Run --evening first before --morning
#
#
# DISCLAIMER
# ──────────────────────────────
#   This tool is for ANALYSIS ONLY.
#   It is NOT financial advice.
#   Always use stop-loss on every trade.
#   Never risk more than 1-2% of capital per trade.
# ═══════════════════════════════════════════════════════════

"""
╔══════════════════════════════════════════════════════════════╗
║     NIFTY F&O COMPLETE ANALYSIS + TOMORROW PREDICTION       ║
║     ✅ 100% Free  ✅ No API Key  ✅ Works on Windows/Mac   ║
║                                                              ║
║  IMPORTANT — HOW NSE REQUESTS WORK:                         ║
║  NSE blocks automated server requests.                       ║
║  This script must be run on YOUR LOCAL MACHINE              ║
║  (not a server/cloud). It mimics a browser session.         ║
║                                                              ║
║  RUN MODES:                                                  ║
║    python fno_complete.py --evening   (after 3:30 PM IST)  ║
║    python fno_complete.py --morning   (before 9:15 AM IST) ║
║    python fno_complete.py             (auto-detect)         ║
╚══════════════════════════════════════════════════════════════╝

SETUP:
    pip install requests pandas numpy yfinance feedparser pytz
"""

import sys, time, json, pytz, requests, feedparser
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────
# CORRECT NIFTY 50 SYMBOLS FOR YFINANCE
# ─────────────────────────────────────────────────────────
# Rules:
#   • All NSE stocks use .NS suffix
#   • Special chars removed: BAJAJ-AUTO → BAJAJ-AUTO.NS (keep hyphen)
#   • M&M → MAHM.NS doesn't work; correct is M%26M.NS via URL but
#     yfinance accepts "M&M.NS" — handled via download(threads=False)
#   • TATAMOTORS.NS is correct (DVR is a different share class)

NIFTY50_YF = [
    "RELIANCE.NS",   "TCS.NS",        "HDFCBANK.NS",   "INFY.NS",
    "ICICIBANK.NS",  "HINDUNILVR.NS", "ITC.NS",        "SBIN.NS",
    "BHARTIARTL.NS", "KOTAKBANK.NS",  "LT.NS",         "AXISBANK.NS",
    "ASIANPAINT.NS", "MARUTI.NS",     "TITAN.NS",      "SUNPHARMA.NS",
    "ULTRACEMCO.NS", "BAJFINANCE.NS", "WIPRO.NS",      "HCLTECH.NS",
    "ADANIENT.NS",   "ADANIPORTS.NS", "POWERGRID.NS",  "NTPC.NS",
    "ONGC.NS",       "JSWSTEEL.NS",   "TATASTEEL.NS",  "COALINDIA.NS",
    "BPCL.NS",       "IOC.NS",        "TECHM.NS",      "DIVISLAB.NS",
    "DRREDDY.NS",    "CIPLA.NS",      "APOLLOHOSP.NS", "EICHERMOT.NS",
    "BAJAJFINSV.NS", "HEROMOTOCO.NS", "TATAMOTORS.NS", "NESTLEIND.NS",
    "BRITANNIA.NS",  "GRASIM.NS",     "INDUSINDBK.NS", "HINDALCO.NS",
    "TATACONSUM.NS", "SBILIFE.NS",    "HDFCLIFE.NS",   "UPL.NS",
    "BAJAJ-AUTO.NS", "M&M.NS",
]

# US + Asian indices (always available on yfinance)
GLOBAL_INDICES = {
    "S&P 500"    : "^GSPC",
    "Dow Jones"  : "^DJI",
    "Nasdaq"     : "^IXIC",
    "US VIX"     : "^VIX",
    "Nikkei 225" : "^N225",
    "Hang Seng"  : "^HSI",
}

NEWS_FEEDS = {
    "ET Markets"   : "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Moneycontrol" : "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "Reuters"      : "https://feeds.reuters.com/reuters/businessNews",
    "CNBC"         : "https://www.cnbc.com/id/10000664/device/rss/rss.html",
}

BULLISH_KW = ["rally","surge","gain","growth","rate cut","stimulus","recovery",
              "strong","upgrade","profit","beat","bullish","inflow","record",
              "rebound","breakout","buying","upside","boost","rise","positive"]
BEARISH_KW = ["crash","fall","decline","sell-off","recession","rate hike",
              "inflation","war","sanctions","downgrade","weak","loss","slowdown",
              "default","crisis","tension","risk","drop","plunge","correction",
              "volatile","fear","slump","tumble"]

IST       = pytz.timezone("Asia/Kolkata")
SAVE_FILE = "evening_snapshot.json"

# ─────────────────────────────────────────────────────────
# NSE SESSION — Must run on local machine
# ─────────────────────────────────────────────────────────

NSE = requests.Session()
NSE_API_HDR = {
    "User-Agent"     : ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"),
    "Accept"         : "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer"        : "https://www.nseindia.com/option-chain",
    "X-Requested-With": "XMLHttpRequest",
    "Sec-Fetch-Dest" : "empty",
    "Sec-Fetch-Mode" : "cors",
    "Sec-Fetch-Site" : "same-origin",
    "Connection"     : "keep-alive",
}
NSE_HTML_HDR = {
    "User-Agent"     : ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"),
    "Accept"         : "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection"     : "keep-alive",
}

def init_nse():
    """
    Warm up NSE session exactly like a browser would:
    1. Visit Google (sets Referer)
    2. Visit NSE homepage (gets initial cookies)
    3. Visit option-chain page (gets NSE auth cookies)
    """
    try:
        print("  → Visiting NSE homepage...")
        NSE.get("https://www.nseindia.com", headers=NSE_HTML_HDR, timeout=15)
        time.sleep(2)
        print("  → Visiting option-chain page...")
        NSE.get("https://www.nseindia.com/option-chain", headers=NSE_HTML_HDR, timeout=15)
        time.sleep(2)
        cookies = list(NSE.cookies.keys())
        print(f"  → Cookies obtained: {cookies}")
        if not cookies:
            print("  ⚠️  No cookies received. NSE may be blocking. Try during market hours.")
        else:
            print("  ✅ NSE session ready")
    except Exception as e:
        print(f"  ⚠️  NSE init error: {e}")


def nse_get(url):
    """GET with retry + session re-init on 403/404"""
    for attempt in range(3):
        try:
            r = NSE.get(url, headers=NSE_API_HDR, timeout=15)
            if r.status_code in (403, 404):
                print(f"  ⚠️  NSE {r.status_code} on attempt {attempt+1}/3 — re-init...")
                init_nse()
                time.sleep(3)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  ⚠️  Attempt {attempt+1} failed: {e}")
            if attempt < 2:
                init_nse()
                time.sleep(3)
    print("  ❌ NSE API failed after 3 attempts.")
    print("     Possible reasons:")
    print("     1. Market is closed (run between 9 AM – 6 PM on weekdays)")
    print("     2. NSE bot protection active (try again in 5 min)")
    print("     3. Running on a cloud/server (must run on local PC)")
    return None


# ─────────────────────────────────────────────────────────
# OPTION CHAIN
# ─────────────────────────────────────────────────────────

def get_option_chain(symbol="NIFTY"):
    print(f"\n📥 Fetching {symbol} option chain from NSE...")
    data = nse_get(f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}")
    if not data:
        return None, 0, "N/A"

    expiries = data["records"]["expiryDates"]
    spot     = data["records"]["underlyingValue"]
    nearest  = expiries[0]
    rows     = []

    for item in data["records"]["data"]:
        if item["expiryDate"] != nearest:
            continue
        row = {"Strike": item["strikePrice"]}
        for side in ("CE", "PE"):
            if side in item:
                row.update({
                    f"{side}_OI"    : item[side].get("openInterest", 0),
                    f"{side}_ChgOI" : item[side].get("changeinOpenInterest", 0),
                    f"{side}_IV"    : item[side].get("impliedVolatility", 0),
                    f"{side}_LTP"   : item[side].get("lastPrice", 0),
                    f"{side}_Volume": item[side].get("totalTradedVolume", 0),
                })
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  ✅ Loaded | Spot: {spot} | Expiry: {nearest} | Strikes: {len(df)}")
    return df, spot, nearest


def calc_pcr(df):
    ce = df["CE_OI"].sum(); pe = df["PE_OI"].sum()
    return round(pe / ce, 2) if ce else 0

def calc_max_pain(df):
    pain = {}
    for s in df["Strike"]:
        ce = ((df[df["Strike"] >= s]["Strike"] - s) * df[df["Strike"] >= s]["CE_OI"]).sum()
        pe = ((s - df[df["Strike"] <= s]["Strike"]) * df[df["Strike"] <= s]["PE_OI"]).sum()
        pain[s] = ce + pe
    return min(pain, key=pain.get)

def calc_support_resistance(df, spot):
    res = df[df["Strike"] > spot].nlargest(3, "CE_OI")["Strike"].tolist()
    sup = df[df["Strike"] < spot].nlargest(3, "PE_OI")["Strike"].tolist()
    return sup, res

def calc_atm_iv(df, spot):
    atm = round(spot / 50) * 50
    row = df[df["Strike"] == atm]
    if row.empty:
        # Try nearest strike
        df2 = df.copy()
        df2["dist"] = abs(df2["Strike"] - spot)
        row = df2.nsmallest(1, "dist")
    return (float(row["CE_IV"].values[0]), float(row["PE_IV"].values[0]))

def calc_oi_buildup(df, spot):
    atm   = round(spot / 50) * 50
    above = df[df["Strike"] > atm].nlargest(5, "CE_ChgOI")
    below = df[df["Strike"] < atm].nlargest(5, "PE_ChgOI")
    ce_b  = above["CE_ChgOI"].sum()
    pe_b  = below["PE_ChgOI"].sum()
    if pe_b > ce_b * 1.5:
        return 1, f"PE OI buildup {pe_b:,.0f} > CE {ce_b:,.0f} → Support forming (bullish)"
    elif ce_b > pe_b * 1.5:
        return -1, f"CE OI buildup {ce_b:,.0f} > PE {pe_b:,.0f} → Resistance forming (bearish)"
    return 0, f"OI buildup balanced (CE:{ce_b:,.0f} / PE:{pe_b:,.0f})"

def get_atm_table(df, spot, rng=300):
    atm = round(spot / 50) * 50
    return df[(df["Strike"] >= atm - rng) & (df["Strike"] <= atm + rng)], atm


# ─────────────────────────────────────────────────────────
# YFINANCE — Nifty 50 stocks (correct symbols)
# ─────────────────────────────────────────────────────────

def fetch_stocks():
    print("\n📥 Fetching Nifty 50 stocks via yfinance...")
    try:
        # threads=False avoids issues with special chars like & and -
        raw = yf.download(
            NIFTY50_YF,
            period="2d", interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=False,   # ← Key fix for M&M.NS and BAJAJ-AUTO.NS
        )
        rows = []
        for sym in NIFTY50_YF:
            try:
                tk_data = raw[sym] if sym in raw.columns.get_level_values(0) else None
                if tk_data is None or len(tk_data) < 2:
                    continue
                today = tk_data.iloc[-1]; yesterday = tk_data.iloc[-2]
                ltp   = round(float(today["Close"]), 2)
                prev  = round(float(yesterday["Close"]), 2)
                chg   = round(((ltp - prev) / prev) * 100, 2) if prev else 0
                rows.append({
                    "Symbol"  : sym.replace(".NS","").replace(".BO",""),
                    "LTP"     : ltp,
                    "Change%" : chg,
                    "Volume"  : int(today["Volume"]) if not pd.isna(today["Volume"]) else 0,
                    "High"    : round(float(today["High"]), 2),
                    "Low"     : round(float(today["Low"]), 2),
                })
            except Exception:
                continue
        df = pd.DataFrame(rows).sort_values("Change%", ascending=False)
        print(f"  ✅ Fetched {len(df)}/50 stocks")
        return df
    except Exception as e:
        print(f"  ⚠️  Stock fetch error: {e}")
        return pd.DataFrame()


def fetch_index():
    """Nifty 50 + BankNifty via yfinance"""
    print("\n📥 Fetching index data via yfinance...")
    try:
        nf  = yf.Ticker("^NSEI").fast_info
        bnf = yf.Ticker("^NSEBANK").fast_info
        nifty     = round(nf.last_price, 2)
        banknifty = round(bnf.last_price, 2)
        prev      = round(nf.previous_close, 2)
        chg       = round(((nifty - prev) / prev) * 100, 2)
        print(f"  ✅ Nifty: {nifty} ({'+' if chg>=0 else ''}{chg}%) | BankNifty: {banknifty}")
        return nifty, banknifty, chg, prev
    except Exception as e:
        print(f"  ⚠️  Index fetch error: {e}")
        return 0, 0, 0, 0


def fetch_global_markets():
    """US + Asian indices via yfinance"""
    print("\n🌍 Fetching global markets...")
    results = {}
    for name, sym in GLOBAL_INDICES.items():
        try:
            info  = yf.Ticker(sym).fast_info
            ltp   = round(info.last_price, 2)
            prev  = round(info.previous_close, 2)
            chg   = round(((ltp - prev) / prev) * 100, 2)
            results[name] = {"price": ltp, "chg": chg}
        except Exception:
            results[name] = {"price": 0, "chg": 0}
    print(f"  ✅ Fetched {len(results)} global indices")
    return results


# ─────────────────────────────────────────────────────────
# TECHNICALS
# ─────────────────────────────────────────────────────────

def fetch_nifty_history(days=60):
    print("\n📥 Fetching Nifty history for technicals...")
    df = yf.Ticker("^NSEI").history(period=f"{days}d", interval="1d")
    df = df[["Open","High","Low","Close","Volume"]].dropna()
    print(f"  ✅ {len(df)} days of data")
    return df

def compute_technicals(hist):
    close = hist["Close"]
    # RSI
    d    = close.diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean()
    rsi  = round(100 - (100 / (1 + gain/loss)).iloc[-1], 2)
    # EMA
    ema20 = round(close.ewm(span=20).mean().iloc[-1], 2)
    ema50 = round(close.ewm(span=50).mean().iloc[-1], 2)
    # MACD
    macd_line = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    sig_line  = macd_line.ewm(span=9).mean()
    macd      = round(macd_line.iloc[-1], 2)
    sig       = round(sig_line.iloc[-1], 2)
    hist_val  = round((macd_line - sig_line).iloc[-1], 2)
    # Bollinger
    sma   = close.rolling(20).mean()
    std   = close.rolling(20).std()
    bb_up = round((sma + 2*std).iloc[-1], 2)
    bb_dn = round((sma - 2*std).iloc[-1], 2)
    bb_mid= round(sma.iloc[-1], 2)
    # Price
    last  = round(close.iloc[-1], 2)
    prev  = round(close.iloc[-2], 2)
    chg   = round(((last - prev)/prev)*100, 2)
    return dict(close=last, prev=prev, day_chg=chg,
                rsi=rsi, ema20=ema20, ema50=ema50,
                macd=macd, macd_sig=sig, macd_hist=hist_val,
                bb_up=bb_up, bb_mid=bb_mid, bb_dn=bb_dn)

def score_technicals(tech, spot):
    score, reasons = 0, []
    if tech["rsi"] < 35:
        score += 2; reasons.append(f"RSI {tech['rsi']} → Oversold, bounce likely 🟢")
    elif tech["rsi"] > 65:
        score -= 2; reasons.append(f"RSI {tech['rsi']} → Overbought, pullback likely 🔴")
    else:
        reasons.append(f"RSI {tech['rsi']} → Neutral zone")

    if tech["ema20"] > tech["ema50"]:
        score += 1; reasons.append(f"EMA20 > EMA50 → Uptrend 🟢")
    else:
        score -= 1; reasons.append(f"EMA20 < EMA50 → Downtrend 🔴")

    if spot > tech["ema20"]:
        score += 1; reasons.append(f"Price above EMA20 → Bullish structure 🟢")
    else:
        score -= 1; reasons.append(f"Price below EMA20 → Bearish structure 🔴")

    if tech["macd"] > tech["macd_sig"]:
        score += 1; reasons.append(f"MACD bullish crossover 🟢")
    else:
        score -= 1; reasons.append(f"MACD bearish crossover 🔴")

    if spot < tech["bb_dn"]:
        score += 1; reasons.append(f"Below Bollinger lower band → Bounce likely 🟢")
    elif spot > tech["bb_up"]:
        score -= 1; reasons.append(f"Above Bollinger upper band → Pullback likely 🔴")

    return score, reasons


# ─────────────────────────────────────────────────────────
# FII/DII
# ─────────────────────────────────────────────────────────

def fetch_fii_dii():
    try:
        data = nse_get("https://www.nseindia.com/api/fiidiiTradeReact")
        if not data or not isinstance(data, list):
            return None
        latest  = data[0]
        fii_net = float(str(latest.get("fiinet","0")).replace(",",""))
        dii_net = float(str(latest.get("diinet","0")).replace(",",""))
        return {"date": latest.get("date","N/A"), "fii_net": fii_net, "dii_net": dii_net}
    except Exception as e:
        print(f"  ⚠️  FII/DII: {e}"); return None

def score_fii_dii(d):
    if not d:
        return 0, ["FII/DII data unavailable"]
    score, reasons = 0, []
    f = d["fii_net"]
    if   f >  500: score += 2; reasons.append(f"FII bought ₹{f:,.0f} Cr → Strong bullish 🟢")
    elif f >    0: score += 1; reasons.append(f"FII bought ₹{f:,.0f} Cr → Mild bullish 🟡")
    elif f < -500: score -= 2; reasons.append(f"FII sold ₹{abs(f):,.0f} Cr → Strong bearish 🔴")
    else:          score -= 1; reasons.append(f"FII sold ₹{abs(f):,.0f} Cr → Mild bearish 🟡")
    dii_sign = "bought" if d["dii_net"] > 0 else "sold"
    reasons.append(f"DII {dii_sign} ₹{abs(d['dii_net']):,.0f} Cr")
    return score, reasons


# ─────────────────────────────────────────────────────────
# INDIA VIX
# ─────────────────────────────────────────────────────────

def fetch_india_vix():
    try:
        data = nse_get("https://www.nseindia.com/api/allIndices")
        if data:
            for idx in data.get("data", []):
                if idx.get("index") == "INDIA VIX":
                    return round(idx.get("last",0),2), round(idx.get("percentChange",0),2)
    except Exception as e:
        print(f"  ⚠️  VIX: {e}")
    return None, 0

def score_vix(vix, chg):
    if not vix:
        return 0, ["India VIX unavailable"]
    score, reasons = 0, []
    if   vix < 13: score += 1; reasons.append(f"VIX {vix} → Low fear 🟢")
    elif vix > 20: score -= 2; reasons.append(f"VIX {vix} → High fear 🔴")
    elif vix > 16: score -= 1; reasons.append(f"VIX {vix} → Elevated caution 🟡")
    else:                       reasons.append(f"VIX {vix} → Normal range")
    if   chg >  5: score -= 1; reasons.append(f"VIX +{chg}% today → Fear rising 🔴")
    elif chg < -5: score += 1; reasons.append(f"VIX {chg}% today → Fear falling 🟢")
    return score, reasons


# ─────────────────────────────────────────────────────────
# GIFT NIFTY
# ─────────────────────────────────────────────────────────

def fetch_gift_nifty():
    try:
        data = nse_get("https://www.nseindia.com/api/giftnifty")
        if data:
            return data.get("lastPrice",0), data.get("pChange",0)
    except Exception:
        pass
    return None, None

def score_gift(pct):
    if pct is None:
        return 0, ["GIFT Nifty unavailable"]
    score, reasons = 0, []
    if   pct >  0.5: score += 2; reasons.append(f"GIFT Nifty +{pct}% → Strong gap-up expected 🟢")
    elif pct >  0.2: score += 1; reasons.append(f"GIFT Nifty +{pct}% → Mild gap-up expected 🟡")
    elif pct < -0.5: score -= 2; reasons.append(f"GIFT Nifty {pct}% → Strong gap-down expected 🔴")
    elif pct < -0.2: score -= 1; reasons.append(f"GIFT Nifty {pct}% → Mild gap-down expected 🟡")
    else:                         reasons.append(f"GIFT Nifty {pct}% → Flat open expected")
    return score, reasons


# ─────────────────────────────────────────────────────────
# NEWS SENTIMENT
# ─────────────────────────────────────────────────────────

def fetch_news():
    print("\n📰 Fetching news sentiment...")
    bull, bear, headlines = 0, 0, []
    for src, url in NEWS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:6]:
                t = e.title.lower()
                headlines.append({"Source": src, "Headline": e.title})
                bull += sum(1 for k in BULLISH_KW if k in t)
                bear += sum(1 for k in BEARISH_KW if k in t)
        except Exception:
            continue
    total = bull + bear
    score = round((bull - bear)/total, 2) if total else 0
    label = "🐂 Bullish" if score > 0.2 else "🐻 Bearish" if score < -0.2 else "⚖️  Neutral"
    print(f"  ✅ Bull:{bull} Bear:{bear} → {label}")
    return score, label, bull, bear, headlines[:8]


# ─────────────────────────────────────────────────────────
# GLOBAL MARKETS SCORING
# ─────────────────────────────────────────────────────────

def score_global(markets):
    score, reasons = 0, []
    sp   = markets.get("S&P 500",   {}).get("chg", 0)
    dow  = markets.get("Dow Jones", {}).get("chg", 0)
    nq   = markets.get("Nasdaq",    {}).get("chg", 0)
    vix  = markets.get("US VIX",    {}).get("price", 20)
    nk   = markets.get("Nikkei 225",{}).get("chg", 0)
    hs   = markets.get("Hang Seng", {}).get("chg", 0)
    us   = round((sp + dow + nq) / 3, 2)
    asia = round((nk + hs) / 2, 2)
    if   us >  0.5: score += 2; reasons.append(f"US markets avg +{us}% → Positive cues 🟢")
    elif us >  0.0: score += 1; reasons.append(f"US markets avg +{us}% → Mildly positive 🟡")
    elif us < -0.5: score -= 2; reasons.append(f"US markets avg {us}% → Negative cues 🔴")
    else:           score -= 1; reasons.append(f"US markets avg {us}% → Mildly negative 🟡")
    if   vix < 15: score += 1; reasons.append(f"US VIX {vix} → Low fear globally 🟢")
    elif vix > 25: score -= 2; reasons.append(f"US VIX {vix} → High fear globally 🔴")
    if   asia >  0.5: score += 1; reasons.append(f"Asia avg +{asia}% → Positive 🟢")
    elif asia < -0.5: score -= 1; reasons.append(f"Asia avg {asia}% → Negative 🔴")
    return score, reasons


# ─────────────────────────────────────────────────────────
# SIGNAL ENGINE
# ─────────────────────────────────────────────────────────

def make_signal(total, spot, sup, res, expiry):
    atm = round(spot / 50) * 50
    if   total >=  4: sig = "📈 STRONG BUY CALL"; strike = res[0] if res else atm+100; conf = min(90, 55+total*6)
    elif total >=  2: sig = "📈 BUY CALL";         strike = res[0] if res else atm+100; conf = min(75, 50+total*6)
    elif total <=  -4: sig = "📉 STRONG BUY PUT";  strike = sup[0] if sup else atm-100; conf = min(90, 55+abs(total)*6)
    elif total <=  -2: sig = "📉 BUY PUT";          strike = sup[0] if sup else atm-100; conf = min(75, 50+abs(total)*6)
    else:              sig = "⚖️  WAIT / NO TRADE"; strike = atm; conf = 25
    return sig, strike, conf


def print_signal_box(sig, strike, expiry, conf, total, spot, sup, res, phase=""):
    atm = round(spot / 50) * 50
    print(f"\n{'═'*66}")
    print(f"  🎯  {'TOMORROW' if phase else 'TODAY'} TRADE SIGNAL  {phase}")
    print(f"{'═'*66}")
    print(f"  Signal     : {sig}")
    print(f"  Strike     : {strike}  (ATM is {atm})")
    print(f"  Expiry     : {expiry}")
    print(f"  Confidence : {conf}%")
    print(f"  Score      : {'+' if total>=0 else ''}{total}  (≥+2=CALL ≥+4=STRONG, ≤-2=PUT ≤-4=STRONG)")
    print(f"  Spot Ref   : {spot:,.2f}")
    print(f"  Support    : {sup}")
    print(f"  Resistance : {res}")
    print(f"\n  💡 ENTRY TIPS:")
    if "CALL" in sig:
        print(f"     • Buy {strike} CE on dip (wait 15 min after market open)")
        print(f"     • Stop-loss: If Nifty breaks {sup[0] if sup else atm-150}")
        print(f"     • Target   : {res[1] if len(res)>1 else atm+200} zone")
    elif "PUT" in sig:
        print(f"     • Buy {strike} PE on bounce (wait 15 min after market open)")
        print(f"     • Stop-loss: If Nifty breaks {res[0] if res else atm+150}")
        print(f"     • Target   : {sup[1] if len(sup)>1 else atm-200} zone")
    else:
        print(f"     • No clear signal. Avoid trading, wait for breakout")
        print(f"     • Watch {sup[0] if sup else atm-100} (support) and {res[0] if res else atm+100} (resistance)")
    print(f"\n  ⚠️  DISCLAIMER: Analysis only. NOT financial advice.")
    print(f"       Always use stop-loss. Never risk more than 1-2% of capital.")
    print(f"{'═'*66}\n")


# ─────────────────────────────────────────────────────────
# PHASE 1 — EVENING SCAN (after 3:30 PM IST)
# ─────────────────────────────────────────────────────────

def evening_scan():
    now      = datetime.now(IST)
    tomorrow = (now + timedelta(days=1 if now.weekday() < 4 else 3)).strftime("%d %b %Y")

    print(f"\n{'═'*66}")
    print(f"  🌙  EVENING SCAN — Predicting tomorrow: {tomorrow}")
    print(f"  🕐  {now.strftime('%d %b %Y  %H:%M IST')}")
    print(f"  📡  Data: NSE EOD + yfinance + FII/DII + VIX + News")
    print(f"{'═'*66}")

    init_nse()

    # Fetch all data
    oc_df, spot, expiry = get_option_chain("NIFTY")
    nifty, bnf, nchg, nprev = fetch_index()
    hist  = fetch_nifty_history(60)
    tech  = compute_technicals(hist)
    fii   = fetch_fii_dii()
    vix, vchg = fetch_india_vix()
    sent_score, sent_lbl, bull_ct, bear_ct, headlines = fetch_news()

    if oc_df is None:
        print("\n❌ Cannot continue without option chain data.")
        print("   Tip: Run between 9 AM – 6 PM IST on weekdays.")
        return

    # Compute metrics
    _pcr   = calc_pcr(oc_df)
    _mp    = calc_max_pain(oc_df)
    sup, res = calc_support_resistance(oc_df, spot)
    ce_iv, pe_iv = calc_atm_iv(oc_df, spot)
    oi_sc, oi_rsn = calc_oi_buildup(oc_df, spot)
    atm_tbl, atm  = get_atm_table(oc_df, spot)
    tech_sc, tech_rsns = score_technicals(tech, spot)
    fii_sc, fii_rsns   = score_fii_dii(fii)
    vix_sc, vix_rsns   = score_vix(vix, vchg)

    pcr_sc = 2 if _pcr>1.3 else -2 if _pcr<0.7 else 0
    mp_sc  = (1 if _mp > spot+150 else -1 if _mp < spot-150 else 0)
    iv_sc  = (-1 if pe_iv > ce_iv*1.1 else 1 if ce_iv > pe_iv*1.1 else 0)
    news_sc= (1 if sent_score>0.2 else -1 if sent_score<-0.2 else 0)

    total = pcr_sc + tech_sc + fii_sc + vix_sc + oi_sc + mp_sc + iv_sc + news_sc
    sig, strike, conf = make_signal(total, spot, sup, res, expiry)

    # Print report
    print(f"\n{'─'*66}")
    print(f"  📊  EOD OPTION CHAIN  |  Expiry: {expiry}")
    print(f"{'─'*66}")
    print(f"  Nifty Spot : {spot:>10,.2f}  |  Close: {tech['close']:,.2f} ({'+' if nchg>=0 else ''}{nchg}%)")
    print(f"  PCR (OI)   : {_pcr:>6}  {'🐂' if _pcr>1.2 else '🐻' if _pcr<0.8 else '⚖️'}")
    print(f"  Max Pain   : {_mp:>6}  |  ATM: {atm}")
    print(f"  CE IV      : {ce_iv}%   PE IV: {pe_iv}%")
    print(f"  Support    : {sup}")
    print(f"  Resistance : {res}")
    print(f"  OI Buildup : {oi_rsn}")

    print(f"\n  ATM Option Chain (±300 pts):")
    print(f"  {'Strike':>8} {'CE OI':>10} {'CE IV':>7} {'CE LTP':>8} {'PE LTP':>8} {'PE IV':>7} {'PE OI':>10}")
    print(f"  {'─'*8} {'─'*10} {'─'*7} {'─'*8} {'─'*8} {'─'*7} {'─'*10}")
    for _, row in atm_tbl.iterrows():
        mk = " ◀ATM" if row["Strike"]==atm else ""
        print(f"  {int(row['Strike']):>8} {int(row.get('CE_OI',0)):>10,} "
              f"{row.get('CE_IV',0):>7.1f} {row.get('CE_LTP',0):>8.2f} "
              f"{row.get('PE_LTP',0):>8.2f} {row.get('PE_IV',0):>7.1f} "
              f"{int(row.get('PE_OI',0)):>10,}{mk}")

    print(f"\n{'─'*66}")
    print(f"  📐  TECHNICALS (RSI / EMA / MACD / Bollinger)")
    print(f"{'─'*66}")
    print(f"  RSI 14 : {tech['rsi']}  |  EMA20: {tech['ema20']}  |  EMA50: {tech['ema50']}")
    print(f"  MACD   : {tech['macd']}  |  Signal: {tech['macd_sig']}  |  Hist: {tech['macd_hist']}")
    print(f"  BBands : {tech['bb_dn']} ── {tech['bb_mid']} ── {tech['bb_up']}")
    for r in tech_rsns: print(f"    • {r}")

    print(f"\n{'─'*66}")
    print(f"  🏦  FII / DII  |  😰  INDIA VIX")
    print(f"{'─'*66}")
    if fii:
        fs = '+' if fii['fii_net']>=0 else ''
        ds = '+' if fii['dii_net']>=0 else ''
        print(f"  FII Net : {fs}₹{fii['fii_net']:,.0f} Cr  |  DII Net: {ds}₹{fii['dii_net']:,.0f} Cr")
    for r in fii_rsns: print(f"    • {r}")
    print(f"  VIX     : {vix}  ({'+' if vchg>=0 else ''}{vchg}% today)")
    for r in vix_rsns: print(f"    • {r}")

    print(f"\n{'─'*66}")
    print(f"  📰  NEWS: {sent_lbl}  (Bull:{bull_ct} Bear:{bear_ct})")
    print(f"{'─'*66}")
    for h in headlines[:5]:
        print(f"    [{h['Source'][:10]:<10}] {h['Headline'][:56]}")

    print(f"\n{'─'*66}")
    print(f"  🔢  SCORE BREAKDOWN")
    print(f"{'─'*66}")
    scores = [
        ("PCR",          pcr_sc,  f"PCR={_pcr}"),
        ("Technicals",   tech_sc, "RSI+EMA+MACD+BB"),
        ("FII/DII",      fii_sc,  "Institutional flow"),
        ("India VIX",    vix_sc,  f"VIX={vix}"),
        ("OI Buildup",   oi_sc,   oi_rsn[:40]),
        ("News",         news_sc, sent_lbl),
        ("Max Pain",     mp_sc,   f"Pain={_mp} vs Spot={spot}"),
        ("IV Skew",      iv_sc,   f"CE={ce_iv}% PE={pe_iv}%"),
    ]
    for name, sc, note in scores:
        bar = "🟢" if sc>0 else "🔴" if sc<0 else "⚪"
        print(f"  {bar} {name:<14} {'+' if sc>=0 else ''}{sc:>3}   {note}")
    print(f"  {'─'*40}")
    print(f"  ⚡ TOTAL SCORE  {'+' if total>=0 else ''}{total:>3}   (≥+2=CALL, ≤-2=PUT, else WAIT)")

    print_signal_box(sig, strike, expiry, conf, total, spot, sup, res,
                     phase=f"| Tomorrow: {tomorrow}")

    # Save snapshot
    snap = dict(
        timestamp=now.isoformat(), tomorrow=tomorrow, expiry=expiry,
        spot=spot, nifty_close=tech["close"],
        pcr=_pcr, max_pain=_mp, ce_iv=ce_iv, pe_iv=pe_iv,
        support=sup, resistance=res,
        evening_total=total, evening_signal=sig, evening_conf=conf,
        pcr_sc=pcr_sc, tech_sc=tech_sc, fii_sc=fii_sc,
        vix_sc=vix_sc, oi_sc=oi_sc, mp_sc=mp_sc, iv_sc=iv_sc,
    )
    Path(SAVE_FILE).write_text(json.dumps(snap, indent=2))
    print(f"✅ Snapshot saved → {SAVE_FILE}")
    print(f"   Run '--morning' at 6:30 AM for final signal with US + GIFT data\n")


# ─────────────────────────────────────────────────────────
# PHASE 2 — MORNING SCAN (6–9 AM IST)
# ─────────────────────────────────────────────────────────

def morning_scan():
    now = datetime.now(IST)
    print(f"\n{'═'*66}")
    print(f"  🌅  MORNING SCAN — Final Pre-Market Signal")
    print(f"  🕐  {now.strftime('%d %b %Y  %H:%M IST')}")
    print(f"{'═'*66}")

    # Load evening snapshot
    if Path(SAVE_FILE).exists():
        snap = json.loads(Path(SAVE_FILE).read_text())
        ev_total = snap.get("evening_total", 0)
        spot     = snap.get("spot", 0)
        expiry   = snap.get("expiry", "N/A")
        sup      = snap.get("support", [])
        res      = snap.get("resistance", [])
        print(f"\n✅ Evening snapshot loaded")
        print(f"   Evening signal : {snap.get('evening_signal','N/A')}")
        print(f"   Evening score  : {'+' if ev_total>=0 else ''}{ev_total}")
    else:
        print("⚠️  No evening snapshot. Fetching fresh data...")
        init_nse()
        oc_df, spot, expiry = get_option_chain("NIFTY")
        if oc_df is not None:
            sup, res = calc_support_resistance(oc_df, spot)
        else:
            spot, expiry, sup, res = 0, "N/A", [], []
        ev_total = 0
        snap = {}

    # Fetch morning data
    markets = fetch_global_markets()
    gift_px, gift_pct = fetch_gift_nifty()
    sent_sc, sent_lbl, bull_ct, bear_ct, headlines = fetch_news()

    # Score morning signals
    global_sc, global_rsns = score_global(markets)
    gift_sc, gift_rsns     = score_gift(gift_pct)
    news_sc  = (1 if sent_sc>0.2 else -1 if sent_sc<-0.2 else 0)

    morning_total = global_sc + gift_sc + news_sc
    final_total   = ev_total + morning_total
    sig, strike, conf = make_signal(final_total, spot, sup, res, expiry)

    # Print
    print(f"\n{'─'*66}")
    print(f"  🌍  GLOBAL MARKETS")
    print(f"{'─'*66}")
    for name, d in markets.items():
        bar  = '🟢' if d['chg']>0 else '🔴' if d['chg']<0 else '⚪'
        sign = '+' if d['chg']>=0 else ''
        print(f"  {bar} {name:<15} {d['price']:>10,.2f}  ({sign}{d['chg']}%)")
    for r in global_rsns: print(f"    • {r}")

    print(f"\n{'─'*66}")
    print(f"  🎁  GIFT NIFTY")
    print(f"{'─'*66}")
    if gift_px:
        print(f"  Price : {gift_px:,.2f}  ({'+' if gift_pct>=0 else ''}{gift_pct}%)")
    else:
        print(f"  GIFT Nifty data unavailable")
    for r in gift_rsns: print(f"    • {r}")

    print(f"\n{'─'*66}")
    print(f"  📰  OVERNIGHT NEWS: {sent_lbl}  (Bull:{bull_ct} Bear:{bear_ct})")
    print(f"{'─'*66}")
    for h in headlines[:5]:
        print(f"    [{h['Source'][:10]:<10}] {h['Headline'][:56]}")

    print(f"\n{'─'*66}")
    print(f"  🔢  COMBINED SCORE")
    print(f"{'─'*66}")
    print(f"  Evening EOD score  : {'+' if ev_total>=0 else ''}{ev_total}")
    print(f"  Global markets     : {'+' if global_sc>=0 else ''}{global_sc}")
    print(f"  GIFT Nifty         : {'+' if gift_sc>=0 else ''}{gift_sc}")
    print(f"  Overnight news     : {'+' if news_sc>=0 else ''}{news_sc}")
    print(f"  {'─'*35}")
    print(f"  FINAL SCORE        : {'+' if final_total>=0 else ''}{final_total}")

    print_signal_box(sig, strike, expiry, conf, final_total, spot, sup, res,
                     phase="| FINAL")

    # Save
    out = dict(timestamp=now.isoformat(), signal=sig, strike=strike,
               expiry=expiry, confidence=conf, final_score=final_total,
               evening_score=ev_total, morning_score=morning_total,
               gift_nifty=gift_px, gift_pct=gift_pct)
    fname = f"signal_{now.strftime('%Y%m%d_%H%M')}.csv"
    pd.DataFrame([out]).to_csv(fname, index=False)
    print(f"✅ Signal saved → {fname}\n")


# ─────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--evening" in args:
        evening_scan()
    elif "--morning" in args:
        morning_scan()
    else:
        hour = datetime.now(IST).hour
        if 6 <= hour < 9:
            print("🌅 Pre-market hours → Morning Scan")
            morning_scan()
        elif hour >= 15 or hour < 6:
            print("🌙 Post-market hours → Evening Scan")
            evening_scan()
        else:
            print("⏰ Market hours — choose a mode:")
            print("   python fno_complete.py --evening")
            print("   python fno_complete.py --morning")