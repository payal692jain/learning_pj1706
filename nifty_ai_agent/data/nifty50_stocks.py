"""NIFTY 50 constituent stock data — advance/decline, top movers."""

import logging
from dataclasses import dataclass

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# NIFTY 50 constituents as of 2025 (NSE symbols with .NS suffix for yfinance)
NIFTY50_SYMBOLS: list[str] = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "BPCL.NS",
    "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS",
    "INFY.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
    "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SHRIRAMFIN.NS",
    "SUNPHARMA.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS",
    "TECHM.NS", "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS",
]


@dataclass
class StockMover:
    symbol: str
    name: str
    change_pct: float
    close: float


@dataclass
class Nifty50Summary:
    advances: int
    declines: int
    unchanged: int
    top_gainers: list[StockMover]
    top_losers: list[StockMover]
    advance_decline_ratio: float


def fetch_nifty50_movers(top_n: int = 5) -> Nifty50Summary:
    """Fetch previous session's performance for all NIFTY 50 stocks.

    Returns advance/decline count and top N gainers/losers.
    """
    logger.info("Fetching NIFTY 50 constituent data (%d stocks)", len(NIFTY50_SYMBOLS))

    movers: list[StockMover] = []

    try:
        # Use 5d so holidays / data-lag gaps don't leave us with only 1 valid row
        df = yf.download(
            NIFTY50_SYMBOLS,
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            raise ValueError("Empty data from yfinance bulk download")

        # Extract Close — yfinance 0.2.x returns MultiIndex ('Close', 'TICKER.NS')
        if isinstance(df.columns, pd.MultiIndex):
            close_df = df["Close"]
        else:
            close_df = df

        # Drop rows where every stock is NaN (holidays / data-lag trailing rows)
        # then take the last two valid rows to compute the actual session change.
        clean = close_df.dropna(how="all")
        if len(clean) < 2:
            raise ValueError(
                f"Only {len(clean)} non-empty row(s) after dropping NaN — "
                "need at least 2 to compute daily change"
            )

        prev_close = clean.iloc[-2]
        last_close = clean.iloc[-1]
        logger.info(
            "Using rows %s → %s for pct change",
            clean.index[-2].date(), clean.index[-1].date(),
        )

        for symbol in NIFTY50_SYMBOLS:
            try:
                last = float(last_close.get(symbol, float("nan")))
                prev = float(prev_close.get(symbol, float("nan")))
                if pd.isna(last) or pd.isna(prev) or prev == 0:
                    continue
                chg = (last - prev) / prev * 100
                movers.append(StockMover(
                    symbol=symbol,
                    name=symbol.replace(".NS", ""),
                    change_pct=round(chg, 2),
                    close=round(last, 2),
                ))
            except Exception:
                pass

        logger.info("Bulk fetch: %d stocks with valid data", len(movers))

    except Exception as exc:
        logger.warning("Bulk NIFTY50 fetch failed (%s) — falling back to individual", exc)
        movers = _fetch_individual_fallback(top_n * 2)

    if not movers:
        logger.warning("No NIFTY 50 mover data available from any source")
        return Nifty50Summary(
            advances=0, declines=0, unchanged=0,
            top_gainers=[], top_losers=[], advance_decline_ratio=0.0,
        )

    advances = sum(1 for m in movers if m.change_pct > 0.1)
    declines = sum(1 for m in movers if m.change_pct < -0.1)
    unchanged = len(movers) - advances - declines
    adr = round(advances / declines, 2) if declines else float("inf")

    sorted_movers = sorted(movers, key=lambda m: m.change_pct, reverse=True)

    logger.info(
        "NIFTY50: Advances=%d Declines=%d Unchanged=%d A/D=%.2f",
        advances, declines, unchanged, adr,
    )
    return Nifty50Summary(
        advances=advances,
        declines=declines,
        unchanged=unchanged,
        top_gainers=sorted_movers[:top_n],
        top_losers=sorted_movers[-top_n:][::-1],
        advance_decline_ratio=adr,
    )


def _fetch_individual_fallback(limit: int) -> list[StockMover]:
    """Fetch a handful of key stocks individually if bulk download fails."""
    key_stocks = ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "ICICIBANK.NS"]
    movers: list[StockMover] = []
    for sym in key_stocks[:limit]:
        try:
            ticker = yf.Ticker(sym)
            info = ticker.fast_info
            price = float(info.last_price)
            prev = float(info.previous_close)
            chg = ((price - prev) / prev * 100) if prev else 0.0
            movers.append(StockMover(symbol=sym, name=sym.replace(".NS", ""), change_pct=round(chg, 2), close=round(price, 2)))
        except Exception:
            pass
    return movers


def format_movers_for_notification(summary: Nifty50Summary) -> str:
    """Compact format for Pushover notification."""
    lines = [
        f"📊 NIFTY 50 A/D: {summary.advances}↑ / {summary.declines}↓ / {summary.unchanged}→",
        f"   A/D Ratio: {summary.advance_decline_ratio}",
        "",
        "Top Gainers:",
    ]
    for m in summary.top_gainers[:3]:
        lines.append(f"  {m.name}: +{m.change_pct}%")
    lines.append("Top Losers:")
    for m in summary.top_losers[:3]:
        lines.append(f"  {m.name}: {m.change_pct}%")
    return "\n".join(lines)
