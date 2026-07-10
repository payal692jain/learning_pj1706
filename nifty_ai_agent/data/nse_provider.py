"""NSE + yfinance market data provider with retry logic."""

import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests
import yfinance as yf

from nifty_ai_agent.data.base import MarketDataProvider, OptionChainData, SpotData

logger = logging.getLogger(__name__)

_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)
_NSE_HEADERS = {
    "User-Agent": _BROWSER_USER_AGENT,
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}
_NSE_BASE = "https://www.nseindia.com"
_RETRY_COUNT = 3
_RETRY_DELAY = 5  # seconds — NSE needs more breathing room than generic APIs
_STRIKE_STEP = 50  # NIFTY strike price interval
_BROWSER_NAV_TIMEOUT_MS = 25_000
_BROWSER_SETTLE_MS = 2_500  # let Akamai's bot-check JS finish before the API call


def _retry(fn, *args, retries: int = _RETRY_COUNT, delay: float = _RETRY_DELAY, **kwargs):
    """Call *fn* up to *retries* times, logging each failure."""
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            logger.warning("Attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(delay)
    logger.error("All %d attempts failed. Last error: %s", retries, last_exc)
    raise last_exc  # type: ignore[misc]


class NSEDataProvider(MarketDataProvider):
    """Fetches NSE index data (NIFTY, BANKNIFTY, ...) from NSE + yfinance + Upstox."""

    def __init__(
        self,
        symbol: str = "^NSEI",
        upstox_access_token: str = "",
        index_name: str = "NIFTY",
        strike_step: int = _STRIKE_STEP,
    ) -> None:
        self._symbol = symbol
        self._upstox_token = upstox_access_token
        self._index_name = index_name
        self._strike_step = strike_step

    # ── NSE session ────────────────────────────────────────────────

    def _get_nse_session(self) -> requests.Session:
        """Return a fresh NSE session with cookies acquired by a two-step warm-up.

        NSE's anti-bot checks that cookies were set by visiting the option-chain
        page (not just the homepage) before the JSON API will respond.  A new
        session is created on every call so a stale / reset connection never
        carries over to the next retry.
        """
        session = requests.Session()
        session.headers.update(_NSE_HEADERS)
        try:
            # Step 1 — homepage (sets initial cookies)
            session.get(_NSE_BASE, timeout=10)
            time.sleep(2)
            # Step 2 — option-chain page (sets the specific cookie NSE checks)
            session.get(f"{_NSE_BASE}/option-chain", timeout=10)
            time.sleep(1)
        except Exception as exc:
            logger.warning("NSE session warm-up incomplete: %s", exc)
        return session

    # ── Public interface ────────────────────────────────────────────

    def get_spot_data(self) -> SpotData:
        """Fetch live spot price — Upstox first (if configured), then yfinance."""
        if self._upstox_token:
            try:
                return self._get_spot_data_via_upstox()
            except Exception as exc:
                logger.warning(
                    "Upstox spot fetch failed (%s: %s) — falling back to yfinance",
                    type(exc).__name__, exc,
                )

        logger.info("Fetching %s spot data via yfinance", self._index_name)

        def _fetch() -> SpotData:
            ticker = yf.Ticker(self._symbol)
            info = ticker.fast_info
            hist = ticker.history(period="1d", interval="1m")
            if hist.empty:
                raise ValueError("yfinance returned empty history for spot data")
            latest = hist.iloc[-1]
            return SpotData(
                symbol=self._symbol,
                price=float(info.last_price),
                timestamp=datetime.now(tz=timezone.utc),
                open=float(latest.get("Open", 0)),
                high=float(latest.get("High", 0)),
                low=float(latest.get("Low", 0)),
                volume=int(latest.get("Volume", 0)),
            )

        return _retry(_fetch)

    def _get_spot_data_via_upstox(self) -> SpotData:
        from nifty_ai_agent.data.upstox_provider import UpstoxClient

        quote = UpstoxClient(self._upstox_token).get_quote(self._index_name)
        logger.info("%s spot fetched via Upstox (live): %.2f", self._index_name, quote["price"])
        return SpotData(
            symbol=self._symbol,
            price=quote["price"],
            timestamp=datetime.now(tz=timezone.utc),
            open=quote["open"],
            high=quote["high"],
            low=quote["low"],
            volume=int(quote["volume"]),
        )

    def get_option_chain(self) -> OptionChainData:
        """Fetch the live option chain; fall back to a VIX-based synthetic estimate.

        Four tiers, cheapest/most-reliable first:
          1. Upstox API (if UPSTOX_ACCESS_TOKEN is configured) — a real authenticated
             REST API, no bot detection to route around.
          2. Plain HTTP session against NSE (fast, ~1-2s) — works if Akamai isn't
             blocking us today.
          3. Headless-browser fetch (slower, ~5-15s) — a real Chromium instance loads
             nseindia.com so Akamai's bot-check JS runs and sets valid cookies, then
             navigates straight to the JSON endpoint and reads its rendered text.
          4. VIX-based synthetic estimate — last resort if every live path fails.
        """
        logger.info("Fetching %s option chain", self._index_name)

        if self._upstox_token:
            try:
                data = self._fetch_option_chain_via_upstox()
                logger.info("%s option chain fetched via Upstox (live)", self._index_name)
                return data
            except Exception as exc:
                logger.warning(
                    "Upstox option chain fetch failed (%s: %s) — falling back to NSE scrape",
                    type(exc).__name__, exc,
                )

        url = f"{_NSE_BASE}/api/option-chain-indices?symbol={self._index_name}"
        try:
            data = self._fetch_option_chain_json_http(url)
            logger.info("NSE option chain fetched via plain HTTP")
        except Exception as exc:
            logger.info(
                "NSE plain HTTP fetch blocked (%s: %s) — retrying via headless browser",
                type(exc).__name__, exc,
            )
            try:
                data = self._fetch_option_chain_json_browser(url)
                logger.info("NSE option chain fetched via headless browser")
            except Exception as exc2:
                logger.warning(
                    "NSE browser fetch also failed — using VIX-based synthetic chain. (%s: %s)",
                    type(exc2).__name__, exc2,
                )
                return self._synthetic_option_chain()

        return self._parse_option_chain_json(data)

    def _fetch_option_chain_via_upstox(self) -> OptionChainData:
        """Fetch weekly + monthly option chains from Upstox's authenticated API."""
        from nifty_ai_agent.data.upstox_provider import UpstoxClient

        client = UpstoxClient(self._upstox_token)
        expiries_iso = client.get_expiries(self._index_name)
        if not expiries_iso:
            raise RuntimeError("Upstox returned no expiry dates")

        expiry_dates_nse_fmt = [_iso_to_nse_date(d) for d in expiries_iso]
        weekly_expiry, monthly_expiry = _identify_expiries(expiry_dates_nse_fmt)

        weekly_df = client.get_option_chain(self._index_name, _nse_date_to_iso(weekly_expiry))
        monthly_df = pd.DataFrame()
        if monthly_expiry and monthly_expiry != weekly_expiry:
            monthly_df = client.get_option_chain(self._index_name, _nse_date_to_iso(monthly_expiry))

        weekly_pcr = _compute_pcr(weekly_df)
        weekly_max_pain = _compute_max_pain(weekly_df) if not weekly_df.empty else 0.0

        return OptionChainData(
            symbol=self._index_name,
            expiry=weekly_expiry,
            monthly_expiry=monthly_expiry if not monthly_df.empty else "",
            timestamp=datetime.now(tz=timezone.utc),
            strikes=weekly_df,
            monthly_strikes=monthly_df,
            pcr=round(weekly_pcr, 3),
            max_pain=weekly_max_pain,
        )

    def _fetch_option_chain_json_http(self, url: str) -> dict[str, Any]:
        """Fast path: plain `requests` session. Raises on any failure (no retries —
        repeated attempts fail identically while Akamai is blocking this client)."""
        session = self._get_nse_session()
        resp = session.get(url, timeout=15, headers={"Referer": f"{_NSE_BASE}/option-chain"})
        resp.raise_for_status()
        return resp.json()

    def _fetch_option_chain_json_browser(self, url: str) -> dict[str, Any]:
        """Slow path: drive a real headless Chromium so Akamai's JS bot-check passes.

        Navigates to the option-chain page first (to run the sensor script and set
        cookies), then straight to the JSON API URL — Chrome renders raw JSON as
        plain text in the page body, which we parse directly.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "playwright not installed — run `pip install playwright` "
                "and `playwright install chromium`"
            ) from exc

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                context = browser.new_context(user_agent=_BROWSER_USER_AGENT, locale="en-US")
                page = context.new_page()
                page.goto(
                    f"{_NSE_BASE}/option-chain",
                    wait_until="domcontentloaded",
                    timeout=_BROWSER_NAV_TIMEOUT_MS,
                )
                page.wait_for_timeout(_BROWSER_SETTLE_MS)
                page.goto(url, wait_until="domcontentloaded", timeout=_BROWSER_NAV_TIMEOUT_MS)
                text = page.evaluate("document.body.innerText")
                return json.loads(text)
            finally:
                browser.close()

    def _parse_option_chain_json(self, data: dict[str, Any]) -> OptionChainData:
        """Split NSE's raw option-chain JSON into weekly/monthly strike DataFrames."""
        records = data.get("records", {})
        expiry_dates: list[str] = _drop_expiring_today(records.get("expiryDates", []))
        weekly_expiry, monthly_expiry = _identify_expiries(expiry_dates)

        raw_data = records.get("data", [])
        weekly_rows: list[dict] = []
        monthly_rows: list[dict] = []

        for entry in raw_data:
            entry_expiry = entry.get("expiryDate", "")
            strike = entry.get("strikePrice", 0)
            ce = entry.get("CE", {})
            pe = entry.get("PE", {})
            row = {
                "strike": strike,
                "ce_oi": ce.get("openInterest", 0),
                "pe_oi": pe.get("openInterest", 0),
                "ce_ltp": ce.get("lastPrice", 0),
                "pe_ltp": pe.get("lastPrice", 0),
                "ce_iv": ce.get("impliedVolatility", 0),
                "pe_iv": pe.get("impliedVolatility", 0),
            }
            if not entry_expiry or entry_expiry == weekly_expiry:
                weekly_rows.append(row)
            if monthly_expiry and entry_expiry == monthly_expiry:
                monthly_rows.append(row)

        weekly_df = pd.DataFrame(weekly_rows)
        monthly_df = pd.DataFrame(monthly_rows)
        weekly_pcr = _compute_pcr(weekly_df)
        weekly_max_pain = _compute_max_pain(weekly_df) if not weekly_df.empty else 0.0

        logger.info(
            "NSE option chain: weekly=%s (%d strikes)  monthly=%s (%d strikes)",
            weekly_expiry, len(weekly_df), monthly_expiry, len(monthly_df),
        )
        return OptionChainData(
            symbol=self._index_name,
            expiry=weekly_expiry,
            monthly_expiry=monthly_expiry,
            timestamp=datetime.now(tz=timezone.utc),
            strikes=weekly_df,
            monthly_strikes=monthly_df,
            pcr=round(weekly_pcr, 3),
            max_pain=weekly_max_pain,
        )

    def _synthetic_option_chain(self) -> OptionChainData:
        """Build a minimal option chain from spot price + India VIX.

        Used when NSE API is blocked.  OI-based levels (CE wall, PE floor,
        max pain) are absent, but PCR is estimated from VIX so the sentiment
        confidence check still runs.

        VIX → PCR mapping (empirical approximation):
          VIX < 12  → PCR 0.65  (complacent, heavy call writing)
          VIX 12-15 → PCR 0.85  (mild optimism)
          VIX 15-18 → PCR 1.00  (neutral)
          VIX 18-22 → PCR 1.25  (cautious, put buying)
          VIX > 22  → PCR 1.55  (fearful, heavy put writing)
        """
        spot = 0.0
        try:
            spot = float(yf.Ticker(self._symbol).fast_info.last_price)
        except Exception:
            pass

        vix = 15.0
        try:
            vix = float(yf.Ticker("^INDIAVIX").fast_info.last_price)
            logger.info("India VIX %.2f → synthetic PCR", vix)
        except Exception:
            logger.warning("India VIX unavailable — defaulting to 15.0")

        if vix < 12:
            pcr = 0.65
        elif vix < 15:
            pcr = 0.85
        elif vix < 18:
            pcr = 1.00
        elif vix < 22:
            pcr = 1.25
        else:
            pcr = 1.55

        atm = int(round(spot / self._strike_step) * self._strike_step) if spot else 24000
        expiry = _next_weekly_expiry()

        logger.info(
            "%s synthetic chain: VIX=%.2f PCR=%.2f ATM=%d expiry=%s",
            self._index_name, vix, pcr, atm, expiry,
        )
        return OptionChainData(
            symbol=self._index_name,
            expiry=expiry,
            monthly_expiry="",
            timestamp=datetime.now(tz=timezone.utc),
            strikes=pd.DataFrame(),
            monthly_strikes=pd.DataFrame(),
            pcr=round(pcr, 2),
            max_pain=float(atm),
            iv_proxy=round(vix / 100, 4),
        )

    def get_historical_data(self, days: int = 60, interval: str = "5m") -> pd.DataFrame:
        """Fetch OHLCV data — Upstox first (if configured), then yfinance.

        For live intraday signals use interval="5m" (default). Use interval="1d"
        for end-of-day / backtesting workflows.
        """
        if self._upstox_token:
            try:
                from nifty_ai_agent.data.upstox_provider import UpstoxClient
                df = UpstoxClient(self._upstox_token).get_historical_ohlcv(
                    self._index_name, days=days, interval=interval,
                )
                logger.info(
                    "%s OHLCV fetched via Upstox (live): %d bars", self._index_name, len(df),
                )
                return df
            except Exception as exc:
                logger.warning(
                    "Upstox historical fetch failed (%s: %s) — falling back to yfinance",
                    type(exc).__name__, exc,
                )

        logger.info(
            "Fetching %d days of %s OHLCV history via yfinance (interval=%s)",
            days, self._index_name, interval,
        )

        def _fetch() -> pd.DataFrame:
            if interval in ("1m", "2m", "5m", "15m", "30m", "60m", "90m"):
                # yfinance intraday: use period string, not start/end
                period_str = f"{min(days, 59)}d"
                df = yf.download(
                    self._symbol,
                    period=period_str,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                )
            else:
                end = datetime.now(tz=timezone.utc)
                start = end - timedelta(days=days + 10)
                df = yf.download(
                    self._symbol,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                )

            if df.empty:
                raise ValueError("yfinance returned empty historical data")

            # Newer yfinance returns a MultiIndex like ('Close', '^NSEI').
            # Flatten to plain strings before lower-casing.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            df.columns = [c.lower() for c in df.columns]
            df.index.name = "datetime"
            return df

        return _retry(_fetch)


def _next_weekly_expiry() -> str:
    """Return the nearest upcoming Tuesday as 'DD-Mon-YYYY' (NIFTY weekly expiry).

    Confirmed live via the Upstox option-chain API — NSE moved NIFTY's weekly
    (and monthly) expiry off the historical Thursday onto Tuesday.
    """
    today = date.today()
    days_ahead = (1 - today.weekday()) % 7  # 1 = Tuesday
    if days_ahead == 0:
        days_ahead = 7
    return (today + timedelta(days=days_ahead)).strftime("%d-%b-%Y")


def _drop_expiring_today(expiry_dates: list[str]) -> list[str]:
    """Drop any expiry dated today from an NSE-format ('DD-Mon-YYYY') list.

    An option expiring within hours has no meaningful time left to trade, so
    the "weekly" pick should roll forward to the next available expiry
    instead. Falls back to the original list if that would leave nothing.
    """
    today_str = date.today().strftime("%d-%b-%Y")
    filtered = [d for d in expiry_dates if d != today_str]
    return filtered if filtered else expiry_dates


def _iso_to_nse_date(iso_date: str) -> str:
    """Convert 'YYYY-MM-DD' (Upstox) to 'DD-Mon-YYYY' (NSE convention used internally)."""
    return datetime.strptime(iso_date, "%Y-%m-%d").strftime("%d-%b-%Y")


def _nse_date_to_iso(nse_date: str) -> str:
    """Convert 'DD-Mon-YYYY' back to 'YYYY-MM-DD' for Upstox API calls."""
    return datetime.strptime(nse_date, "%d-%b-%Y").strftime("%Y-%m-%d")


def _identify_expiries(expiry_dates: list[str]) -> tuple[str, str]:
    """Return *(weekly_expiry, monthly_expiry)* from the NSE expiry date list.

    weekly  = nearest available expiry (index 0 after sorting by date).
    monthly = last expiry in the same calendar month as weekly, provided it
              differs from weekly.  If weekly IS the last of its month (i.e.
              it is the monthly), return the last expiry of the following month
              as the monthly instead.  Falls back to weekly if no future month
              data is present.
    """
    if not expiry_dates:
        return "", ""

    parsed: list[tuple[date, str]] = []
    for d_str in expiry_dates:
        try:
            d = datetime.strptime(d_str, "%d-%b-%Y").date()
            parsed.append((d, d_str))
        except Exception:
            pass

    if not parsed:
        w = expiry_dates[0]
        return w, w

    parsed.sort(key=lambda x: x[0])
    weekly_date, weekly_str = parsed[0]

    # Build a map: (year, month) → last expiry date in that month
    by_month: dict[tuple[int, int], tuple[date, str]] = {}
    for d, s in parsed:
        key = (d.year, d.month)
        if key not in by_month or d > by_month[key][0]:
            by_month[key] = (d, s)

    # Check whether weekly is the last expiry of its month
    same_month_key = (weekly_date.year, weekly_date.month)
    same_month_last_date, same_month_last_str = by_month[same_month_key]

    if same_month_last_date != weekly_date:
        # Monthly is a later date in the same calendar month
        return weekly_str, same_month_last_str

    # Weekly IS the last of its month — look for next month's monthly expiry
    future_months = sorted(k for k in by_month if k > same_month_key)
    if future_months:
        _, monthly_str = by_month[future_months[0]]
        return weekly_str, monthly_str

    return weekly_str, weekly_str  # fallback: no future month data


def _compute_pcr(df: pd.DataFrame) -> float:
    """Put-Call Ratio from a strikes DataFrame."""
    if df.empty or "ce_oi" not in df.columns:
        return 0.0
    total_ce = df["ce_oi"].sum()
    total_pe = df["pe_oi"].sum()
    return total_pe / total_ce if total_ce > 0 else 0.0


def _compute_max_pain(df: pd.DataFrame) -> float:
    """Calculate max pain strike price."""
    if df.empty or "strike" not in df.columns:
        return 0.0
    strikes = df["strike"].tolist()
    min_pain = float("inf")
    max_pain_strike = 0.0
    for s in strikes:
        pain = 0.0
        for _, row in df.iterrows():
            k = row["strike"]
            pain += row["ce_oi"] * max(0, k - s)
            pain += row["pe_oi"] * max(0, s - k)
        if pain < min_pain:
            min_pain = pain
            max_pain_strike = s
    return float(max_pain_strike)
