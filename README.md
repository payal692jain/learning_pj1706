# NIFTY + SENSEX AI Signal Agent

An AI-assisted trading signal agent for Indian markets — **NIFTY 50** and **BSE SENSEX** weekly **and monthly** expiry options.
Runs two independent strategies (EMA Crossover + VWAP Breakout) every 5 minutes during market hours, each scored through a multi-layer confidence pipeline (RSI analysis + option chain OI levels + heavyweight breadth), fetches **live** option chain premiums via the Upstox API (with automatic fallback to an NSE scrape and finally a VIX-based estimate if live data is unavailable), sends a full pre-market morning brief at 8 AM, and delivers every prediction to your iPhone via Pushover notifications and a Streamlit dashboard.

> **No automated order placement. Signal generation only.**

---

## Table of Contents

1. [What This Does](#what-this-does)
2. [Tech Stack](#tech-stack)
3. [Architecture & Flow Diagram](#architecture--flow-diagram)
4. [Component Details](#component-details)
5. [Signal Confidence Scoring](#signal-confidence-scoring)
6. [Setup & Installation](#setup--installation)
7. [How to Use](#how-to-use)
8. [What You Receive](#what-you-receive)
9. [Book-Grounded AI Knowledge Base](#book-grounded-ai-knowledge-base)
10. [Future Improvements](#future-improvements)

---

## What This Does

```
Every morning at 8 AM (pre-market):
  → Fetches global indices (S&P 500, Dow, Nasdaq, Nikkei, Hang Seng, FTSE, DAX, India VIX)
  → Fetches GIFT Nifty pre-market price
  → Fetches all 50 NIFTY constituent stocks (advance/decline ratio, top movers)
  → Fetches WEEKLY expiry option chain (CE/PE table, max pain, PCR, OI support/resistance)
  → Fetches MONTHLY expiry option chain (structural levels, monthly max pain, PCR)
  → Fetches Indian + global market news (RSS feeds)
  → Claude AI synthesises everything into a daily trading plan
  → Sends 5–6 Pushover notifications to your iPhone

Every 5 minutes during market hours (9:15 AM – 3:30 PM) — for BOTH NIFTY and SENSEX:
  → Fetches live spot price
  → Fetches 5-minute intraday OHLCV bars (10 days ≈ 700 candles)
  → Calculates RSI, EMA20, EMA50, MACD, ATR, VWAP
  → Fetches the live option chain (see "Live Option Chain Data" below) — weekly AND
    monthly premiums, skipping any contract expiring today in favour of the next one

  SIGNAL GENERATION — every strategy runs independently, every cycle:
  → EMA Crossover strategy   → raw BUY_CE / BUY_PE / HOLD signal
  → VWAP Breakout strategy   → raw BUY_CE / BUY_PE / HOLD signal

  MULTI-LAYER CONFIDENCE SCORING (applied separately to each strategy's signal):
  Layer 1 — RSI Analysis
    · Zone: DEEPLY_OVERSOLD / OVERSOLD / NEUTRAL / OVERBOUGHT / DEEPLY_OVERBOUGHT
    · Trend: RSI rising or falling over last 5 bars
    · Divergence: price/RSI bearish or bullish divergence detection
    → Adjusts confidence −12 to +11 points

  Layer 2 — Weekly Option Chain Filter (cached 15 min)
    · PCR direction vs signal
    · Spot proximity to CE OI resistance wall
    · Spot proximity to PE OI support floor
    · Max pain pinning risk on expiry day
    → Adjusts confidence −22 to +9 points

  Layer 3 — Monthly Option Chain Note (cached 15 min)
    · Monthly PCR structural bias
    · Monthly CE OI wall proximity
    · Monthly PE OI floor proximity
    · Monthly max pain pinning (DTE ≤ 3)
    → Adjusts confidence −14 to +3 points

  Layer 4 — Heavyweight Breadth (live, top 10 stocks)
    · Advance/decline among HDFCBANK, Reliance, ICICI, Infy, TCS, L&T, Kotak, Axis, SBI, Airtel
    → Adjusts confidence −12 to +8 points

  → Every strategy's final signal saved to SQLite, sent via Pushover (one message
    listing all predictions side by side), shown on dashboard
  → Claude AI explains each signal using concepts from your trading book library
```

### Live Option Chain Data (weekly + monthly, both indices)

Real option premiums are hard to get reliably — NSE's website sits behind Akamai bot
detection, and there is no official public BSE option chain API. The agent tries three
tiers, cheapest/most-reliable first, and always shows which tier produced the price:

```
1. Upstox API (if UPSTOX_ACCESS_TOKEN is set)  — real authenticated REST API, live LTP/OI
2. NSE plain HTTP scrape                       — fast, works when Akamai isn't blocking today
3. NSE headless-browser scrape (Playwright)    — a real Chromium loads nseindia.com so
                                                  Akamai's bot-check JS passes, then reads
                                                  the JSON endpoint directly
4. VIX-based synthetic estimate (last resort)  — Black-Scholes price from spot + India VIX;
                                                  notifications label this "(Est.)" so it's
                                                  never mistaken for a real traded premium
```
Tiers 2–4 apply to NIFTY only (BSE has no scrape-based middle tier — SENSEX goes straight
from Upstox to the synthetic estimate if Upstox isn't configured). Whichever contract is
nearest to expiring **today is skipped** — an option with hours left to trade isn't useful
for a fresh signal, so "weekly" always means the *next* available expiry.

---

## Tech Stack

| Layer | Library / Service | Purpose |
|-------|------------------|---------|
| **Language** | Python 3.10+ | Primary language |
| **Data Models** | Pydantic v2, Dataclasses | Type-safe structured data |
| **Configuration** | pydantic-settings, python-dotenv | `.env` file management |
| **Market Data** | yfinance | NIFTY/SENSEX OHLCV, global indices, NIFTY 50 stocks |
| **Option Chain (live)** | Upstox API | Real weekly + monthly LTP/OI for NIFTY *and* SENSEX |
| **Option Chain (fallback)** | NSE India API, Playwright | HTTP scrape, then headless-browser scrape if Akamai blocks |
| **Option Chain (last resort)** | Black-Scholes (`math.erf`) | VIX-based theoretical price, labelled "(Est.)" |
| **News** | feedparser | RSS from ET Markets, Moneycontrol, Reuters, Livemint |
| **Indicators** | pandas, numpy | RSI, EMA, MACD, ATR, VWAP (stateless) |
| **Strategies** | Custom, run independently | EMA20/50 Crossover + RSI confirmation; VWAP Breakout + momentum |
| **Confidence Scoring** | 4-layer system, per strategy | RSI analysis + weekly OC + monthly OC + breadth |
| **Risk** | ATR-based | SL/Target/RR calculator |
| **AI** | anthropic SDK | Claude `claude-opus-4-8` — signal explanation + daily plan |
| **Knowledge Base** | Hardcoded patterns | 23 patterns from 6 trading books |
| **Database** | SQLAlchemy 2.0, SQLite | Signals (one row per strategy per cycle), market data, trades |
| **Notifications** | Pushover HTTP API | iPhone push alerts, one message listing every strategy's prediction |
| **Dashboard** | Streamlit, Plotly | Live candlestick + indicator chart, per-strategy prediction cards |
| **Scheduler** | schedule | 8 AM daily + 5-min intraday loop, both indices |
| **Timezone** | pytz | IST market hours enforcement |
| **Testing** | pytest, pytest-cov | ~79% coverage, 250+ tests |
| **Version Control** | Git | `.env`, `.venv/`, `.idea/` excluded via `.gitignore` |

---

## Architecture & Flow Diagram

```
╔══════════════════════════════════════════════════════════════════════╗
║               NIFTY + SENSEX AI SIGNAL AGENT                        ║
╚══════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────┐
│                        SCHEDULE LAYER                               │
│  ┌──────────────────────┐      ┌──────────────────────────────┐    │
│  │  08:00 AM IST Daily  │      │  Every 5 min (9:15–15:30)   │    │
│  │   Morning Report     │      │   Intraday Signal Pipeline   │    │
│  └──────────┬───────────┘      └──────────────┬───────────────┘    │
└─────────────┼────────────────────────────────┼────────────────────-┘
              │                                │
              ▼                                ▼
╔═════════════════════════╗    ╔══════════════════════════════════════╗
║    MORNING REPORT       ║    ║        INTRADAY PIPELINE             ║
╠═════════════════════════╣    ╠══════════════════════════════════════╣
║                         ║    ║                                      ║
║  Global Indices         ║    ║  ┌──────────────────────────────┐   ║
║  (yfinance, 8 markets)  ║    ║  │ DataProvider (NSE+BSE)       │   ║
║         +               ║    ║  │ · Spot+OHLCV: yfinance       │   ║
║  GIFT Nifty             ║    ║  │ · Options: Upstox→NSE→VIX est│   ║
║  (NSE IFSC API)         ║    ║  └──────────────┬───────────────┘   ║
║         +               ║    ║                 ▼                    ║
║  NIFTY 50 Stocks        ║    ║  ┌──────────────────────────────┐   ║
║  · Advance/Decline      ║    ║  │     Indicator Engine         │   ║
║  · Top 5 gainers        ║    ║  │  EMA20, EMA50, RSI(14)       │   ║
║  · Top 5 losers         ║    ║  │  MACD(12,26,9), ATR(14)      │   ║
║         +               ║    ║  │  VWAP                        │   ║
║  WEEKLY Option Chain    ║    ║  └──────────────┬───────────────┘   ║
║  · ATM ± 3 strikes      ║    ║                 ▼                    ║
║  · CE/PE LTP & OI       ║    ║  ┌──────────────────────────────┐   ║
║  · PCR, Max Pain        ║    ║  │  Strategy Engine (2, indep.) │   ║
║  · OI resistance/support║    ║  │  1. EMA Crossover (EMA/RSI)  │   ║
║  · Black-Scholes CE/PE  ║    ║  │  2. VWAP Breakout (VWAP/mom.)│   ║
║         +               ║    ║  │  Each: BUY_CE/BUY_PE/HOLD    │   ║
║  MONTHLY Option Chain   ║    ║  │  Confidence: 50–95% each     │   ║
║  · ATM ± 5 strikes      ║    ║  └──────────────┬───────────────┘   ║
║  · Monthly max pain     ║    ║                 ▼                    ║
║  · Monthly PCR          ║    ║  ┌──────────────────────────────┐   ║
║  · Monthly OI wall/floor║    ║  │   Layer 1: RSI Analyser      │   ║
║         +               ║    ║  │  · Zone (5 levels)           │   ║
║  News (RSS, 4 feeds)    ║    ║  │  · Trend (5-bar lookback)    │   ║
║         +               ║    ║  │  · Divergence detection      │   ║
║  Claude Daily Plan      ║    ║  │  Δ confidence: −12 to +11   │   ║
║  (if API key set)       ║    ║  └──────────────┬───────────────┘   ║
║                         ║    ║                 ▼                    ║
║  Pushover: 5–6 msgs     ║    ║  ┌──────────────────────────────┐   ║
║  ① Global + GIFT Nifty  ║    ║  │   Layer 2: Weekly OC Filter  │   ║
║  ② NIFTY 50 A/D         ║    ║  │  · PCR direction vs signal   │   ║
║  ③ Weekly OC            ║    ║  │  · CE OI wall proximity      │   ║
║  ④ Monthly OC           ║    ║  │  · PE OI floor proximity     │   ║
║  ⑤ News (silent)        ║    ║  │  · Max pain pinning risk     │   ║
║  ⑥ Claude Plan (HIGH)   ║    ║  │  Cached 15 min (NSE limits)  │   ║
║                         ║    ║  │  Δ confidence: −22 to +9    │   ║
╚═════════════════════════╝    ║  └──────────────┬───────────────┘   ║
                               ║                 ▼                    ║
                               ║  ┌──────────────────────────────┐   ║
                               ║  │  Layer 3: Monthly OC Note    │   ║
                               ║  │  · Monthly PCR bias          │   ║
                               ║  │  · Monthly CE wall proximity │   ║
                               ║  │  · Monthly PE floor          │   ║
                               ║  │  · Monthly pin risk (DTE≤3)  │   ║
                               ║  │  Δ confidence: −14 to +3    │   ║
                               ║  └──────────────┬───────────────┘   ║
                               ║                 ▼                    ║
                               ║  ┌──────────────────────────────┐   ║
                               ║  │  Layer 4: Breadth Check      │   ║
                               ║  │  Top 10 heavyweights live:   │   ║
                               ║  │  HDFCBANK, Reliance, ICICI   │   ║
                               ║  │  Infy, TCS, L&T, Kotak       │   ║
                               ║  │  Axis, SBI, Bharti Airtel    │   ║
                               ║  │  Advance/Decline → score     │   ║
                               ║  │  Δ confidence: −12 to +8    │   ║
                               ║  └──────────────┬───────────────┘   ║
                               ║                 ▼                    ║
                               ║  ┌──────────────────────────────┐   ║
                               ║  │      Risk Management         │   ║
                               ║  │  SL  = Entry ∓ 1.5 × ATR    │   ║
                               ║  │  Tgt = Entry ± 3.0 × ATR    │   ║
                               ║  │  RR ≥ 1:2  |  Max risk 1%   │   ║
                               ║  └──────────────┬───────────────┘   ║
                               ║                 ▼                    ║
                               ║  ┌──────────────────────────────┐   ║
                               ║  │   AI Explainer (optional)    │   ║
                               ║  │  4 book patterns (RAG)       │   ║
                               ║  │  All 4 layer deltas visible  │   ║
                               ║  │  Claude explains in context  │   ║
                               ║  └──────────────┬───────────────┘   ║
                               ║                 ▼                    ║
                               ║  SQLite ──────── Pushover ── Dashboard║
                               ╚══════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────┐
│                  STREAMLIT DASHBOARD (always on)                     │
│  Candlestick + EMA20/50 · RSI panel · MACD panel                    │
│  Signal box (colour-coded) · Entry/SL/Target/RR                      │
│  AI explanation · Indicator snapshot · Signal history                │
│  Auto-refresh every 60 seconds                                       │
└──────────────────────────────────────────────────────────────────────┘
```

> The Intraday Pipeline box runs **twice** every 5 minutes — once for NIFTY
> (`NSEDataProvider`, ₹50 strikes, weekly expiry Tuesday) and once for SENSEX
> (`BSEDataProvider`, ₹100 strikes, weekly expiry Thursday) — and within each
> run, both strategies fire independently, so every cycle produces up to
> **4 predictions** (2 indices × 2 strategies).

---

## Component Details

### 1. Configuration (`nifty_ai_agent/config.py`)
Pydantic Settings loads all config from `.env`. Single `get_settings()` singleton cached with `@lru_cache`. `ANTHROPIC_API_KEY` is optional — AI features degrade gracefully if unset.

---

### 2. Data Layer (`nifty_ai_agent/data/`)

#### `upstox_provider.py` — UpstoxOptionChainClient (live data, both indices)
The primary source for real option chain data — an authenticated REST API, so there's
no bot detection to work around.
- **`get_expiries(index_name)`** — fetches available expiry dates from `/v2/option/contract`. Instrument keys: `NSE_INDEX|Nifty 50` (NIFTY), `BSE_INDEX|SENSEX` (SENSEX).
- **`get_option_chain(index_name, expiry_date)`** — fetches strike-level OI/LTP/IV from `/v2/option/chain`, normalised into the same `strike/ce_oi/pe_oi/ce_ltp/pe_ltp/ce_iv/pe_iv` shape the rest of the pipeline expects.
- **`drop_expiring_today()`** — an option expiring within hours has no meaningful time left to trade, so any expiry dated today is dropped before picking "weekly" — it always rolls forward to the next available expiry.
- Access tokens expire nightly (~3:30 AM IST) — see [`scripts/upstox_login.py`](#setup--installation) for the daily refresh flow.

#### `nse_provider.py` — NSEDataProvider (NIFTY)
- **`get_spot_data()`** — yfinance `fast_info.last_price` for live price
- **`get_historical_data(days, interval)`** — yfinance 5-minute bars (10 days ≈ 700 candles). Auto-flattens yfinance MultiIndex columns.
- **`get_option_chain()`** — four-tier fallback, cheapest/most-reliable first:
  1. **Upstox** (if `UPSTOX_ACCESS_TOKEN` is set) — real live weekly + monthly LTP/OI
  2. **Plain HTTP** against NSE's API — fast, works if Akamai isn't blocking today
  3. **Headless-browser scrape** (Playwright) — loads `nseindia.com/option-chain` for real so Akamai's bot-check JS runs and sets valid cookies, then reads the JSON endpoint's rendered text directly
  4. **VIX-based synthetic estimate** — Black-Scholes theoretical price from spot + India VIX; marked `is_live=False` so notifications label it "(Est.)"
  - `_identify_expiries()` scans the expiry dates list (after dropping any expiring today), groups by calendar month, and returns:
    - **weekly** = nearest remaining expiry
    - **monthly** = last expiry of the same calendar month (if different from weekly), or last expiry of the following month if weekly IS the monthly
  - Computes PCR and max pain separately for each expiry

#### `bse_provider.py` — BSEDataProvider (SENSEX)
Same shape as `NSEDataProvider`, but there's no scrape-based middle tier — BSE has no
reliable public option chain API to fall back on, so it's Upstox or straight to the
VIX-based synthetic estimate.

#### `market_context.py` — Global Indices + GIFT Nifty
8 global indices via yfinance + GIFT Nifty from NSE IFSC API. Computes weighted `global_bias`.

#### `news_fetcher.py` — RSS News
feedparser on 4 feeds: ET Markets, Moneycontrol, Reuters, Livemint.

#### `nifty50_stocks.py` — Constituent Movers
Bulk `yf.download()` for all 50 stocks → advance/decline/unchanged count + top 5 gainers/losers.

#### `breadth.py` — Real-Time Heavyweight Breadth
Fetches live price vs previous close for the **top 10 NIFTY 50 heavyweights** (by index weight):

| Symbol | Approx weight |
|--------|--------------|
| HDFCBANK | ~13% |
| RELIANCE | ~10% |
| ICICIBANK | ~8% |
| INFY | ~6% |
| TCS | ~4% |
| LT | ~4% |
| KOTAKBANK | ~3.5% |
| AXISBANK | ~3% |
| SBIN | ~3% |
| BHARTIARTL | ~2.5% |

Returns `BreadthSnapshot(advancing, declining, unchanged, score, bias, leaders, laggards)`.
A stock is counted as advancing if it is up ≥ 0.15% from previous close (declining if down ≥ 0.15%).

---

### 3. Indicator Engine (`nifty_ai_agent/indicators/`)
All stateless pure functions — take a DataFrame, return a DataFrame with an extra column.

| Function | Column(s) added | Notes |
|----------|----------------|-------|
| `compute_rsi(df, 14)` | `rsi` | EWM smoothing; RSI=100 when all moves upward |
| `compute_ema(df, [20,50])` | `ema_20`, `ema_50` | `adjust=False` EWM |
| `compute_macd(df, 12, 26, 9)` | `macd`, `macd_signal`, `macd_histogram` | Standard MACD |
| `compute_atr(df, 14)` | `atr` | Wilder's smoothing of True Range |
| `compute_vwap(df)` | `vwap` | Cumulative (typical price × volume) / volume |

---

### 4. Strategy Engine (`nifty_ai_agent/strategies/`)

Both strategies below run **independently, every cycle** — neither one "wins"; each
produces its own signal, gets its own confidence score, and is saved/notified separately
(`main.py`'s `_STRATEGIES` list). Adding a third strategy is a one-line addition to that list.

#### `ema_crossover.py` — EMA Crossover
```
BUY_CE  →  EMA20 > EMA50  AND  RSI > 60
BUY_PE  →  EMA20 < EMA50  AND  RSI < 40
HOLD    →  all other
```
Base confidence 50–95% from EMA separation % and RSI distance from threshold.

#### `vwap_breakout.py` — VWAP Breakout
```
BUY_CE  →  close > VWAP by ≥ 0.15%  AND  price rising over the last 3 bars
BUY_PE  →  close < VWAP by ≥ 0.15%  AND  price falling over the last 3 bars
HOLD    →  all other
```
Base confidence 50–95% from breakout distance from VWAP and momentum strength.

#### `rsi_analyser.py` — RSI Analysis (Layer 1)
Goes beyond the binary threshold with 3 independent checks:

**Zone classification:**
| RSI | Zone |
|-----|------|
| < 30 | DEEPLY_OVERSOLD |
| 30–40 | OVERSOLD |
| 40–60 | NEUTRAL |
| 60–70 | OVERBOUGHT |
| > 70 | DEEPLY_OVERBOUGHT |

**Trend:** compares RSI now vs 5 bars ago → RISING (Δ > 2.5) / FALLING (Δ < −2.5) / FLAT

**Divergence:** over last 14 bars —
- Price higher high + RSI lower high = BEARISH_DIV (−12 on BUY_CE)
- Price lower low + RSI higher low = BULLISH_DIV (−12 on BUY_PE)

#### `option_analyser.py` — Option Chain Analysis (Layers 2 & 3)

**Weekly OC filter** — applied every intraday run (15-min cached):
1. PCR direction (> 1.2 = bullish, < 0.8 = bearish): ±5
2. CE OI resistance proximity (wall < 0.5% away: −12; 0.5–1.2%: −5; > 1.2%: +4)
3. PE OI support proximity (floor < 0.5% below: −10; > 1.5%: +4)
4. Max pain pinning (DTE ≤ 1, within 0.3%): −10

**Monthly OC note** — lighter structural filter (same 15-min cache):
1. Monthly PCR structural bias (> 1.5 confirms BUY_CE: +3; < 0.7 confirms BUY_PE: +3)
2. Monthly CE wall proximity (< 0.5%: −8; 0.5–1%: −3)
3. Monthly PE floor proximity (< 0.5%: −6; 0.5–1%: −2)
4. Monthly max pain pinning (DTE ≤ 3, within 0.5%): −6

**Weekly CE/PE table** (ATM ± 3 strikes) is shown in the morning report.
**Monthly key levels only** (no per-strike table) to keep Pushover messages concise.

**Max pain** — the strike that minimises total OI-weighted loss to all option buyers (where the market tends to pin near expiry).

**Black-Scholes pricing** — ATM CE/PE theoretical prices using actual chain IV. Implemented with `math.erf` (no scipy dependency). Also used as the last-resort synthetic estimate when no live chain is available (weekly *and* monthly, both marked `is_live=False`).

**`ExpiryAnalysis.is_live`** — `True` when the price came from Upstox or the NSE scrape, `False` for the VIX-based synthetic estimate. The notification layer uses this to label prices "(Est.)" so a theoretical estimate is never mistaken for a real traded premium.

---

### 5. Risk Management (`nifty_ai_agent/risk/calculator.py`)
```
Stop Loss  = Entry ∓ (ATR × 1.5)
Target     = Entry ± (ATR × 3.0)
RR Ratio   ≥ 1:2
Max risk   ≤ 1% of capital
Daily loss ≤ 3% of capital
```
Returns `is_valid=False` with `rejection_reason` if rules are violated. ATR-based stops widen automatically during high-VIX regimes.

---

### 6. AI Layer (`nifty_ai_agent/ai/`)

#### `knowledge_base.py` — 23 Book Patterns
| Book | Key Patterns Encoded |
|------|---------------------|
| Al Brooks — Trading Price Action Trends | With-Trend Bar, EMA Pullback, Breakout Pullback, Bull/Bear Flag, Climax Reversal, Two-Legged Pullback |
| Anna Coulling — Volume Price Analysis | High Volume Confirmation, No-Demand Bar, Selling Climax, Effort vs Result |
| Thomas Bulkowski — Encyclopedia of Chart Patterns | Bull Flag (54%), Double Bottom (64%), H&S Top (93%), Ascending Triangle (68%) |
| Adam Grimes — Art and Science of TA | Pullback in Trend, Failed Breakout, Momentum Divergence |
| Martin Pring — Price Action | S/R Reversal, Trendline Break + Momentum |
| Bob Volman — Forex Price Action Scalping | Tight Range Breakout, Round Number Barrier |

`get_relevant_patterns()` selects up to 4 matching signal direction, rotating across books.

#### `explainer.py` — Signal Explainer
Claude `claude-opus-4-8` with adaptive thinking + streaming. Receives:
- Signal + all 4 layer confidence deltas in the reason string
- RSI zone, trend, divergence
- Weekly PCR, max pain, CE resistance, PE support
- Monthly PCR, max pain, CE wall, PE floor
- Breadth score, advances, declines
- 4 matched book patterns

Returns a 3–5 sentence explanation grounded in book concepts. Falls back to raw reason on API failure.

#### `morning_analyser.py` — Daily Trading Plan
Structured prompt produces 5-section output under 180 words:
1. BIAS — one word + reason
2. KEY LEVELS — resistance, support, weekly max pain
3. BUY_CE TRIGGER — exact condition
4. BUY_PE TRIGGER — exact condition
5. RISK TO WATCH — main invalidation scenario

Also receives **monthly option chain levels** (monthly max pain, PCR, CE wall, PE floor) as additional context for Claude when setting key levels.

---

### 7. Database Layer (`nifty_ai_agent/database/`)
SQLite via SQLAlchemy 2.0 ORM.

```
market_data          signals                        trades
──────────────       ──────────────────────────     ─────────────────
datetime (idx)       datetime (idx)                 signal_id
symbol               signal (BUY_CE/PE/HOLD)        entry_price
open/high/low        confidence (0–100)             exit_price
close/volume         strategy                       pnl
                     reason (full multi-layer text) strategy
                     ai_explanation                 result
                     entry_price / stop_loss        entry_time / exit_time
                     target / risk_reward
                     status
```

---

### 8. Notification Layer (`nifty_ai_agent/notifier/pushover.py`)
Direct Pushover HTTP API — no extra SDK.

**`send_multi_signal()`** sends one notification per cycle per index, listing every
strategy's prediction side by side — title summarises all of them
(`🔔 NIFTY — EMA_Crossover: BUY_CE 82% | VWAP_Breakout: HOLD 50%`), and the body has a
section per strategy with its own contract line(s), risk levels, and reasoning.

**Contract line(s)** show weekly *and* monthly together whenever both are available,
each labelled with its source:
```
📌 Buy (Weekly): NIFTY 24400 CE  14-Jul-2026  @ ₹142
📌 Buy (Monthly): NIFTY 24400 CE  28-Jul-2026  @ ₹351
```
If a price came from the VIX-based synthetic estimate instead of a live chain, the
label becomes `(Weekly, Est.)` / `(Monthly, Est.)`. Sub-₹10 premiums (common for options
near expiry) show two decimals instead of rounding to a misleading "₹0".

| Message | Priority | Behaviour |
|---------|----------|-----------|
| Any strategy signals BUY_CE/BUY_PE | 0 (Normal) | Sound + lock screen |
| All strategies HOLD | −1 (Low) | Silent |
| Claude daily plan | +1 (High) | Bypasses Do Not Disturb; requires acknowledgement |
| Monthly OC (morning report) | 0 (Normal) | Sound + lock screen |
| Startup ping | 0 (Normal) | Agent alive confirmation |

3 retries with 2-second backoff. Returns `False` on total failure — never crashes.

---

### 9. Dashboard (`dashboard/app.py`)
Streamlit + Plotly, reads live from SQLite.

- **One prediction card per strategy**, side by side — signal, confidence, entry/SL/target/RR, and AI reasoning shown independently for EMA Crossover and VWAP Breakout
- 3-panel Plotly chart: Candlestick + EMA20/50, RSI (with 60/40 lines), MACD histogram
- Signal history table (last 50) with a **Strategy column and filter**
- Auto-refresh every 60 seconds

---

## Signal Confidence Scoring

Every BUY signal's confidence — from **either** strategy, scored independently — is
adjusted by the same 4 layers before it fires:

```
Base confidence (EMA Crossover OR VWAP Breakout)  50 – 95%
  + Layer 1: RSI analysis                          −12 to +11
  + Layer 2: Weekly option chain                    −22 to  +9
  + Layer 3: Monthly option chain                   −14 to  +3
  + Layer 4: Breadth (10 heavyweights)               −12 to  +8
  ─────────────────────────────────────────────────────────────
Final confidence (clamped 10–95%)
```

**Example — BUY_CE at 9:30 AM:**
```
Base:             68%   EMA20 > EMA50 with RSI 67
RSI:             +7    RSI OVERBOUGHT zone, RISING trend
Weekly OC:       +4    PCR 1.26 (bullish), CE wall 1.8% away (room to run)
Monthly OC:      −3    Monthly PCR 0.71 (bearish backdrop)
Breadth:         +8    7/10 heavyweights advancing (HDFCBANK, INFY, TCS leading)
─────────────────────
Final:            84%
Reason: EMA20 (24,120) > EMA50 (24,050) with RSI at 67.1. RSI OVERBOUGHT
        zone: RSI 67.1 — overbought zone, momentum confirmed; RSI trending up.
        OC (27-Jun, PCR 1.26): PCR 1.26 (high put writing — bullish); open air
        to CE resistance at 24,400 (1.8% away). [Monthly 31-Jul: Monthly PCR 0.71
        — bearish backdrop against CE]. Breadth confirms: 7/10 heavyweights
        advancing (HDFCBANK, INFY, TCS).
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- Git
- Pushover account (pushover.net — $5 one-time licence after 30-day trial)
- Anthropic API key (console.anthropic.com — optional)
- Upstox trading account + developer app (upstox.com — free; optional but strongly recommended for live option prices instead of the synthetic estimate)

### Step 1 — Install
```bash
cd c:\my_learnings\market_analysyis
pip install -r requirements.txt
playwright install chromium   # one-time — used as the NSE-scrape fallback tier
```

### Step 2 — Configure
```bash
copy .env.example .env
```

Edit `.env`:
```env
# Required
PUSHOVER_USER_KEY=your_user_key        # from pushover.net account page
PUSHOVER_API_TOKEN=your_app_token      # create an app at pushover.net

# Optional — enables AI explanations and daily plan
ANTHROPIC_API_KEY=sk-ant-...           # from console.anthropic.com

# Optional — enables live weekly + monthly option chain prices for NIFTY + SENSEX.
# Without these, the agent falls back to an NSE scrape and finally a VIX-based
# estimate (clearly labelled "(Est.)" in notifications).
UPSTOX_API_KEY=your_api_key            # from account.upstox.com/developer/apps
UPSTOX_API_SECRET=your_api_secret
UPSTOX_REDIRECT_URI=https://www.google.com/upstox-callback  # must be https://, not localhost
UPSTOX_ACCESS_TOKEN=                   # filled in daily — see Step 3

# Market data defaults (fine as-is)
NIFTY_SYMBOL=^NSEI
SENSEX_SYMBOL=^BSESN
DATA_FETCH_INTERVAL_MINUTES=5
DATA_INTERVAL=5m
HISTORICAL_DAYS=10
```

### Step 3 — Refresh the Upstox token (skip if not using Upstox)
Upstox access tokens expire nightly (~3:30 AM IST) — there's no long-lived refresh
token, so this runs once each morning before market open:
```bash
python scripts/upstox_login.py
# Opens the Upstox login page — log in, then paste back the redirected URL.
# Writes a fresh UPSTOX_ACCESS_TOKEN into .env automatically.
```

### Step 4 — Run
```bash
# Terminal 1 — signal agent + morning reports (both NIFTY and SENSEX)
python main.py

# Terminal 2 — dashboard
streamlit run dashboard/app.py
# Opens at http://localhost:8501
```

### Step 5 — Add to iPhone home screen
Safari → `http://<your-laptop-ip>:8501` → Share → Add to Home Screen

### Step 6 — Auto-start on Windows (optional)
`Win + R` → `shell:startup` → create `start_agent.bat`:
```bat
@echo off
cd C:\my_learnings\market_analysyis
python main.py
```

---

## How to Use

### Daily workflow
```
07:55 AM  →  Ensure python main.py is running
             (If using Upstox: run scripts/upstox_login.py first — token expired overnight)
08:00 AM  →  Pushover messages arrive (NIFTY):
              ① Global bias + GIFT Nifty level
              ② NIFTY 50 advance/decline + top movers
              ③ Weekly option chain: CE/PE table, max pain, PCR, OI levels
              ④ Monthly option chain: structural levels, monthly max pain
              ⑤ News headlines (silent)
              ⑥ Claude daily plan (HIGH priority, if API key set)

09:15 AM  →  Market opens; 5-min signal loop begins for NIFTY and SENSEX
              Each cycle → both EMA Crossover and VWAP Breakout fire independently
              Any strategy signalling BUY_CE/BUY_PE → sound alert, full breakdown
              All strategies HOLD → silent (agent heartbeat)

03:30 PM  →  Market closes; signal loop stops automatically
16:00 PM  →  EOD prediction run (after-hours outlook for next session)
```

### Reading a signal notification
```
🔔 NIFTY — EMA_Crossover: BUY_CE 84% | VWAP_Breakout: HOLD 50%

📈 EMA_Crossover — BUY_CE (84%)
📌 Buy (Weekly): NIFTY 24400 CE  14-Jul-2026  @ ₹142
📌 Buy (Monthly): NIFTY 24400 CE  28-Jul-2026  @ ₹351
Entry 24,120  SL 23,984  Target 24,390  RR 1:2.0
EMA20 (24,120) > EMA50 (24,050) RSI 67.1 — bullish momentum.
RSI (Overbought): RSI 67.1 overbought zone, momentum confirmed; RSI trending up.
OC (14-Jul, PCR 1.26): PCR 1.26 bullish; open air to CE wall at 24,400 (1.8% away).
[Monthly 28-Jul: Monthly PCR 0.71 — bearish backdrop].
Breadth confirms: 7/10 heavyweights advancing (HDFCBANK, INFY, TCS).

Analysis: EMA20 crossed above EMA50 on an expanding bullish bar — Al Brooks
calls this an EMA Pullback Entry. PCR above 1.2 indicates aggressive put
writing at lower strikes, confirming institutional bullish bias.
Key risk: if RSI turns below 60 on the next bar, momentum is fading — exit.
────────────────────────
⏸ VWAP_Breakout — HOLD (50%)
[No trade — HOLD]
No confirmed VWAP breakout. Close=24,120, VWAP=24,095 (+0.10%).
```
Both weekly and monthly prices come from whichever tier actually produced them (Upstox
live → NSE scrape → synthetic) — if it's a theoretical estimate rather than a real
traded premium, the label reads `(Weekly, Est.)` / `(Monthly, Est.)` instead.

### Run tests
```bash
python -m pytest tests/ -v
# 250+ tests, ~79% coverage
```

---

## What You Receive

### Without any optional keys (free)
- Morning report: global markets, NIFTY 50 A/D, weekly OC, monthly OC, news
- 5-min intraday signals for NIFTY + SENSEX, 2 strategies each, full 4-layer confidence scoring
- Option chain prices via NSE scrape / VIX-based synthetic estimate (clearly labelled if not live)
- Risk parameters (SL, target, RR)
- Streamlit dashboard
- No Claude AI explanations, no Claude daily trading plan

### With Upstox configured (free)
All of the above, plus:
- **Real live weekly + monthly option premiums** for both NIFTY and SENSEX instead of estimates

### With Anthropic API key (~$0.05–0.20/day)
All of the above, plus:
- Book-grounded signal explanation (references Al Brooks, Bulkowski, etc.) — generated per strategy
- Claude daily trading plan at 8 AM (HIGH priority alert, bypasses silent mode)

---

## Book-Grounded AI Knowledge Base

When a signal fires, Claude receives the 4 most relevant patterns from your library and explains the signal using book terminology, historical success rates, and specific risk notes.

| Book | Author | Patterns |
|------|--------|---------|
| Trading Price Action Trends | Al Brooks | With-Trend Bar, EMA Pullback, BPB, Bull/Bear Flag, Climax, Two-Legged Pullback |
| A Complete Guide to VPA | Anna Coulling | Volume Confirmation, No-Demand Bar, Selling Climax, Effort vs Result |
| Encyclopedia of Chart Patterns | Thomas Bulkowski | Bull Flag (54%), Double Bottom (64%), H&S Top (93%), Ascending Triangle (68%) |
| Art and Science of Technical Analysis | Adam Grimes | Pullback in Trend, Failed Breakout, Momentum Divergence |
| Martin Pring on Price Action | Martin Pring | S/R Reversal, Trendline Break + Momentum |
| Forex Price Action Scalping | Bob Volman | Tight Range Breakout, Round Number Barrier |

---

## Project Structure

```
market_analysyis/
├── main.py                              # Entry point + scheduler (NIFTY + SENSEX)
├── requirements.txt                     # includes playwright (NSE scrape fallback)
├── pytest.ini
├── .env                                 # Secrets (gitignored)
├── .env.example                         # Template
├── .gitignore
│
├── scripts/
│   └── upstox_login.py                  # Daily Upstox access-token refresh helper
│
├── nifty_ai_agent/
│   ├── config.py                        # Pydantic settings (incl. Upstox)
│   │
│   ├── data/
│   │   ├── base.py                      # MarketDataProvider + OptionChainData (weekly + monthly)
│   │   ├── upstox_provider.py           # Live option chain — NIFTY + SENSEX  [NEW]
│   │   ├── nse_provider.py              # NIFTY: Upstox → NSE scrape → VIX synthetic (4 tiers)
│   │   ├── bse_provider.py              # SENSEX: Upstox → VIX synthetic
│   │   ├── breadth.py                   # Live breadth — top 10 heavyweights
│   │   ├── sensex_breadth.py            # SENSEX breadth equivalent
│   │   ├── market_context.py            # Global indices + GIFT Nifty
│   │   ├── news_fetcher.py              # RSS news (4 feeds)
│   │   └── nifty50_stocks.py            # NIFTY 50 constituent movers
│   │
│   ├── indicators/
│   │   ├── rsi.py / ema.py / macd.py / atr.py / vwap.py
│   │
│   ├── strategies/
│   │   ├── base.py                      # BaseStrategy, Signal, SignalType
│   │   ├── ema_crossover.py             # Strategy 1 — EMA crossover + RSI
│   │   ├── vwap_breakout.py             # Strategy 2 — VWAP breakout + momentum  [NEW]
│   │   ├── rsi_analyser.py              # RSI zone + trend + divergence
│   │   └── option_analyser.py           # Weekly + monthly OC analysis, is_live flag
│   │
│   ├── risk/
│   │   └── calculator.py                # ATR-based SL/Target/RR
│   │
│   ├── ai/
│   │   ├── knowledge_base.py            # 23 patterns from 6 books
│   │   ├── explainer.py                 # Signal explanation via Claude (per strategy)
│   │   └── morning_analyser.py          # Daily plan via Claude
│   │
│   ├── database/
│   │   ├── models.py                    # SQLAlchemy ORM
│   │   └── repository.py                # CRUD operations
│   │
│   ├── notifier/
│   │   └── pushover.py                  # send_multi_signal() — all strategies, one message
│   │
│   └── reports/
│       └── morning_report.py            # 8 AM orchestrator
│
├── dashboard/
│   └── app.py                           # Streamlit — per-strategy prediction cards
│
└── tests/                               # 250+ tests, ~79% coverage
    ├── test_indicators.py
    ├── test_strategies.py
    ├── test_vwap_breakout.py            # [NEW]
    ├── test_risk.py
    ├── test_database.py
    ├── test_config.py
    ├── test_data_base.py
    ├── test_nse_provider.py             # incl. Upstox wiring + skip-expiring-today
    ├── test_bse_provider.py             # [NEW]
    ├── test_upstox_provider.py          # [NEW]
    ├── test_main_option_estimate.py     # [NEW] synthetic monthly-expiry date math
    ├── test_ai_explainer.py
    ├── test_knowledge_base.py
    ├── test_option_analyser.py
    ├── test_rsi_analyser.py
    ├── test_breadth.py
    ├── test_notifier.py                 # incl. weekly+monthly dual pricing, Est. labels
    ├── test_news_and_context.py
    └── test_morning_report.py
```

---

## Future Improvements

### Phase 2 — Enhanced Analysis
- **Option Greeks** (Delta, Theta, Gamma, Vega) derived from chain IV — show P/L sensitivity for each signal
- **OI change tracking** — compare current OI to previous fetch to detect fresh accumulation vs unwinding
- **Bank Nifty support** — separate weekly expiry chain for BANKNIFTY with adjusted strike step (100)
- **PCR trend** — 5-session PCR moving average to identify momentum in put/call writing

### Phase 3 — Multi-Strategy Engine
- ✅ **Two independent strategies** — EMA Crossover + VWAP Breakout, both run every cycle, saved and notified separately *(done)*
- **Strategy ranking** — score all strategies; only alert (or boost confidence) when ≥ 2 agree
- **Volume-Price Analysis strategy** — implement Coulling's effort-vs-result as a standalone signal module
- **Chart pattern detection** — programmatically detect Bulkowski patterns (flags, triangles, double bottoms)
- **Multi-timeframe confirmation** — require 15-min and 5-min agreement before firing

### Phase 4 — Learning
- **Signal outcome tracking** — log result (win/loss/partial) against each signal record
- **Win rate by configuration** — which RSI zone + PCR combination has the best historical accuracy
- **Backtesting module** — replay historical 5-minute data through the full 4-layer pipeline
- **PDF book ingestion** — ChromaDB + sentence-transformers for true vector RAG instead of hardcoded patterns

### Phase 5 — Infrastructure
- **PostgreSQL migration** — concurrent dashboard + agent DB access without SQLite locking
- **Docker + Raspberry Pi** — run 24/7 without a laptop
- **Streamlit Cloud / ngrok** — access dashboard from iPhone anywhere, not just home WiFi
- **Windows Task Scheduler** — auto-restart `main.py` on boot or crash
- **Paper trading tracker** — log your manual trades against signals; calculate real P/L and win rate

### Phase 6 — Broker Integration
- ✅ **Upstox API** — live option contract prices (weekly + monthly) for the specific CE/PE to trade, for both NIFTY and SENSEX *(done)*
- **Order preview** — generate the exact order parameters but require manual confirmation to place
- **Automated daily token refresh** — currently a manual `scripts/upstox_login.py` run each morning; could be scripted around market-open time

---

## Security Notes
- `.env` is in `.gitignore` — never committed (Upstox/Pushover/Anthropic keys all live here)
- `.venv/` and `.idea/` are gitignored too — keep local environments and IDE config out of version control
- Logs never record API key values — only key presence
- Pushover and Upstox credentials only read at runtime via `get_settings()`
- **Upstox access tokens expire nightly** (~3:30 AM IST) — a leaked token has a short shelf life, but treat `.env` as sensitive regardless
- ⚠️ This repository currently tracks `nifty_ai_agent.db` (SQLite) in git — if the repo is public, anyone can see your accumulated signal history. Add `*.db` back to `.gitignore` and `git rm --cached nifty_ai_agent.db` if you'd rather keep that private.

---

## No Live Trading
This agent generates **signals only**. It never places, modifies, or cancels any orders. All signals are for manual decision-making by the trader.
