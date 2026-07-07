# NIFTY AI Signal Agent

An AI-assisted trading signal agent for Indian markets — NIFTY 50 weekly **and monthly** expiry options.
Generates BUY_CE / BUY_PE / HOLD signals every 5 minutes during market hours, with multi-layer confidence scoring (RSI analysis + option chain OI levels + heavyweight breadth), sends a full pre-market morning brief at 8 AM, and delivers everything to your iPhone via Pushover notifications and a Streamlit dashboard.

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

Every 5 minutes during market hours (9:15 AM – 3:30 PM):
  → Fetches live NIFTY spot price
  → Fetches 5-minute intraday OHLCV bars (10 days ≈ 700 candles)
  → Calculates RSI, EMA20, EMA50, MACD, ATR, VWAP

  SIGNAL GENERATION:
  → EMA Crossover strategy → raw BUY_CE / BUY_PE / HOLD signal

  MULTI-LAYER CONFIDENCE SCORING:
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

  → Final signal saved to SQLite, sent via Pushover, shown on dashboard
  → Claude AI explains the signal using concepts from your trading book library
```

---

## Tech Stack

| Layer | Library / Service | Purpose |
|-------|------------------|---------|
| **Language** | Python 3.10+ | Primary language |
| **Data Models** | Pydantic v2, Dataclasses | Type-safe structured data |
| **Configuration** | pydantic-settings, python-dotenv | `.env` file management |
| **Market Data** | yfinance | NIFTY OHLCV, global indices, NIFTY 50 stocks |
| **Option Chain** | NSE India API | Weekly + monthly chain, PCR, max pain |
| **News** | feedparser | RSS from ET Markets, Moneycontrol, Reuters, Livemint |
| **Indicators** | pandas, numpy | RSI, EMA, MACD, ATR, VWAP (stateless) |
| **Strategy** | Custom | EMA20/50 Crossover + RSI confirmation |
| **Confidence Scoring** | 4-layer system | RSI analysis + weekly OC + monthly OC + breadth |
| **Risk** | ATR-based | SL/Target/RR calculator |
| **AI** | anthropic SDK | Claude `claude-opus-4-8` — signal explanation + daily plan |
| **Knowledge Base** | Hardcoded patterns | 23 patterns from 6 trading books |
| **Database** | SQLAlchemy 2.0, SQLite | Signals, market data, trades |
| **Notifications** | Pushover HTTP API | iPhone push alerts (priority-based) |
| **Dashboard** | Streamlit, Plotly | Live candlestick + indicator chart |
| **Scheduler** | schedule | 8 AM daily + 5-min intraday loop |
| **Timezone** | pytz | IST market hours enforcement |
| **Testing** | pytest, pytest-cov | 82%+ coverage, 193 tests |
| **Version Control** | Git | `.env` and DB excluded via `.gitignore` |

---

## Architecture & Flow Diagram

```
╔══════════════════════════════════════════════════════════════════════╗
║                     NIFTY AI SIGNAL AGENT                           ║
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
║  (yfinance, 8 markets)  ║    ║  │ NSEDataProvider              │   ║
║         +               ║    ║  │ · Spot: yfinance fast_info   │   ║
║  GIFT Nifty             ║    ║  │ · OHLCV: 5m bars, 10 days   │   ║
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
║  · PCR, Max Pain        ║    ║  │   EMA Crossover Strategy     │   ║
║  · OI resistance/support║    ║  │  BUY_CE: EMA20>EMA50 RSI>60  │   ║
║  · Black-Scholes CE/PE  ║    ║  │  BUY_PE: EMA20<EMA50 RSI<40  │   ║
║         +               ║    ║  │  HOLD:   all other           │   ║
║  MONTHLY Option Chain   ║    ║  │  Confidence: 50–95%          │   ║
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

---

## Component Details

### 1. Configuration (`nifty_ai_agent/config.py`)
Pydantic Settings loads all config from `.env`. Single `get_settings()` singleton cached with `@lru_cache`. `ANTHROPIC_API_KEY` is optional — AI features degrade gracefully if unset.

---

### 2. Data Layer (`nifty_ai_agent/data/`)

#### `nse_provider.py` — NSEDataProvider
- **`get_spot_data()`** — yfinance `fast_info.last_price` for live price
- **`get_historical_data(days, interval)`** — yfinance 5-minute bars (10 days ≈ 700 candles). Auto-flattens yfinance MultiIndex columns.
- **`get_option_chain()`** — NSE India API with browser-mimicking headers. **Returns both weekly and monthly expiry data in a single API call:**
  - Filters raw entries by `expiryDate` field per record
  - `_identify_expiries()` scans the expiry dates list, groups by calendar month, and returns:
    - **weekly** = nearest expiry
    - **monthly** = last expiry of the same calendar month (if different from weekly), or last expiry of the following month if weekly IS the monthly (last week of the month)
  - Computes PCR and max pain separately for each expiry

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

#### `ema_crossover.py` — Primary Signal
```
BUY_CE  →  EMA20 > EMA50  AND  RSI > 60
BUY_PE  →  EMA20 < EMA50  AND  RSI < 40
HOLD    →  all other
```
Base confidence 50–95% from EMA separation % and RSI distance from threshold.

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

**Black-Scholes pricing** — ATM CE/PE theoretical prices using actual chain IV. Implemented with `math.erf` (no scipy dependency).

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

| Message | Priority | Behaviour |
|---------|----------|-----------|
| BUY_CE / BUY_PE | 0 (Normal) | Sound + lock screen |
| HOLD heartbeat | −1 (Low) | Silent |
| Claude daily plan | +1 (High) | Bypasses Do Not Disturb; requires acknowledgement |
| Monthly OC | 0 (Normal) | Sound + lock screen |
| Startup ping | 0 (Normal) | Agent alive confirmation |

3 retries with 2-second backoff. Returns `False` on total failure — never crashes.

---

### 9. Dashboard (`dashboard/app.py`)
Streamlit + Plotly, reads live from SQLite.

- 3-panel Plotly chart: Candlestick + EMA20/50, RSI (with 60/40 lines), MACD histogram
- Signal box (green/red/yellow) with confidence %
- Entry / SL / Target / RR metric cards
- Claude AI explanation card
- Indicator snapshot (EMA, RSI, MACD, breadth score, PCR, max pain)
- Signal history table (last 50)
- Auto-refresh every 60 seconds

---

## Signal Confidence Scoring

Every BUY signal's confidence is adjusted by 4 independent layers before it fires:

```
Base confidence (EMA crossover)         50 – 95%
  + Layer 1: RSI analysis               −12 to +11
  + Layer 2: Weekly option chain        −22 to  +9
  + Layer 3: Monthly option chain       −14 to  +3
  + Layer 4: Breadth (10 heavyweights)  −12 to  +8
  ─────────────────────────────────────────────────
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

### Step 1 — Install
```bash
cd c:\my_learnings\market_analysyis
pip install -r requirements.txt
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

# Market data defaults (fine as-is)
NIFTY_SYMBOL=^NSEI
DATA_FETCH_INTERVAL_MINUTES=5
DATA_INTERVAL=5m
HISTORICAL_DAYS=10
```

### Step 3 — Run
```bash
# Terminal 1 — signal agent + morning reports
python main.py

# Terminal 2 — dashboard
streamlit run dashboard/app.py
# Opens at http://localhost:8501
```

### Step 4 — Add to iPhone home screen
Safari → `http://<your-laptop-ip>:8501` → Share → Add to Home Screen

### Step 5 — Auto-start on Windows (optional)
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
08:00 AM  →  Pushover messages arrive:
              ① Global bias + GIFT Nifty level
              ② NIFTY 50 advance/decline + top movers
              ③ Weekly option chain: CE/PE table, max pain, PCR, OI levels
              ④ Monthly option chain: structural levels, monthly max pain
              ⑤ News headlines (silent)
              ⑥ Claude daily plan (HIGH priority, if API key set)

09:15 AM  →  Market opens; 5-min signal loop begins
              BUY_CE or BUY_PE → sound alert with full confidence breakdown
              HOLD → silent (agent heartbeat)

03:30 PM  →  Market closes; signal loop stops automatically
```

### Reading a signal notification
```
📈 NIFTY — BUY_CE · 84% confidence
Entry: 24,120  SL: 23,984  Target: 24,390  RR: 1:2.0

EMA20 (24,120) > EMA50 (24,050) RSI 67.1 — bullish momentum.
RSI (Overbought): RSI 67.1 overbought zone, momentum confirmed; RSI trending up.
OC (27-Jun, PCR 1.26): PCR 1.26 bullish; open air to CE wall at 24,400 (1.8% away).
[Monthly 31-Jul: Monthly PCR 0.71 — bearish backdrop].
Breadth confirms: 7/10 heavyweights advancing (HDFCBANK, INFY, TCS).

Analysis: EMA20 crossed above EMA50 on an expanding bullish bar — Al Brooks
calls this an EMA Pullback Entry. PCR above 1.2 indicates aggressive put
writing at lower strikes, confirming institutional bullish bias. Monthly CE
wall at 24,650 leaves ~2.2% upside before hitting structural resistance.
Key risk: if RSI turns below 60 on the next bar, momentum is fading — exit.
```

### Run tests
```bash
python -m pytest tests/ -v
# 193 tests, 82%+ coverage
```

---

## What You Receive

### Without API key (free)
- Morning report: global markets, NIFTY 50 A/D, weekly OC, monthly OC, news
- 5-min intraday signals with full 4-layer confidence scoring
- Risk parameters (SL, target, RR)
- Streamlit dashboard
- No Claude AI explanations
- No Claude daily trading plan

### With API key (~$0.05–0.20/day)
All of the above, plus:
- Book-grounded signal explanation (references Al Brooks, Bulkowski, etc.)
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
├── main.py                              # Entry point + scheduler
├── requirements.txt
├── pytest.ini
├── .env                                 # Secrets (gitignored)
├── .env.example                         # Template
├── .gitignore
│
├── nifty_ai_agent/
│   ├── config.py                        # Pydantic settings
│   │
│   ├── data/
│   │   ├── base.py                      # MarketDataProvider + OptionChainData (weekly + monthly)
│   │   ├── nse_provider.py              # NSE API + yfinance  [weekly + monthly expiry]
│   │   ├── breadth.py                   # Live breadth — top 10 heavyweights  [NEW]
│   │   ├── market_context.py            # Global indices + GIFT Nifty
│   │   ├── news_fetcher.py              # RSS news (4 feeds)
│   │   └── nifty50_stocks.py            # NIFTY 50 constituent movers
│   │
│   ├── indicators/
│   │   ├── rsi.py / ema.py / macd.py / atr.py / vwap.py
│   │
│   ├── strategies/
│   │   ├── base.py                      # BaseStrategy, Signal, SignalType
│   │   ├── ema_crossover.py             # Primary signal strategy
│   │   ├── rsi_analyser.py              # RSI zone + trend + divergence  [NEW]
│   │   └── option_analyser.py           # Weekly + monthly OC analysis  [EXTENDED]
│   │
│   ├── risk/
│   │   └── calculator.py                # ATR-based SL/Target/RR
│   │
│   ├── ai/
│   │   ├── knowledge_base.py            # 23 patterns from 6 books
│   │   ├── explainer.py                 # Signal explanation via Claude
│   │   └── morning_analyser.py          # Daily plan via Claude  [monthly OC context]
│   │
│   ├── database/
│   │   ├── models.py                    # SQLAlchemy ORM
│   │   └── repository.py                # CRUD operations
│   │
│   ├── notifier/
│   │   └── pushover.py                  # Pushover HTTP API
│   │
│   └── reports/
│       └── morning_report.py            # 8 AM orchestrator  [sends monthly OC message]
│
├── dashboard/
│   └── app.py                           # Streamlit dashboard
│
└── tests/                               # 193 tests, 82%+ coverage
    ├── test_indicators.py
    ├── test_strategies.py
    ├── test_risk.py
    ├── test_database.py
    ├── test_config.py
    ├── test_data_base.py
    ├── test_nse_provider.py             # includes _identify_expiries + _compute_pcr tests
    ├── test_ai_explainer.py
    ├── test_knowledge_base.py
    ├── test_option_analyser.py          # includes weekly + monthly OC confidence tests
    ├── test_rsi_analyser.py             # zone, trend, divergence, adjustment  [NEW]
    ├── test_breadth.py                  # heavyweight advance/decline  [NEW]
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
- **Strategy ranking** — score EMA crossover, VPA, and pattern strategies; only alert when ≥ 2 agree
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
- **Zerodha Kite API** — display live option contract prices for the specific CE/PE to trade
- **Order preview** — generate the exact order parameters but require manual confirmation to place

---

## Security Notes
- `.env` is in `.gitignore` — never committed
- Logs never record API key values — only key presence
- SQLite file excluded from git
- Pushover credentials only read at runtime via `get_settings()`

---

## No Live Trading
This agent generates **signals only**. It never places, modifies, or cancels any orders. All signals are for manual decision-making by the trader.
