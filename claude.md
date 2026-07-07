# CLAUDE.md

## Project Overview

This project is an AI-assisted trading signal agent focused on Indian markets.

Initial scope:

* NIFTY 50 only
* Weekly expiry option suggestions
* Signal generation only
* No automated order placement
* Telegram notifications
* AI-generated trade explanations
* SQLite for storage
* Python as primary language

The system should be modular, testable, and production-ready.

---

## Architecture

Market Data Layer
↓
Indicator Engine
↓
Strategy Engine
↓
Risk Management
↓
AI Analysis Layer
↓
Notification Layer

---

## Tech Stack

Language:

* Python 3.12+

Database:

* SQLite (MVP)
* PostgreSQL (future)

Libraries:

* pandas
* numpy
* requests
* ta
* sqlalchemy
* python-telegram-bot
* pydantic
* pytest

AI:

* Anthropic Claude API
* OpenAI API (optional)

---

## Directory Structure

nifty_ai_agent/

data/
indicators/
strategies/
risk/
ai/
database/
notifier/
backtesting/
tests/
logs/

main.py
config.py

---

## Coding Standards

### General

* Use type hints everywhere
* Use dataclasses or Pydantic models
* Avoid global variables
* Keep functions under 50 lines
* Follow SOLID principles
* Prefer composition over inheritance

### Logging

Use Python logging module.

Log:

* Data fetches
* Signals generated
* Errors
* Notifications sent

Never log API secrets.

### Error Handling

All external API calls must:

* Retry 3 times
* Log failures
* Return graceful errors

Never allow application crashes due to API failures.

---

## Data Layer

Responsibilities:

* Fetch NIFTY spot data
* Fetch option chain data
* Store OHLC history
* Cache requests

Required Interfaces:

MarketDataProvider

Methods:

* get_spot_data()
* get_option_chain()
* get_historical_data()

---

## Indicator Engine

Indicators:

* RSI
* EMA20
* EMA50
* MACD
* ATR
* VWAP

Indicators should be stateless.

Each indicator returns a DataFrame.

---

## Strategy Engine

Every strategy must implement:

generate_signal()

Output:

{
"signal": "BUY_CE",
"confidence": 80,
"reason": "EMA crossover"
}

Allowed signals:

* BUY_CE
* BUY_PE
* HOLD

Strategies must not place trades.

---

## Initial Strategy

EMA Crossover Strategy

BUY_CE Conditions:

* EMA20 > EMA50
* RSI > 60

BUY_PE Conditions:

* EMA20 < EMA50
* RSI < 40

Else:

HOLD

---

## Risk Management

Rules:

Maximum risk per trade:
1%

Daily loss limit:
3%

Risk module must calculate:

* Stop Loss
* Target
* Risk Reward Ratio

Minimum RR:

1:2

---

## AI Layer

Purpose:

Explain signals.

AI must never generate trading decisions.

Input:

* Signal
* Indicators
* Market conditions

Output:

Human-readable explanation.

Example:

"EMA20 crossed EMA50 while RSI remained above 60, indicating bullish momentum."

---

## Notification Layer

Primary:

Telegram

Future:

* WhatsApp
* Email

Notification Example:

NIFTY SIGNAL

Signal: BUY CE

Confidence: 82%

SL: 25180

Target: 25340

Reason:
EMA20 > EMA50
RSI = 64

---

## Database Schema

market_data

* datetime
* open
* high
* low
* close
* volume

signals

* datetime
* signal
* confidence
* strategy
* status

trades

* entry_price
* exit_price
* pnl
* strategy
* result

---

## Testing Requirements

Minimum coverage:

80%

Tests required for:

* Indicators
* Strategies
* Risk calculations
* Database operations

Use pytest.

---

## Future Roadmap

Phase 2

* Option chain analysis
* PCR
* Max Pain
* OI Build-up

Phase 3

* Multi-agent architecture
* Strategy ranking
* Learning from historical trades

Phase 4

* Paper trading
* Broker integrations

Phase 5

* Reinforcement learning experiments

---

## Constraints

* No live order execution in MVP
* No hardcoded secrets
* Store secrets in .env
* Modular architecture only
* All logic must be unit-testable

---

## Success Criteria

The system should:

1. Fetch NIFTY market data.
2. Calculate indicators.
3. Generate signals.
4. Explain signals using AI.
5. Send Telegram notifications.
6. Log all activity.
7. Run continuously every 5 minutes.
