"""BANKNIFTY constituent options — CE/PE suggestions on the individual banks.

BANKNIFTY is not a market, it is twelve banks in a trenchcoat, and on most days the
index move is carried by two or three of them while the rest drag. Trading the index
option means paying for the average; trading the leader's own option means expressing
the same view on the stock actually doing the work.

So when the index signal is bullish this suggests calls on the banks LEADING the move,
and when it is bearish it suggests puts on the banks LAGGING it. A call on a bank that
is falling while the index rises is not a bank-nifty trade, it is a bet against the
one constituent that disagrees — the opposite of confirmation.

Everything is skipped (never guessed) when the Upstox token is dead: a suggested strike
with a made-up premium is worse than no suggestion.
"""

import logging
from dataclasses import dataclass

import yfinance as yf

from nifty_ai_agent.data.instrument_master import InstrumentMaster, get_instrument_master
from nifty_ai_agent.strategies.base import SignalType

logger = logging.getLogger(__name__)

# The BANKNIFTY constituents liquid enough to have a tradable option chain. Ordered
# by index weight; the smaller PSU banks are excluded on purpose — their option
# spreads are wide enough to eat the edge before the trade starts.
TRADABLE_BANKS: list[str] = [
    "HDFCBANK",
    "ICICIBANK",
    "KOTAKBANK",
    "AXISBANK",
    "SBIN",
    "INDUSINDBK",
]

# A bank must be moving at least this much, in the signal's direction, to qualify.
# Below it the stock is drifting, not leading, and its option is just a slower
# version of the index one.
_MIN_MOVE_PCT = 0.3


@dataclass
class BankMove:
    symbol: str
    price: float
    change_pct: float


@dataclass
class BankOptionIdea:
    symbol: str
    spot: float
    change_pct: float       # the move that qualified this bank
    strike: float
    opt_type: str           # CE / PE
    expiry: str             # 'DD-Mon'
    premium: float
    lot_size: int
    trading_symbol: str

    @property
    def cost_per_lot(self) -> float:
        return self.premium * self.lot_size


def fetch_bank_moves(symbols: list[str] | None = None) -> list[BankMove]:
    """Today's % move for each tradable bank, strongest mover first."""
    moves: list[BankMove] = []
    for symbol in symbols or TRADABLE_BANKS:
        try:
            info = yf.Ticker(f"{symbol}.NS").fast_info
            price = float(info.last_price)
            prev = float(info.previous_close)
            if prev <= 0:
                continue
            moves.append(
                BankMove(
                    symbol=symbol,
                    price=round(price, 2),
                    change_pct=round((price - prev) / prev * 100, 2),
                )
            )
        except Exception as exc:
            logger.debug("Bank move: skipping %s — %s", symbol, exc)

    moves.sort(key=lambda m: abs(m.change_pct), reverse=True)
    return moves


def suggest_bank_options(
    signal: SignalType,
    upstox_client,
    top_n: int = 3,
    master: InstrumentMaster | None = None,
    moves: list[BankMove] | None = None,
) -> list[BankOptionIdea]:
    """Suggest CE/PE on the constituent banks confirming a BANKNIFTY *signal*.

    Args:
        signal: The BANKNIFTY index consensus — BUY_CE, BUY_PE, or HOLD.
        upstox_client: An UpstoxClient; needed for live option premiums.
        top_n: How many banks to suggest at most.
        master / moves: Injectable for tests.

    Returns:
        Ideas for the banks moving WITH the signal, strongest first. Empty on HOLD,
        when no bank confirms, or when live prices are unavailable.
    """
    if signal == SignalType.HOLD:
        return []

    bullish = signal == SignalType.BUY_CE
    opt_type = "CE" if bullish else "PE"
    master = master or get_instrument_master()
    moves = moves if moves is not None else fetch_bank_moves()

    # Only banks moving in the signal's direction, by a margin that means something.
    if bullish:
        confirming = [m for m in moves if m.change_pct >= _MIN_MOVE_PCT]
    else:
        confirming = [m for m in moves if m.change_pct <= -_MIN_MOVE_PCT]

    if not confirming:
        logger.info(
            "No BANKNIFTY constituent confirms %s by >=%.1f%% — no bank options suggested.",
            signal.value, _MIN_MOVE_PCT,
        )
        return []

    selected = confirming[:top_n]
    contracts = {}
    for move in selected:
        contract = master.atm_contract(move.symbol, move.price, opt_type)
        if contract:
            contracts[move.symbol] = contract
        else:
            logger.warning("No %s contract found for %s", opt_type, move.symbol)

    if not contracts:
        return []

    try:
        prices = upstox_client.get_ltp([c.instrument_key for c in contracts.values()])
    except Exception as exc:
        # A strike with an invented premium is worse than silence — bail out entirely.
        logger.warning("Bank option premiums unavailable (%s) — skipping suggestions.", exc)
        return []

    ideas: list[BankOptionIdea] = []
    for move in selected:
        contract = contracts.get(move.symbol)
        if not contract:
            continue
        premium = prices.get(contract.instrument_key, 0.0)
        if premium <= 0:
            logger.debug("No live premium for %s — skipping.", contract.trading_symbol)
            continue
        ideas.append(
            BankOptionIdea(
                symbol=move.symbol,
                spot=move.price,
                change_pct=move.change_pct,
                strike=contract.strike,
                opt_type=opt_type,
                expiry=contract.expiry.strftime("%d-%b"),
                premium=premium,
                lot_size=contract.lot_size,
                trading_symbol=contract.trading_symbol,
            )
        )

    logger.info(
        "Bank options for %s: %s",
        signal.value, ", ".join(f"{i.symbol} {i.strike:.0f}{i.opt_type}" for i in ideas) or "none",
    )
    return ideas


def format_bank_options(ideas: list[BankOptionIdea], capital: float) -> list[str]:
    """Render bank ideas as monospace notification lines."""
    if not ideas:
        return []

    lines = ["── BANKS CONFIRMING ──"]
    for idea in ideas:
        affordable = int(capital // idea.cost_per_lot) if idea.cost_per_lot > 0 else 0
        lines.append(
            f"{idea.symbol[:10]:<10} {idea.change_pct:+.1f}%  "
            f"{idea.strike:.0f}{idea.opt_type} @{idea.premium:,.1f}"
        )
        lines.append(
            f"  lot {idea.lot_size} = ₹{idea.cost_per_lot:,.0f}  "
            f"({affordable} lot(s) affordable)"
        )
    return lines
