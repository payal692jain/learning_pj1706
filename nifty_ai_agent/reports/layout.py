"""Shared formatting for Pushover notification bodies.

Pushover REJECTS a message over 1024 characters — it does not truncate it. An
over-long body therefore means no notification arrives at all, which is the worst
possible failure mode for an alert whose entire job is to arrive. Every report
composes through fit_sections() so that constraint is enforced in one place.
"""

import logging

logger = logging.getLogger(__name__)

PUSHOVER_LIMIT = 1024


def fit_sections(essential: list[str], optional: list[str], footer: list[str]) -> str:
    """Join the body, dropping *optional* sections from the end until it fits.

    *essential* and *footer* always survive: they carry the verdict and the risk
    disclaimer, and a notification that dropped either would be actively misleading.
    Sections in *optional* are separated by blank lines and shed tail-first.
    """
    while True:
        body = "\n".join(essential + optional + footer)
        if len(body) <= PUSHOVER_LIMIT or not optional:
            return body

        last_break = max(
            (i for i, line in enumerate(optional[:-1]) if line == ""), default=-1
        )
        if last_break < 0:
            optional = []
        else:
            optional = optional[: last_break + 1]
        logger.warning("Body over %d chars — dropped a detail section.", PUSHOVER_LIMIT)


def row(label: str, cells: list[str], label_width: int = 11, cell_width: int = 9) -> str:
    """One monospace table row.

    *label_width* must clear the longest label in the table — a label that overflows
    shifts that row's columns out of alignment with every other row.
    """
    return f"{label:<{label_width}}" + "".join(f"{c:>{cell_width}}" for c in cells)


def inr(value: float) -> str:
    """Indian-notation money for a narrow cell: 6760 → '6.8k', 340080 → '3.40L'."""
    if abs(value) >= 100_000:
        return f"{value / 100_000:.2f}L"
    if abs(value) >= 1_000:
        return f"{value / 1000:.1f}k"
    return f"{value:,.0f}"


def premium(value: float) -> str:
    """Option premium: two decimals below ₹10 so near-expiry paise premiums do not
    round to a misleading '0'."""
    if value and value < 10:
        return f"{value:.2f}"
    return f"{value:,.0f}"
