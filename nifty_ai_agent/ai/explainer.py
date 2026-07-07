"""AI explainer — uses Anthropic Claude to narrate trading signals in plain English,
grounded in patterns from the user's trading book library.

The AI layer NEVER generates trading decisions.
It only explains a signal that the strategy engine already produced.
"""

import logging
import time
from dataclasses import dataclass

import anthropic

from nifty_ai_agent.ai.knowledge_base import (
    format_patterns_for_prompt,
    get_relevant_patterns,
)
from nifty_ai_agent.risk.calculator import RiskParameters
from nifty_ai_agent.strategies.base import Signal

logger = logging.getLogger(__name__)

_RETRY_COUNT = 3
_RETRY_DELAY = 2  # seconds

_SYSTEM_PROMPT = """\
You are a concise trading analyst for Indian equity markets (NIFTY 50 / BSE SENSEX).
Your role is to explain a generated trading signal using concepts from the \
trader's personal book library.

Rules:
- Never generate new trading decisions or signals — only explain the one provided.
- Never recommend specific option strikes or quantities.
- Always cite the book source when referencing a pattern (e.g., "Al Brooks calls this...").
- Keep the explanation under 150 words.
- End with one sentence on the key risk to watch (from the book's risk note).
- Use plain language — avoid academic jargon.
"""


@dataclass
class Explanation:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


class SignalExplainer:
    """Wraps the Anthropic SDK to explain a generated signal using book-grounded context."""

    def __init__(self, api_key: str, model: str = "claude-opus-4-8") -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def explain(
        self,
        signal: Signal,
        risk: RiskParameters,
        indicators: dict[str, float],
        index_name: str = "NIFTY 50",
    ) -> Explanation:
        """Generate a plain-English, book-grounded explanation of *signal*.

        Args:
            signal: The signal produced by the strategy engine.
            risk: The corresponding risk parameters.
            indicators: Dict of indicator name → current value.

        Returns:
            Explanation with the narrative text and token usage.
        """
        # Retrieve matching book patterns for this market state
        patterns = get_relevant_patterns(
            signal=signal.signal,
            rsi=indicators.get("rsi", 50.0),
            ema20=indicators.get("ema_20", 0.0),
            ema50=indicators.get("ema_50", 0.0),
        )
        pattern_context = format_patterns_for_prompt(patterns)
        prompt = self._build_prompt(signal, risk, indicators, pattern_context, index_name)

        for attempt in range(1, _RETRY_COUNT + 1):
            try:
                return self._call_api(prompt)
            except anthropic.APIError as exc:
                logger.warning(
                    "Claude API attempt %d/%d failed: %s", attempt, _RETRY_COUNT, exc
                )
                if attempt < _RETRY_COUNT:
                    time.sleep(_RETRY_DELAY)

        logger.error("All Claude API attempts failed. Returning fallback explanation.")
        return Explanation(
            text=signal.reason,
            model=self._model,
            input_tokens=0,
            output_tokens=0,
        )

    def _call_api(self, prompt: str) -> Explanation:
        with self._client.messages.stream(
            model=self._model,
            max_tokens=400,
            thinking={"type": "adaptive"},
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            message = stream.get_final_message()

        text = "".join(
            block.text
            for block in message.content
            if hasattr(block, "text")
        )
        return Explanation(
            text=text.strip(),
            model=message.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )

    @staticmethod
    def _build_prompt(
        signal: Signal,
        risk: RiskParameters,
        indicators: dict[str, float],
        pattern_context: str,
        index_name: str = "NIFTY 50",
    ) -> str:
        ind_lines = "\n".join(f"  {k}: {v:.2f}" for k, v in indicators.items())
        return f"""\
A trading strategy has generated the following signal for {index_name}.
Explain in plain English why the current market conditions support this signal, \
referencing the relevant patterns from the book library below.

Signal: {signal.signal.value}
Confidence: {signal.confidence}%
Strategy reason: {signal.reason}

Current indicator values:
{ind_lines}

Risk parameters:
  Entry price: {risk.entry_price}
  Stop loss:   {risk.stop_loss}
  Target:      {risk.target}
  Risk/Reward: {risk.risk_reward_ratio}

{pattern_context}

Write a 3–5 sentence explanation for a retail trader, citing the most relevant \
book patterns above. End with the key risk to watch."""
