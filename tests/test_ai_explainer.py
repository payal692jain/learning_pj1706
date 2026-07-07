"""Tests for the AI explainer (mocked Anthropic SDK)."""

from unittest.mock import MagicMock, patch

import pytest

from nifty_ai_agent.ai.explainer import Explanation, SignalExplainer
from nifty_ai_agent.risk.calculator import RiskCalculator
from nifty_ai_agent.strategies.base import Signal, SignalType


def _dummy_signal():
    return Signal(
        signal=SignalType.BUY_CE,
        confidence=80,
        reason="EMA crossover with bullish RSI",
        strategy="EMA_Crossover",
    )


def _dummy_risk():
    return RiskCalculator().calculate(SignalType.BUY_CE, 24000.0, 100.0)


def _dummy_indicators():
    return {"ema_20": 24050.0, "ema_50": 23950.0, "rsi": 65.0, "atr": 100.0}


def _mock_stream(text: str = "Bullish momentum confirmed."):
    """Build a mock that mimics the Anthropic streaming context manager."""
    content_block = MagicMock()
    content_block.text = text

    usage = MagicMock()
    usage.input_tokens = 150
    usage.output_tokens = 30

    final_message = MagicMock()
    final_message.content = [content_block]
    final_message.model = "claude-opus-4-8"
    final_message.usage = usage

    stream_ctx = MagicMock()
    stream_ctx.__enter__ = MagicMock(return_value=stream_ctx)
    stream_ctx.__exit__ = MagicMock(return_value=False)
    stream_ctx.get_final_message = MagicMock(return_value=final_message)
    return stream_ctx


class TestSignalExplainer:
    def setup_method(self):
        self.explainer = SignalExplainer(api_key="sk-test", model="claude-opus-4-8")

    def test_returns_explanation_object(self):
        stream = _mock_stream("Market shows bullish crossover.")
        with patch.object(self.explainer._client.messages, "stream", return_value=stream):
            result = self.explainer.explain(
                _dummy_signal(), _dummy_risk(), _dummy_indicators()
            )
        assert isinstance(result, Explanation)

    def test_explanation_text_populated(self):
        stream = _mock_stream("EMA20 crossed above EMA50.")
        with patch.object(self.explainer._client.messages, "stream", return_value=stream):
            result = self.explainer.explain(
                _dummy_signal(), _dummy_risk(), _dummy_indicators()
            )
        assert result.text == "EMA20 crossed above EMA50."

    def test_token_counts(self):
        stream = _mock_stream()
        with patch.object(self.explainer._client.messages, "stream", return_value=stream):
            result = self.explainer.explain(
                _dummy_signal(), _dummy_risk(), _dummy_indicators()
            )
        assert result.input_tokens == 150
        assert result.output_tokens == 30

    def test_api_failure_falls_back_to_reason(self):
        import anthropic
        with patch.object(
            self.explainer._client.messages,
            "stream",
            side_effect=anthropic.APIError("fail", request=MagicMock(), body={}),
        ):
            with patch("time.sleep"):
                result = self.explainer.explain(
                    _dummy_signal(), _dummy_risk(), _dummy_indicators()
                )
        assert result.text == _dummy_signal().reason
        assert result.input_tokens == 0

    def test_prompt_contains_signal(self):
        captured: list[str] = []

        def capture_stream(**kwargs):
            captured.append(kwargs.get("messages", [{}])[-1].get("content", ""))
            return _mock_stream()

        with patch.object(self.explainer._client.messages, "stream", side_effect=capture_stream):
            self.explainer.explain(_dummy_signal(), _dummy_risk(), _dummy_indicators())

        assert "BUY_CE" in captured[0]
