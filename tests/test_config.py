"""Tests for config.py."""

import os
from unittest.mock import patch

import pytest

from nifty_ai_agent.config import Settings


class TestSettings:
    def _make_env(self, **overrides):
        base = {
            "ANTHROPIC_API_KEY": "sk-test",
            "PUSHOVER_USER_KEY": "user_key_123",
            "PUSHOVER_API_TOKEN": "app_token_456",
        }
        base.update(overrides)
        return base

    def test_loads_required_fields(self):
        with patch.dict(os.environ, self._make_env(), clear=False):
            s = Settings()
            assert s.anthropic_api_key == "sk-test"
            assert s.pushover_user_key == "user_key_123"
            assert s.pushover_api_token == "app_token_456"

    def test_default_model(self):
        with patch.dict(os.environ, self._make_env(), clear=False):
            s = Settings()
            assert s.claude_model == "claude-opus-4-8"

    def test_default_interval(self):
        with patch.dict(os.environ, self._make_env(), clear=False):
            s = Settings()
            assert s.data_fetch_interval_minutes == 5

    def test_default_risk_pct(self):
        with patch.dict(os.environ, self._make_env(), clear=False):
            s = Settings()
            assert s.max_risk_per_trade_pct == 1.0
            assert s.daily_loss_limit_pct == 3.0
            assert s.min_risk_reward_ratio == 2.0

    def test_override_via_env(self):
        env = self._make_env(HISTORICAL_DAYS="90", LOG_LEVEL="DEBUG")
        with patch.dict(os.environ, env, clear=False):
            s = Settings()
            assert s.historical_days == 90
            assert s.log_level == "DEBUG"

    def test_missing_required_field_raises(self):
        # Clear env AND bypass .env file so required fields are truly absent
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(Exception):
                Settings(_env_file=None)
