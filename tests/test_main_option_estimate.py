"""Tests for the synthetic monthly-expiry date estimate in main.py.

Used only when live NSE/BSE option chain data is unavailable and the pipeline
falls back to a VIX-based synthetic chain, which has no real expiry calendar.
"""

from datetime import date
from unittest.mock import patch

import main


class TestLastWeekdayOfMonth:
    def test_last_thursday_of_july_2026(self):
        # July 2026: 30th is a Thursday, 31st a Friday.
        result = main._last_weekday_of_month(2026, 7, 3)  # 3 = Thursday
        assert result == date(2026, 7, 30)

    def test_last_friday_of_july_2026(self):
        result = main._last_weekday_of_month(2026, 7, 4)  # 4 = Friday
        assert result == date(2026, 7, 31)

    def test_december_rolls_into_january(self):
        # Last Thursday of December 2026 — must not crash on year rollover.
        result = main._last_weekday_of_month(2026, 12, 3)
        assert result.year == 2026
        assert result.month == 12
        assert result.weekday() == 3


class TestEstimatedMonthlyExpiry:
    def test_returns_last_thursday_when_still_upcoming(self):
        with patch("main.date") as mock_date:
            mock_date.today.return_value = date(2026, 7, 1)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            result = main._estimated_monthly_expiry(3)
        assert result == "30-Jul-2026"

    def test_rolls_to_next_month_when_already_passed(self):
        # "Today" is after this month's last Thursday (30-Jul-2026).
        with patch("main.date") as mock_date:
            mock_date.today.return_value = date(2026, 7, 31)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            result = main._estimated_monthly_expiry(3)
        assert result == "27-Aug-2026"

    def test_rolls_from_december_into_next_january(self):
        with patch("main.date") as mock_date:
            mock_date.today.return_value = date(2026, 12, 31)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            result = main._estimated_monthly_expiry(3)
        assert result == "28-Jan-2027"

    def test_result_is_parseable_and_matches_weekday(self):
        from datetime import datetime as dt
        result = main._estimated_monthly_expiry(4)  # Friday
        parsed = dt.strptime(result, "%d-%b-%Y").date()
        assert parsed.weekday() == 4
        assert parsed >= date.today()
