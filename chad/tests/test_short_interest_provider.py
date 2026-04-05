"""Tests for ShortInterestProvider."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chad.intel.short_interest_provider import (
    HIGH_THRESHOLD,
    LOW_THRESHOLD,
    MODERATE_THRESHOLD,
    ShortInterestProvider,
    ShortInterestSignal,
    _classify_short_float,
    _parse_short_float,
)


# ---------------------------------------------------------------------------
# Signal classification
# ---------------------------------------------------------------------------


class TestClassifyShortFloat:
    def test_low(self):
        assert _classify_short_float(0.02) == "LOW"
        assert _classify_short_float(0.049) == "LOW"

    def test_moderate(self):
        assert _classify_short_float(0.05) == "MODERATE"
        assert _classify_short_float(0.10) == "MODERATE"
        assert _classify_short_float(0.149) == "MODERATE"

    def test_high(self):
        assert _classify_short_float(0.15) == "HIGH"
        assert _classify_short_float(0.20) == "HIGH"
        assert _classify_short_float(0.299) == "HIGH"

    def test_extreme(self):
        assert _classify_short_float(0.30) == "EXTREME"
        assert _classify_short_float(0.50) == "EXTREME"
        assert _classify_short_float(1.0) == "EXTREME"

    def test_zero(self):
        assert _classify_short_float(0.0) == "LOW"

    def test_boundary_low_moderate(self):
        assert _classify_short_float(LOW_THRESHOLD) == "MODERATE"

    def test_boundary_moderate_high(self):
        assert _classify_short_float(MODERATE_THRESHOLD) == "HIGH"

    def test_boundary_high_extreme(self):
        assert _classify_short_float(HIGH_THRESHOLD) == "EXTREME"


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------


class TestParseShortFloat:
    def test_typical_finviz_format(self):
        html = '<td>Short Float</td><td class="snapshot-td2"><b>3.21%</b></td>'
        result = _parse_short_float(html)
        assert result is not None
        assert abs(result - 0.0321) < 0.0001

    def test_high_short_float(self):
        html = 'Short Float</td><td><b>35.50%</b></td>'
        result = _parse_short_float(html)
        assert result is not None
        assert abs(result - 0.355) < 0.001

    def test_integer_percentage(self):
        html = 'Short Float</td><td><b>5%</b></td>'
        result = _parse_short_float(html)
        assert result is not None
        assert abs(result - 0.05) < 0.001

    def test_no_short_float(self):
        html = '<html><body>No short interest data here</body></html>'
        result = _parse_short_float(html)
        assert result is None

    def test_empty_html(self):
        result = _parse_short_float("")
        assert result is None

    def test_multiline_html(self):
        html = """
        <td>Short Float</td>
        <td class="snapshot-td2">
            <b>12.34%</b>
        </td>
        """
        result = _parse_short_float(html)
        assert result is not None
        assert abs(result - 0.1234) < 0.0001


# ---------------------------------------------------------------------------
# Squeeze risk detection
# ---------------------------------------------------------------------------


class TestSqueezeRisk:
    def test_high_short_uptrend_is_squeeze(self, tmp_path: Path):
        provider = ShortInterestProvider(runtime_dir=tmp_path)
        # Mock the HTML response and price uptrend
        html = 'Short Float</td><td><b>25.0%</b></td>'
        mock_resp = MagicMock()
        mock_resp.read.return_value = html.encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("chad.intel.short_interest_provider.urllib.request.urlopen",
                   return_value=mock_resp):
            with patch("chad.intel.short_interest_provider._check_price_uptrend",
                       return_value=True):
                with patch("chad.intel.short_interest_provider.time.sleep"):
                    signals = provider._fetch_short_interest(["GME"])

        if "GME" in signals:
            assert signals["GME"].squeeze_risk is True
            assert signals["GME"].signal == "HIGH"

    def test_high_short_downtrend_no_squeeze(self, tmp_path: Path):
        provider = ShortInterestProvider(runtime_dir=tmp_path)
        html = 'Short Float</td><td><b>25.0%</b></td>'
        mock_resp = MagicMock()
        mock_resp.read.return_value = html.encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("chad.intel.short_interest_provider.urllib.request.urlopen",
                   return_value=mock_resp):
            with patch("chad.intel.short_interest_provider._check_price_uptrend",
                       return_value=False):
                with patch("chad.intel.short_interest_provider.time.sleep"):
                    signals = provider._fetch_short_interest(["GME"])

        if "GME" in signals:
            assert signals["GME"].squeeze_risk is False

    def test_low_short_no_squeeze(self, tmp_path: Path):
        provider = ShortInterestProvider(runtime_dir=tmp_path)
        html = 'Short Float</td><td><b>2.0%</b></td>'
        mock_resp = MagicMock()
        mock_resp.read.return_value = html.encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("chad.intel.short_interest_provider.urllib.request.urlopen",
                   return_value=mock_resp):
            with patch("chad.intel.short_interest_provider._check_price_uptrend",
                       return_value=True):
                with patch("chad.intel.short_interest_provider.time.sleep"):
                    signals = provider._fetch_short_interest(["AAPL"])

        if "AAPL" in signals:
            assert signals["AAPL"].squeeze_risk is False
            assert signals["AAPL"].signal == "LOW"


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


class TestCacheBehavior:
    def test_cache_fresh(self, tmp_path: Path):
        provider = ShortInterestProvider(runtime_dir=tmp_path)
        sig = ShortInterestSignal(
            symbol="SPY", short_float_pct=0.032, signal="LOW",
            squeeze_risk=False, ts_utc="2026-04-05T10:00:00Z",
        )
        from dataclasses import asdict
        provider._cache = {"SPY": asdict(sig)}
        provider._cache_ts = time.time()  # Fresh

        result = provider.get_batch_short_interest(["SPY"])
        assert "SPY" in result
        assert result["SPY"].signal == "LOW"

    def test_cache_stale(self, tmp_path: Path):
        provider = ShortInterestProvider(runtime_dir=tmp_path)
        provider._cache = {"SPY": {"symbol": "SPY", "short_float_pct": 0.03,
                                    "signal": "LOW", "squeeze_risk": False,
                                    "ts_utc": "2026-04-01T00:00:00Z"}}
        provider._cache_ts = time.time() - 7 * 3600  # 7 hours ago (> 6 hour TTL)

        with patch.object(provider, "_fetch_short_interest", side_effect=Exception("no network")):
            result = provider.get_batch_short_interest(["SPY"])
        assert result == {}


# ---------------------------------------------------------------------------
# Fail-silent behavior
# ---------------------------------------------------------------------------


class TestFailSilent:
    def test_network_error_returns_empty(self, tmp_path: Path):
        provider = ShortInterestProvider(runtime_dir=tmp_path)
        with patch("chad.intel.short_interest_provider.urllib.request.urlopen",
                   side_effect=Exception("Connection refused")):
            result = provider.get_batch_short_interest(["SPY"])
        assert result == {}

    def test_empty_symbols(self, tmp_path: Path):
        provider = ShortInterestProvider(runtime_dir=tmp_path)
        result = provider.get_batch_short_interest([])
        assert result == {}

    def test_single_symbol_fail_silent(self, tmp_path: Path):
        provider = ShortInterestProvider(runtime_dir=tmp_path)
        with patch.object(provider, "_fetch_short_interest", side_effect=Exception("fail")):
            result = provider.get_short_interest("SPY")
        assert result is None


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


class TestStatePersistence:
    def test_save_and_load(self, tmp_path: Path):
        provider = ShortInterestProvider(runtime_dir=tmp_path)
        sig = ShortInterestSignal(
            symbol="AAPL", short_float_pct=0.045, signal="LOW",
            squeeze_risk=False, ts_utc="2026-04-05T10:00:00Z",
        )
        # Populate cache as _fetch_short_interest would
        from dataclasses import asdict
        provider._cache = {"AAPL": asdict(sig)}
        provider._cache_ts = time.time()
        provider._save_state({"AAPL": sig})

        provider2 = ShortInterestProvider(runtime_dir=tmp_path)
        assert "AAPL" in provider2._cache
