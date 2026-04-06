"""
Tests for DailyCHADReport — plain English translations and report generation.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from chad.ops.daily_chad_report import (
    DailyCHADReport,
    MorningBrief,
    TradeRow,
    translate_instrument,
    translate_strategy,
    vix_description,
    load_today_trades,
    _format_money,
)


# ---------------------------------------------------------------------------
# Instrument translations
# ---------------------------------------------------------------------------

class TestInstrumentTranslation:
    def test_mes_translation(self):
        result = translate_instrument("MES")
        assert "S&P 500 futures" in result

    def test_es_translation(self):
        result = translate_instrument("ES")
        assert "S&P 500 futures" in result

    def test_mnq_translation(self):
        result = translate_instrument("MNQ")
        assert "Nasdaq futures" in result
        assert "tech stocks" in result

    def test_mcl_translation(self):
        result = translate_instrument("MCL")
        assert "Oil futures" in result

    def test_spy_translation(self):
        result = translate_instrument("SPY")
        assert "S&P 500 ETF" in result

    def test_btc_translation(self):
        result = translate_instrument("BTC-USD")
        assert result == "Bitcoin"

    def test_eth_translation(self):
        result = translate_instrument("ETH-USD")
        assert result == "Ethereum"

    def test_sol_translation(self):
        result = translate_instrument("SOL-USD")
        assert result == "Solana"

    def test_unknown_symbol_passthrough(self):
        result = translate_instrument("ZZZZ")
        assert result == "ZZZZ"

    def test_case_insensitive(self):
        result = translate_instrument("mes")
        assert "S&P 500 futures" in result

    def test_crash_protection(self):
        result = translate_instrument("SH")
        assert "Crash protection" in result

    def test_volatility_fund(self):
        result = translate_instrument("SVXY")
        assert "Volatility fund" in result


# ---------------------------------------------------------------------------
# Strategy translations
# ---------------------------------------------------------------------------

class TestStrategyTranslation:
    def test_alpha(self):
        assert "Momentum" in translate_strategy("alpha")

    def test_beta(self):
        assert "Long-term" in translate_strategy("beta")

    def test_gamma(self):
        assert "Swing" in translate_strategy("gamma")

    def test_omega(self):
        assert "Crash protection" in translate_strategy("omega")

    def test_delta(self):
        assert "Execution optimizer" in translate_strategy("delta")

    def test_alpha_futures(self):
        assert "Futures momentum" in translate_strategy("alpha_futures")

    def test_crypto(self):
        assert "Crypto" in translate_strategy("crypto")


# ---------------------------------------------------------------------------
# VIX descriptions
# ---------------------------------------------------------------------------

class TestVixDescription:
    def test_very_calm(self):
        result = vix_description(12.0)
        assert "Very calm" in result
        assert "😴" in result

    def test_normal(self):
        result = vix_description(17.0)
        assert "Normal" in result
        assert "🌤️" in result

    def test_bit_nervous(self):
        result = vix_description(23.0)
        assert "A bit nervous" in result
        assert "🌥️" in result

    def test_worried(self):
        result = vix_description(27.0)
        assert "Worried" in result
        assert "🌧️" in result

    def test_fearful(self):
        result = vix_description(35.0)
        assert "Fearful" in result
        assert "⛈️" in result

    def test_boundary_15(self):
        # 15 is "Normal", not "Very calm"
        result = vix_description(15.0)
        assert "Normal" in result

    def test_boundary_20(self):
        result = vix_description(20.0)
        assert "A bit nervous" in result

    def test_boundary_25(self):
        result = vix_description(25.0)
        assert "Worried" in result

    def test_boundary_30(self):
        result = vix_description(30.0)
        assert "Fearful" in result


# ---------------------------------------------------------------------------
# Money formatting
# ---------------------------------------------------------------------------

class TestFormatMoney:
    def test_large_amount(self):
        assert _format_money(1234.56) == "$1,235"

    def test_small_amount(self):
        assert _format_money(42.50) == "$42.50"

    def test_negative(self):
        assert _format_money(-500.0) == "$-500.00"

    def test_zero(self):
        assert _format_money(0.0) == "$0.00"


# ---------------------------------------------------------------------------
# Trade loading from temp files
# ---------------------------------------------------------------------------

class TestTradeLoading:
    def _write_trades(self, tmpdir: Path, date: str, trades: list):
        ledger = tmpdir / f"trade_history_{date}.ndjson"
        lines = []
        for i, t in enumerate(trades):
            rec = {
                "payload": t,
                "timestamp_utc": f"2026-03-28T00:00:0{i}Z",
                "record_hash": f"hash_{i}",
                "sequence_id": i,
            }
            lines.append(json.dumps(rec))
        ledger.write_text("\n".join(lines), encoding="utf-8")

    def test_loads_trusted_trades(self, tmp_path):
        from chad.ops.daily_chad_report import _today_yyyymmdd
        today = _today_yyyymmdd()
        self._write_trades(tmp_path, today, [
            {"strategy": "alpha", "symbol": "SPY", "side": "BUY", "pnl": 100.0, "is_live": False},
            {"strategy": "beta", "symbol": "QQQ", "side": "SELL", "pnl": -50.0, "is_live": False},
        ])
        rows = load_today_trades(tmp_path)
        assert len(rows) == 2
        assert rows[0].pnl == 100.0
        assert rows[1].pnl == -50.0

    def test_skips_untrusted(self, tmp_path):
        from chad.ops.daily_chad_report import _today_yyyymmdd
        today = _today_yyyymmdd()
        self._write_trades(tmp_path, today, [
            {"strategy": "alpha", "symbol": "SPY", "side": "BUY", "pnl": 100.0, "is_live": False},
            {"strategy": "alpha_futures", "symbol": "MCL", "side": "BUY", "pnl": 0.0, "is_live": False,
             "extra": {"pnl_untrusted": True}},
        ])
        rows = load_today_trades(tmp_path)
        assert len(rows) == 1

    def test_skips_live_trades(self, tmp_path):
        from chad.ops.daily_chad_report import _today_yyyymmdd
        today = _today_yyyymmdd()
        self._write_trades(tmp_path, today, [
            {"strategy": "alpha", "symbol": "SPY", "side": "BUY", "pnl": 100.0, "is_live": True},
        ])
        rows = load_today_trades(tmp_path)
        assert len(rows) == 0

    def test_empty_dir(self, tmp_path):
        rows = load_today_trades(tmp_path)
        assert rows == []


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

class TestDailyCHADReport:
    def _make_report_with_trades(self, tmp_path, trades):
        from chad.ops.daily_chad_report import _today_yyyymmdd
        today = _today_yyyymmdd()
        ledger = tmp_path / f"trade_history_{today}.ndjson"
        lines = []
        for i, t in enumerate(trades):
            rec = {
                "payload": t,
                "timestamp_utc": f"2026-04-04T12:00:0{i}Z",
                "record_hash": f"h{i}",
                "sequence_id": i,
            }
            lines.append(json.dumps(rec))
        ledger.write_text("\n".join(lines), encoding="utf-8")

        # Create runtime dir with pnl_state
        runtime = tmp_path.parent / "runtime"
        runtime.mkdir(exist_ok=True)
        pnl = {"account_equity": 998000.0, "realized_pnl": 0.0}
        (runtime / "pnl_state.json").write_text(json.dumps(pnl), encoding="utf-8")
        (runtime / "live_readiness.json").write_text(json.dumps({"ready_for_live": False}), encoding="utf-8")

        root = tmp_path.parent
        report = DailyCHADReport(repo_root=root, trades_dir=tmp_path)
        return report.generate()

    def test_no_trades_report(self, tmp_path):
        trades_dir = tmp_path / "trades"
        trades_dir.mkdir()
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        (runtime / "pnl_state.json").write_text(json.dumps({"account_equity": 998000}))
        (runtime / "live_readiness.json").write_text(json.dumps({"ready_for_live": False}))

        report = DailyCHADReport(repo_root=tmp_path, trades_dir=trades_dir)
        msg = report.generate()
        assert "CHAD's End of Day" in msg
        assert "Quiet day" in msg or "no trades" in msg.lower()
        assert "Practice mode" in msg
        # Fix 1: quiet days show condensed strategy line, not full breakdown
        assert "All 12 strategies are loaded and ready for Monday." in msg
        assert "😴 No trades today" not in msg
        # Fix 4: CHAD's Take always present
        assert "CHAD'S TAKE" in msg

    def test_winning_day(self, tmp_path):
        trades_dir = tmp_path / "trades"
        trades_dir.mkdir()
        msg = self._make_report_with_trades(trades_dir, [
            {"strategy": "alpha", "symbol": "SPY", "side": "BUY", "pnl": 200.0, "is_live": False},
            {"strategy": "beta", "symbol": "QQQ", "side": "BUY", "pnl": 150.0, "is_live": False},
            {"strategy": "gamma", "symbol": "GLD", "side": "SELL", "pnl": -30.0, "is_live": False},
        ])
        assert "we made" in msg.lower()
        assert "🟢" in msg
        assert "WHAT DID WE DO" in msg
        assert "3 trades" in msg
        assert "2 were winners" in msg
        assert "BEST MOVE" in msg
        assert "CHAD 🤝" in msg

    def test_losing_day(self, tmp_path):
        trades_dir = tmp_path / "trades"
        trades_dir.mkdir()
        msg = self._make_report_with_trades(trades_dir, [
            {"strategy": "alpha", "symbol": "SPY", "side": "BUY", "pnl": -200.0, "is_live": False},
            {"strategy": "beta", "symbol": "QQQ", "side": "SELL", "pnl": -50.0, "is_live": False},
        ])
        assert "lost" in msg.lower()
        assert "🔴" in msg
        assert "WORST MOVE" in msg

    def test_win_rate_above_target(self, tmp_path):
        trades_dir = tmp_path / "trades"
        trades_dir.mkdir()
        msg = self._make_report_with_trades(trades_dir, [
            {"strategy": "alpha", "symbol": "SPY", "side": "BUY", "pnl": 100.0, "is_live": False},
            {"strategy": "alpha", "symbol": "QQQ", "side": "BUY", "pnl": 100.0, "is_live": False},
            {"strategy": "alpha", "symbol": "GLD", "side": "BUY", "pnl": 100.0, "is_live": False},
            {"strategy": "alpha", "symbol": "TLT", "side": "SELL", "pnl": -50.0, "is_live": False},
        ])
        # 75% win rate > 55% target
        assert "above our 55% target" in msg

    def test_win_rate_below_target(self, tmp_path):
        trades_dir = tmp_path / "trades"
        trades_dir.mkdir()
        msg = self._make_report_with_trades(trades_dir, [
            {"strategy": "alpha", "symbol": "SPY", "side": "BUY", "pnl": 100.0, "is_live": False},
            {"strategy": "alpha", "symbol": "QQQ", "side": "BUY", "pnl": -50.0, "is_live": False},
            {"strategy": "alpha", "symbol": "GLD", "side": "BUY", "pnl": -50.0, "is_live": False},
            {"strategy": "alpha", "symbol": "TLT", "side": "SELL", "pnl": -50.0, "is_live": False},
        ])
        # 25% win rate < 55% target
        assert "below our 55% target" in msg

    def test_strategy_section_present(self, tmp_path):
        trades_dir = tmp_path / "trades"
        trades_dir.mkdir()
        msg = self._make_report_with_trades(trades_dir, [
            {"strategy": "alpha", "symbol": "SPY", "side": "BUY", "pnl": 100.0, "is_live": False},
        ])
        assert "WHAT'S WORKING" in msg
        assert "Momentum trading" in msg

    def test_instrument_in_best_move(self, tmp_path):
        trades_dir = tmp_path / "trades"
        trades_dir.mkdir()
        msg = self._make_report_with_trades(trades_dir, [
            {"strategy": "alpha_futures", "symbol": "MES", "side": "BUY", "pnl": 500.0, "is_live": False},
        ])
        assert "S&P 500 futures" in msg

    def test_sign_off(self, tmp_path):
        trades_dir = tmp_path / "trades"
        trades_dir.mkdir()
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        (runtime / "pnl_state.json").write_text(json.dumps({"account_equity": 998000}))
        (runtime / "live_readiness.json").write_text(json.dumps({"ready_for_live": False}))

        report = DailyCHADReport(repo_root=tmp_path, trades_dir=trades_dir)
        msg = report.generate()
        assert "See you tomorrow" in msg
        assert "CHAD 🤝" in msg

    def test_win_rate_from_scr(self, tmp_path):
        """Fix 3: Win rate should be read from scr_state.json."""
        trades_dir = tmp_path / "trades"
        trades_dir.mkdir()
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        (runtime / "pnl_state.json").write_text(json.dumps({"account_equity": 998000}))
        (runtime / "live_readiness.json").write_text(json.dumps({"ready_for_live": False}))
        (runtime / "scr_state.json").write_text(json.dumps({
            "stats": {"win_rate": 0.58},
        }))

        report = DailyCHADReport(repo_root=tmp_path, trades_dir=trades_dir)
        msg = report.generate()
        assert "Win rate: 58%" in msg
        assert "above our 55% target" in msg
        assert "🎯" in msg

    def test_vix_omitted_when_unavailable(self, tmp_path):
        """Fix 2: VIX line should be omitted, not show 'data not available'."""
        trades_dir = tmp_path / "trades"
        trades_dir.mkdir()
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        (runtime / "pnl_state.json").write_text(json.dumps({"account_equity": 998000}))
        (runtime / "live_readiness.json").write_text(json.dumps({"ready_for_live": False}))

        report = DailyCHADReport(repo_root=tmp_path, trades_dir=trades_dir)
        msg = report.generate()
        assert "data not available" not in msg

    def test_chads_take_present_on_quiet_day(self, tmp_path):
        """Fix 4: CHAD's Take should always appear, even on quiet days."""
        trades_dir = tmp_path / "trades"
        trades_dir.mkdir()
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        (runtime / "pnl_state.json").write_text(json.dumps({"account_equity": 998000}))
        (runtime / "live_readiness.json").write_text(json.dumps({"ready_for_live": False}))

        report = DailyCHADReport(repo_root=tmp_path, trades_dir=trades_dir)
        msg = report.generate()
        assert "CHAD'S TAKE" in msg
        # Fallback text should mention quiet/rest/Monday
        assert "quiet" in msg.lower() or "rest" in msg.lower() or "monday" in msg.lower()

    def test_practice_mode_label(self, tmp_path):
        trades_dir = tmp_path / "trades"
        trades_dir.mkdir()
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        (runtime / "pnl_state.json").write_text(json.dumps({"account_equity": 998000}))
        (runtime / "live_readiness.json").write_text(json.dumps({"ready_for_live": False}))

        report = DailyCHADReport(repo_root=tmp_path, trades_dir=trades_dir)
        msg = report.generate()
        assert "Practice mode" in msg
        # Should NOT use jargon
        assert "DRY_RUN" not in msg
        assert "paper trading" not in msg.lower().replace("paper account", "").replace("paper today", "").replace("on paper", "")


# ---------------------------------------------------------------------------
# Morning Brief
# ---------------------------------------------------------------------------

class TestMorningBrief:
    def test_morning_brief_structure(self, tmp_path):
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        brief = MorningBrief(repo_root=tmp_path)
        msg = brief.generate()
        assert "Good morning" in msg
        assert "Markets open" in msg
        assert "CHAD 🤝" in msg
        assert "full report" in msg

    def test_morning_brief_has_sections(self, tmp_path):
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        brief = MorningBrief(repo_root=tmp_path)
        msg = brief.generate()
        assert "Good morning" in msg
        # VIX line is omitted when data is unavailable (no "waiting for data")
        assert "waiting for data" not in msg
