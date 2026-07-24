"""
chad/tests/test_w6b_options_universe_coverage.py

W6B-11 (D6) — options-chain refresh universe is config-declared with an
asserted expected count.

The gap being closed is narrow and specific. R17 already promotes an empty
chains map or a non-empty ``error`` field to CRITICAL, and ``symbol_errors``
covers every symbol that was ATTEMPTED and failed. What nothing caught is a
symbol that is never attempted at all: the universe shrinks, ``chains`` is
smaller but non-empty, no error is set, the unit exits 0, and coverage has
silently dropped.

These tests pin:
  * the declared universe loads from config, and matches today's SPY-only reality
  * a symbols/expected_count contradiction REFUSES rather than falling back
  * absence/malformation falls back to the historical default (never an outage)
  * a never-attempted symbol produces a loud coverage_shortfall
  * a fully-covered run stays clean (no false alarm on the happy path)
"""

from __future__ import annotations

import json

import pytest

from chad.market_data import options_chain_refresh as ocr


# --------------------------------------------------------------------------
# The declared universe as it actually ships
# --------------------------------------------------------------------------

def test_shipped_config_is_spy_only_and_self_consistent():
    """D6 ruling: SPY-only is confirmed. Pin it so a silent edit is a test failure."""
    raw = json.loads(ocr.UNIVERSE_CONFIG_PATH.read_text(encoding="utf-8"))
    assert raw["schema_version"] == "options_universe.v1"
    assert raw["symbols"] == ["SPY"]
    assert raw["expected_count"] == 1
    assert len(raw["symbols"]) == raw["expected_count"]


def test_loader_reads_shipped_config():
    symbols, expected, source = ocr._load_declared_universe()
    assert symbols == ["SPY"]
    assert expected == 1
    assert source == "config"


# --------------------------------------------------------------------------
# Contradiction is fatal, not fallback
# --------------------------------------------------------------------------

def _point_at(tmp_path, monkeypatch, payload) -> None:
    p = tmp_path / "options_universe.json"
    p.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    monkeypatch.setattr(ocr, "UNIVERSE_CONFIG_PATH", p)


def test_emptied_symbol_list_with_stale_count_refuses(tmp_path, monkeypatch):
    """The dangerous edit: someone empties `symbols` and leaves expected_count=1.
    Falling back here would let a zero-coverage run report success — exactly the
    outcome the expected-count assertion exists to prevent."""
    _point_at(tmp_path, monkeypatch, {"symbols": [], "expected_count": 1})
    with pytest.raises(ocr.UniverseContradiction) as exc:
        ocr._load_declared_universe()
    assert "expected_count=1" in str(exc.value)


def test_added_symbol_without_bumping_count_refuses(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch, {"symbols": ["SPY", "QQQ"], "expected_count": 1})
    with pytest.raises(ocr.UniverseContradiction):
        ocr._load_declared_universe()


def test_contradiction_makes_main_exit_nonzero(tmp_path, monkeypatch):
    """A contradiction must stop the run, not just log. Proven through main(),
    and it must never reach run() (which would connect to IBKR)."""
    _point_at(tmp_path, monkeypatch, {"symbols": [], "expected_count": 3})
    monkeypatch.setattr("sys.argv", ["options_chain_refresh"])

    def _explode(*_a, **_kw):
        raise AssertionError("run() must not be reached on a contradiction")

    monkeypatch.setattr(ocr, "run", _explode)
    assert ocr.main() == 1


# --------------------------------------------------------------------------
# Absence must never harden into an outage
# --------------------------------------------------------------------------

def test_absent_config_falls_back_to_historical_default(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "UNIVERSE_CONFIG_PATH", tmp_path / "nope.json")
    symbols, expected, source = ocr._load_declared_universe()
    assert symbols == ["SPY"], "fallback must reproduce the pre-W6B-11 default exactly"
    assert expected == 1
    assert source == "fallback:absent"


def test_unreadable_config_falls_back(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch, "{not json at all")
    symbols, _, source = ocr._load_declared_universe()
    assert symbols == ["SPY"]
    assert source == "fallback:unreadable"


def test_malformed_config_falls_back(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch, {"symbols": "SPY", "expected_count": 1})
    symbols, _, source = ocr._load_declared_universe()
    assert symbols == ["SPY"]
    assert source == "fallback:malformed"


def test_boolean_expected_count_is_rejected_not_coerced(tmp_path, monkeypatch):
    """bool subclasses int in Python; `true` must not silently mean 1."""
    _point_at(tmp_path, monkeypatch, {"symbols": ["SPY"], "expected_count": True})
    symbols, _, source = ocr._load_declared_universe()
    assert source == "fallback:malformed"
    assert symbols == ["SPY"]


def test_non_object_config_falls_back(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch, ["SPY"])
    _, _, source = ocr._load_declared_universe()
    assert source == "fallback:not_an_object"


# --------------------------------------------------------------------------
# Coverage shortfall detection
# --------------------------------------------------------------------------

def test_never_attempted_symbol_is_detected():
    missing = ocr._coverage_shortfall(["SPY", "QQQ"], {"SPY": {"strikes": []}})
    assert missing == ["QQQ"]


def test_full_coverage_reports_no_shortfall():
    missing = ocr._coverage_shortfall(["SPY"], {"SPY": {"strikes": []}})
    assert missing == []


def test_shortfall_is_case_insensitive_on_fetched_keys():
    """Declared symbols are upper-cased; a lower-cased chain key is still a hit,
    not a phantom shortfall."""
    assert ocr._coverage_shortfall(["SPY"], {"spy": {}}) == []


def test_empty_declared_universe_reports_no_shortfall():
    """Degenerate input must not invent a shortfall — the empty-universe case is
    caught upstream by the contradiction check and the main() empty guard."""
    assert ocr._coverage_shortfall([], {"SPY": {}}) == []


# --------------------------------------------------------------------------
# main() wiring
# --------------------------------------------------------------------------

def test_main_uses_declared_universe_when_no_flag(monkeypatch):
    seen = {}

    def _capture(syms):
        seen["syms"] = list(syms)
        return 0

    monkeypatch.setattr(ocr, "run", _capture)
    monkeypatch.setattr("sys.argv", ["options_chain_refresh"])
    assert ocr.main() == 0
    assert seen["syms"] == ["SPY"]


def test_explicit_symbols_flag_overrides_declared_universe(monkeypatch):
    """An ad-hoc operator run must still work; it is an override, and logged as one."""
    seen = {}

    def _capture(syms):
        seen["syms"] = list(syms)
        return 0

    monkeypatch.setattr(ocr, "run", _capture)
    monkeypatch.setattr("sys.argv", ["options_chain_refresh", "--symbols", "qqq", "iwm"])
    assert ocr.main() == 0
    assert seen["syms"] == ["QQQ", "IWM"]


def test_main_refuses_an_empty_resolved_universe(tmp_path, monkeypatch):
    """symbols=[] with expected_count=0 is self-consistent, so it passes the
    contradiction check — but refreshing nothing must not report success."""
    _point_at(tmp_path, monkeypatch, {"symbols": [], "expected_count": 0})
    monkeypatch.setattr("sys.argv", ["options_chain_refresh"])

    def _explode(*_a, **_kw):
        raise AssertionError("run() must not be reached with an empty universe")

    monkeypatch.setattr(ocr, "run", _explode)
    assert ocr.main() == 1
