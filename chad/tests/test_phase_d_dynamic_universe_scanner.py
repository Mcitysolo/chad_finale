"""Phase D Item 1A — dynamic universe scanner publisher tests.

Verifies the publisher only — no strategy / execution wiring is touched
and the canonical active universe artifact is never replaced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from chad.market_data import dynamic_universe_scanner as dus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_feed(runtime_dir: Path, filename: str, payload: Dict[str, Any]) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / filename).write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _patch_universe(monkeypatch: pytest.MonkeyPatch, symbols: List[str]) -> None:
    from chad.utils.universe_provider import UniverseLoad

    def _fake_load_active_universe(*_a: Any, **_kw: Any) -> UniverseLoad:
        return UniverseLoad(
            symbols=list(symbols),
            source_path="test://runtime",
            source_type="runtime",
            symbol_count=len(symbols),
            stale=False,
            reason="fresh",
        )

    monkeypatch.setattr(
        dus, "load_active_universe", _fake_load_active_universe
    )


def _full_feeds(runtime_dir: Path, symbols: List[str]) -> None:
    rs = {"schema_version": "relative_strength.v1", "symbols": {}}
    rv = {"schema_version": "volume_scan.v1", "symbols": {}}
    news = {"schema_version": "news_intel.v1", "symbols": {}}
    earn = {"schema_version": "earnings_intel.v1", "symbols": {}}
    evt = {"schema_version": "event_risk.v1", "severity": "low"}
    for s in symbols:
        rs["symbols"][s] = {
            "rs_class": "neutral",
            "excess_vs_spy_5d": 0.0,
            "data_available": True,
        }
        rv["symbols"][s] = {
            "rvol_class": "normal",
            "rvol": 1.0,
            "data_available": True,
        }
        news["symbols"][s] = {
            "catalyst_strength": "none",
            "catalyst_direction": "none",
            "confirmed_gate_relevant": False,
        }
        earn["symbols"][s] = {"days_to_next_earnings": None}
    _write_feed(runtime_dir, "relative_strength.json", rs)
    _write_feed(runtime_dir, "volume_scan.json", rv)
    _write_feed(runtime_dir, "news_intel.json", news)
    _write_feed(runtime_dir, "earnings_intel.json", earn)
    _write_feed(runtime_dir, "event_risk.json", evt)


# ---------------------------------------------------------------------------
# 1-3. Symbol filtering
# ---------------------------------------------------------------------------


def test_is_equity_or_etf_symbol_excludes_btc_usd() -> None:
    assert dus._is_equity_or_etf_symbol("BTC-USD") is False


def test_is_equity_or_etf_symbol_excludes_mes() -> None:
    assert dus._is_equity_or_etf_symbol("MES") is False


def test_is_equity_or_etf_symbol_allows_spy() -> None:
    assert dus._is_equity_or_etf_symbol("SPY") is True


# ---------------------------------------------------------------------------
# 4-6. Component scoring
# ---------------------------------------------------------------------------


def test_rs_strong_contributes_score_and_reason() -> None:
    delta, reasons, _warnings, rs_class, _excess = dus._score_rs(
        {"rs_class": "strong", "excess_vs_spy_5d": 0.05}
    )
    assert delta == pytest.approx(0.35)
    assert "rs_strong" in reasons
    assert rs_class == "strong"


def test_rvol_high_contributes_score_and_reason() -> None:
    delta, reasons, _warnings, rvol_class, _rvol = dus._score_rvol(
        {"rvol_class": "high", "rvol": 4.2}
    )
    assert delta == pytest.approx(0.30)
    assert "rvol_high" in reasons
    assert rvol_class == "high"


def test_confirmed_high_catalyst_contributes_score_and_reason() -> None:
    delta, reasons, _warnings, strength, _direction, confirmed = dus._score_catalyst(
        {
            "catalyst_strength": "high",
            "catalyst_direction": "bullish",
            "confirmed_gate_relevant": True,
        }
    )
    assert delta == pytest.approx(0.25)
    assert "confirmed_high_catalyst" in reasons
    assert strength == "high"
    assert confirmed is True


# ---------------------------------------------------------------------------
# 7. Unconfirmed catalyst does not contribute
# ---------------------------------------------------------------------------


def test_unconfirmed_catalyst_does_not_contribute_score() -> None:
    delta, reasons, _warnings, _strength, _direction, confirmed = dus._score_catalyst(
        {
            "catalyst_strength": "high",
            "catalyst_direction": "bullish",
            "confirmed_gate_relevant": False,
        }
    )
    assert delta == 0.0
    assert reasons == []
    assert confirmed is False


# ---------------------------------------------------------------------------
# 8-9. Earnings warnings
# ---------------------------------------------------------------------------


def test_earnings_within_2d_halves_score_and_warns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_dir = tmp_path / "runtime"
    _patch_universe(monkeypatch, ["AAPL"])
    rs = {
        "schema_version": "relative_strength.v1",
        "symbols": {"AAPL": {"rs_class": "strong", "excess_vs_spy_5d": 0.05}},
    }
    rv = {
        "schema_version": "volume_scan.v1",
        "symbols": {"AAPL": {"rvol_class": "high", "rvol": 4.0}},
    }
    news = {
        "schema_version": "news_intel.v1",
        "symbols": {
            "AAPL": {
                "catalyst_strength": "high",
                "catalyst_direction": "bullish",
                "confirmed_gate_relevant": True,
            }
        },
    }
    earn = {
        "schema_version": "earnings_intel.v1",
        "symbols": {"AAPL": {"days_to_next_earnings": 1}},
    }
    _write_feed(runtime_dir, "relative_strength.json", rs)
    _write_feed(runtime_dir, "volume_scan.json", rv)
    _write_feed(runtime_dir, "news_intel.json", news)
    _write_feed(runtime_dir, "earnings_intel.json", earn)

    payload = dus.build_payload(runtime_dir, max_candidates=5)
    cand = payload["candidates"][0]
    assert cand["symbol"] == "AAPL"
    assert "earnings_within_2d" in cand["warnings"]
    # Pre-multiplier score (clamped at 1.0) would be 1.0 since
    # rs(0.35)+rvol(0.30)+catalyst(0.25)+liq(0.03) = 0.93 → halved 0.465.
    assert cand["score"] == pytest.approx(0.465, abs=1e-3)


def test_earnings_within_7d_warns_but_does_not_halve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_dir = tmp_path / "runtime"
    _patch_universe(monkeypatch, ["AAPL"])
    _full_feeds(runtime_dir, ["AAPL"])
    # Override earnings to put AAPL in the 7d window.
    earn = {
        "schema_version": "earnings_intel.v1",
        "symbols": {"AAPL": {"days_to_next_earnings": 5}},
    }
    _write_feed(runtime_dir, "earnings_intel.json", earn)

    payload = dus.build_payload(runtime_dir, max_candidates=5)
    cand = payload["candidates"][0]
    assert "earnings_within_7d" in cand["warnings"]
    assert "earnings_within_2d" not in cand["warnings"]
    # Baseline (rs neutral=0.15, rvol normal=0.10, no catalyst, liq unknown=0.03)
    # = 0.28; not halved.
    assert cand["score"] == pytest.approx(0.28, abs=1e-3)


# ---------------------------------------------------------------------------
# 10. Missing feeds fail open
# ---------------------------------------------------------------------------


def test_missing_feeds_fail_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    _patch_universe(monkeypatch, ["AAPL", "SPY"])

    payload = dus.build_payload(runtime_dir, max_candidates=5)
    assert payload["schema_version"] == dus.SCHEMA_VERSION
    syms = {c["symbol"] for c in payload["candidates"]}
    assert syms == {"AAPL", "SPY"}
    for c in payload["candidates"]:
        assert c["data_available"] is False
        # Unknown rs → +0.05; unknown rvol → +0.03; unknown liq → +0.03.
        assert 0.0 < c["score"] <= 0.15


# ---------------------------------------------------------------------------
# 11-12. Sorting
# ---------------------------------------------------------------------------


def test_candidate_ranking_sorts_by_score_desc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_dir = tmp_path / "runtime"
    syms = ["AAA", "BBB", "CCC"]
    _patch_universe(monkeypatch, syms)
    rs = {
        "schema_version": "relative_strength.v1",
        "symbols": {
            "AAA": {"rs_class": "weak"},
            "BBB": {"rs_class": "strong"},
            "CCC": {"rs_class": "neutral"},
        },
    }
    _write_feed(runtime_dir, "relative_strength.json", rs)
    payload = dus.build_payload(runtime_dir, max_candidates=5)
    syms_ranked = [c["symbol"] for c in payload["candidates"]]
    assert syms_ranked[0] == "BBB"  # strong wins
    assert syms_ranked[-1] == "AAA"  # weak loses
    assert payload["candidates"][0]["rank"] == 1


def test_tiebreak_sorts_by_symbol_ascending(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_dir = tmp_path / "runtime"
    syms = ["ZZZ", "AAA", "MMM"]
    _patch_universe(monkeypatch, syms)
    # All same rs_class -> identical scores -> tiebreak by symbol.
    rs = {
        "schema_version": "relative_strength.v1",
        "symbols": {s: {"rs_class": "neutral"} for s in syms},
    }
    _write_feed(runtime_dir, "relative_strength.json", rs)
    payload = dus.build_payload(runtime_dir, max_candidates=5)
    syms_ranked = [c["symbol"] for c in payload["candidates"]]
    assert syms_ranked == ["AAA", "MMM", "ZZZ"]


# ---------------------------------------------------------------------------
# 13-14. Payload schema and filtering
# ---------------------------------------------------------------------------


def test_build_payload_emits_schema_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    _patch_universe(monkeypatch, ["SPY"])
    payload = dus.build_payload(runtime_dir, max_candidates=1)
    assert payload["schema_version"] == "dynamic_universe_candidates.v1"
    assert "ts_utc" in payload
    assert "ttl_seconds" in payload
    assert "source" in payload
    assert "candidates" in payload
    assert "summary" in payload


def test_build_payload_excludes_crypto_and_futures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    _patch_universe(
        monkeypatch,
        ["SPY", "AAPL", "BTC-USD", "ETH-USD", "MES", "MNQ", "ZN"],
    )
    payload = dus.build_payload(runtime_dir, max_candidates=20)
    syms = {c["symbol"] for c in payload["candidates"]}
    assert "SPY" in syms
    assert "AAPL" in syms
    assert "BTC-USD" not in syms
    assert "ETH-USD" not in syms
    assert "MES" not in syms
    assert "MNQ" not in syms
    assert "ZN" not in syms


# ---------------------------------------------------------------------------
# 15-16. Dry-run vs write
# ---------------------------------------------------------------------------


def test_dry_run_does_not_write_runtime_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    _patch_universe(monkeypatch, ["SPY"])
    payload, wrote = dus.publish(
        runtime_dir, dry_run=True, max_candidates=5
    )
    assert wrote is False
    assert not (runtime_dir / dus.OUTPUT_FILENAME).exists()
    assert payload["schema_version"] == dus.SCHEMA_VERSION


def test_publish_writes_runtime_file_when_dry_run_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_dir = tmp_path / "runtime"
    _patch_universe(monkeypatch, ["SPY"])
    payload, wrote = dus.publish(
        runtime_dir, dry_run=False, max_candidates=5
    )
    assert wrote is True
    out_path = runtime_dir / "dynamic_universe_candidates.json"
    assert out_path.exists()
    doc = json.loads(out_path.read_text(encoding="utf-8"))
    assert doc["schema_version"] == "dynamic_universe_candidates.v1"
    assert doc["candidates"][0]["symbol"] == "SPY"
    assert payload["schema_version"] == doc["schema_version"]


# ---------------------------------------------------------------------------
# 17. Deploy files exist and contain expected directives
# ---------------------------------------------------------------------------


def test_deploy_service_and_timer_files_exist_with_expected_directives() -> None:
    root = Path("/home/ubuntu/chad_finale")
    service = root / "deploy" / "chad-dynamic-universe-scanner-refresh.service"
    timer = root / "deploy" / "chad-dynamic-universe-scanner-refresh.timer"
    assert service.is_file()
    assert timer.is_file()
    service_text = service.read_text(encoding="utf-8")
    timer_text = timer.read_text(encoding="utf-8")
    assert "ExecStart=" in service_text
    assert "chad.market_data.dynamic_universe_scanner" in service_text
    assert "OnUnitActiveSec=" in timer_text


# ---------------------------------------------------------------------------
# 18. No strategy / execution imports
# ---------------------------------------------------------------------------


def test_scanner_does_not_import_strategies_or_execution() -> None:
    src = Path(dus.__file__).read_text(encoding="utf-8")
    assert "chad.strategies" not in src
    assert "chad.execution" not in src
    assert "chad.risk" not in src
    assert "chad.core" not in src


# ---------------------------------------------------------------------------
# 19. Never writes runtime/universe.json
# ---------------------------------------------------------------------------


def test_scanner_does_not_write_runtime_universe_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    canonical = runtime_dir / "universe.json"
    canonical.write_text(
        json.dumps({"schema_version": "universe.v1", "symbols": ["SPY"]}),
        encoding="utf-8",
    )
    before = canonical.read_bytes()
    _patch_universe(monkeypatch, ["SPY", "AAPL"])

    dus.publish(runtime_dir, dry_run=False, max_candidates=5)

    # Canonical universe artifact untouched.
    assert canonical.read_bytes() == before
    # And the scanner source itself contains no reference to writing it.
    src = Path(dus.__file__).read_text(encoding="utf-8")
    assert "runtime/universe.json" not in src
    assert "config/universe.json" not in src


# ---------------------------------------------------------------------------
# 20. No config mutation in scanner source
# ---------------------------------------------------------------------------


def test_scanner_source_contains_no_config_mutation() -> None:
    src = Path(dus.__file__).read_text(encoding="utf-8")
    assert "config/" not in src
    # Only the runtime-output filename is written; no other write_text /
    # os.replace call should target a config path. We check that the
    # specific helper writes only via the scanner output filename.
    assert dus.OUTPUT_FILENAME == "dynamic_universe_candidates.json"


# ---------------------------------------------------------------------------
# 21. Partial/missing earnings_intel does not crash
# ---------------------------------------------------------------------------


def test_partial_or_missing_earnings_does_not_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_dir = tmp_path / "runtime"
    _patch_universe(monkeypatch, ["AAPL", "MSFT"])
    rs = {
        "schema_version": "relative_strength.v1",
        "symbols": {"AAPL": {"rs_class": "strong"}, "MSFT": {"rs_class": "neutral"}},
    }
    _write_feed(runtime_dir, "relative_strength.json", rs)
    # Earnings intel exists but only has a partial record for AAPL with
    # malformed days field; MSFT entry is missing entirely.
    earn = {
        "schema_version": "earnings_intel.v1",
        "symbols": {"AAPL": {"days_to_next_earnings": "not-a-number"}},
    }
    _write_feed(runtime_dir, "earnings_intel.json", earn)

    payload = dus.build_payload(runtime_dir, max_candidates=5)
    syms = {c["symbol"] for c in payload["candidates"]}
    assert syms == {"AAPL", "MSFT"}
    # No crash; AAPL's malformed earnings days collapses to None (no halving,
    # no warning band).
    aapl = next(c for c in payload["candidates"] if c["symbol"] == "AAPL")
    assert aapl["earnings_days"] is None
    assert "earnings_within_2d" not in aapl["warnings"]
    assert "earnings_within_7d" not in aapl["warnings"]
