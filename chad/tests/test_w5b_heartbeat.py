"""
W5B-4 — allocator_state.v1 heartbeat tests.

The heartbeat doctrine (fuse_box.publish_state, the XOV lesson): publish every
cycle INCLUDING all-off, so `intents_evaluated=0` is distinguishable from a
dead publisher. Plus the standing findings that bound what the shadow corpus
may claim, which must ride on every report rather than living in a doc.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chad.risk.allocator_limits import PortfolioLimits
from chad.risk.allocator_shadow_gate import (
    DEFAULT_STATE_PATH,
    MODE_OFF,
    MODE_SHADOW,
    STANDING_FINDINGS,
    STATE_SCHEMA,
    STATE_TTL_SECONDS,
    AllocatorShadowGate,
    build_state,
    publish_cycle_state,
    publish_state,
)
from chad.tests.test_w5b_exposure_core import REAL_BOOK, REAL_PRICES


@pytest.fixture()
def sectors():
    from chad.risk.fuse_box import load_sector_map, make_sector_lookup

    return make_sector_lookup(load_sector_map())


@pytest.fixture()
def gate(tmp_path, sectors):
    return AllocatorShadowGate(
        mode=MODE_SHADOW, positions=REAL_BOOK, prices=REAL_PRICES,
        sector_lookup=sectors, limits=PortfolioLimits.load(),
        evidence_dir=tmp_path / "ev",
    )


def _entry(symbol="AAPL", qty=10, price=321.01, side="BUY"):
    return {"symbol": symbol, "side": side, "quantity": qty, "sec_type": "STK",
            "limit_price": price, "strategy": "gamma", "meta": {}}


# --------------------------------------------------------------------------- #
# Heartbeat doctrine
# --------------------------------------------------------------------------- #

def test_heartbeat_written_when_gate_is_none(tmp_path):
    """A cycle with no intents constructs no gate — it must still publish."""
    p = tmp_path / "state.json"
    publish_cycle_state(None, p)
    s = json.loads(p.read_text(encoding="utf-8"))
    assert s["schema_version"] == STATE_SCHEMA
    assert s["mode"] == MODE_OFF
    assert s["active"] is False
    assert s["cycle"]["intents_evaluated"] == 0


def test_heartbeat_written_when_off(tmp_path, sectors):
    off = AllocatorShadowGate(mode=MODE_OFF, positions=REAL_BOOK,
                              prices=REAL_PRICES, sector_lookup=sectors,
                              evidence_dir=tmp_path / "ev")
    p = tmp_path / "state.json"
    publish_cycle_state(off, p)
    s = json.loads(p.read_text(encoding="utf-8"))
    assert s["mode"] == MODE_OFF
    assert s["active"] is False
    assert s["book"] is None
    assert s["cycle"]["intents_evaluated"] == 0


def test_heartbeat_carries_ts_and_ttl(tmp_path):
    p = tmp_path / "state.json"
    publish_state(build_state(None), p)
    s = json.loads(p.read_text(encoding="utf-8"))
    assert s["ts_utc"].endswith("Z")
    assert s["ttl_seconds"] == STATE_TTL_SECONDS == 180


def test_heartbeat_reports_the_cycle(gate, tmp_path):
    gate.observe(_entry())                      # approve
    gate.observe(_entry("LLY", 100, 1184.0))    # reject (per_symbol)
    gate.observe({"symbol": "SPY", "side": "EXIT", "quantity": 1,
                  "sec_type": "STK", "meta": {}})   # bypassed

    p = tmp_path / "state.json"
    publish_cycle_state(gate, p)
    s = json.loads(p.read_text(encoding="utf-8"))

    assert s["active"] is True
    assert s["mode"] == MODE_SHADOW
    c = s["cycle"]
    assert c["intents_evaluated"] == 2
    assert c["bypassed"] == 1
    assert c["would_approve"] == 1
    assert c["would_reject"] == 1
    assert c["by_limit"]["per_symbol"] == 1


def test_heartbeat_book_summary(gate, tmp_path):
    p = tmp_path / "state.json"
    publish_cycle_state(gate, p)
    book = json.loads(p.read_text(encoding="utf-8"))["book"]
    assert book["symbols"] == 11
    assert book["gross_usd"] == pytest.approx(671_037.40, abs=0.5)
    assert book["net_usd"] == pytest.approx(671_037.40, abs=0.5)
    assert book["by_sector"]["healthcare"] == pytest.approx(316_936.0, abs=0.5)
    assert book["currency_mix"] == {"USD": pytest.approx(671_037.40, abs=0.5)}


# --------------------------------------------------------------------------- #
# The derived-cap disclosure
# --------------------------------------------------------------------------- #

def test_heartbeat_names_the_unratified_binding_limits(gate, tmp_path):
    """A limit that BINDS but is not RATIFIED must be visible on the report,
    not only in the config — otherwise a reader of the heartbeat could take a
    would-reject for a ratified breach."""
    p = tmp_path / "state.json"
    publish_cycle_state(gate, p)
    limits = json.loads(p.read_text(encoding="utf-8"))["limits"]
    assert limits["unratified_derived"] == ["gross_exposure", "per_sector"]

    dims = limits["dimensions"]
    assert dims["gross_exposure"]["basis"] == "shadow_derivation_2026-07"
    assert dims["gross_exposure"]["ratified"] is False
    assert dims["per_symbol_concentration"]["ratified"] is True
    assert dims["net_exposure"]["binds"] is False


def test_heartbeat_records_equity_is_not_a_divisor(gate, tmp_path):
    p = tmp_path / "state.json"
    publish_cycle_state(gate, p)
    eq = json.loads(p.read_text(encoding="utf-8"))["limits"]["equity_basis"]
    assert eq["currency"] == "CAD"
    assert eq["used_as_divisor"] is False


# --------------------------------------------------------------------------- #
# Standing findings (§13.4)
# --------------------------------------------------------------------------- #

def test_standing_findings_ride_on_every_heartbeat(gate, tmp_path):
    """Including the OFF and no-gate heartbeats: the bypass finding bounds
    what the evidence claims whether or not the observer ran."""
    for g in (None, gate):
        p = tmp_path / f"state_{'none' if g is None else 'on'}.json"
        publish_cycle_state(g, p)
        findings = json.loads(p.read_text(encoding="utf-8"))["standing_findings"]
        ids = [f["id"] for f in findings]
        assert "W5B-SF1" in ids
        assert "W5B-SF2" in ids
        assert "W5B-F1" in ids


def test_sf1_states_what_the_evidence_cannot_claim():
    sf1 = next(f for f in STANDING_FINDINGS if f["id"] == "W5B-SF1")
    assert "does NOT support" in sf1["bounds"]
    assert "gross never exceeded" in sf1["bounds"]
    assert "enforce-flip PA" in sf1["blocks"]
    for word in ("overlay", "reconciler", "flatten", "flip"):
        assert word in sf1["detail"].lower()


def test_sf2_names_the_downstream_suppression_bound():
    sf2 = next(f for f in STANDING_FINDINGS if f["id"] == "W5B-SF2")
    assert "upper bound" in sf2["title"]
    for word in ("cooldown", "veto", "risk gate"):
        assert word in sf2["detail"].lower()


def test_every_standing_finding_is_well_formed():
    for f in STANDING_FINDINGS:
        assert set(f) >= {"id", "severity", "title", "detail", "bounds"}
        assert f["id"].startswith("W5B-")
        assert f["detail"] and f["bounds"]


# --------------------------------------------------------------------------- #
# Correlation is declarative on the heartbeat too
# --------------------------------------------------------------------------- #

def test_heartbeat_correlation_is_declarative_only(gate, tmp_path):
    """§13.3: no rho anywhere. The heartbeat names the regime and the
    deferral and carries no numeric correlation."""
    p = tmp_path / "state.json"
    publish_cycle_state(gate, p)
    corr = json.loads(p.read_text(encoding="utf-8"))["correlation"]
    assert corr["mode"] == "static_sector_buckets"
    assert corr["rolling_deferred_to"] == "R2"
    assert not [v for v in corr.values() if isinstance(v, (int, float))]


# --------------------------------------------------------------------------- #
# Degradation + leak guard
# --------------------------------------------------------------------------- #

def test_state_leak_guard_fires_on_default_path():
    with pytest.raises(RuntimeError, match="ALLOCATOR_ERROR"):
        publish_state({"x": 1}, None)
    with pytest.raises(RuntimeError, match="ALLOCATOR_ERROR"):
        publish_state({"x": 1}, DEFAULT_STATE_PATH)


def test_heartbeat_failure_is_not_fatal(tmp_path, monkeypatch):
    """A heartbeat failure must never end a trading cycle."""
    import chad.risk.allocator_shadow_gate as m

    monkeypatch.setattr(m, "build_state", lambda g: (_ for _ in ()).throw(ValueError("x")))
    publish_cycle_state(None, tmp_path / "s.json")  # must not raise


def test_construction_error_surfaces_on_the_heartbeat(tmp_path, monkeypatch, sectors):
    import chad.risk.allocator_shadow_gate as m

    monkeypatch.setattr(m, "build_base_book",
                        lambda **k: (_ for _ in ()).throw(OSError("no book")))
    gate = AllocatorShadowGate(mode=MODE_SHADOW, prices=REAL_PRICES,
                               sector_lookup=sectors,
                               evidence_dir=tmp_path / "ev")
    p = tmp_path / "state.json"
    publish_cycle_state(gate, p)
    s = json.loads(p.read_text(encoding="utf-8"))
    assert s["active"] is False
    assert "no book" in s["construction_error"]


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #

def test_live_loop_publishes_on_both_exits():
    """Both the no-intents early return and the end of the intent loop must
    publish, or a quiet cycle would look like a dead publisher."""
    src = (Path(__file__).resolve().parents[2] / "chad" / "core" / "live_loop.py").read_text(
        encoding="utf-8"
    )
    assert src.count("publish_cycle_state(") == 2
    assert "publish_cycle_state(None)" in src
    assert "publish_cycle_state(_allocator_gate)" in src

    early = src.index("No executable IBKR intents.")
    late = src.index("All intents skipped by signal/position guard.")
    assert early < src.index("publish_cycle_state(None)") < late
