"""W4A-3 — LC2 buckets (family + setup grains) + eventing + evaluator cycle.

Eventing pins (PLAN_W4A §7 / CTF-T2): marker line, evidence row, coach
generic-kind NOTIFY with warning severity, dedupe identity = rule+entity with
digits stripped and no values.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from chad.risk.fuse_box import (
    BucketSpec,
    FuseBoxConfig,
    FuseEvent,
    TrustedClose,
    build_lc2_buckets,
    dedupe_identity,
    emit_fuse_events,
    run_evaluator_cycle,
)

NOW = datetime(2026, 7, 23, 14, 0, 0, tzinfo=timezone.utc)


def _close(strategy="gamma", symbol="PSQ", pnl=-1.0, setup=None,
           regime="unknown", ts=NOW):
    return TrustedClose(
        strategy=strategy, symbol=symbol, side="BUY", pnl=pnl, exit_ts=ts,
        regime=regime, setup_family=setup, fill_ids=("a", "b"),
    )


def _cfg(tmp_path, obj):
    p = tmp_path / "fuse_box.json"
    p.write_text(json.dumps(obj))
    return FuseBoxConfig.load(p)


BASE_CFG = {
    "schema_version": "fuse_box.v1",
    "defaults": {"consecutive_losers": 3, "session_net_pnl_usd": None},
    "families": {
        "gamma": ["gamma", "gamma_futures", "gamma_reversion"],
        "alpha": ["alpha", "alpha_intraday"],
    },
    "setup_fuses": {"enabled_strategies": ["alpha", "alpha_intraday"]},
}


# --------------------------------------------------------------------------- #
# Family buckets
# --------------------------------------------------------------------------- #

def test_family_buckets_from_config(tmp_path):
    cfg = _cfg(tmp_path, BASE_CFG)
    buckets = build_lc2_buckets(cfg, [])
    fams = {b.fuse_id: b for b in buckets if b.kind == "family"}
    assert set(fams) == {"family:gamma", "family:alpha"}
    assert fams["family:gamma"].members == frozenset(
        {"gamma", "gamma_futures", "gamma_reversion"}
    )
    assert fams["family:gamma"].consecutive_losers == 3
    assert fams["family:gamma"].session_net_pnl_usd is None
    assert fams["family:gamma"].regimes is None  # GLOBAL leg by default


def test_family_threshold_overrides(tmp_path):
    obj = dict(BASE_CFG)
    obj["family_thresholds"] = {
        "gamma": {
            "consecutive_losers": 5,
            "session_net_pnl_usd": -500.0,
            "regimes": ["ranging", "unknown"],  # unknown must be stripped
        }
    }
    cfg = _cfg(tmp_path, obj)
    fam = {b.fuse_id: b for b in build_lc2_buckets(cfg, [])}["family:gamma"]
    assert fam.consecutive_losers == 5
    assert fam.session_net_pnl_usd == -500.0
    assert fam.regimes == frozenset({"ranging"})  # D2 rider: no unknown scope


def test_family_null_override_disables_leg(tmp_path):
    obj = dict(BASE_CFG)
    obj["family_thresholds"] = {"gamma": {"consecutive_losers": None}}
    cfg = _cfg(tmp_path, obj)
    fam = {b.fuse_id: b for b in build_lc2_buckets(cfg, [])}["family:gamma"]
    assert fam.consecutive_losers is None


def test_registry_parity_warns_not_raises(tmp_path, caplog):
    """Runtime warn leg: a families map missing active strategies warns loud
    and keeps building (config drift must never brick the engine)."""
    cfg = _cfg(tmp_path, {
        "families": {"tiny": ["gamma"]},
        "defaults": {"consecutive_losers": 3},
    })
    with caplog.at_level(logging.WARNING, logger="chad.risk.fuse_box"):
        buckets = build_lc2_buckets(cfg, [])
    assert len(buckets) == 1
    assert any("FUSE_CONFIG_FAMILY_PARITY" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# Setup buckets — derived from observed stamps (P13)
# --------------------------------------------------------------------------- #

def test_setup_buckets_from_observed_stamps(tmp_path):
    cfg = _cfg(tmp_path, BASE_CFG)
    closes = [
        _close(strategy="alpha_intraday", setup="ORB", pnl=-1),
        _close(strategy="alpha_intraday", setup="ORB", pnl=-2),
        _close(strategy="alpha_intraday", setup="VWAP_RECLAIM", pnl=3),
        _close(strategy="gamma", setup="ORB", pnl=-1),  # gamma not setup-enabled
        _close(strategy="alpha", setup=None, pnl=-1),   # unstamped: no bucket
    ]
    buckets = build_lc2_buckets(cfg, closes)
    setups = {b.fuse_id for b in buckets if b.kind == "setup"}
    assert setups == {
        "setup:alpha_intraday:ORB",
        "setup:alpha_intraday:VWAP_RECLAIM",
    }


def test_setup_bucket_matches_only_its_pair(tmp_path):
    cfg = _cfg(tmp_path, BASE_CFG)
    closes = [_close(strategy="alpha_intraday", setup="ORB", pnl=-1)]
    bucket = next(
        b for b in build_lc2_buckets(cfg, closes) if b.kind == "setup"
    )
    assert bucket.matches(closes[0])
    assert not bucket.matches(_close(strategy="alpha", setup="ORB"))
    assert not bucket.matches(_close(strategy="alpha_intraday", setup="GAP_GO"))


# --------------------------------------------------------------------------- #
# Eventing (§7 surfaces)
# --------------------------------------------------------------------------- #

def test_dedupe_identity_value_free():
    assert dedupe_identity("trip", "family:gamma") == "fuse_trip_family:gamma"
    # digits stripped — a numeric symbol can't mint per-value identities
    assert dedupe_identity("trip", "symbol:M2K") == "fuse_trip_symbol:mk"
    assert dedupe_identity("clear", "setup:alpha:ORB2") == (
        "fuse_clear_setup:alpha:orb"
    )


def test_emit_fuse_events_all_surfaces(tmp_path, caplog):
    sent = []

    def fake_notify(msg, **kw):
        sent.append((msg, kw))
        return True

    ev = FuseEvent(
        event="trip", fuse_id="family:gamma", kind="family",
        trip_rule="consecutive_losers", consecutive_losers=3,
        session_net_pnl=-112.28,
    )
    with caplog.at_level(logging.WARNING, logger="chad.risk.fuse_box"):
        n = emit_fuse_events(
            [ev], mode="shadow", evidence_dir=tmp_path / "ev",
            notify_fn=fake_notify, now=NOW,
        )
    assert n == 1
    # 1. marker
    assert any("FUSE_TRIP" in r.getMessage() for r in caplog.records)
    # 2. evidence row
    rows = [
        json.loads(l)
        for l in (tmp_path / "ev" / "fuse_box_20260723.ndjson")
        .read_text().splitlines()
    ]
    assert rows[0]["event"] == "trip" and rows[0]["mode"] == "shadow"
    assert rows[0]["session_net_pnl"] == -112.28
    # 3+4. notify with warning severity + value-free dedupe key
    msg, kw = sent[0]
    assert kw["severity"] == "warning"
    assert kw["dedupe_key"] == "fuse_trip_family:gamma"
    assert "-112" not in kw["dedupe_key"]
    assert msg  # coach or fallback text present


def test_emit_fuse_events_notify_failure_never_raises(tmp_path):
    def broken_notify(msg, **kw):
        raise ConnectionError("telegram down")

    ev = FuseEvent(
        event="clear", fuse_id="family:gamma", kind="family",
        trip_rule=None, consecutive_losers=0, session_net_pnl=0.0,
    )
    n = emit_fuse_events(
        [ev], mode="shadow", evidence_dir=tmp_path / "ev",
        notify_fn=broken_notify, now=NOW,
    )
    assert n == 0  # counted as not emitted, but no exception escaped


# --------------------------------------------------------------------------- #
# Evaluator cycle (the W4A-5 wiring surface)
# --------------------------------------------------------------------------- #

def _cycle_kw(tmp_path, env):
    return dict(
        env=env,
        trades_dir=tmp_path / "trades",
        fills_dir=tmp_path / "fills",
        runtime_dir=tmp_path / "runtime",
        epoch_state_path=tmp_path / "runtime" / "epoch_state.json",
        state_path=tmp_path / "runtime" / "fuse_box_state.json",
        evidence_dir=tmp_path / "ev",
        notify_fn=lambda *a, **k: True,
    )


def test_cycle_all_off_publishes_bare_heartbeat(tmp_path):
    state = run_evaluator_cycle(NOW, **_cycle_kw(tmp_path, env={}))
    assert state["schema_version"] == "fuse_box_state.v1"
    assert state["modes"] == {"lc2": "off", "lc3": "off", "lc5": "off", "dq": "off"}
    assert state["fuses"] == []
    assert state["ts_utc"] and state["ttl_seconds"] == 180
    on_disk = json.loads(
        (tmp_path / "runtime" / "fuse_box_state.json").read_text()
    )
    assert on_disk["counting"]["trusted_closes"] == 0


def test_cycle_shadow_counts_and_trips(tmp_path):
    """End-to-end: synthetic trusted ledger → family trip event + state."""
    import pathlib

    from chad.tests.test_w4a_fuse_box import (  # reuse incident-shape builders
        _fill_row, _trade_row, _write_ndjson,
    )

    DATE = "20260723"
    fills = [
        _fill_row(f"o{i}", "gamma", "PSQ", "BUY", 1.0, 26.0, "paper_fill",
                  "2026-07-23T10:00:00+00:00", i)
        for i in range(3)
    ] + [
        _fill_row(f"c{i}", "gamma", "PSQ", "SELL", 1.0, 25.0, "paper_fill",
                  "2026-07-23T13:00:00+00:00", 10 + i)
        for i in range(3)
    ]
    trades = [
        _trade_row("gamma", "PSQ", -5.0, [f"o{i}", f"c{i}"],
                   exit_ts=f"2026-07-23T13:0{i}:00+00:00")
        for i in range(3)
    ]
    kw = _cycle_kw(tmp_path, env={"CHAD_FUSE_LC2": "shadow"})
    _write_ndjson(pathlib.Path(kw["fills_dir"]) / f"FILLS_{DATE}.ndjson", fills)
    _write_ndjson(
        pathlib.Path(kw["trades_dir"]) / f"trade_history_{DATE}.ndjson", trades
    )

    state = run_evaluator_cycle(NOW, **kw)
    assert state["counting"]["trusted_closes"] == 3
    fam = {r["fuse_id"]: r for r in state["fuses"]}
    assert fam["family:gamma"]["tripped"] is True
    assert fam["family:gamma"]["trip_rule"] == "consecutive_losers"
    # trip evidence written
    ev_rows = [
        json.loads(l)
        for l in (tmp_path / "ev" / f"fuse_box_{DATE}.ndjson")
        .read_text().splitlines()
    ]
    assert any(r["event"] == "trip" and r["fuse_id"] == "family:gamma"
               for r in ev_rows)

    # Second cycle: still tripped, edge-triggered → no new trip evidence.
    n_before = len(ev_rows)
    state2 = run_evaluator_cycle(NOW, **kw)
    fam2 = {r["fuse_id"]: r for r in state2["fuses"]}
    assert fam2["family:gamma"]["tripped"] is True
    assert (
        fam2["family:gamma"]["tripped_at_utc"]
        == fam["family:gamma"]["tripped_at_utc"]
    )
    ev_rows2 = [
        json.loads(l)
        for l in (tmp_path / "ev" / f"fuse_box_{DATE}.ndjson")
        .read_text().splitlines()
    ]
    assert len(ev_rows2) == n_before  # no re-emit while tripped


def test_cycle_failure_soft_still_heartbeats(tmp_path, monkeypatch):
    """A counting failure must not kill the cycle — wait, stronger: the
    evaluator itself is wrapped failure-soft at the live_loop call site
    (W4A-5); HERE we pin that a corrupt prior state file cannot break the
    pass (read_json canon returns None for garbage)."""
    kw = _cycle_kw(tmp_path, env={})
    (tmp_path / "runtime").mkdir(parents=True, exist_ok=True)
    (tmp_path / "runtime" / "fuse_box_state.json").write_text("{corrupt")
    state = run_evaluator_cycle(NOW, **kw)
    assert state["schema_version"] == "fuse_box_state.v1"
