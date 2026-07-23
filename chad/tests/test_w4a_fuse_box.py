"""W4A-1 — fuse box core: trusted counting, trip/clear engine, state publisher.

Covers the GO-record binding requirements (audits/W4A_GO_RECORD.md):
- §5a INCIDENT-0723 inheritance: a drill's dry_run exhaust must never move a
  fuse counter — replayed with the exact incident row shapes
  (test_incident_0723_dry_run_exclusion.py lineage), both directions
  (dry_run exit leg AND dry_run entry leg with genuine exit).
- D2 rider: unknown-regime rows count toward GLOBAL legs only; a
  regime-scoped leg can never count them, and "unknown" is stripped from
  config scopes.
- Heartbeat doctrine: state written including all-modes-OFF; margin-gate-style
  test-write leak guards on state/evidence default paths.
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone

import pytest

from chad.risk.fuse_box import (
    ENV_LC2,
    GENUINE_FILL_STATUSES,
    BucketSpec,
    FuseBoxConfig,
    TrustedClose,
    append_evidence,
    build_state,
    compute_bucket_stats,
    evaluate_buckets,
    fuse_mode,
    load_trusted_session_closes,
    load_window_fill_statuses,
    publish_state,
    read_modes,
    sanitize_regime_scope,
    session_window_start,
)

DATE = "20260723"
NOW = datetime(2026, 7, 23, 14, 0, 0, tzinfo=timezone.utc)
TS_OPEN = "2026-07-23T10:00:00+00:00"
TS_DRILL = "2026-07-23T10:26:04+00:00"
TS_CLOSE = "2026-07-23T13:31:30+00:00"


# --------------------------------------------------------------------------- #
# Fixture builders (incident row shapes)
# --------------------------------------------------------------------------- #

def _fill_row(fid, strategy, symbol, side, qty, px, status, ts, seq):
    """FILLS_*.ndjson row in the exact INCIDENT-0723 shape (paper_exec_fill.v4)."""
    return {
        "payload": {
            "schema_version": "paper_exec_fill.v4",
            "fill_id": fid,
            "strategy": strategy,
            "symbol": symbol,
            "side": side,
            "quantity": qty,
            "fill_price": px,
            "fill_time_utc": ts,
            "is_live": False,
            "reject": False,
            "status": status,
            "source": "paper_trade_executor",
            "order_type": "SIM",
        },
        "sequence_id": seq,
        "timestamp_utc": ts,
        "prev_hash": "GENESIS",
        "record_hash": f"rh_{fid}",
    }


def _trade_row(
    strategy,
    symbol,
    pnl,
    fill_ids,
    *,
    exit_ts=TS_CLOSE,
    schema="closed_trade.v1",
    tags=None,
    extra=None,
    meta=None,
    regime=None,
):
    # Base price 200 (not 100) so a pnl of -0.0 can never collide with the
    # $100 placeholder fingerprint trust_exclusion hunts first.
    payload = {
        "schema_version": schema,
        "strategy": strategy,
        "symbol": symbol,
        "side": "BUY",
        "pnl": pnl,
        "net_pnl": pnl,
        "entry_time_utc": TS_OPEN,
        "exit_time_utc": exit_ts,
        "entry_price": 200.0,
        "exit_price": 200.0 + pnl,
        "fill_price": 200.0 + pnl,
        "quantity": 1.0,
        "contract_multiplier": 1.0,
        "fill_ids": list(fill_ids),
        "broker": "paper_exec",
        "tags": tags if tags is not None else ["paper", "closed", strategy],
        "meta": meta if meta is not None else {},
    }
    if extra is not None:
        payload["extra"] = extra
    if regime is not None:
        payload["regime"] = regime
    return {"payload": payload, "record_hash": f"rh_trade_{symbol}_{pnl}"}


def _write_ndjson(path: pathlib.Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _dirs(tmp_path):
    return dict(
        trades_dir=tmp_path / "trades",
        fills_dir=tmp_path / "fills",
        runtime_dir=tmp_path / "runtime",
        epoch_state_path=tmp_path / "runtime" / "epoch_state.json",
    )


def _genuine_pair(fid_open, fid_close, symbol, strategy="gamma"):
    """Two genuine paper_fill FILLS rows for one round-trip."""
    return [
        _fill_row(fid_open, strategy, symbol, "BUY", 1.0, 100.0, "paper_fill", TS_OPEN, 1),
        _fill_row(fid_close, strategy, symbol, "SELL", 1.0, 99.0, "paper_fill", TS_CLOSE, 2),
    ]


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #

def test_mode_defaults_off():
    assert fuse_mode(ENV_LC2, env={}) == "off"


@pytest.mark.parametrize("raw,expect", [
    ("off", "off"), ("shadow", "shadow"), ("enforce", "enforce"),
    ("SHADOW", "shadow"), (" enforce ", "enforce"),
    ("1", "off"), ("true", "off"), ("garbage", "off"), ("", "off"),
])
def test_mode_tristate_garbage_off(raw, expect):
    assert fuse_mode(ENV_LC2, env={ENV_LC2: raw}) == expect


def test_read_modes_all_four_keys():
    assert set(read_modes(env={})) == {"lc2", "lc3", "lc5", "dq"}


# --------------------------------------------------------------------------- #
# Session window (GAP-026)
# --------------------------------------------------------------------------- #

def test_window_is_midnight_without_epoch(tmp_path):
    start = session_window_start(NOW, tmp_path / "missing_epoch.json")
    assert start == datetime(2026, 7, 23, tzinfo=timezone.utc)


def test_window_is_epoch_start_when_later(tmp_path):
    ep = tmp_path / "epoch_state.json"
    ep.write_text(json.dumps({"epoch_started_at_utc": "2026-07-23T09:30:00Z"}))
    start = session_window_start(NOW, ep)
    assert start == datetime(2026, 7, 23, 9, 30, tzinfo=timezone.utc)


def test_window_epoch_in_past_uses_midnight(tmp_path):
    ep = tmp_path / "epoch_state.json"
    ep.write_text(json.dumps({"epoch_started_at_utc": "2026-06-30T12:17:42Z"}))
    start = session_window_start(NOW, ep)
    assert start == datetime(2026, 7, 23, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def test_config_load_missing_file_safe_defaults(tmp_path):
    cfg = FuseBoxConfig.load(tmp_path / "nope.json")
    assert cfg.default_consecutive_losers == 3
    assert cfg.families == {}
    assert cfg.setup_fuse_strategies == ()


def test_config_load_corrupt_file_safe_defaults(tmp_path):
    p = tmp_path / "fuse_box.json"
    p.write_text("{not json")
    cfg = FuseBoxConfig.load(p)
    assert cfg.default_consecutive_losers == 3
    assert cfg.raw == {}


def test_config_load_real_repo_config_parses():
    cfg = FuseBoxConfig.load()
    assert cfg.default_consecutive_losers == 3
    assert cfg.default_session_net_pnl_usd is None


# --------------------------------------------------------------------------- #
# Counting predicates
# --------------------------------------------------------------------------- #

def test_trusted_close_admitted(tmp_path):
    d = _dirs(tmp_path)
    _write_ndjson(d["fills_dir"] / f"FILLS_{DATE}.ndjson",
                  _genuine_pair("f_open", "f_close", "PSQ"))
    _write_ndjson(d["trades_dir"] / f"trade_history_{DATE}.ndjson",
                  [_trade_row("gamma", "PSQ", -1.0, ["f_open", "f_close"])])
    closes, tally = load_trusted_session_closes(NOW, **d)
    assert len(closes) == 1
    assert closes[0].strategy == "gamma"
    assert closes[0].pnl == -1.0
    assert closes[0].regime == "unknown"  # unstamped (pre-W4A-2 rows)
    assert tally == {}


@pytest.mark.parametrize("mutate,expected_reason", [
    (lambda r: r["payload"].__setitem__("schema_version", "paper_trade_result.v1"),
     "non_closed_trade"),
    (lambda r: r["payload"].__setitem__("exit_time_utc", "2026-07-20T00:00:00Z"),
     "out_of_window"),
    # extra/tags pnl_untrusted is caught by the QUARANTINE belt first
    # (is_record_quarantined precedes trust_exclusion — same net exclusion,
    # earlier reason). A meta-only marker reaches trust_exclusion.
    (lambda r: r["payload"].__setitem__("extra", {"pnl_untrusted": True}),
     "quarantined"),
    (lambda r: r["payload"].__setitem__("meta", {"pnl_untrusted": True}),
     "pnl_untrusted"),
    (lambda r: r["payload"].__setitem__("extra", {"scoring_excluded": True}),
     "scoring_excluded"),
    (lambda r: r["payload"]["tags"].append("validate_only"), "validate_only"),
    (lambda r: r["payload"]["tags"].append("manual"), "manual"),
    (lambda r: r["payload"]["tags"].append("warmup_sim"), "warmup_sim"),
    (lambda r: r["payload"].__setitem__("strategy", "broker_sync"),
     "strategy_excluded"),
    (lambda r: r["payload"].__setitem__("symbol", "MES"), "futures_bug_b"),
    (lambda r: r["payload"].__setitem__("net_pnl", "zilch") or
     r["payload"].__setitem__("pnl", "zilch"), "malformed_pnl"),
])
def test_exclusion_reasons(tmp_path, mutate, expected_reason):
    d = _dirs(tmp_path)
    _write_ndjson(d["fills_dir"] / f"FILLS_{DATE}.ndjson",
                  _genuine_pair("f_open", "f_close", "PSQ"))
    row = _trade_row("gamma", "PSQ", -1.0, ["f_open", "f_close"])
    mutate(row)
    _write_ndjson(d["trades_dir"] / f"trade_history_{DATE}.ndjson", [row])
    closes, tally = load_trusted_session_closes(NOW, **d)
    assert closes == []
    assert tally.get(expected_reason) == 1, tally


def test_quarantine_manifest_pin_excludes(tmp_path):
    d = _dirs(tmp_path)
    _write_ndjson(d["fills_dir"] / f"FILLS_{DATE}.ndjson",
                  _genuine_pair("f_open", "f_close", "PSQ"))
    _write_ndjson(d["trades_dir"] / f"trade_history_{DATE}.ndjson",
                  [_trade_row("gamma", "PSQ", -1.0, ["f_open", "f_close"])])
    d["runtime_dir"].mkdir(parents=True, exist_ok=True)
    (d["runtime_dir"] / "quarantine_manifest_test.json").write_text(json.dumps({
        "schema_version": "fills_quarantine.v1",
        "invalid_fills": ["f_close"],
        "invalid_trades": [],
    }))
    closes, tally = load_trusted_session_closes(NOW, **d)
    assert closes == []
    assert tally.get("quarantined") == 1


# --------------------------------------------------------------------------- #
# INCIDENT-0723 inheritance (a): provenance verification, by construction
# --------------------------------------------------------------------------- #

def test_window_fill_statuses_maps_window_files_only(tmp_path):
    fills = tmp_path / "fills"
    _write_ndjson(fills / f"FILLS_{DATE}.ndjson",
                  [_fill_row("in_window", "gamma", "PSQ", "SELL", 5.0, 26.2,
                             "dry_run", TS_DRILL, 1)])
    _write_ndjson(fills / "FILLS_20260601.ndjson",
                  [_fill_row("out_window", "gamma", "PSQ", "BUY", 5.0, 26.2,
                             "paper_fill", "2026-06-01T14:00:00Z", 1)])
    window_start = datetime(2026, 7, 23, tzinfo=timezone.utc)
    statuses = load_window_fill_statuses(window_start, NOW, fills)
    assert statuses == {"in_window": "dry_run"}


def test_dry_run_exit_leg_condemns_row(tmp_path):
    """The 8 fake 07-23 rows: genuine entry, dry_run exit → excluded."""
    d = _dirs(tmp_path)
    _write_ndjson(d["fills_dir"] / f"FILLS_{DATE}.ndjson", [
        _fill_row("open_v", "gamma", "V", "BUY", 9.0, 349.0, "paper_fill", TS_OPEN, 1),
        _fill_row("drill_v", "gamma", "V", "SELL", 9.0, 349.0, "dry_run", TS_DRILL, 2),
    ])
    _write_ndjson(d["trades_dir"] / f"trade_history_{DATE}.ndjson",
                  [_trade_row("gamma", "V", -964.73, ["open_v", "drill_v"])])
    closes, tally = load_trusted_session_closes(NOW, **d)
    assert closes == []
    assert tally.get("unverified_provenance") == 1


def test_dry_run_entry_leg_condemns_row(tmp_path):
    """The reverse incident shape (13:31:30 PSQ): dry_run entry, genuine
    paper_fill exit — exit-only verification would admit it; ours must not."""
    d = _dirs(tmp_path)
    _write_ndjson(d["fills_dir"] / f"FILLS_{DATE}.ndjson", [
        _fill_row("drill_short", "g", "PSQ", "SELL", 5.0, 26.2, "dry_run", TS_DRILL, 1),
        _fill_row("buyback", "g", "PSQ", "BUY", 5.0, 26.61, "paper_fill", TS_CLOSE, 2),
    ])
    _write_ndjson(d["trades_dir"] / f"trade_history_{DATE}.ndjson",
                  [_trade_row("g", "PSQ", -0.0, ["drill_short", "buyback"])])
    closes, tally = load_trusted_session_closes(NOW, **d)
    assert closes == []
    assert tally.get("unverified_provenance") == 1


def test_unresolvable_old_entry_fill_does_not_condemn(tmp_path):
    """An entry fill older than the window (not in any window FILLS file)
    must NOT condemn — it is vetted by the quarantine/trust belts instead."""
    d = _dirs(tmp_path)
    _write_ndjson(d["fills_dir"] / f"FILLS_{DATE}.ndjson", [
        _fill_row("f_close", "gamma", "UNH", "SELL", 1.0, 428.0, "paper_fill", TS_CLOSE, 1),
    ])
    _write_ndjson(d["trades_dir"] / f"trade_history_{DATE}.ndjson",
                  [_trade_row("gamma", "UNH", 12.5, ["ancient_open", "f_close"])])
    closes, tally = load_trusted_session_closes(NOW, **d)
    assert len(closes) == 1
    assert tally == {}


def test_drill_never_trips_a_fuse(tmp_path):
    """GO record §5a, the named inheritance test: a full drill replay —
    incident FILLS shapes + the fake trade_history rows they minted —
    produces ZERO fuse counter movement. One genuine losing close proves the
    counter still works; three fake gamma losses (enough to trip N=3 if
    counted) are all condemned by provenance."""
    d = _dirs(tmp_path)
    _write_ndjson(d["fills_dir"] / f"FILLS_{DATE}.ndjson", [
        # Real opens (paper_fill) — the incident's 10:00 book.
        _fill_row("open_ma", "gamma", "MA", "BUY", 10.0, 590.0, "paper_fill", TS_OPEN, 1),
        _fill_row("open_v", "gamma", "V", "BUY", 9.0, 349.0, "paper_fill", TS_OPEN, 2),
        _fill_row("open_svxy", "gamma", "SVXY", "BUY", 156.0, 56.2, "paper_fill", TS_OPEN, 3),
        _fill_row("open_unh", "gamma", "UNH", "BUY", 1.0, 428.0, "paper_fill", TS_OPEN, 4),
        # Drill exhaust: dry_run SELLs (adapter dry-run short-circuit).
        _fill_row("drill_ma", "gamma", "MA", "SELL", 10.0, 579.2, "dry_run", TS_DRILL, 5),
        _fill_row("drill_v", "gamma", "V", "SELL", 9.0, 349.0, "dry_run", TS_DRILL, 6),
        _fill_row("drill_svxy", "gamma", "SVXY", "SELL", 156.0, 56.2, "dry_run", TS_DRILL, 7),
        # One genuine close (both legs real).
        _fill_row("close_unh", "gamma", "UNH", "SELL", 1.0, 427.0, "paper_fill", TS_CLOSE, 8),
    ])
    _write_ndjson(d["trades_dir"] / f"trade_history_{DATE}.ndjson", [
        # The fake round-trips the incident minted (§3.2 pnls: MA −107.60,
        # V −964.73, SVXY −4.68) — three gamma losses in a row, exactly what
        # would trip family:gamma at N=3 if counted.
        _trade_row("gamma", "MA", -107.6, ["open_ma", "drill_ma"],
                   exit_ts="2026-07-23T10:26:30+00:00"),
        _trade_row("gamma", "V", -964.73, ["open_v", "drill_v"],
                   exit_ts="2026-07-23T10:26:31+00:00"),
        _trade_row("gamma", "SVXY", -4.68, ["open_svxy", "drill_svxy"],
                   exit_ts="2026-07-23T10:26:32+00:00"),
        # The genuine loser.
        _trade_row("gamma", "UNH", -1.0, ["open_unh", "close_unh"]),
    ])
    closes, tally = load_trusted_session_closes(NOW, **d)
    assert len(closes) == 1 and closes[0].symbol == "UNH"
    assert tally.get("unverified_provenance") == 3

    buckets = [
        BucketSpec(fuse_id="family:gamma", kind="family",
                   members=frozenset({"gamma"}), consecutive_losers=3),
        BucketSpec(fuse_id="symbol:V", kind="symbol",
                   members=frozenset({"V"}), consecutive_losers=1),
    ]
    rows, events = evaluate_buckets(buckets, closes, now=NOW)
    assert all(not r["tripped"] for r in rows), rows
    assert events == []
    fam = next(r for r in rows if r["fuse_id"] == "family:gamma")
    assert fam["consecutive_losers"] == 1  # the genuine loser only


def test_genuine_fill_statuses_census_membership():
    """Belt over the 8f census extension: the fuse allowlist never admits a
    rehearsal status."""
    assert "dry_run" not in GENUINE_FILL_STATUSES
    assert "market_closed" not in GENUINE_FILL_STATUSES
    assert GENUINE_FILL_STATUSES == frozenset({"paper_fill", "fill", "filled"})


# --------------------------------------------------------------------------- #
# Streak + trip engine
# --------------------------------------------------------------------------- #

def _close(strategy="gamma", symbol="PSQ", pnl=-1.0, ts=NOW, regime="unknown",
           setup=None):
    return TrustedClose(
        strategy=strategy, symbol=symbol, side="BUY", pnl=pnl, exit_ts=ts,
        regime=regime, setup_family=setup, fill_ids=("a", "b"),
    )


def test_streak_counts_trailing_losers():
    spec = BucketSpec(fuse_id="family:g", kind="family",
                      members=frozenset({"gamma"}), consecutive_losers=3)
    closes = [_close(pnl=-1), _close(pnl=-2), _close(pnl=-3)]
    stats = compute_bucket_stats(spec, closes)
    assert stats.consecutive_losers == 3
    assert stats.tripped and stats.trip_rule == "consecutive_losers"


def test_streak_reset_by_winner():
    spec = BucketSpec(fuse_id="family:g", kind="family",
                      members=frozenset({"gamma"}), consecutive_losers=3)
    closes = [_close(pnl=-1), _close(pnl=-2), _close(pnl=5.0), _close(pnl=-3)]
    stats = compute_bucket_stats(spec, closes)
    assert stats.consecutive_losers == 1
    assert not stats.tripped


def test_streak_scratch_neither_extends_nor_resets():
    spec = BucketSpec(fuse_id="family:g", kind="family",
                      members=frozenset({"gamma"}), consecutive_losers=3)
    closes = [_close(pnl=-1), _close(pnl=-2), _close(pnl=0.0), _close(pnl=-3)]
    stats = compute_bucket_stats(spec, closes)
    assert stats.consecutive_losers == 3
    assert stats.tripped


def test_pnl_threshold_trips():
    spec = BucketSpec(fuse_id="family:g", kind="family",
                      members=frozenset({"gamma"}),
                      consecutive_losers=None, session_net_pnl_usd=-100.0)
    closes = [_close(pnl=-60.0), _close(pnl=-45.0)]
    stats = compute_bucket_stats(spec, closes)
    assert stats.tripped and stats.trip_rule == "session_net_pnl"


def test_pnl_threshold_needs_rows():
    """An empty bucket can never trip on the $ leg (no rows, no evidence)."""
    spec = BucketSpec(fuse_id="family:g", kind="family",
                      members=frozenset({"gamma"}),
                      consecutive_losers=None, session_net_pnl_usd=-0.0)
    stats = compute_bucket_stats(spec, [])
    assert not stats.tripped


def test_trip_event_edge_triggered_and_tripped_at_preserved():
    spec = BucketSpec(fuse_id="family:g", kind="family",
                      members=frozenset({"gamma"}), consecutive_losers=2)
    closes = [_close(pnl=-1), _close(pnl=-2)]
    rows1, events1 = evaluate_buckets([spec], closes, now=NOW)
    assert [e.event for e in events1] == ["trip"]
    t0 = rows1[0]["tripped_at_utc"]

    later = NOW.replace(minute=30)
    rows2, events2 = evaluate_buckets(
        [spec], closes, prior_state={"fuses": rows1}, now=later
    )
    assert events2 == []  # still tripped → no re-emit
    assert rows2[0]["tripped_at_utc"] == t0  # preserved


def test_clear_event_on_session_roll():
    spec = BucketSpec(fuse_id="family:g", kind="family",
                      members=frozenset({"gamma"}), consecutive_losers=2)
    rows1, _ = evaluate_buckets([spec], [_close(pnl=-1), _close(pnl=-2)], now=NOW)
    # Next session: counters re-derive empty.
    rows2, events2 = evaluate_buckets(
        [spec], [], prior_state={"fuses": rows1}, now=NOW
    )
    assert [e.event for e in events2] == ["clear"]
    assert not rows2[0]["tripped"]
    assert "tripped_at_utc" not in rows2[0]


# --------------------------------------------------------------------------- #
# D2 rider — regime legs
# --------------------------------------------------------------------------- #

def test_regime_scoped_leg_never_counts_unknown():
    spec = BucketSpec(fuse_id="family:g@ranging", kind="family",
                      members=frozenset({"gamma"}), consecutive_losers=2,
                      regimes=frozenset({"ranging"}))
    closes = [
        _close(pnl=-1, regime="unknown"),
        _close(pnl=-2, regime="unknown"),
        _close(pnl=-3, regime="ranging"),
    ]
    stats = compute_bucket_stats(spec, closes)
    assert stats.matched == 1  # only the stamped ranging row
    assert stats.consecutive_losers == 1
    assert not stats.tripped


def test_global_leg_counts_unknown_rows():
    spec = BucketSpec(fuse_id="family:g", kind="family",
                      members=frozenset({"gamma"}), consecutive_losers=3)
    closes = [
        _close(pnl=-1, regime="unknown"),
        _close(pnl=-2, regime="unknown"),
        _close(pnl=-3, regime="ranging"),
    ]
    stats = compute_bucket_stats(spec, closes)
    assert stats.matched == 3
    assert stats.tripped


def test_scoped_leg_refuses_unknown_even_if_configured():
    """Structural belt: a spec hand-built with 'unknown' in scope still
    cannot count unstamped rows."""
    spec = BucketSpec(fuse_id="family:g@bad", kind="family",
                      members=frozenset({"gamma"}), consecutive_losers=1,
                      regimes=frozenset({"unknown"}))
    stats = compute_bucket_stats(spec, [_close(pnl=-1, regime="unknown")])
    assert stats.matched == 0
    assert not stats.tripped


def test_sanitize_regime_scope_strips_unknown_and_garbage():
    assert sanitize_regime_scope(None) is None
    assert sanitize_regime_scope([]) is None
    assert sanitize_regime_scope(["ranging", "volatile"]) == frozenset(
        {"ranging", "volatile"}
    )
    assert sanitize_regime_scope(["unknown"]) is None  # all stripped → GLOBAL
    assert sanitize_regime_scope(["unknown", "ranging"]) == frozenset({"ranging"})
    assert sanitize_regime_scope(["sideways_chop"]) is None  # unrecognised


# --------------------------------------------------------------------------- #
# State publisher + heartbeat doctrine + leak guards
# --------------------------------------------------------------------------- #

def _state(modes=None):
    return build_state(
        modes=modes or {"lc2": "off", "lc3": "off", "lc5": "off", "dq": "off"},
        fuse_rows=[],
        counting_tally={},
        trusted_count=0,
        regime_unknown_rows=0,
        session_window_start_utc=NOW,
        epoch_started_at_utc=None,
    )


def test_publish_state_heartbeat_including_off(tmp_path):
    path = tmp_path / "fuse_box_state.json"
    written = publish_state(_state(), path)
    obj = json.loads(path.read_text())
    assert obj["schema_version"] == "fuse_box_state.v1"
    assert obj["ttl_seconds"] == 180
    assert obj["ts_utc"]  # stamped by runtime_json canon
    assert obj["modes"] == {"lc2": "off", "lc3": "off", "lc5": "off", "dq": "off"}
    assert obj["counting"]["trusted_closes"] == 0
    assert obj["lc5"]["factor"] == 1.0
    assert written["ttl_seconds"] == 180


def test_publish_state_default_path_refused_under_pytest():
    with pytest.raises(RuntimeError, match="REQUIRED-explicit"):
        publish_state(_state())


def test_append_evidence_writes_dated_ndjson(tmp_path):
    n = append_evidence(
        [{"event": "trip", "fuse_id": "family:gamma"}],
        evidence_dir=tmp_path / "ev",
        now=NOW,
    )
    assert n == 1
    path = tmp_path / "ev" / f"fuse_box_{DATE}.ndjson"
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert rows[0]["event"] == "trip"
    assert rows[0]["ts_utc"].startswith("2026-07-23")


def test_append_evidence_default_dir_refused_under_pytest():
    with pytest.raises(RuntimeError, match="REQUIRED-explicit"):
        append_evidence([{"event": "trip"}])
