#!/usr/bin/env python3
"""
Tests for the soak evidence writers (PA SOAK_STATUS_HISTORY_WRITER_2026-06-20).

Covers, per PA §7:
  - PRIMARY schema + deterministic sizing_digest (criterion 4)
  - PRIMARY partial-read resilience (read_errors[], no exception)
  - PRIMARY append-only (two invocations -> two lines, line 1 byte-identical)
  - PRIMARY isolation (CRITICAL): a forced internal write failure does NOT
    propagate out of lifecycle_truth_publisher.publish_once(), and the authority
    artifacts (positions_truth.json + trade_lifecycle_state.json) are still
    written.
  - Companion #2 route() emits one row per RoutedSignal; empty route emits 0;
    isolation (writer raise does not perturb route()).
  - Companion #3 ENTRY/EXIT/FLIP classification + audit rows; isolation;
    placement downstream of same-side suppression.
  - Cross-condition replay / criterion-8 de-dup (stub classifier): consecutive
    rows sharing a reconciliation_ts_utc collapse to one logical cycle, and the
    "next cycle after the window" resolves to the first distinct
    reconciliation_ts_utc > window_close_utc.

All writes go to pytest tmp_path; no live runtime/ or data/ path is touched.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from chad.ops.soak import evidence_writers as ew


@pytest.fixture(autouse=True)
def _enable_soak_writers(monkeypatch):
    """All write tests run with the activation gate ON. Default is OFF (see
    test_writers_disabled_by_default and the module docstring)."""
    monkeypatch.setenv("CHAD_SOAK_EVIDENCE_WRITERS", "1")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _raise(*_a, **_k):
    raise RuntimeError("forced soak write failure")


def _read_rows(path: Path):
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _write_runtime_sources(
    runtime_dir: Path,
    *,
    broker_status: str = "GREEN",
    caps=None,
    recon_status: str = "GREEN",
    recon_ts: str = "2026-06-20T13:30:00Z",
):
    runtime_dir.mkdir(parents=True, exist_ok=True)
    caps = caps if caps is not None else {"alpha": 1000.0, "beta": 2000.0}
    files = {
        "positions_truth.json": {
            "broker_authority_status": broker_status,
            "as_of_event_ts_utc": "2026-06-20T13:30:00.333713Z",
            "ts_utc": "2026-06-20T13:30:36.807927Z",
            "ttl_seconds": 90,
        },
        "reconciliation_state.json": {"status": recon_status, "ts_utc": recon_ts, "ttl_seconds": 360},
        "scr_state.json": {"state": "CAUTIOUS", "ts_utc": "2026-06-20T13:29:30Z"},
        "dynamic_caps.json": {"strategy_caps": caps, "ts_utc": "2026-06-20T13:29:58.740740Z"},
        "profit_lock_state.json": {"mode": "NORMAL", "sizing_factor": 1.0, "ts_utc": "2026-06-20T13:34:46Z"},
        "stop_bus.json": {
            "active": False,
            "triggered_at": None,
            "cleared_at": "2026-05-28T18:38:41.794703+00:00",
        },
        "tier_state.json": {"tier_name": "PRO_GROWTH", "ts_utc": "2026-06-20T04:06:38.290073Z"},
    }
    for name, payload in files.items():
        (runtime_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    return files


# ===========================================================================
# Activation gate (default OFF) — production inertness
# ===========================================================================
def test_writers_disabled_by_default(tmp_path, monkeypatch):
    """With the env flag unset, every writer is a no-op: no dir, no file, no row.

    This is what keeps the in-tree wiring inert in the running services until the
    operator explicitly activates the soak clock (PA §6)."""
    monkeypatch.delenv("CHAD_SOAK_EVIDENCE_WRITERS", raising=False)
    assert ew.writers_enabled() is False

    rt, dd = tmp_path / "runtime", tmp_path / "data"
    _write_runtime_sources(rt)
    ew.emit_status_history(repo_root=tmp_path, runtime_dir=rt, data_dir=dd)
    ew.emit_signal_router_emissions([object()], data_dir=dd)
    ew.emit_entry_intent_audit(
        intent_type="ENTRY", symbol="AAPL", side="BUY", strategy="alpha",
        admitted=True, data_dir=dd,
    )
    assert not (dd / "soak").exists()


# ===========================================================================
# PRIMARY — status-history writer
# ===========================================================================
def test_status_history_schema_fields(tmp_path):
    rt, dd = tmp_path / "runtime", tmp_path / "data"
    _write_runtime_sources(rt)
    ew.emit_status_history(
        repo_root=tmp_path, runtime_dir=rt, data_dir=dd, ts_utc="2026-06-20T13:30:36Z"
    )
    out = dd / "soak" / "status_history_20260620.ndjson"
    rows = _read_rows(out)
    assert len(rows) == 1
    row = rows[0]
    assert row["schema_version"] == "soak_status_history.v1"
    assert row["ts_utc"] == "2026-06-20T13:30:36Z"
    assert row["writer_identity"] == "chad.ops.soak.status_history_writer"
    assert row["cadence_source"] == "lifecycle_truth_publisher"
    assert row["broker_authority_status"] == "GREEN"
    assert row["broker_authority_as_of_utc"] == "2026-06-20T13:30:00.333713Z"
    assert row["broker_authority_publish_ts_utc"] == "2026-06-20T13:30:36.807927Z"
    assert isinstance(row["broker_authority_age_seconds"], int)
    assert row["reconciliation_status"] == "GREEN"
    assert row["reconciliation_ts_utc"] == "2026-06-20T13:30:00Z"
    assert row["scr_band"] == "CAUTIOUS"
    assert row["scr_ts_utc"] == "2026-06-20T13:29:30Z"
    assert row["sizing_digest"].startswith("sha256:")
    assert row["dynamic_caps_ts_utc"] == "2026-06-20T13:29:58.740740Z"
    assert row["profit_lock_mode"] == "NORMAL"
    assert row["profit_lock_sizing_factor"] == 1.0
    assert row["profit_lock_ts_utc"] == "2026-06-20T13:34:46Z"
    assert row["stop_bus_active"] is False
    # stop_bus.json has NO ts_utc; triggered_at None -> "" proxy (PA §5.4)
    assert row["stop_bus_triggered_at"] == ""
    assert row["stop_bus_cleared_at"] == "2026-05-28T18:38:41.794703+00:00"
    assert row["tier_name"] == "PRO_GROWTH"
    assert row["tier_ts_utc"] == "2026-06-20T04:06:38.290073Z"
    assert set(row["source_sha256"].keys()) == set(ew._PRIMARY_SOURCE_FILES)
    assert all(v.startswith("sha256:") for v in row["source_sha256"].values())
    assert row["read_errors"] == []


def test_sizing_digest_is_canonical_and_change_sensitive(tmp_path):
    rt1, rt2, rt3 = tmp_path / "r1", tmp_path / "r2", tmp_path / "r3"
    _write_runtime_sources(rt1, caps={"alpha": 1000.0, "beta": 2000.0})
    # reordered keys -> same canonical (sorted-key) digest
    _write_runtime_sources(rt2, caps={"beta": 2000.0, "alpha": 1000.0})
    # one cap changed -> different digest
    _write_runtime_sources(rt3, caps={"alpha": 1000.0, "beta": 2500.0})
    d1 = ew.build_status_history_row(runtime_dir=rt1, ts_utc="t")["sizing_digest"]
    d2 = ew.build_status_history_row(runtime_dir=rt2, ts_utc="t")["sizing_digest"]
    d3 = ew.build_status_history_row(runtime_dir=rt3, ts_utc="t")["sizing_digest"]
    assert d1 == d2
    assert d1 != d3


def test_status_history_partial_read_resilience(tmp_path):
    rt, dd = tmp_path / "runtime", tmp_path / "data"
    _write_runtime_sources(rt)
    (rt / "scr_state.json").write_text("{ not valid json", encoding="utf-8")  # corrupt
    (rt / "tier_state.json").unlink()  # missing
    # must not raise
    ew.emit_status_history(
        repo_root=tmp_path, runtime_dir=rt, data_dir=dd, ts_utc="2026-06-20T13:30:36Z"
    )
    rows = _read_rows(dd / "soak" / "status_history_20260620.ndjson")
    assert len(rows) == 1
    row = rows[0]
    assert "scr_state.json" in row["read_errors"]
    assert "tier_state.json" in row["read_errors"]
    assert row["scr_band"] is None
    assert row["tier_name"] is None
    assert row["source_sha256"]["scr_state.json"] == ""
    assert row["source_sha256"]["tier_state.json"] == ""
    # the readable sources are still populated
    assert row["broker_authority_status"] == "GREEN"
    assert row["source_sha256"]["positions_truth.json"].startswith("sha256:")


def test_status_history_append_only(tmp_path):
    rt, dd = tmp_path / "runtime", tmp_path / "data"
    _write_runtime_sources(rt)
    out = dd / "soak" / "status_history_20260620.ndjson"
    ew.emit_status_history(
        repo_root=tmp_path, runtime_dir=rt, data_dir=dd, ts_utc="2026-06-20T13:30:36Z"
    )
    first_bytes = out.read_bytes()
    ew.emit_status_history(
        repo_root=tmp_path, runtime_dir=rt, data_dir=dd, ts_utc="2026-06-20T13:31:36Z"
    )
    after_bytes = out.read_bytes()
    rows = _read_rows(out)
    assert len(rows) == 2
    # line 1 byte-identical after the second append
    assert after_bytes.startswith(first_bytes)


def test_publish_once_emits_soak_row_positive_control(tmp_path):
    """Wiring is live: publish_once produces a soak row from real authority writes."""
    from chad.ops import lifecycle_truth_publisher as ltp

    rt, dd = tmp_path / "runtime", tmp_path / "data"
    ltp.publish_once(repo_root=tmp_path, runtime_dir=rt, data_dir=dd)
    files = list((dd / "soak").glob("status_history_*.ndjson"))
    assert len(files) == 1
    rows = _read_rows(files[0])
    assert len(rows) == 1
    row = rows[0]
    assert row["schema_version"] == "soak_status_history.v1"
    # the row read the just-written positions_truth.json
    assert row["broker_authority_status"] in {"GREEN", "RED"}
    assert row["source_sha256"]["positions_truth.json"].startswith("sha256:")


def test_status_history_isolation_publish_once(tmp_path, monkeypatch):
    """CRITICAL isolation proof: a forced writer failure must NOT break publish_once.

    publish_once must still return normally and both authority artifacts
    (positions_truth.json + trade_lifecycle_state.json) must be written.
    """
    from chad.ops import lifecycle_truth_publisher as ltp

    rt, dd = tmp_path / "runtime", tmp_path / "data"
    # Force the writer's sole disk-write primitive to raise.
    monkeypatch.setattr(ew, "_append_row", _raise)

    result = ltp.publish_once(repo_root=tmp_path, runtime_dir=rt, data_dir=dd)

    # host completed normally, return value unchanged (None)
    assert result is None
    # the authority artifacts produced at :719-720 are present + valid
    pt_path = rt / "positions_truth.json"
    lc_path = rt / "trade_lifecycle_state.json"
    assert pt_path.is_file()
    assert lc_path.is_file()
    pt = json.loads(pt_path.read_text(encoding="utf-8"))
    assert "broker_authority_status" in pt
    json.loads(lc_path.read_text(encoding="utf-8"))  # parses
    # the writer was actually reached (it created data/soak/) but produced no row
    assert (dd / "soak").is_dir()
    assert _read_rows_safe(dd / "soak", "status_history_*.ndjson") == []


def _read_rows_safe(soak_dir: Path, pattern: str):
    rows = []
    for f in soak_dir.glob(pattern):
        rows += _read_rows(f)
    return rows


# ===========================================================================
# COMPANION #2 — signal-router emission log
# ===========================================================================
def _trade_signal(symbol, side, size=100.0):
    from chad.types import AssetClass, SignalSide, StrategyName, TradeSignal

    return TradeSignal(
        strategy=StrategyName.ALPHA,
        symbol=symbol,
        side=getattr(SignalSide, side),
        size=size,
        confidence=0.8,
        asset_class=AssetClass.EQUITY,
    )


def test_signal_router_row_schema():
    from chad.types import AssetClass, RoutedSignal, SignalSide, StrategyName

    rs = RoutedSignal(
        symbol="AAPL",
        side=SignalSide.BUY,
        net_size=100.0,
        source_strategies=(StrategyName.ALPHA, StrategyName.BETA),
        confidence=0.8,
        asset_class=AssetClass.EQUITY,
        primary_strategy="alpha",
    )
    row = ew.build_signal_router_row(rs, "2026-06-20T13:30:36Z")
    assert row["schema_version"] == "soak_signal_router.v1"
    assert row["ts_utc"] == "2026-06-20T13:30:36Z"
    assert row["symbol"] == "AAPL"
    assert row["side"] == "BUY"
    assert row["primary_strategy"] == "alpha"
    assert row["source_strategies"] == ["alpha", "beta"]
    assert row["net_size"] == 100.0


def test_route_emits_one_row_per_routed_signal(tmp_path, monkeypatch):
    from chad.utils.signal_router import SignalRouter

    monkeypatch.setenv("CHAD_DATA_DIR", str(tmp_path))
    sigs = [
        _trade_signal("AAPL", "BUY"),
        _trade_signal("MSFT", "BUY", 50.0),
        _trade_signal("TSLA", "SELL", 20.0),
    ]
    routed = SignalRouter().route(sigs)
    assert len(routed) == 3
    files = list((tmp_path / "soak").glob("signal_router_emissions_*.ndjson"))
    assert len(files) == 1
    rows = _read_rows(files[0])
    assert len(rows) == 3
    assert all(r["schema_version"] == "soak_signal_router.v1" for r in rows)
    assert {r["symbol"] for r in rows} == {"AAPL", "MSFT", "TSLA"}


def test_route_empty_emits_zero_rows(tmp_path, monkeypatch):
    from chad.utils.signal_router import SignalRouter

    monkeypatch.setenv("CHAD_DATA_DIR", str(tmp_path))
    routed = SignalRouter().route([])
    assert routed == []
    # no emission rows for a zero-signal route (early return before any write)
    assert _read_rows_safe(tmp_path / "soak", "signal_router_emissions_*.ndjson") == []


def test_route_isolation_writer_raise_does_not_perturb(tmp_path, monkeypatch):
    from chad.utils.signal_router import SignalRouter

    monkeypatch.setenv("CHAD_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(ew, "_append_row", _raise)
    routed = SignalRouter().route([_trade_signal("AAPL", "BUY")])
    # routing output unaffected
    assert len(routed) == 1
    assert routed[0].symbol == "AAPL"
    assert routed[0].side.value == "BUY"


# ===========================================================================
# COMPANION #3 — ENTRY-intent audit log
# ===========================================================================
@pytest.mark.parametrize(
    "is_exit,is_flip,side,expected",
    [
        (False, False, "BUY", "ENTRY"),
        (False, False, "SELL", "ENTRY"),
        (True, False, "BUY", "EXIT"),
        (False, False, "EXIT", "EXIT"),
        (False, False, "CLOSE", "EXIT"),
        (False, True, "BUY", "FLIP"),
        (True, True, "BUY", "EXIT"),  # exit takes precedence over flip
    ],
)
def test_classify_intent_type(is_exit, is_flip, side, expected):
    assert ew.classify_intent_type(is_exit=is_exit, is_flip=is_flip, side=side) == expected


def test_entry_intent_audit_rows(tmp_path):
    for it in ("ENTRY", "EXIT", "FLIP"):
        ew.emit_entry_intent_audit(
            intent_type=it,
            symbol="AAPL",
            side="BUY",
            strategy="alpha",
            admitted=True,
            data_dir=tmp_path,
            ts_utc="2026-06-20T13:30:36Z",
        )
    rows = _read_rows(tmp_path / "soak" / "entry_intent_audit_20260620.ndjson")
    assert [r["intent_type"] for r in rows] == ["ENTRY", "EXIT", "FLIP"]
    assert all(r["schema_version"] == "soak_entry_intent.v1" for r in rows)
    assert all(r["admitted"] is True for r in rows)
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["side"] == "BUY"
    assert rows[0]["strategy"] == "alpha"


def test_entry_intent_audit_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(ew, "_append_row", _raise)
    # must not raise
    ew.emit_entry_intent_audit(
        intent_type="ENTRY",
        symbol="AAPL",
        side="BUY",
        strategy="alpha",
        admitted=True,
        data_dir=tmp_path,
    )
    assert _read_rows_safe(tmp_path / "soak", "entry_intent_audit_*.ndjson") == []


def test_entry_intent_attach_point_is_downstream_of_same_side_suppression():
    """A same-side-suppressed intent yields no ENTRY row because the audit emit is
    placed AFTER the same-side `continue` and before the broker submit (PA §4.3).

    Structural guard: assert the source ordering in live_loop.py."""
    src = Path(__file__).resolve().parents[1] / "core" / "live_loop.py"
    text = src.read_text(encoding="utf-8")
    i_same_side = text.index("if is_same_side_open(intent):")
    i_emit = text.index("_soak_emit_entry_intent(")
    i_submit = text.index("submit_strategy_trade_intents([intent])")
    assert i_same_side < i_emit < i_submit


# ===========================================================================
# Cross-condition replay — criterion-8 de-dup (stub classifier, separate-PA-shape)
# ===========================================================================
def _status_row(
    *,
    ts_utc,
    broker,
    recon_ts,
    recon_status,
    sizing_digest="sha256:STABLE",
    scr_band="CAUTIOUS",
    profit_lock_ts="2026-06-20T10:00:00Z",
    tier_ts="2026-06-20T04:00:00Z",
    stop_bus_active=False,
):
    return {
        "schema_version": "soak_status_history.v1",
        "ts_utc": ts_utc,
        "broker_authority_status": broker,
        "reconciliation_status": recon_status,
        "reconciliation_ts_utc": recon_ts,
        "sizing_digest": sizing_digest,
        "scr_band": scr_band,
        "profit_lock_ts_utc": profit_lock_ts,
        "tier_ts_utc": tier_ts,
        "stop_bus_active": stop_bus_active,
        "stop_bus_triggered_at": "",
        "stop_bus_cleared_at": "2026-05-28T18:38:41Z",
    }


def _synth_history():
    """60s-cadence history with a 12-min RED window and 300s reconciliation snapshots.

    Reconciliation cycles (de-dup keys): 13:00, 13:05, 13:10, 13:15.
    Broker RED 13:01..13:12 -> window_open=13:01:00, window_close=13:13:00.
    Next distinct reconciliation_ts_utc > window_close is 13:15:00, status GREEN.
    """
    rows = []

    def mm(minute):
        return f"2026-06-20T13:{minute:02d}:00Z"

    # 20 rows: minutes 13:00..13:19 -> reconciliation buckets 13:00/05/10/15
    for minute in range(0, 20):
        if 1 <= minute <= 12:
            broker = "RED"
        else:
            broker = "GREEN"
        bucket = (minute // 5) * 5
        # contemporaneous reconciliation_status: GREEN except during the unsettled buckets
        recon_status = "GREEN" if bucket in (0, 15) else "YELLOW"
        rows.append(
            _status_row(
                ts_utc=mm(minute),
                broker=broker,
                recon_ts=mm(bucket),
                recon_status=recon_status,
            )
        )
    return rows


def _stub_grade(rows):
    """Stub RED-window classifier (the real one is a separate later PA).

    Implements only enough of the PA §2 grading procedure to exercise the
    writer's output contract — especially criterion-8 de-dup."""
    open_i = next((i for i, r in enumerate(rows) if r["broker_authority_status"] == "RED"), None)
    assert open_i is not None
    close_i = next(
        (i for i in range(open_i, len(rows)) if rows[i]["broker_authority_status"] != "RED"),
        None,
    )
    assert close_i is not None
    window_open = rows[open_i]["ts_utc"]
    window_close = rows[close_i]["ts_utc"]
    in_window = rows[open_i:close_i]  # the RED rows

    dur = (ew._parse_ts(window_close) - ew._parse_ts(window_open)).total_seconds()
    c1 = dur <= 900
    c4 = len({r["sizing_digest"] for r in in_window}) == 1
    c5 = len({r["scr_band"] for r in in_window}) == 1
    c6 = (
        len({r["profit_lock_ts_utc"] for r in in_window}) == 1
        and all(r["stop_bus_active"] is False for r in in_window)
        and len({r["stop_bus_triggered_at"] for r in in_window}) == 1
        and len({r["stop_bus_cleared_at"] for r in in_window}) == 1
        and len({r["tier_ts_utc"] for r in in_window}) == 1
    )

    # criterion 8 — de-dup by reconciliation_ts_utc into distinct logical cycles
    distinct = []
    seen = set()
    for r in rows:
        rt = r["reconciliation_ts_utc"]
        if rt not in seen:
            seen.add(rt)
            distinct.append((rt, r))  # first row carrying this reconciliation cycle
    distinct.sort(key=lambda d: ew._parse_ts(d[0]))
    wclose = ew._parse_ts(window_close)
    nxt = next((d for d in distinct if ew._parse_ts(d[0]) > wclose), None)
    c8 = (
        nxt is not None
        and nxt[1]["reconciliation_status"] == "GREEN"
        and nxt[1]["broker_authority_status"] == "GREEN"
    )
    return {
        "window_open": window_open,
        "window_close": window_close,
        "distinct_recon_count": len(distinct),
        "next_recon_ts": nxt[0] if nxt else None,
        "1": c1,
        "4": c4,
        "5": c5,
        "6": c6,
        "8": c8,
    }


def test_criterion8_dedup_and_full_grade_pass():
    rows = _synth_history()
    g = _stub_grade(rows)
    assert g["window_open"] == "2026-06-20T13:01:00Z"
    assert g["window_close"] == "2026-06-20T13:13:00Z"
    # 20 rows (13:00..13:19) collapse to 4 distinct reconciliation cycles
    assert g["distinct_recon_count"] == 4
    # "next cycle after the window" = first distinct reconciliation_ts_utc > window_close
    assert g["next_recon_ts"] == "2026-06-20T13:15:00Z"
    assert g["1"] is True
    assert g["4"] is True
    assert g["5"] is True
    assert g["6"] is True
    assert g["8"] is True


def test_criterion_flips_to_fail_when_a_field_changes():
    # crit 1: stretch the RED window past 900s (open 13:01, close now 13:17 = 960s)
    rows = _synth_history()
    for r in rows:
        m = int(r["ts_utc"][14:16])
        if 13 <= m <= 16:
            r["broker_authority_status"] = "RED"
    assert _stub_grade(rows)["1"] is False

    # crit 4: change one in-window sizing_digest
    rows = _synth_history()
    rows[5]["sizing_digest"] = "sha256:CHANGED"
    assert _stub_grade(rows)["4"] is False

    # crit 5: change one in-window scr_band
    rows = _synth_history()
    rows[6]["scr_band"] = "CONFIDENT"
    assert _stub_grade(rows)["5"] is False

    # crit 6: change one in-window profit_lock_ts_utc
    rows = _synth_history()
    rows[7]["profit_lock_ts_utc"] = "2026-06-20T13:07:00Z"
    assert _stub_grade(rows)["6"] is False

    # crit 8: next reconciliation cycle after the window is not GREEN
    rows = _synth_history()
    for r in rows:
        if r["reconciliation_ts_utc"] == "2026-06-20T13:15:00Z":
            r["reconciliation_status"] = "RED"
    assert _stub_grade(rows)["8"] is False
