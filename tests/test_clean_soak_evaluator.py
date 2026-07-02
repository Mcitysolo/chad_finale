"""Fixture-only tests for chad.ops.clean_soak_evaluator (GAP-039).

No broker, no network, no live runtime: every test builds its own tmp
runtime/trades tree and asserts against the emitted state dict. The evaluator
under test writes at most ONE new file (clean_soak_state.json) and only when
explicitly asked to.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from chad.ops import clean_soak_evaluator as ev

ANCHOR = "2026-06-30T12:00:00Z"


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_state_file(
    path: Path, obj: Dict[str, Any], *, ttl_seconds: int, ts_utc: Optional[str] = None
) -> None:
    body = dict(obj)
    body["ts_utc"] = ts_utc if ts_utc is not None else _now_iso()
    body["ttl_seconds"] = ttl_seconds
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")


def _write_scr(runtime: Path, *, effective: int, stale: bool = False) -> None:
    ts = "2000-01-01T00:00:00Z" if stale else None
    _write_state_file(
        runtime / ev.SCR_STATE_FILENAME,
        {
            "schema_version": "scr_state.v1",
            "state": "WARMUP",
            "sizing_factor": 0.1,
            "paper_only": True,
            "stats": {
                "effective_trades": effective,
                "total_trades": effective,
                "paper_trades": effective,
                "live_trades": 0,
                "excluded_untrusted": 0,
                "win_rate": 0.0,
                "sharpe_like": 0.0,
                "max_drawdown": 0.0,
                "total_pnl": 0.0,
            },
        },
        ttl_seconds=1 if stale else 600,
        ts_utc=ts,
    )


def _write_epoch(runtime: Path, *, anchor: str = ANCHOR, label: str = "CHAD_v8.9_Paper_Epoch_3") -> None:
    # epoch_reset_state.json intentionally has NO ts_utc / ttl_seconds.
    (runtime).mkdir(parents=True, exist_ok=True)
    (runtime / ev.EPOCH_RESET_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": "epoch_reset_state.v1",
                "target_epoch_label": label,
                "completed_at_utc": anchor,
            }
        ),
        encoding="utf-8",
    )


def _write_positions_truth(runtime: Path, *, status: str = "GREEN", stale: bool = False) -> None:
    ts = "2000-01-01T00:00:00Z" if stale else None
    _write_state_file(
        runtime / ev.POSITIONS_TRUTH_FILENAME,
        {
            "schema_version": "positions_truth.v1",
            "broker_authority_status": status,
            "truth_ok": status == "GREEN",
        },
        ttl_seconds=1 if stale else 90,
        ts_utc=ts,
    )


def _payload(
    *,
    exit_utc: str,
    pnl: float,
    is_live: bool = False,
    untrusted: bool = False,
    entry_utc: Optional[str] = None,
) -> Dict[str, Any]:
    tags: List[str] = []
    extra: Dict[str, Any] = {}
    if untrusted:
        # Mirror the live kraken_paper shape: pnl_untrusted lives in tags + extra,
        # never as a top-level payload key.
        tags.append("pnl_untrusted")
        extra["pnl_untrusted"] = True
    return {
        "entry_time_utc": entry_utc if entry_utc is not None else exit_utc,
        "exit_time_utc": exit_utc,
        "pnl": pnl,
        "is_live": is_live,
        "strategy": "test_strategy",
        "symbol": "AAA",
        "side": "BUY",
        "notional": 100.0,
        "quantity": 1.0,
        "tags": tags,
        "extra": extra,
    }


def _write_trades(trades: Path, ymd: str, payloads: List[Dict[str, Any]]) -> None:
    trades.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, p in enumerate(payloads, start=1):
        lines.append(
            json.dumps(
                {
                    "payload": p,
                    "prev_hash": "GENESIS" if i == 1 else "x",
                    "record_hash": f"hash-{ymd}-{i}",
                    "sequence_id": i,
                    "timestamp_utc": p["exit_time_utc"],
                }
            )
        )
    (trades / f"trade_history_{ymd}.ndjson").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture()
def tree(tmp_path: Path):
    runtime = tmp_path / "runtime"
    trades = tmp_path / "data" / "trades"
    runtime.mkdir(parents=True, exist_ok=True)
    trades.mkdir(parents=True, exist_ok=True)
    return runtime, trades


# --------------------------------------------------------------------------- #
# Scenario (a): clean flat sample — 440 zero-pnl post-anchor => effective 0
# --------------------------------------------------------------------------- #
def test_a_flat_zero_pnl_sample(tree) -> None:
    runtime, trades = tree
    _write_scr(runtime, effective=0)
    _write_epoch(runtime)
    _write_positions_truth(runtime, status="GREEN")
    _write_trades(
        trades,
        "20260701",
        [_payload(exit_utc="2026-07-01T14:00:00Z", pnl=0.0) for _ in range(440)],
    )

    state = ev.evaluate(runtime_dir=runtime, trades_dir=trades)

    assert state["effective_trades"] == 0
    assert state["diagnostics"]["rows_seen"] == 440
    assert state["diagnostics"]["rows_post_anchor"] == 440
    assert state["diagnostics"]["rows_zero_pnl"] == 440
    assert state["diagnostics"]["rows_untrusted"] == 0
    assert state["diagnostics"]["evaluator_effective_tally"] == 0
    assert state["sessions_with_effective"] == 0
    assert state["untrusted_ratio"] == 0.0
    assert state["discrepancy_flag"] is False
    # A flat sample cannot satisfy the trade/session gates; only authority,
    # untrusted_ok and no_discrepancy can pass.
    assert state["gates"]["effective_met"] is False
    assert state["gates"]["sessions_met"] is False
    assert state["gates"]["untrusted_ok"] is True
    assert state["soak_passed"] is False


# --------------------------------------------------------------------------- #
# Scenario (b): 120 trusted effective trades across 6 sessions => 5/5 PASS
# --------------------------------------------------------------------------- #
def test_b_clean_soak_passes(tree) -> None:
    runtime, trades = tree
    _write_scr(runtime, effective=120)
    _write_epoch(runtime)
    _write_positions_truth(runtime, status="GREEN")
    for day in range(1, 7):  # 2026-07-01 .. 2026-07-06 == 6 sessions
        ymd = f"202607{day:02d}"
        exit_utc = f"2026-07-{day:02d}T14:00:00Z"
        _write_trades(
            trades, ymd, [_payload(exit_utc=exit_utc, pnl=10.0) for _ in range(20)]
        )

    state = ev.evaluate(runtime_dir=runtime, trades_dir=trades)

    assert state["effective_trades"] == 120
    assert state["diagnostics"]["evaluator_effective_tally"] == 120
    assert state["sessions_with_effective"] == 6
    assert state["untrusted_ratio"] == 0.0
    assert state["discrepancy_flag"] is False
    assert state["gates"] == {
        "authority_green": True,
        "effective_met": True,
        "sessions_met": True,
        "untrusted_ok": True,
        "no_discrepancy": True,
    }
    assert state["gates_passed"] == 5
    assert state["soak_passed"] is True


# --------------------------------------------------------------------------- #
# Scenario (c): evaluator tally != scr_state.effective => discrepancy_flag True
# --------------------------------------------------------------------------- #
def test_c_discrepancy_flag(tree) -> None:
    runtime, trades = tree
    _write_scr(runtime, effective=999)  # SCR disagrees with the file tally (120)
    _write_epoch(runtime)
    _write_positions_truth(runtime, status="GREEN")
    for day in range(1, 7):
        ymd = f"202607{day:02d}"
        exit_utc = f"2026-07-{day:02d}T14:00:00Z"
        _write_trades(
            trades, ymd, [_payload(exit_utc=exit_utc, pnl=10.0) for _ in range(20)]
        )

    state = ev.evaluate(runtime_dir=runtime, trades_dir=trades)

    assert state["effective_trades"] == 999  # SCR value preserved, never overridden
    assert state["diagnostics"]["evaluator_effective_tally"] == 120
    assert state["discrepancy_flag"] is True
    assert state["gates"]["no_discrepancy"] is False
    assert state["soak_passed"] is False
    assert any("discrepancy" in r for r in state["reasons"])


# --------------------------------------------------------------------------- #
# Scenario (d): missing scr_state => valid file, gates forced false, reason
# --------------------------------------------------------------------------- #
def test_d_missing_scr_state(tree) -> None:
    runtime, trades = tree
    # deliberately NO scr_state.json
    _write_epoch(runtime)
    _write_positions_truth(runtime, status="GREEN")
    _write_trades(trades, "20260701", [_payload(exit_utc="2026-07-01T14:00:00Z", pnl=10.0)])

    state = ev.evaluate(runtime_dir=runtime, trades_dir=trades)

    assert state["schema_version"] == ev.SCHEMA_VERSION  # still a valid state doc
    assert state["gates_passed"] == 0
    assert all(v is False for v in state["gates"].values())
    assert state["soak_passed"] is False
    assert any("scr_state_missing" in r for r in state["reasons"])
    assert any("insufficient_state_data" in r for r in state["reasons"])


def test_d2_stale_scr_state_forces_gates_false(tree) -> None:
    runtime, trades = tree
    _write_scr(runtime, effective=120, stale=True)  # ts far in the past, ttl=1
    _write_epoch(runtime)
    _write_positions_truth(runtime, status="GREEN")

    state = ev.evaluate(runtime_dir=runtime, trades_dir=trades)

    assert state["gates_passed"] == 0
    assert state["soak_passed"] is False
    assert any("scr_state_stale" in r for r in state["reasons"])


# --------------------------------------------------------------------------- #
# Scenario (e): untrusted_ratio breach => untrusted_ok False
# --------------------------------------------------------------------------- #
def test_e_untrusted_ratio_breach(tree) -> None:
    runtime, trades = tree
    _write_scr(runtime, effective=100)
    _write_epoch(runtime)
    _write_positions_truth(runtime, status="GREEN")
    # 100 trusted effective trades across 5 sessions ...
    for day in range(1, 6):  # 2026-07-01 .. 2026-07-05
        ymd = f"202607{day:02d}"
        exit_utc = f"2026-07-{day:02d}T14:00:00Z"
        rows = [_payload(exit_utc=exit_utc, pnl=10.0) for _ in range(20)]
        if day == 1:
            # ... plus 20 untrusted post-anchor rows on day 1 -> ratio 20/120.
            rows += [
                _payload(exit_utc=exit_utc, pnl=5.0, untrusted=True) for _ in range(20)
            ]
        _write_trades(trades, ymd, rows)

    state = ev.evaluate(runtime_dir=runtime, trades_dir=trades)

    assert state["effective_trades"] == 100
    assert state["diagnostics"]["evaluator_effective_tally"] == 100
    assert state["diagnostics"]["rows_post_anchor"] == 120
    assert state["diagnostics"]["rows_untrusted"] == 20
    # output untrusted_ratio is rounded to 6 decimals by design
    assert state["untrusted_ratio"] == pytest.approx(20 / 120, abs=1e-6)
    assert state["gates"]["untrusted_ok"] is False
    assert state["gates"]["effective_met"] is True
    assert state["gates"]["sessions_met"] is True
    assert state["soak_passed"] is False


# --------------------------------------------------------------------------- #
# Authority gate variants
# --------------------------------------------------------------------------- #
def test_authority_red_when_not_green(tree) -> None:
    runtime, trades = tree
    _write_scr(runtime, effective=120)
    _write_epoch(runtime)
    _write_positions_truth(runtime, status="RED")
    for day in range(1, 7):
        _write_trades(
            trades,
            f"202607{day:02d}",
            [_payload(exit_utc=f"2026-07-{day:02d}T14:00:00Z", pnl=10.0) for _ in range(20)],
        )

    state = ev.evaluate(runtime_dir=runtime, trades_dir=trades)
    assert state["authority_status"] == "RED"
    assert state["gates"]["authority_green"] is False
    assert state["soak_passed"] is False


def test_authority_stale_positions_truth_fails_closed(tree) -> None:
    runtime, trades = tree
    _write_scr(runtime, effective=120)
    _write_epoch(runtime)
    _write_positions_truth(runtime, status="GREEN", stale=True)
    for day in range(1, 7):
        _write_trades(
            trades,
            f"202607{day:02d}",
            [_payload(exit_utc=f"2026-07-{day:02d}T14:00:00Z", pnl=10.0) for _ in range(20)],
        )

    state = ev.evaluate(runtime_dir=runtime, trades_dir=trades)
    assert state["authority_status"] == "RED"
    assert state["gates"]["authority_green"] is False


# --------------------------------------------------------------------------- #
# Effective-trade predicate details (is_live / untrusted / zero-pnl / pre-anchor)
# --------------------------------------------------------------------------- #
def test_predicate_excludes_live_and_pre_anchor(tree) -> None:
    runtime, trades = tree
    _write_scr(runtime, effective=1)
    _write_epoch(runtime)
    _write_positions_truth(runtime, status="GREEN")
    _write_trades(
        trades,
        "20260701",
        [
            _payload(exit_utc="2026-07-01T14:00:00Z", pnl=10.0),                 # effective
            _payload(exit_utc="2026-07-01T15:00:00Z", pnl=10.0, is_live=True),   # live -> excluded
            _payload(exit_utc="2026-06-30T11:00:00Z", pnl=10.0),                 # pre-anchor -> excluded
            _payload(exit_utc="2026-07-01T16:00:00Z", pnl=0.0),                  # zero pnl -> excluded
            _payload(exit_utc="2026-07-01T17:00:00Z", pnl=10.0, untrusted=True), # untrusted -> excluded
        ],
    )

    state = ev.evaluate(runtime_dir=runtime, trades_dir=trades)
    assert state["diagnostics"]["rows_seen"] == 5
    assert state["diagnostics"]["rows_post_anchor"] == 4  # the pre-anchor row drops out
    assert state["diagnostics"]["evaluator_effective_tally"] == 1
    assert state["diagnostics"]["rows_untrusted"] == 1
    assert state["diagnostics"]["rows_zero_pnl"] == 1
    assert state["sessions_with_effective"] == 1
    assert state["discrepancy_flag"] is False  # scr=1 matches tally=1


# --------------------------------------------------------------------------- #
# Env-driven criteria overrides
# --------------------------------------------------------------------------- #
def test_env_override_raises_effective_threshold(tree) -> None:
    runtime, trades = tree
    _write_scr(runtime, effective=120)
    _write_epoch(runtime)
    _write_positions_truth(runtime, status="GREEN")
    for day in range(1, 7):
        _write_trades(
            trades,
            f"202607{day:02d}",
            [_payload(exit_utc=f"2026-07-{day:02d}T14:00:00Z", pnl=10.0) for _ in range(20)],
        )

    env = {"CHAD_SOAK_MIN_EFFECTIVE_TRADES": "200"}
    state = ev.evaluate(runtime_dir=runtime, trades_dir=trades, env=env)
    assert state["criteria"]["min_effective_trades"] == 200
    assert state["gates"]["effective_met"] is False  # 120 < 200
    assert state["soak_passed"] is False


def test_load_criteria_defaults_and_invalid_fallback() -> None:
    default = ev.load_criteria(env={})
    assert default.min_effective_trades == 100
    assert default.min_sessions == 5
    assert default.max_untrusted_ratio == 0.05

    overridden = ev.load_criteria(
        env={
            "CHAD_SOAK_MIN_EFFECTIVE_TRADES": "42",
            "CHAD_SOAK_MIN_SESSIONS": "3",
            "CHAD_SOAK_MAX_UNTRUSTED_RATIO": "0.2",
        }
    )
    assert overridden == ev.Criteria(42, 3, 0.2)

    # Garbage values fall back to defaults rather than raising.
    fallback = ev.load_criteria(
        env={
            "CHAD_SOAK_MIN_EFFECTIVE_TRADES": "not-an-int",
            "CHAD_SOAK_MAX_UNTRUSTED_RATIO": "xyz",
        }
    )
    assert fallback.min_effective_trades == 100
    assert fallback.max_untrusted_ratio == 0.05


# --------------------------------------------------------------------------- #
# Pure-helper unit tests
# --------------------------------------------------------------------------- #
def test_parse_iso_utc_variants() -> None:
    assert ev.parse_iso_utc("2026-06-30T12:17:42Z") == datetime(
        2026, 6, 30, 12, 17, 42, tzinfo=timezone.utc
    )
    off = ev.parse_iso_utc("2026-07-02T02:46:43.171578+00:00")
    assert off is not None and off.tzinfo == timezone.utc
    # naive -> assumed UTC
    naive = ev.parse_iso_utc("2026-07-02T02:46:43")
    assert naive is not None and naive.tzinfo == timezone.utc
    assert ev.parse_iso_utc("") is None
    assert ev.parse_iso_utc(None) is None
    assert ev.parse_iso_utc("garbage") is None
    assert ev.parse_iso_utc(12345) is None


def test_row_is_untrusted_all_shapes() -> None:
    assert ev.row_is_untrusted({"tags": ["pnl_untrusted"]}) is True
    assert ev.row_is_untrusted({"tags": ["PNL_UNTRUSTED"]}) is True
    assert ev.row_is_untrusted({"extra": {"pnl_untrusted": True}}) is True
    assert ev.row_is_untrusted({"pnl_untrusted": True}) is True  # documented top-level
    assert ev.row_is_untrusted({"tags": ["buy"], "extra": {}}) is False
    assert ev.row_is_untrusted({}) is False


def test_iter_soak_trade_files_filters(tmp_path: Path) -> None:
    trades = tmp_path / "data" / "trades"
    trades.mkdir(parents=True, exist_ok=True)
    (trades / "trade_history_20260630.ndjson").write_text("", encoding="utf-8")   # on anchor
    (trades / "trade_history_20260701.ndjson").write_text("", encoding="utf-8")   # after
    (trades / "trade_history_20260629.ndjson").write_text("", encoding="utf-8")   # before -> out
    (trades / "trade_history_20260702.ndjson.scr_reset_bak").write_text("", encoding="utf-8")  # bak -> out
    (trades / "trade_history_enriched_20260701.ndjson").write_text("", encoding="utf-8")  # aux -> out

    from datetime import date

    found = {p.name for p in ev.iter_soak_trade_files(trades, date(2026, 6, 30))}
    assert found == {"trade_history_20260630.ndjson", "trade_history_20260701.ndjson"}


# --------------------------------------------------------------------------- #
# evaluate() writes nothing; write_state / CLI behaviour
# --------------------------------------------------------------------------- #
def test_evaluate_writes_nothing(tree) -> None:
    runtime, trades = tree
    _write_scr(runtime, effective=0)
    _write_epoch(runtime)
    _write_positions_truth(runtime, status="GREEN")
    ev.evaluate(runtime_dir=runtime, trades_dir=trades)
    assert not (runtime / ev.OUTPUT_FILENAME).exists()


def test_write_state_atomic_roundtrip(tree, tmp_path: Path) -> None:
    runtime, trades = tree
    _write_scr(runtime, effective=0)
    _write_epoch(runtime)
    _write_positions_truth(runtime, status="GREEN")
    state = ev.evaluate(runtime_dir=runtime, trades_dir=trades)

    out = tmp_path / "out" / ev.OUTPUT_FILENAME
    ev.write_state(state, out)
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "clean_soak_state.v1"
    assert loaded["ttl_seconds"] == 300
    assert "ts_utc" in loaded
    assert set(loaded["gates"].keys()) == set(ev.GATE_KEYS)
    assert not (out.parent / (ev.OUTPUT_FILENAME + ".tmp")).exists()  # tmp cleaned up


def test_main_print_does_not_write(tree, capsys) -> None:
    runtime, trades = tree
    _write_scr(runtime, effective=0)
    _write_epoch(runtime)
    _write_positions_truth(runtime, status="GREEN")
    out = runtime / ev.OUTPUT_FILENAME

    rc = ev.main(["--print"], runtime_dir=runtime, trades_dir=trades, out_path=out)
    assert rc == 0
    assert not out.exists()
    captured = capsys.readouterr().out
    assert "soak_passed" in captured
    assert "gates" in captured


def test_main_json_does_not_write(tree, capsys) -> None:
    runtime, trades = tree
    _write_scr(runtime, effective=0)
    _write_epoch(runtime)
    _write_positions_truth(runtime, status="GREEN")
    out = runtime / ev.OUTPUT_FILENAME

    rc = ev.main(["--json"], runtime_dir=runtime, trades_dir=trades, out_path=out)
    assert rc == 0
    assert not out.exists()
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["schema_version"] == "clean_soak_state.v1"


def test_main_default_writes_file(tree, capsys) -> None:
    runtime, trades = tree
    _write_scr(runtime, effective=0)
    _write_epoch(runtime)
    _write_positions_truth(runtime, status="GREEN")
    out = runtime / ev.OUTPUT_FILENAME

    rc = ev.main([], runtime_dir=runtime, trades_dir=trades, out_path=out)
    assert rc == 0
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "clean_soak_state.v1"
    # exactly one summary line on the write path
    summary = capsys.readouterr().out.strip().splitlines()
    assert len(summary) == 1
    assert "clean_soak" in summary[0]
