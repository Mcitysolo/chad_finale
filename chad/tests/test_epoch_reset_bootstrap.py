"""Tests for ops/epoch_reset_bootstrap.py.

All tests run against tmp fixtures only — the live runtime/ tree is never
touched. The tool module is loaded by file path so no ops/ package layout is
assumed.

Coverage:
  * dry-run (default)            -> prints plan, ZERO mutations
  * apply (+confirm)            -> archives (SHA256SUMS+manifest), bumps epoch,
                                   fences day-files, clears DBs/equity/queues/
                                   guard/HWM-carriers/second-family, writes marker
  * idempotent re-run           -> no-op (no new archive, no re-truncate)
  * refusal: ready_for_live=true
  * refusal: exec_mode=live
  * --apply without token       -> stays dry-run
  * protected-path guard        -> rejects chad/, configs, .claude/, audits/
  * before-today vs all cutoff  -> same-day file handling
"""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL_PATH = _REPO_ROOT / "ops" / "epoch_reset_bootstrap.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("epoch_reset_bootstrap", _TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = mod  # required so dataclasses can resolve annotations
    spec.loader.exec_module(mod)
    return mod


ERB = _load_tool()


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #

CURRENT_EPOCH = "CHAD_v8.9_Paper_Epoch_2"
EXPECTED_TARGET = "CHAD_v8.9_Paper_Epoch_3"


def _make_sqlite(path: Path, table: str) -> None:
    con = sqlite3.connect(str(path))
    try:
        con.execute(f"CREATE TABLE IF NOT EXISTS {table} (k TEXT PRIMARY KEY, v TEXT)")
        con.execute(f"INSERT OR REPLACE INTO {table} (k, v) VALUES ('a','1')")
        con.commit()
    finally:
        con.close()
    # simulate quiescent WAL/SHM sidecars present on disk
    path.with_name(path.name + "-wal").write_bytes(b"")
    path.with_name(path.name + "-shm").write_bytes(b"\x00" * 16)


def make_dirty(tmp_path: Path):
    runtime = tmp_path / "runtime"
    data_trades = tmp_path / "data" / "trades"
    runtime.mkdir(parents=True)
    data_trades.mkdir(parents=True)

    (runtime / "epoch_state.json").write_text(json.dumps({
        "schema_version": "epoch_state.v1",
        "active_epoch": CURRENT_EPOCH,
        "epoch_started_at_utc": "2026-05-04T00:54:30Z",
        "paper_only": True,
        "ready_for_live": False,
        "previous_epoch_archive": "/old/epoch_1_pre",
        "quarantine_manifest": "/old/quarantine.json",
    }, indent=2))

    (runtime / "live_readiness.json").write_text(json.dumps({
        "ready_for_live": False,
        "schema_version": "live_readiness_state.v1",
        "ts_utc": "2026-06-21T18:50:22Z",
        "ttl_seconds": 604800,
    }))

    # contaminated equity history incl the phantom $313k row
    (runtime / "equity_history.ndjson").write_text(
        json.dumps({"date_utc": "2026-06-02", "total_equity_usd": 240331.0,
                    "schema_version": "equity_history.v1"}) + "\n" +
        json.dumps({"date_utc": "2026-06-03", "total_equity_usd": 313124.35261756467,
                    "schema_version": "equity_history.v1"}) + "\n" +
        json.dumps({"date_utc": "2026-06-11", "total_equity_cad": 219865.0,
                    "schema_version": "equity_history.v2"}) + "\n"
    )

    _make_sqlite(runtime / "ibkr_adapter_state.sqlite3", "ibkr_exec_state")
    _make_sqlite(runtime / "exec_state_paper.sqlite3", "paper_runs")

    (runtime / "trade_closer_state.json").write_text(json.dumps({
        "processed_fill_ids": ["fid-1", "fid-2", "fid-3"],
        "queues": [
            {"strategy": "omega_macro", "symbol": "MES",
             "lots": [{"side": "BUY", "quantity": 1, "fill_price": 5000.0,
                       "lot_ts_utc": "2026-06-04T00:00:00Z", "fill_id": "fid-9"}]},
        ],
        "saved_at_utc": "2026-06-18T12:00:00Z",
    }, indent=2))

    (runtime / "position_guard.json").write_text(json.dumps({
        "_version": 123, "_written_by": "test",
        "omega_macro|MES": {"open": True, "strategy": "omega_macro",
                            "symbol": "MES", "side": "BUY", "quantity": 1},
        "broker_sync|MES": {"open": True, "strategy": "broker_sync",
                            "symbol": "MES", "side": "BUY", "quantity": 1},
    }, indent=2))

    (runtime / "drawdown_state.json").write_text(json.dumps({"hwm_cad": 313124.35}))
    (runtime / "withdrawal_authorization.json").write_text(
        json.dumps({"high_water_mark_usd": 141124.33}))
    (runtime / "business_phase.json").write_text(
        json.dumps({"high_water_mark_usd": 141124.33}))

    for name, body in {
        "expectancy_state.json": {"strategies": {"x": 1}},
        "winner_scaling.json": {"scale": {"x": 1.2}},
        "strategy_health.json": {"health": {"x": "GREEN"}},
        "strategy_allocations.json": {"omega_macro": {"halted": True,
                                                      "consecutive_negative": 10}},
        "setup_family_expectancy.json": {"fam": {"x": 0.1}},
        "strategy_throttle_state.json": {"paused_until": {}},
    }.items():
        (runtime / name).write_text(json.dumps(body))

    # day-files: clearly-past dates (pre-cutoff under both modes) + a non-canonical
    # sibling that must NOT be touched + an already-fenced file (idempotent skip).
    (data_trades / "trade_history_20260101.ndjson").write_text('{"x":1}\n')
    (data_trades / "trade_history_20260102.ndjson").write_text('{"x":2}\n')
    (data_trades / "trade_history_20260101.ndjson.pre_pnl_fix_bak").write_text("old\n")
    (data_trades / "trade_history_20251231.ndjson.scr_reset_bak").write_text("fenced\n")

    return runtime, data_trades


def _snapshot(paths):
    snap = {}
    for p in paths:
        snap[str(p)] = p.read_bytes() if p.exists() else None
    return snap


def _base_args(runtime, data_trades):
    return ["--runtime-dir", str(runtime), "--data-trades-dir", str(data_trades)]


@pytest.fixture(autouse=True)
def _paper_env(monkeypatch):
    # Default to a paper (not-live) execution mode for every test; refusal tests
    # override this explicitly.
    monkeypatch.setenv("CHAD_EXECUTION_MODE", "paper")


# --------------------------------------------------------------------------- #
# Dry-run: zero mutations
# --------------------------------------------------------------------------- #

def test_dry_run_default_mutates_nothing(tmp_path, capsys):
    runtime, data_trades = make_dirty(tmp_path)
    watched = list(runtime.iterdir()) + list(data_trades.iterdir())
    before = _snapshot(watched)

    rc = ERB.main(_base_args(runtime, data_trades))
    out = capsys.readouterr().out

    assert rc == 0
    assert "DRY-RUN" in out
    assert EXPECTED_TARGET in out  # plan shows the resolved target label
    # nothing changed on disk
    assert _snapshot(watched) == before
    assert not (runtime / "archive").exists()
    assert not (runtime / ERB.MARKER_FILENAME).exists()


def test_dry_run_reports_truthful_would_archive_bytes(tmp_path, capsys):
    """Regression for the 'bytes to archive: 0' display artifact: dry-run never
    ran execute_plan, so every actionable a.bytes was None and the summed total
    rendered as 0 despite real content. The preview must now stat each target and
    report a truthful, non-zero would-preserve total — split copy vs fence — and
    that total must equal exactly what APPLY archives on the same fixture."""
    runtime, data_trades = make_dirty(tmp_path)

    rc = ERB.main(_base_args(runtime, data_trades) + ["--json"])
    out = capsys.readouterr().out
    assert rc == 0

    # --- parse the dry-run JSON block (printed after the rendered plan) ---
    jstart = out.index("\n{", out.index("DRY-RUN")) if "DRY-RUN" in out else out.index("\n{")
    payload = json.loads(out[jstart:])
    pv = payload["would_preserve_bytes"]

    # headline regression: both legs are non-zero (the bug rendered 0)
    assert pv["to_archive_dir"] > 0
    assert pv["fenced_in_place"] > 0
    assert pv["total"] == pv["to_archive_dir"] + pv["fenced_in_place"]

    # the rendered human summary must echo the same non-zero total, not 0
    assert f"bytes preserved total   : {pv['total']:,}" in out
    assert "bytes to archive : 0" not in out

    # --- independent ground truth from a freshly built plan over the fixture ---
    ctx = ERB.resolve_context(ERB.build_arg_parser().parse_args(
        _base_args(runtime, data_trades)))
    plan = ERB.build_plan(ctx, CURRENT_EPOCH, EXPECTED_TARGET)
    assert all(a.bytes is None for a in plan)  # dry-run plan never stats (root cause)
    exp_copy = sum(a.src.stat().st_size for a in plan
                   if a.archive_mode == "copy" and a.status in ("PLANNED", "DONE"))
    exp_fence = sum(a.src.stat().st_size for a in plan
                    if a.archive_mode == "rename_in_place" and a.status in ("PLANNED", "DONE"))
    assert (pv["to_archive_dir"], pv["fenced_in_place"]) == (exp_copy, exp_fence)
    assert ERB.plan_preserved_bytes(plan) == (exp_copy, exp_fence)

    # --- preview must equal what APPLY actually preserves on the same fixture ---
    assert ERB.main(_base_args(runtime, data_trades) +
                    ["--apply", "--confirm", ERB.CONFIRM_TOKEN]) == 0
    arch = next((runtime / "archive").glob("epoch_2_pre_*"))
    manifest = json.loads((arch / "archive_manifest.json").read_text())
    archived_copy = sum(f["bytes"] for f in manifest["files_archived"])
    fenced_bytes = sum(f["bytes"] for f in manifest["day_files_fenced"])
    assert pv["to_archive_dir"] == archived_copy
    assert pv["fenced_in_place"] == fenced_bytes


def test_apply_without_token_stays_dry_run(tmp_path, capsys):
    runtime, data_trades = make_dirty(tmp_path)
    before = _snapshot(list(runtime.iterdir()) + list(data_trades.iterdir()))

    rc = ERB.main(_base_args(runtime, data_trades) + ["--apply"])  # no --confirm
    out = capsys.readouterr().out

    assert rc == 0
    assert "confirm token missing" in out
    assert _snapshot(list(runtime.iterdir()) + list(data_trades.iterdir())) == before
    assert not (runtime / "archive").exists()


# --------------------------------------------------------------------------- #
# Apply: full reset
# --------------------------------------------------------------------------- #

def test_apply_full_reset(tmp_path):
    runtime, data_trades = make_dirty(tmp_path)
    rc = ERB.main(_base_args(runtime, data_trades) +
                  ["--apply", "--confirm", ERB.CONFIRM_TOKEN])
    assert rc == 0

    # archive dir + integrity artifacts
    arch_dirs = list((runtime / "archive").glob("epoch_2_pre_*"))
    assert len(arch_dirs) == 1
    arch = arch_dirs[0]
    assert (arch / "SHA256SUMS").is_file()
    manifest = json.loads((arch / "archive_manifest.json").read_text())
    assert manifest["schema_version"] == ERB.MANIFEST_SCHEMA
    assert manifest["from_epoch"] == CURRENT_EPOCH
    assert manifest["to_epoch"] == EXPECTED_TARGET

    # H4-A: epoch bumped
    epoch = json.loads((runtime / "epoch_state.json").read_text())
    assert epoch["active_epoch"] == EXPECTED_TARGET
    assert epoch["previous_epoch_archive"] == str(arch)
    assert epoch["ready_for_live"] is False  # never flipped
    assert epoch["epoch_started_at_utc"].endswith("Z")

    # H4-B: canonical pre-cutoff day-files fenced; siblings untouched
    assert not (data_trades / "trade_history_20260101.ndjson").exists()
    assert (data_trades / "trade_history_20260101.ndjson.scr_reset_bak").is_file()
    assert (data_trades / "trade_history_20260102.ndjson.scr_reset_bak").is_file()
    assert (data_trades / "trade_history_20260101.ndjson.pre_pnl_fix_bak").is_file()
    assert manifest["day_files_fenced_count"] == 2

    # H2: both DBs + sidecars moved out (self-recreate on restart), archived
    for db in ("ibkr_adapter_state.sqlite3", "exec_state_paper.sqlite3"):
        assert not (runtime / db).exists()
        assert not (runtime / (db + "-wal")).exists()
        assert not (runtime / (db + "-shm")).exists()
        assert (arch / db).is_file()
        assert (arch / (db + "-wal")).is_file()
        assert (arch / (db + "-shm")).is_file()

    # H3: equity truncated + archived non-empty; HWM carriers removed+archived
    assert (runtime / "equity_history.ndjson").read_text() == ""
    assert (arch / "equity_history.ndjson").stat().st_size > 0
    assert "313124.35261756467" in (arch / "equity_history.ndjson").read_text()
    for name in ERB.HWM_CARRIER_FILES:
        assert not (runtime / name).exists()
        assert (arch / name).is_file()

    # H5: queues cleared, processed_fill_ids RETAINED; guard removed+archived
    tc = json.loads((runtime / "trade_closer_state.json").read_text())
    assert tc["queues"] == []
    assert tc["processed_fill_ids"] == ["fid-1", "fid-2", "fid-3"]
    assert not (runtime / "position_guard.json").exists()
    assert (arch / "position_guard.json").is_file()

    # H4 second family: all removed + archived
    for name in ERB.SECOND_FAMILY_FILES:
        assert not (runtime / name).exists()
        assert (arch / name).is_file()

    # completion marker written
    marker = json.loads((runtime / ERB.MARKER_FILENAME).read_text())
    assert marker["target_epoch_label"] == EXPECTED_TARGET
    assert marker["archive_dir"] == str(arch)

    # SHA256SUMS integrity: every listed hash matches the archived copy
    for line in (arch / "SHA256SUMS").read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        assert ERB._sha256_file(arch / name) == digest


def test_idempotent_rerun_is_noop(tmp_path, capsys):
    runtime, data_trades = make_dirty(tmp_path)
    args = _base_args(runtime, data_trades) + ["--apply", "--confirm", ERB.CONFIRM_TOKEN]
    assert ERB.main(args) == 0
    capsys.readouterr()

    archives_after_first = sorted((runtime / "archive").glob("epoch_2_pre_*"))
    marker_first = (runtime / ERB.MARKER_FILENAME).read_bytes()
    equity_first = (runtime / "equity_history.ndjson").read_bytes()

    rc = ERB.main(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "NO-OP" in out
    # no second archive, marker + (empty) equity unchanged
    assert sorted((runtime / "archive").glob("epoch_2_pre_*")) == archives_after_first
    assert (runtime / ERB.MARKER_FILENAME).read_bytes() == marker_first
    assert (runtime / "equity_history.ndjson").read_bytes() == equity_first


# --------------------------------------------------------------------------- #
# Refusal gates
# --------------------------------------------------------------------------- #

def test_refuse_when_ready_for_live_true(tmp_path, capsys):
    runtime, data_trades = make_dirty(tmp_path)
    (runtime / "live_readiness.json").write_text(json.dumps({
        "ready_for_live": True, "schema_version": "live_readiness_state.v1",
        "ts_utc": "2026-06-21T18:50:22Z", "ttl_seconds": 604800,
    }))
    before = _snapshot(list(runtime.iterdir()) + list(data_trades.iterdir()))

    rc = ERB.main(_base_args(runtime, data_trades) +
                  ["--apply", "--confirm", ERB.CONFIRM_TOKEN])
    err = capsys.readouterr().err

    assert rc == 3
    assert "REFUSED" in err
    assert _snapshot(list(runtime.iterdir()) + list(data_trades.iterdir())) == before
    assert not (runtime / "archive").exists()


def test_refuse_when_exec_mode_live(tmp_path, monkeypatch, capsys):
    runtime, data_trades = make_dirty(tmp_path)
    monkeypatch.setenv("CHAD_EXECUTION_MODE", "live")
    before = _snapshot(list(runtime.iterdir()) + list(data_trades.iterdir()))

    rc = ERB.main(_base_args(runtime, data_trades) +
                  ["--apply", "--confirm", ERB.CONFIRM_TOKEN])
    assert rc == 3
    assert _snapshot(list(runtime.iterdir()) + list(data_trades.iterdir())) == before
    assert not (runtime / "archive").exists()


def test_refuse_when_live_readiness_absent(tmp_path):
    runtime, data_trades = make_dirty(tmp_path)
    (runtime / "live_readiness.json").unlink()
    rc = ERB.main(_base_args(runtime, data_trades) +
                  ["--apply", "--confirm", ERB.CONFIRM_TOKEN])
    assert rc == 3
    assert not (runtime / "archive").exists()


# --------------------------------------------------------------------------- #
# Protected-path guard + scope
# --------------------------------------------------------------------------- #

def test_protected_path_guard_rejects_outside_and_protected(tmp_path):
    runtime, data_trades = make_dirty(tmp_path)
    ctx = ERB.resolve_context(ERB.build_arg_parser().parse_args(
        _base_args(runtime, data_trades)))

    # legitimate runtime/data targets pass
    ERB.assert_safe_target(ctx, runtime / "equity_history.ndjson")
    ERB.assert_safe_target(ctx, data_trades / "trade_history_20260101.ndjson")

    # outside the two roots -> refuse
    with pytest.raises(SystemExit):
        ERB.assert_safe_target(ctx, _REPO_ROOT / "chad" / "core" / "live_loop.py")
    with pytest.raises(SystemExit):
        ERB.assert_safe_target(ctx, _REPO_ROOT / "venv" / "x")
    with pytest.raises(SystemExit):
        ERB.assert_safe_target(ctx, Path.home() / ".claude" / "x")
    with pytest.raises(SystemExit):
        ERB.assert_safe_target(ctx, _REPO_ROOT / "audits" / "x")
    # protected config basename inside runtime -> refuse
    with pytest.raises(SystemExit):
        ERB.assert_safe_target(ctx, runtime / "tiers.json")
    with pytest.raises(SystemExit):
        ERB.assert_safe_target(ctx, runtime / "withdrawal_policy.json")


def test_plan_targets_only_runtime_and_data(tmp_path):
    runtime, data_trades = make_dirty(tmp_path)
    ctx = ERB.resolve_context(ERB.build_arg_parser().parse_args(
        _base_args(runtime, data_trades)))
    plan = ERB.build_plan(ctx, CURRENT_EPOCH, EXPECTED_TARGET)
    roots = (str(runtime.resolve()), str(data_trades.resolve()))
    for a in plan:
        rp = str(a.src.resolve()) if a.src.exists() else str(a.src)
        assert any(rp == r or rp.startswith(r + os.sep) for r in roots), rp


# --------------------------------------------------------------------------- #
# Cutoff modes + DB both-or-neither + label increment
# --------------------------------------------------------------------------- #

def test_cutoff_before_today_leaves_same_day_file(tmp_path):
    runtime, data_trades = make_dirty(tmp_path)
    today = ERB._utcnow().strftime("%Y%m%d")
    same_day = data_trades / f"trade_history_{today}.ndjson"
    same_day.write_text('{"today":1}\n')

    ERB.main(_base_args(runtime, data_trades) +
             ["--apply", "--confirm", ERB.CONFIRM_TOKEN])
    # before-today (default) must leave today's file alone
    assert same_day.exists()
    assert not (data_trades / f"trade_history_{today}.ndjson.scr_reset_bak").exists()


def test_cutoff_all_fences_same_day_file(tmp_path):
    runtime, data_trades = make_dirty(tmp_path)
    today = ERB._utcnow().strftime("%Y%m%d")
    same_day = data_trades / f"trade_history_{today}.ndjson"
    same_day.write_text('{"today":1}\n')

    ERB.main(_base_args(runtime, data_trades) +
             ["--apply", "--confirm", ERB.CONFIRM_TOKEN, "--cutoff", "all"])
    assert not same_day.exists()
    assert (data_trades / f"trade_history_{today}.ndjson.scr_reset_bak").exists()


def test_db_half_state_refuses(tmp_path):
    runtime, data_trades = make_dirty(tmp_path)
    # remove exactly one DB -> half state
    (runtime / "exec_state_paper.sqlite3").unlink()
    with pytest.raises(SystemExit):
        ERB.main(_base_args(runtime, data_trades) +
                 ["--apply", "--confirm", ERB.CONFIRM_TOKEN])
    # the present DB must remain untouched (no half-reset)
    assert (runtime / "ibkr_adapter_state.sqlite3").exists()


def test_epoch_label_increment_helper():
    assert ERB._increment_epoch_label("CHAD_v8.9_Paper_Epoch_2") == "CHAD_v8.9_Paper_Epoch_3"
    assert ERB._increment_epoch_label("Epoch_9") == "Epoch_10"
    assert ERB._increment_epoch_label("no_number") is None


def test_no_hwm_carriers_flag_excludes_them(tmp_path):
    runtime, data_trades = make_dirty(tmp_path)
    ERB.main(_base_args(runtime, data_trades) +
             ["--apply", "--confirm", ERB.CONFIRM_TOKEN, "--no-hwm-carriers"])
    # equity still cleared, but carriers preserved in place
    assert (runtime / "equity_history.ndjson").read_text() == ""
    for name in ERB.HWM_CARRIER_FILES:
        assert (runtime / name).exists()
