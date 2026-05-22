#!/usr/bin/env python3
"""
CHAD — Lifecycle Truth Bootstrap Publisher (SSOT v5.0)

Publishes SSOT v5.0 lifecycle truth artifacts (safe, fail-closed bootstrap):
- runtime/trade_lifecycle_state.json
- runtime/positions_truth.json

Why this exists:
- Your active repo does NOT currently implement the full broker_events/fills/fees truth-rebuild engine.
- SSOT v5.0 requires these runtime artifacts to exist and to fail closed when truth can't be proven.

Guarantees:
- Never enables trading.
- Never mutates config.
- Never performs network calls.
- Never crashes if inputs are missing.
- Always writes outputs each run with ts_utc + ttl_seconds (atomic writes via runtime_json helper).
- Conservative flags:
    - gap_flag/backlog_flag default TRUE if broker_events evidence is missing
    - truth_ok FALSE unless we can prove a broker_events stream is present AND recent (basic evidence only)

Inputs (best-effort):
- runtime/positions_snapshot.json (may include positions + cash)
- data/broker_events/ ledger directory (if present)
- runtime/ibkr_paper_ledger_state.json (optional hints)
- runtime/reconciliation_state.json (optional hints)

Outputs:
- trade_lifecycle_state.json: stream health summary + backlog/gap flags
- positions_truth.json: best-effort truth snapshot (mirrors positions_snapshot), marked truth_ok=false unless proven

Later upgrade path:
- Replace evidence checks with deterministic rebuild over broker_events + fills + fees
- Set truth_ok true only when rebuild is complete and gap/backlog flags are false
"""

# BROKER_EVENTS_MAX_AGE_ENV_FIX_V1
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from chad.utils.runtime_json import read_runtime_state_json, stable_json_dumps, write_runtime_state_json

LOG = logging.getLogger("chad.ops.lifecycle_truth_publisher")


# ----------------------------
# Env + path helpers
# ----------------------------

def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    try:
        v = int(raw)
        return v if v > 0 else default
    except Exception:
        return default


def _repo_root() -> Path:
    root = str(os.environ.get("CHAD_ROOT", "")).strip()
    if root:
        p = Path(root).expanduser()
        if p.is_dir():
            return p.resolve()
    return Path(__file__).resolve().parents[2]


def _runtime_dir(repo_root: Path) -> Path:
    rd = str(os.environ.get("CHAD_RUNTIME_DIR", "")).strip()
    if rd:
        return Path(rd).expanduser().resolve()
    return (repo_root / "runtime").resolve()


def _data_dir(repo_root: Path) -> Path:
    dd = str(os.environ.get("CHAD_DATA_DIR", "")).strip()
    if dd:
        return Path(dd).expanduser().resolve()
    return (repo_root / "data").resolve()


def _read_json_best_effort(path: Path) -> Dict[str, Any]:
    try:
        obj, _fresh = read_runtime_state_json(path, default_ttl_seconds=0)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        try:
            import json
            if path.is_file():
                x = json.loads(path.read_text(encoding="utf-8"))
                return x if isinstance(x, dict) else {}
        except Exception:
            return {}
    return {}


def _sha256_prefixed(payload: Dict[str, Any]) -> str:
    b = stable_json_dumps(payload).encode("utf-8")
    return f"sha256:{hashlib.sha256(b).hexdigest()}"


def _normalize_ledger_open_records(ledger_state: Any) -> Dict[str, Dict[str, Any]]:
    """Return ledger open records keyed by their original id, regardless of schema.

    Accepts both:
      - Wrapped schema: {"open": {<id>: {"symbol": ..., "qty": ..., ...}, ...}}
      - Flat schema (current writer in chad/portfolio/ibkr_paper_ledger_watcher.py):
          {<id>: {"symbol": ..., "qty": ..., ...}, ...}

    Includes only dict records with symbol present, qty present, and float(qty) != 0.0.
    """
    if not isinstance(ledger_state, dict):
        return {}

    candidate: Any = ledger_state.get("open")
    if not isinstance(candidate, dict):
        candidate = ledger_state

    out: Dict[str, Dict[str, Any]] = {}
    for key, row in candidate.items():
        if not isinstance(row, dict):
            continue
        sym = row.get("symbol")
        if not isinstance(sym, str) or not sym.strip():
            continue
        if "qty" not in row:
            continue
        try:
            qty = float(row.get("qty") or 0.0)
        except (TypeError, ValueError):
            continue
        if qty == 0.0:
            continue
        out[str(key)] = row
    return out


# ----------------------------
# Broker events evidence (bootstrap only)
# ----------------------------

@dataclass(frozen=True)
class BrokerEventsEvidence:
    exists: bool
    newest_file: Optional[str]
    newest_mtime_unix: Optional[float]
    last_event_ts_utc: Optional[str]
    event_count_hint: int


@dataclass(frozen=True)
class LedgerEvidence:
    exists: bool
    newest_file: Optional[str]
    newest_mtime_unix: Optional[float]
    line_count_hint: int


def _find_broker_events(data_dir: Path) -> BrokerEventsEvidence:
    """
    Evidence check:
    - Looks for data/broker_events/*.ndjson
    - Uses newest file mtime as a freshness proxy
    - Attempts to parse the last ~50 lines to find a 'ts_utc' field (best effort)
    """
    be_dir = data_dir / "broker_events"
    if not be_dir.is_dir():
        return BrokerEventsEvidence(False, None, None, None, 0)

    files = [p for p in be_dir.glob("*.ndjson") if p.is_file()]
    if not files:
        return BrokerEventsEvidence(False, None, None, None, 0)

    newest = max(files, key=lambda p: p.stat().st_mtime)
    newest_mtime = float(newest.stat().st_mtime)
    last_ts: Optional[str] = None
    count_hint = 0

    try:
        with newest.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[-50:]
        count_hint = len(lines)
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                import json
                obj = json.loads(line)
                if isinstance(obj, dict):
                    ts = obj.get("ts_utc")
                    if isinstance(ts, str) and ts:
                        last_ts = ts
                        break
            except Exception:
                continue
    except Exception:
        pass

    return BrokerEventsEvidence(True, str(newest), newest_mtime, last_ts, int(count_hint))


def _find_ledger_evidence(data_dir: Path, subdir: str) -> LedgerEvidence:
    d = data_dir / subdir
    if not d.is_dir():
        return LedgerEvidence(False, None, None, 0)

    files = [p for p in d.glob("*.ndjson") if p.is_file()]
    if not files:
        return LedgerEvidence(False, None, None, 0)

    newest = max(files, key=lambda p: p.stat().st_mtime)
    newest_mtime = float(newest.stat().st_mtime)
    line_count_hint = 0

    try:
        with newest.open("r", encoding="utf-8", errors="ignore") as f:
            line_count_hint = sum(1 for _ in f)
    except Exception:
        line_count_hint = 0

    return LedgerEvidence(True, str(newest), newest_mtime, int(line_count_hint))


def _now_unix() -> float:
    return time.time()


def _is_recent(mtime_unix: Optional[float], *, max_age_s: int) -> bool:
    if mtime_unix is None:
        return False
    return (_now_unix() - float(mtime_unix)) <= float(max_age_s)


def _parse_iso8601_to_unix(ts: Any) -> Optional[float]:
    """Best-effort ISO-8601 -> POSIX seconds. Accepts trailing 'Z' and naive
    datetimes (treated as UTC). Returns None on failure — caller decides how
    to fail closed."""
    if not isinstance(ts, str) or not ts:
        return None
    s = ts.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _count_recent_non_heartbeats(
    newest_file: Optional[str],
    *,
    max_age_s: int,
    scan_lines: int = 400,
) -> int:
    """GAP-053 helper: count broker_events that are (a) NOT heartbeats and
    (b) within max_age_s seconds of now. Used by build_trade_lifecycle_state
    to distinguish "pipeline working but no fills happened" (legitimate quiet
    window, backlog_flag can be False) from "fills happened but ledger
    writer broke" (backlog_flag must stay True).

    Reads up to the last `scan_lines` lines of the newest broker_events
    ndjson; that bound keeps cost O(1) on long files. Heartbeats dominate
    by ~25:1, so 400 lines covers ~6 minutes of heartbeats plus any fills
    in the trailing window — comfortably more than the default 900s window
    needs.

    Returns 0 on any failure (path missing, parse error, etc.); callers
    treat 0-count as "no recent fills" which keeps the quiet-window branch
    safe (it only relaxes backlog_flag when this is 0 AND broker_events
    mtime is fresh).
    """
    if not newest_file:
        return 0
    try:
        p = Path(newest_file)
        if not p.is_file():
            return 0
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[-int(scan_lines):]
    except Exception:
        return 0

    now = _now_unix()
    cutoff_unix = now - float(max_age_s)
    count = 0
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            import json
            obj = json.loads(raw)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else obj
        event_type = str(payload.get("event_type") or "").strip().lower()
        if event_type == "heartbeat":
            continue
        ts_unix = _parse_iso8601_to_unix(payload.get("ts_utc"))
        if ts_unix is None:
            continue
        if ts_unix >= cutoff_unix and ts_unix <= now + 1.0:
            count += 1
    return count


# ----------------------------
# Build payloads
# ----------------------------

def build_trade_lifecycle_state(
    *,
    repo_root: Path,
    runtime_dir: Path,
    data_dir: Path,
    evidence: BrokerEventsEvidence,
    fills_evidence: LedgerEvidence,
    fees_evidence: LedgerEvidence,
) -> Dict[str, Any]:
    """
    SSOT v5.0 expects:
      ts_utc, ttl_seconds,
      venue, last_event_ts_utc, event_count_rolling, backlog_flag, gap_flag

    Bootstrap semantics:
    - If broker_events evidence missing => gap_flag=True and backlog_flag=True (fail closed).
    - If evidence exists but stale => backlog_flag=True (fail closed).
    """
    venue = str(os.environ.get("CHAD_LIFECYCLE_VENUE", "IBKR")).strip() or "IBKR"
    max_age_s = _env_int("CHAD_BROKER_EVENTS_MAX_AGE_SECONDS", 900)

    have_events = evidence.exists
    events_fresh = have_events and _is_recent(evidence.newest_mtime_unix, max_age_s=max_age_s)

    have_fills = fills_evidence.exists and fills_evidence.line_count_hint > 0
    fills_fresh = have_fills and _is_recent(fills_evidence.newest_mtime_unix, max_age_s=max_age_s)

    have_fees = fees_evidence.exists and fees_evidence.line_count_hint > 0
    fees_fresh = have_fees and _is_recent(fees_evidence.newest_mtime_unix, max_age_s=max_age_s)

    gap_flag = not (have_events and have_fills and have_fees)

    # GAP-053 quiet-window policy: distinguish "pipeline broken" from
    # "no fills happened recently". The pre-fix logic treated fills/fees
    # mtime aging as backlog, which produced false positives during long
    # cooldown / no-signal periods. Policy is documented in
    # ops/pending_actions/GAP-053_lifecycle_replay_coverage_policy.md.
    #
    # Fail-closed invariants preserved:
    #   * Missing files (gap_flag) ALWAYS forces backlog_flag=true.
    #   * Stale broker_events (no heartbeats) ALWAYS forces backlog_flag=true
    #     (this is the canonical "lifecycle pipeline alive" signal).
    #   * If a non-heartbeat broker event arrived within max_age_s but the
    #     fills/fees writers fell behind, backlog_flag=true (writer outage).
    #   * Only when broker_events are fresh AND no recent fill-class events
    #     occurred do we accept stale fills/fees as a legitimate quiet
    #     window (backlog_flag=false). The accepted-policy field below
    #     surfaces the decision to operators for audit.
    recent_non_heartbeats = _count_recent_non_heartbeats(
        evidence.newest_file, max_age_s=max_age_s
    )
    had_recent_broker_fill_activity = recent_non_heartbeats > 0
    quiet_window_accepted = bool(
        not gap_flag
        and events_fresh
        and not had_recent_broker_fill_activity
        and (not fills_fresh or not fees_fresh)
    )

    if gap_flag:
        backlog_flag = True
    elif not events_fresh:
        # Pipeline outage — heartbeats stopped.
        backlog_flag = True
    elif had_recent_broker_fill_activity and (not fills_fresh or not fees_fresh):
        # Writer fell behind real fill activity.
        backlog_flag = True
    else:
        backlog_flag = False

    payload: Dict[str, Any] = {
        "schema_version": "trade_lifecycle_state.v1",
        "venue": venue,
        "last_event_ts_utc": evidence.last_event_ts_utc,
        "event_count_rolling": int(evidence.event_count_hint),
        "backlog_flag": bool(backlog_flag),
        "gap_flag": bool(gap_flag),
        "quiet_window_accepted": bool(quiet_window_accepted),
        "replay_coverage_policy": "GAP-053",
        "evidence": {
            "broker_events": {
                "dir": str((data_dir / "broker_events").resolve()),
                "exists": bool(evidence.exists),
                "newest_file": evidence.newest_file,
                "newest_mtime_unix": evidence.newest_mtime_unix,
                "line_count_hint": int(evidence.event_count_hint),
                "events_fresh": bool(events_fresh),
                "recent_non_heartbeat_count": int(recent_non_heartbeats),
            },
            "fills": {
                "dir": str((data_dir / "fills").resolve()),
                "exists": bool(fills_evidence.exists),
                "newest_file": fills_evidence.newest_file,
                "newest_mtime_unix": fills_evidence.newest_mtime_unix,
                "line_count_hint": int(fills_evidence.line_count_hint),
                "fresh": bool(fills_fresh),
            },
            "fees": {
                "dir": str((data_dir / "fees").resolve()),
                "exists": bool(fees_evidence.exists),
                "newest_file": fees_evidence.newest_file,
                "newest_mtime_unix": fees_evidence.newest_mtime_unix,
                "line_count_hint": int(fees_evidence.line_count_hint),
                "fresh": bool(fees_fresh),
            },
            "max_age_seconds": int(max_age_s),
            "had_recent_broker_fill_activity": bool(had_recent_broker_fill_activity),
        },
        "notes": (
            "EVIDENCE-GATED BOOTSTRAP with GAP-053 quiet-window policy: "
            "broker_events heartbeat freshness is the canonical lifecycle "
            "liveness signal. fills/fees mtime staleness is only treated as "
            "backlog when broker_events report non-heartbeat activity within "
            "max_age_seconds (writer-outage signal). When the broker emitted "
            "no fill-class events in that window, stale ledger mtimes reflect "
            "a quiet market window, not a backlog. gap_flag still trips on "
            "missing files; downstream live-readiness still ANDs lifecycle_truth, "
            "reconciliation, mode, scr, operator_intent, etc."
        ),
    }
    return payload


def build_positions_truth(
    *,
    repo_root: Path,
    runtime_dir: Path,
    data_dir: Path,
    evidence: BrokerEventsEvidence,
    fills_evidence: LedgerEvidence,
    fees_evidence: LedgerEvidence,
) -> Dict[str, Any]:
    """
    SSOT v5.0 expects positions_truth to be rebuilt from broker_events + fills + fees.
    Bootstrap behavior:
    - Mirror positions_snapshot if present, but mark truth_ok=false unless broker_events evidence exists and is fresh.
    """
    pos_snap = _read_json_best_effort(runtime_dir / "positions_snapshot.json")
    cash = pos_snap.get("cash")
    positions = pos_snap.get("positions")

    if not isinstance(cash, dict):
        cash = {}
    if not isinstance(positions, list):
        positions = []

    max_age_s = _env_int("CHAD_BROKER_EVENTS_MAX_AGE_SECONDS", 900)
    fills_present = bool(fills_evidence.exists and fills_evidence.line_count_hint > 0)
    fees_present = bool(fees_evidence.exists and fees_evidence.line_count_hint > 0)

    reconciliation = _read_json_best_effort(runtime_dir / "reconciliation_state.json")
    ledger_state = _read_json_best_effort(runtime_dir / "ibkr_paper_ledger_state.json")
    replay_state = _read_json_best_effort(runtime_dir / "lifecycle_replay_state.json")
    replay_coverage = _read_json_best_effort(runtime_dir / "lifecycle_replay_coverage.json")

    reconciliation_status_upstream = str(reconciliation.get("status") or "").strip().upper()
    upstream_green = reconciliation_status_upstream == "GREEN"

    ledger_open = _normalize_ledger_open_records(ledger_state)
    ledger_state_positions_count = int(len(ledger_open))

    replay_positions = replay_state.get("positions")
    if not isinstance(replay_positions, dict):
        replay_positions = {}
    replay_positions_count = int(len(replay_positions))

    replay_coverage_status = str(replay_coverage.get("status") or "").strip().upper()
    replay_summary = replay_coverage.get("summary")
    if not isinstance(replay_summary, dict):
        replay_summary = {}

    missing_from_replay = replay_coverage.get("missing_from_replay")
    if not isinstance(missing_from_replay, list):
        missing_from_replay = []

    replay_only = replay_coverage.get("replay_only")
    if not isinstance(replay_only, list):
        replay_only = []

    qty_mismatches = replay_coverage.get("qty_mismatches")
    if not isinstance(qty_mismatches, list):
        qty_mismatches = []

    snapshot_positions_count = int(len(positions))

    replay_inputs = replay_state.get("inputs")
    if not isinstance(replay_inputs, dict):
        replay_inputs = {}

    broker_non_heartbeat_count = int(replay_inputs.get("broker_non_heartbeat_count", 0) or 0)
    replay_match_confirmed = replay_coverage_status == "REPLAY_MATCH_CONFIRMED"

    scope_mismatch = replay_coverage_status == "SCOPE_MISMATCH_MANUAL_VS_PAPER_EXEC"

    qty_mismatches_count = int(len(qty_mismatches))
    missing_from_replay_count = int(len(missing_from_replay))

    # GAP-A009: classify reconciliation_status with replay/scope evidence.
    # RED   = upstream RED, or quantity/symbol mismatches detected.
    # YELLOW = scope mismatch OR replay not confirmed.
    # GREEN = upstream GREEN AND replay confirmed AND no scope mismatch AND no qty/symbol mismatches.
    if reconciliation_status_upstream == "RED" or qty_mismatches_count > 0 or missing_from_replay_count > 0:
        reconciliation_status = "RED"
        if reconciliation_status_upstream == "RED":
            reconciliation_status_reason = (
                f"UPSTREAM_RED: snapshot={snapshot_positions_count} "
                f"replay={replay_positions_count} ledger={ledger_state_positions_count}"
            )
        else:
            reconciliation_status_reason = (
                f"QTY_OR_SYMBOL_MISMATCH: qty_mismatches={qty_mismatches_count} "
                f"missing_from_replay={missing_from_replay_count} "
                f"snapshot={snapshot_positions_count} replay={replay_positions_count} "
                f"ledger={ledger_state_positions_count}"
            )
    elif scope_mismatch:
        reconciliation_status = "YELLOW"
        reconciliation_status_reason = (
            f"SCOPE_MISMATCH_MANUAL_VS_PAPER_EXEC: snapshot={snapshot_positions_count} "
            f"replay={replay_positions_count} ledger={ledger_state_positions_count}"
        )
    elif not replay_match_confirmed:
        reconciliation_status = "YELLOW"
        reconciliation_status_reason = (
            f"REPLAY_NOT_CONFIRMED: replay_coverage_status={replay_coverage_status or 'UNKNOWN'} "
            f"snapshot={snapshot_positions_count} replay={replay_positions_count} "
            f"ledger={ledger_state_positions_count}"
        )
    elif upstream_green:
        reconciliation_status = "GREEN"
        reconciliation_status_reason = (
            f"GREEN_CONFIRMED: snapshot={snapshot_positions_count} "
            f"replay={replay_positions_count} ledger={ledger_state_positions_count}"
        )
    else:
        # Upstream is missing/unknown — surface it explicitly.
        reconciliation_status = "YELLOW"
        reconciliation_status_reason = (
            f"UPSTREAM_NOT_GREEN: upstream={reconciliation_status_upstream or 'UNKNOWN'} "
            f"snapshot={snapshot_positions_count} replay={replay_positions_count} "
            f"ledger={ledger_state_positions_count}"
        )

    reconciliation_green = reconciliation_status == "GREEN"

    truth_ok = bool(
        upstream_green
        and bool(ledger_state)
        and snapshot_positions_count > 0
        and ledger_state_positions_count == snapshot_positions_count
    )

    if truth_ok and scope_mismatch:
        truth_source = "BROKER_SNAPSHOT_RECONCILED_WITH_LEDGER_SCOPE_MISMATCH_REPLAY_DIAGNOSTIC_ONLY"
    elif truth_ok:
        truth_source = "BROKER_SNAPSHOT_RECONCILED_WITH_LEDGER"
    else:
        truth_source = "FAIL_CLOSED_BOOTSTRAP_SCOPE_UNPROVEN"

    payload_nohash: Dict[str, Any] = {
        "schema_version": "positions_truth.v1",
        "as_of_event_ts_utc": evidence.last_event_ts_utc,
        "cash": cash,
        "positions": positions,
        "fees_included": bool(fees_present),
        "truth_ok": bool(truth_ok),
        "truth_source": truth_source,
        "evidence": {
            "broker_events_present": bool(evidence.exists),
            "broker_events_newest_file": evidence.newest_file,
            "broker_events_newest_mtime_unix": evidence.newest_mtime_unix,
            "fills_present": fills_present,
            "fills_newest_file": fills_evidence.newest_file,
            "fills_newest_mtime_unix": fills_evidence.newest_mtime_unix,
            "fills_line_count_hint": int(fills_evidence.line_count_hint),
            "fees_present": fees_present,
            "fees_newest_file": fees_evidence.newest_file,
            "fees_newest_mtime_unix": fees_evidence.newest_mtime_unix,
            "fees_line_count_hint": int(fees_evidence.line_count_hint),
            "reconciliation_status": reconciliation_status,
            "reconciliation_status_reason": reconciliation_status_reason,
            "reconciliation_status_upstream": reconciliation_status_upstream,
            "reconciliation_green": bool(reconciliation_green),
            "reconciliation_ts_utc": reconciliation.get("ts_utc"),
            "ledger_state_present": bool(ledger_state),
            "ledger_state_positions_count": int(ledger_state_positions_count),
            "ledger_state_last_run_utc": ledger_state.get("last_run_utc"),
            "snapshot_positions_count": int(snapshot_positions_count),
            "replay_state_present": bool(replay_state),
            "replay_positions_count": int(replay_positions_count),
            "replay_coverage_status": replay_coverage_status,
            "replay_summary": replay_summary,
            "missing_from_replay": missing_from_replay,
            "replay_only": replay_only,
            "qty_mismatches_count": int(len(qty_mismatches)),
            "broker_non_heartbeat_count": int(broker_non_heartbeat_count),
            "replay_match_confirmed": bool(replay_match_confirmed),
            "max_age_seconds": int(max_age_s),
        },
        "notes": (
            "BROKER-AUTHORITY PATH: positions_truth is marked true when runtime/positions_snapshot.json "
            "and runtime/ibkr_paper_ledger_state.json reconcile GREEN and position counts align. "
            "Lifecycle replay evidence remains diagnostic only. "
            "If replay scope is mismatched against a manual/IBKR holdings book, that mismatch does not block "
            "broker-authority truth for this path."
        ),
    }
    payload_nohash["hash_sha256"] = _sha256_prefixed({k: v for k, v in payload_nohash.items() if k != "hash_sha256"})
    return payload_nohash


# ----------------------------
# Publish
# ----------------------------

def publish_once(*, repo_root: Path, runtime_dir: Path, data_dir: Path) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)

    ttl_lifecycle = _env_int("CHAD_TRADE_LIFECYCLE_TTL_SECONDS", 60)
    ttl_truth = _env_int("CHAD_POSITIONS_TRUTH_TTL_SECONDS", 60)

    evidence = _find_broker_events(data_dir)
    fills_evidence = _find_ledger_evidence(data_dir, "fills")
    fees_evidence = _find_ledger_evidence(data_dir, "fees")

    lifecycle_payload = build_trade_lifecycle_state(
        repo_root=repo_root,
        runtime_dir=runtime_dir,
        data_dir=data_dir,
        evidence=evidence,
        fills_evidence=fills_evidence,
        fees_evidence=fees_evidence,
    )
    truth_payload = build_positions_truth(
        repo_root=repo_root,
        runtime_dir=runtime_dir,
        data_dir=data_dir,
        evidence=evidence,
        fills_evidence=fills_evidence,
        fees_evidence=fees_evidence,
    )

    lifecycle_path = runtime_dir / "trade_lifecycle_state.json"
    truth_path = runtime_dir / "positions_truth.json"

    write_runtime_state_json(lifecycle_path, lifecycle_payload, ttl_seconds=int(ttl_lifecycle), inject_ts=True)
    write_runtime_state_json(truth_path, truth_payload, ttl_seconds=int(ttl_truth), inject_ts=True)

    LOG.info("published trade_lifecycle_state=%s ttl=%ss", lifecycle_path, ttl_lifecycle)
    LOG.info("published positions_truth=%s ttl=%ss truth_ok=%s", truth_path, ttl_truth, truth_payload.get("truth_ok"))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Publish SSOT v5.0 trade_lifecycle_state + positions_truth (bootstrap).")
    ap.add_argument("--once", action="store_true", help="Publish once and exit (default).")
    ap.add_argument("--loop", action="store_true", help="Loop forever, publishing at a fixed interval.")
    ap.add_argument("--interval-seconds", type=int, default=_env_int("CHAD_LIFECYCLE_LOOP_INTERVAL_SECONDS", 60))
    ap.add_argument("--log-level", type=str, default=os.environ.get("CHAD_LOG_LEVEL", "INFO"))
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)sZ %(levelname)s %(name)s %(message)s",
    )

    repo_root = _repo_root()
    runtime_dir = _runtime_dir(repo_root)
    data_dir = _data_dir(repo_root)

    if args.loop:
        interval = int(max(10, args.interval_seconds))
        LOG.info("starting loop publisher: runtime_dir=%s data_dir=%s interval=%ss", runtime_dir, data_dir, interval)
        while True:
            try:
                publish_once(repo_root=repo_root, runtime_dir=runtime_dir, data_dir=data_dir)
            except Exception as e:
                LOG.exception("publish failed (will retry): %s", e)
            time.sleep(interval)
        return 0

    try:
        publish_once(repo_root=repo_root, runtime_dir=runtime_dir, data_dir=data_dir)
        return 0
    except Exception as e:
        LOG.exception("publish failed: %s", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
