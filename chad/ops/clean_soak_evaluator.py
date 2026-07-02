"""chad.ops.clean_soak_evaluator — GAP-039 clean-soak evaluator.

Reads the honest post-Epoch-3 SCR sample and emits a single gradable
``runtime/clean_soak_state.json``.

Design contract
---------------
* The evaluator MIRRORS SCR's own definition of an effective trade; it never
  uses a naive file row-count. The AUTHORITATIVE effective count is read
  straight from ``scr_state.stats.effective_trades`` (the Shadow Confidence
  Router is the source of truth). An independent, best-effort tally over
  post-anchor rows only populates the diagnostic fields and a discrepancy
  tripwire — it never silently overrides SCR. If the independent tally
  disagrees with SCR by more than 0 the evaluator records BOTH and sets
  ``discrepancy_flag=true``.
* Additive only: reads state, writes ONE new state file atomically, mutates
  no existing file and changes no existing behaviour.

Effective-trade predicate (mirrors ``chad.analytics.trade_stats_engine``)
-------------------------------------------------------------------------
A soak-effective row is a payload where::

    (exit_time_utc OR entry_time_utc) >= soak_anchor
    AND pnl_untrusted is NOT truthy   (checked the way SCR checks it:
                                       'pnl_untrusted' in tags, OR
                                       extra.pnl_untrusted, OR the documented
                                       top-level payload.pnl_untrusted)
    AND pnl is finite
    AND is_live is False              (paper lane)
    AND pnl != 0.0                    (zero-P&L flats are excluded exactly as
                                       SCR's effective_trades already reflects)

Documentation grounding
------------------------
Standard library only (json, os, re, math, argparse, logging, datetime,
dataclasses, pathlib, typing) plus the in-repo ``chad.utils.runtime_json``
atomic-write/freshness helpers. No third-party API is used, so no Context7
lookup was required.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from chad.utils.runtime_json import (
    atomic_write_json,
    read_json,
    read_runtime_state_json,
    utc_now,
)

log = logging.getLogger("chad.ops.clean_soak_evaluator")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SCHEMA_VERSION = "clean_soak_state.v1"
OUTPUT_TTL_SECONDS = 300

SCR_STATE_FILENAME = "scr_state.json"
EPOCH_RESET_FILENAME = "epoch_reset_state.json"
POSITIONS_TRUTH_FILENAME = "positions_truth.json"
OUTPUT_FILENAME = "clean_soak_state.json"

# Exact daily-ledger filename form. `.scr_reset_bak`, `.pre_pnl_fix_bak`,
# enriched/aux ledgers, etc. all fail this match and are therefore ignored.
TRADE_FILE_RE = re.compile(r"^trade_history_(\d{8})\.ndjson$")

GATE_KEYS: Tuple[str, ...] = (
    "authority_green",
    "effective_met",
    "sessions_met",
    "untrusted_ok",
    "no_discrepancy",
)

# Config-driven grade thresholds. Overridable via environment.
DEFAULT_SOAK_CRITERIA: Dict[str, Any] = {
    "min_effective_trades": 100,
    "min_sessions": 5,
    "max_untrusted_ratio": 0.05,
}

ENV_MIN_EFFECTIVE = "CHAD_SOAK_MIN_EFFECTIVE_TRADES"
ENV_MIN_SESSIONS = "CHAD_SOAK_MIN_SESSIONS"
ENV_MAX_UNTRUSTED_RATIO = "CHAD_SOAK_MAX_UNTRUSTED_RATIO"


# --------------------------------------------------------------------------- #
# Value types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Criteria:
    min_effective_trades: int
    min_sessions: int
    max_untrusted_ratio: float


@dataclass(frozen=True)
class Diagnostics:
    rows_seen: int
    rows_post_anchor: int
    rows_zero_pnl: int
    rows_untrusted: int
    evaluator_effective_tally: int


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #
def _default_repo_root() -> Path:
    # chad/ops/clean_soak_evaluator.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


def _default_runtime_dir() -> Path:
    return _default_repo_root() / "runtime"


def _default_trades_dir() -> Path:
    return _default_repo_root() / "data" / "trades"


# --------------------------------------------------------------------------- #
# Criteria / env parsing (fail-safe: log and fall back to default, never raise)
# --------------------------------------------------------------------------- #
def load_criteria(env: Optional[Mapping[str, str]] = None) -> Criteria:
    """Build :class:`Criteria` from ``env`` (defaults to ``os.environ``)."""
    source: Mapping[str, str] = env if env is not None else os.environ

    def _coerce_int(key: str, default: int) -> int:
        raw = source.get(key)
        if raw is None or str(raw).strip() == "":
            return default
        try:
            return int(str(raw).strip())
        except (ValueError, TypeError):
            log.warning("invalid %s=%r; falling back to default %d", key, raw, default)
            return default

    def _coerce_float(key: str, default: float) -> float:
        raw = source.get(key)
        if raw is None or str(raw).strip() == "":
            return default
        try:
            return float(str(raw).strip())
        except (ValueError, TypeError):
            log.warning("invalid %s=%r; falling back to default %g", key, raw, default)
            return default

    return Criteria(
        min_effective_trades=_coerce_int(
            ENV_MIN_EFFECTIVE, int(DEFAULT_SOAK_CRITERIA["min_effective_trades"])
        ),
        min_sessions=_coerce_int(
            ENV_MIN_SESSIONS, int(DEFAULT_SOAK_CRITERIA["min_sessions"])
        ),
        max_untrusted_ratio=_coerce_float(
            ENV_MAX_UNTRUSTED_RATIO, float(DEFAULT_SOAK_CRITERIA["max_untrusted_ratio"])
        ),
    )


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #
def parse_iso_utc(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp (``Z`` or offset) into an aware UTC datetime.

    Returns ``None`` on any non-string / unparseable input — never raises.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _finite_float(value: Any) -> Optional[float]:
    """Return ``float(value)`` iff finite, else ``None`` (non-raising)."""
    try:
        f = float(value)
    except (ValueError, TypeError):
        return None
    return f if math.isfinite(f) else None


def row_is_untrusted(payload: Mapping[str, Any]) -> bool:
    """Mirror ``trade_stats_engine._is_untrusted`` plus the documented shape.

    Untrusted when any of:
      * ``'pnl_untrusted'`` appears in ``tags`` (case-insensitive), OR
      * ``extra.pnl_untrusted`` is truthy, OR
      * the documented top-level ``payload.pnl_untrusted`` is truthy.
    """
    tags = payload.get("tags")
    if isinstance(tags, (list, tuple)):
        for tag in tags:
            if str(tag).strip().lower() == "pnl_untrusted":
                return True
    extra = payload.get("extra")
    if isinstance(extra, Mapping) and bool(extra.get("pnl_untrusted")):
        return True
    if bool(payload.get("pnl_untrusted")):
        return True
    return False


def _row_representative_time(payload: Mapping[str, Any]) -> Optional[datetime]:
    """Realised time for session bucketing: exit_time_utc, else entry_time_utc."""
    for key in ("exit_time_utc", "entry_time_utc"):
        dt = parse_iso_utc(payload.get(key))
        if dt is not None:
            return dt
    return None


def _row_is_post_anchor(payload: Mapping[str, Any], anchor: datetime) -> bool:
    """True if EITHER exit_time_utc OR entry_time_utc is >= anchor."""
    for key in ("exit_time_utc", "entry_time_utc"):
        dt = parse_iso_utc(payload.get(key))
        if dt is not None and dt >= anchor:
            return True
    return False


# --------------------------------------------------------------------------- #
# Trade-file discovery + streaming
# --------------------------------------------------------------------------- #
def iter_soak_trade_files(trades_dir: Path, anchor_day: date) -> List[Path]:
    """Newest-first daily ledgers dated on/after ``anchor_day``.

    Only exact ``trade_history_YYYYMMDD.ndjson`` files qualify; ``.scr_reset_bak``
    and other suffixed/aux files are excluded by construction.
    """
    anchor_ymd = anchor_day.strftime("%Y%m%d")
    out: List[Path] = []
    try:
        candidates = sorted(trades_dir.glob("trade_history_*.ndjson"), reverse=True)
    except OSError as exc:
        log.warning("cannot list trades dir %s: %s", trades_dir, exc)
        return out
    for path in candidates:
        m = TRADE_FILE_RE.match(path.name)
        if m is None:
            continue
        # 8-digit YYYYMMDD is zero-padded, so lexicographic == chronological.
        if m.group(1) >= anchor_ymd:
            out.append(path)
    return out


def _iter_ndjson_payloads(path: Path):
    """Yield ``payload`` dicts from an NDJSON ledger. Never raises on bad rows."""
    try:
        handle = path.open("r", encoding="utf-8", errors="ignore")
    except OSError as exc:
        log.warning("cannot open %s: %s", path, exc)
        return
    with handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            payload = record.get("payload")
            if isinstance(payload, dict):
                yield payload


def tally_diagnostics(
    files: Sequence[Path], anchor: datetime
) -> Tuple[Diagnostics, Set[date]]:
    """Independent, best-effort diagnostic tally over ``files``.

    Returns the :class:`Diagnostics` counters and the set of distinct UTC session
    days that contain at least one soak-effective trade.
    """
    rows_seen = 0
    rows_post_anchor = 0
    rows_zero_pnl = 0
    rows_untrusted = 0
    effective_tally = 0
    session_days: Set[date] = set()

    for path in files:
        for payload in _iter_ndjson_payloads(path):
            rows_seen += 1

            if not _row_is_post_anchor(payload, anchor):
                continue
            rows_post_anchor += 1

            untrusted = row_is_untrusted(payload)
            if untrusted:
                rows_untrusted += 1

            pnl = _finite_float(payload.get("pnl"))
            if pnl is not None and pnl == 0.0:
                rows_zero_pnl += 1

            is_live = bool(payload.get("is_live", False))

            # Soak-effective predicate (mirrors SCR semantics).
            if untrusted or is_live or pnl is None or pnl == 0.0:
                continue

            effective_tally += 1
            rep = _row_representative_time(payload)
            if rep is not None:
                session_days.add(rep.date())

    diag = Diagnostics(
        rows_seen=rows_seen,
        rows_post_anchor=rows_post_anchor,
        rows_zero_pnl=rows_zero_pnl,
        rows_untrusted=rows_untrusted,
        evaluator_effective_tally=effective_tally,
    )
    return diag, session_days


# --------------------------------------------------------------------------- #
# Authority gate input
# --------------------------------------------------------------------------- #
def read_authority(runtime_dir: Path) -> Tuple[str, bool, str]:
    """Return ``(status, is_green, reason)`` for the broker-authority gate.

    Fails closed: a missing / stale / non-GREEN ``positions_truth.json`` yields
    ``is_green=False``.
    """
    path = runtime_dir / POSITIONS_TRUTH_FILENAME
    obj, freshness = read_runtime_state_json(path)
    if obj is None:
        return "RED", False, f"positions_truth_{freshness.reason}"
    raw_status = str(obj.get("broker_authority_status") or "").strip().upper()
    if not freshness.ok:
        return "RED", False, f"positions_truth_stale:{freshness.reason}"
    if raw_status == "GREEN":
        return "GREEN", True, "broker_authority_green"
    return "RED", False, f"broker_authority_status={raw_status or 'MISSING'}"


# --------------------------------------------------------------------------- #
# Core evaluation
# --------------------------------------------------------------------------- #
def evaluate(
    *,
    runtime_dir: Optional[Path] = None,
    trades_dir: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Perform one clean-soak evaluation and return the state dict.

    Pure with respect to the filesystem: reads only; writes nothing. Robust to
    absent/stale/corrupt inputs — always returns a valid, gradable dict with a
    ``reasons`` list rather than raising.
    """
    runtime = runtime_dir if runtime_dir is not None else _default_runtime_dir()
    trades = trades_dir if trades_dir is not None else _default_trades_dir()
    criteria = load_criteria(env)
    now_dt = (now.astimezone(timezone.utc) if now is not None else utc_now())
    now_iso = now_dt.isoformat().replace("+00:00", "Z")

    reasons: List[str] = []

    # ---- SCR authoritative effective count (freshness-gated) ---------------- #
    scr_obj, scr_fresh = read_runtime_state_json(runtime / SCR_STATE_FILENAME)
    scr_ok = scr_obj is not None and scr_fresh.ok
    if scr_obj is None:
        reasons.append(f"scr_state_missing_or_corrupt:{scr_fresh.reason}")
    elif not scr_fresh.ok:
        reasons.append(
            f"scr_state_stale:{scr_fresh.reason}:age={scr_fresh.age_seconds:.0f}s"
        )
    scr_stats = {}
    if isinstance(scr_obj, dict):
        maybe_stats = scr_obj.get("stats")
        if isinstance(maybe_stats, dict):
            scr_stats = maybe_stats
    effective_trades = _coerce_int(scr_stats.get("effective_trades", 0), 0)

    # ---- Soak anchor from the epoch-reset state ----------------------------- #
    epoch_obj = read_json(runtime / EPOCH_RESET_FILENAME)
    if epoch_obj is None:
        reasons.append("epoch_reset_state_missing_or_corrupt")
    anchor_utc = ""
    epoch_label = ""
    if isinstance(epoch_obj, dict):
        anchor_utc = str(epoch_obj.get("completed_at_utc") or "")
        epoch_label = str(epoch_obj.get("target_epoch_label") or "")
    anchor_dt = parse_iso_utc(anchor_utc) if anchor_utc else None
    if isinstance(epoch_obj, dict) and anchor_dt is None:
        reasons.append(f"epoch_anchor_unparseable:{anchor_utc!r}")

    # ---- Independent diagnostic tally over post-anchor rows ----------------- #
    if anchor_dt is not None:
        files = iter_soak_trade_files(trades, anchor_dt.date())
        diag, session_days = tally_diagnostics(files, anchor_dt)
    else:
        diag = Diagnostics(0, 0, 0, 0, 0)
        session_days = set()
    sessions_count = len(session_days)

    untrusted_ratio = (
        (diag.rows_untrusted / diag.rows_post_anchor)
        if diag.rows_post_anchor > 0
        else 0.0
    )

    # ---- Authority gate ----------------------------------------------------- #
    authority_status, authority_green, authority_reason = read_authority(runtime)
    if not authority_green:
        reasons.append(authority_reason)

    # ---- Discrepancy tripwire ---------------------------------------------- #
    # Only meaningful when both inputs are trustworthy: SCR fresh + real anchor.
    data_ok = scr_ok and anchor_dt is not None
    discrepancy_flag = bool(
        data_ok and diag.evaluator_effective_tally != effective_trades
    )
    if discrepancy_flag:
        reasons.append(
            "effective_tally_discrepancy:"
            f"scr={effective_trades},evaluator={diag.evaluator_effective_tally}"
        )

    # ---- Gates -------------------------------------------------------------- #
    if data_ok:
        gates = {
            "authority_green": authority_green,
            "effective_met": effective_trades >= criteria.min_effective_trades,
            "sessions_met": sessions_count >= criteria.min_sessions,
            "untrusted_ok": untrusted_ratio <= criteria.max_untrusted_ratio,
            "no_discrepancy": not discrepancy_flag,
        }
    else:
        # Absent/stale SCR or unusable anchor: cannot grade — fail closed.
        gates = {key: False for key in GATE_KEYS}
        reasons.append("insufficient_state_data:gates_forced_false")

    gates_passed = sum(1 for value in gates.values() if value)
    soak_passed = gates_passed == len(GATE_KEYS)

    state: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "soak_anchor_utc": anchor_utc,
        "epoch_label": epoch_label,
        "authority_status": authority_status,
        "effective_trades": effective_trades,
        "sessions_with_effective": sessions_count,
        "untrusted_ratio": round(untrusted_ratio, 6),
        "diagnostics": {
            "rows_seen": diag.rows_seen,
            "rows_post_anchor": diag.rows_post_anchor,
            "rows_zero_pnl": diag.rows_zero_pnl,
            "rows_untrusted": diag.rows_untrusted,
            "evaluator_effective_tally": diag.evaluator_effective_tally,
        },
        "discrepancy_flag": discrepancy_flag,
        "criteria": {
            "min_effective_trades": criteria.min_effective_trades,
            "min_sessions": criteria.min_sessions,
            "max_untrusted_ratio": criteria.max_untrusted_ratio,
        },
        "gates": gates,
        "gates_passed": gates_passed,
        "soak_passed": soak_passed,
        "reasons": reasons,
        "ts_utc": now_iso,
        "ttl_seconds": OUTPUT_TTL_SECONDS,
    }
    return state


def write_state(state: Mapping[str, Any], out_path: Path) -> None:
    """Atomically write ``state`` to ``out_path`` via the canonical helper."""
    atomic_write_json(out_path, dict(state))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _one_line_summary(state: Mapping[str, Any], out_path: Optional[Path]) -> str:
    where = f" -> {out_path}" if out_path is not None else ""
    return (
        f"clean_soak{where}: {state['gates_passed']}/{len(GATE_KEYS)} gates, "
        f"soak_passed={state['soak_passed']} "
        f"effective={state['effective_trades']} "
        f"sessions={state['sessions_with_effective']} "
        f"untrusted_ratio={state['untrusted_ratio']} "
        f"discrepancy={state['discrepancy_flag']}"
    )


def _human_summary(state: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"schema_version   : {state['schema_version']}")
    lines.append(f"soak_anchor_utc  : {state['soak_anchor_utc'] or '<none>'}")
    lines.append(f"epoch_label      : {state['epoch_label'] or '<none>'}")
    lines.append(f"authority_status : {state['authority_status']}")
    lines.append(f"effective_trades : {state['effective_trades']} (SCR authoritative)")
    lines.append(f"sessions         : {state['sessions_with_effective']}")
    lines.append(f"untrusted_ratio  : {state['untrusted_ratio']}")
    lines.append(f"discrepancy_flag : {state['discrepancy_flag']}")
    diag = state["diagnostics"]
    lines.append(
        "diagnostics      : "
        f"rows_seen={diag['rows_seen']} "
        f"post_anchor={diag['rows_post_anchor']} "
        f"zero_pnl={diag['rows_zero_pnl']} "
        f"untrusted={diag['rows_untrusted']} "
        f"evaluator_effective_tally={diag['evaluator_effective_tally']}"
    )
    crit = state["criteria"]
    lines.append(
        "criteria         : "
        f"min_effective={crit['min_effective_trades']} "
        f"min_sessions={crit['min_sessions']} "
        f"max_untrusted_ratio={crit['max_untrusted_ratio']}"
    )
    gates = state["gates"]
    lines.append("gates            :")
    for key in GATE_KEYS:
        mark = "PASS" if gates.get(key) else "FAIL"
        lines.append(f"    [{mark}] {key}")
    lines.append(
        f"gates_passed     : {state['gates_passed']}/{len(GATE_KEYS)}  "
        f"=> soak_passed={state['soak_passed']}"
    )
    if state["reasons"]:
        lines.append("reasons          :")
        for reason in state["reasons"]:
            lines.append(f"    - {reason}")
    return "\n".join(lines)


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    runtime_dir: Optional[Path] = None,
    trades_dir: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m chad.ops.clean_soak_evaluator",
        description="GAP-039 clean-soak evaluator — grade the post-Epoch-3 SCR sample.",
    )
    parser.add_argument(
        "--print",
        dest="do_print",
        action="store_true",
        help="Evaluate and print a human summary WITHOUT writing the state file.",
    )
    parser.add_argument(
        "--json",
        dest="do_json",
        action="store_true",
        help="Evaluate and print machine JSON to stdout WITHOUT writing the file.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    runtime = runtime_dir if runtime_dir is not None else _default_runtime_dir()
    trades = trades_dir if trades_dir is not None else _default_trades_dir()
    target = out_path if out_path is not None else (runtime / OUTPUT_FILENAME)

    state = evaluate(runtime_dir=runtime, trades_dir=trades)

    if args.do_json:
        # Machine-readable path: pure stdout, no write, no log noise.
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0

    if args.do_print:
        # Dry run: human summary, no write.
        print(_human_summary(state))
        return 0

    # Default: atomically write the state file; exactly one summary line printed.
    write_state(state, target)
    log.info(
        "clean_soak_state written: %s (gates_passed=%d/%d soak_passed=%s)",
        target,
        state["gates_passed"],
        len(GATE_KEYS),
        state["soak_passed"],
    )
    print(_one_line_summary(state, target))
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    raise SystemExit(main())
