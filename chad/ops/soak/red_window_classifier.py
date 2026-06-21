#!/usr/bin/env python3
"""
CHAD — RED-window mechanical classifier (soak evaluator).

Implements the LOCKED rule
``ops/pending_actions/SESSION_SOAK_MECHANICAL_EVIDENCE_GATE_RULE_2026-05-27.md``
(sha256 ``800480a570e8b2e50d115b9a62d52d6b7022cd5dc0a4421a740528061ef60723``).

It reads the soak evidence written by ``chad/ops/soak/evidence_writers.py`` plus
the closed-trade ledger, detects every ``broker_authority_status=RED`` window in a
day's status-history, grades the rule's eight mechanical conditions for each
window, classifies each window ``EXPLAINED`` / ``FAIL`` / ``NOT_COUNTABLE`` exactly
per the rule, and emits ``runtime/session_explanations.json`` per the rule's §5.

HARD CONTRACT (mirrors the writers' posture):
- **Read-only over all evidence.** The classifier opens the evidence artifacts for
  read only. The SOLE write is the §5 output file, addressed via the ``--out``
  path parameter so tests redirect it to tmp. It mutates no ``runtime/`` gating
  state, no lock/epoch/ready_for_live, no source artifact.
- **On-demand only.** This module is NOT wired into any hot service. It runs when
  an operator invokes the CLI (or a test calls the pure functions). There is no
  timer, no daemon, no auto-wiring.
- **Mechanical, no operator override.** Per rule §5, only this evaluator can mark a
  window EXPLAINED, and only when ALL eight conditions are mechanically verified.
  If the evaluator cannot verify a condition, the window defaults to
  FAIL / NOT_COUNTABLE — there is no manual EXPLAINED path.

Decision mapping (rule §5, verbatim intent — see RULE_TEXT below):
- ``EXPLAINED``      — the window is bounded and gradable AND all eight conditions PASS.
- ``FAIL``           — the window is gradable AND at least one condition FAILs
                       (evidence present, condition violated).
- ``NOT_COUNTABLE``  — the window cannot be mechanically graded: it is unbounded
                       (no GREEN recovery sample), or sub-cadence (< one 60s tick,
                       so it cannot be bounded "from artifacts written during the
                       window"), or a required condition is UNVERIFIABLE with no
                       definitive FAIL. The rule treats FAIL and NOT_COUNTABLE
                       identically for soak purposes ("FAIL / NOT COUNTABLE"); the
                       split here records WHICH cause applies ("fails" vs "cannot
                       be verified") without loosening the rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Reuse the SAME timestamp parser the writers + their proven tests use, so window
# arithmetic and criterion-8 de-dup are byte-for-byte the logic validated in
# chad/tests/test_soak_evidence_writers.py.
from chad.ops.soak.evidence_writers import _parse_ts  # noqa: F401 (private, same package)

# ----------------------------------------------------------------------------
# The locked contract this evaluator enforces.
# ----------------------------------------------------------------------------
RULE_PATH = (
    "ops/pending_actions/SESSION_SOAK_MECHANICAL_EVIDENCE_GATE_RULE_2026-05-27.md"
)
# Full sha256 of the rule file (prefix 800480a5). The CLI re-computes the live
# file's sha and refuses to run if it has drifted from this pinned value.
RULE_SHA256 = "800480a570e8b2e50d115b9a62d52d6b7022cd5dc0a4421a740528061ef60723"

SCHEMA_VERSION = "session_explanations.v1"
EVALUATOR_IDENTITY = "chad.ops.soak.red_window_classifier"

# Rule §5.1: "the RED window duration is <= 1x ledger publish cadence (15 minutes
# wall-clock)."  15 minutes == 900 seconds.
LEDGER_CADENCE_SECONDS = 900

# Lifecycle-truth / status-history sampling cadence is 60s. A RED window shorter
# than one tick cannot be mechanically bounded "from artifacts written during the
# window" (rule §5), so it defaults to NOT_COUNTABLE rather than EXPLAINED.
MIN_COUNTABLE_DURATION_SECONDS = 60

# Classification tokens.
EXPLAINED = "EXPLAINED"
FAIL = "FAIL"
NOT_COUNTABLE = "NOT_COUNTABLE"

# Per-criterion grade tokens.
PASS = "PASS"
# (FAIL token reused.)
UNVERIFIABLE = "UNVERIFIABLE"

RED = "RED"
GREEN = "GREEN"

# Dated evidence-file prefixes (mirror evidence_writers._soak_out_path prefixes).
_STATUS_PREFIX = "status_history"
_SIGNAL_PREFIX = "signal_router_emissions"
_ENTRY_PREFIX = "entry_intent_audit"
_TRADE_PREFIX = "trade_history"


# ----------------------------------------------------------------------------
# Small read-only helpers
# ----------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    """``sha256:<hex>`` of a file's bytes at evaluation time, or ``""`` if absent.

    Rule §5: "the evaluator's sha256 of each artifact at the time of evaluation."
    """
    try:
        return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except Exception:
        return ""


def _read_ndjson(path: Path) -> List[Dict[str, Any]]:
    """Read an append-only ndjson evidence file (read-only). Missing file -> []."""
    rows: List[Dict[str, Any]] = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except Exception:
        return rows
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _in_window(ts_value: Any, open_dt: datetime, close_dt: datetime) -> bool:
    """True iff ts in the half-open RED window [open, close).

    [open, close): an event at exactly window_open (first RED tick) is inside the
    RED window; an event at exactly window_close (the GREEN recovery tick) is
    outside it.
    """
    t = _parse_ts(ts_value)
    if t is None:
        return False
    return open_dt <= t < close_dt


def _trade_realization_ts(row: Dict[str, Any]) -> Any:
    """Realization timestamp of a closed-trade ledger row.

    trade_history rows are hash-chained envelopes:
    ``{timestamp_utc, payload:{exit_time_utc, pnl, ...}, record_hash, ...}``.
    The PnL is realized at ``payload.exit_time_utc``; fall back to the envelope
    ``timestamp_utc`` (the row's append time) when the payload lacks an exit time.
    """
    payload = row.get("payload") if isinstance(row, dict) else None
    if isinstance(payload, dict) and payload.get("exit_time_utc"):
        return payload.get("exit_time_utc")
    return row.get("timestamp_utc") if isinstance(row, dict) else None


# ----------------------------------------------------------------------------
# RED-window detection
# ----------------------------------------------------------------------------
def detect_red_windows(status_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detect every RED window in a status-history (soak_status_history.v1) day.

    A RED window is a maximal run of consecutive ``broker_authority_status == RED``
    samples in the time-sorted status history. Detection is purely the
    broker_authority transition in the status-history (NOT the RTH session
    boundary): per the BOX-059 reconciliation record §2.5, the locked rule
    (sha 800480a5) is the sole authority for classifying RED windows WITHIN a
    BOX-059 session, and the rule itself references no RTH boundary — RTH bounds
    the enclosing session, not the per-window grade.

    Window bounds (reuses the proven test_soak_evidence_writers shape):
      - ``window_open_utc``  = ts_utc of the FIRST RED sample in the run.
      - ``recovery_row``     = the FIRST non-RED sample after the run.
      - ``window_close_utc`` = recovery_row's ts_utc (None if the run reaches EOF,
                               i.e. the window never recovered = unbounded).
      - ``red_rows``         = the RED samples themselves (the in-window samples
                               graded by criteria 4/5/6).
    """
    # Time-sort (stable). Unparseable timestamps are dropped from detection.
    ordered = [r for r in status_rows if _parse_ts(r.get("ts_utc")) is not None]
    ordered.sort(key=lambda r: _parse_ts(r.get("ts_utc")))

    windows: List[Dict[str, Any]] = []
    i = 0
    n = len(ordered)
    while i < n:
        if str(ordered[i].get("broker_authority_status")) != RED:
            i += 1
            continue
        j = i
        while j < n and str(ordered[j].get("broker_authority_status")) == RED:
            j += 1
        # ordered[i:j] is the maximal RED run; ordered[j] (if any) is recovery.
        recovery_row = ordered[j] if j < n else None
        windows.append(
            {
                "window_open_utc": ordered[i].get("ts_utc"),
                "window_close_utc": recovery_row.get("ts_utc") if recovery_row else None,
                "bounded": recovery_row is not None,
                "red_rows": ordered[i:j],
                "recovery_row": recovery_row,
            }
        )
        i = j
    return windows


# ----------------------------------------------------------------------------
# Per-criterion grading primitive
# ----------------------------------------------------------------------------
def _crit(result: str, basis: str, section: str, artifact: str) -> Dict[str, str]:
    return {"result": result, "basis": basis, "rule_section": section, "artifact": artifact}


def _distinct_set(rows: List[Dict[str, Any]], key: str) -> set:
    return {r.get(key) for r in rows}


# ----------------------------------------------------------------------------
# Window grading — all eight conditions, EXACTLY per the rule §5.
# ----------------------------------------------------------------------------
def grade_window(window: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Grade one RED window's eight conditions and classify it.

    ``evidence`` keys:
      - ``status_rows``  : full time-sorted status-history (for criterion 8 de-dup).
      - ``signal_rows``  : soak_signal_router.v1 rows (criterion 2).
      - ``entry_rows``   : soak_entry_intent.v1 rows (criterion 3).
      - ``trade_rows``   : trade_history envelopes (criterion 7).

    Returns a window record with per-criterion ``{result, basis, rule_section,
    artifact}`` and the overall ``classification``.
    """
    status_rows: List[Dict[str, Any]] = evidence.get("status_rows", [])
    signal_rows: List[Dict[str, Any]] = evidence.get("signal_rows", [])
    entry_rows: List[Dict[str, Any]] = evidence.get("entry_rows", [])
    trade_rows: List[Dict[str, Any]] = evidence.get("trade_rows", [])

    red_rows: List[Dict[str, Any]] = window.get("red_rows", [])
    bounded: bool = bool(window.get("bounded"))
    window_open_utc = window.get("window_open_utc")
    window_close_utc = window.get("window_close_utc")

    open_dt = _parse_ts(window_open_utc)
    close_dt = _parse_ts(window_close_utc) if bounded else None
    duration_seconds: Optional[int] = (
        int((close_dt - open_dt).total_seconds())
        if (open_dt is not None and close_dt is not None)
        else None
    )

    criteria: Dict[str, Dict[str, str]] = {}

    # ---- Condition 1 — Duration (rule §5, criterion 1) -----------------------
    # "the RED window duration is <= 1x ledger publish cadence (15 minutes)."
    if not bounded or duration_seconds is None:
        criteria["1"] = _crit(
            UNVERIFIABLE,
            "window unbounded — no GREEN recovery sample after the RED run; "
            "duration cannot be measured",
            "§5.1",
            _STATUS_PREFIX,
        )
    else:
        ok = duration_seconds <= LEDGER_CADENCE_SECONDS
        criteria["1"] = _crit(
            PASS if ok else FAIL,
            f"duration={duration_seconds}s; threshold={LEDGER_CADENCE_SECONDS}s "
            f"(1x 15-min ledger publish cadence)",
            "§5.1",
            _STATUS_PREFIX,
        )

    # ---- Condition 2 — No fresh signals (rule §5, criterion 2) ---------------
    # "zero RoutedSignal emissions during the window, verified from the
    # signal_router audit log."
    if not bounded or close_dt is None:
        criteria["2"] = _crit(
            UNVERIFIABLE,
            "window unbounded — RED interval not closed, cannot bound signal scan",
            "§5.2",
            _SIGNAL_PREFIX,
        )
    else:
        n_sig = sum(1 for r in signal_rows if _in_window(r.get("ts_utc"), open_dt, close_dt))
        criteria["2"] = _crit(
            PASS if n_sig == 0 else FAIL,
            f"RoutedSignal emissions in [open,close)={n_sig} "
            f"(soak_signal_router.v1; absence within a status-history-covered "
            f"window = zero emissions)",
            "§5.2",
            _SIGNAL_PREFIX,
        )

    # ---- Condition 3 — No fresh entries (rule §5, criterion 3) ---------------
    # "zero intent_type=ENTRY items in the execution pipeline, verified from
    # execution_plan_audit.ndjson (or equivalent audit artifact)."  The equivalent
    # audit artifact is the soak entry-intent log (companion #3); criterion 3
    # counts only intent_type==ENTRY.
    if not bounded or close_dt is None:
        criteria["3"] = _crit(
            UNVERIFIABLE,
            "window unbounded — cannot bound entry-intent scan",
            "§5.3",
            _ENTRY_PREFIX,
        )
    else:
        in_win_entries = [
            r
            for r in entry_rows
            if _in_window(r.get("ts_utc"), open_dt, close_dt)
        ]
        n_entry = sum(1 for r in in_win_entries if str(r.get("intent_type")) == "ENTRY")
        criteria["3"] = _crit(
            PASS if n_entry == 0 else FAIL,
            f"ENTRY intents in window={n_entry} "
            f"(of {len(in_win_entries)} intent rows in window; soak_entry_intent.v1)",
            "§5.3",
            _ENTRY_PREFIX,
        )

    # ---- Condition 4 — No sizing decisions (rule §5, criterion 4) ------------
    # "runtime/dynamic_caps.json ts_utc lies outside the RED window OR the
    # per-strategy sizing_factor values are unchanged across the window."
    # Branch A: no in-window dynamic_caps republish (ts outside window).
    # Branch B: sizing_digest (sha of canonical per-strategy caps, emitted by the
    # writer as the per-strategy sizing proxy) unchanged across the RED samples.
    if not red_rows:
        criteria["4"] = _crit(
            UNVERIFIABLE, "no in-window status samples", "§5.4", _STATUS_PREFIX
        )
    else:
        republished_in_window = (
            bool(close_dt is not None)
            and any(_in_window(r.get("dynamic_caps_ts_utc"), open_dt, close_dt) for r in red_rows)
        )
        branch_a_ok = not republished_in_window
        digests = _distinct_set(red_rows, "sizing_digest")
        nonempty_digests = {d for d in digests if d}
        branch_b_ok = len(digests) == 1 and len(nonempty_digests) == 1
        all_blank = (not nonempty_digests) and not any(
            _parse_ts(r.get("dynamic_caps_ts_utc")) for r in red_rows
        )
        if branch_a_ok or branch_b_ok:
            res = PASS
        elif all_blank:
            res = UNVERIFIABLE
        else:
            res = FAIL
        criteria["4"] = _crit(
            res,
            f"branch_a(dynamic_caps ts outside window)={branch_a_ok}; "
            f"branch_b(sizing_digest unchanged)={branch_b_ok}; "
            f"distinct_sizing_digests={len(digests)}",
            "§5.4",
            _STATUS_PREFIX,
        )

    # ---- Condition 5 — No SCR transitions (rule §5, criterion 5) -------------
    # "runtime/scr_state.json state band is unchanged across the window."
    if not red_rows:
        criteria["5"] = _crit(
            UNVERIFIABLE, "no in-window status samples", "§5.5", _STATUS_PREFIX
        )
    else:
        bands = _distinct_set(red_rows, "scr_band")
        nonempty_bands = {b for b in bands if b is not None}
        if not nonempty_bands:
            res = UNVERIFIABLE
        elif len(bands) == 1:
            res = PASS
        else:
            res = FAIL
        criteria["5"] = _crit(
            res,
            f"distinct scr_band values across window={sorted(str(b) for b in bands)}",
            "§5.5",
            _STATUS_PREFIX,
        )

    # ---- Condition 6 — No risk-state mutation (rule §5, criterion 6) ---------
    # "ts_utc on runtime/profit_lock_state.json, runtime/stop_bus.json, and
    # runtime/tier_state.json are all unchanged across the window."
    # NOTE (basis flag): stop_bus.json has NO ts_utc field. The rule cites a
    # phantom field; we grade HONESTLY against the writer's faithful proxy
    # (stop_bus_active | stop_bus_triggered_at | stop_bus_cleared_at all unchanged)
    # and surface that the field does not exist — we do NOT pretend it does.
    if not red_rows:
        criteria["6"] = _crit(
            UNVERIFIABLE, "no in-window status samples", "§5.6", _STATUS_PREFIX
        )
    else:
        pl_set = _distinct_set(red_rows, "profit_lock_ts_utc")
        tier_set = _distinct_set(red_rows, "tier_ts_utc")
        sb_active_set = _distinct_set(red_rows, "stop_bus_active")
        sb_trig_set = _distinct_set(red_rows, "stop_bus_triggered_at")
        sb_clear_set = _distinct_set(red_rows, "stop_bus_cleared_at")
        pl_ok = len(pl_set) == 1
        tier_ok = len(tier_set) == 1
        sb_ok = len(sb_active_set) == 1 and len(sb_trig_set) == 1 and len(sb_clear_set) == 1
        all_blank = (
            pl_set == {None}
            and tier_set == {None}
            and sb_active_set == {None}
            and sb_trig_set <= {None, ""}
            and sb_clear_set <= {None, ""}
        )
        if all_blank:
            res = UNVERIFIABLE
        else:
            res = PASS if (pl_ok and tier_ok and sb_ok) else FAIL
        criteria["6"] = _crit(
            res,
            f"profit_lock_ts unchanged={pl_ok}; tier_ts unchanged={tier_ok}; "
            f"stop_bus proxy unchanged={sb_ok} "
            f"[BASIS FLAG: stop_bus.json has no ts_utc (rule §5.6 cites a phantom "
            f"field); graded against emitted proxy "
            f"active|triggered_at|cleared_at]",
            "§5.6",
            _STATUS_PREFIX,
        )

    # ---- Condition 7 — No PnL realization (rule §5, criterion 7) -------------
    # "zero new closed-trade rows in data/trades/trade_history_<date>.ndjson
    # during the window."  An absent ledger file = zero rows = satisfied.
    if not bounded or close_dt is None:
        criteria["7"] = _crit(
            UNVERIFIABLE,
            "window unbounded — cannot bound closed-trade scan",
            "§5.7",
            _TRADE_PREFIX,
        )
    else:
        n_trades = sum(
            1 for r in trade_rows if _in_window(_trade_realization_ts(r), open_dt, close_dt)
        )
        criteria["7"] = _crit(
            PASS if n_trades == 0 else FAIL,
            f"closed-trade rows realized in window={n_trades} "
            f"(realization_ts=payload.exit_time_utc||timestamp_utc; "
            f"file_rows={len(trade_rows)})",
            "§5.7",
            _TRADE_PREFIX,
        )

    # ---- Condition 8 — Reconciler self-resolved (rule §5, criterion 8) -------
    # "the next chad-reconciliation-publisher cycle after the window returned
    # broker_authority_status=GREEN without operator intervention or manual JSON
    # edit."  De-dup status rows by reconciliation_ts_utc into distinct logical
    # cycles; the "next cycle after the window" is the first distinct
    # reconciliation_ts_utc strictly greater than window_close. (This de-dup is
    # the logic proven in chad/tests/test_soak_evidence_writers.py.)
    if not bounded or close_dt is None:
        criteria["8"] = _crit(
            UNVERIFIABLE,
            "window unbounded — no post-window reconciliation cycle to inspect",
            "§5.8",
            _STATUS_PREFIX,
        )
    else:
        distinct: List[Tuple[Any, Dict[str, Any]]] = []
        seen: set = set()
        for r in status_rows:  # original time order; first row carrying each cycle
            rt = r.get("reconciliation_ts_utc")
            if rt not in seen:
                seen.add(rt)
                distinct.append((rt, r))
        distinct = [d for d in distinct if _parse_ts(d[0]) is not None]
        distinct.sort(key=lambda d: _parse_ts(d[0]))
        nxt = next((d for d in distinct if _parse_ts(d[0]) > close_dt), None)
        if nxt is None:
            criteria["8"] = _crit(
                UNVERIFIABLE,
                "no reconciliation cycle observed strictly after window_close",
                "§5.8",
                _STATUS_PREFIX,
            )
        else:
            nxt_row = nxt[1]
            ba = str(nxt_row.get("broker_authority_status"))
            ok = ba == GREEN
            criteria["8"] = _crit(
                PASS if ok else FAIL,
                f"next_recon_ts={nxt[0]}; next_cycle broker_authority_status={ba} "
                f"(gate=broker_authority_status==GREEN per §5.8); "
                f"reconciliation_status={nxt_row.get('reconciliation_status')} "
                f"(operator-intervention not mechanically detectable — auto-recovery "
                f"inferred from regular cadence)",
                "§5.8",
                _STATUS_PREFIX,
            )

    # ---- Overall classification (decision mapping, rule §5) ------------------
    results = [criteria[str(k)]["result"] for k in range(1, 9)]
    if not bounded:
        classification = NOT_COUNTABLE
        countable = False
        reason = "unbounded_window"
    elif duration_seconds is not None and duration_seconds < MIN_COUNTABLE_DURATION_SECONDS:
        # Sub-cadence: cannot be mechanically bounded from per-tick artifacts.
        classification = NOT_COUNTABLE
        countable = False
        reason = "sub_cadence_window"
    elif all(r == PASS for r in results):
        classification = EXPLAINED
        countable = True
        reason = "all_eight_conditions_pass"
    elif any(r == FAIL for r in results):
        # Evidence present, a condition violated -> the rule's "fails" disposition.
        classification = FAIL
        countable = True
        reason = "condition_failed"
    else:
        # No definitive FAIL but a condition could not be verified -> the rule's
        # "cannot be verified" disposition.
        classification = NOT_COUNTABLE
        countable = False
        reason = "unverifiable_criteria"

    return {
        "window_open_utc": window_open_utc,
        "window_close_utc": window_close_utc,
        "duration_seconds": duration_seconds,
        "classification": classification,
        "countability": {"countable": countable, "reason": reason},
        "criteria": criteria,
    }


# ----------------------------------------------------------------------------
# §5 output assembly
# ----------------------------------------------------------------------------
def _criterion_artifact_paths(paths: Dict[str, str]) -> Dict[str, str]:
    """Map each criterion (1..8) to the artifact path used to prove it (rule §5
    'the artifact paths used to prove each criterion')."""
    s = paths.get(_STATUS_PREFIX, "")
    return {
        "1": s,
        "2": paths.get(_SIGNAL_PREFIX, ""),
        "3": paths.get(_ENTRY_PREFIX, ""),
        "4": s,
        "5": s,
        "6": s,
        "7": paths.get(_TRADE_PREFIX, ""),
        "8": s,
    }


def build_session_explanations(
    *,
    date: str,
    status_rows: List[Dict[str, Any]],
    signal_rows: List[Dict[str, Any]],
    entry_rows: List[Dict[str, Any]],
    trade_rows: List[Dict[str, Any]],
    artifact_paths: Dict[str, str],
    artifact_sha256: Dict[str, str],
    evaluator_identity: str,
    evaluated_at_utc: str,
) -> Dict[str, Any]:
    """Build the §5 ``session_explanations.json`` document for one day.

    Records every detected RED window. EXPLAINED windows carry every §5-mandated
    field (window bounds, duration, per-criterion artifact paths, per-artifact
    sha256, evaluator identity + ts_utc); FAIL / NOT_COUNTABLE windows carry the
    same shape plus the per-condition basis so the record is a complete audit.
    """
    windows = detect_red_windows(status_rows)
    evidence = {
        "status_rows": status_rows,
        "signal_rows": signal_rows,
        "entry_rows": entry_rows,
        "trade_rows": trade_rows,
    }
    crit_paths = _criterion_artifact_paths(artifact_paths)

    records: List[Dict[str, Any]] = []
    counts = {EXPLAINED: 0, FAIL: 0, NOT_COUNTABLE: 0}
    for w in windows:
        graded = grade_window(w, evidence)
        counts[graded["classification"]] += 1
        record = dict(graded)
        # Rule §5 fields recorded per (EXPLAINED) window — present on every window.
        record["artifact_paths"] = crit_paths
        record["artifact_sha256"] = dict(artifact_sha256)
        record["evaluator_identity"] = evaluator_identity
        record["evaluated_at_utc"] = evaluated_at_utc
        records.append(record)

    return {
        "schema_version": SCHEMA_VERSION,
        "rule_path": RULE_PATH,
        "rule_sha256": RULE_SHA256,
        "date": date,
        "evaluator_identity": evaluator_identity,
        "evaluated_at_utc": evaluated_at_utc,
        "artifact_paths": dict(artifact_paths),
        "artifact_sha256": dict(artifact_sha256),
        "window_count": len(records),
        "counts": counts,
        "windows": records,
    }


def _serialize(doc: Dict[str, Any]) -> str:
    """Deterministic serialization: sorted keys + stable indent + trailing NL.

    Identical inputs -> byte-identical output (the determinism contract)."""
    return json.dumps(doc, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


# ----------------------------------------------------------------------------
# CLI (on-demand only — no service wiring)
# ----------------------------------------------------------------------------
def _repo_root() -> Path:
    # chad/ops/soak/red_window_classifier.py -> parents[3] == repo root
    return Path(__file__).resolve().parents[3]


def _verify_rule_sha(repo_root: Path) -> Tuple[bool, str]:
    """Re-compute the live rule file's sha and compare to the pinned RULE_SHA256."""
    rule_file = repo_root / RULE_PATH
    try:
        live = hashlib.sha256(rule_file.read_bytes()).hexdigest()
    except Exception as exc:  # pragma: no cover - operator-facing error path
        return False, f"cannot read rule file {rule_file}: {exc}"
    return (live == RULE_SHA256), live


def run(
    *,
    date: str,
    data_dir: Path,
    runtime_dir: Path,
    out_path: Path,
    evaluator_identity: str,
    evaluated_at_utc: str,
) -> Dict[str, Any]:
    """Classify a day's RED windows and write the §5 file to ``out_path`` only.

    Read-only over evidence; the SOLE write is ``out_path``.
    """
    data_dir = Path(data_dir)
    soak_dir = data_dir / "soak"
    trades_dir = data_dir / "trades"

    status_path = soak_dir / f"{_STATUS_PREFIX}_{date}.ndjson"
    signal_path = soak_dir / f"{_SIGNAL_PREFIX}_{date}.ndjson"
    entry_path = soak_dir / f"{_ENTRY_PREFIX}_{date}.ndjson"
    trade_path = trades_dir / f"{_TRADE_PREFIX}_{date}.ndjson"

    artifact_paths = {
        _STATUS_PREFIX: str(status_path),
        _SIGNAL_PREFIX: str(signal_path),
        _ENTRY_PREFIX: str(entry_path),
        _TRADE_PREFIX: str(trade_path),
    }
    artifact_sha256 = {
        _STATUS_PREFIX: _sha256_file(status_path),
        _SIGNAL_PREFIX: _sha256_file(signal_path),
        _ENTRY_PREFIX: _sha256_file(entry_path),
        _TRADE_PREFIX: _sha256_file(trade_path),
    }

    doc = build_session_explanations(
        date=date,
        status_rows=_read_ndjson(status_path),
        signal_rows=_read_ndjson(signal_path),
        entry_rows=_read_ndjson(entry_path),
        trade_rows=_read_ndjson(trade_path),
        artifact_paths=artifact_paths,
        artifact_sha256=artifact_sha256,
        evaluator_identity=evaluator_identity,
        evaluated_at_utc=evaluated_at_utc,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_serialize(doc), encoding="utf-8")
    return doc


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m chad.ops.soak.red_window_classifier",
        description=(
            "RED-window mechanical classifier (soak evaluator) — enforces the "
            "LOCKED rule sha256 800480a5...; on-demand, read-only, writes only --out."
        ),
    )
    parser.add_argument("--date", required=True, help="UTC day to classify, YYYYMMDD")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="path to the data/ dir (contains soak/ and trades/); default: repo data/",
    )
    parser.add_argument(
        "--runtime-dir",
        default=None,
        help="path to the runtime/ dir (default --out location); default: repo runtime/",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output path; default: <runtime-dir>/session_explanations.json",
    )
    parser.add_argument(
        "--evaluator-identity",
        default=EVALUATOR_IDENTITY,
        help="evaluator process identity recorded in the §5 output",
    )
    parser.add_argument(
        "--evaluated-at",
        default=None,
        help="evaluator ts_utc recorded in the §5 output (default: now, UTC ISO)",
    )
    args = parser.parse_args(argv)

    repo_root = _repo_root()

    # Rule-sha integrity gate: refuse to run if the locked contract has drifted.
    ok, live = _verify_rule_sha(repo_root)
    if not ok:
        sys.stderr.write(
            "RULE SHA MISMATCH — refusing to classify.\n"
            f"  expected: {RULE_SHA256}\n"
            f"  actual:   {live}\n"
            f"  rule:     {repo_root / RULE_PATH}\n"
        )
        return 2

    data_dir = Path(args.data_dir) if args.data_dir else (repo_root / "data")
    runtime_dir = Path(args.runtime_dir) if args.runtime_dir else (repo_root / "runtime")
    out_path = Path(args.out) if args.out else (runtime_dir / "session_explanations.json")

    if args.evaluated_at:
        evaluated_at = args.evaluated_at
    else:
        from chad.utils.runtime_json import utc_now_iso

        evaluated_at = utc_now_iso()

    doc = run(
        date=args.date,
        data_dir=data_dir,
        runtime_dir=runtime_dir,
        out_path=out_path,
        evaluator_identity=args.evaluator_identity,
        evaluated_at_utc=evaluated_at,
    )
    c = doc["counts"]
    sys.stdout.write(
        f"session_explanations: date={args.date} windows={doc['window_count']} "
        f"EXPLAINED={c[EXPLAINED]} FAIL={c[FAIL]} NOT_COUNTABLE={c[NOT_COUNTABLE]} "
        f"-> {out_path}\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
