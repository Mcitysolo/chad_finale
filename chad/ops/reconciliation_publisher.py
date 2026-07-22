#!/usr/bin/env python3
"""
CHAD Reconciliation Publisher — scheduled CLI wrapper.

Compares CHAD's position_guard.json against IBKR broker truth
(connected with clientId=83, readonly) and writes
runtime/reconciliation_state.json.

Status rules:
  GREEN  — every open CHAD position matches broker within 1 unit
  YELLOW — minor discrepancy (within 2 units) OR broker has no-guard symbols
  RED    — major discrepancy OR IBKR unavailable

Fail-soft: always writes an output file. Never raises.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

LOG = logging.getLogger("chad.ops.reconciliation_publisher")

RUNTIME_DIR = Path("/home/ubuntu/chad_finale/runtime")
GUARD_PATH = RUNTIME_DIR / "position_guard.json"
OUT_PATH = RUNTIME_DIR / "reconciliation_state.json"
DRIFT_OUT_PATH = RUNTIME_DIR / "position_guard_drift.json"
# W1A-5 / D2: the CHAD_DRIFT_V4 independent-leg view writes a SIBLING file only.
# It never touches DRIFT_OUT_PATH and never feeds the live-readiness RED gate.
SNAPSHOT_PATH = RUNTIME_DIR / "positions_snapshot.json"
DRIFT_V4_OUT_PATH = RUNTIME_DIR / "position_guard_drift_v4.json"

IBKR_HOST = "127.0.0.1"
IBKR_PORT = 4002
IBKR_CLIENT_ID = 83
IBKR_TIMEOUT_SEC = 15
TTL_SECONDS = 360

# Pre-existing paper account positions not opened by CHAD. Excluded
# from the mismatch check so they do not flip status to RED.
try:
    from chad.core.position_reconciler import KNOWN_NON_CHAD_SYMBOLS as _RECONCILER_NON_CHAD  # type: ignore
except Exception:  # noqa: BLE001
    _RECONCILER_NON_CHAD = frozenset({"AAPL", "MSFT"})

_EXCLUSIONS_CONFIG_PATH = Path("/home/ubuntu/chad_finale/config/reconciliation_exclusions.json")

# Hardcoded fallback used only when the JSON config is missing or unreadable.
# Source of truth is config/reconciliation_exclusions.json.
_FALLBACK_BROKER_PREEXISTING = frozenset({"NVDA"})
_FALLBACK_EXCLUSION_POLICY: Dict[str, Dict[str, Any]] = {
    "AAPL": {
        "reason": "pre-existing broker position",
        "owner": "operator",
        "added_utc": "2026-04-01",
        "expires_utc": None,
        "reviewed_utc": "2026-05-03",
    },
    "MSFT": {
        "reason": "pre-existing broker position",
        "owner": "operator",
        "added_utc": "2026-04-01",
        "expires_utc": None,
        "reviewed_utc": "2026-05-03",
    },
    "NVDA": {
        "reason": "pre-existing broker position",
        "owner": "operator",
        "added_utc": "2026-04-01",
        "expires_utc": None,
        "reviewed_utc": "2026-05-03",
    },
}


def _load_exclusion_config() -> Dict[str, Any]:
    """Load reconciliation exclusion policy from config JSON.

    Returns dict with broker_preexisting (frozenset) and exclusion_policy
    (dict). Falls back to hardcoded values if the file is missing or
    malformed so reconciliation never crashes on a config error.
    """
    try:
        raw = json.loads(_EXCLUSIONS_CONFIG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "broker_preexisting": _FALLBACK_BROKER_PREEXISTING,
            "exclusion_policy": dict(_FALLBACK_EXCLUSION_POLICY),
        }
    except Exception as exc:  # noqa: BLE001
        LOG.warning(
            "reconciliation_exclusions.json unreadable (%s) — falling back to hardcoded policy",
            exc,
        )
        return {
            "broker_preexisting": _FALLBACK_BROKER_PREEXISTING,
            "exclusion_policy": dict(_FALLBACK_EXCLUSION_POLICY),
        }
    bp = raw.get("broker_preexisting_symbols") or []
    pol = raw.get("exclusion_policy") or {}
    if not isinstance(bp, list) or not isinstance(pol, dict):
        return {
            "broker_preexisting": _FALLBACK_BROKER_PREEXISTING,
            "exclusion_policy": dict(_FALLBACK_EXCLUSION_POLICY),
        }
    return {
        "broker_preexisting": frozenset(str(s) for s in bp),
        "exclusion_policy": pol,
    }


_EXCLUSION_CFG = _load_exclusion_config()

# Publisher-only augmentation: symbols present at broker as pre-existing
# paper positions that CHAD never opened. Kept separate from the
# position_reconciler set so CHAD can still auto-close its own future
# positions on these symbols via thesis-flip reconciliation.
_BROKER_PREEXISTING: frozenset = _EXCLUSION_CFG["broker_preexisting"]
KNOWN_NON_CHAD_SYMBOLS = _RECONCILER_NON_CHAD | _BROKER_PREEXISTING

# DS08: bounded exclusion policy. Every symbol skipped from reconciliation
# carries reason/owner/added/expires/reviewed metadata so excluded items
# stay auditable instead of becoming permanent blind spots.
EXCLUSION_POLICY: Dict[str, Dict[str, Any]] = _EXCLUSION_CFG["exclusion_policy"]

# Futures symbols whose IBKR positions cannot be reliably reconciled
# without explicit contract_month resolution (ISSUE-29 companion). Any
# diff on these is skipped rather than flagged as a mismatch until the
# futures contract-resolution path is complete. MCL surfaced in retry-3
# as chad=2 vs broker=0 (ghost position from ContractResolutionError in
# ibkr_adapter). Do not flip status RED for futures-reconciliation gaps.
KNOWN_FUTURES_SYMBOLS = frozenset({"MCL", "ES", "NQ", "CL", "GC", "RTY", "MES", "MNQ", "MYM", "M2K"})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_guard_positions() -> Dict[str, float]:
    """Aggregate open CHAD positions by symbol (signed quantity).

    In paper/DRY_RUN mode, only broker_sync positions are reconciled
    against IBKR — strategy positions are never submitted to the broker
    in paper mode and will always appear as mismatches if included.
    """
    from chad.execution.execution_config import is_paper_mode
    is_paper = is_paper_mode()

    if not GUARD_PATH.exists():
        return {}
    try:
        raw = json.loads(GUARD_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    agg: Dict[str, float] = defaultdict(float)
    for entry in raw.values():
        if not isinstance(entry, dict) or not entry.get("open"):
            continue
        sym = entry.get("symbol")
        qty = entry.get("quantity", 0) or 0
        side = str(entry.get("side", "")).upper()
        strategy = str(entry.get("strategy", "")).lower()
        if not sym:
            continue
        # In paper mode: only reconcile broker_sync (real IBKR positions)
        if is_paper and strategy != "broker_sync":
            continue
        signed = -abs(float(qty)) if side in ("SELL", "SHORT") else abs(float(qty))
        agg[sym] += signed
    return dict(agg)


def _load_guard_breakdown() -> Dict[str, Dict[str, float]]:
    """Per-symbol breakdown: broker_sync contribution vs strategy contribution.

    Used by drift detection — a diff on a symbol whose chad-side is entirely
    from broker_sync (no strategy attribution) indicates IBKR moved without
    CHAD initiating the change. Treated as drift, not mismatch.
    """
    if not GUARD_PATH.exists():
        return {}
    try:
        raw = json.loads(GUARD_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    from chad.execution.execution_config import is_paper_mode
    is_paper = is_paper_mode()

    agg: Dict[str, Dict[str, float]] = {}
    for entry in raw.values():
        if not isinstance(entry, dict) or not entry.get("open"):
            continue
        sym = entry.get("symbol")
        if not sym:
            continue
        qty = entry.get("quantity", 0) or 0
        side = str(entry.get("side", "")).upper()
        strategy = str(entry.get("strategy", "")).lower()
        # In paper mode: only include broker_sync in breakdown so
        # broker_sync diffs are correctly classified as drift not mismatch
        if is_paper and strategy != "broker_sync":
            continue
        signed = -abs(float(qty)) if side in ("SELL", "SHORT") else abs(float(qty))
        bucket = agg.setdefault(sym, {"broker_sync": 0.0, "strategies": 0.0})
        if strategy == "broker_sync":
            bucket["broker_sync"] += signed
        else:
            bucket["strategies"] += signed
    return agg


def _load_broker_positions() -> Dict[str, float]:
    """Connect to IBKR with clientId=83 and return signed positions by symbol."""
    from ib_async import IB

    ib = IB()
    try:
        ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID,
                   readonly=True, timeout=IBKR_TIMEOUT_SEC)
        positions = ib.positions()
        out: Dict[str, float] = defaultdict(float)
        for p in positions:
            sym = getattr(p.contract, "symbol", None)
            if not sym:
                continue
            out[sym] += float(p.position)
        return dict(out)
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


def _write(payload: Dict[str, Any]) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    payload["ts_utc"] = _utc_now_iso()
    payload["ttl_seconds"] = TTL_SECONDS
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(OUT_PATH)


def _emit_position_guard_drift() -> int:
    """GAP-028 §5.1: emit per-strategy ↔ broker_sync drift findings.

    Pure read-only: invokes detect_guard_vs_broker_truth_drift() against the
    current on-disk position_guard.json and writes the v1 advisory file at
    DRIFT_OUT_PATH. No mutation of position_guard.json or trade_closer_state.json.
    Returns the number of drift records emitted.
    """
    try:
        # WKF U3: v2 drift semantics — like-with-like symbol totals, broker truth
        # read from the recorded broker_sync quantity (NOT gated on `open`), and
        # a single atomic read of position_guard.json so guard + broker truth
        # share one `_version` generation (kills the stale-snapshot false positive).
        from chad.core.position_guard import detect_guard_vs_broker_drift_v2
    except Exception as exc:  # noqa: BLE001
        LOG.warning("position_guard import failed for drift emit: %s", exc)
        return 0

    if GUARD_PATH.is_file():
        try:
            guard_state = json.loads(GUARD_PATH.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            LOG.warning("position_guard.json unreadable for drift emit: %s", exc)
            guard_state = {}
    else:
        guard_state = {}

    # A5: operator-owned symbols (exclusion policy + broker-preexisting set).
    # Drift on these is reported as mixed_ownership_info, never actionable
    # qty_mismatch/broker_untracked, so the operator's pre-existing holdings
    # (BAC/SPY/LLY/MSFT ...) do not inflate drift_count or flip live-readiness
    # RED. operator_baselines is sourced from any signed baseline recorded in
    # the exclusion policy (none today -> deltas reported as unattributable).
    excluded_symbols = {
        str(s).strip().upper()
        for s in (set(EXCLUSION_POLICY.keys()) | set(_BROKER_PREEXISTING))
        if s
    }
    operator_baselines = {
        str(sym).strip().upper(): float(meta["operator_baseline_qty"])
        for sym, meta in EXCLUSION_POLICY.items()
        if isinstance(meta, dict) and meta.get("operator_baseline_qty") is not None
    }

    result = (
        detect_guard_vs_broker_drift_v2(
            guard_state,
            excluded_symbols=excluded_symbols,
            operator_baselines=operator_baselines,
        )
        if isinstance(guard_state, dict)
        else {"drift_count": 0, "info_count": 0, "drifts": [],
              "snapshot_generation": None, "counts_by_kind": {}}
    )
    drifts = result.get("drifts", [])
    drift_count = int(result.get("drift_count", len(drifts)) or 0)
    payload = {
        # v3: adds mixed_ownership_info drift_kind (A5) for operator-owned
        # symbols; drift_count now counts ACTIONABLE drift only (phantom /
        # broker_untracked / qty_mismatch) and remains the authoritative
        # live-readiness gate (GAP-041 / PR-09). info_count exposes the
        # informational mixed-ownership records. ops.live_readiness_publish
        # accepts v1/v2/v3.
        "schema_version": "position_guard_drift.v3",
        "ts_utc": _utc_now_iso(),
        "ttl_seconds": TTL_SECONDS,
        "drift_count": drift_count,
        "info_count": int(result.get("info_count", 0) or 0),
        "excluded_symbols": sorted(excluded_symbols),
        "snapshot_generation": result.get("snapshot_generation"),
        "counts_by_kind": result.get("counts_by_kind", {}),
        "drifts": drifts,
    }
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    tmp = DRIFT_OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(DRIFT_OUT_PATH)
    LOG.info(
        "position_guard_drift emitted schema=v3 drift_count=%d info_count=%d by_kind=%s path=%s",
        drift_count, payload["info_count"], payload["counts_by_kind"], DRIFT_OUT_PATH,
    )
    # W3B-4: drift-content transition alerting (appeared/resolved). The helper
    # never raises; this belt-and-braces guard keeps reconciliation unblockable.
    try:
        _maybe_alert_drift_transitions(payload)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("drift transition alerting failed (non-blocking): %s", exc)
    return drift_count


# ---------------------------------------------------------------------------
# W3B-4: drift-content transition alerting
#
# Before this, a drift-only condition (reconciliation GREEN + drift_count>0)
# flipped live-readiness RED with zero notification -- the publisher pages on
# reconciliation_state RED (:478/:542) but nothing anywhere read the drift
# file's CONTENT. This alerts on TRANSITIONS of the actionable drift set
# (appeared / resolved), write-site pattern like check_and_send_scr_milestone.
# ---------------------------------------------------------------------------

DRIFT_ALERT_STATE_NAME = "position_guard_drift_last_alerted.json"

# Actionable kinds only. mixed_ownership_info is EXCLUDED BY DESIGN: A5
# (b27890a) reclassified operator-owned symbols out of drift_count precisely so
# they never page; alerting on them here would resurrect that false-positive
# class. Pinned by test_mixed_ownership_info_can_never_alert.
_DRIFT_ALERT_KINDS = frozenset(
    {"phantom_guard_entry", "broker_untracked_position", "qty_mismatch"}
)


def _drift_identity_set(payload: Dict[str, Any]) -> set:
    """Stable identities for the actionable drift records: ``kind|SYMBOL``.

    CTF-T2 rule: values belong in evidence, never in identity. Quantities,
    deltas, snapshot_generation and guard_keys all fluctuate cycle-to-cycle;
    a qty_mismatch whose delta drifts 5 -> 7 is the SAME condition and must
    not re-alert.
    """
    out = set()
    for d in payload.get("drifts") or []:
        if not isinstance(d, dict):
            continue
        kind = str(d.get("drift_kind") or "").strip()
        if kind not in _DRIFT_ALERT_KINDS:
            continue
        symbol = str(d.get("symbol") or "").strip().upper()
        out.add(f"{kind}|{symbol}")
    return out


def _drift_set_dedupe_suffix(identities) -> str:
    """Short stable digest of an identity set for the Telegram dedupe key.

    A changed set mints a new key, so a genuinely new transition is never
    TTL-suppressed; the transition state file remains the primary gate and
    the 900s dedupe TTL is belt-and-braces against a crash-loop re-sending.
    """
    import hashlib

    joined = "\n".join(sorted(str(i) for i in identities))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def _coach_drift_message(kind: str, facts: Dict[str, Any], fallback: str) -> str:
    """Coach-voiced rendering with plain-text fallback (presentation-only)."""
    try:
        from chad.utils.coach_voice import format_alert

        text = format_alert(kind, facts)
        if isinstance(text, str) and text.strip():
            return text
    except Exception as exc:  # noqa: BLE001 — presentation must never block
        LOG.warning("coach_voice unavailable for %s: %s", kind, exc)
    return fallback


def _maybe_alert_drift_transitions(
    payload: Dict[str, Any],
    *,
    state_path: "Path | None" = None,
    notify_fn=None,
) -> Dict[str, Any]:
    """Alert on appeared/resolved transitions of the actionable drift set.

    Never raises. State-file semantics: the last-alerted set is only advanced
    when every attempted send was delivered (SENT) or suppressed by the
    Telegram dedupe (an identical message just went out) -- a transport
    failure leaves the state untouched so the next 5-min cycle retries.
    """
    disposition: Dict[str, Any] = {
        "appeared": [], "resolved": [], "sent": [], "state_advanced": False,
    }
    try:
        # Test-leak guard (mirrors ExterminatorSentinel's explicit-reports_dir
        # rule): under pytest both the state path and the notifier must be
        # injected, so an existing test that calls _emit_position_guard_drift
        # can never write real runtime state or attempt a real Telegram send.
        if "pytest" in sys.modules and (state_path is None or notify_fn is None):
            disposition["skipped"] = "pytest_requires_explicit_injection"
            return disposition
        # Resolved at call time from the module attribute so tests that
        # monkeypatch RUNTIME_DIR redirect this file with everything else.
        state_path = state_path or (RUNTIME_DIR / DRIFT_ALERT_STATE_NAME)
        current = _drift_identity_set(payload)

        last: set = set()
        try:
            raw = json.loads(Path(state_path).read_text(encoding="utf-8"))
            last = {str(x) for x in raw.get("identities") or []}
        except Exception:  # noqa: BLE001 — first run / corrupt state
            last = set()

        appeared = sorted(current - last)
        resolved = sorted(last - current)
        disposition["appeared"] = appeared
        disposition["resolved"] = resolved

        if notify_fn is None:
            from chad.utils.telegram_notify import notify_detailed as notify_fn  # noqa: PLW2901

        ok_statuses = {"sent", "suppressed_dedupe"}

        def _delivered(outcome: Any) -> bool:
            status = getattr(getattr(outcome, "status", None), "value", None)
            if status is None:
                status = str(getattr(outcome, "status", outcome))
            return str(status).strip().lower() in ok_statuses

        all_delivered = True
        if appeared:
            facts = {
                "appeared": appeared,
                "counts_by_kind": payload.get("counts_by_kind") or {},
                "drift_count": payload.get("drift_count"),
            }
            fallback = (
                "position drift appeared: " + ", ".join(appeared)
                + f" (drift_count={payload.get('drift_count')})"
            )
            msg = _coach_drift_message("position_drift", facts, fallback)
            outcome = notify_fn(
                msg,
                severity="warning",
                dedupe_key=f"position_drift_appeared_{_drift_set_dedupe_suffix(appeared)}",
            )
            delivered = _delivered(outcome)
            all_delivered = all_delivered and delivered
            disposition["sent"].append({"event": "appeared", "delivered": delivered})
        if resolved:
            facts = {"resolved": resolved, "still_active": len(current)}
            fallback = "position drift resolved: " + ", ".join(resolved)
            msg = _coach_drift_message("position_drift_resolved", facts, fallback)
            outcome = notify_fn(
                msg,
                severity="info",
                dedupe_key=f"position_drift_resolved_{_drift_set_dedupe_suffix(resolved)}",
            )
            delivered = _delivered(outcome)
            all_delivered = all_delivered and delivered
            disposition["sent"].append({"event": "resolved", "delivered": delivered})

        if (appeared or resolved) and not all_delivered:
            # Transport failure: leave the last-alerted state untouched so the
            # next publisher cycle retries (dedupe marks only on SENT, so the
            # retry is not self-suppressed).
            return disposition

        state_payload = {
            "schema_version": "position_guard_drift_alert_state.v1",
            "ts_utc": _utc_now_iso(),
            "identities": sorted(current),
        }
        sp = Path(state_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        tmp = sp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")
        tmp.replace(sp)
        disposition["state_advanced"] = True
    except Exception as exc:  # noqa: BLE001 — alerting must never block reconciliation
        LOG.warning("drift transition alerting failed (non-blocking): %s", exc)
    return disposition


def _flag_on(name: str) -> bool:
    """Strict truthy env parse — '0'/'false'/'no'/'off'/'' all read as OFF."""
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _emit_position_guard_drift_v4(
    *,
    guard_path: "Path | None" = None,
    snapshot_path: "Path | None" = None,
    out_path: "Path | None" = None,
    now: "float | None" = None,
) -> "Dict[str, Any] | None":
    """W1A-5 / CHAD_DRIFT_V4 (default OFF) — observability-only sibling drift file.

    Emits an INDEPENDENT-leg drift view (``position_guard_drift.v4``) to
    ``runtime/position_guard_drift_v4.json`` via
    ``detect_guard_vs_independent_snapshot_drift`` — which compares the guard's
    two same-source legs SEPARATELY against the ONE independent
    ``positions_snapshot.json`` leg (the EXS4 model; never sums the dual-booked
    legs, freshness-gates the independent leg to blind).

    D2 (Wave-1): SIBLING file only. It does NOT touch
    ``position_guard_drift.json`` and does NOT feed the live-readiness RED gate —
    a new comparator must soak before it can flip live-readiness. When the flag is
    unset this is a **no-op**: nothing is read, nothing is written, and the v3
    path (``_emit_position_guard_drift``) is entirely unaffected.

    Returns the v4 payload dict when it ran, else ``None`` (flag off).
    """
    if not _flag_on("CHAD_DRIFT_V4"):
        return None
    try:
        from chad.core.position_guard import detect_guard_vs_independent_snapshot_drift
    except Exception as exc:  # noqa: BLE001
        LOG.warning("position_guard import failed for v4 drift emit: %s", exc)
        return None

    gpath = guard_path or GUARD_PATH
    spath = snapshot_path or SNAPSHOT_PATH
    opath = out_path or DRIFT_V4_OUT_PATH

    try:
        guard_state = json.loads(gpath.read_text(encoding="utf-8")) if gpath.is_file() else {}
    except Exception as exc:  # noqa: BLE001
        LOG.warning("position_guard.json unreadable for v4 drift emit: %s", exc)
        guard_state = {}
    try:
        snapshot = json.loads(spath.read_text(encoding="utf-8")) if spath.is_file() else None
    except Exception as exc:  # noqa: BLE001
        LOG.warning("positions_snapshot.json unreadable for v4 drift emit: %s", exc)
        snapshot = None

    excluded_symbols = {
        str(s).strip().upper()
        for s in (set(EXCLUSION_POLICY.keys()) | set(_BROKER_PREEXISTING))
        if s
    }
    result = detect_guard_vs_independent_snapshot_drift(
        guard_state if isinstance(guard_state, dict) else {},
        snapshot if isinstance(snapshot, dict) else None,
        excluded_symbols=excluded_symbols,
        now=now,
    )
    payload = {
        **result,
        "ts_utc": _utc_now_iso(),
        "ttl_seconds": TTL_SECONDS,
        "excluded_symbols": sorted(excluded_symbols),
    }
    opath.parent.mkdir(parents=True, exist_ok=True)
    tmp = opath.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(opath)
    LOG.info(
        "position_guard_drift_v4 (observability sibling) leg=%s drift_count=%d info_count=%d path=%s",
        result.get("independent_leg"), int(result.get("drift_count", 0) or 0),
        int(result.get("info_count", 0) or 0), opath,
    )
    return payload


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    # GAP-028 §5.1: emit per-strategy ↔ broker_sync drift advisory file
    # (read-only; aligned with this publisher's TTL). Failure here must not
    # block the rest of reconciliation, so wrap defensively.
    try:
        _emit_position_guard_drift()
    except Exception as exc:  # noqa: BLE001
        LOG.warning("position_guard_drift emit failed: %s", exc)

    # W1A-5 (CHAD_DRIFT_V4, default OFF): observability-only independent-leg
    # sibling file. No-op unless the flag is on; never touches the v3 file or the
    # RED gate. Defensive so it can never block reconciliation.
    try:
        _emit_position_guard_drift_v4()
    except Exception as exc:  # noqa: BLE001
        LOG.warning("position_guard_drift_v4 emit failed: %s", exc)

    # GAP-032 preventive: refresh runtime/systemd_wants_lint.json on the
    # publisher's existing ~5-min cadence so a regression to the
    # regular-file-in-wants/ corruption signature is detected without
    # adding a new (itself-corruptible) systemd timer. Best-effort.
    try:
        from chad.ops.systemd_wants_lint import main as _systemd_wants_lint_main
        _systemd_wants_lint_main(["--quiet"])
    except Exception as exc:  # noqa: BLE001
        LOG.warning("systemd_wants_lint refresh failed: %s", exc)

    chad_side = _load_guard_positions()

    # Count open strategy-attribution entries (non-broker_sync) for visibility
    # alongside chad_open. In paper mode chad_open intentionally filters to
    # broker_sync; this exposes the strategy layer that is otherwise hidden.
    try:
        _raw_guard = (
            json.loads(GUARD_PATH.read_text(encoding="utf-8"))
            if GUARD_PATH.exists() else {}
        )
        chad_strategy_open = sum(
            1 for k, v in (_raw_guard or {}).items()
            if isinstance(v, dict) and v.get("open") and not k.startswith("broker_sync|")
        )
    except Exception:
        chad_strategy_open = 0

    try:
        broker_side = _load_broker_positions()
    except Exception as exc:
        LOG.warning("IBKR unavailable for reconciliation: %s", exc)
        _write({
            "status": "RED",
            "broker_source": f"unavailable:{exc}",
            "chad_state_source": "position_guard.json",
            "counts": {
                "chad_open": len(chad_side),
                "chad_strategy_open": chad_strategy_open,
                "broker_positions": 0,
            },
            "mismatches": [],
            "exclusion_policy": EXCLUSION_POLICY,
            "notes": ["ibkr_unavailable"],
        })
        try:
            from chad.utils.telegram_notify import notify
            notify(
                f"🚨 Reconciliation RED — IBKR unavailable: {exc}",
                severity="critical",
                dedupe_key="reconciliation_red",
            )
        except Exception:
            pass
        return 0

    breakdown = _load_guard_breakdown()

    mismatches: List[Dict[str, Any]] = []
    drifts: List[Dict[str, Any]] = []
    diagnostic_drifts: List[Dict[str, Any]] = []
    excluded: List[str] = []
    futures_excluded: List[str] = []
    symbols = set(chad_side) | set(broker_side)
    worst = 0.0
    for sym in sorted(symbols):
        if sym in KNOWN_NON_CHAD_SYMBOLS:
            excluded.append(sym)
            continue
        if sym in KNOWN_FUTURES_SYMBOLS:
            # ISSUE-29 companion: futures reconciliation requires explicit
            # contract_month resolution (see chad/execution/ibkr_adapter.py
            # ContractResolutionError). Skip until that path is complete.
            futures_excluded.append(sym)
            continue
        c = chad_side.get(sym, 0.0)
        b = broker_side.get(sym, 0.0)
        diff = abs(c - b)
        if diff > 0:
            # Classify diff as BROKER_DRIFT vs real mismatch.
            # Drift criterion: CHAD's side is zero OR all from broker_sync
            # (no strategy attribution). IBKR position moved without CHAD
            # initiating — a reality to log, not a bug to flag RED.
            bd = breakdown.get(sym, {"broker_sync": 0.0, "strategies": 0.0})
            strategy_contrib = abs(bd.get("strategies", 0.0))
            if strategy_contrib < 1e-6:
                LOG.warning(
                    "BROKER_DRIFT symbol=%s guard_qty=%s broker_qty=%s diff=%s "
                    "(no strategy attribution; broker moved independently)",
                    sym, c, b, diff,
                )
                # PR-09: route broker_sync-only advisory entries to
                # diagnostic_drifts. The drifts[] field is reserved for
                # strategy-attributable drifts that MUST trip live_readiness
                # RED (GAP-041 safety net preserved). Today's classifier
                # never lands here when strategies are involved — see else.
                diagnostic_drifts.append(
                    {"symbol": sym, "chad": c, "broker": b, "diff": diff, "kind": "broker_sync_only"}
                )
            else:
                mismatches.append({"symbol": sym, "chad": c, "broker": b, "diff": diff})
                worst = max(worst, diff)

    if worst <= 1.0:
        status = "GREEN"
    elif worst <= 2.0:
        status = "YELLOW"
    else:
        status = "RED"
        try:
            from chad.utils.telegram_notify import notify
            _red_reason = f"worst_diff={worst:.2f} mismatches={len(mismatches)}"
            notify(
                f"🚨 Reconciliation RED — {_red_reason}",
                severity="critical",
                dedupe_key="reconciliation_red",
            )
        except Exception:
            pass

    _write({
        "status": status,
        "broker_source": f"ibkr:clientId={IBKR_CLIENT_ID}",
        "chad_state_source": "position_guard.json",
        "counts": {
            "chad_open": len(chad_side),
            "chad_strategy_open": chad_strategy_open,
            "broker_positions": len(broker_side),
        },
        "worst_diff": worst,
        "mismatches": mismatches,
        "drifts": drifts,
        "diagnostic_drifts": diagnostic_drifts,
        "excluded_symbols": excluded,
        "futures_excluded_symbols": futures_excluded,
        "exclusion_policy": EXCLUSION_POLICY,
        "notes": [],
    })
    LOG.info(
        "reconciliation status=%s worst_diff=%.2f mismatches=%d drifts=%d diagnostic_drifts=%d futures_excluded=%d",
        status, worst, len(mismatches), len(drifts), len(diagnostic_drifts), len(futures_excluded),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
