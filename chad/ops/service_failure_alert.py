"""CHAD service-failure alert (Decision 3 / institutional closeout).

Invoked by ``chad-service-alert@<unit>.service`` (a systemd template unit)
whenever an ``OnFailure=`` directive on a CHAD service fires. It captures
a journal tail, optional runtime metadata snapshot (sizes + mtimes only —
no contents), writes a structured artifact under
``reports/service_failures/``, and sends a Telegram alert via the existing
``chad.utils.telegram_notify.notify`` path.

CLI:
    python -m chad.ops.service_failure_alert \\
      --failed-unit chad-options-chain-refresh.service \\
      --severity HIGH \\
      --include-journal-tail 50 \\
      --include-runtime-snapshot \\
      [--dry-run]

Exit codes:
    0 success (delivered OR intentionally dedupe-suppressed)
    2 unit name invalid / not a chad-* unit
    3 journal read failed (artifact still written)
    4 telegram send genuinely failed — config or transport (artifact still written)

Delivery auditability (schema v2)
---------------------------------
Every artifact records ``telegram_sent`` (bool), ``telegram_delivery_status``
(sent / suppressed_dedupe / dry_run / config_error / transport_error /
exception), and ``delivery_error`` (str|null). Prior to v2 a *successfully
suppressed duplicate* alert (dedupe TTL) returned ``notify_returned_false`` and
mapped to exit 4, latching the systemd handler into ``failed`` whenever a
service flapped (root cause of the 2026-06-27 chad-ibkr-bar-provider latch).
A suppressed duplicate is now exit 0 and only genuine config/transport failures
exit 4.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "service_failure_alert.v2"

EXIT_OK = 0
EXIT_INVALID_UNIT = 2
EXIT_JOURNAL_FAILED = 3
EXIT_TELEGRAM_FAILED = 4

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "reports" / "service_failures"

RUNTIME_SNAPSHOT_FILES = (
    "runtime/live_readiness.json",
    "runtime/scr_state.json",
    "runtime/stop_bus.json",
    "runtime/positions_truth.json",
    "runtime/position_guard_drift.json",
    "runtime/reconciliation_state.json",
)


@dataclass
class AlertResult:
    artifact_path: Path
    payload: dict[str, Any]
    exit_code: int


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_now_compact() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _is_chad_unit(unit: str) -> bool:
    return unit.startswith("chad-") and (
        unit.endswith(".service") or unit.endswith(".timer") or unit.endswith(".socket")
    )


def _read_journal_tail(unit: str, n: int) -> tuple[list[str], str | None]:
    """Return (lines, error). Error is None on success."""
    try:
        out = subprocess.check_output(
            ["journalctl", "-u", unit, "-n", str(n), "--no-pager", "--output=short-iso"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except subprocess.CalledProcessError as exc:
        return [], f"journalctl_exit_{exc.returncode}:{(exc.output or '').strip()[:200]}"
    except FileNotFoundError:
        return [], "journalctl_not_installed"
    except subprocess.TimeoutExpired:
        return [], "journalctl_timeout"
    except Exception as exc:  # pragma: no cover - defensive
        return [], f"journalctl_unexpected:{type(exc).__name__}:{exc}"
    return [ln for ln in out.splitlines() if ln], None


def _runtime_snapshot() -> dict[str, dict[str, Any]]:
    """Capture only metadata (mtime_utc, size_bytes). Never read contents."""
    snap: dict[str, dict[str, Any]] = {}
    for rel in RUNTIME_SNAPSHOT_FILES:
        fp = REPO_ROOT / rel
        key = Path(rel).name
        if not fp.is_file():
            snap[key] = {"present": False}
            continue
        st = fp.stat()
        snap[key] = {
            "present": True,
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "size_bytes": st.st_size,
        }
    return snap


def _systemctl_active(unit: str) -> str:
    try:
        out = subprocess.check_output(
            ["systemctl", "is-active", unit],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
        )
        return out.strip()
    except subprocess.CalledProcessError as exc:
        # is-active returns non-zero for failed/inactive; the output text is the truth
        return (exc.output or "unknown").strip()
    except FileNotFoundError:
        return "systemctl_not_installed"
    except Exception:  # pragma: no cover
        return "unknown"


def _format_telegram_message(payload: dict[str, Any]) -> str:
    """Compose the operator-facing message.

    COACH-VOICE-L1 U3: the message is rendered through the coach-voice
    presentation layer (calm plain English, no codes in SIMPLE mode). This is
    presentation only — the raw artifact JSON written to disk is unchanged, and
    the delivery-audit fields (telegram_sent / telegram_delivery_status /
    delivery_error) are derived from the notifier outcome, not from this text.
    If the coach layer is unavailable, fall back to the legacy machine-formatted
    message so the alert still delivers.
    """
    coached = _coach_message(payload)
    return coached if coached else _legacy_telegram_message(payload)


def _coach_message(payload: dict[str, Any]) -> str | None:
    """Render this service-failure payload via chad.utils.coach_voice.

    Returns None on any failure so the caller falls back to legacy text.
    """
    try:
        from chad.utils.coach_voice import format_alert

        facts = {
            "failed_unit": payload.get("failed_unit", ""),
            "severity": payload.get("severity", "HIGH"),
            "active_unit_status": payload.get("active_unit_status", "unknown"),
            "journal_tail": payload.get("journal_tail") or [],
            "journal_error": payload.get("journal_error"),
            "host": payload.get("host", ""),
            "ts_utc": payload.get("ts_utc", ""),
            "artifact_path": payload.get("artifact_path", ""),
        }
        text = format_alert("service_failure", facts)
        return text if isinstance(text, str) and text.strip() else None
    except Exception:
        return None


def _legacy_telegram_message(payload: dict[str, Any]) -> str:
    sev = payload.get("severity", "HIGH")
    icon = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "ℹ️"}.get(sev, "🚨")
    head = (
        f"{icon} CHAD {sev}: service failure — {payload['failed_unit']} "
        f"(state={payload.get('active_unit_status', 'unknown')})"
    )
    tail = payload.get("journal_tail") or []
    last_lines = "\n".join(tail[-5:]) if tail else "(no journal lines captured)"
    artifact = payload.get("artifact_path", "")
    return (
        f"{head}\n"
        f"host={payload.get('host', '?')}  ts={payload.get('ts_utc', '?')}\n"
        f"artifact={artifact}\n"
        f"--- journal tail (last 5 of {len(tail)}) ---\n"
        f"{last_lines}"
    )


def _send_telegram(payload: dict[str, Any], *, dry_run: bool) -> tuple[bool, str | None, str]:
    """Return (delivered, delivery_error, delivery_status).

    ``delivered`` is True when the incident is considered handled at the
    telegram layer: either the message was actually sent OR a recent identical
    alert was intentionally dedupe-suppressed (both mean the operator is/was
    notified). Only genuine config/transport failures return delivered=False,
    which is what should drive EXIT_TELEGRAM_FAILED. A per-invocation
    ``telegram_sent`` bool (message actually delivered *this* call) is derived
    from ``delivery_status`` by the caller.
    """
    message = _format_telegram_message(payload)
    if dry_run:
        print("=== DRY RUN: Telegram message that would be sent ===")
        print(message)
        return True, None, "dry_run"
    try:
        from chad.utils.telegram_notify import notify_detailed  # local import: optional dep

        outcome = notify_detailed(
            message,
            severity="critical",
            dedupe_key=f"service_failure:{payload['failed_unit']}",
        )
    except Exception as exc:
        # Import failure or an unexpected defect in the notifier itself.
        return False, f"telegram_exception:{type(exc).__name__}:{exc}", "exception"

    status = outcome.status.value
    if outcome.sent:
        return True, None, status
    if outcome.suppressed:
        # A recent identical alert for this unit already delivered within the
        # dedupe TTL — a successful no-op, NOT a delivery failure. This is the
        # v1 defect: a flapping service latched the handler FAILED here.
        return True, None, status
    # config_error / transport_error / empty_message — a genuine failure.
    return False, outcome.error, status


def _artifact_path_for_payload(artifact_path: Path) -> str:
    """Return repo-relative path when possible; else absolute string."""
    try:
        return str(artifact_path.relative_to(REPO_ROOT))
    except ValueError:
        return str(artifact_path)


def build_payload(
    *,
    failed_unit: str,
    severity: str,
    journal_tail: list[str],
    journal_error: str | None,
    include_runtime_snapshot: bool,
    artifact_path: Path,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "ts_utc": _utc_now_iso(),
        "failed_unit": failed_unit,
        "severity": severity,
        "host": socket.gethostname(),
        "journal_tail": journal_tail,
        "journal_error": journal_error,
        "active_unit_status": _systemctl_active(failed_unit),
        "artifact_path": _artifact_path_for_payload(artifact_path),
    }
    if include_runtime_snapshot:
        payload["runtime_snapshot"] = _runtime_snapshot()
    return payload


def write_artifact(payload: dict[str, Any], artifact_path: Path) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run(
    *,
    failed_unit: str,
    severity: str = "HIGH",
    journal_tail_n: int = 50,
    include_runtime_snapshot: bool = False,
    dry_run: bool = False,
    artifact_dir: Path | None = None,
) -> AlertResult:
    """Library entry point. Returns AlertResult with payload + chosen exit code."""
    if not _is_chad_unit(failed_unit):
        return AlertResult(
            artifact_path=Path("/dev/null"),
            payload={"error": f"refused: {failed_unit!r} is not a chad-* systemd unit"},
            exit_code=EXIT_INVALID_UNIT,
        )

    journal_lines, journal_error = _read_journal_tail(failed_unit, journal_tail_n)

    out_dir = artifact_dir or ARTIFACT_DIR
    artifact_path = out_dir / f"{_utc_now_compact()}__{failed_unit}.json"

    payload = build_payload(
        failed_unit=failed_unit,
        severity=severity,
        journal_tail=journal_lines,
        journal_error=journal_error,
        include_runtime_snapshot=include_runtime_snapshot,
        artifact_path=artifact_path,
    )
    write_artifact(payload, artifact_path)

    telegram_ok, telegram_err, delivery_status = _send_telegram(payload, dry_run=dry_run)

    # v2 delivery-audit fields (recorded for EVERY incident, not just failures):
    #   telegram_sent           — a message was actually delivered THIS invocation
    #   telegram_delivery_status — sent / suppressed_dedupe / dry_run / config_error /
    #                              transport_error / exception / empty_message
    #   delivery_error          — error text, or null on success/suppression
    payload["telegram_sent"] = bool(delivery_status == "sent")
    payload["telegram_delivery_status"] = delivery_status
    payload["delivery_error"] = telegram_err
    # Back-compat: keep telegram_error populated on genuine failures only.
    if not telegram_ok and not dry_run:
        payload["telegram_error"] = telegram_err

    exit_code = EXIT_OK
    if journal_error is not None:
        exit_code = EXIT_JOURNAL_FAILED
    # EXIT_TELEGRAM_FAILED only on a GENUINE delivery failure — never on a
    # dedupe-suppressed duplicate (delivered=True), which is the v1 latch bug.
    if not telegram_ok and not dry_run:
        exit_code = EXIT_TELEGRAM_FAILED

    # Rewrite the artifact so the delivery-audit fields are always persisted.
    write_artifact(payload, artifact_path)

    return AlertResult(artifact_path=artifact_path, payload=payload, exit_code=exit_code)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="chad.ops.service_failure_alert",
        description="Emit a structured CHAD service-failure alert (artifact + Telegram).",
    )
    p.add_argument("--failed-unit", required=True, help="systemd unit name (e.g. chad-x.service)")
    p.add_argument(
        "--severity",
        choices=["HIGH", "MEDIUM", "LOW"],
        default="HIGH",
        help="Alert severity (default HIGH).",
    )
    p.add_argument(
        "--include-journal-tail",
        type=int,
        default=50,
        help="Number of journal lines to include (default 50).",
    )
    p.add_argument(
        "--include-runtime-snapshot",
        action="store_true",
        help="Include metadata-only snapshot of key runtime/*.json files.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not send Telegram; print payload + would-send message.",
    )
    args = p.parse_args(argv)

    result = run(
        failed_unit=args.failed_unit,
        severity=args.severity,
        journal_tail_n=args.include_journal_tail,
        include_runtime_snapshot=args.include_runtime_snapshot,
        dry_run=args.dry_run,
    )

    if args.dry_run or result.exit_code == EXIT_INVALID_UNIT:
        print(json.dumps(result.payload, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "artifact_path": str(result.artifact_path),
                    "exit_code": result.exit_code,
                    "schema_version": SCHEMA_VERSION,
                },
                indent=2,
            )
        )

    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
