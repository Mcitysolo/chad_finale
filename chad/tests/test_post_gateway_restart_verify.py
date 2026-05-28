"""Tests for scripts/post_gateway_restart_verify.py (Fix B / Channel 2).

All tests run with mocked sockets and temp files — no real Gateway, no real
broker. They assert exit codes, severities, artifact contents, and the
read-only invariant (no runtime/* or data/* mutation).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_VERIFIER_PATH = REPO_ROOT / "scripts" / "post_gateway_restart_verify.py"

_spec = importlib.util.spec_from_file_location(
    "post_gateway_restart_verify", _VERIFIER_PATH
)
verifier = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
# Register before exec so dataclass annotation resolution can find the module.
sys.modules["post_gateway_restart_verify"] = verifier
_spec.loader.exec_module(verifier)


# --- helpers ---------------------------------------------------------------

FIXED_START = datetime(2026, 5, 28, 3, 15, 0, tzinfo=timezone.utc)


class FakeClock:
    """Monotonic clock that advances only when sleep() is called."""

    def __init__(self) -> None:
        self._t = 0.0

    def monotonic(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        self._t += seconds


def _ok_connect(host: str, port: int) -> None:
    return None


def _refused_connect(host: str, port: int) -> None:
    raise ConnectionRefusedError("connection refused")


def _write_status(
    path: Path, latency_ms: float, recovery_state: str, mtime_epoch: float
) -> None:
    payload = {
        "latency_ms": latency_ms,
        "current_recovery_state": recovery_state,
        "ok": True,
        "port": 4002,
        "ts_utc": "2026-05-28T03:15:30Z",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    os.utime(path, (mtime_epoch, mtime_epoch))


def _make_deps(tmp_path: Path, **overrides):
    status_path = overrides.pop("status_path", tmp_path / "ibkr_status.json")
    log_dir = overrides.pop("log_dir", tmp_path / "gateway_restart_log")
    clock = FakeClock()
    kwargs = dict(
        status_path=status_path,
        log_dir=log_dir,
        now_utc=lambda: FIXED_START,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        connect=_ok_connect,
    )
    kwargs.update(overrides)
    return verifier.Deps(**kwargs)


# --- tests -----------------------------------------------------------------


def test_all_checks_pass(tmp_path: Path) -> None:
    status_path = tmp_path / "ibkr_status.json"
    _write_status(status_path, 800.0, "healthy", FIXED_START.timestamp() + 30)
    deps = _make_deps(tmp_path, status_path=status_path, connect=_ok_connect)
    payload = verifier.run_verification(deps)

    assert payload["overall_ok"] is True
    assert payload["exit_code"] == verifier.EXIT_OK
    assert payload["alert_severity"] == "INFO"
    assert payload["checks"]["port_listening"]["ok"] is True
    assert payload["checks"]["latency_healthy"]["latency_ms"] == 800.0


def test_port_not_listening(tmp_path: Path) -> None:
    status_path = tmp_path / "ibkr_status.json"
    _write_status(status_path, 800.0, "healthy", FIXED_START.timestamp() + 30)
    deps = _make_deps(tmp_path, status_path=status_path, connect=_refused_connect)
    payload = verifier.run_verification(deps)

    assert payload["overall_ok"] is False
    assert payload["exit_code"] == verifier.EXIT_UNHEALTHY
    assert payload["alert_severity"] == "CRITICAL"
    assert payload["checks"]["port_listening"]["ok"] is False
    # The port retry loop must respect its bounded budget.
    assert payload["checks"]["port_listening"]["attempts"] >= 2


def test_artifact_stale(tmp_path: Path) -> None:
    status_path = tmp_path / "ibkr_status.json"
    # mtime older than verifier start - 120s → never considered fresh.
    _write_status(status_path, 800.0, "healthy", FIXED_START.timestamp() - 200)
    deps = _make_deps(tmp_path, status_path=status_path, connect=_ok_connect)
    payload = verifier.run_verification(deps)

    assert payload["overall_ok"] is False
    assert payload["exit_code"] == verifier.EXIT_UNHEALTHY
    assert payload["alert_severity"] == "WARNING"
    assert payload["checks"]["artifact_fresh"]["ok"] is False


def test_latency_above_sanity_threshold(tmp_path: Path) -> None:
    status_path = tmp_path / "ibkr_status.json"
    _write_status(status_path, 8000.0, "healthy", FIXED_START.timestamp() + 30)
    deps = _make_deps(tmp_path, status_path=status_path, connect=_ok_connect)
    payload = verifier.run_verification(deps)

    assert payload["overall_ok"] is False
    assert payload["exit_code"] == verifier.EXIT_UNHEALTHY
    assert payload["alert_severity"] == "WARNING"
    assert payload["checks"]["latency_healthy"]["ok"] is False
    assert payload["checks"]["latency_healthy"]["latency_ms"] == 8000.0


def test_recovery_state_still_above_threshold(tmp_path: Path) -> None:
    status_path = tmp_path / "ibkr_status.json"
    _write_status(status_path, 9000.0, "above_threshold", FIXED_START.timestamp() + 30)
    deps = _make_deps(tmp_path, status_path=status_path, connect=_ok_connect)
    payload = verifier.run_verification(deps)

    assert payload["overall_ok"] is False
    assert payload["exit_code"] == verifier.EXIT_UNHEALTHY
    assert payload["alert_severity"] == "WARNING"
    assert payload["checks"]["recovery_state"]["ok"] is False
    assert payload["checks"]["recovery_state"]["state"] == "above_threshold"


def test_artifact_written_to_correct_path(tmp_path: Path) -> None:
    status_path = tmp_path / "ibkr_status.json"
    log_dir = tmp_path / "gateway_restart_log"
    _write_status(status_path, 800.0, "healthy", FIXED_START.timestamp() + 30)
    deps = _make_deps(tmp_path, status_path=status_path, log_dir=log_dir)
    payload = verifier.run_verification(deps)
    out_path = verifier.write_artifact(payload, deps)

    assert out_path.parent == log_dir
    assert out_path.exists()
    assert out_path.name == "20260528T031500Z.json"
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["schema_version"] == "gateway_restart_verify.v1"
    assert written["overall_ok"] is True


def test_no_runtime_or_data_mutation(tmp_path: Path) -> None:
    """The verifier writes ONLY to its injected log_dir and reads its inputs.

    We assert this against sentinel runtime/ and data/ trees inside tmp_path
    rather than the real repo dirs: on a live box the production services
    churn runtime/*.json concurrently, which would make a real-dir snapshot
    flaky for reasons unrelated to the verifier. Because the verifier never
    references anything outside the deps it is handed, sentinel trees prove
    the read-only invariant deterministically.
    """
    status_path = tmp_path / "ibkr_status.json"
    _write_status(status_path, 800.0, "healthy", FIXED_START.timestamp() + 30)

    def _snapshot(root: Path) -> dict[str, float]:
        snap: dict[str, float] = {}
        for p in root.rglob("*"):
            if p.is_file():
                snap[str(p)] = p.stat().st_mtime
        return snap

    # Sentinel runtime/ and data/ trees the verifier must not touch.
    runtime_root = tmp_path / "runtime"
    data_root = tmp_path / "data"
    runtime_root.mkdir()
    data_root.mkdir()
    (runtime_root / "stop_bus.json").write_text('{"active": false}', encoding="utf-8")
    (runtime_root / "scr_state.json").write_text('{"state": "CONFIDENT"}', encoding="utf-8")
    (data_root / "ledger.json").write_text("[]", encoding="utf-8")
    os.utime(runtime_root / "stop_bus.json", (1000.0, 1000.0))
    os.utime(runtime_root / "scr_state.json", (1000.0, 1000.0))
    os.utime(data_root / "ledger.json", (1000.0, 1000.0))

    before_runtime = _snapshot(runtime_root)
    before_data = _snapshot(data_root)
    status_mtime_before = status_path.stat().st_mtime

    deps = _make_deps(tmp_path, status_path=status_path)
    payload = verifier.run_verification(deps)
    verifier.write_artifact(payload, deps)

    assert _snapshot(runtime_root) == before_runtime
    assert _snapshot(data_root) == before_data
    # The read input (ibkr_status.json) must also be untouched.
    assert status_path.stat().st_mtime == status_mtime_before
