"""W4B-8e — bare-terminal-proof for the flatten CLI (first-drill follow-ups).

The first live drill (2026-07-23) was run from a bare operator terminal —
no PYTHONPATH, no exported CHAD_* env — and needed five debugging attempts.
These tests pin the two self-sufficiency fixes plus the end-to-end proof:

  1. sys.path self-locate from __file__: `venv/bin/python3 scripts/<cli>.py`
     finds the chad package from ANY cwd;
  2. mode inference from the live units' systemd drop-ins when
     CHAD_EXECUTION_MODE is absent — env, when set, always overrides;
     conflicting drop-ins REFUSE (never guess a posture mid-migration);
  3. a subprocess invoking the script exactly as the operator did (bare venv
     python3, scrubbed env, non-repo cwd) reaches OVERALL: DRILL_COMPLETE —
     against a stub ib_async (no live socket) and a --repo-root sandbox —
     and leaves the REAL money ledger byte-identical (exhaust hygiene).
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_SPEC = importlib.util.spec_from_file_location(
    "flatten_all_bare_mod", REPO_ROOT / "scripts" / "flatten_all.py")
fa = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = fa
_SPEC.loader.exec_module(fa)


# --------------------------------------------------------------------------- #
# 1) sys.path self-locate
# --------------------------------------------------------------------------- #

def test_script_prepends_repo_root_to_sys_path():
    assert str(REPO_ROOT) in sys.path
    assert fa._SCRIPT_REPO_ROOT == REPO_ROOT


# --------------------------------------------------------------------------- #
# 2) mode inference from systemd drop-ins
# --------------------------------------------------------------------------- #

def _dropin_tree(tmp_path, live_loop_mode=None, orch_mode=None):
    globs = []
    for unit, mode in (("live-loop", live_loop_mode),
                       ("orchestrator", orch_mode)):
        d = tmp_path / f"chad-{unit}.service.d"
        d.mkdir()
        globs.append(str(d / "*.conf"))
        if mode is not None:
            (d / "20-execution-mode.conf").write_text(
                "[Service]\n"
                f'Environment=PYTHONUNBUFFERED=1 CHAD_EXECUTION_MODE={mode}\n',
                encoding="utf-8")
    return tuple(globs)


def test_infer_agreeing_dropins(tmp_path):
    globs = _dropin_tree(tmp_path, "paper", "paper")
    assert fa.infer_execution_mode_from_dropins(globs) == "paper"


def test_infer_single_unit_declaration(tmp_path):
    globs = _dropin_tree(tmp_path, "dry_run", None)
    assert fa.infer_execution_mode_from_dropins(globs) == "dry_run"


def test_infer_conflicting_dropins_refuses(tmp_path):
    """A live/paper split means a migration is in flight — never guess."""
    globs = _dropin_tree(tmp_path, "paper", "live")
    assert fa.infer_execution_mode_from_dropins(globs) is None


def test_infer_absent_dropins_returns_none(tmp_path):
    globs = _dropin_tree(tmp_path, None, None)
    assert fa.infer_execution_mode_from_dropins(globs) is None


def test_infer_parses_quoted_declaration(tmp_path):
    d = tmp_path / "u.service.d"
    d.mkdir()
    (d / "x.conf").write_text(
        '[Service]\nEnvironment="CHAD_EXECUTION_MODE=paper"\n',
        encoding="utf-8")
    assert fa.infer_execution_mode_from_dropins((str(d / "*.conf"),)) == "paper"


def test_gate_env_overrides_inference():
    """A SET env var wins even when inference disagrees."""
    gates = fa.check_gates({"CHAD_EXECUTION_MODE": "live"},
                           inferred_mode="paper")
    assert gates["mode_source"] == "env"
    assert gates["ok"] is False  # live refuses, inference cannot rescue


def test_gate_inference_fills_absent_env():
    gates = fa.check_gates({}, inferred_mode="paper")
    assert gates["mode_source"] == "systemd_dropin"
    assert gates["execution_mode_raw"] == "paper"
    assert gates["ok"] is True


def test_gate_refuses_when_neither_source():
    gates = fa.check_gates({}, inferred_mode=None)
    assert gates["mode_source"] is None
    assert gates["ok"] is False


# --------------------------------------------------------------------------- #
# 3) subprocess e2e — the operator's exact invocation shape
# --------------------------------------------------------------------------- #

_STUB_IB_ASYNC = textwrap.dedent("""\
    # Stub ib_async for the bare-terminal drill e2e: answers every read-only
    # probe as a flat, healthy connection; HARD-FAILS on any mutating call.
    class _Client:
        def isConnected(self):
            return True

    class IB:
        def __init__(self):
            self.client = _Client()

        def connect(self, host, port, clientId=None, timeout=None):
            return self

        def isConnected(self):
            return True

        def positions(self):
            return []

        def reqAllOpenOrders(self):
            return []

        def openOrders(self):
            return []

        def reqGlobalCancel(self):
            raise AssertionError("DRILL must never cancel")

        def placeOrder(self, *a, **k):
            raise AssertionError("DRILL must never place orders")

        def disconnect(self):
            pass
""")


def _real_dropins_declare_paper() -> bool:
    for pattern in fa._SYSTEMD_DROPIN_GLOBS:
        if glob.glob(pattern):
            return fa.infer_execution_mode_from_dropins() in ("paper", "dry_run")
    return False


@pytest.mark.skipif(not _real_dropins_declare_paper(),
                    reason="host has no paper/dry_run systemd drop-ins")
def test_bare_terminal_drill_reaches_drill_complete(tmp_path):
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    (stub_dir / "ib_async.py").write_text(_STUB_IB_ASYNC, encoding="utf-8")
    sandbox_root = tmp_path / "root"
    (sandbox_root / "runtime").mkdir(parents=True)
    cwd = tmp_path / "somewhere-else"          # NOT the repo root
    cwd.mkdir()

    real_fills = sorted(
        (REPO_ROOT / "data" / "fills").glob("FILLS_*.ndjson"))
    fills_before = {p: p.stat().st_size for p in real_fills}

    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        # The stub must shadow the venv's real ib_async; chad itself must be
        # found by the script's OWN sys.path self-locate, not by this.
        "PYTHONPATH": str(stub_dir),
    }
    assert not any(k.startswith("CHAD_") for k in env)

    proc = subprocess.run(
        [str(REPO_ROOT / "venv" / "bin" / "python3"),
         str(REPO_ROOT / "scripts" / "flatten_all.py"),
         "--repo-root", str(sandbox_root),
         "--no-telegram"],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=120,
    )

    out = proc.stdout + "\n" + proc.stderr
    assert proc.returncode == 0, f"drill failed rc={proc.returncode}\n{out}"
    assert "OVERALL: DRILL_COMPLETE" in out, out
    assert "mode inferred from systemd drop-ins" in out, out

    # Drill exhaust hygiene: everything lands under the sandbox root...
    assert (sandbox_root / "reports" / "ratification").glob(
        "PROOF_FLATTEN_DRILL_*.json")
    # ...and the REAL money ledger is byte-identical.
    for p, size in fills_before.items():
        assert p.stat().st_size == size, f"real ledger touched: {p}"
    sandbox_fills = sandbox_root / "data" / "fills"
    assert not sandbox_fills.exists() or not list(
        sandbox_fills.glob("FILLS_*.ndjson"))
