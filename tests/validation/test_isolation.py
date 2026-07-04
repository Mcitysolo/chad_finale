"""Transitive-import-closure isolation test for chad.validation (SSOT §1.2, §2).

The edge-validation harness is the wall between paper and live money. For it to
be *unable to touch what it judges*, its entire import closure must be free of
the live trading loop, broker/exchange adapters, and any ``runtime/`` reader.
This test enforces that mechanically rather than by convention:

  * A subprocess imports ``chad.validation`` **and every submodule of it** in a
    clean interpreter (under paper / futures-disable env guards), then dumps the
    full ``sys.modules`` set mapped to each module's ``__file__`` — the true,
    fully transitive closure (catches forward-imports through any submodule, not
    just what ``__init__`` re-exports).
  * ``first-party`` check (strongest): NO in-repo module — resolved by its file
    path under the repo root — may appear in the closure unless it lives under
    ``chad/validation/`` or ``chad/constants/``. This generalizes the allowlist
    beyond ``chad.*`` to first-party packages that carry no forbidden token in
    their name (e.g. ``backend.api_gateway``, which holds the live execution-mode
    gate ``ExecutionMode.IBKR_LIVE`` / ``paper_only``).
  * ``allowlist`` check: every ``chad.*`` module in that closure must be under
    ``chad.validation`` or the pure ``chad.constants`` package — a name-based
    restatement of the first-party check for the ``chad`` namespace.
  * ``denylist`` check: no module name (chad or third-party) contains a
    broker/live/exchange/network token — catches an external adapter or transport
    lib (``ib_async``, ``ibapi``, a Kraken client, ``websockets``, ``ccxt``) that
    lives in site-packages, not the repo, where the first-party check can't see it.

The two pure checker functions are exercised by **negative-control** tests
(``test_*_catches_*``) that inject synthetic forbidden module names and assert
they are flagged — proving the checks are not vacuous. That is how we know the
isolation test would actually fail if a forbidden import were ever added.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

import pytest

# Repo root = two levels up from tests/validation/.
ROOT = Path(__file__).resolve().parents[2]

# The ONLY in-repo directories the harness may transitively import from. Any
# first-party module whose file lives outside these (e.g. ``backend/*.py``,
# ``chad/core/*.py``, a ``runtime/`` reader) is a violation by construction.
ALLOWED_FIRST_PARTY_DIRS: tuple[Path, ...] = (
    ROOT / "chad" / "validation",
    ROOT / "chad" / "constants",
)

# The ONLY chad packages the harness may transitively import. chad.constants is a
# pure, side-effect-free constants package (imports only ``__future__``); the
# canonical FX constant lives there (SSOT §5). Everything else under ``chad.`` is
# live/broker/runtime-adjacent and therefore forbidden.
ALLOWED_CHAD_PREFIXES: tuple[str, ...] = ("chad.validation", "chad.constants")

# Tokens that must never appear in any module name in the closure. Substring match
# (lower-cased) so ``chad.core.live_loop``, ``chad.execution.ibkr_adapter``,
# ``ib_async``, ``ibapi``, and ``kraken_client`` are all caught wherever they sit.
FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "live_loop",
    "orchestrator",
    "ibkr",
    "ib_async",
    "ib_insync",
    "ibapi",
    "kraken",
    "broker",
    "telegram",
    "execution",   # chad.execution.* — order placement path
    "position_guard",
    "trade_closer",
    "profit_lock",
    # Network / exchange transports: these live in site-packages, not the repo,
    # so the first-party check can't see them — only the denylist can.
    "websocket",   # covers websocket + websockets
    "ccxt",
    "requests",
    "aiohttp",
    "urllib3",
    # First-party live-adjacent packages whose names carry no broker token
    # (belt-and-suspenders with the file-path first-party check below).
    "backend",
    "api_gateway",
    "operator_surface",
    "polygon",
    "runtime",
)


# --------------------------------------------------------------------------- #
# Pure checkers (exercised both against the real closure and synthetic leaks).
# --------------------------------------------------------------------------- #
def disallowed_chad_modules(module_names: Iterable[str]) -> list[str]:
    """Return every ``chad*`` module NOT under an allowed prefix (allowlist check)."""
    bad: list[str] = []
    for name in module_names:
        if name == "chad" or name.startswith("chad."):
            if name == "chad":
                continue  # the namespace package itself is fine
            if not any(
                name == p or name.startswith(p + ".") for p in ALLOWED_CHAD_PREFIXES
            ):
                bad.append(name)
    return sorted(bad)


def forbidden_modules(module_names: Iterable[str]) -> list[str]:
    """Return every module whose name contains a forbidden token (denylist check)."""
    lowered = [(name, name.lower()) for name in module_names]
    return sorted(
        name for name, low in lowered if any(tok in low for tok in FORBIDDEN_SUBSTRINGS)
    )


# Path segments marking installed third-party code. The venv lives at ``ROOT/venv``
# (CLAUDE.md), so site-packages resolves *under* the repo root — such files are
# third-party, not first-party repo source, and are governed by the denylist.
_THIRD_PARTY_PATH_MARKERS: tuple[str, ...] = ("site-packages", "dist-packages", "venv", ".venv")


def _is_installed_third_party(resolved_path: str) -> bool:
    parts = resolved_path.split(os.sep)
    return any(marker in parts for marker in _THIRD_PARTY_PATH_MARKERS)


def first_party_violations(module_files: dict[str, Optional[str]]) -> list[str]:
    """Return every in-repo (first-party) module NOT under an allowed dir.

    Generalizes the ``chad.*`` allowlist to ANY package that lives inside this
    repo (``backend.*``, ``chad.core.*``, a ``runtime/`` reader, …), so a
    first-party live/exec module whose name carries no forbidden token is still
    caught by its file path. Out of scope here (the denylist governs them):
    builtins with no ``__file__``, stdlib/site-packages outside the repo, and
    installed packages inside the in-repo venv (``ROOT/venv/.../site-packages``).
    """
    root_str = str(ROOT)
    allowed = [str(d) for d in ALLOWED_FIRST_PARTY_DIRS]
    bad: list[str] = []
    for name, path in module_files.items():
        if name == "chad":
            continue  # the empty namespace-package __init__ is benign
        if not path:
            continue  # builtin / namespace with no file → not first-party by file
        rp = os.path.realpath(path)
        if not (rp == root_str or rp.startswith(root_str + os.sep)):
            continue  # outside the repo → third-party/stdlib, denylist's job
        if _is_installed_third_party(rp):
            continue  # inside the in-repo venv / site-packages → third-party
        if not any(rp == d or rp.startswith(d + os.sep) for d in allowed):
            bad.append(name)
    return sorted(bad)


# --------------------------------------------------------------------------- #
# Closure capture — fresh interpreter, import package + all submodules.
# --------------------------------------------------------------------------- #
_CLOSURE_SNIPPET = """
import importlib, pkgutil, sys, json
import chad.validation as pkg
for info in pkgutil.walk_packages(pkg.__path__, prefix="chad.validation."):
    importlib.import_module(info.name)
mods = {name: getattr(mod, "__file__", None) for name, mod in sys.modules.items()}
print(json.dumps(mods, sort_keys=True))
"""


def _import_closure() -> dict[str, Optional[str]]:
    """Map every module in sys.modules after importing chad.validation + all its
    submodules (fresh interpreter) to its ``__file__`` (None for builtins).

    The probe runs under paper / futures-disable guards so that even if a future
    validation submodule grew an import-time side effect, it could not reach a
    live broker or a futures venue while this test runs (defense-in-depth; the
    closure is import-side-effect-free today).
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    env["CHAD_DISABLE_FUTURES_EXECUTION"] = "1"
    env["CHAD_EXECUTION_MODE"] = "paper"
    env["CHAD_SKIP_IB_CONNECT"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _CLOSURE_SNIPPET],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
        check=False,
    )
    assert proc.returncode == 0, f"closure subprocess failed:\nSTDERR:\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def closure() -> dict[str, Optional[str]]:
    return _import_closure()


# --------------------------------------------------------------------------- #
# The guarantees.
# --------------------------------------------------------------------------- #
def test_closure_pulls_in_expected_validation_modules(closure: dict[str, Optional[str]]) -> None:
    """Sanity: the closure actually loaded the Phase 0 + Phase 1 modules (not a no-op)."""
    assert "chad.validation" in closure
    assert "chad.validation.bar_audit" in closure
    assert "chad.validation.scoring_spine" in closure


def test_only_allowed_chad_modules_in_closure(closure: dict[str, Optional[str]]) -> None:
    """No chad.* module outside chad.validation / chad.constants is imported."""
    bad = disallowed_chad_modules(closure)
    assert bad == [], f"forbidden chad.* modules reachable from chad.validation: {bad}"


def test_no_forbidden_tokens_in_closure(closure: dict[str, Optional[str]]) -> None:
    """No live-loop / broker / exchange / runtime-writer module in the closure."""
    bad = forbidden_modules(closure)
    assert bad == [], f"forbidden modules reachable from chad.validation: {bad}"


def test_specific_live_modules_absent(closure: dict[str, Optional[str]]) -> None:
    """Explicit spot-checks named in SSOT §1.2 / the /goal brief."""
    for banned in (
        "chad.core.live_loop",
        "chad.execution.ibkr_adapter",
        "chad.core.orchestrator",
    ):
        assert banned not in closure, f"{banned} must not be in the harness closure"


def test_no_numpy_in_closure(closure: dict[str, Optional[str]]) -> None:
    """Stdlib-only hygiene (module docstring): numpy is present in the env but the
    spine deliberately avoids it for determinism, so it must not be pulled in."""
    assert not any(m == "numpy" or m.startswith("numpy.") for m in closure)


# --------------------------------------------------------------------------- #
# Negative controls — prove the checkers are NOT vacuous (would catch a leak).
# --------------------------------------------------------------------------- #
def test_allowlist_catches_injected_chad_leak(closure: dict[str, Optional[str]]) -> None:
    """Injecting a live chad module makes the allowlist check fail as intended."""
    leaked = list(closure) + ["chad.core.live_loop", "chad.risk.dynamic_risk_allocator"]
    bad = disallowed_chad_modules(leaked)
    assert "chad.core.live_loop" in bad
    assert "chad.risk.dynamic_risk_allocator" in bad


def test_denylist_catches_injected_broker_leak(closure: dict[str, Optional[str]]) -> None:
    """Injecting broker/exchange/network modules trips the denylist as intended."""
    for leak in (
        "chad.execution.ibkr_adapter",
        "ib_async",
        "kraken.client",
        "ibapi.wrapper",
        "websockets",
        "ccxt",
        "backend.polygon_stocks_stream",
    ):
        bad = forbidden_modules(list(closure) + [leak])
        assert leak in bad, f"denylist failed to catch {leak}"


def test_first_party_check_is_clean_today(closure: dict[str, Optional[str]]) -> None:
    """No in-repo module outside chad.validation / chad.constants is in the closure.

    This is the strongest guarantee — it catches first-party live/exec packages
    (e.g. ``backend.api_gateway``) that the ``chad.*``-only allowlist and the token
    denylist would both miss, by resolving each module's file path.
    """
    bad = first_party_violations(closure)
    assert bad == [], f"first-party modules outside the harness are reachable: {bad}"


def test_first_party_check_catches_backend_leak(closure: dict[str, Optional[str]]) -> None:
    """Negative control: a synthetic in-repo backend.* leak (no forbidden token in a
    stripped name, and not under chad.*) is still caught by the file-path check —
    the exact blind spot the ``chad.*``-only allowlist had."""
    leaked = dict(closure)
    leaked["some_first_party_gate"] = str(ROOT / "backend" / "api_gateway.py")
    bad = first_party_violations(leaked)
    assert "some_first_party_gate" in bad
    # And a legitimately-allowed in-repo path must NOT be flagged.
    clean = dict(closure)
    clean["chad.validation.future_module"] = str(
        ROOT / "chad" / "validation" / "future_module.py"
    )
    assert "chad.validation.future_module" not in first_party_violations(clean)


def test_checkers_pass_on_clean_synthetic_set() -> None:
    """A hand-built clean set trips neither checker (guards against false positives)."""
    clean = [
        "chad",
        "chad.validation",
        "chad.validation.bar_audit",
        "chad.validation.scoring_spine",
        "chad.constants",
        "chad.constants.fx",
        "math",
        "json",
        "dataclasses",
    ]
    assert disallowed_chad_modules(clean) == []
    assert forbidden_modules(clean) == []


# --------------------------------------------------------------------------- #
# Public API re-export (Phase 1 spine reachable from the package root).
# --------------------------------------------------------------------------- #
def test_scoring_spine_reexported_from_package() -> None:
    """The spine is re-exported at chad.validation (SSOT §1.3 single-spine access)."""
    import chad.validation as v

    for name in ("ScoreResult", "score_returns", "score_trades", "score_equity"):
        assert name in v.__all__, f"{name} missing from chad.validation.__all__"
        assert hasattr(v, name), f"{name} not importable from chad.validation"
