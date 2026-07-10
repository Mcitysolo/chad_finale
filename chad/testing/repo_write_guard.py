"""G3C-HF repo-write leak guard.

Fails any test that **creates or modifies** a file (or directory) under the repository
working-tree ``data/`` or ``runtime/`` trees. Kills the whole *class* of leaks where a
test constructs a production writer with its real default path instead of ``tmp_path``
— the ``data/margin_shadow/margin_shadow_20270115.ndjson`` incident (2026-07-10), where a
fixture using a fixed future epoch wrote through the real evidence path.

Wired by the repo-root ``conftest.py``:

  * the write primitives are patched **once** for the whole session (``install``);
  * enforcement is toggled **on only while a test is running** (``set_enforcing`` — the
    conftest autouse fixture brackets each test), so collection / import-time module
    wiring is never affected;
  * when enforcing, a guarded write is **BLOCKED** — it raises :class:`RepoWriteLeakError`
    *before* it touches disk — **and** the attempt is recorded, so the offending test
    fails even if the immediate caller swallows the exception (e.g. a best-effort
    ``except Exception: pass`` evidence writer such as
    ``MarginShadowGate._safe_write_evidence``).

Reads, and writes anywhere outside ``data/`` | ``runtime/`` (``tmp_path``, ``/tmp``,
the pytest cache …), are never touched. The guard patches the primitives that ``pathlib``
funnels through on CPython 3.12 (``builtins.open``/``io.open`` for ``Path.open``/
``write_text``/``write_bytes``; ``os.mkdir`` for ``Path.mkdir``; ``os.open`` for the
low-level path; ``os.rename``/``os.replace``/``os.remove``/``os.unlink``/``os.rmdir`` for
atomic-replace and delete flows).
"""

from __future__ import annotations

import builtins
import contextlib
import io
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# chad/testing/repo_write_guard.py -> parents[2] == repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _guarded_roots() -> Tuple[str, ...]:
    roots: List[str] = []
    for name in ("data", "runtime"):
        target = str(REPO_ROOT / name)
        try:
            roots.append(os.path.realpath(target))
        except OSError:
            roots.append(os.path.abspath(target))
    return tuple(roots)


_GUARDED_ROOTS: Tuple[str, ...] = _guarded_roots()

# --------------------------------------------------------------------------- #
# Ratchet baseline: KNOWN, PRE-EXISTING leak sinks discovered when this guard was first
# switched on (G3C-HF, 2026-07-10 full-suite run). These are production TELEMETRY / STATE
# writers with hardcoded runtime/ | data/ paths that tests exercise but cannot fully isolate
# without production-side runtime_dir injection (a separate cleanup epic — see the G3C-HF
# findings report). They are grandfathered so the guard is shippable-green and blocks only
# NEW leaks; each entry below is a tracked finding to redirect at the writer/test.
#
# NOTE: ``data/margin_shadow`` (the incident that motivated this guard) is deliberately NOT
# baselined — the margin_shadow leak is fixed at its source and stays hard-blocked here.
#
# Matching (against a realpath): a target is baselined iff it equals an entry, is *under* an
# entry directory, or is a ``.tmp``/``.<pid>`` sibling of an entry file (atomic-write temps).
_BASELINE_REL: Tuple[str, ...] = (
    # runtime STATE files (overwrite semantics; live-service-owned; self-heal next cycle)
    "runtime/dynamic_caps_correlation.json",     # chad.core.orchestrator caps refresh
    "runtime/decision_trace_heartbeat.json",     # chad.core.decision_trace heartbeat
    "runtime/correlation_overlay_health.json",   # chad.risk.correlation_strategy health
    "runtime/claude_usage.json",                 # chad.intel claude client usage meter
    "runtime/strategy_throttle_state.json",      # chad.risk strategy throttle gate
    "runtime/execution_environment.json",        # chad.ops.execution_environment_publisher
    "runtime/options_chain_refresh_failure.json",# chad.market_data.options_chain_refresh
    "runtime/locks",                             # transient *.lock files (fills/broker locks)
    # append-only DATA evidence dirs (spurious-row risk — highest-priority follow-up)
    "data/traces",                               # chad.core.decision_trace ndjson
    "data/slippage",                             # chad.analytics.slippage_tracker
    "data/signal_decay",                         # strategy signal-decay ledgers
)


def _baseline_abs() -> Tuple[str, ...]:
    out: List[str] = []
    for rel in _BASELINE_REL:
        target = str(REPO_ROOT / rel)
        try:
            out.append(os.path.realpath(target))
        except OSError:
            out.append(os.path.abspath(target))
    return tuple(out)


_BASELINE_ABS: Tuple[str, ...] = _baseline_abs()

# Write-ish open() mode characters and os.open() flags.
_WRITE_MODE_CHARS = ("w", "a", "x", "+")
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


class RepoWriteLeakError(RuntimeError):
    """Raised when a test tries to create/modify a file under the repo data/|runtime/ tree."""


# Attempts recorded for the currently-running test: (resolved_path, mode, primitive).
_attempts: List[Tuple[str, str, str]] = []
_enforcing: bool = False
_installed: bool = False
_orig: Dict[str, object] = {}


# --------------------------------------------------------------------------- #
# Enforcement state (driven by the conftest autouse fixture).
# --------------------------------------------------------------------------- #
def set_enforcing(on: bool) -> None:
    global _enforcing
    _enforcing = bool(on)


def take_attempts() -> List[Tuple[str, str, str]]:
    """Return and clear the recorded attempts (used by the fixture teardown check and by
    :func:`expect_blocked` so an intentional guard self-test does not double-fail)."""
    out = list(_attempts)
    _attempts.clear()
    return out


# --------------------------------------------------------------------------- #
# Path classification.
# --------------------------------------------------------------------------- #
def _resolve(target: object) -> Optional[str]:
    if isinstance(target, int):  # already-open fd → not a path
        return None
    try:
        p = os.fspath(target)  # type: ignore[arg-type]
    except TypeError:
        return None
    if isinstance(p, bytes):
        try:
            p = os.fsdecode(p)
        except Exception:  # noqa: BLE001
            return None
    try:
        return os.path.realpath(p)
    except OSError:
        return os.path.abspath(p)


def _is_baselined(rp: str) -> bool:
    """True if the resolved path is a grandfathered pre-existing leak sink (``_BASELINE_REL``)."""
    for entry in _BASELINE_ABS:
        if rp == entry or rp.startswith(entry + os.sep) or rp.startswith(entry + "."):
            return True
    return False


def _is_guarded(target: object) -> Optional[str]:
    """Resolved path if it is under the guarded tree (whether or not it is baselined), else None.

    Baselining does NOT make a path un-guarded: the write is still BLOCKED (never reaches disk —
    keeps the running live service's runtime/ intact). It only changes whether the block is
    *recorded* as a test failure — see :func:`_flag`."""
    rp = _resolve(target)
    if rp is None:
        return None
    for root in _GUARDED_ROOTS:
        if rp == root or rp.startswith(root + os.sep):
            return rp
    return None


def _path_exists(target: object) -> bool:
    """True if the target already exists (uses stat, which is NOT guarded). Lets the guard
    ignore *no-op* tree operations: ``mkdir(exist_ok=True)`` on an existing dir, or a delete of
    something already absent — neither creates nor modifies anything."""
    try:
        return os.path.exists(target)  # type: ignore[arg-type]
    except (TypeError, ValueError, OSError):
        return False


def _flag(target: object, mode: str, primitive: str) -> None:
    """Block the write (raise, so it never reaches disk — live-safe) and, UNLESS the sink is a
    grandfathered pre-existing leaker (``_BASELINE_REL``), record it so the test fails. Baselined
    sinks are still blocked but not recorded: a best-effort writer swallows the block and the test
    passes; only a test that genuinely *depends* on the write fails (and should move to tmp_path)."""
    rp = _resolve(target) or str(target)
    if not _is_baselined(rp):
        _attempts.append((rp, mode, primitive))
    raise RepoWriteLeakError(
        f"G3C-HF repo-write-guard: test tried to {primitive} a working-tree path: {rp} "
        f"(mode={mode!r}). Point the writer at tmp_path — NEVER the real data/ | runtime/ "
        f"path. The write was BLOCKED before touching disk."
    )


# --------------------------------------------------------------------------- #
# Guarded primitive wrappers.
# --------------------------------------------------------------------------- #
def _guarded_open(file, mode="r", *args, **kwargs):
    if _enforcing and any(c in str(mode) for c in _WRITE_MODE_CHARS) and _is_guarded(file):
        _flag(file, str(mode), "open")
    return _orig["open"](file, mode, *args, **kwargs)  # type: ignore[operator]


def _guarded_os_open(path, flags, *args, **kwargs):
    if _enforcing and (flags & _WRITE_FLAGS) and _is_guarded(path):
        _flag(path, oct(flags), "os.open")
    return _orig["os.open"](path, flags, *args, **kwargs)  # type: ignore[operator]


def _guarded_mkdir(path, *args, **kwargs):
    # Only a NEW directory is a tree change; mkdir(exist_ok=True) on an existing dir (the
    # common `parent.mkdir(parents=True, exist_ok=True)` setup on runtime/) is a no-op.
    if _enforcing and _is_guarded(path) and not _path_exists(path):
        _flag(path, "mkdir", "os.mkdir")
    return _orig["os.mkdir"](path, *args, **kwargs)  # type: ignore[operator]


def _guarded_rename(src, dst, *args, **kwargs):
    if _enforcing:
        hit = _is_guarded(dst) or _is_guarded(src)
        if hit:
            _flag(hit, "rename", "os.rename")
    return _orig["os.rename"](src, dst, *args, **kwargs)  # type: ignore[operator]


def _guarded_replace(src, dst, *args, **kwargs):
    if _enforcing:
        hit = _is_guarded(dst) or _is_guarded(src)
        if hit:
            _flag(hit, "replace", "os.replace")
    return _orig["os.replace"](src, dst, *args, **kwargs)  # type: ignore[operator]


def _guarded_remove(path, *args, **kwargs):
    # Deleting an existing guarded file modifies the tree; removing an absent path is a no-op
    # (would raise FileNotFoundError anyway) so it is not flagged.
    if _enforcing and _is_guarded(path) and _path_exists(path):
        _flag(path, "remove", "os.remove")
    return _orig["os.remove"](path, *args, **kwargs)  # type: ignore[operator]


def _guarded_unlink(path, *args, **kwargs):
    if _enforcing and _is_guarded(path) and _path_exists(path):
        _flag(path, "unlink", "os.unlink")
    return _orig["os.unlink"](path, *args, **kwargs)  # type: ignore[operator]


def _guarded_rmdir(path, *args, **kwargs):
    if _enforcing and _is_guarded(path) and _path_exists(path):
        _flag(path, "rmdir", "os.rmdir")
    return _orig["os.rmdir"](path, *args, **kwargs)  # type: ignore[operator]


# --------------------------------------------------------------------------- #
# Install / uninstall (idempotent).
# --------------------------------------------------------------------------- #
def install() -> None:
    global _installed
    if _installed:
        return
    _orig["open"] = builtins.open
    _orig["os.open"] = os.open
    _orig["os.mkdir"] = os.mkdir
    _orig["os.rename"] = os.rename
    _orig["os.replace"] = os.replace
    _orig["os.remove"] = os.remove
    _orig["os.unlink"] = os.unlink
    _orig["os.rmdir"] = os.rmdir

    builtins.open = _guarded_open  # type: ignore[assignment]
    io.open = _guarded_open  # type: ignore[assignment]  # pathlib.Path.open funnels here
    os.open = _guarded_os_open  # type: ignore[assignment]
    os.mkdir = _guarded_mkdir  # type: ignore[assignment]
    os.rename = _guarded_rename  # type: ignore[assignment]
    os.replace = _guarded_replace  # type: ignore[assignment]
    os.remove = _guarded_remove  # type: ignore[assignment]
    os.unlink = _guarded_unlink  # type: ignore[assignment]
    os.rmdir = _guarded_rmdir  # type: ignore[assignment]
    _installed = True


def uninstall() -> None:
    global _installed
    if not _installed:
        return
    builtins.open = _orig["open"]  # type: ignore[assignment]
    io.open = _orig["open"]  # type: ignore[assignment]
    os.open = _orig["os.open"]  # type: ignore[assignment]
    os.mkdir = _orig["os.mkdir"]  # type: ignore[assignment]
    os.rename = _orig["os.rename"]  # type: ignore[assignment]
    os.replace = _orig["os.replace"]  # type: ignore[assignment]
    os.remove = _orig["os.remove"]  # type: ignore[assignment]
    os.unlink = _orig["os.unlink"]  # type: ignore[assignment]
    os.rmdir = _orig["os.rmdir"]  # type: ignore[assignment]
    _installed = False


@contextlib.contextmanager
def expect_blocked():
    """Assert the wrapped block triggers exactly one guard block, and CONSUME the record so
    the conftest autouse teardown does not also fail the test. For guard self-tests only."""
    try:
        yield
    except RepoWriteLeakError:
        take_attempts()  # consume so the per-test teardown check sees a clean slate
        return
    raise AssertionError("expected a RepoWriteLeakError (guarded write) but none was raised")
