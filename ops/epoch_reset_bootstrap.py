#!/usr/bin/env python3
"""ops/epoch_reset_bootstrap.py

Guarded, dry-run-first, idempotent CHAD-side clean reset for a fresh
Epoch-3 paper baseline. Resolves the four master-ledger must-resolve hazards:

  H2 — idempotency-DB / runtime split-brain (the two exec-dedup sqlite DBs)
  H3 — phantom $313k / $141k high-water-mark (equity_history + HWM carriers)
  H4 — epoch boundary honored by only one of several stats engines
        (H4-A epoch bump; H4-B day-file fence; second-stats-family caches)
  H5 — guard / FIFO-queue split-brain (trade_closer queues + position_guard)

DESIGN PRINCIPLES
-----------------
* Never hard-deletes. Every file that is mutated, truncated, or removed is
  archived FIRST to runtime/archive/epoch_2_pre_<UTC>/ with a SHA256SUMS file
  and an archive_manifest.json. Day-files are fenced in place by renaming to
  `.scr_reset_bak` (data preserved on disk, fenced from the live globs) and
  recorded in the manifest with their pre-rename sha256.
* Dry-run is the DEFAULT. It prints the full plan + blast radius and mutates
  NOTHING. `--apply` requires the typed token `--confirm RESET-EPOCH-3`.
  `--apply` without the token degrades to dry-run.
* Fail-closed gates: refuses unless execution mode is not-live, ibkr_dry_run is
  True, and ready_for_live is False (read from live_readiness.json).
* Idempotent: a completed reset writes a marker (runtime/epoch_reset_state.json).
  Re-running the same reset is a no-op ("already-archived detected").
* Touches ONLY runtime/ and data/trades/. A hard protected-path guard rejects
  any target that resolves outside those roots or into a protected prefix
  (chad/, venv/, .claude/, audits/, /etc/chad) or onto a protected config file.
* No git operations, no broker calls, no systemd, no Bucket-C config edits.

The CHAD-side behavioral facts this tool relies on (verified read-before-write):

* position_guard.json — absent file rebuilds CLEAN: `_load_state()` returns {}
  when the file is missing (chad/core/position_guard.py:61-68); the paper-ledger
  rebuilder (chad/core/live_loop.py:621) mirrors trade_closer_state.queues, and
  the broker rebuilder (live_loop.py:762) closes any stale entry against a flat
  broker book. So archive (remove) it -> next cycle rebuilds an all-closed guard.
* trade_closer queues — clearing `queues` is safe ONLY if `processed_fill_ids`
  is RETAINED. load_state() seeds processed ids from trade_history (skipping
  fenced `.scr_reset_bak` files) then unions the state file; process_fills()
  re-opens any FILLS_<today> fill NOT in processed_fill_ids
  (chad/execution/trade_closer.py:559-623, 692-, 792-). This tool clears
  `queues` to [] and KEEPS `processed_fill_ids` — the self-revert-safe form.
* idempotency DBs — both self-recreate empty with correct schema on next connect
  (CREATE TABLE IF NOT EXISTS; chad/execution/ibkr_adapter.py:822 and
  chad/execution/idempotency_store.py:53). WAL/SHM sidecars are archived with
  the main file. Archived both-or-neither.
* equity_history.ndjson — the publisher reads only the set of existing date_utc
  values for same-day dedup, never a "last equity" delta
  (chad/ops/equity_history_publisher.py:38-131). An empty file reseeds forward
  from broker-authoritative CAD with no seed row needed.
* second stats family — expectancy/winner_scaler/strategy_health/edge_decay and
  the throttle gate read ALL trade history with NO epoch awareness; each
  persists a derived state file that survives a reset. Bumping epoch_state alone
  is insufficient, so those caches are archived too and regenerate from the
  fenced ledger.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Ensure the repo root is importable so the canonical `chad.*` gate readers work
# regardless of launch style (python3 ops/epoch_reset_bootstrap.py vs -m). If the
# import still fails, check_gates() fails closed (refuses).
_REPO_ROOT_BOOTSTRAP = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT_BOOTSTRAP not in sys.path:
    sys.path.insert(0, _REPO_ROOT_BOOTSTRAP)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

CONFIRM_TOKEN = "RESET-EPOCH-3"

MANIFEST_SCHEMA = "epoch_reset_manifest.v1"
MARKER_SCHEMA = "epoch_reset_state.v1"
MARKER_FILENAME = "epoch_reset_state.json"
EPOCH_STATE_FILENAME = "epoch_state.json"
LIVE_READINESS_FILENAME = "live_readiness.json"

# Hazard buckets (for reporting / manifest grouping)
H2 = "H2:idempotency-dbs"
H3 = "H3:phantom-hwm"
H3S = "H3:hwm-carriers(supplementary)"
H4A = "H4-A:epoch-bump"
H4B = "H4-B:day-file-fence"
H4F = "H4:second-stats-family-cache"
H5 = "H5:queues+guard"

# The two execution-dedup sqlite DBs (archived both-or-neither). Sidecars are
# discovered dynamically (`-wal`, `-shm`).
DB_FILES = ("ibkr_adapter_state.sqlite3", "exec_state_paper.sqlite3")
DB_SIDECAR_SUFFIXES = ("-wal", "-shm")

# H3 high-water-mark carriers (stale phantom values; regenerate from history).
HWM_CARRIER_FILES = (
    "drawdown_state.json",         # holds the $313k CAD phantom hwm_cad
    "withdrawal_authorization.json",  # holds the $141k true-USD hwm
    "business_phase.json",         # mirrors withdrawal_authorization hwm
)

# H4 second-stats-family derived caches (no epoch awareness; survive a reset).
SECOND_FAMILY_FILES = (
    "expectancy_state.json",
    "winner_scaling.json",
    "strategy_health.json",
    "strategy_allocations.json",   # highest risk: sticky edge-decay halts
    "setup_family_expectancy.json",
    "strategy_throttle_state.json",
)

# Canonical day-file name: trade_history_YYYYMMDD.ndjson  (no suffix variants).
_DAYFILE_RE = re.compile(r"^trade_history_(\d{8})\.ndjson$")
FENCE_SUFFIX = ".scr_reset_bak"

# Protected-path defense in depth. No target may resolve into any of these.
_PROTECTED_SUBSTRINGS = (
    os.sep + "chad" + os.sep,
    os.sep + "venv" + os.sep,
    os.sep + ".claude" + os.sep,
    os.sep + "audits" + os.sep,
    os.sep + "etc" + os.sep + "chad",
    os.sep + ".git" + os.sep,
)
_PROTECTED_BASENAMES = (
    "tiers.json",
    "withdrawal_policy.json",
)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _iso_z(dt: _dt.datetime) -> str:
    return dt.astimezone(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp(dt: _dt.datetime) -> str:
    return dt.astimezone(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_json(path: Path, obj) -> None:
    _atomic_write_text(path, json.dumps(obj, indent=2, default=str) + "\n")


def _git_head_best_effort(repo_root: Path) -> Optional[str]:
    """Read .git/HEAD by plain file I/O (NO git binary invoked). Best-effort
    provenance only; returns None on any error so we never depend on git."""
    try:
        head = (repo_root / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    except Exception:
        return None
    if head.startswith("ref:"):
        ref = head.split(":", 1)[1].strip()
        try:
            return (repo_root / ".git" / ref).read_text(encoding="utf-8").strip()
        except Exception:
            return None
    return head or None


def _repo_root() -> Path:
    # ops/ lives directly under the repo root.
    return Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Reset context
# --------------------------------------------------------------------------- #

@dataclass
class ResetContext:
    repo_root: Path
    runtime_dir: Path
    data_trades_dir: Path
    archive_root: Path
    now: _dt.datetime
    run_stamp: str
    archive_dir: Path
    cutoff_mode: str            # "before-today" | "all"
    include_hwm_carriers: bool
    epoch_label_override: Optional[str]

    @property
    def marker_path(self) -> Path:
        return self.runtime_dir / MARKER_FILENAME


def resolve_context(args: argparse.Namespace) -> ResetContext:
    repo_root = _repo_root()

    if args.runtime_dir:
        runtime_dir = Path(args.runtime_dir).resolve()
    else:
        env = os.environ.get("CHAD_RUNTIME_DIR", "").strip()
        runtime_dir = Path(env).resolve() if env else (repo_root / "runtime")

    if args.data_trades_dir:
        data_trades_dir = Path(args.data_trades_dir).resolve()
    else:
        data_trades_dir = repo_root / "data" / "trades"

    now = _utcnow()
    stamp = _stamp(now)
    archive_root = runtime_dir / "archive"
    archive_dir = archive_root / f"epoch_2_pre_{stamp}"

    return ResetContext(
        repo_root=repo_root,
        runtime_dir=runtime_dir,
        data_trades_dir=data_trades_dir,
        archive_root=archive_root,
        now=now,
        run_stamp=stamp,
        archive_dir=archive_dir,
        cutoff_mode=args.cutoff,
        include_hwm_carriers=not args.no_hwm_carriers,
        epoch_label_override=args.epoch_label,
    )


# --------------------------------------------------------------------------- #
# Fail-closed gates
# --------------------------------------------------------------------------- #

@dataclass
class GateResult:
    exec_mode: str
    ibkr_dry_run: Optional[bool]
    ready_for_live: Optional[bool]
    ready_source: str
    refusals: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.refusals


def check_gates(ctx: ResetContext) -> GateResult:
    """Evaluate the three fail-closed gates. Returns a GateResult; .ok is True
    only when every gate passes. Reads exec mode / ibkr_dry_run via the canonical
    chad readers (env-driven) and ready_for_live from the runtime live_readiness
    file (TTL-aware, fail-closed)."""
    refusals: List[str] = []
    exec_mode = "<unknown>"
    ibkr_dry_run: Optional[bool] = None

    # (a) execution mode + (b) ibkr_dry_run — canonical readers (env-driven).
    try:
        from chad.execution.execution_config import (  # type: ignore
            get_execution_mode,
            get_execution_config,
        )
        mode = get_execution_mode()
        exec_mode = getattr(mode, "value", str(mode))
        is_live = bool(getattr(mode, "name", "").upper().find("LIVE") >= 0)
        # Robust live detection across enum spellings.
        if exec_mode.lower() in ("live", "ibkr_live"):
            is_live = True
        if is_live:
            refusals.append(
                f"execution mode is LIVE ({exec_mode}); reset refuses on live."
            )
        cfg = get_execution_config()
        ibkr_dry_run = bool(getattr(cfg, "ibkr_dry_run", False))
        if not ibkr_dry_run:
            refusals.append(
                f"ibkr_dry_run is {ibkr_dry_run}; reset requires dry-run broker."
            )
    except Exception as exc:  # fail closed if we cannot prove not-live
        refusals.append(f"cannot read execution config ({exc!r}); fail-closed.")

    # (c) ready_for_live — must be present AND false (fail-closed on absence).
    ready_for_live: Optional[bool] = None
    ready_path = ctx.runtime_dir / LIVE_READINESS_FILENAME
    ready_source = str(ready_path)
    if not ready_path.is_file():
        refusals.append(
            f"live_readiness.json absent at {ready_path}; cannot confirm "
            "ready_for_live=false (fail-closed)."
        )
    else:
        try:
            obj = json.loads(ready_path.read_text(encoding="utf-8"))
            ready_for_live = bool(obj.get("ready_for_live", False))
            # TTL freshness (mirrors live_gate fail-closed semantics): a stale
            # file only makes the system MORE not-ready, so stale+false is still
            # safe-to-reset; we surface staleness but do not refuse on it.
            ts_raw = obj.get("ts_utc")
            ttl = obj.get("ttl_seconds")
            stale = False
            if isinstance(ts_raw, str) and isinstance(ttl, (int, float)):
                try:
                    ts = ts_raw[:-1] + "+00:00" if ts_raw.endswith("Z") else ts_raw
                    age = (ctx.now - _dt.datetime.fromisoformat(ts)).total_seconds()
                    stale = age > float(ttl)
                except Exception:
                    stale = False
            ready_source = f"{ready_path} (ready_for_live={ready_for_live}" + (
                ", STALE)" if stale else ")"
            )
            if ready_for_live:
                refusals.append(
                    "ready_for_live=true; system is live-ready, reset refuses."
                )
        except Exception as exc:
            refusals.append(f"live_readiness.json unreadable ({exc!r}); fail-closed.")

    return GateResult(
        exec_mode=exec_mode,
        ibkr_dry_run=ibkr_dry_run,
        ready_for_live=ready_for_live,
        ready_source=ready_source,
        refusals=refusals,
    )


# --------------------------------------------------------------------------- #
# Epoch label resolution + completion marker (idempotency)
# --------------------------------------------------------------------------- #

def _read_epoch_state(ctx: ResetContext) -> Optional[dict]:
    p = ctx.runtime_dir / EPOCH_STATE_FILENAME
    if not p.is_file():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _increment_epoch_label(label: str) -> Optional[str]:
    """Increment the trailing integer of an epoch label.
    'CHAD_v8.9_Paper_Epoch_2' -> 'CHAD_v8.9_Paper_Epoch_3'. Returns None when
    no trailing integer can be found (caller must require --epoch-label)."""
    m = re.search(r"(\d+)(\D*)$", label or "")
    if not m:
        return None
    num = int(m.group(1)) + 1
    start, end = m.span(1)
    return label[:start] + str(num) + label[end:]


def resolve_target_label(ctx: ResetContext, current_label: str) -> (str):
    if ctx.epoch_label_override:
        return ctx.epoch_label_override
    nxt = _increment_epoch_label(current_label)
    if nxt is None:
        raise SystemExit(
            "ERROR: cannot auto-derive next epoch label from "
            f"{current_label!r}; pass --epoch-label explicitly."
        )
    return nxt


def detect_completed_reset(ctx: ResetContext, current_label: str) -> Optional[dict]:
    """Return the completion marker iff a prior reset already landed for the
    current live epoch (and its archive still exists) AND the operator did not
    request a different target via --epoch-label. This is the idempotency key:
    re-running the same reset is a no-op."""
    if not ctx.marker_path.is_file():
        return None
    try:
        marker = json.loads(ctx.marker_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    target = marker.get("target_epoch_label")
    arch = marker.get("archive_dir")
    if target != current_label:
        return None
    if not (arch and Path(arch).exists()):
        return None
    # A fresh, different override means a deliberate NEW reset (e.g. 3->4).
    if ctx.epoch_label_override and ctx.epoch_label_override != current_label:
        return None
    return marker


# --------------------------------------------------------------------------- #
# Plan model
# --------------------------------------------------------------------------- #

@dataclass
class PlannedAction:
    hazard: str
    label: str
    src: Path
    live_action: str        # edit_epoch|truncate|clear_queues|remove|rename_fence
    archive_mode: str       # copy|rename_in_place
    status: str = "PLANNED"  # PLANNED|SKIP_ABSENT|SKIP_DONE|DONE|ERROR
    detail: str = ""
    archive_name: Optional[str] = None
    sha256: Optional[str] = None
    bytes: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "hazard": self.hazard,
            "label": self.label,
            "src": str(self.src),
            "live_action": self.live_action,
            "archive_mode": self.archive_mode,
            "status": self.status,
            "detail": self.detail,
            "archive_name": self.archive_name,
            "sha256": self.sha256,
            "bytes": self.bytes,
        }


# --------------------------------------------------------------------------- #
# Plan builders (pure — no mutation)
# --------------------------------------------------------------------------- #

def _present(p: Path) -> bool:
    try:
        return p.is_file()
    except Exception:
        return False


def plan_epoch_bump(ctx: ResetContext, current_label: str, target_label: str) -> List[PlannedAction]:
    p = ctx.runtime_dir / EPOCH_STATE_FILENAME
    a = PlannedAction(
        hazard=H4A, label="epoch_state.json -> bump to new epoch",
        src=p, live_action="edit_epoch", archive_mode="copy",
        archive_name=EPOCH_STATE_FILENAME,
    )
    if not _present(p):
        a.status, a.detail = "SKIP_ABSENT", "epoch_state.json absent"
    elif current_label == target_label:
        a.status = "SKIP_DONE"
        a.detail = f"active_epoch already == {target_label!r}"
    else:
        a.detail = (
            f"active_epoch {current_label!r} -> {target_label!r}; "
            f"epoch_started_at_utc -> {_iso_z(ctx.now)}; "
            "previous_epoch_archive -> <archive dir>"
        )
    return [a]


def plan_fence_dayfiles(ctx: ResetContext) -> List[PlannedAction]:
    out: List[PlannedAction] = []
    d = ctx.data_trades_dir
    if not d.is_dir():
        return out
    today = ctx.now.astimezone(_dt.timezone.utc).strftime("%Y%m%d")
    for entry in sorted(d.iterdir()):
        if not entry.is_file():
            continue
        m = _DAYFILE_RE.match(entry.name)
        if not m:
            continue  # ignore non-canonical (already-fenced .scr_reset_bak, .bak, etc.)
        date_str = m.group(1)
        if ctx.cutoff_mode == "before-today" and not (date_str < today):
            continue  # leave same-day / future file alone in before-today mode
        fenced = entry.with_name(entry.name + FENCE_SUFFIX)
        a = PlannedAction(
            hazard=H4B, label=f"fence {entry.name}",
            src=entry, live_action="rename_fence", archive_mode="rename_in_place",
            archive_name=fenced.name,
            detail=f"{entry.name} -> {fenced.name}",
        )
        if fenced.exists():
            a.status, a.detail = "SKIP_DONE", f"{fenced.name} already exists"
        out.append(a)
    return out


def plan_dbs(ctx: ResetContext) -> List[PlannedAction]:
    """Both-or-neither: if exactly one main DB is present, raise (never half)."""
    mains = [ctx.runtime_dir / n for n in DB_FILES]
    present = [p for p in mains if _present(p)]
    if len(present) == 1:
        raise SystemExit(
            "ERROR (H2 both-or-neither): exactly one idempotency DB present "
            f"({present[0].name}); refusing to half-reset the dedup pair. "
            "Resolve manually before re-running."
        )
    out: List[PlannedAction] = []
    for main in mains:
        files = [main] + [main.with_name(main.name + sfx) for sfx in DB_SIDECAR_SUFFIXES]
        for f in files:
            a = PlannedAction(
                hazard=H2, label=f"archive+remove {f.name}",
                src=f, live_action="remove", archive_mode="copy",
                archive_name=f.name,
                detail="self-recreates empty (CREATE TABLE IF NOT EXISTS) on restart",
            )
            if not _present(f):
                a.status, a.detail = "SKIP_ABSENT", f"{f.name} absent"
            out.append(a)
    return out


def plan_equity(ctx: ResetContext) -> List[PlannedAction]:
    out: List[PlannedAction] = []
    eq = ctx.runtime_dir / "equity_history.ndjson"
    a = PlannedAction(
        hazard=H3, label="equity_history.ndjson -> archive+truncate",
        src=eq, live_action="truncate", archive_mode="copy",
        archive_name="equity_history.ndjson",
        detail="reseeds forward from broker-authoritative CAD (no seed row needed)",
    )
    if not _present(eq):
        a.status, a.detail = "SKIP_ABSENT", "equity_history.ndjson absent"
    elif eq.stat().st_size == 0:
        a.status, a.detail = "SKIP_DONE", "already empty"
    out.append(a)

    if ctx.include_hwm_carriers:
        for name in HWM_CARRIER_FILES:
            p = ctx.runtime_dir / name
            ca = PlannedAction(
                hazard=H3S, label=f"archive+remove {name}",
                src=p, live_action="remove", archive_mode="copy",
                archive_name=name,
                detail="stale HWM carrier; recomputes from history next cycle",
            )
            if not _present(p):
                ca.status, ca.detail = "SKIP_ABSENT", f"{name} absent"
            out.append(ca)
    return out


def plan_queues_guard(ctx: ResetContext) -> List[PlannedAction]:
    out: List[PlannedAction] = []

    tc = ctx.runtime_dir / "trade_closer_state.json"
    a = PlannedAction(
        hazard=H5, label="trade_closer_state.json -> clear queues (retain processed_fill_ids)",
        src=tc, live_action="clear_queues", archive_mode="copy",
        archive_name="trade_closer_state.json",
        detail="queues -> []; processed_fill_ids RETAINED (self-revert-safe)",
    )
    if not _present(tc):
        a.status, a.detail = "SKIP_ABSENT", "trade_closer_state.json absent"
    else:
        try:
            cur = json.loads(tc.read_text(encoding="utf-8"))
            if isinstance(cur, dict) and not (cur.get("queues") or []):
                a.status, a.detail = "SKIP_DONE", "queues already empty"
        except Exception:
            pass
    out.append(a)

    pg = ctx.runtime_dir / "position_guard.json"
    g = PlannedAction(
        hazard=H5, label="position_guard.json -> archive+remove (rebuilds clean)",
        src=pg, live_action="remove", archive_mode="copy",
        archive_name="position_guard.json",
        detail="absent file rebuilds all-closed from empty queues + flat broker",
    )
    if not _present(pg):
        g.status, g.detail = "SKIP_ABSENT", "position_guard.json absent"
    out.append(g)
    return out


def plan_second_family(ctx: ResetContext) -> List[PlannedAction]:
    out: List[PlannedAction] = []
    for name in SECOND_FAMILY_FILES:
        p = ctx.runtime_dir / name
        a = PlannedAction(
            hazard=H4F, label=f"archive+remove {name}",
            src=p, live_action="remove", archive_mode="copy",
            archive_name=name,
            detail="no epoch awareness; regenerates from fenced ledger next cycle",
        )
        if not _present(p):
            a.status, a.detail = "SKIP_ABSENT", f"{name} absent"
        out.append(a)
    return out


def build_plan(ctx: ResetContext, current_label: str, target_label: str) -> List[PlannedAction]:
    plan: List[PlannedAction] = []
    plan += plan_epoch_bump(ctx, current_label, target_label)
    plan += plan_fence_dayfiles(ctx)
    plan += plan_dbs(ctx)
    plan += plan_equity(ctx)
    plan += plan_queues_guard(ctx)
    plan += plan_second_family(ctx)
    return plan


# --------------------------------------------------------------------------- #
# Protected-path guard (defense in depth, applied before every mutation)
# --------------------------------------------------------------------------- #

def assert_safe_target(ctx: ResetContext, path: Path) -> None:
    rp = str(path.resolve()) if path.exists() else str(Path(os.path.abspath(str(path))))
    # Must live under runtime/ or data/trades/.
    roots = (str(ctx.runtime_dir.resolve()), str(ctx.data_trades_dir.resolve()))
    if not any(rp == r or rp.startswith(r + os.sep) for r in roots):
        raise SystemExit(f"PROTECTED-PATH GUARD: {rp} is outside runtime/ and data/trades/.")
    low = rp
    for sub in _PROTECTED_SUBSTRINGS:
        if sub in low:
            raise SystemExit(f"PROTECTED-PATH GUARD: {rp} hits protected prefix {sub!r}.")
    if Path(rp).name in _PROTECTED_BASENAMES:
        raise SystemExit(f"PROTECTED-PATH GUARD: {rp} is a protected config file.")


# --------------------------------------------------------------------------- #
# Execution (archive-first, then mutate)
# --------------------------------------------------------------------------- #

def _epoch_bump_payload(cur: dict, target_label: str, ctx: ResetContext) -> dict:
    out = dict(cur)
    out["active_epoch"] = target_label
    out["epoch_started_at_utc"] = _iso_z(ctx.now)
    out["previous_epoch_archive"] = str(ctx.archive_dir)
    out.setdefault("schema_version", "epoch_state.v1")
    out.setdefault("paper_only", True)
    out.setdefault("ready_for_live", False)
    # Never flip these here — preserve whatever was set (must already be safe).
    return out


def execute_plan(
    ctx: ResetContext,
    plan: List[PlannedAction],
    current_label: str,
    target_label: str,
    gates: GateResult,
) -> dict:
    """Phase A: archive every actionable file (copy / record sha). Then write
    SHA256SUMS + manifest. Phase B: perform the live mutations. The complete,
    hashed archive therefore always exists BEFORE any live mutation."""
    ctx.archive_dir.mkdir(parents=True, exist_ok=False)

    sha_lines: List[str] = []
    files_archived: List[dict] = []
    day_files_fenced: List[dict] = []

    # ---- Phase A: archive --------------------------------------------------
    for a in plan:
        if a.status in ("SKIP_ABSENT", "SKIP_DONE"):
            continue
        assert_safe_target(ctx, a.src)
        a.sha256 = _sha256_file(a.src)
        a.bytes = a.src.stat().st_size
        if a.archive_mode == "copy":
            dest = ctx.archive_dir / a.archive_name
            shutil.copy2(a.src, dest)
            sha_lines.append(f"{a.sha256}  {a.archive_name}")
            files_archived.append({
                "name": a.archive_name, "src": str(a.src), "sha256": a.sha256,
                "bytes": a.bytes, "live_action": a.live_action, "hazard": a.hazard,
                "detail": a.detail,
            })
        elif a.archive_mode == "rename_in_place":
            day_files_fenced.append({
                "src": str(a.src), "renamed_to": a.archive_name,
                "sha256": a.sha256, "bytes": a.bytes,
            })

    # ---- manifest + SHA256SUMS (written BEFORE any live mutation) ----------
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "archive_name": ctx.archive_dir.name,
        "created_at_utc": _iso_z(ctx.now),
        "reason": "CHAD-side Epoch reset (H2/H3/H4/H5) — paper baseline",
        "tool": "ops/epoch_reset_bootstrap.py",
        "git_head_best_effort": _git_head_best_effort(ctx.repo_root),
        "from_epoch": current_label,
        "to_epoch": target_label,
        "cutoff_mode": ctx.cutoff_mode,
        "include_hwm_carriers": ctx.include_hwm_carriers,
        "gates": {
            "exec_mode": gates.exec_mode,
            "ibkr_dry_run": gates.ibkr_dry_run,
            "ready_for_live": gates.ready_for_live,
            "ready_source": gates.ready_source,
        },
        "files_archived": files_archived,
        "files_archived_count": len(files_archived),
        "day_files_fenced": day_files_fenced,
        "day_files_fenced_count": len(day_files_fenced),
    }
    _atomic_write_text(ctx.archive_dir / "SHA256SUMS", "\n".join(sha_lines) + "\n")
    _atomic_write_json(ctx.archive_dir / "archive_manifest.json", manifest)

    # ---- Phase B: live mutations ------------------------------------------
    for a in plan:
        if a.status in ("SKIP_ABSENT", "SKIP_DONE"):
            continue
        assert_safe_target(ctx, a.src)
        try:
            if a.live_action == "edit_epoch":
                cur = _read_epoch_state(ctx) or {}
                _atomic_write_json(a.src, _epoch_bump_payload(cur, target_label, ctx))
            elif a.live_action == "truncate":
                _atomic_write_text(a.src, "")
            elif a.live_action == "clear_queues":
                cur = json.loads(a.src.read_text(encoding="utf-8"))
                payload = {
                    "processed_fill_ids": sorted(cur.get("processed_fill_ids", []) or []),
                    "queues": [],
                    "saved_at_utc": _iso_z(ctx.now),
                }
                _atomic_write_json(a.src, payload)
            elif a.live_action == "remove":
                a.src.unlink()
            elif a.live_action == "rename_fence":
                a.src.rename(a.src.with_name(a.src.name + FENCE_SUFFIX))
            else:
                raise RuntimeError(f"unknown live_action {a.live_action!r}")
            a.status = "DONE"
        except Exception as exc:
            a.status = "ERROR"
            a.detail = f"{a.detail} | MUTATION FAILED: {exc!r}"
            raise

    # ---- completion marker (idempotency sentinel) -------------------------
    marker = {
        "schema_version": MARKER_SCHEMA,
        "target_epoch_label": target_label,
        "from_epoch": current_label,
        "archive_dir": str(ctx.archive_dir),
        "manifest": str(ctx.archive_dir / "archive_manifest.json"),
        "completed_at_utc": _iso_z(ctx.now),
        "files_archived_count": len(files_archived),
        "day_files_fenced_count": len(day_files_fenced),
    }
    _atomic_write_json(ctx.marker_path, marker)

    return {"manifest": manifest, "marker": marker}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _fmt_bytes(n: Optional[int]) -> str:
    if n is None:
        return "-"
    return f"{n:,}"


def _action_bytes(a: PlannedAction) -> int:
    """Bytes this action would preserve, resolved for display (no mutation).

    APPLY stats each source at archive time (a.bytes set in execute_plan Phase
    A), so the rendered figure matches exactly what was archived. DRY-RUN never
    mutates, so a.bytes is None and we stat the still-present live source here —
    this is what makes the preview total truthful. Non-actionable actions
    (SKIP_ABSENT/SKIP_DONE/ERROR) and sources that have gone missing contribute
    0, so the reported total is zero only when nothing is genuinely preserved.
    """
    if a.status not in ("PLANNED", "DONE"):
        return 0
    if a.bytes is not None:
        return a.bytes
    try:
        return a.src.stat().st_size
    except OSError:
        return 0


def plan_preserved_bytes(plan: List[PlannedAction]) -> tuple:
    """(copy_bytes, fence_bytes) the plan would preserve. copy_bytes are moved
    into the archive dir (the sole rollback source); fence_bytes are renamed to
    `.scr_reset_bak` and stay on disk in place. Display-only — never mutates."""
    copy_b = sum(_action_bytes(a) for a in plan if a.archive_mode == "copy")
    fence_b = sum(_action_bytes(a) for a in plan if a.archive_mode == "rename_in_place")
    return copy_b, fence_b


def render_plan(ctx: ResetContext, plan: List[PlannedAction], gates: GateResult,
                current_label: str, target_label: str, mode: str) -> str:
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append(f"CHAD Epoch-Reset Bootstrap — {mode}")
    lines.append("=" * 78)
    lines.append(f"  repo_root        : {ctx.repo_root}")
    lines.append(f"  runtime_dir      : {ctx.runtime_dir}")
    lines.append(f"  data/trades dir  : {ctx.data_trades_dir}")
    lines.append(f"  archive_dir      : {ctx.archive_dir}")
    lines.append(f"  run_utc          : {_iso_z(ctx.now)}")
    lines.append(f"  from_epoch       : {current_label}")
    lines.append(f"  to_epoch         : {target_label}")
    lines.append(f"  cutoff_mode      : {ctx.cutoff_mode}")
    lines.append(f"  hwm_carriers     : {'INCLUDED' if ctx.include_hwm_carriers else 'EXCLUDED'}")
    lines.append("")
    lines.append("GATES (fail-closed):")
    lines.append(f"  exec_mode        : {gates.exec_mode}")
    lines.append(f"  ibkr_dry_run     : {gates.ibkr_dry_run}")
    lines.append(f"  ready_for_live   : {gates.ready_for_live}  [{gates.ready_source}]")
    lines.append(f"  gate status      : {'PASS' if gates.ok else 'REFUSE'}")
    if gates.refusals:
        for r in gates.refusals:
            lines.append(f"    - REFUSAL: {r}")
    lines.append("")

    # group by hazard
    by_haz: dict = {}
    for a in plan:
        by_haz.setdefault(a.hazard, []).append(a)

    act = sum(1 for a in plan if a.status == "PLANNED" or a.status == "DONE")
    skip = sum(1 for a in plan if a.status.startswith("SKIP"))
    copy_bytes, fence_bytes = plan_preserved_bytes(plan)
    fence_n = sum(1 for a in plan if a.live_action == "rename_fence" and a.status in ("PLANNED", "DONE"))

    lines.append("PLAN / BLAST RADIUS:")
    for haz in sorted(by_haz):
        lines.append(f"  [{haz}]")
        for a in by_haz[haz]:
            flag = {
                "PLANNED": "WOULD" if mode != "APPLY" else "PLAN",
                "DONE": "DONE ", "SKIP_ABSENT": "skip ", "SKIP_DONE": "skip ",
                "ERROR": "ERROR",
            }.get(a.status, a.status)
            nb = _action_bytes(a)
            szs = f"  ({_fmt_bytes(nb)} B)" if nb else ""
            lines.append(f"    {flag:5s} {a.live_action:13s} {a.label}{szs}")
            if a.detail:
                lines.append(f"          - {a.detail}")
    lines.append("")
    lines.append("SUMMARY:")
    lines.append(f"  actionable ops          : {act}")
    lines.append(f"  skipped (no-op)         : {skip}")
    lines.append(f"  day-files fenced        : {fence_n}")
    lines.append(f"  bytes -> archive dir    : {_fmt_bytes(copy_bytes)}"
                 "  (copied; sole rollback source)")
    lines.append(f"  bytes fenced in place   : {_fmt_bytes(fence_bytes)}"
                 "  (.scr_reset_bak renames; data stays on disk)")
    lines.append(f"  bytes preserved total   : {_fmt_bytes(copy_bytes + fence_bytes)}")
    lines.append("=" * 78)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="epoch_reset_bootstrap",
        description="Guarded, dry-run-first, idempotent CHAD-side epoch reset.",
    )
    p.add_argument("--apply", action="store_true",
                   help="Perform mutations. Requires --confirm RESET-EPOCH-3. "
                        "Without the token this degrades to dry-run.")
    p.add_argument("--confirm", default="",
                   help=f"Typed safety token. Must equal {CONFIRM_TOKEN!r} to mutate.")
    p.add_argument("--cutoff", choices=("before-today", "all"), default="before-today",
                   help="Day-file fence cutoff: before-today (default) fences files "
                        "strictly before today's UTC date; all fences every "
                        "canonical day-file.")
    p.add_argument("--epoch-label", default=None,
                   help="Override the new active_epoch label. Default increments "
                        "the trailing integer of the current label.")
    p.add_argument("--no-hwm-carriers", action="store_true",
                   help="Exclude the supplementary H3 HWM-carrier state files "
                        "(drawdown_state/withdrawal_authorization/business_phase). "
                        "equity_history truncation alone clears the COMPUTED HWM.")
    p.add_argument("--runtime-dir", default=None,
                   help="Override runtime dir (testing). Default CHAD_RUNTIME_DIR "
                        "or <repo>/runtime.")
    p.add_argument("--data-trades-dir", default=None,
                   help="Override data/trades dir (testing).")
    p.add_argument("--json", action="store_true",
                   help="Also emit the plan/result as JSON on stdout.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    ctx = resolve_context(args)

    epoch_obj = _read_epoch_state(ctx)
    current_label = str((epoch_obj or {}).get("active_epoch") or "")
    if not current_label:
        print("ERROR: cannot read current active_epoch from epoch_state.json; "
              f"looked in {ctx.runtime_dir / EPOCH_STATE_FILENAME}", file=sys.stderr)
        return 2

    # --- idempotency: completed reset for this epoch -> no-op ---------------
    done = detect_completed_reset(ctx, current_label)
    if done is not None:
        print("=" * 78)
        print("CHAD Epoch-Reset Bootstrap — NO-OP (already reset)")
        print("=" * 78)
        print(f"  active_epoch     : {current_label}")
        print(f"  completed_at_utc : {done.get('completed_at_utc')}")
        print(f"  archive_dir      : {done.get('archive_dir')}")
        print(f"  day-files fenced : {done.get('day_files_fenced_count')}")
        print("  Nothing to do. Pass --epoch-label for a deliberate new reset.")
        if args.json:
            print(json.dumps({"status": "noop", "marker": done}, indent=2, default=str))
        return 0

    target_label = resolve_target_label(ctx, current_label)
    gates = check_gates(ctx)

    # Build plan (plan_dbs may SystemExit on a half-DB state).
    plan = build_plan(ctx, current_label, target_label)

    # Decide mode.
    want_apply = bool(args.apply)
    token_ok = (args.confirm == CONFIRM_TOKEN)
    if want_apply and not token_ok:
        mode = "DRY-RUN (apply requested, confirm token missing/incorrect)"
        want_apply = False
    elif want_apply:
        mode = "APPLY"
    else:
        mode = "DRY-RUN (default)"

    # Gates must pass to apply. Dry-run still renders gate status.
    if want_apply and not gates.ok:
        print(render_plan(ctx, plan, gates, current_label, target_label, "REFUSED"))
        print("\nREFUSED: one or more fail-closed gates did not pass. No mutations performed.",
              file=sys.stderr)
        return 3

    if not want_apply:
        out = render_plan(ctx, plan, gates, current_label, target_label, mode)
        print(out)
        if mode.startswith("DRY-RUN"):
            print("\nDRY-RUN: nothing was mutated. Re-run with "
                  f"--apply --confirm {CONFIRM_TOKEN} to execute.")
        if args.json:
            copy_b, fence_b = plan_preserved_bytes(plan)
            print(json.dumps({
                "status": "dry-run", "mode": mode,
                "from_epoch": current_label, "to_epoch": target_label,
                "gates": {"ok": gates.ok, "refusals": gates.refusals},
                "would_preserve_bytes": {
                    "to_archive_dir": copy_b,
                    "fenced_in_place": fence_b,
                    "total": copy_b + fence_b,
                },
                "plan": [a.to_dict() for a in plan],
            }, indent=2, default=str))
        return 0

    # ---- APPLY ----
    result = execute_plan(ctx, plan, current_label, target_label, gates)
    print(render_plan(ctx, plan, gates, current_label, target_label, "APPLY"))
    print(f"\nAPPLIED. Archive: {ctx.archive_dir}")
    print(f"  manifest : {ctx.archive_dir / 'archive_manifest.json'}")
    print(f"  marker   : {ctx.marker_path}")
    if args.json:
        print(json.dumps({
            "status": "applied",
            "archive_dir": str(ctx.archive_dir),
            "result": result,
            "plan": [a.to_dict() for a in plan],
        }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
