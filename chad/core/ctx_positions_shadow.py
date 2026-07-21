#!/usr/bin/env python3
"""
chad/core/ctx_positions_shadow.py  —  Wave-2 Lane B (W2B-3)

Shadow-compare recorder for CHAD_CTX_POSITIONS. In ``shadow`` mode the live
cycle acts on the OFF (empty-book) context exactly as today, and this module
computes what the strategies WOULD emit with the injected book, logs the
signal-set diff, and closes nothing — the mechanism that lets the ON flip be
judged on evidence (the go/no-go: does injecting positions add any equity/ETF
SELL, or only remove BUYs?).

Contract (mirrors the exit-overlay / margin-shadow shadows):
  * compute the counterfactual every shadow cycle;
  * write one evidence record per cycle to ``data/ctx_positions_shadow/`` (never
    ``runtime/``), so an ON flip can be judged on a corpus;
  * write a heartbeat to ``runtime/ctx_positions_heartbeat.json`` so "on/shadow"
    stays distinguishable from "dead" (heartbeat is written in shadow AND on —
    NOT in off, which stays strictly inert / byte-identical to legacy);
  * act on NOTHING and — critically — leave the strategies' in-memory exit state
    (gamma/delta/omega ``_STATE``) exactly as the acting OFF run left it: the
    counterfactual run is bracketed by a snapshot/restore so it is provably
    side-effect-free.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional

SCHEMA_VERSION = "ctx_positions_shadow.v1"

# Position-aware strategies that carry module-level in-memory state. The
# counterfactual observation run is snapshot/restore-bracketed around these so it
# cannot perturb the acting OFF run's state.
_STATEFUL_MODULES = (
    "chad.strategies.gamma",
    "chad.strategies.delta",
    "chad.strategies.omega",
    "chad.strategies.beta",
    "chad.strategies.beta_trend",
)

_EVIDENCE_DIR_ENV = "CHAD_CTX_POSITIONS_EVIDENCE_DIR"
_HEARTBEAT_ENV = "CHAD_CTX_POSITIONS_HEARTBEAT_PATH"


# --------------------------------------------------------------------------- #
# signal-key extraction
# --------------------------------------------------------------------------- #

def _sym(sig: Any) -> str:
    return str(getattr(sig, "symbol", "") or "").upper()


def _side(sig: Any) -> str:
    s = getattr(sig, "side", None)
    return str(getattr(s, "value", None) or getattr(s, "name", None) or s or "")


def signal_keys(available: Mapping[str, List[Any]]) -> set:
    """{(strategy, symbol, side)} over a strategy->signals map."""
    out = set()
    for name, sigs in (available or {}).items():
        for s in (sigs or []):
            out.add((str(name), _sym(s), _side(s)))
    return out


# --------------------------------------------------------------------------- #
# counterfactual context (clone ctx_off, swap the book) + state isolation
# --------------------------------------------------------------------------- #

def build_ctx_on(ctx_off: Any, view: Any) -> Any:
    """ctx_off with ``portfolio.positions`` replaced by the injected book. Clones
    the acting context (identical bars/ticks/prices) rather than rebuilding, so
    the diff is a pure function of the injected positions and there is no second
    market-data load."""
    old = getattr(ctx_off, "portfolio", None)
    new_pf = SimpleNamespace(
        timestamp=getattr(old, "timestamp", None),
        cash=getattr(old, "cash", 0.0),
        positions=dict(view.positions),
        total_equity=getattr(old, "total_equity", 0.0),
        equity=getattr(old, "equity", 0.0),
        net_liq=getattr(old, "net_liq", 0.0),
        extra=dict(getattr(old, "extra", {}) or {}),
    )
    if dataclasses.is_dataclass(ctx_off):
        return dataclasses.replace(ctx_off, portfolio=new_pf)
    if hasattr(ctx_off, "__dict__"):
        return SimpleNamespace(**{**ctx_off.__dict__, "portfolio": new_pf})
    return ctx_off


def _snapshot_strategy_state() -> Dict[str, Any]:
    snaps: Dict[str, Any] = {}
    for modname in _STATEFUL_MODULES:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        st = getattr(mod, "_STATE", None)
        if st is not None:
            try:
                snaps[modname] = copy.deepcopy(st)
            except Exception:
                pass
    return snaps


def _restore_strategy_state(snaps: Mapping[str, Any]) -> None:
    for modname, snap in snaps.items():
        try:
            mod = importlib.import_module(modname)
            setattr(mod, "_STATE", snap)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# evidence + heartbeat writers (best-effort; never raise)
# --------------------------------------------------------------------------- #

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _evidence_dir(explicit: Optional[Any]) -> Path:
    if explicit is not None:
        return Path(explicit)
    env = os.getenv(_EVIDENCE_DIR_ENV)
    if env:
        return Path(env)
    return Path.cwd() / "data" / "ctx_positions_shadow"


def _heartbeat_path(explicit: Optional[Any]) -> Path:
    if explicit is not None:
        return Path(explicit)
    env = os.getenv(_HEARTBEAT_ENV)
    if env:
        return Path(env)
    return Path.cwd() / "runtime" / "ctx_positions_heartbeat.json"


def _append_ndjson(directory: Path, record: Mapping[str, Any], now: datetime) -> None:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        fname = directory / f"ctx_positions_{now.astimezone(timezone.utc).strftime('%Y%m%d')}.ndjson"
        with fname.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception:
        pass


def _write_heartbeat(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# public entry — called by the router in shadow/on modes
# --------------------------------------------------------------------------- #

def record_cycle(
    *,
    mode: str,
    regs: List[Any],
    run_handler: Any,
    ctx_off: Any,
    available_off: Mapping[str, List[Any]],
    view: Any,
    logger: Any = None,
    now: Optional[datetime] = None,
    evidence_dir: Optional[Any] = None,
    heartbeat_path: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """In shadow: compute + persist the OFF-vs-ON signal-set diff and heartbeat,
    acting on nothing and perturbing no strategy state. In on: heartbeat only.
    Returns the shadow record (shadow mode) or ``None``. Never raises.

    ``available_off`` is the signal set the cycle ACTS on: the OFF (empty-book)
    set in shadow mode; the ON (injected) set in on mode.
    """
    now = now or _utcnow()
    hb_path = _heartbeat_path(heartbeat_path)
    n_acted = len(signal_keys(available_off))

    record: Optional[Dict[str, Any]] = None
    if mode == "shadow" and view is not None and getattr(view, "known", False):
        try:
            ctx_on = build_ctx_on(ctx_off, view)
            snaps = _snapshot_strategy_state()
            try:
                available_on = {}
                for reg in regs:
                    name = reg.name.value
                    available_on[name] = run_handler(name, reg.handler, ctx_on, logger)
            finally:
                _restore_strategy_state(snaps)

            off = signal_keys(available_off)
            on = signal_keys(available_on)
            removed = sorted(off - on)
            added = sorted(on - off)
            added_sells = [k for k in added if str(k[2]).upper() == "SELL"]
            record = {
                "schema_version": SCHEMA_VERSION,
                "ts_utc": _iso(now),
                "mode": "shadow",
                "positions_status": getattr(view, "status", None),
                "n_injected": (getattr(view, "evidence", {}) or {}).get("n_injected"),
                "n_off": len(off),
                "n_on": len(on),
                "unchanged": len(off & on),
                "removed": [list(k) for k in removed],
                "added": [list(k) for k in added],
                # the go/no-go danger signal: any NEW unclamped equity/ETF SELL.
                "added_sells_count": len(added_sells),
                "added_sells": [list(k) for k in added_sells],
            }
            _append_ndjson(_evidence_dir(evidence_dir), record, now)
            if logger is not None:
                logger.info(
                    "CTX_POSITIONS_SHADOW removed=%d added=%d added_sells=%d n_injected=%s",
                    len(removed), len(added), len(added_sells), record["n_injected"],
                )
        except Exception as exc:  # pragma: no cover - best-effort
            if logger is not None:
                logger.warning("ctx_positions shadow diff failed (non-fatal): %s", exc)

    _write_heartbeat(hb_path, {
        "schema_version": "ctx_positions_heartbeat.v1",
        "ts_utc": _iso(now),
        "mode": mode,
        "positions_status": getattr(view, "status", None) if view is not None else None,
        "n_injected": (getattr(view, "evidence", {}) or {}).get("n_injected") if view is not None else None,
        "n_signals_acted": n_acted,
        "added_sells_count": (record or {}).get("added_sells_count"),
    })
    return record
