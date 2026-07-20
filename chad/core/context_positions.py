#!/usr/bin/env python3
"""
chad/core/context_positions.py  —  Wave-2 Lane B (W2B)

The single authoritative position view a strategy's ``MarketContext`` should
see: netted to **ONE** :class:`chad.types.Position` per symbol, **CHAD-attributed
only**, and **fail-closed to UNKNOWN** (never empty-as-truth).

Read-only. No side effects. Never raises.

Why this module exists
----------------------
``ContextBuilder`` already accepts ``current_positions`` but no caller passes it,
so ``ctx.portfolio.positions`` is hardwired ``{}`` and every strategy is
position-blind (root cause of the gamma re-buy churn, beta's perpetual
underweight-BUY, and dormant native exits). This module produces the value that
the W2B injection feeds into ``ContextBuilder.build(current_positions=...)``.

Sources (D1)
------------
* PRIMARY broker truth — ``runtime/positions_snapshot.json``: an *independent*
  collector (a different clientId from the guard writers), carrying already-signed
  ``position`` and ``avgCost`` per row. This is the only leg that does not share a
  source with the strategy legs, so it cannot false-flat the way the two guard
  legs can (the XOV-2345 trap).
* CROSS-CHECK mirror — ``position_guard.json`` ``broker_sync|<SYM>`` legs.
* ATTRIBUTION — ``position_guard.json`` strategy legs (``gamma|<SYM>``, ...).

Netting rule (D1 / D2 — never sum the dual-booked copies)
---------------------------------------------------------
``gamma|UNH`` and ``broker_sync|UNH`` are the SAME shares booked twice; summing
invents a 2x phantom (see ``position_guard._agg_guard_strategy`` docstring). So::

    injected_qty(S) = clamp_to_broker( strategy_net(S), broker_signed(S) )
                    = sign(broker) * min(|strategy_net|, |broker|)   # signs agree
                    = 0                                              # otherwise

Only CHAD-attributed shares that the broker confirms are injected. Operator-only
inventory (a broker position with no strategy leg — e.g. LLY) is invisible;
over-attribution (CHAD claims more than the broker holds — e.g. AAPL 14 vs 7) is
clamped down to broker truth. No strategy ever sees or acts on operator inventory.

Fail-closed (D3)
----------------
Snapshot missing / stale / malformed, or the snapshot disagrees with the guard
broker mirror beyond tolerance  ->  ``status == UNKNOWN``, ``positions == {}``.
The caller MUST treat UNKNOWN as "idle this cycle", NEVER as an empty book —
empty-as-truth is the original disease.

Asset-class classifier (D6 — MAINTENANCE SURFACE)
-------------------------------------------------
``asset_class`` reuses the exit overlay's ETF/futures allow-lists via
``position_exit_overlay._asset_class`` (single source of truth). That allow-list
is a maintenance surface: an ETF **not** on the list classifies as EQUITY.
Candidate hardening (future): a sentinel that warns on an unknown symbol instead
of silently guessing EQUITY.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from chad.types import AssetClass, Position
from chad.core import position_guard as _pg

# D6: reuse the overlay's classifier (ETF/futures allow-list). Fail-soft so the
# loader never hard-depends on the overlay import succeeding.
try:  # pragma: no cover - exercised indirectly
    from chad.risk.position_exit_overlay import _asset_class as _overlay_asset_class
except Exception:  # pragma: no cover - defensive fallback
    def _overlay_asset_class(symbol: str) -> str:  # type: ignore[misc]
        return "equity"


STATUS_KNOWN = "KNOWN"
STATUS_UNKNOWN = "UNKNOWN"

_DEFAULT_TTL_SECONDS = 300
# Snapshot vs guard broker-mirror agreement tolerance, in shares. Larger than
# _pg._V4_TOL because the two are independently-timed broker reads; a sub-share
# rounding delta must not idle the whole cycle, a real position delta must.
_MIRROR_TOL = 0.5
_SNAPSHOT_ENV = "CHAD_POSITIONS_SNAPSHOT_PATH"


@dataclass(frozen=True)
class PositionsView:
    """Result of :func:`load_context_positions`.

    ``status`` is ``KNOWN`` or ``UNKNOWN``. ``positions`` is a symbol -> Position
    map, empty whenever ``status == UNKNOWN``. ``reason`` and ``evidence`` are for
    markers / the W2B shadow record.
    """

    status: str
    positions: Mapping[str, Position]
    reason: str
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def known(self) -> bool:
        return self.status == STATUS_KNOWN


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _asset_class_enum(symbol: str) -> AssetClass:
    """Symbol -> AssetClass enum (D6). Overlay classifier returns the lowercase
    strings that ARE the enum values ('equity'/'etf'/'futures')."""
    try:
        return AssetClass(_overlay_asset_class(symbol))
    except Exception:  # pragma: no cover - defensive
        return AssetClass.EQUITY


def _clamp_to_broker(strategy_qty: float, broker_qty: float) -> float:
    """CHAD-attributed shares the broker confirms (D2): same sign as the broker,
    magnitude ``min(|strategy|, |broker|)``; ``0.0`` if the broker does not hold
    the symbol on CHAD's side."""
    if broker_qty == 0.0:
        return 0.0
    if (strategy_qty > 0.0) != (broker_qty > 0.0):
        return 0.0  # opposite sides — broker does not confirm CHAD's claim
    mag = min(abs(strategy_qty), abs(broker_qty))
    return mag if broker_qty > 0.0 else -mag


def _parse_ts(ts_raw: Any) -> Optional[datetime]:
    if not isinstance(ts_raw, str) or not ts_raw:
        return None
    try:
        clean = ts_raw[:-1] if ts_raw.endswith("Z") else ts_raw
        dt = datetime.fromisoformat(clean)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _resolve_snapshot_path(snapshot_path: Optional[Any]) -> Path:
    if snapshot_path is not None:
        return Path(snapshot_path)
    env = os.getenv(_SNAPSHOT_ENV)
    if env:
        return Path(env)
    return Path.cwd() / "runtime" / "positions_snapshot.json"


def _parse_snapshot(
    path: Path, now: datetime
) -> Tuple[bool, str, Dict[str, float], Dict[str, float], Optional[str], Optional[float]]:
    """Returns (fresh, reason, signed_by_symbol, avgcost_by_symbol, ts_utc, age_s)."""
    try:
        if not path.is_file():
            return False, "snapshot_missing", {}, {}, None, None
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, f"snapshot_read_error:{type(exc).__name__}", {}, {}, None, None
    if not isinstance(doc, dict):
        return False, "snapshot_malformed", {}, {}, None, None

    ts_raw = doc.get("ts_utc")
    try:
        ttl = int(doc.get("ttl_seconds", _DEFAULT_TTL_SECONDS))
    except (TypeError, ValueError):
        ttl = _DEFAULT_TTL_SECONDS
    feed_ts = _parse_ts(ts_raw)
    if feed_ts is None:
        return False, "snapshot_bad_ts", {}, {}, (ts_raw if isinstance(ts_raw, str) else None), None
    age_s = (now - feed_ts).total_seconds()
    if age_s > ttl:
        return False, "snapshot_stale", {}, {}, ts_raw, age_s

    signed = _pg._agg_snapshot_positions(doc)  # {SYM: signed qty}, filtered by _V4_TOL
    avgcost: Dict[str, float] = {}
    for row in doc.get("positions") or []:
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol") or "").strip().upper()
        if not sym:
            continue
        try:
            ac = float(row.get("avgCost"))
        except (TypeError, ValueError):
            continue
        if ac > 0.0:
            avgcost[sym] = ac
    return True, "fresh", signed, avgcost, ts_raw, age_s


def _read_guard(path: Path) -> Mapping[str, Any]:
    try:
        if not path.is_file():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, ValueError):
        return {}


def _first_mirror_conflict(
    snapshot_signed: Mapping[str, float],
    mirror_signed: Mapping[str, float],
    tol: float,
) -> Optional[Tuple[str, float, float]]:
    """The two broker views (independent snapshot vs guard ``broker_sync`` mirror)
    must agree on symbols they share; a material disagreement means we cannot
    trust the book and must idle (D3). Returns the first conflicting symbol."""
    for sym in sorted(set(snapshot_signed) & set(mirror_signed)):
        sv = snapshot_signed[sym]
        mv = mirror_signed[sym]
        if abs(sv - mv) > tol:
            return (sym, sv, mv)
    return None


def _excluded_symbols() -> set:
    try:
        from chad.core.position_reconciler import _EFFECTIVE_NON_CHAD_SYMBOLS
        return set(_EFFECTIVE_NON_CHAD_SYMBOLS)
    except Exception:  # pragma: no cover - defensive
        return set()


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def load_context_positions(
    *,
    snapshot_path: Optional[Any] = None,
    guard_path: Optional[Any] = None,
    now: Optional[datetime] = None,
    mirror_tol: float = _MIRROR_TOL,
) -> PositionsView:
    """Load the CHAD-attributed, broker-confirmed position view (see module docstring).

    Never raises. Fail-closed to ``UNKNOWN`` on any unreadable/stale/conflicting
    broker truth. All paths are injectable for hermetic tests.
    """
    now = now or datetime.now(timezone.utc)
    snap_path = _resolve_snapshot_path(snapshot_path)
    g_path = Path(guard_path) if guard_path is not None else _pg.STATE_PATH

    # 1) PRIMARY broker truth — the independent snapshot is REQUIRED for KNOWN.
    fresh, reason, broker_signed, avgcost, ts_utc, age_s = _parse_snapshot(snap_path, now)
    if not fresh:
        return PositionsView(
            STATUS_UNKNOWN, {}, reason,
            {"snapshot_ts_utc": ts_utc, "snapshot_age_s": age_s},
        )

    # 2) Guard — strategy attribution + broker mirror (cross-check only).
    guard = _read_guard(g_path)
    strat_net = _pg._agg_guard_strategy(guard)      # NEVER summed with the mirror
    mirror = _pg._agg_guard_broker_mirror(guard)

    # 3) Cross-check: independent snapshot must agree with the guard's own mirror
    #    on shared symbols; a conflict means we do not guess (D3).
    conflict = _first_mirror_conflict(broker_signed, mirror, mirror_tol)
    if conflict is not None:
        sym, sv, mv = conflict
        return PositionsView(
            STATUS_UNKNOWN, {}, "snapshot_mirror_conflict",
            {"symbol": sym, "snapshot_qty": sv, "mirror_qty": mv, "snapshot_ts_utc": ts_utc},
        )

    # 4) Net to ONE CHAD-attributed Position per symbol (D1/D2 — never sum legs).
    excluded = _excluded_symbols()
    positions: Dict[str, Position] = {}
    injected: list = []
    for sym, s_qty in strat_net.items():
        b_qty = broker_signed.get(sym, 0.0)
        qty = _clamp_to_broker(s_qty, b_qty)
        if abs(qty) <= _pg._V4_TOL:
            continue
        positions[sym] = Position(
            symbol=sym,
            asset_class=_asset_class_enum(sym),
            quantity=qty,
            avg_price=float(avgcost.get(sym, 0.0)),
        )
        injected.append({
            "symbol": sym,
            "qty": qty,
            "strategy_net": s_qty,
            "broker": b_qty,
            "operator_mixed": sym in excluded,
        })

    return PositionsView(
        STATUS_KNOWN, positions, "fresh",
        {
            "snapshot_ts_utc": ts_utc,
            "snapshot_age_s": age_s,
            "n_injected": len(positions),
            "n_strategy_symbols": len(strat_net),
            "injected": injected,
        },
    )
