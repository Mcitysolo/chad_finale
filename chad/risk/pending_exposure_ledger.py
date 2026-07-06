"""chad/risk/pending_exposure_ledger.py — in-flight order reservation ledger.

Phase B (Part 1) of the Margin / Buying-Power BLOCK
(docs/CHAD_MARGIN_BLOCK_DESIGN_v2.2.md, §1.3, Part 4, Part 7). PURE in-memory
state — no I/O, no broker call, no live-order-path import. It does NOT decide
ALLOW/BLOCK (that is the gate, margin_block.py, Phase B Part 2) and it places NO
orders.

Why it exists (the 17.9× lesson, §0). The runaway book was built by a *burst* of
orders each passing an individual check against a stale snapshot, with nothing
counting the orders already in flight. This ledger is that missing counter: the
gate reserves EVERY order it ALLOWs — increasing and reducing alike — so each
subsequent order is judged net of what is already working.

Gate-authoritative, and the reducing bound is load-bearing (§1.3, CHANGELOG
2.1→2.2). Reducing orders MUST reserve (``reducing=True``); otherwise
``pending_reducing_qty`` never accumulates, the gate's reducible remainder
(``last_known_position − pending_reducing_qty``) stays at the full position, and
a reduce-burst (three SELL-100s vs long 100) over-flattens into a short — the
exact bug the remainder was meant to close.

Public API (all pure state, deterministic):
  * ``reserve(order_id, side, symbol, reducing, margin, notional, *, qty, now)``
        — register an ALLOWed order. Duplicate ``order_id`` → ValueError
          (reject, loud; the gate's modify-ban BLOCKs a live orderId before it
          ever reaches reserve, so a duplicate here is a real invariant breach).
  * ``release(order_id, *, filled_qty=None)`` — drop a reservation on
        fill/cancel/reject. Unknown WELL-FORMED id → no-op (returns False). A
        malformed id (None/empty) or invalid ``filled_qty`` still raises.
  * ``expire(now, broker_open_order_ids)`` — TTL expiry RECONCILED against an
        INJECTED broker open-orders truth set. An entry past TTL is released
        ONLY when broker truth confirms it is no longer live; it NEVER
        blind-releases. ``broker_open_order_ids=None`` means truth is
        UNAVAILABLE → release nothing (a lost lifecycle event must not silently
        free reserved exposure).
  * ``rebuild_from_broker(open_orders, *, now)`` — clear and rebuild the whole
        ledger from the broker's actual open orders. Callable at startup so the
        gate can require a rebuild before serving any verdict (``rebuilt``).
  * ``total_reserved()`` — aggregate reserved margin AND notional (typed pair,
        never conflated into one number) across all live reservations.
  * ``pending_reducing_qty(symbol)`` — summed qty of live REDUCING reservations
        on that symbol; this is exactly what the gate's remainder subtracts.

Partial fills (deliberate, stated — §1.3, CHANGELOG 2.1→2.2). Target semantics:
a partial fill shrinks the reservation by the filled qty. Until that lands, the
FULL reservation persists on partial fill (conservative — may over-block further
reduces, never under-blocks). ``release`` already carries the ``filled_qty``
keyword so per-fill shrink can be implemented later WITHOUT breaking callers;
today a ``filled_qty``-bearing call is a deliberate no-op that RETAINS the full
reservation, and only a terminal ``release(order_id)`` removes it.

Determinism (§0, Part 7). Wall-clock time is never read here — every timestamp
is injected (``now``), so the same sequence of operations always yields the same
state. Money/qty inputs are validated finite and non-negative at the door; a
non-finite or negative amount is corruption and raises, never silently poisons
an aggregate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

# The reservation TTL is the age past which an entry becomes eligible for
# reconciliation against broker truth (expire). It is deliberately NOT
# safety-critical: expire NEVER blind-releases, so a shorter/longer TTL only
# changes WHEN reconciliation is attempted, never WHETHER exposure is wrongly
# freed. A working resting order can legitimately live minutes, so the default
# is generous; operators may configure per lane.
DEFAULT_RESERVATION_TTL_SECONDS: float = 300.0

# Fields every broker-open-order spec must supply for rebuild_from_broker. The
# caller (Phase B/C, from broker truth) is responsible for reconstructing the
# CHAD-side classification (reducing) and the margin/notional/qty estimates; the
# ledger stays agnostic and only validates + stores them.
_REBUILD_FIELDS: Tuple[str, ...] = (
    "order_id",
    "side",
    "symbol",
    "reducing",
    "margin",
    "notional",
    "qty",
)


@dataclass(frozen=True)
class Reservation:
    """A single in-flight order's reserved exposure. Immutable.

    ``qty`` is the order quantity in position units (shares/contracts) — the
    magnitude the gate's reducible-remainder bound subtracts against; ``margin``
    and ``notional`` are money figures for the aggregate/exposure checks.
    ``reserved_at_epoch`` is the injected reserve time (drives TTL).
    """

    order_id: str
    side: str
    symbol: str
    reducing: bool
    margin: float
    notional: float
    qty: float
    reserved_at_epoch: float

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable view (observability only — no decision)."""
        return {
            "order_id": self.order_id,
            "side": self.side,
            "symbol": self.symbol,
            "reducing": self.reducing,
            "margin": self.margin,
            "notional": self.notional,
            "qty": self.qty,
            "reserved_at_epoch": self.reserved_at_epoch,
        }


@dataclass(frozen=True)
class ReservedTotals:
    """Aggregate reserved exposure across all live reservations.

    ``margin`` and ``notional`` are kept SEPARATE (never summed into one
    number): the gate needs reserved notional for the aggregate gross-exposure
    cap and reserved margin for the init-margin / excess-liquidity projections.
    ``count`` is the number of live reservations (observability).
    """

    margin: float
    notional: float
    count: int

    def to_dict(self) -> Dict[str, Any]:
        return {"margin": self.margin, "notional": self.notional, "count": self.count}


class PendingExposureLedger:
    """In-memory, gate-authoritative ledger of in-flight order reservations.

    Pure state: no I/O, no broker call, no live import. Not thread-safe by
    design (the gate is single-threaded on the submit path); callers serialize
    access.
    """

    def __init__(self, *, ttl_seconds: float = DEFAULT_RESERVATION_TTL_SECONDS) -> None:
        ttl = float(ttl_seconds)
        if not math.isfinite(ttl) or ttl <= 0.0:
            # Programmer/config error — refuse to construct rather than serve a
            # ledger whose TTL can never trigger reconciliation sanely.
            raise ValueError(f"ttl_seconds must be finite and > 0, got {ttl_seconds!r}")
        self._ttl: float = ttl
        # Insertion order preserved (Python dict) → deterministic iteration.
        self._reservations: Dict[str, Reservation] = {}
        # Flips True once rebuilt from broker truth. The gate (Phase B Part 2)
        # should require this before serving a verdict (§1.3, Part 7: "on
        # process restart the ledger rebuilds from broker open-orders before any
        # verdict is served").
        self._rebuilt: bool = False

    # ---- properties -------------------------------------------------------
    @property
    def ttl_seconds(self) -> float:
        return self._ttl

    @property
    def rebuilt(self) -> bool:
        """True once rebuild_from_broker has run at least once this process."""
        return self._rebuilt

    def __len__(self) -> int:
        return len(self._reservations)

    # ---- normalization / validation helpers -------------------------------
    @staticmethod
    def _norm_order_id(order_id: Any) -> str:
        # None must be rejected, not coerced to the string "None" (which would
        # become a live key). An int orderId (ib_async uses int) stringifies fine.
        if order_id is None:
            raise ValueError("order_id must not be None")
        oid = str(order_id).strip()
        if oid == "":
            raise ValueError("order_id must be a non-empty string")
        return oid

    @staticmethod
    def _norm_side(side: Any) -> str:
        s = str(side).strip().upper()
        if s == "":
            raise ValueError("side must be a non-empty string")
        return s

    @staticmethod
    def _norm_symbol(symbol: Any) -> str:
        sym = str(symbol).strip().upper()
        if sym == "":
            raise ValueError("symbol must be a non-empty string")
        return sym

    @staticmethod
    def _as_bool(value: Any, *, name: str) -> bool:
        # Strict: a string "false" or an int 0 must NOT masquerade as a bool and
        # silently mis-classify a reducer as increasing (or vice-versa).
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a bool, got {type(value).__name__}")
        return value

    @staticmethod
    def _as_amount(value: Any, *, name: str) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} not numeric: {value!r}") from exc
        # Non-finite (nan/inf) or negative money/qty is corruption — never let it
        # into an aggregate that a margin decision reads.
        if not math.isfinite(v) or v < 0.0:
            raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
        return v

    @staticmethod
    def _as_epoch(value: Any, *, name: str) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} not numeric: {value!r}") from exc
        if not math.isfinite(v):
            raise ValueError(f"{name} must be a finite epoch, got {value!r}")
        return v

    def _validated_reservation(
        self,
        *,
        order_id: Any,
        side: Any,
        symbol: Any,
        reducing: Any,
        margin: Any,
        notional: Any,
        qty: Any,
        reserved_at: Any,
    ) -> Reservation:
        """Validate every field and construct a ``Reservation`` — no store, no
        duplicate check. The shared validation path for ``reserve`` and
        ``rebuild_from_broker`` so both reject the same corrupt inputs. Raises
        ValueError on any invalid field.
        """
        return Reservation(
            order_id=self._norm_order_id(order_id),
            side=self._norm_side(side),
            symbol=self._norm_symbol(symbol),
            reducing=self._as_bool(reducing, name="reducing"),
            margin=self._as_amount(margin, name="margin"),
            notional=self._as_amount(notional, name="notional"),
            qty=self._as_amount(qty, name="qty"),
            reserved_at_epoch=self._as_epoch(reserved_at, name="reserved_at"),
        )

    def _insert(
        self,
        *,
        order_id: Any,
        side: Any,
        symbol: Any,
        reducing: Any,
        margin: Any,
        notional: Any,
        qty: Any,
        reserved_at: Any,
    ) -> Reservation:
        """Validate and insert one reservation into live state (the ``reserve``
        path). Duplicate order_id → ValueError.
        """
        res = self._validated_reservation(
            order_id=order_id,
            side=side,
            symbol=symbol,
            reducing=reducing,
            margin=margin,
            notional=notional,
            qty=qty,
            reserved_at=reserved_at,
        )
        if res.order_id in self._reservations:
            # Reject duplicates loudly (documented choice). The gate blocks a
            # live orderId (MODIFY_NOT_ALLOWED) before reserve is reached, so a
            # duplicate here means an id collision or double-reserve bug — surface
            # it, never silently keep-first (which could under-count exposure if
            # the two calls describe different orders).
            raise ValueError(f"duplicate order_id reserved: {res.order_id!r}")
        self._reservations[res.order_id] = res
        return res

    # ---- reserve / release ------------------------------------------------
    def reserve(
        self,
        order_id: Any,
        side: Any,
        symbol: Any,
        reducing: bool,
        margin: float,
        notional: float,
        *,
        qty: float,
        now: float,
    ) -> Reservation:
        """Register an ALLOWed order (increasing OR reducing) at ALLOW time.

        Reducing orders MUST reserve (``reducing=True``) or the gate's
        reducible-remainder bound is defeated (§1.3). ``qty`` is the order
        quantity in position units — the design's §2.2 reserve signature
        abbreviated it, but ``pending_reducing_qty`` and future per-fill shrink
        both require it, so it is an explicit keyword here. ``now`` is the
        injected reserve time (drives TTL — no wall-clock is read).

        Raises ValueError on a duplicate ``order_id`` or any invalid field.
        Returns the created immutable ``Reservation``.
        """
        return self._insert(
            order_id=order_id,
            side=side,
            symbol=symbol,
            reducing=reducing,
            margin=margin,
            notional=notional,
            qty=qty,
            reserved_at=now,
        )

    def release(self, order_id: Any, *, filled_qty: Optional[float] = None) -> bool:
        """Drop a reservation on a TERMINAL event (full fill / cancel / reject).

        Releasing an unknown WELL-FORMED id is a no-op — returns False, does not
        raise (a late/duplicate lifecycle event must not crash the gate). A
        malformed id (None/empty) or an invalid ``filled_qty`` (non-finite /
        negative) still raises ValueError — those are corruption, not a
        benign late event.

        ``filled_qty`` reserves the future per-fill-shrink API. TODAY it is a
        deliberate no-op that RETAINS the full reservation (conservative
        partial-fill semantics, §1.3): a partial fill must not free exposure, so
        a ``filled_qty``-bearing call returns False and changes nothing. Only a
        terminal ``release(order_id)`` (no ``filled_qty``) removes the entry.
        When per-fill shrink lands, that same call site starts shrinking without
        any signature change — callers do not break.
        """
        oid = self._norm_order_id(order_id)
        if filled_qty is not None:
            # Partial fill: conservative retention. Validate the input (so a
            # malformed partial surfaces) but do NOT shrink or remove.
            self._as_amount(filled_qty, name="filled_qty")
            return False
        if oid in self._reservations:
            del self._reservations[oid]
            return True
        return False

    # ---- TTL expiry reconciled against broker truth -----------------------
    def expire(self, now: float, broker_open_order_ids: Optional[Any]) -> List[str]:
        """Release past-TTL reservations ONLY where broker truth confirms gone.

        ``broker_open_order_ids`` — the injected broker open-orders truth:
          * ``None`` → truth UNAVAILABLE. Release NOTHING (never blind-release; a
            lost lifecycle event must not silently free reserved exposure).
          * an iterable of order ids → AUTHORITATIVE set of currently-live broker
            orders. A reservation is released iff it is BOTH past TTL AND absent
            from this set. An entry past TTL that is STILL in broker truth is
            KEPT (it is genuinely working — not leaked). An entry not yet past
            TTL is kept regardless (too soon to reconcile; its terminal event
            should arrive, and TTL will catch a truly lost one later).

        Note the empty-set vs None distinction is load-bearing: an EMPTY set is a
        valid truth ("broker has zero open orders") and DOES release past-TTL
        entries; ``None`` is "I could not read broker truth" and releases none.

        Returns the list of released order ids (sorted, for deterministic tests).
        """
        if broker_open_order_ids is None:
            return []
        now_e = self._as_epoch(now, name="now")
        truth = {str(x).strip() for x in broker_open_order_ids}
        to_release: List[str] = []
        for oid, res in self._reservations.items():
            age = now_e - res.reserved_at_epoch
            if age > self._ttl and oid not in truth:
                to_release.append(oid)
        for oid in to_release:
            del self._reservations[oid]
        return sorted(to_release)

    # ---- restart rebuild from broker truth --------------------------------
    def rebuild_from_broker(self, open_orders: Any, *, now: float) -> int:
        """Rebuild the ledger from the broker's actual open orders. TRANSACTIONAL.

        On process restart the in-memory ledger is empty while orders may still
        be working at the broker; this restores gate-authoritative truth before
        any verdict (§1.3, Part 7). On SUCCESS it fully REPLACES existing state
        with the rebuilt set and flips ``rebuilt`` True.

        Each element of ``open_orders`` is a Mapping or an object exposing the
        fields in ``_REBUILD_FIELDS`` (order_id, side, symbol, reducing, margin,
        notional, qty). All rebuilt entries are stamped ``reserved_at = now`` so
        TTL restarts fresh. A missing/invalid field, or a duplicate order_id
        within the input, raises ValueError — a rebuild must be trustworthy, so
        it fails loud rather than serve a partially-reconstructed ledger.

        Atomicity is the safety property: the new set is assembled in a LOCAL
        map and swapped in only after EVERY spec validates. A mid-rebuild error
        therefore leaves the prior ``_reservations`` and the prior ``rebuilt``
        flag completely UNCHANGED — never a partially-reconstructed ledger that
        under-counts in-flight exposure. At startup (prior state empty, ``rebuilt``
        False) a failed rebuild stays not-``rebuilt``, so a gate that requires a
        successful rebuild before serving a verdict stays fail-closed.

        Returns the number of reservations rebuilt.
        """
        now_e = self._as_epoch(now, name="now")
        built: Dict[str, Reservation] = {}
        for spec in open_orders:
            fields = {name: _spec_get(spec, name) for name in _REBUILD_FIELDS}
            res = self._validated_reservation(
                order_id=fields["order_id"],
                side=fields["side"],
                symbol=fields["symbol"],
                reducing=fields["reducing"],
                margin=fields["margin"],
                notional=fields["notional"],
                qty=fields["qty"],
                reserved_at=now_e,
            )
            if res.order_id in built:
                raise ValueError(f"duplicate order_id reserved: {res.order_id!r}")
            built[res.order_id] = res
        # Commit point: reached only if no spec raised. A mid-loop raise above
        # never gets here, so live state is left untouched (last-good preserved).
        self._reservations = built
        self._rebuilt = True
        return len(self._reservations)

    # ---- aggregates read by the gate --------------------------------------
    def total_reserved(self) -> ReservedTotals:
        """Aggregate reserved margin and notional across all live reservations.

        Every stored amount is already validated finite/non-negative, so these
        sums are finite. Empty ledger → zeros.
        """
        margin = 0.0
        notional = 0.0
        for res in self._reservations.values():
            margin += res.margin
            notional += res.notional
        return ReservedTotals(margin=margin, notional=notional, count=len(self._reservations))

    def pending_reducing_qty(self, symbol: Any) -> float:
        """Summed qty of live REDUCING reservations on ``symbol``.

        This is exactly the term the gate subtracts:
        ``remainder = last_known_position − pending_reducing_qty(symbol)``. Only
        reservations with ``reducing=True`` on the (normalized) symbol count;
        increasing reservations are excluded.
        """
        sym = self._norm_symbol(symbol)
        total = 0.0
        for res in self._reservations.values():
            if res.reducing and res.symbol == sym:
                total += res.qty
        return total

    # ---- queries / observability ------------------------------------------
    def is_reserved(self, order_id: Any) -> bool:
        """True if ``order_id`` currently has a live reservation.

        Supports the gate's modify-ban (§2.2 branch (a): an orderId already live
        → BLOCK MODIFY_NOT_ALLOWED). Pure query.
        """
        return self._norm_order_id(order_id) in self._reservations

    def get(self, order_id: Any) -> Optional[Reservation]:
        """Return the live ``Reservation`` for ``order_id``, or None."""
        return self._reservations.get(self._norm_order_id(order_id))

    def live_reservations(self) -> Tuple[Reservation, ...]:
        """Immutable snapshot of all live reservations (insertion order)."""
        return tuple(self._reservations.values())

    def live_order_ids(self) -> Tuple[str, ...]:
        """Immutable snapshot of all live reservation order ids."""
        return tuple(self._reservations.keys())


def _spec_get(spec: Any, key: str) -> Any:
    """Read ``key`` from a broker-open-order spec (Mapping or attribute object).

    Mirrors the Phase-A providers' dual-access pattern so the same rebuild path
    works against dict fixtures and ib_async-style order objects. A missing key
    yields None, which then fails closed in ``_validated_reservation``'s checks.
    """
    if isinstance(spec, Mapping):
        return spec.get(key)
    return getattr(spec, key, None)


__all__ = [
    "DEFAULT_RESERVATION_TTL_SECONDS",
    "Reservation",
    "ReservedTotals",
    "PendingExposureLedger",
]
