"""chad/risk/buying_power_provider.py — IBKR account/margin buying-power provider.

Phase A of the Margin / Buying-Power BLOCK (docs/CHAD_MARGIN_BLOCK_DESIGN_v2.2.md,
§2.1, Part 4, Part 8). READ-ONLY, additive, offline: this module supplies the
account/margin fields the (future) gate needs. It does NOT decide ALLOW/BLOCK
(that is Phase B — margin_block.py) and it places NO orders.

What it does (only):
  * Parse an ``accountSummary()``-shaped input into a typed snapshot of the six
    IBKR fields the gate needs (§2.1):
        NetLiquidation, BuyingPower, ExcessLiquidity, AvailableFunds,
        FullInitMarginReq, FullMaintMarginReq
    mirroring the existing collector's row parsing
    (chad/portfolio/ibkr_portfolio_collector_v2.py:172-186 — rows are
    ``AccountValue(account, tag, value, currency, modelCode)`` read via getattr).
  * Stamp a per-snapshot capture time and check freshness against a configurable
    TTL (default 30s per design §3: "stale → treat as missing").
  * Cache the last usable snapshot and re-check its freshness on read.

Fail-closed doctrine (Part 4): on ANY uncertainty — a missing field, an
unparseable/empty value, an absent/wrong/mixed currency, an ambiguous
per-account duplicate, no capture time, or a stale snapshot — the provider
returns a typed sentinel flagged ``usable=False`` with a specific reason. It
NEVER returns a silent zero. A parsed *0.0* (e.g. FullInitMarginReq on a flat
account) is a real, usable value and is distinguished from missing/empty.

Currency (§2.1): values are kept in their REPORTED currency and tagged. This
provider does NOT convert — the gate converts, conservatively, using
``chad/constants/fx.py::USDCAD_CONVERSION_CONSTANT`` (the single FX source). The
real paper account's base currency is CAD (verified: portfolio_snapshot.json
``ibkr_equity_currency == "CAD"``), so the default expected currency is CAD; a
row reported in another currency fails closed rather than being silently mixed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

# Required IBKR accountSummary tags (design §2.1). All six must be present,
# parseable, and reported in one expected currency for the snapshot to be USABLE.
REQUIRED_TAGS: Tuple[str, ...] = (
    "NetLiquidation",
    "BuyingPower",
    "ExcessLiquidity",
    "AvailableFunds",
    "FullInitMarginReq",
    "FullMaintMarginReq",
)

# Design §3: "Account-data freshness TTL — start 30s (stale → treat as missing)."
DEFAULT_FRESHNESS_TTL_SECONDS: float = 30.0

# Real paper account base currency (verified: portfolio_snapshot.json
# ibkr_equity_currency == "CAD"). Values are tagged, not converted, here.
DEFAULT_EXPECTED_CURRENCY: str = "CAD"


class BPReason:
    """String reason codes for a buying-power snapshot verdict.

    ``OK`` is the only usable reason; every other value marks a fail-closed
    sentinel. Kept as plain strings (not an Enum) so snapshots serialize
    trivially and reasons read cleanly in logs.
    """

    OK = "OK"
    EMPTY_INPUT = "EMPTY_INPUT"
    MISSING_FIELD = "MISSING_FIELD"
    UNPARSEABLE = "UNPARSEABLE"
    ABSENT_CURRENCY = "ABSENT_CURRENCY"
    WRONG_CURRENCY = "WRONG_CURRENCY"
    MIXED_CURRENCY = "MIXED_CURRENCY"
    AMBIGUOUS_FIELD = "AMBIGUOUS_FIELD"
    BAD_TTL = "BAD_TTL"
    NO_CAPTURE_TIME = "NO_CAPTURE_TIME"
    STALE = "STALE"
    NO_DATA = "NO_DATA"


@dataclass(frozen=True)
class BuyingPowerSnapshot:
    """A typed, immutable snapshot of the IBKR account/margin fields.

    When ``usable`` is False this is a fail-closed sentinel: every numeric field
    is ``None`` (never a silent zero) and ``reason``/``detail`` explain why. When
    ``usable`` is True every numeric field is a parsed float in ``currency``.
    """

    usable: bool
    reason: str
    detail: str
    currency: Optional[str]
    captured_at_epoch: Optional[float]
    age_seconds: Optional[float]
    ttl_seconds: float
    net_liquidation: Optional[float]
    buying_power: Optional[float]
    excess_liquidity: Optional[float]
    available_funds: Optional[float]
    full_init_margin_req: Optional[float]
    full_maint_margin_req: Optional[float]

    @property
    def is_fail_closed(self) -> bool:
        return not self.usable

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable view (observability only — no decision)."""
        return {
            "usable": self.usable,
            "reason": self.reason,
            "detail": self.detail,
            "currency": self.currency,
            "captured_at_epoch": self.captured_at_epoch,
            "age_seconds": self.age_seconds,
            "ttl_seconds": self.ttl_seconds,
            "net_liquidation": self.net_liquidation,
            "buying_power": self.buying_power,
            "excess_liquidity": self.excess_liquidity,
            "available_funds": self.available_funds,
            "full_init_margin_req": self.full_init_margin_req,
            "full_maint_margin_req": self.full_maint_margin_req,
        }

    @classmethod
    def fail_closed(
        cls,
        reason: str,
        detail: str = "",
        *,
        ttl_seconds: float,
        captured_at_epoch: Optional[float] = None,
        age_seconds: Optional[float] = None,
        currency: Optional[str] = None,
    ) -> "BuyingPowerSnapshot":
        """Construct a not-usable sentinel with all numeric fields cleared."""
        return cls(
            usable=False,
            reason=reason,
            detail=detail,
            currency=currency,
            captured_at_epoch=captured_at_epoch,
            age_seconds=age_seconds,
            ttl_seconds=float(ttl_seconds),
            net_liquidation=None,
            buying_power=None,
            excess_liquidity=None,
            available_funds=None,
            full_init_margin_req=None,
            full_maint_margin_req=None,
        )


def _row_get(row: Any, key: str) -> Any:
    """Read ``key`` from an accountSummary row.

    Mirrors the collector's getattr access (rows are ib_async
    ``AccountValue`` NamedTuples) while also accepting plain dict rows so the
    same parser can be exercised against serialized fixtures / real snapshots.
    """
    if isinstance(row, Mapping):
        return row.get(key)
    return getattr(row, key, None)


def _parse_float(raw: Any) -> Tuple[bool, float]:
    """Parse a numeric AccountValue string. Returns (ok, value).

    Deliberately does NOT mimic the collector's ``float(... or 0.0)``: an empty
    or missing value must fail closed, never collapse to a silent zero. A real
    ``"0"`` / ``"0.00"`` parses fine (a flat account legitimately has 0 margin).
    """
    if raw is None:
        return (False, 0.0)
    text = str(raw).strip()
    if text == "":
        return (False, 0.0)
    try:
        val = float(text)
    except (TypeError, ValueError):
        return (False, 0.0)
    # nan / +inf / -inf are not usable money figures. float() accepts the
    # strings "nan"/"inf"/"Infinity", so reject non-finite here — fail closed,
    # never a silent non-finite value threaded into a margin decision.
    if not math.isfinite(val):
        return (False, 0.0)
    return (True, val)


def parse_account_summary(
    rows: Iterable[Any],
    *,
    captured_at_epoch: Optional[float],
    now_epoch: float,
    ttl_seconds: float = DEFAULT_FRESHNESS_TTL_SECONDS,
    expected_currency: Optional[str] = DEFAULT_EXPECTED_CURRENCY,
    account: Optional[str] = None,
) -> BuyingPowerSnapshot:
    """Parse accountSummary()-shaped ``rows`` into a fail-closed-typed snapshot.

    Pure function (no I/O, no broker call). ``captured_at_epoch`` is the wall
    clock (epoch seconds) at which the caller obtained ``rows``; ``now_epoch`` is
    the evaluation time. Freshness = ``now_epoch - captured_at_epoch <=
    ttl_seconds`` (and non-negative — a future capture time is clock skew and
    fails closed).

    ``account`` optionally restricts parsing to a single IBKR account; without
    it, a tag appearing for more than one account with differing values is
    AMBIGUOUS and fails closed (margin must never be blindly summed across
    accounts — unlike the collector's NetLiquidation sum).
    """
    ttl = float(ttl_seconds)
    if ttl <= 0.0:
        return BuyingPowerSnapshot.fail_closed(
            BPReason.BAD_TTL, f"ttl_seconds={ttl_seconds!r}", ttl_seconds=ttl
        )

    # ---- collect required-tag rows (optionally filtered by account) ----------
    values_seen: Dict[str, List[Tuple[str, str]]] = {}  # tag -> [(value_str, ccy)]
    saw_any_row = False
    for row in rows:
        saw_any_row = True
        tag = str(_row_get(row, "tag") or "").strip()
        if tag not in REQUIRED_TAGS:
            continue
        acct = str(_row_get(row, "account") or "").strip()
        if account is not None and acct != account:
            continue
        raw_value = _row_get(row, "value")
        ccy = str(_row_get(row, "currency") or "").strip().upper()
        values_seen.setdefault(tag, []).append(
            ("" if raw_value is None else str(raw_value), ccy)
        )

    if not saw_any_row:
        return BuyingPowerSnapshot.fail_closed(BPReason.EMPTY_INPUT, ttl_seconds=ttl)

    # ---- resolve each required field: present, unambiguous, parseable --------
    parsed: Dict[str, float] = {}
    currencies: Dict[str, str] = {}
    for tag in REQUIRED_TAGS:
        entries = values_seen.get(tag)
        if not entries:
            return BuyingPowerSnapshot.fail_closed(
                BPReason.MISSING_FIELD, tag, ttl_seconds=ttl
            )
        # Ambiguity: multiple rows for this tag that do not agree on BOTH value
        # and currency. Key on the (value, currency) pair — not the value alone
        # — so two rows of the same magnitude in different currencies are
        # caught, never silently collapsed to the first row's currency (a
        # currency-provenance hole in exactly the class CHAD regresses on).
        distinct_pairs = {(v, c) for (v, c) in entries}
        if len(distinct_pairs) > 1:
            return BuyingPowerSnapshot.fail_closed(
                BPReason.AMBIGUOUS_FIELD,
                f"{tag}: {sorted(distinct_pairs)}",
                ttl_seconds=ttl,
            )
        value_str, ccy = entries[0]
        ok, val = _parse_float(value_str)
        if not ok:
            return BuyingPowerSnapshot.fail_closed(
                BPReason.UNPARSEABLE, f"{tag}={value_str!r}", ttl_seconds=ttl
            )
        if ccy == "":
            return BuyingPowerSnapshot.fail_closed(
                BPReason.ABSENT_CURRENCY, tag, ttl_seconds=ttl
            )
        parsed[tag] = val
        currencies[tag] = ccy

    # ---- currency must be single and (if expected given) the expected one ----
    distinct_ccy = set(currencies.values())
    if len(distinct_ccy) > 1:
        return BuyingPowerSnapshot.fail_closed(
            BPReason.MIXED_CURRENCY,
            ",".join(f"{t}={currencies[t]}" for t in REQUIRED_TAGS),
            ttl_seconds=ttl,
        )
    unified_ccy = next(iter(distinct_ccy))
    if expected_currency is not None and unified_ccy != expected_currency.strip().upper():
        return BuyingPowerSnapshot.fail_closed(
            BPReason.WRONG_CURRENCY,
            f"expected={expected_currency.strip().upper()} got={unified_ccy}",
            ttl_seconds=ttl,
            currency=unified_ccy,
        )

    # ---- freshness (checked last, so structural/currency reasons win) --------
    if captured_at_epoch is None:
        return BuyingPowerSnapshot.fail_closed(
            BPReason.NO_CAPTURE_TIME, ttl_seconds=ttl, currency=unified_ccy
        )
    age = float(now_epoch) - float(captured_at_epoch)
    if age < 0.0 or age > ttl:
        return BuyingPowerSnapshot.fail_closed(
            BPReason.STALE,
            f"age={age:.3f}s ttl={ttl:.3f}s",
            ttl_seconds=ttl,
            captured_at_epoch=float(captured_at_epoch),
            age_seconds=age,
            currency=unified_ccy,
        )

    return BuyingPowerSnapshot(
        usable=True,
        reason=BPReason.OK,
        detail="",
        currency=unified_ccy,
        captured_at_epoch=float(captured_at_epoch),
        age_seconds=age,
        ttl_seconds=ttl,
        net_liquidation=parsed["NetLiquidation"],
        buying_power=parsed["BuyingPower"],
        excess_liquidity=parsed["ExcessLiquidity"],
        available_funds=parsed["AvailableFunds"],
        full_init_margin_req=parsed["FullInitMarginReq"],
        full_maint_margin_req=parsed["FullMaintMarginReq"],
    )


class BuyingPowerProvider:
    """Caches the last usable IBKR buying-power snapshot; re-checks freshness.

    Report-only. No allow/block decision, no order method, no broker import at
    module scope. ``fetch`` is a thin optional convenience wrapper around a
    passed-in ``ib``-like object; it is deliberately NOT exercised by the unit
    tests (no network in tests).
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_FRESHNESS_TTL_SECONDS,
        expected_currency: Optional[str] = DEFAULT_EXPECTED_CURRENCY,
        account: Optional[str] = None,
    ) -> None:
        self._ttl = float(ttl_seconds)
        self._expected_currency = expected_currency
        self._account = account
        self._cache: Optional[BuyingPowerSnapshot] = None

    @property
    def ttl_seconds(self) -> float:
        return self._ttl

    def update(
        self,
        rows: Iterable[Any],
        *,
        captured_at_epoch: Optional[float],
        now_epoch: float,
    ) -> BuyingPowerSnapshot:
        """Parse ``rows`` and cache the result if (and only if) it is usable."""
        snap = parse_account_summary(
            rows,
            captured_at_epoch=captured_at_epoch,
            now_epoch=now_epoch,
            ttl_seconds=self._ttl,
            expected_currency=self._expected_currency,
            account=self._account,
        )
        if snap.usable:
            self._cache = snap
        return snap

    def get(self, now_epoch: float) -> BuyingPowerSnapshot:
        """Return the cached snapshot re-evaluated for freshness at ``now_epoch``.

        Fail-closed if nothing is cached or the cached snapshot has aged past its
        TTL — a cached-but-now-stale snapshot must not be served as usable.
        """
        snap = self._cache
        if snap is None or snap.captured_at_epoch is None:
            return BuyingPowerSnapshot.fail_closed(BPReason.NO_DATA, ttl_seconds=self._ttl)
        age = float(now_epoch) - float(snap.captured_at_epoch)
        if age < 0.0 or age > snap.ttl_seconds:
            return BuyingPowerSnapshot.fail_closed(
                BPReason.STALE,
                f"age={age:.3f}s ttl={snap.ttl_seconds:.3f}s",
                ttl_seconds=snap.ttl_seconds,
                captured_at_epoch=snap.captured_at_epoch,
                age_seconds=age,
                currency=snap.currency,
            )
        return replace(snap, age_seconds=age)

    def fetch(
        self,
        ib: Any,
        *,
        now_epoch: float,
        captured_at_epoch: Optional[float] = None,
    ) -> BuyingPowerSnapshot:
        """Thin convenience wrapper: ``ib.accountSummary()`` → parse → cache.

        NOT unit-tested (network/broker). ``ib`` is any object exposing
        ``accountSummary()`` (e.g. an ib_async ``IB``). This method reads only;
        it never places or modifies an order.
        """
        rows = ib.accountSummary()
        cap = now_epoch if captured_at_epoch is None else captured_at_epoch
        return self.update(rows, captured_at_epoch=cap, now_epoch=now_epoch)


__all__ = [
    "REQUIRED_TAGS",
    "DEFAULT_FRESHNESS_TTL_SECONDS",
    "DEFAULT_EXPECTED_CURRENCY",
    "BPReason",
    "BuyingPowerSnapshot",
    "parse_account_summary",
    "BuyingPowerProvider",
]
