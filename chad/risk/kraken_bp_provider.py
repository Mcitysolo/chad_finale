"""chad/risk/kraken_bp_provider.py — Kraken (crypto lane) buying-power provider.

Phase A of the Margin / Buying-Power BLOCK (docs/CHAD_MARGIN_BLOCK_DESIGN_v2.2.md,
§1.3, §2.1, Part 4, Part 8). READ-ONLY, additive, offline.

The crypto lane is its OWN lane (§1.3/§2.1): no leverage, no cross-margin with
IBKR. This provider reports available crypto buying capacity in the account's
native base currency (CAD) from a ``runtime/kraken_balances.json``-shaped input
(verified real schema: ``{ts_utc, ok, balances{...}, raw, usd_equivalent,
cad_equivalent, error}``). It does NOT decide ALLOW/BLOCK (that is Phase B) and
places NO orders.

Valuation (mirrors chad/market_data/kraken_balance_provider._value_balances_in_cad
and tests/test_kraken_cad_equity):
  * CAD cash → valued 1:1 (native, no FX).
  * A crypto balance → valued only if a USD price is supplied, as
        qty * price_usd * USDCAD_CONVERSION_CONSTANT
    using the single sanctioned FX source (chad/constants/fx.py). An UNPRICED
    crypto balance is dropped from capacity (conservative — never counted as
    buying power it cannot be proven to have).

Fail-closed doctrine (Part 4): unreadable (``ok`` not True / ``error`` set /
non-mapping), missing/unparseable balances, a negative balance (unlevered lane
→ impossible → treat as corruption), no capture time, or a stale snapshot →
typed sentinel flagged ``usable=False``. Never a silent zero.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from chad.constants.fx import USDCAD_CONVERSION_CONSTANT

# Design §3 shared account-data freshness TTL default (start 30s). NOTE: the
# Kraken snapshot producer refreshes on a ~300s cadence, so an operator may
# configure a longer TTL for this lane; the conservative default fails closed
# rather than serve data of unknown age.
DEFAULT_FRESHNESS_TTL_SECONDS: float = 30.0

# The Kraken account is natively CAD (kraken_balances.json balances.CAD + a
# small crypto sliver). Capacity is reported in this base and tagged.
ACCOUNT_CURRENCY: str = "CAD"

# Cash key valued 1:1 into the CAD base (no FX applied).
_CAD_CASH_KEY: str = "CAD"


class KrakenBPReason:
    """String reason codes for a Kraken buying-power snapshot verdict."""

    OK = "OK"
    NOT_A_MAPPING = "NOT_A_MAPPING"
    UNREADABLE = "UNREADABLE"
    MISSING_BALANCES = "MISSING_BALANCES"
    UNPARSEABLE = "UNPARSEABLE"
    NEGATIVE_BALANCE = "NEGATIVE_BALANCE"
    NO_CAPTURE_TIME = "NO_CAPTURE_TIME"
    STALE = "STALE"
    BAD_TTL = "BAD_TTL"
    NO_DATA = "NO_DATA"


@dataclass(frozen=True)
class KrakenBuyingPowerSnapshot:
    """Typed, immutable Kraken crypto-lane capacity snapshot.

    When ``usable`` is False this is a fail-closed sentinel with numeric fields
    ``None`` (never a silent zero). When True, ``available_cad`` is the native
    CAD buying capacity (cash 1:1 + priced crypto), ``currency`` == "CAD".
    """

    usable: bool
    reason: str
    detail: str
    currency: Optional[str]
    captured_at_epoch: Optional[float]
    age_seconds: Optional[float]
    ttl_seconds: float
    available_cad: Optional[float]
    cad_cash: Optional[float]
    crypto_cad: Optional[float]
    unpriced_symbols: Tuple[str, ...]

    @property
    def is_fail_closed(self) -> bool:
        return not self.usable

    def to_dict(self) -> Dict[str, Any]:
        return {
            "usable": self.usable,
            "reason": self.reason,
            "detail": self.detail,
            "currency": self.currency,
            "captured_at_epoch": self.captured_at_epoch,
            "age_seconds": self.age_seconds,
            "ttl_seconds": self.ttl_seconds,
            "available_cad": self.available_cad,
            "cad_cash": self.cad_cash,
            "crypto_cad": self.crypto_cad,
            "unpriced_symbols": list(self.unpriced_symbols),
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
    ) -> "KrakenBuyingPowerSnapshot":
        return cls(
            usable=False,
            reason=reason,
            detail=detail,
            currency=None,
            captured_at_epoch=captured_at_epoch,
            age_seconds=age_seconds,
            ttl_seconds=float(ttl_seconds),
            available_cad=None,
            cad_cash=None,
            crypto_cad=None,
            unpriced_symbols=tuple(),
        )


def _iso_to_epoch(text: Any) -> Optional[float]:
    """Parse an ISO-8601 UTC timestamp (trailing 'Z' tolerated) to epoch secs.

    Returns None on anything unparseable (fail-closed at the call site).
    """
    if not isinstance(text, str) or text.strip() == "":
        return None
    normalized = text.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(normalized)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _price_for(symbol: str, prices_usd: Optional[Mapping[str, float]]) -> Optional[float]:
    """Look up a USD price for ``symbol`` under a couple of common key shapes."""
    if not prices_usd:
        return None
    for key in (symbol, f"{symbol}-USD", f"{symbol}USD"):
        if key in prices_usd:
            raw = prices_usd[key]
            try:
                price = float(raw)
            except (TypeError, ValueError):
                return None
            # Require a finite, strictly-positive price. A +inf price would pass
            # a bare `> 0.0` and poison crypto_cad; nan already fails `> 0.0`.
            if math.isfinite(price) and price > 0.0:
                return price
            return None
    return None


def parse_kraken_balances(
    payload: Any,
    *,
    now_epoch: float,
    ttl_seconds: float = DEFAULT_FRESHNESS_TTL_SECONDS,
    prices_usd: Optional[Mapping[str, float]] = None,
) -> KrakenBuyingPowerSnapshot:
    """Parse a kraken_balances.json-shaped ``payload`` into a typed snapshot.

    Pure function (no I/O). ``now_epoch`` is the evaluation time; capture time is
    derived from the payload's ``ts_utc``. Freshness = ``now_epoch -
    captured_at_epoch <= ttl_seconds`` (and non-negative).
    """
    ttl = float(ttl_seconds)
    if ttl <= 0.0:
        # Dedicated code (harmonized with the IBKR provider's BPReason.BAD_TTL)
        # so a Phase-B caller switching on `reason` sees the same code for the
        # same programmer/config error across both lanes.
        return KrakenBuyingPowerSnapshot.fail_closed(
            KrakenBPReason.BAD_TTL, f"bad ttl_seconds={ttl_seconds!r}", ttl_seconds=ttl
        )

    if not isinstance(payload, Mapping):
        return KrakenBuyingPowerSnapshot.fail_closed(
            KrakenBPReason.NOT_A_MAPPING, type(payload).__name__, ttl_seconds=ttl
        )

    # Producer-declared readability: ok must be exactly True and error absent.
    if payload.get("ok") is not True:
        return KrakenBuyingPowerSnapshot.fail_closed(
            KrakenBPReason.UNREADABLE, f"ok={payload.get('ok')!r}", ttl_seconds=ttl
        )
    err = payload.get("error")
    if err is not None:
        return KrakenBuyingPowerSnapshot.fail_closed(
            KrakenBPReason.UNREADABLE, f"error={err!r}", ttl_seconds=ttl
        )

    captured = _iso_to_epoch(payload.get("ts_utc"))
    if captured is None:
        return KrakenBuyingPowerSnapshot.fail_closed(
            KrakenBPReason.NO_CAPTURE_TIME,
            f"ts_utc={payload.get('ts_utc')!r}",
            ttl_seconds=ttl,
        )

    balances = payload.get("balances")
    if not isinstance(balances, Mapping) or len(balances) == 0:
        return KrakenBuyingPowerSnapshot.fail_closed(
            KrakenBPReason.MISSING_BALANCES, ttl_seconds=ttl, captured_at_epoch=captured
        )

    cad_cash = 0.0
    crypto_cad = 0.0
    unpriced: list[str] = []
    for raw_sym, raw_qty in balances.items():
        symbol = str(raw_sym).strip().upper()
        try:
            qty = float(raw_qty)
        except (TypeError, ValueError):
            return KrakenBuyingPowerSnapshot.fail_closed(
                KrakenBPReason.UNPARSEABLE,
                f"{symbol}={raw_qty!r}",
                ttl_seconds=ttl,
                captured_at_epoch=captured,
            )
        # json.loads accepts bare NaN/Infinity by default, so a corrupted
        # balances file can round-trip a non-finite qty. A non-finite balance is
        # not a usable buying-power figure — fail closed, never sum nan/inf into
        # available_cad and hand a usable=True snapshot to the gate.
        if not math.isfinite(qty):
            return KrakenBuyingPowerSnapshot.fail_closed(
                KrakenBPReason.UNPARSEABLE,
                f"{symbol}={raw_qty!r} (non-finite)",
                ttl_seconds=ttl,
                captured_at_epoch=captured,
            )
        if qty < 0.0:
            return KrakenBuyingPowerSnapshot.fail_closed(
                KrakenBPReason.NEGATIVE_BALANCE,
                f"{symbol}={qty}",
                ttl_seconds=ttl,
                captured_at_epoch=captured,
            )
        if symbol == _CAD_CASH_KEY:
            cad_cash += qty  # native CAD cash, valued 1:1 (no FX)
            continue
        price = _price_for(symbol, prices_usd)
        if price is None:
            # Conservative: unpriced crypto is NOT counted as buying power.
            if qty > 0.0:
                unpriced.append(symbol)
            continue
        crypto_cad += qty * price * USDCAD_CONVERSION_CONSTANT

    age = float(now_epoch) - captured
    if age < 0.0 or age > ttl:
        return KrakenBuyingPowerSnapshot.fail_closed(
            KrakenBPReason.STALE,
            f"age={age:.3f}s ttl={ttl:.3f}s",
            ttl_seconds=ttl,
            captured_at_epoch=captured,
            age_seconds=age,
        )

    available = cad_cash + crypto_cad
    return KrakenBuyingPowerSnapshot(
        usable=True,
        reason=KrakenBPReason.OK,
        detail="",
        currency=ACCOUNT_CURRENCY,
        captured_at_epoch=captured,
        age_seconds=age,
        ttl_seconds=ttl,
        available_cad=available,
        cad_cash=cad_cash,
        crypto_cad=crypto_cad,
        unpriced_symbols=tuple(unpriced),
    )


class KrakenBuyingPowerProvider:
    """Caches the last usable Kraken capacity snapshot; re-checks freshness.

    Report-only. No allow/block decision, no order method. ``load_from_file``
    reads the runtime snapshot read-only (the only I/O this class does).
    """

    def __init__(self, *, ttl_seconds: float = DEFAULT_FRESHNESS_TTL_SECONDS) -> None:
        self._ttl = float(ttl_seconds)
        self._cache: Optional[KrakenBuyingPowerSnapshot] = None

    @property
    def ttl_seconds(self) -> float:
        return self._ttl

    def update(
        self,
        payload: Any,
        *,
        now_epoch: float,
        prices_usd: Optional[Mapping[str, float]] = None,
    ) -> KrakenBuyingPowerSnapshot:
        snap = parse_kraken_balances(
            payload,
            now_epoch=now_epoch,
            ttl_seconds=self._ttl,
            prices_usd=prices_usd,
        )
        if snap.usable:
            self._cache = snap
        return snap

    def load_from_file(
        self,
        path: Path,
        *,
        now_epoch: float,
        prices_usd: Optional[Mapping[str, float]] = None,
    ) -> KrakenBuyingPowerSnapshot:
        """Read-only load + parse of a kraken_balances.json file.

        A missing/unreadable file or malformed JSON fails closed (UNREADABLE);
        it never raises into the caller and never writes.
        """
        try:
            text = Path(path).read_text(encoding="utf-8")
        except (OSError, ValueError) as exc:
            return KrakenBuyingPowerSnapshot.fail_closed(
                KrakenBPReason.UNREADABLE, f"read: {exc}", ttl_seconds=self._ttl
            )
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            return KrakenBuyingPowerSnapshot.fail_closed(
                KrakenBPReason.UNREADABLE, f"json: {exc}", ttl_seconds=self._ttl
            )
        return self.update(payload, now_epoch=now_epoch, prices_usd=prices_usd)

    def get(self, now_epoch: float) -> KrakenBuyingPowerSnapshot:
        snap = self._cache
        if snap is None or snap.captured_at_epoch is None:
            return KrakenBuyingPowerSnapshot.fail_closed(
                KrakenBPReason.NO_DATA, ttl_seconds=self._ttl
            )
        age = float(now_epoch) - float(snap.captured_at_epoch)
        if age < 0.0 or age > snap.ttl_seconds:
            return KrakenBuyingPowerSnapshot.fail_closed(
                KrakenBPReason.STALE,
                f"age={age:.3f}s ttl={snap.ttl_seconds:.3f}s",
                ttl_seconds=snap.ttl_seconds,
                captured_at_epoch=snap.captured_at_epoch,
                age_seconds=age,
            )
        return replace(snap, age_seconds=age)


__all__ = [
    "DEFAULT_FRESHNESS_TTL_SECONDS",
    "ACCOUNT_CURRENCY",
    "KrakenBPReason",
    "KrakenBuyingPowerSnapshot",
    "parse_kraken_balances",
    "KrakenBuyingPowerProvider",
]
