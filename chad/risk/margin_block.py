"""chad/risk/margin_block.py — the pre-trade ALLOW/BLOCK margin gate.

Phase B (Part 2) of the Margin / Buying-Power BLOCK
(docs/CHAD_MARGIN_BLOCK_DESIGN_v2.2.md — §1.2/§1.3, §2.1, §2.2 decision flow,
Part 3 thresholds, Part 4 fail-closed-vs-bounded-reduce, Part 6 BLOCK contract).

This is the heart: a per-order pre-trade gate that projects post-fill
margin/exposure — counting in-flight orders in BOTH directions via the
PendingExposureLedger — and returns ALLOW or BLOCK(reason) BEFORE the order
reaches the broker. It is PURE decision logic:

  * no I/O, no network, no broker call, no live/execution/broker import;
  * it places NO orders (no order-submission call, no submit path) — it DECIDES;
  * deterministic — every timestamp is injected (``now``); no wall-clock read;
  * reuses the committed ledger + provider snapshots; reimplements neither.

The 17.9× lesson (§0) and its exit mirror (§1.3 / CHANGELOG 2.1→2.2) are closed
structurally: EVERY ALLOW reserves in the ledger (increasing AND reducing), so
each subsequent order is judged net of what is already working, and a
reduce-burst can never over-flatten past flat into a new opposite position.

Decision flow — EXACT §2.2 branch order (this ordering is load-bearing):
  (a) orderId already live in the ledger → BLOCK MODIFY_NOT_ALLOWED. FIRST, so a
      modify (a re-submitted order carrying a live orderId) can never masquerade as a
      reduce. The ledger rebuilds from broker open-orders on restart, so it is
      the authoritative in-flight truth.
  (b) REDUCE-ONLY within the reducible remainder
      (remainder = |last_known_position| − pending_reducing_qty(symbol); order
      opposes the position sign AND qty ≤ remainder) → ALLOW + reserve(reducing=
      True). Allowed even on stale/missing equity (an exit is never trapped);
      MUST reserve so pending_reducing_qty accumulates and the remainder shrinks
      across a burst.
  (c) else (increasing, OR reduce beyond the remainder) → full fail-closed
      checks in order: snapshot fresh & complete → margin =
      max(whatIf, independent-on-same-basis) computable → aggregate gross ≤
      max(N×NetLiq, current) → ExcessLiquidity ≥ floor → InitMargin ≤ ceiling →
      single-order notional ≤ cap → ALLOW + reserve(reducing=False). Any
      uncertainty BLOCKs.

Crypto orders route to their OWN lane (§1.3): a notional-vs-balance check against
the Kraken buying-power snapshot (no leverage, no cross-margin with IBKR).

Two rules that govern everything (§0): when in doubt BLOCK (except to reduce, up
to what is actually reducible); and always trust the scarier number — margin is
computed two independent ways and the LARGER (more conservative) governs.

Mode (Part 7 shadow→enforce ladder), from the frozen config:
  * A normal ALLOW/BLOCK verdict is the TRUE decision in every mode — this pure
    gate returns it and reserves on ALLOW. Suppressing a BLOCK's outcome in
    SHADOW (submitting anyway while still exercising reserve/release) is the
    Phase-C wiring layer's job, not this function's.
  * The ONE mode-dependent behavior here is an unexpected INTERNAL exception:
    ENFORCE → BLOCK INTERNAL_ERROR (fail-closed); SHADOW → log SHADOW_ERROR,
    ALLOW, and STILL reserve (so a leaked-reservation bug surfaces in shadow
    logs, never as a false block later — §7 step 2).

Config (Part 3): thresholds are read ONLY from the frozen, schema-validated
``config/margin_block.json`` via ``load_frozen_config`` — STRICT validation
(unknown non-underscore key / missing required key / wrong type → refuse to
start), NO env-var overrides.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from chad.constants.fx import USDCAD_CONVERSION_CONSTANT
from chad.risk.pending_exposure_ledger import PendingExposureLedger

logger = logging.getLogger(__name__)

# Float comparison tolerance. Threshold comparisons are INCLUSIVE per §2.2
# (≤ cap, ≥ floor): an order exactly AT a limit passes. A tiny relative epsilon
# keeps floating-point round-off at an exact boundary from spuriously blocking a
# legitimate order (or spuriously allowing one) — it never widens a limit by a
# material amount.
_EPS_REL: float = 1e-9


def _le(value: float, limit: float) -> bool:
    """value ≤ limit, tolerant of float round-off at the boundary."""
    return value <= limit + _EPS_REL * max(1.0, abs(limit))


def _ge(value: float, floor: float) -> bool:
    """value ≥ floor, tolerant of float round-off at the boundary."""
    return value >= floor - _EPS_REL * max(1.0, abs(floor))


# ---------------------------------------------------------------------------
# Verdict + reason enum
# ---------------------------------------------------------------------------
class MarginBlockReason(str, Enum):
    """Every ALLOW/BLOCK outcome the gate can return.

    ``str``-valued so a verdict serializes trivially and reads cleanly in logs.
    The two ``ALLOWED_*`` members are the only allowing reasons; every other is
    a BLOCK (except ``SHADOW_ERROR``, an allow-with-error specific to SHADOW).
    """

    ALLOWED = "ALLOWED"                              # increasing order passed all checks
    ALLOWED_REDUCE_ONLY = "ALLOWED_REDUCE_ONLY"      # reduce within the reducible remainder
    MODIFY_NOT_ALLOWED = "MODIFY_NOT_ALLOWED"        # branch (a): orderId already live
    STALE_OR_MISSING_MARGIN_DATA = "STALE_OR_MISSING_MARGIN_DATA"
    MARGIN_UNCOMPUTABLE = "MARGIN_UNCOMPUTABLE"
    AGGREGATE_EXPOSURE = "AGGREGATE_EXPOSURE"
    EXCESS_LIQUIDITY_FLOOR = "EXCESS_LIQUIDITY_FLOOR"
    INIT_MARGIN_CEILING = "INIT_MARGIN_CEILING"
    SINGLE_ORDER_CAP = "SINGLE_ORDER_CAP"
    CRYPTO_INSUFFICIENT_BALANCE = "CRYPTO_INSUFFICIENT_BALANCE"
    INTERNAL_ERROR = "INTERNAL_ERROR"                # ENFORCE fail-closed on internal exception
    SHADOW_ERROR = "SHADOW_ERROR"                    # SHADOW allow-with-error on internal exception


@dataclass(frozen=True)
class AllowOrBlock:
    """The gate's verdict for a single order. Immutable.

    ``allowed`` is the true decision; ``reason`` the enum; ``reserved`` whether
    THIS call registered a ledger reservation (an ALLOW reserves; a SHADOW
    internal-error allow reserves; a BLOCK does not). ``mode`` records the config
    mode the verdict was produced under (observability).
    """

    allowed: bool
    reason: MarginBlockReason
    detail: str
    reserved: bool
    mode: str

    @property
    def blocked(self) -> bool:
        return not self.allowed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason.value,
            "detail": self.detail,
            "reserved": self.reserved,
            "mode": self.mode,
        }


# ---------------------------------------------------------------------------
# Frozen config: strict load + typed accessor
# ---------------------------------------------------------------------------
_VALID_MODES: Tuple[str, ...] = ("shadow", "enforce_paper", "enforce_live")
_ENFORCE_MODES: Tuple[str, ...] = ("enforce_paper", "enforce_live")

# Asset-class → the margin-rate key in config.margin_rates_by_asset_class.
_ASSET_RATE_KEY: Dict[str, str] = {
    "equity": "equity_regt",
    "crypto": "crypto_unlevered",
    "futures": "futures_blanket_conservative",
}

# The strict schema for config/margin_block.json. Leaves are a type token
# ("str"/"bool"/"numeric"); a nested dict is a sub-object validated the same
# way. Keys beginning "_" in the FILE are documentation and always allowed; the
# schema lists only the required non-underscore keys. A key present in the file
# but absent from the schema (a typo'd threshold) → refuse to start, so a typo
# can never validate as a harmless extra while the real threshold silently falls
# to a code default (the loss-guard drift lesson, config safety._note).
_CONFIG_SCHEMA: Dict[str, Any] = {
    "schema_version": "str",
    "design_ref": "str",
    "frozen_utc": "str",
    "mode": "str",
    "account_size_basis": "str",
    "thresholds": {
        "aggregate_gross_exposure_max_mult_netliq": "numeric",
        "excess_liquidity_floor_pct_netliq": "numeric",
        "init_margin_ceiling_pct_netliq": "numeric",
        "single_order_notional_max_pct_netliq": "numeric",
        "whatif_vs_independent_log_gap_tolerance_pct": "numeric",
    },
    "margin_rates_by_asset_class": {
        "equity_regt": "numeric",
        "crypto_unlevered": "numeric",
        "futures_blanket_conservative": "numeric",
    },
    "freshness": {
        "account_data_ttl_seconds": "numeric",
        "fx_conservative_bias_mult": "numeric",
    },
    "whatif_pacing": {
        "max_calls_per_second": "numeric",
    },
    "safety": {
        "no_env_overrides": "bool",
        "refuse_start_on_invalid": "bool",
    },
}


class MarginBlockConfigError(ValueError):
    """Raised when the frozen config fails STRICT validation — refuse to start.

    A ValueError subclass so callers can catch either the specific type or the
    broad ``ValueError`` fail-closed family.
    """


def _check_leaf_type(value: Any, kind: str, path: str) -> None:
    """Validate one leaf value against a schema type token. Raises on mismatch."""
    if kind == "str":
        if not isinstance(value, str):
            raise MarginBlockConfigError(f"{path}: expected str, got {type(value).__name__}")
    elif kind == "bool":
        # bool is an int subclass — check bool FIRST/exactly so a 0/1 cannot pass
        # for a bool nor a True/False pass for a number.
        if not isinstance(value, bool):
            raise MarginBlockConfigError(f"{path}: expected bool, got {type(value).__name__}")
    elif kind == "numeric":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MarginBlockConfigError(f"{path}: expected number, got {type(value).__name__}")
        if not math.isfinite(float(value)):
            raise MarginBlockConfigError(f"{path}: number must be finite, got {value!r}")
    else:  # pragma: no cover - guards against a schema authoring typo
        raise MarginBlockConfigError(f"{path}: unknown schema kind {kind!r}")


def _validate_config_obj(obj: Any, schema: Mapping[str, Any], path: str) -> None:
    """Recursively STRICT-validate ``obj`` against ``schema``.

    Unknown non-underscore key → raise. Missing required key → raise. Wrong type
    → raise. Underscore keys in ``obj`` are documentation and skipped.
    """
    if not isinstance(obj, Mapping):
        raise MarginBlockConfigError(
            f"{path or '<root>'}: expected object, got {type(obj).__name__}"
        )
    for key in obj:
        if isinstance(key, str) and key.startswith("_"):
            continue
        if key not in schema:
            raise MarginBlockConfigError(f"unknown key: {path + '.' if path else ''}{key}")
    for key, spec in schema.items():
        sub_path = f"{path}.{key}" if path else key
        if key not in obj:
            raise MarginBlockConfigError(f"missing required key: {sub_path}")
        value = obj[key]
        if isinstance(spec, Mapping):
            _validate_config_obj(value, spec, sub_path)
        else:
            _check_leaf_type(value, spec, sub_path)


@dataclass(frozen=True)
class MarginBlockConfig:
    """Typed, validated view of the frozen margin-block thresholds.

    Constructed only via ``load_frozen_config`` (production) or directly (test
    fixtures). All numeric thresholds are denominated in NetLiquidation per the
    config's ``account_size_basis`` ("equity" in the design maps to NetLiq here).
    """

    mode: str
    account_size_basis: str
    schema_version: str
    aggregate_gross_exposure_max_mult_netliq: float
    excess_liquidity_floor_pct_netliq: float
    init_margin_ceiling_pct_netliq: float
    single_order_notional_max_pct_netliq: float
    whatif_vs_independent_log_gap_tolerance_pct: float
    margin_rates_by_asset_class: Mapping[str, float]
    account_data_ttl_seconds: float
    fx_conservative_bias_mult: float

    @property
    def is_enforce(self) -> bool:
        return self.mode in _ENFORCE_MODES

    def margin_rate(self, asset_class: str) -> Optional[float]:
        """Independent-estimate margin rate for an asset class, or None if the
        class is unknown / the rate is absent (→ margin may be uncomputable)."""
        key = _ASSET_RATE_KEY.get(asset_class)
        if key is None:
            return None
        rate = self.margin_rates_by_asset_class.get(key)
        return None if rate is None else float(rate)

    @classmethod
    def from_validated_raw(cls, raw: Mapping[str, Any]) -> "MarginBlockConfig":
        """Build from an already schema-validated raw mapping."""
        th = raw["thresholds"]
        fr = raw["freshness"]
        mr = raw["margin_rates_by_asset_class"]
        rates = {k: float(v) for k, v in mr.items() if not (isinstance(k, str) and k.startswith("_"))}
        return cls(
            mode=str(raw["mode"]),
            account_size_basis=str(raw["account_size_basis"]),
            schema_version=str(raw["schema_version"]),
            aggregate_gross_exposure_max_mult_netliq=float(th["aggregate_gross_exposure_max_mult_netliq"]),
            excess_liquidity_floor_pct_netliq=float(th["excess_liquidity_floor_pct_netliq"]),
            init_margin_ceiling_pct_netliq=float(th["init_margin_ceiling_pct_netliq"]),
            single_order_notional_max_pct_netliq=float(th["single_order_notional_max_pct_netliq"]),
            whatif_vs_independent_log_gap_tolerance_pct=float(th["whatif_vs_independent_log_gap_tolerance_pct"]),
            margin_rates_by_asset_class=rates,
            account_data_ttl_seconds=float(fr["account_data_ttl_seconds"]),
            fx_conservative_bias_mult=float(fr["fx_conservative_bias_mult"]),
        )


def load_frozen_config(path: Any) -> MarginBlockConfig:
    """Load + STRICT-validate the frozen margin-block config. Refuse-to-start.

    Fail-closed at startup (Part 3, config ``safety``): a missing/unreadable
    file, malformed JSON, a non-object root, an unknown non-underscore key, a
    missing required key, a wrong-type value, an unknown ``mode``, or the safety
    flags not both True → raise ``MarginBlockConfigError``. NO env-var overrides
    are read (a risk threshold must not be silently overridable — the loss-guard
    drift lesson).
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise MarginBlockConfigError(f"cannot read config {path!r}: {exc}") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MarginBlockConfigError(f"invalid JSON in config {path!r}: {exc}") from exc

    _validate_config_obj(raw, _CONFIG_SCHEMA, "")

    mode = raw["mode"]
    if mode not in _VALID_MODES:
        raise MarginBlockConfigError(f"mode: expected one of {_VALID_MODES}, got {mode!r}")
    safety = raw["safety"]
    if safety["no_env_overrides"] is not True:
        raise MarginBlockConfigError("safety.no_env_overrides must be true")
    if safety["refuse_start_on_invalid"] is not True:
        raise MarginBlockConfigError("safety.refuse_start_on_invalid must be true")

    return MarginBlockConfig.from_validated_raw(raw)


# ---------------------------------------------------------------------------
# Field access + parsing helpers (dual Mapping/attribute access, fail-closed)
# ---------------------------------------------------------------------------
def _get(obj: Any, key: str) -> Any:
    """Read ``key`` from a Mapping or attribute object (mirrors the providers'
    dual-access convention). Missing → None."""
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _finite_float(value: Any) -> Optional[float]:
    """Parse to a finite float (any sign). None / unparseable / non-finite → None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _nonneg_amount(value: Any) -> Optional[float]:
    """Parse to a finite, non-negative float. Otherwise None."""
    f = _finite_float(value)
    if f is None or f < 0.0:
        return None
    return f


def _asset_class(order: Any) -> str:
    return str(_get(order, "asset_class") or "").strip().lower()


def _is_crypto(order: Any) -> bool:
    return _asset_class(order) == "crypto"


def _order_side(order: Any) -> str:
    return str(_get(order, "side") or "").strip().upper()


def _order_qty(order: Any) -> Optional[float]:
    """Positive order quantity magnitude (shares/contracts), or None if absent/
    non-positive. Side carries direction; qty is a magnitude (ib convention)."""
    f = _finite_float(_get(order, "qty"))
    if f is None:
        return None
    f = abs(f)
    return f if f > 0.0 else None


def _order_currency(order: Any) -> str:
    """Notional currency of the order. Defaults to USD (most CHAD instruments are
    USD-quoted); the account base is CAD, so conversion is applied downstream."""
    ccy = str(_get(order, "currency") or "").strip().upper()
    return ccy if ccy else "USD"


def _order_notional_native(order: Any) -> Optional[float]:
    """Gross notional in the order's own currency: explicit ``notional`` if
    given, else ``price`` × ``qty`` × ``multiplier`` (multiplier default 1).
    None if neither is computable."""
    explicit = _nonneg_amount(_get(order, "notional"))
    if explicit is not None:
        return explicit
    price = _nonneg_amount(_get(order, "price"))
    qty = _order_qty(order)
    if price is None or qty is None:
        return None
    mult = _get(order, "multiplier")
    mult_f = 1.0 if mult is None else _finite_float(mult)
    if mult_f is None or mult_f <= 0.0:
        return None
    return price * qty * mult_f


def _to_cad(amount: float, currency: str, config: MarginBlockConfig) -> Optional[float]:
    """Convert an EXPOSURE amount to the CAD account base, biased to OVERSTATE
    USD exposure (FX fails-closed, §2.1). CAD passes through; USD is scaled by
    the sanctioned constant × the pre-registered conservative bias; any other
    currency is unknown → None (fail-closed). The bias applies to exposure, not
    to capacity figures (which the snapshot already reports natively in CAD)."""
    c = currency.strip().upper()
    if c == "CAD":
        return amount
    if c == "USD":
        return amount * USDCAD_CONVERSION_CONSTANT * config.fx_conservative_bias_mult
    return None


def _order_notional_cad(order: Any, config: MarginBlockConfig) -> Optional[float]:
    native = _order_notional_native(order)
    if native is None:
        return None
    return _to_cad(native, _order_currency(order), config)


def _whatif_margin_cad(order: Any) -> Optional[float]:
    """The broker whatIf init-margin change for this order (already in the CAD
    account base per §2.1), clamped to ≥ 0 (a margin requirement is never
    negative). None if absent/unusable → the independent estimate governs."""
    f = _finite_float(_get(order, "whatif_init_margin"))
    return None if f is None else max(0.0, f)


def _independent_margin_cad(order: Any, config: MarginBlockConfig) -> Optional[float]:
    """Own margin estimate = CAD notional × asset-class rate (§2.1). None if the
    notional or the rate is uncomputable."""
    notional_cad = _order_notional_cad(order, config)
    if notional_cad is None:
        return None
    rate = config.margin_rate(_asset_class(order))
    if rate is None:
        return None
    return notional_cad * rate


def _conservative_margin(order: Any, config: MarginBlockConfig) -> Optional[float]:
    """margin = max(whatIf, independent-on-same-basis) — the LARGER (scarier)
    governs (§0 rule 2). Both are CAD. None only if NEITHER is computable."""
    ind = _independent_margin_cad(order, config)
    whatif = _whatif_margin_cad(order)
    candidates = [c for c in (ind, whatif) if c is not None]
    if not candidates:
        return None
    margin = max(candidates)
    if ind is not None and whatif is not None:
        larger = max(ind, whatif)
        if larger > 0.0:
            gap_pct = abs(ind - whatif) / larger * 100.0
            if gap_pct > config.whatif_vs_independent_log_gap_tolerance_pct:
                logger.info(
                    "MARGIN_BLOCK whatif/independent gap %.1f%% (whatif=%.2f independent=%.2f)"
                    " — conservative %.2f used",
                    gap_pct, whatif, ind, margin,
                )
    return margin


# ---------------------------------------------------------------------------
# Positions access
# ---------------------------------------------------------------------------
def _positions_get(positions: Any, symbol: str) -> Any:
    """Look up one position record by symbol (Mapping or attribute object)."""
    if positions is None:
        return None
    getter = getattr(positions, "get", None)
    if callable(getter):
        return getter(symbol)
    if isinstance(positions, Mapping):
        return positions[symbol] if symbol in positions else None
    return getattr(positions, symbol, None)


def _last_known_position(positions: Any, symbol: str) -> float:
    """Signed last-known filled position for ``symbol`` (0.0 if flat/absent).

    A record may be a bare signed number (qty) or a Mapping/object exposing
    ``qty``. This is the reduce-classification anchor and is a pure function of
    last-known sign — robust when equity/margin figures are unavailable."""
    rec = _positions_get(positions, symbol)
    if rec is None:
        return 0.0
    if isinstance(rec, (int, float)) and not isinstance(rec, bool):
        val = _finite_float(rec)
        return val if val is not None else 0.0
    val = _finite_float(_get(rec, "qty"))
    return val if val is not None else 0.0


def _position_notional_cad(rec: Any) -> Optional[float]:
    """Gross CAD notional of one held position, or None if uncomputable.

    A bare-qty record (a number, no market value) has UNKNOWN notional → None,
    which fails the aggregate check closed for an increasing order (we cannot
    prove we are under the cap). A record with an explicit ``notional_cad``
    (finite ≥ 0, 0 is valid for a flat line) is used directly."""
    if isinstance(rec, (int, float)) and not isinstance(rec, bool):
        return None
    return _nonneg_amount(_get(rec, "notional_cad"))


def _filled_gross_notional_cad(positions: Any) -> Optional[float]:
    """Sum of held-position gross CAD notionals (the aggregate check's
    ``current`` term). Returns None when positions cannot be enumerated or a held
    line lacks a usable notional → the aggregate check then fails CLOSED for an
    increasing order (we cannot prove we are under the cap).

    Contract: ``positions is None`` means a FLAT book (no open positions) → 0.0.
    "Unreadable" is a DIFFERENT state — a positions object that raises on lookup
    surfaces as an internal exception (→ INTERNAL_ERROR, fail-closed), and an
    object with no ``items()`` returns None here. The Phase-C caller MUST NOT
    pass ``None`` to mean "could not read positions" (that would fail OPEN); it
    passes ``None``/``{}`` only for a genuinely empty book."""
    if positions is None:
        return 0.0
    items = getattr(positions, "items", None)
    if not callable(items):
        return None
    total = 0.0
    for _sym, rec in items():
        n = _position_notional_cad(rec)
        if n is None:
            return None
        total += n
    return total


def _is_reducing(last_known: float, side: str) -> bool:
    """True iff the order opposes the position sign (moves toward flat). A flat
    line (0) has nothing to reduce → not reducing (any order there increases)."""
    if last_known > 0.0 and side == "SELL":
        return True
    if last_known < 0.0 and side == "BUY":
        return True
    return False


# ---------------------------------------------------------------------------
# verdict constructors
# ---------------------------------------------------------------------------
def _allow(reason: MarginBlockReason, detail: str, config: MarginBlockConfig, *, reserved: bool) -> AllowOrBlock:
    return AllowOrBlock(allowed=True, reason=reason, detail=detail, reserved=reserved, mode=config.mode)


def _block(reason: MarginBlockReason, detail: str, config: MarginBlockConfig) -> AllowOrBlock:
    return AllowOrBlock(allowed=False, reason=reason, detail=detail, reserved=False, mode=config.mode)


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
def decide(
    order: Any,
    account_snapshot: Any,
    positions: Any,
    ledger: PendingExposureLedger,
    config: MarginBlockConfig,
    *,
    now: float,
) -> AllowOrBlock:
    """Return ALLOW or BLOCK(reason) for one order, before it reaches the broker.

    Pure decision logic (no I/O, no broker, no order placement). ``now`` is the
    injected evaluation epoch (determinism — the ledger reserve needs it; no
    wall-clock is read). For an equity/futures order ``account_snapshot`` is a
    BuyingPowerSnapshot; for a crypto order it is a KrakenBuyingPowerSnapshot
    (the caller supplies the right lane's snapshot; the gate routes on the
    order's ``asset_class``). ``positions`` maps symbol → a signed qty or a
    record exposing ``qty`` (and, for held lines an increasing order must clear,
    ``notional_cad``). ``ledger`` is the gate-authoritative PendingExposureLedger
    (rebuilt from broker truth on restart before any verdict is served).

    An unexpected internal exception is isolated per mode: ENFORCE → BLOCK
    INTERNAL_ERROR (fail-closed); SHADOW → log SHADOW_ERROR, ALLOW, and still
    reserve. Every other path is a normal, mode-independent verdict.
    """
    try:
        return _decide_inner(order, account_snapshot, positions, ledger, config, now=now)
    except MarginBlockConfigError:
        # A config problem is not a per-order internal error — surface it.
        raise
    except Exception as exc:  # noqa: BLE001 - deliberate internal-error isolation (§2.2)
        return _handle_internal_error(exc, order, ledger, config, now=now)


def _decide_inner(
    order: Any,
    account_snapshot: Any,
    positions: Any,
    ledger: PendingExposureLedger,
    config: MarginBlockConfig,
    *,
    now: float,
) -> AllowOrBlock:
    order_id = _get(order, "order_id")
    symbol = str(_get(order, "symbol") or "").strip().upper()
    side = _order_side(order)

    # (a) MODIFY BAN — FIRST, so a modify can never masquerade as a reduce.
    # The ledger is the authoritative in-flight truth (rebuilt from broker
    # open-orders on restart); an orderId already live there is a re-submitted order.
    if ledger.is_reserved(order_id):
        return _block(
            MarginBlockReason.MODIFY_NOT_ALLOWED,
            f"orderId {order_id!r} already live in ledger",
            config,
        )

    # (b) REDUCE-ONLY within the reducible remainder — allowed even on stale/
    # missing equity, but MUST reserve so pending_reducing_qty accumulates.
    last_known = _last_known_position(positions, symbol)
    qty = _order_qty(order)
    if qty is not None and _is_reducing(last_known, side):
        remainder = abs(last_known) - ledger.pending_reducing_qty(symbol)
        if remainder > 0.0 and _le(qty, remainder):
            reduce_notional = _order_notional_cad(order, config)
            ledger.reserve(
                order_id, side, symbol, True, 0.0,
                reduce_notional if reduce_notional is not None else 0.0,
                qty=qty, now=now,
            )
            return _allow(
                MarginBlockReason.ALLOWED_REDUCE_ONLY,
                f"reduce {qty} within remainder {remainder} on {symbol}",
                config, reserved=True,
            )

    # (c) increasing, OR reduce beyond the remainder → full fail-closed checks.
    if _is_crypto(order):
        return _decide_crypto(order, account_snapshot, ledger, config,
                              order_id=order_id, symbol=symbol, side=side, qty=qty, now=now)
    return _decide_ibkr(order, account_snapshot, positions, ledger, config,
                        order_id=order_id, symbol=symbol, side=side, qty=qty, now=now)


def _decide_ibkr(
    order: Any,
    account_snapshot: Any,
    positions: Any,
    ledger: PendingExposureLedger,
    config: MarginBlockConfig,
    *,
    order_id: Any,
    symbol: str,
    side: str,
    qty: Optional[float],
    now: float,
) -> AllowOrBlock:
    """Full fail-closed increasing-order checks for the IBKR (margin) lane, in
    the EXACT §2.2 order."""
    # snapshot fresh & complete?
    if account_snapshot is None or not bool(getattr(account_snapshot, "usable", False)):
        reason_detail = getattr(account_snapshot, "reason", "missing") if account_snapshot is not None else "missing"
        return _block(MarginBlockReason.STALE_OR_MISSING_MARGIN_DATA, f"snapshot {reason_detail}", config)

    netliq = _finite_float(getattr(account_snapshot, "net_liquidation", None))
    excess_liq = _finite_float(getattr(account_snapshot, "excess_liquidity", None))
    cur_init = _finite_float(getattr(account_snapshot, "full_init_margin_req", None))
    if netliq is None or netliq <= 0.0 or excess_liq is None or cur_init is None:
        return _block(MarginBlockReason.STALE_OR_MISSING_MARGIN_DATA, "snapshot fields incomplete", config)

    # margin = max(whatIf, independent-on-same-basis) computable?
    margin = _conservative_margin(order, config)
    if margin is None:
        return _block(MarginBlockReason.MARGIN_UNCOMPUTABLE, "neither whatIf nor independent margin computable", config)

    this_notional = _order_notional_cad(order, config)
    if this_notional is None:
        return _block(MarginBlockReason.MARGIN_UNCOMPUTABLE, "order notional uncomputable", config)

    totals = ledger.total_reserved()
    reserved_notional = totals.notional
    reserved_margin = totals.margin

    # aggregate: (filled + reserved + this) gross ≤ max(N×NetLiq, current)
    filled_gross = _filled_gross_notional_cad(positions)
    if filled_gross is None:
        return _block(MarginBlockReason.AGGREGATE_EXPOSURE, "current gross exposure unreadable", config)
    projected_gross = filled_gross + reserved_notional + this_notional
    agg_cap = max(config.aggregate_gross_exposure_max_mult_netliq * netliq, filled_gross)
    if not _le(projected_gross, agg_cap):
        return _block(
            MarginBlockReason.AGGREGATE_EXPOSURE,
            f"gross {projected_gross:.2f} > cap {agg_cap:.2f}",
            config,
        )

    # projected ExcessLiquidity ≥ floor (in-flight margin counted)
    projected_excess = excess_liq - reserved_margin - margin
    excess_floor = config.excess_liquidity_floor_pct_netliq / 100.0 * netliq
    if not _ge(projected_excess, excess_floor):
        return _block(
            MarginBlockReason.EXCESS_LIQUIDITY_FLOOR,
            f"excess {projected_excess:.2f} < floor {excess_floor:.2f}",
            config,
        )

    # projected InitMargin ≤ ceiling (in-flight margin counted)
    projected_init = cur_init + reserved_margin + margin
    init_ceiling = config.init_margin_ceiling_pct_netliq / 100.0 * netliq
    if not _le(projected_init, init_ceiling):
        return _block(
            MarginBlockReason.INIT_MARGIN_CEILING,
            f"init {projected_init:.2f} > ceiling {init_ceiling:.2f}",
            config,
        )

    # single-order notional ≤ cap
    single_cap = config.single_order_notional_max_pct_netliq / 100.0 * netliq
    if not _le(this_notional, single_cap):
        return _block(
            MarginBlockReason.SINGLE_ORDER_CAP,
            f"notional {this_notional:.2f} > cap {single_cap:.2f}",
            config,
        )

    # all checks pass → ALLOW + reserve (increasing)
    ledger.reserve(order_id, side, symbol, False, margin, this_notional, qty=qty if qty is not None else 0.0, now=now)
    return _allow(MarginBlockReason.ALLOWED, f"increasing {symbol} within all caps", config, reserved=True)


def _decide_crypto(
    order: Any,
    account_snapshot: Any,
    ledger: PendingExposureLedger,
    config: MarginBlockConfig,
    *,
    order_id: Any,
    symbol: str,
    side: str,
    qty: Optional[float],
    now: float,
) -> AllowOrBlock:
    """Crypto own-lane check (§1.3): notional-vs-balance, unlevered.

    ``account_snapshot`` is a KrakenBuyingPowerSnapshot. The CAPACITY SOURCE is
    the Kraken balance (``available_cad``) — never an IBKR margin figure, so
    there is no cross-margin: crypto gets no IBKR leverage/netting benefit.

    Like the IBKR lane (and per §1.1/§1.3 — "every order, every lane … counting
    in-flight" / "the aggregate check counts … all live reservations …"), the
    candidate is judged NET OF WHAT IS ALREADY IN FLIGHT, so a same-cycle burst
    of crypto BUYs cannot each pass against the same static balance and
    over-commit (the 17.9× mechanism, on this lane). The in-flight term is the
    ledger's GLOBAL ``total_reserved().notional`` because the frozen ledger
    exposes no lane-scoped total; that is deliberately CONSERVATIVE (a busy IBKR
    book can only over-block a crypto INCREASE, never under-block one — and a
    crypto REDUCE never reaches here, it is allowed in branch (b)). A future
    lane-tagged ledger would let this bound be strictly crypto-scoped."""
    if account_snapshot is None or not bool(getattr(account_snapshot, "usable", False)):
        reason_detail = getattr(account_snapshot, "reason", "missing") if account_snapshot is not None else "missing"
        return _block(MarginBlockReason.STALE_OR_MISSING_MARGIN_DATA, f"kraken snapshot {reason_detail}", config)

    this_notional = _order_notional_cad(order, config)
    if this_notional is None:
        return _block(MarginBlockReason.MARGIN_UNCOMPUTABLE, "crypto order notional uncomputable", config)

    available = _nonneg_amount(getattr(account_snapshot, "available_cad", None))
    if available is None:
        return _block(MarginBlockReason.STALE_OR_MISSING_MARGIN_DATA, "kraken available_cad unusable", config)

    # Count in-flight reservations, so a burst is bounded by the balance.
    reserved_notional = ledger.total_reserved().notional
    projected_notional = reserved_notional + this_notional
    if not _le(projected_notional, available):
        return _block(
            MarginBlockReason.CRYPTO_INSUFFICIENT_BALANCE,
            f"projected {projected_notional:.2f} (reserved {reserved_notional:.2f} + this {this_notional:.2f})"
            f" > available {available:.2f}",
            config,
        )

    # Unlevered lane: margin == notional. Reserve (increasing) in the shared
    # ledger; the crypto verdict itself never reads the IBKR aggregate.
    ledger.reserve(order_id, side, symbol, False, this_notional, this_notional, qty=qty if qty is not None else 0.0, now=now)
    return _allow(MarginBlockReason.ALLOWED, f"crypto {symbol} within balance", config, reserved=True)


def _handle_internal_error(
    exc: BaseException,
    order: Any,
    ledger: PendingExposureLedger,
    config: MarginBlockConfig,
    *,
    now: float,
) -> AllowOrBlock:
    """Isolate an unexpected internal exception per mode (§2.2).

    ENFORCE → BLOCK INTERNAL_ERROR (fail-closed). SHADOW → log SHADOW_ERROR,
    ALLOW, and STILL reserve (best-effort) so a leaked-reservation bug surfaces
    in shadow logs rather than as a false block later."""
    if config.is_enforce:
        logger.error("MARGIN_BLOCK internal error (ENFORCE → BLOCK): %r", exc)
        return _block(MarginBlockReason.INTERNAL_ERROR, repr(exc), config)
    logger.error("MARGIN_BLOCK SHADOW_ERROR (allow + best-effort reserve): %r", exc)
    reserved = _best_effort_reserve(order, ledger, config, now=now)
    return _allow(MarginBlockReason.SHADOW_ERROR, repr(exc), config, reserved=reserved)


def _best_effort_reserve(order: Any, ledger: PendingExposureLedger, config: MarginBlockConfig, *, now: float) -> bool:
    """Reserve an order with best-effort fields after a SHADOW internal error.

    In SHADOW the order proceeds despite the error, so the ledger must still
    count it (never silently under-count in-flight exposure). Classification is
    unknown after an error → reserve as increasing (reducing=False, conservative)
    with a best-effort notional (0 if uncomputable). Guarded so a failure here
    never escapes the error handler."""
    try:
        order_id = _get(order, "order_id")
        symbol = str(_get(order, "symbol") or "?").strip().upper() or "?"
        side = _order_side(order) or "?"
        qty = _order_qty(order) or 0.0
        notional_cad = _order_notional_cad(order, config)
        ledger.reserve(
            order_id, side, symbol, False, 0.0,
            notional_cad if notional_cad is not None else 0.0,
            qty=qty, now=now,
        )
        return True
    except Exception as exc:  # noqa: BLE001 - never let the fallback crash the gate
        logger.error("MARGIN_BLOCK shadow best-effort reserve failed: %r", exc)
        return False


__all__ = [
    "MarginBlockReason",
    "AllowOrBlock",
    "MarginBlockConfig",
    "MarginBlockConfigError",
    "load_frozen_config",
    "decide",
]
