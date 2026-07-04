"""chad/validation/cost_model.py — Phase 2 trading-cost model + pessimistic execution.

The single, offline place where the edge-validation harness charges the friction
of trading. SSOT ``docs/CHAD_EDGE_VALIDATION_HARNESS_DESIGN_v1.1.md`` §3.5 (S4 +
V2): *every* trade — synthetic Stage-1 backtest fills **and** real Stage-2 paper
fills — is charged commission + half-spread (entry AND exit) + volatility/volume
-linked slippage, with an optional (default-off) market-impact term. Under-charging
is the classic backtest lie; this module over-charges on purpose (conservative
defaults) so a surviving edge is genuinely a surviving edge.

One path, both stages: :func:`apply_costs` is the ONLY function that turns a
:class:`Trade` into a :class:`CostBreakdown`. Stage 1 (backtest engine, Phase 4)
and Stage 2 (trade-log adapter, Phase 6) both construct a :class:`Trade` and call
this same function — there is no second haircut anywhere in the harness. A real
IBKR paper fill is charged identically to a synthetic one (SSOT §3.5: IBKR paper
fills are optimistic — mid-ish, no queue/partials — so they get the same haircut).

Pessimistic intrabar rule (V2): :func:`resolve_intrabar` implements the "when a
daily bar's range contains BOTH stop and target, assume the stop hit first"
discipline. Order-within-the-bar is unknowable from a daily OHLC bar, so the worse
(stop) outcome is assumed and the fill is flagged ambiguous for counting.

Isolation (SSOT §1.2 / §2): pure, offline, deterministic, standard-library only.
No numpy, no broker, no ``runtime/`` reader, no live-loop dependency.

--------------------------------------------------------------------------------
Cost model (documented once, applied uniformly across instrument classes)
--------------------------------------------------------------------------------
A :class:`Trade` models one **round trip** (an entry fill and an exit fill). Costs
are charged on BOTH legs. Per leg, at that leg's price ``P``:

  notional      = P * quantity * multiplier
  commission    = per instrument class (see below), charged on each leg
  spread cost   = half_spread_frac * notional        (half_spread_frac = tier_bps/1e4)
  slippage cost = slip_frac        * notional
  impact cost   = impact_frac      * notional         (0.0 unless enabled)

with, expressed as plain fractions of notional (1 bp = 1e-4):

  slip_frac   = slippage_base_bps/1e4
                + slippage_vol_coeff           * volatility      (a fraction, e.g. 0.02)
                + slippage_participation_coeff * participation    (order_qty / bar_volume)
  impact_frac = impact_coeff * sqrt(participation)                (square-root impact law)

Commission by instrument class (each leg):
  * STK    — ``max(per_share * qty, min_order)``          (IBKR-style per-share + floor)
  * FUT    — ``per_contract * qty``                        (flat per contract)
  * CRYPTO — ``max(rate * notional, min_order)``           (percentage of notional)

``total_cost = commission + spread_cost + slippage_cost + impact_cost`` is always
``>= 0`` and, when a pre-cost ``gross_pnl`` is supplied, ``net_pnl = gross_pnl -
total_cost``. Every parameter that produced the number is echoed into the output
(``config_echo``) so a report is self-describing and reproducible.

Sentinel / raise convention (mirrors ``scoring_spine`` and the SSOT honesty rule):
  * Degenerate-but-valid data never raises — it degrades to a documented,
    conservative default and records a human-readable ``warning``:
      - missing ``volatility``      → volatility slippage term = 0 (+ warning),
      - missing volume/participation → volume slippage & impact term = 0 (+ warning),
      - zero volatility / zero participation → genuinely-computed 0 term (no warning).
  * Structurally-invalid *input* is a caller bug and fails fast with ``ValueError``:
      - non-positive price, quantity, or multiplier on a :class:`Trade`,
      - negative volatility / participation / bar_volume,
      - any negative rate/bp/coefficient in :class:`CostConfig` (invalid config),
      - a bar with ``high < low`` in :func:`resolve_intrabar`.
    These are treated like invalid configuration (fail-fast), never as data to
    silently paper over.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Union

__all__ = [
    "InstrumentClass",
    "LiquidityTier",
    "CostConfig",
    "DEFAULT_COST_CONFIG",
    "Trade",
    "CostBreakdown",
    "apply_costs",
    "IntrabarResolution",
    "resolve_intrabar",
]

Number = Union[int, float]
_BPS: float = 1e-4  # one basis point as a fraction


# --------------------------------------------------------------------------- #
# Categorical inputs.
# --------------------------------------------------------------------------- #
class InstrumentClass(Enum):
    """Instrument class selecting the commission schedule (SSOT §3.5)."""

    STK = "STK"
    FUT = "FUT"
    CRYPTO = "CRYPTO"


class LiquidityTier(Enum):
    """Liquidity tier selecting the half-spread charged on each leg (SSOT §3.5).

    Ordered most→least liquid; a wider tier costs more half-spread per side.
    """

    LIQUID = "LIQUID"        # e.g. SPY, mega-cap, front-month index futures
    MID = "MID"              # e.g. mid-cap equities, second-tier futures
    ILLIQUID = "ILLIQUID"    # e.g. small-cap / thin names


# --------------------------------------------------------------------------- #
# Frozen cost configuration — conservative defaults, echoed into every output.
# --------------------------------------------------------------------------- #
def _reject_negative(name: str, value: float) -> None:
    """Fail-fast on a negative config parameter (invalid configuration)."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value}")


@dataclass(frozen=True)
class CostConfig:
    """All cost parameters in one frozen, JSON-serialisable, hashable record.

    Defaults are deliberately conservative (over-charge, never under-charge). A
    later phase (SSOT §3.2 ``config_freeze``) may hash-freeze a chosen instance;
    every field is a plain number/bool so :meth:`to_dict` is ``json.dumps``-able.

    Half-spreads and the base slippage are expressed in **basis points** (of the
    per-leg notional); the volatility and participation slippage coefficients and
    the impact coefficient are **dimensionless multipliers on a fraction** (a
    volatility fraction such as 0.02, and a participation fraction ``qty/volume``).
    """

    # --- commissions -------------------------------------------------------- #
    stk_commission_per_share: float = 0.005     # $/share (IBKR tiered-ish)
    stk_commission_min: float = 1.00            # $/order floor
    fut_commission_per_contract: float = 0.85   # $/contract (conservative full+micro)
    crypto_commission_rate: float = 0.0018      # fraction of notional (0.18%)
    crypto_commission_min: float = 0.0          # $/order floor

    # --- half-spread per liquidity tier, basis points, charged on EACH leg -- #
    half_spread_bps_liquid: float = 2.0
    half_spread_bps_mid: float = 5.0
    half_spread_bps_illiquid: float = 15.0

    # --- slippage (each leg) ------------------------------------------------ #
    slippage_base_bps: float = 1.0              # always-on floor, bps of notional
    slippage_vol_coeff: float = 0.10            # * volatility fraction
    slippage_participation_coeff: float = 0.10  # * participation fraction (qty/volume)

    # --- optional market impact (OFF by default, SSOT §3.5) ----------------- #
    enable_market_impact: bool = False
    impact_coeff: float = 0.10                  # impact_frac = coeff * sqrt(participation)

    def __post_init__(self) -> None:
        for name in (
            "stk_commission_per_share",
            "stk_commission_min",
            "fut_commission_per_contract",
            "crypto_commission_rate",
            "crypto_commission_min",
            "half_spread_bps_liquid",
            "half_spread_bps_mid",
            "half_spread_bps_illiquid",
            "slippage_base_bps",
            "slippage_vol_coeff",
            "slippage_participation_coeff",
            "impact_coeff",
        ):
            _reject_negative(name, getattr(self, name))
        if not isinstance(self.enable_market_impact, bool):
            raise ValueError(
                f"enable_market_impact must be bool, got {self.enable_market_impact!r}"
            )

    def half_spread_bps_for(self, tier: LiquidityTier) -> float:
        """Half-spread (bps of per-leg notional) for a liquidity ``tier``."""
        if tier is LiquidityTier.LIQUID:
            return self.half_spread_bps_liquid
        if tier is LiquidityTier.MID:
            return self.half_spread_bps_mid
        if tier is LiquidityTier.ILLIQUID:
            return self.half_spread_bps_illiquid
        raise ValueError(f"unknown liquidity tier: {tier!r}")  # defensive; enum-exhaustive

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict echo of every parameter (stable key order, JSON-serialisable)."""
        return {
            "stk_commission_per_share": self.stk_commission_per_share,
            "stk_commission_min": self.stk_commission_min,
            "fut_commission_per_contract": self.fut_commission_per_contract,
            "crypto_commission_rate": self.crypto_commission_rate,
            "crypto_commission_min": self.crypto_commission_min,
            "half_spread_bps_liquid": self.half_spread_bps_liquid,
            "half_spread_bps_mid": self.half_spread_bps_mid,
            "half_spread_bps_illiquid": self.half_spread_bps_illiquid,
            "slippage_base_bps": self.slippage_base_bps,
            "slippage_vol_coeff": self.slippage_vol_coeff,
            "slippage_participation_coeff": self.slippage_participation_coeff,
            "enable_market_impact": self.enable_market_impact,
            "impact_coeff": self.impact_coeff,
        }


DEFAULT_COST_CONFIG: CostConfig = CostConfig()


# --------------------------------------------------------------------------- #
# Trade input — one round trip. Constructable from synthetic fields (Stage 1) or
# from a real fill mapping (Stage 2) via :meth:`from_fill`; both feed apply_costs.
# --------------------------------------------------------------------------- #
def _require_positive(name: str, value: Number) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    v = float(value)
    if not (v > 0.0):
        raise ValueError(f"{name} must be > 0, got {v}")
    return v


def _require_nonneg_optional(name: str, value: Optional[Number]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number or None, got {value!r}")
    v = float(value)
    if v < 0.0:
        raise ValueError(f"{name} must be >= 0 when supplied, got {v}")
    return v


@dataclass(frozen=True)
class Trade:
    """One round-trip trade (an entry fill + an exit fill) to be costed.

    ``quantity``/``entry_price``/``exit_price``/``multiplier`` must be strictly
    positive (a malformed trade is a caller bug → ``ValueError``). ``volatility``
    (a fraction, e.g. 0.02 for 2%), ``participation`` (``qty/bar_volume`` fraction),
    and ``bar_volume`` are optional; a missing one degrades to a conservative
    default term and records a warning (see module docstring). If both
    ``participation`` and ``bar_volume`` are given, ``participation`` wins.
    ``gross_pnl`` is the pre-cost PnL (currency); when present, ``net_pnl`` is
    computed. ``multiplier`` is the contract/point multiplier (futures); 1.0 for
    equities/crypto.
    """

    instrument_class: InstrumentClass
    quantity: float
    entry_price: float
    exit_price: float
    liquidity_tier: LiquidityTier = LiquidityTier.LIQUID
    multiplier: float = 1.0
    volatility: Optional[float] = None
    participation: Optional[float] = None
    bar_volume: Optional[float] = None
    gross_pnl: Optional[float] = None
    label: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.instrument_class, InstrumentClass):
            raise ValueError(
                f"instrument_class must be an InstrumentClass, got {self.instrument_class!r}"
            )
        if not isinstance(self.liquidity_tier, LiquidityTier):
            raise ValueError(
                f"liquidity_tier must be a LiquidityTier, got {self.liquidity_tier!r}"
            )
        _require_positive("quantity", self.quantity)
        _require_positive("entry_price", self.entry_price)
        _require_positive("exit_price", self.exit_price)
        _require_positive("multiplier", self.multiplier)
        _require_nonneg_optional("volatility", self.volatility)
        _require_nonneg_optional("participation", self.participation)
        _require_nonneg_optional("bar_volume", self.bar_volume)
        if self.gross_pnl is not None and (
            isinstance(self.gross_pnl, bool) or not isinstance(self.gross_pnl, (int, float))
        ):
            raise ValueError(f"gross_pnl must be a real number or None, got {self.gross_pnl!r}")

    @classmethod
    def from_fill(cls, fill: Mapping[str, Any]) -> "Trade":
        """Build a :class:`Trade` from a real-fill mapping (Stage-2 seam, SSOT §3.5).

        Accepts common aliases so a broker/paper-fill record maps onto the same
        costing path as a synthetic Stage-1 trade — proving one haircut for both
        stages. String ``instrument_class`` / ``liquidity_tier`` are coerced to
        their enums. Unknown keys are ignored; required keys missing → ``KeyError``
        via the explicit lookups below (a caller bug, surfaced fast).
        """

        def _first(*keys: str, default: Any = None) -> Any:
            for k in keys:
                if k in fill:
                    return fill[k]
            return default

        inst_raw = _first("instrument_class", "instrument", "asset_class")
        inst = inst_raw if isinstance(inst_raw, InstrumentClass) else InstrumentClass(str(inst_raw))
        tier_raw = _first("liquidity_tier", "tier", default=LiquidityTier.LIQUID)
        tier = tier_raw if isinstance(tier_raw, LiquidityTier) else LiquidityTier(str(tier_raw))
        return cls(
            instrument_class=inst,
            quantity=_first("quantity", "qty", "size"),
            entry_price=_first("entry_price", "entry", "open_price", "fill_price_in"),
            exit_price=_first("exit_price", "exit", "close_price", "fill_price_out"),
            liquidity_tier=tier,
            multiplier=_first("multiplier", "mult", default=1.0),
            volatility=_first("volatility", "vol", "realized_vol"),
            participation=_first("participation"),
            bar_volume=_first("bar_volume", "volume"),
            gross_pnl=_first("gross_pnl", "pnl", "pnl_gross"),
            label=_first("label", "head", "strategy"),
        )


# --------------------------------------------------------------------------- #
# Output — flat, serialisable, self-describing (echoes the config that made it).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CostBreakdown:
    """Every cost component of a trade plus its provenance (SSOT §3.5 "printed")."""

    commission: float
    spread_cost: float
    slippage_cost: float
    impact_cost: float
    total_cost: float
    instrument_class: str
    liquidity_tier: str
    market_impact_enabled: bool
    gross_pnl: Optional[float]
    net_pnl: Optional[float]
    warnings: tuple[str, ...]
    config_echo: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Plain dict of every field (JSON-serialisable, stable key order)."""
        return {
            "commission": self.commission,
            "spread_cost": self.spread_cost,
            "slippage_cost": self.slippage_cost,
            "impact_cost": self.impact_cost,
            "total_cost": self.total_cost,
            "instrument_class": self.instrument_class,
            "liquidity_tier": self.liquidity_tier,
            "market_impact_enabled": self.market_impact_enabled,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "warnings": list(self.warnings),
            "config_echo": self.config_echo,
        }


def _commission_leg(
    inst: InstrumentClass, qty: float, notional: float, cfg: CostConfig
) -> float:
    """Commission for a single leg (entry or exit) by instrument class."""
    if inst is InstrumentClass.STK:
        return max(cfg.stk_commission_per_share * qty, cfg.stk_commission_min)
    if inst is InstrumentClass.FUT:
        return cfg.fut_commission_per_contract * qty
    if inst is InstrumentClass.CRYPTO:
        return max(cfg.crypto_commission_rate * notional, cfg.crypto_commission_min)
    raise ValueError(f"unknown instrument class: {inst!r}")  # defensive; enum-exhaustive


def apply_costs(trade: Trade, config: CostConfig = DEFAULT_COST_CONFIG) -> CostBreakdown:
    """Charge commission + spread (both legs) + slippage (+ optional impact) on ``trade``.

    THE single costing path (SSOT §3.5 / S4): used by synthetic Stage-1 trades and
    real Stage-2 paper fills alike. Deterministic and pure. Never raises on
    degenerate-but-valid data (missing volatility/volume degrade to conservative
    zero-terms with a recorded warning); only a malformed :class:`Trade` (rejected
    at construction) or invalid :class:`CostConfig` can raise.

    Returns a :class:`CostBreakdown` with each component, ``total_cost >= 0``, the
    echoed config, any warnings, and ``net_pnl = gross_pnl - total_cost`` when a
    pre-cost ``gross_pnl`` was supplied (else ``None``).
    """
    if not isinstance(trade, Trade):
        raise ValueError(f"apply_costs expects a Trade, got {type(trade).__name__}")
    if not isinstance(config, CostConfig):
        raise ValueError(f"config must be a CostConfig, got {type(config).__name__}")

    warnings: list[str] = []
    inst = trade.instrument_class
    qty = trade.quantity
    mult = trade.multiplier

    # --- resolve volume-linked participation (fraction of bar volume) ------- #
    if trade.participation is not None:
        participation = trade.participation
    elif trade.bar_volume is not None and trade.bar_volume > 0.0:
        participation = qty / trade.bar_volume
    else:
        participation = 0.0
        warnings.append(
            "no participation/bar_volume provided; volume-linked slippage term = 0 "
            "(and market impact = 0 if enabled)"
        )

    # --- resolve volatility fraction ---------------------------------------- #
    if trade.volatility is not None:
        volatility = trade.volatility
    else:
        volatility = 0.0
        warnings.append("no volatility provided; volatility-linked slippage term = 0")

    # --- fractional rates (of per-leg notional) ----------------------------- #
    half_spread_frac = config.half_spread_bps_for(trade.liquidity_tier) * _BPS
    slip_frac = (
        config.slippage_base_bps * _BPS
        + config.slippage_vol_coeff * volatility
        + config.slippage_participation_coeff * participation
    )
    if config.enable_market_impact:
        impact_frac = config.impact_coeff * math.sqrt(participation)
    else:
        impact_frac = 0.0

    # --- charge BOTH legs (entry price, exit price) ------------------------- #
    commission = 0.0
    spread_cost = 0.0
    slippage_cost = 0.0
    impact_cost = 0.0
    for leg_price in (trade.entry_price, trade.exit_price):
        notional = leg_price * qty * mult
        commission += _commission_leg(inst, qty, notional, config)
        spread_cost += half_spread_frac * notional
        slippage_cost += slip_frac * notional
        impact_cost += impact_frac * notional

    total_cost = commission + spread_cost + slippage_cost + impact_cost
    net_pnl: Optional[float] = None
    if trade.gross_pnl is not None:
        net_pnl = float(trade.gross_pnl) - total_cost

    return CostBreakdown(
        commission=commission,
        spread_cost=spread_cost,
        slippage_cost=slippage_cost,
        impact_cost=impact_cost,
        total_cost=total_cost,
        instrument_class=inst.value,
        liquidity_tier=trade.liquidity_tier.value,
        market_impact_enabled=config.enable_market_impact,
        gross_pnl=None if trade.gross_pnl is None else float(trade.gross_pnl),
        net_pnl=net_pnl,
        warnings=tuple(warnings),
        config_echo=config.to_dict(),
    )


# --------------------------------------------------------------------------- #
# Pessimistic intrabar resolution (V2, SSOT §3.5).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IntrabarResolution:
    """Which of stop/target a daily bar resolved to, and whether it was ambiguous.

    ``outcome`` is ``"stop"`` | ``"target"`` | ``"none"``. ``fill_price`` is the
    level that was hit (``None`` for ``"none"``). ``ambiguous`` is ``True`` iff both
    stop AND target lay inside the bar range — in which case the STOP is returned
    (the worse assumption) and the fill is flagged for counting (SSOT §3.5 "log
    every ambiguous fill").
    """

    outcome: str
    fill_price: Optional[float]
    ambiguous: bool
    stop_in_range: bool
    target_in_range: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "fill_price": self.fill_price,
            "ambiguous": self.ambiguous,
            "stop_in_range": self.stop_in_range,
            "target_in_range": self.target_in_range,
            "reason": self.reason,
        }


def _bar_high_low(bar: Mapping[str, Any]) -> tuple[float, float]:
    """Extract ``(high, low)`` from a bar mapping (canonical Phase-0 schema keys)."""
    try:
        high = float(bar["high"])
        low = float(bar["low"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"bar must have numeric 'high' and 'low': {exc}") from exc
    return high, low


def resolve_intrabar(
    bar: Mapping[str, Any], stop: Number, target: Number
) -> IntrabarResolution:
    """Resolve stop-vs-target for a daily OHLC ``bar`` under the pessimistic rule (V2).

    A daily bar cannot reveal the *order* in which prices were touched, so when its
    ``[low, high]`` range (inclusive) contains BOTH the ``stop`` and the ``target``,
    the STOP is assumed to have hit first (the worse outcome) and ``ambiguous`` is
    ``True``. When exactly one level is in range, that level is returned
    unambiguously; when neither is in range, ``outcome == "none"`` (no fill this
    bar). Direction-agnostic: it only asks which levels the bar traversed, so it is
    correct for both long (stop<entry<target) and short (target<entry<stop) trades.

    Raises ``ValueError`` only on a malformed bar (missing/non-numeric high/low, or
    ``high < low``) — a data-integrity bug, not degenerate-but-valid data.
    """
    high, low = _bar_high_low(bar)
    if high < low:
        raise ValueError(f"malformed bar: high ({high}) < low ({low})")
    if isinstance(stop, bool) or not isinstance(stop, (int, float)):
        raise ValueError(f"stop must be a real number, got {stop!r}")
    if isinstance(target, bool) or not isinstance(target, (int, float)):
        raise ValueError(f"target must be a real number, got {target!r}")
    stop_f = float(stop)
    target_f = float(target)

    stop_in = low <= stop_f <= high
    target_in = low <= target_f <= high

    if stop_in and target_in:
        return IntrabarResolution(
            outcome="stop",
            fill_price=stop_f,
            ambiguous=True,
            stop_in_range=True,
            target_in_range=True,
            reason="both stop and target within bar range; pessimistic rule assumes stop first",
        )
    if stop_in:
        return IntrabarResolution(
            outcome="stop",
            fill_price=stop_f,
            ambiguous=False,
            stop_in_range=True,
            target_in_range=False,
            reason="only stop within bar range",
        )
    if target_in:
        return IntrabarResolution(
            outcome="target",
            fill_price=target_f,
            ambiguous=False,
            stop_in_range=False,
            target_in_range=True,
            reason="only target within bar range",
        )
    return IntrabarResolution(
        outcome="none",
        fill_price=None,
        ambiguous=False,
        stop_in_range=False,
        target_in_range=False,
        reason="neither stop nor target within bar range",
    )
