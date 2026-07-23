"""
chad/risk/allocator_limits.py — W5B-2 limits loading + marginal evaluation.

Consumes `config/portfolio_limits.json` (portfolio_limits.v1) and renders a
would-verdict for ONE intent's *marginal* contribution to a `ProvisionalBook`.

THE HONESTY SPLIT. Every limit declares `binds`. A binding limit produces
WOULD_REJECT / WOULD_RESIZE evidence; a non-binding one is computed, reported,
and never rejected against. This is not timidity — a would-reject rendered
against a number nobody sourced is an unfalsifiable verdict, and a corpus full
of those cannot be used to argue for enforcement later. Today:

  binds     gross_exposure ($750k, DERIVED — basis shadow_derivation_2026-07)
            per_symbol_concentration ($150k, sourced + already enforced)
            per_sector ($375k, DERIVED — basis shadow_derivation_2026-07)
            per_venue KRAKEN ($184.58, the real wallet)
  reports   net_exposure (no cap exists anywhere)
            per_venue IBKR (no venue cap exists; falls under firm gross)
            the 5%-of-equity per-symbol leg (needs an FX rate that doesn't exist)
            the `unmapped` sector bucket (W4A LC3 idiom: never trippable)

"DERIVED" means exactly what the config says: a shadow threshold that makes
would-reject evidence possible, not a ratified limit. Nothing here enforces —
there is no enforce path in W5B at all.

WOULD_RESIZE is offered only when a smaller quantity would genuinely fit under
every binding limit; otherwise the verdict is WOULD_REJECT. The resize quantity
is floored to whole units, because no venue here accepts fractional shares or
contracts, and a resize suggestion that cannot be submitted is noise.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from chad.risk.portfolio_allocator import (
    UNMAPPED_SECTOR,
    ExposureVector,
    ProvisionalBook,
)

LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
LIMITS_PATH = REPO_ROOT / "config" / "portfolio_limits.json"

SCHEMA_VERSION = "portfolio_limits.v1"

# Verdicts.
WOULD_APPROVE = "WOULD_APPROVE"
WOULD_RESIZE = "WOULD_RESIZE"
WOULD_REJECT = "WOULD_REJECT"
ERROR = "ERROR"

# Limit dimension identifiers — also the coach's dedupe dimension (W5B-5) and
# the `which_limit` field in evidence.
LIMIT_GROSS = "gross"
LIMIT_NET = "net"
LIMIT_PER_SYMBOL = "per_symbol"
LIMIT_PER_SECTOR = "per_sector"
LIMIT_VENUE = "venue"


@dataclasses.dataclass(frozen=True)
class LimitCheck:
    """One dimension's arithmetic. `binds` decides whether a breach can turn
    into a WOULD_REJECT or is merely reported."""

    limit: str
    binds: bool
    cap_usd: Optional[float]
    basis: str
    ratified: bool
    projected_usd: float          # the dimension's value AFTER this intent
    prior_usd: float              # ... and before it
    headroom_usd: Optional[float]  # cap - projected (None when no cap)
    breached: bool
    breach_by_usd: Optional[float]
    breach_by_pct: Optional[float]
    scope: Optional[str] = None    # e.g. the symbol / sector / venue in question

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        for k in ("cap_usd", "projected_usd", "prior_usd", "headroom_usd",
                  "breach_by_usd", "breach_by_pct"):
            if isinstance(d[k], float):
                d[k] = round(d[k], 4)
        return d


@dataclasses.dataclass(frozen=True)
class AllocatorVerdict:
    verdict: str
    which_limit: Optional[str]
    reason: str
    checks: Tuple[LimitCheck, ...]
    would_resize_to_qty: Optional[float] = None
    breach_by_usd: Optional[float] = None
    breach_by_pct: Optional[float] = None
    headroom_usd: Optional[float] = None

    @property
    def is_breach(self) -> bool:
        return self.verdict in (WOULD_REJECT, WOULD_RESIZE)


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class PortfolioLimits:
    raw: Mapping[str, Any]

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "PortfolioLimits":
        """Missing/corrupt config ⇒ an all-non-binding limits object. A broken
        config must never invent a binding cap, and it must never crash the
        cycle: the allocator is an observer and its failure mode is silence."""
        target = Path(path) if path is not None else LIMITS_PATH
        try:
            obj = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(obj, dict):
                raise ValueError("not an object")
            if str(obj.get("schema_version")) != SCHEMA_VERSION:
                LOG.warning(
                    "ALLOCATOR_LIMITS_SCHEMA_MISMATCH got=%s want=%s",
                    obj.get("schema_version"), SCHEMA_VERSION,
                )
        except Exception as exc:  # noqa: BLE001
            LOG.warning("ALLOCATOR_LIMITS_UNREADABLE path=%s err=%s", target, exc)
            return cls(raw={})
        return cls(raw=obj)

    # -- accessors --------------------------------------------------------- #

    def _firm(self, key: str) -> Mapping[str, Any]:
        firm = self.raw.get("firm")
        if not isinstance(firm, Mapping):
            return {}
        node = firm.get(key)
        return node if isinstance(node, Mapping) else {}

    def cap(self, key: str) -> Tuple[Optional[float], bool, str, bool]:
        """(cap_usd, binds, basis, ratified) for a firm-level dimension.

        A cap of None can never bind, whatever the config claims — that pairing
        would otherwise mean "reject everything against no number".
        """
        node = self._firm(key)
        raw_cap = node.get("cap_notional_usd")
        try:
            cap_usd = None if raw_cap is None else float(raw_cap)
        except (TypeError, ValueError):
            cap_usd = None
        binds = bool(node.get("binds", False)) and cap_usd is not None
        return cap_usd, binds, str(node.get("basis") or "unknown"), bool(node.get("ratified", False))

    def venue_cap(self, venue: str) -> Tuple[Optional[float], bool, str, bool]:
        node = self._firm("per_venue").get(str(venue).upper()) if self._firm("per_venue") else None
        if not isinstance(node, Mapping):
            return None, False, "unknown", False
        raw_cap = node.get("cap_usd")
        try:
            cap_usd = None if raw_cap is None else float(raw_cap)
        except (TypeError, ValueError):
            cap_usd = None
        binds = bool(node.get("binds", False)) and cap_usd is not None
        return cap_usd, binds, str(node.get("basis") or "unknown"), bool(node.get("ratified", False))

    @property
    def sector_unmapped_binds(self) -> bool:
        return bool(self._firm("per_sector").get("unmapped_bucket_binds", False))

    @property
    def reject_streak_n(self) -> int:
        coach = self.raw.get("coach")
        if isinstance(coach, Mapping):
            try:
                return max(1, int(coach.get("reject_streak_n", 3)))
            except (TypeError, ValueError):
                pass
        return 3

    @property
    def correlation_mode(self) -> Mapping[str, Any]:
        """Declarative only — mode + deferral, never a computed rho (§13.3)."""
        node = self.raw.get("correlation")
        if not isinstance(node, Mapping):
            return {"mode": "static_sector_buckets", "rolling_deferred_to": "R2"}
        return {
            "mode": str(node.get("mode") or "static_sector_buckets"),
            "rolling_deferred_to": str(node.get("rolling_deferred_to") or "R2"),
            "existing_per_order_reducer": (
                (node.get("existing_per_order_reducer") or {}).get("module")
                if isinstance(node.get("existing_per_order_reducer"), Mapping)
                else None
            ),
        }

    @property
    def equity_basis(self) -> Dict[str, Any]:
        node = self.raw.get("equity_basis")
        if not isinstance(node, Mapping):
            return {}
        return {
            "value_cad": node.get("value_cad"),
            "currency": node.get("currency"),
            "used_as_divisor": bool(node.get("used_as_divisor", False)),
        }


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def _mk_check(
    *,
    limit: str,
    cap_usd: Optional[float],
    binds: bool,
    basis: str,
    ratified: bool,
    prior: float,
    projected: float,
    scope: Optional[str] = None,
) -> LimitCheck:
    breached = cap_usd is not None and projected > cap_usd
    breach_by = (projected - cap_usd) if (breached and cap_usd is not None) else None
    breach_pct = (
        (breach_by / cap_usd * 100.0)
        if (breach_by is not None and cap_usd not in (None, 0))
        else None
    )
    return LimitCheck(
        limit=limit,
        binds=binds,
        cap_usd=cap_usd,
        basis=basis,
        ratified=ratified,
        projected_usd=projected,
        prior_usd=prior,
        headroom_usd=(None if cap_usd is None else cap_usd - projected),
        breached=breached,
        breach_by_usd=breach_by,
        breach_by_pct=breach_pct,
        scope=scope,
    )


def evaluate_marginal(
    vector: ExposureVector,
    book: ProvisionalBook,
    limits: PortfolioLimits,
) -> AllocatorVerdict:
    """Would-verdict for one intent's marginal contribution.

    `book` is the provisional book BEFORE this vector is added; the projected
    values are computed by adding it. The caller adds the vector to the book
    afterwards regardless of verdict (shadow blocks nothing, so the honest
    counterfactual is the book enforcement would have seen).

    An intent whose exposure is not computable (unmapped futures root, no price)
    yields WOULD_APPROVE with an explicit reason — the allocator must not
    manufacture a verdict from an unknown, and must not silently treat it as 0.
    """
    if not vector.computable:
        return AllocatorVerdict(
            verdict=WOULD_APPROVE,
            which_limit=None,
            reason=f"not_evaluable:{vector.null_reason or 'unknown'}",
            checks=(),
        )

    delta = float(vector.delta_usd or 0.0)
    checks: List[LimitCheck] = []

    # -- firm gross (absolute value adds; a short adds to gross too) -------- #
    cap, binds, basis, ratified = limits.cap("gross_exposure")
    prior_gross = book.gross_usd
    checks.append(_mk_check(
        limit=LIMIT_GROSS, cap_usd=cap, binds=binds, basis=basis, ratified=ratified,
        prior=prior_gross, projected=prior_gross + abs(delta),
    ))

    # -- firm net (signed; report-only, no cap exists) ---------------------- #
    cap, binds, basis, ratified = limits.cap("net_exposure")
    prior_net = book.net_usd
    checks.append(_mk_check(
        limit=LIMIT_NET, cap_usd=cap, binds=binds, basis=basis, ratified=ratified,
        prior=abs(prior_net), projected=abs(prior_net + delta),
    ))

    # -- per-symbol concentration ------------------------------------------- #
    # Measured on the COMBINED position+intent ticket — the grain policy.py's
    # per-order cap structurally cannot see.
    cap, binds, basis, ratified = limits.cap("per_symbol_concentration")
    prior_sym = book.by_symbol().get(vector.symbol, 0.0)
    checks.append(_mk_check(
        limit=LIMIT_PER_SYMBOL, cap_usd=cap, binds=binds, basis=basis,
        ratified=ratified, prior=abs(prior_sym), projected=abs(prior_sym + delta),
        scope=vector.symbol,
    ))

    # -- per-sector --------------------------------------------------------- #
    cap, binds, basis, ratified = limits.cap("per_sector")
    sector = vector.sector or UNMAPPED_SECTOR
    if sector == UNMAPPED_SECTOR and not limits.sector_unmapped_binds:
        binds = False  # W4A LC3 idiom: the unmapped bucket is never trippable.
    prior_sec = book.by_sector().get(sector, 0.0)
    checks.append(_mk_check(
        limit=LIMIT_PER_SECTOR, cap_usd=cap, binds=binds, basis=basis,
        ratified=ratified, prior=abs(prior_sec), projected=abs(prior_sec + delta),
        scope=sector,
    ))

    # -- per-venue ---------------------------------------------------------- #
    cap, binds, basis, ratified = limits.venue_cap(vector.venue)
    prior_venue = book.by_venue().get(vector.venue, 0.0)
    checks.append(_mk_check(
        limit=LIMIT_VENUE, cap_usd=cap, binds=binds, basis=basis,
        ratified=ratified, prior=prior_venue, projected=prior_venue + abs(delta),
        scope=vector.venue,
    ))

    binding_breaches = [c for c in checks if c.binds and c.breached]
    if not binding_breaches:
        tightest = min(
            (c for c in checks if c.headroom_usd is not None and c.binds),
            key=lambda c: c.headroom_usd,
            default=None,
        )
        return AllocatorVerdict(
            verdict=WOULD_APPROVE,
            which_limit=None,
            reason="within_all_binding_limits",
            checks=tuple(checks),
            headroom_usd=(tightest.headroom_usd if tightest else None),
        )

    # Worst breach by absolute dollars decides `which_limit`.
    worst = max(binding_breaches, key=lambda c: c.breach_by_usd or 0.0)
    resize_qty = _resize_quantity(vector, book, binding_breaches)
    if resize_qty is not None and resize_qty > 0:
        return AllocatorVerdict(
            verdict=WOULD_RESIZE,
            which_limit=worst.limit,
            reason=f"binding_breach:{worst.limit}",
            checks=tuple(checks),
            would_resize_to_qty=resize_qty,
            breach_by_usd=worst.breach_by_usd,
            breach_by_pct=worst.breach_by_pct,
            headroom_usd=worst.headroom_usd,
        )
    return AllocatorVerdict(
        verdict=WOULD_REJECT,
        which_limit=worst.limit,
        reason=f"binding_breach:{worst.limit}",
        checks=tuple(checks),
        breach_by_usd=worst.breach_by_usd,
        breach_by_pct=worst.breach_by_pct,
        headroom_usd=worst.headroom_usd,
    )


def _resize_quantity(
    vector: ExposureVector,
    book: ProvisionalBook,
    breaches: List[LimitCheck],
) -> Optional[float]:
    """Largest whole quantity that fits under EVERY breached binding limit.

    Uses the pre-intent headroom of each breached dimension (cap - prior), takes
    the tightest, and converts back to units at this ticket's price × multiplier.
    Floored to whole units: no venue here accepts fractional shares or contracts,
    so a fractional suggestion would be unsubmittable. Returns None when nothing
    whole fits — the caller then renders WOULD_REJECT.
    """
    try:
        unit_value = abs(float(vector.price or 0.0) * float(vector.multiplier or 0.0))
        if unit_value <= 0:
            return None
        room = min(
            (c.cap_usd - c.prior_usd)
            for c in breaches
            if c.cap_usd is not None
        )
        if room <= 0:
            return None
        qty = math.floor(room / unit_value)
        return float(qty) if qty >= 1 else None
    except Exception:  # noqa: BLE001 — a resize hint is never load-bearing
        return None


__all__ = [
    "AllocatorVerdict",
    "LimitCheck",
    "PortfolioLimits",
    "SCHEMA_VERSION",
    "ERROR",
    "WOULD_APPROVE",
    "WOULD_REJECT",
    "WOULD_RESIZE",
    "LIMIT_GROSS",
    "LIMIT_NET",
    "LIMIT_PER_SECTOR",
    "LIMIT_PER_SYMBOL",
    "LIMIT_VENUE",
    "evaluate_marginal",
]
