"""
chad/analytics/pnl_breakdown.py

Per-trade P&L breakdown — canonical schema for explaining how a closed
trade's net P&L decomposes into gross price movement, commissions, fees,
and slippage.

Goals:

  * Provide a single, well-documented nested object (``pnl_breakdown``)
    that future readers can rely on instead of grepping six top-level
    fields with inconsistent default semantics.
  * Never silently turn an *unknown* cost into a zero. When a cost is
    not actually known, the breakdown stores ``None`` and flags
    ``cost_basis_status`` accordingly.
  * Stay backwards compatible with the existing top-level fields
    (``pnl``, ``gross_pnl``, ``commission``, ``slippage``, ``net_pnl``,
    ``fees``) so today's profit_lock / SCR / weekly-investor readers
    are not disturbed.
  * Be a pure, side-effect-free helper. It builds dicts; it never reads
    or writes files, never mutates inputs, never reaches into runtime.

The canonical structure produced by this module is::

    {
        "schema_version": "pnl_breakdown.v1",
        "gross_price_pnl": float,
        "commission": float | None,
        "fees": float | None,
        "slippage": float | None,
        "net_pnl": float,
        "entry_price": float | None,
        "exit_price": float | None,
        "quantity": float | None,
        "contract_multiplier": float,
        "currency": "USD",
        "source": "paper_exec" | "ibkr" | "kraken" | "legacy" | "unknown",
        "cost_basis_status": "real" | "estimated" | "unavailable" |
                             "legacy" | "untrusted",
        "notes": [str, ...],
    }
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

PNL_BREAKDOWN_SCHEMA = "pnl_breakdown.v1"

VALID_COST_BASIS_STATUSES = (
    "real",
    "estimated",
    "unavailable",
    "legacy",
    "untrusted",
)

VALID_SOURCES = (
    "paper_exec",
    "ibkr",
    "kraken",
    "legacy",
    "unknown",
)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f


def _round_money(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    return round(v, 6)


def _is_untrusted(payload: Mapping[str, Any]) -> bool:
    """Mirror chad/utils/quarantine._payload_is_untrusted but local + pure."""
    if not isinstance(payload, Mapping):
        return False
    if payload.get("pnl_untrusted") is True:
        return True
    extra = payload.get("extra")
    if isinstance(extra, Mapping) and extra.get("pnl_untrusted") is True:
        return True
    tags = payload.get("tags")
    if isinstance(tags, list) and any(
        str(t).strip().lower() == "pnl_untrusted" for t in tags
    ):
        return True
    return False


def _coerce_status(status: Any) -> str:
    s = str(status or "").strip().lower()
    if s in VALID_COST_BASIS_STATUSES:
        return s
    return "unavailable"


def _coerce_source(source: Any) -> str:
    s = str(source or "").strip().lower()
    if s in VALID_SOURCES:
        return s
    return "unknown"


def build_pnl_breakdown(
    *,
    gross_price_pnl: float,
    entry_price: Optional[float] = None,
    exit_price: Optional[float] = None,
    quantity: Optional[float] = None,
    contract_multiplier: float = 1.0,
    commission: Optional[float] = None,
    fees: Optional[float] = None,
    slippage: Optional[float] = None,
    source: str = "unknown",
    cost_basis_status: Optional[str] = None,
    currency: str = "USD",
    notes: Optional[list] = None,
) -> dict:
    """Construct a canonical pnl_breakdown.v1 dict.

    Cost components are accepted as ``None`` to signal *unavailable*; the
    helper will not coerce them to ``0.0``. ``net_pnl`` is computed as
    ``gross_price_pnl`` minus only the cost components that are actually
    known (non-None). When all three are ``None`` the breakdown is marked
    ``cost_basis_status="unavailable"`` and ``net_pnl`` equals
    ``gross_price_pnl`` (best-effort, with the unavailability flag making
    the uncertainty explicit to consumers).
    """
    note_list = list(notes or [])
    gross = float(_safe_float(gross_price_pnl) or 0.0)

    comm = _safe_float(commission)
    fees_v = _safe_float(fees)
    slip = _safe_float(slippage)

    known_costs = [c for c in (comm, fees_v, slip) if c is not None]
    deductions = sum(known_costs) if known_costs else 0.0
    net = gross - deductions

    if cost_basis_status is None:
        if comm is None and fees_v is None and slip is None:
            status = "unavailable"
        elif comm is not None and fees_v is not None and slip is not None:
            status = "real"
        else:
            status = "estimated"
    else:
        status = _coerce_status(cost_basis_status)

    if comm is None and "commission_unavailable" not in note_list:
        note_list.append("commission_unavailable")
    if fees_v is None and "fees_unavailable" not in note_list:
        note_list.append("fees_unavailable")
    if slip is None and "slippage_unavailable" not in note_list:
        note_list.append("slippage_unavailable")

    return {
        "schema_version": PNL_BREAKDOWN_SCHEMA,
        "gross_price_pnl": _round_money(gross),
        "commission": _round_money(comm),
        "fees": _round_money(fees_v),
        "slippage": _round_money(slip),
        "net_pnl": _round_money(net),
        "entry_price": _safe_float(entry_price),
        "exit_price": _safe_float(exit_price),
        "quantity": _safe_float(quantity),
        "contract_multiplier": float(_safe_float(contract_multiplier) or 1.0),
        "currency": str(currency or "USD"),
        "source": _coerce_source(source),
        "cost_basis_status": status,
        "notes": note_list,
    }


def normalize_to_breakdown(payload: Mapping[str, Any]) -> dict:
    """Return a canonical breakdown for *payload* without mutating it.

    Behavior:

    * If ``payload['pnl_breakdown']`` already exists and is a mapping,
      return a shallow copy with missing canonical keys defaulted (so
      the caller never has to defensively ``.get`` after this).
    * Otherwise (legacy / external record), derive a breakdown from the
      top-level fields. Such records are tagged ``cost_basis_status =
      "legacy"`` because we cannot tell whether their stored
      ``commission`` / ``slippage`` were truly zero or silently defaulted.
    * If the payload carries any ``pnl_untrusted`` marker, override
      ``cost_basis_status`` to ``"untrusted"`` and add a note. This
      mirrors the existing quarantine semantics so downstream
      analytics can refuse to count the trade.
    """
    if not isinstance(payload, Mapping):
        return build_pnl_breakdown(gross_price_pnl=0.0, source="unknown")

    existing = payload.get("pnl_breakdown")
    if isinstance(existing, Mapping):
        out = {
            "schema_version": str(existing.get("schema_version", PNL_BREAKDOWN_SCHEMA)),
            "gross_price_pnl": _round_money(_safe_float(existing.get("gross_price_pnl"))),
            "commission": _round_money(_safe_float(existing.get("commission"))),
            "fees": _round_money(_safe_float(existing.get("fees"))),
            "slippage": _round_money(_safe_float(existing.get("slippage"))),
            "net_pnl": _round_money(_safe_float(existing.get("net_pnl"))),
            "entry_price": _safe_float(existing.get("entry_price")),
            "exit_price": _safe_float(existing.get("exit_price")),
            "quantity": _safe_float(existing.get("quantity")),
            "contract_multiplier": float(
                _safe_float(existing.get("contract_multiplier")) or 1.0
            ),
            "currency": str(existing.get("currency") or "USD"),
            "source": _coerce_source(existing.get("source")),
            "cost_basis_status": _coerce_status(existing.get("cost_basis_status")),
            "notes": list(existing.get("notes") or []),
        }
    else:
        gross = _safe_float(payload.get("gross_pnl"))
        if gross is None:
            gross = _safe_float(payload.get("pnl"))
        if gross is None:
            gross = _safe_float(payload.get("net_pnl"))
        gross = gross if gross is not None else 0.0

        comm = _safe_float(payload.get("commission"))
        fees_v = _safe_float(payload.get("fees"))
        slip = _safe_float(payload.get("slippage"))
        net = _safe_float(payload.get("net_pnl"))
        if net is None:
            net = _safe_float(payload.get("pnl"))
        if net is None:
            net = gross

        out = {
            "schema_version": PNL_BREAKDOWN_SCHEMA,
            "gross_price_pnl": _round_money(gross),
            "commission": _round_money(comm),
            "fees": _round_money(fees_v),
            "slippage": _round_money(slip),
            "net_pnl": _round_money(net),
            "entry_price": _safe_float(payload.get("entry_price")),
            "exit_price": _safe_float(payload.get("exit_price")),
            "quantity": _safe_float(payload.get("quantity")),
            "contract_multiplier": float(
                _safe_float(payload.get("contract_multiplier")) or 1.0
            ),
            "currency": "USD",
            "source": _coerce_source(payload.get("broker") or payload.get("source")),
            "cost_basis_status": "legacy",
            "notes": ["normalized_from_legacy_record"],
        }

    if _is_untrusted(payload):
        out["cost_basis_status"] = "untrusted"
        if "pnl_untrusted" not in out["notes"]:
            out["notes"].append("pnl_untrusted")

    return out


__all__ = [
    "PNL_BREAKDOWN_SCHEMA",
    "VALID_COST_BASIS_STATUSES",
    "VALID_SOURCES",
    "build_pnl_breakdown",
    "normalize_to_breakdown",
]
