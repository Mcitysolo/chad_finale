"""
chad/analytics/implementation_shortfall.py — W5A E2/E2-B Implementation Shortfall.

Per closed lap, compute IS = realized slippage + explicit fees + funding +
opportunity cost, in $, bps, and R units, benchmarked to the DECISION price.
This module is a pure JOINER (audit finding W5A §1-P2): the raw materials
already exist, scattered across the per-fill evidence — the decision reference
and realized slippage are on the FILL row's own `extra`
(`submit_quote.ref_price` / `expected_price` / `slippage_bps`, PA-EP2a), the
fee dollar amount is in `FEES_*.ndjson`, keyed to the fill by the shared tuple
`(symbol, side, fill_time_utc, strategy)`. E2 reconnects them; it computes
nothing the executor didn't already measure.

HONEST NULLS (rider R1): any leg that cannot be resolved from real evidence is
recorded as `None` with a reason code — never estimated, never zero-filled. A
lap with unknown costs is distinguishable from a lap with genuinely zero costs
(`cost_basis_status` + per-leg reasons). The harness decides what to do with a
null; this writer never guesses.

Output is the `implementation_shortfall.v1` sub-schema block, stamped additively
onto the `closed_trade.v1` payload (no top-level schema bump — the ledger is
hash-chained and exact-matched by execution-critical readers; see
audits/W5A_BASELINE.md and PLAN_W5A §1-P7). Both lanes: IBKR reads the sibling
evidence; the Kraken trusted lane's round-trip row carries entry/exit fee +
expected_price natively (richer) and is preferred when present.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILLS_DIR = REPO_ROOT / "data" / "fills"

IS_SCHEMA_VERSION = "implementation_shortfall.v1"

TCA_STAMP_ENV = "CHAD_TCA_STAMP"


def tca_stamp_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """CHAD_TCA_STAMP=on enables the closed_trade IS stamp. Default (and any
    non-'on' value) is off ⇒ byte-identical closed_trade rows."""
    src = env if env is not None else os.environ
    return str(src.get(TCA_STAMP_ENV, "")).strip().lower() == "on"

_FILLS_RE = re.compile(r"^FILLS_(\d{8})\.ndjson$")
_FEES_RE = re.compile(r"^FEES_(\d{8})\.ndjson$")

# Genuine executed-fill statuses (W4B-8f census parity — a rehearsal/exhaust
# status must never contribute a cost leg).
_GENUINE_FILL_STATUSES = frozenset({"paper_fill", "fill", "filled"})


# --------------------------------------------------------------------------- #
# Per-fill resolved cost (one leg)
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class FillCost:
    fill_id: str
    found: bool
    symbol: str
    side: str
    strategy: str
    quantity: float
    fill_price: float
    fill_time_utc: str
    status: str
    decision_price: Optional[float]
    decision_price_source: Optional[str]   # submit_quote_ref_price | expected_price | None
    decision_price_reason: str
    slippage_usd: Optional[float]
    slippage_reason: str
    fee_usd: Optional[float]
    fee_reason: str


def _f(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _fee_join_key(symbol: str, side: str, fill_time_utc: str, strategy: str) -> Tuple[str, str, str, str]:
    return (
        str(symbol or "").upper(),
        str(side or "").upper(),
        str(fill_time_utc or ""),
        str(strategy or "").lower(),
    )


def _dated_files(directory: Path, regex: re.Pattern, date: str) -> List[Path]:
    if not directory.is_dir():
        return []
    return [p for p in directory.iterdir() if regex.match(p.name) and regex.match(p.name).group(1) == date]


def build_fee_index(
    date: str, fills_dir: Optional[Path] = None
) -> Dict[Tuple[str, str, str, str], float]:
    """`(symbol, side, fill_time_utc, strategy)` → summed `fee_amount` from the
    day's FEES ledger. The fill and its fee are written from the same evidence
    object, so this tuple links them (fill_id and fee_id are distinct hashes of
    overlapping field sets, so neither derives from the other)."""
    src = Path(fills_dir) if fills_dir is not None else DEFAULT_FILLS_DIR
    out: Dict[Tuple[str, str, str, str], float] = {}
    for path in _dated_files(src, _FEES_RE, date):
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            p = rec.get("payload", rec)
            if not isinstance(p, Mapping):
                continue
            amt = _f(p.get("fee_amount"))
            if amt is None:
                continue
            key = _fee_join_key(
                p.get("symbol", ""), p.get("side", ""),
                p.get("fill_time_utc", ""), p.get("strategy", ""),
            )
            out[key] = out.get(key, 0.0) + amt
    return out


def build_fill_cost_index(
    date: str,
    *,
    fills_dir: Optional[Path] = None,
) -> Dict[str, FillCost]:
    """Build `fill_id → FillCost` for one day. Decision price + slippage come
    from the FILL's own `extra` (self-sufficient, PA-EP2a); fee $ is joined
    from the FEES ledger. Every unresolved leg carries a reason code (R1)."""
    src = Path(fills_dir) if fills_dir is not None else DEFAULT_FILLS_DIR
    fee_index = build_fee_index(date, fills_dir=src)
    out: Dict[str, FillCost] = {}
    for path in _dated_files(src, _FILLS_RE, date):
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            p = rec.get("payload", rec)
            if not isinstance(p, Mapping):
                continue
            fid = str(p.get("fill_id") or "").strip()
            if not fid:
                continue
            extra = p.get("extra") if isinstance(p.get("extra"), Mapping) else {}
            status = str(p.get("status") or "").strip().lower()
            symbol = str(p.get("symbol") or "").upper()
            side = str(p.get("side") or "").upper()
            strategy = str(p.get("strategy") or "").lower()
            fill_time = str(p.get("fill_time_utc") or "")
            qty = _f(p.get("quantity")) or 0.0
            fill_price = _f(p.get("fill_price")) or 0.0

            # Decision price: submit_quote.ref_price primary, expected_price
            # fallback, else honest null (R1). D2.
            decision_price: Optional[float] = None
            dp_source: Optional[str] = None
            dp_reason = "no_decision_stamp"
            sq = extra.get("submit_quote") if isinstance(extra.get("submit_quote"), Mapping) else None
            if sq is not None:
                ref = _f(sq.get("ref_price"))
                if ref is not None and ref > 0:
                    decision_price, dp_source, dp_reason = ref, "submit_quote_ref_price", "resolved"
            if decision_price is None:
                exp = _f(extra.get("expected_price"))
                if exp is not None and exp > 0:
                    decision_price, dp_source, dp_reason = exp, "expected_price", "resolved_fallback"

            # Slippage $: from the fill's own slippage_bps (self-contained).
            # slippage_bps sign convention (slippage_tracker): positive = adverse
            # (a cost). $ = bps/1e4 * fill_price * |qty|. Genuine fills only.
            slippage_usd: Optional[float] = None
            slip_reason = "no_slippage_bps"
            if status not in _GENUINE_FILL_STATUSES:
                slip_reason = "non_genuine_fill_status"
            else:
                sbps = _f(extra.get("slippage_bps"))
                if sbps is not None:
                    slippage_usd = round(sbps / 10000.0 * fill_price * abs(qty), 6)
                    slip_reason = "resolved_from_fill_bps"

            # Fee $: joined from FEES ledger.
            fee_usd: Optional[float] = None
            fee_reason = "no_fee_row"
            if status not in _GENUINE_FILL_STATUSES:
                fee_reason = "non_genuine_fill_status"
            else:
                fee = fee_index.get(_fee_join_key(symbol, side, fill_time, strategy))
                if fee is not None:
                    fee_usd = round(fee, 6)
                    fee_reason = "resolved_from_fees_ledger"

            out[fid] = FillCost(
                fill_id=fid, found=True, symbol=symbol, side=side,
                strategy=strategy, quantity=qty, fill_price=fill_price,
                fill_time_utc=fill_time, status=status,
                decision_price=decision_price, decision_price_source=dp_source,
                decision_price_reason=dp_reason,
                slippage_usd=slippage_usd, slippage_reason=slip_reason,
                fee_usd=fee_usd, fee_reason=fee_reason,
            )
    return out


def _absent_leg(fill_id: str) -> FillCost:
    """A lap fill_id with no FILLS row (e.g. a RECON_ADOPT seed lot / harvester-
    only close). Every cost leg is honest-null with reason `fill_not_found`."""
    return FillCost(
        fill_id=fill_id, found=False, symbol="", side="", strategy="",
        quantity=0.0, fill_price=0.0, fill_time_utc="", status="",
        decision_price=None, decision_price_source=None,
        decision_price_reason="fill_not_found",
        slippage_usd=None, slippage_reason="fill_not_found",
        fee_usd=None, fee_reason="fill_not_found",
    )


# --------------------------------------------------------------------------- #
# Lap-level aggregation
# --------------------------------------------------------------------------- #

def compute_lap_is(
    *,
    fill_ids: Iterable[str],
    quantity: float,
    contract_multiplier: float,
    broker: str,
    stop_width_usd: Optional[float],
    index: Mapping[str, FillCost],
    kraken_native: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Aggregate the per-leg costs of one closed lap into the
    `implementation_shortfall.v1` block. Honest nulls throughout (R1).

    `kraken_native` (optional): when the lap is a Kraken trusted round-trip
    whose row already carries `entry_fee`/`exit_fee`/`expected_price`, pass
    those here — they are preferred over the FILLS/FEES join (the crypto lane
    is richer, audit §1-P4).
    """
    legs: List[FillCost] = []
    for fid in fill_ids:
        legs.append(index.get(fid) or _absent_leg(str(fid)))

    # --- fees ---
    fees_reasons: List[str] = []
    if kraken_native and (
        kraken_native.get("entry_fee") is not None or kraken_native.get("exit_fee") is not None
    ):
        ef = _f(kraken_native.get("entry_fee")) or 0.0
        xf = _f(kraken_native.get("exit_fee")) or 0.0
        fees_usd: Optional[float] = round(ef + xf, 6)
        fees_reasons = ["resolved_kraken_native"]
    else:
        resolved_fees = [l.fee_usd for l in legs if l.fee_usd is not None]
        fees_usd = round(sum(resolved_fees), 6) if resolved_fees else None
        fees_reasons = sorted({l.fee_reason for l in legs})

    # --- slippage ---
    resolved_slip = [l.slippage_usd for l in legs if l.slippage_usd is not None]
    slippage_usd: Optional[float] = round(sum(resolved_slip), 6) if resolved_slip else None
    slip_reasons = sorted({l.slippage_reason for l in legs})

    # --- funding / opportunity cost: honest nulls, no source (R1, D3/D4) ---
    funding_usd = None
    funding_reason = "not_modeled_paper_lane"
    opportunity_cost_usd = None
    opportunity_cost_reason = "no_unfilled_qty_evidence"

    # --- decision price = the OPEN (entry) leg's reference ---
    entry_leg = legs[0] if legs else _absent_leg("")
    if kraken_native and _f(kraken_native.get("expected_price")):
        decision_price: Optional[float] = _f(kraken_native.get("expected_price"))
        decision_price_source = "kraken_expected_price"
        decision_price_reason = "resolved_kraken_native"
    else:
        decision_price = entry_leg.decision_price
        decision_price_source = entry_leg.decision_price_source
        decision_price_reason = entry_leg.decision_price_reason

    # --- IS total: sum of the RESOLVED cost legs (R1 — null when none) ---
    resolvable = [v for v in (slippage_usd, fees_usd, funding_usd, opportunity_cost_usd) if v is not None]
    is_usd: Optional[float] = round(sum(resolvable), 6) if resolvable else None

    # cost_basis_status: real only when BOTH slippage AND fees resolved for
    # EVERY leg; partial when some resolved; unavailable when none.
    all_slip = all(l.slippage_usd is not None for l in legs) and bool(legs)
    all_fee = (fees_usd is not None) and (
        bool(kraken_native and kraken_native.get("entry_fee") is not None)
        or all(l.fee_usd is not None for l in legs) and bool(legs)
    )
    if all_slip and all_fee:
        cost_basis_status = "real"
    elif is_usd is not None:
        cost_basis_status = "partial"
    else:
        cost_basis_status = "unavailable"

    # --- bps / R ---
    is_bps: Optional[float] = None
    is_bps_reason = "resolved"
    denom = None
    if decision_price is not None and quantity and contract_multiplier:
        denom = decision_price * abs(float(quantity)) * float(contract_multiplier)
    if is_usd is not None and denom:
        is_bps = round(is_usd / denom * 10000.0, 4)
    else:
        is_bps_reason = "no_is_usd" if is_usd is None else "no_decision_price_or_notional"

    is_r: Optional[float] = None
    sw = _f(stop_width_usd)
    if is_usd is not None and sw and sw > 0:
        is_r = round(is_usd / sw, 4)
        is_r_reason = "resolved"
    else:
        is_r_reason = "no_is_usd" if is_usd is None else "no_stop_width_usd"

    return {
        "schema_version": IS_SCHEMA_VERSION,
        "is_usd": is_usd,
        "is_bps": is_bps,
        "is_bps_reason": is_bps_reason,
        "is_r": is_r,
        "is_r_reason": is_r_reason,
        "slippage_usd": slippage_usd,
        "slippage_reason": slip_reasons,
        "fees_usd": fees_usd,
        "fees_reason": fees_reasons,
        "funding_usd": funding_usd,
        "funding_reason": funding_reason,
        "opportunity_cost_usd": opportunity_cost_usd,
        "opportunity_cost_reason": opportunity_cost_reason,
        "decision_price": decision_price,
        "decision_price_source": decision_price_source,
        "decision_price_reason": decision_price_reason,
        "cost_basis_status": cost_basis_status,
        "legs": [
            {
                "fill_id": l.fill_id, "role": ("open" if i == 0 else "close"),
                "found": l.found, "slippage_usd": l.slippage_usd,
                "fee_usd": l.fee_usd, "decision_price": l.decision_price,
            }
            for i, l in enumerate(legs)
        ],
    }


# --------------------------------------------------------------------------- #
# Convenience for the mint-time stamp (memoized per date+dir, mtime-invalidated)
# --------------------------------------------------------------------------- #

_INDEX_CACHE: Dict[Tuple[str, str], Tuple[float, Dict[str, FillCost]]] = {}


def _date_of(iso_ts: Any) -> Optional[str]:
    s = str(iso_ts or "")
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:4] + s[5:7] + s[8:10]
    return None


def _mtime_sig(date: str, fills_dir: Path) -> float:
    sig = 0.0
    for regex in (_FILLS_RE, _FEES_RE):
        for p in _dated_files(fills_dir, regex, date):
            try:
                sig = max(sig, p.stat().st_mtime)
            except OSError:
                pass
    return sig


def _cached_index(date: str, fills_dir: Path) -> Dict[str, FillCost]:
    key = (date, str(fills_dir))
    sig = _mtime_sig(date, fills_dir)
    hit = _INDEX_CACHE.get(key)
    if hit is not None and hit[0] == sig:
        return hit[1]
    idx = build_fill_cost_index(date, fills_dir=fills_dir)
    _INDEX_CACHE[key] = (sig, idx)
    return idx


def compute_is_for_lap(
    *,
    fill_ids: Iterable[str],
    entry_time_utc: Any,
    exit_time_utc: Any,
    quantity: float,
    contract_multiplier: float,
    broker: str = "paper_exec",
    stop_width_usd: Optional[float] = None,
    fills_dir: Optional[Path] = None,
    kraken_native: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Mint-time convenience: build (memoized) the fill-cost index for the
    lap's entry+exit dates and return the implementation_shortfall.v1 block.
    Returns None only on hard failure (caller treats as absent). A lap whose
    fills resolve nothing still returns a well-formed block with honest nulls
    (R1) — that is DIFFERENT from None."""
    try:
        src = Path(fills_dir) if fills_dir is not None else DEFAULT_FILLS_DIR
        dates = {d for d in (_date_of(entry_time_utc), _date_of(exit_time_utc)) if d}
        merged: Dict[str, FillCost] = {}
        for d in sorted(dates):
            merged.update(_cached_index(d, src))
        return compute_lap_is(
            fill_ids=fill_ids, quantity=quantity,
            contract_multiplier=contract_multiplier, broker=broker,
            stop_width_usd=stop_width_usd, index=merged,
            kraken_native=kraken_native,
        )
    except Exception as exc:  # noqa: BLE001 — observer-class, never break a mint
        LOG.warning("compute_is_for_lap failed (skipped): %s", exc)
        return None


__all__ = [
    "IS_SCHEMA_VERSION",
    "TCA_STAMP_ENV",
    "FillCost",
    "build_fee_index",
    "build_fill_cost_index",
    "compute_is_for_lap",
    "compute_lap_is",
    "tca_stamp_enabled",
]
