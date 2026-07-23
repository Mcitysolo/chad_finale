"""
Trade Closer — FIFO position matcher.

Reads paper fills from data/fills/FILLS_{date}.ndjson, matches opening
legs against closing legs FIFO per (strategy, symbol), computes realized
PnL using contract multipliers, and writes closed_trade.v1 records to
data/trades/trade_history_{date}.ndjson so downstream consumers (daily
report, SCR learner, profit lock) see real PnL.

State (open FIFO queues + already-processed fill_ids) is persisted to
runtime/trade_closer_state.json so partial days survive restarts.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import logging
import pathlib
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

_LOG = logging.getLogger("chad.execution.trade_closer")


# ---------------------------------------------------------------------------
# Contract multipliers (mirror chad/execution/ibkr_adapter.py FuturesContractSpec)
# ---------------------------------------------------------------------------

CONTRACT_MULTIPLIERS: Dict[str, float] = {
    # Micro futures
    "MES": 5.0,
    "MNQ": 2.0,
    "MCL": 100.0,
    "MGC": 10.0,
    # ETFs ($1 per share)
    "SPY": 1.0,
    "QQQ": 1.0,
    "GLD": 1.0,
    "TLT": 1.0,
    "IWM": 1.0,
    # Crypto (per coin)
    "BTC-USD": 1.0,
    "ETH-USD": 1.0,
    "SOL-USD": 1.0,
    # Inverse / vol ETFs
    "SVXY": 1.0,
    "UVXY": 1.0,
    "SH": 1.0,
    "PSQ": 1.0,
    # Larger futures
    "ZN": 1000.0,
    "ZB": 1000.0,
    "M6E": 12500.0,
    "SIL": 1000.0,
}
DEFAULT_MULTIPLIER = 1.0


def get_multiplier(symbol: str) -> float:
    if not symbol:
        return DEFAULT_MULTIPLIER
    return CONTRACT_MULTIPLIERS.get(symbol.upper(), DEFAULT_MULTIPLIER)


import math as _math  # local import keeps the module-level surface tidy


def _coerce_finite_positive(v: Any) -> Optional[float]:
    """Return float(v) iff it's a finite, strictly positive number.

    Anything that fails float() (str garbage, None, dicts), or that parses to
    NaN / +inf / -inf / 0 / negative, is rejected so the caller can fall back.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not _math.isfinite(f):
        return None
    if f <= 0.0:
        return None
    return f


def _fill_multiplier(fill: Dict[str, Any]) -> float:
    """Pick the multiplier to use when computing closed-trade PnL.

    Selection rule (first match wins):
      1. If the fill carries a finite-positive ``options_multiplier`` value
         (either at the top level — which is what _extract_fill promotes —
         or under ``extra.options_multiplier``), use it. This is the alpha
         options BAG case where the writer stamped 100x on the record.
      2. If asset_class == "options" AND extra.options_multiplier exists,
         use it (covers fills where the top-level promotion was bypassed
         but the structured ``extra`` is intact).
      3. Otherwise fall back to the per-symbol table via get_multiplier().

    Stocks / futures / crypto are untouched: they don't carry
    options_multiplier, so they hit step 3 unchanged.
    """
    if not isinstance(fill, dict):
        return DEFAULT_MULTIPLIER

    # Step 1: flat top-level options_multiplier (promoted by _extract_fill).
    direct = _coerce_finite_positive(fill.get("options_multiplier"))
    if direct is not None:
        return direct

    # Step 2: structured extra path. Defensive against missing/garbage extra.
    extra = fill.get("extra")
    if isinstance(extra, dict):
        nested = _coerce_finite_positive(extra.get("options_multiplier"))
        if nested is not None:
            asset_class = str(fill.get("asset_class") or "").strip().lower()
            extra_ac = str(extra.get("asset_class") or "").strip().lower()
            if asset_class == "options" or extra_ac == "options":
                return nested
            # No asset_class signal anywhere — still trust the explicit
            # options_multiplier value the writer placed in extra.
            return nested

    # Step 3: fall back to the per-symbol table.
    return get_multiplier(fill.get("symbol", ""))


# ---------------------------------------------------------------------------
# ClosedTrade dataclass
# ---------------------------------------------------------------------------

# B2 (FLIP-UNBLOCK 2026-07-17): the trust markers that matter live on the
# OPENING lot's meta. The Epoch-3 reconciler seeds adopted broker positions
# with a fabricated cost basis and stamps the lot
# ``meta={"pnl_untrusted": True, "scoring_excluded": True,
#         "provenance": "UNATTRIBUTED_EPOCH3_ACCUMULATION", ...}``.
# Those markers reached the closed_trade.v1 payload under ``payload["meta"]``
# only — but every downstream trust gate (SCR ``trade_stats_engine._is_untrusted``
# and Stage-2 ``trade_log_adapter.trust_exclusion``) reads ``payload["extra"]``
# and ``payload["tags"]`` and NEVER reads ``meta``. A seed-lot round-trip
# therefore scored as clean alpha: PnL measured against a cost basis that was
# invented by the rebuild rather than paid in the market.
# Mirroring the markers into ``extra``/``tags`` here — at the single point where
# a closed trade becomes a payload — makes the exclusion un-missable for every
# reader, present and future, without asking each one to learn about ``meta``.
_TRUST_EXCLUSION_META_KEYS: Tuple[str, ...] = ("pnl_untrusted", "scoring_excluded")

# Provenance carried alongside the markers so an excluded row can be explained
# without re-reading the FIFO state. Copied only when a marker is present.
_TRUST_PROVENANCE_META_KEYS: Tuple[str, ...] = (
    "provenance", "source", "seeded_from", "reconciled",
)


def _trust_extra_from_meta(meta: Any) -> Dict[str, Any]:
    """Return the ``extra`` trust-marker block implied by an opening lot's *meta*.

    Empty dict when *meta* carries no exclusion marker — so a clean trade's
    payload shape is byte-for-byte what it was before B2 (no ``extra`` key at
    all). Only a genuinely tainted lot grows the block.
    """
    if not isinstance(meta, dict):
        return {}
    markers = {k: True for k in _TRUST_EXCLUSION_META_KEYS if bool(meta.get(k))}
    if not markers:
        return {}
    out: Dict[str, Any] = dict(markers)
    for k in _TRUST_PROVENANCE_META_KEYS:
        v = meta.get(k)
        if v is not None:
            out[k] = v
    # A human/grep-readable cause. SCR's placeholder detector keys off
    # `pnl_untrusted_reason`, so keep the phrasing distinct from the $100
    # placeholder literal it hunts for.
    out.setdefault(
        "pnl_untrusted_reason",
        "seed_lot_fabricated_cost_basis (provenance={}; scoring_excluded={})".format(
            meta.get("provenance", "unknown"), bool(meta.get("scoring_excluded")),
        ),
    )
    return out


def _trust_tags_from_meta(meta: Any) -> List[str]:
    """Tag mirror of :func:`_trust_extra_from_meta` — tags are the other read path."""
    if not isinstance(meta, dict):
        return []
    return [k for k in _TRUST_EXCLUSION_META_KEYS if bool(meta.get(k))]


def _sanitize_meta(value: Any) -> Dict[str, Any]:
    """Gap-4 (v9.1 audit): return a JSON-safe shallow copy of *value*.

    Used everywhere meta crosses a serialization boundary (fill record →
    FIFO lot → closed_trade.v1 payload → trade_closer_state.json). Returns
    {} when *value* is None or not a dict. For dict input, every key/value
    is checked with `json.dumps` (no `default=` fallback so the check is
    real); a value that fails serialization is converted to ``str(v)``.
    A value that also fails ``str()`` is dropped with a WARNING log. Keys
    are coerced to ``str`` to keep the resulting dict round-trip safe.
    """
    if not isinstance(value, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in value.items():
        try:
            key_str = str(k)
        except Exception:  # noqa: BLE001 — defensive; str() basically never raises
            _LOG.warning("trade_closer_meta_drop reason=key_not_stringable")
            continue
        try:
            json.dumps(v)
            out[key_str] = v
            continue
        except (TypeError, ValueError):
            pass
        try:
            out[key_str] = str(v)
            _LOG.warning(
                "trade_closer_meta_sanitize key=%s converted_to_str source_type=%s",
                key_str, type(v).__name__,
            )
        except Exception:  # noqa: BLE001
            _LOG.warning(
                "trade_closer_meta_drop key=%s reason=value_unsanitizable source_type=%s",
                key_str, type(v).__name__,
            )
    return out


@dataclasses.dataclass
class ClosedTrade:
    strategy: str
    symbol: str
    side: str  # direction of the OPENING leg: BUY (long) or SELL (short)
    entry_price: float
    exit_price: float
    quantity: float
    entry_time_utc: str
    exit_time_utc: str
    pnl: float
    contract_multiplier: float
    fill_ids: List[str]
    schema: str = "closed_trade.v1"
    # Gap-4 (v9.1 audit): forwarded strategy meta block (setup_family,
    # stop_width_usd, session_window, r_target_1, r_target_2, ...). Empty
    # default keeps existing callers and legacy state files compatible.
    meta: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        # Local import: keeps trade_closer's import surface narrow and avoids
        # any analytics→execution import cycle if it ever surfaces.
        from chad.analytics.pnl_breakdown import build_pnl_breakdown

        notional = self.entry_price * self.quantity * self.contract_multiplier
        gross = float(self.pnl)
        # Per-fill cost enrichment is not yet wired into FIFO matching, so
        # commission/slippage/fees are *unknown* — not zero. The breakdown
        # carries cost_basis_status="unavailable" so downstream readers see
        # the uncertainty explicitly instead of inheriting a silent default.
        breakdown = build_pnl_breakdown(
            gross_price_pnl=gross,
            entry_price=self.entry_price,
            exit_price=self.exit_price,
            quantity=self.quantity,
            contract_multiplier=self.contract_multiplier,
            commission=None,
            fees=None,
            slippage=None,
            source="paper_exec",
            cost_basis_status="unavailable",
        )
        net = float(breakdown["net_pnl"])
        # DS07: pnl is deprecated alias for net_pnl; use net_pnl.
        # Top-level commission/slippage are kept as numeric for backwards
        # compatibility with old report tooling that pre-dates pnl_breakdown.
        # When costs are unknown we keep them at 0.0 here ONLY to preserve
        # the existing top-level shape — the canonical truth lives in
        # pnl_breakdown.cost_basis_status / pnl_breakdown.commission etc.
        commission_top = breakdown["commission"] if breakdown["commission"] is not None else 0.0
        slippage_top = breakdown["slippage"] if breakdown["slippage"] is not None else 0.0
        fees_top = breakdown["fees"]  # None preserved; legacy field already nullable
        # B2: mirror the opening lot's trust markers onto the two fields the
        # downstream gates actually read. Clean trades add neither key.
        sanitized_meta = _sanitize_meta(self.meta)
        trust_extra = _trust_extra_from_meta(sanitized_meta)
        tags = ["paper", "closed", self.strategy] + _trust_tags_from_meta(sanitized_meta)
        payload: Dict[str, Any] = {
            "schema_version": self.schema,
            "strategy": self.strategy,
            "symbol": self.symbol,
            "side": self.side,
            "pnl": round(float(self.pnl), 4),
            "gross_pnl": round(gross, 4),
            "commission": commission_top,
            "slippage": slippage_top,
            "fees": fees_top,
            "net_pnl": round(net, 4),
            "entry_time_utc": self.entry_time_utc,
            "exit_time_utc": self.exit_time_utc,
            "fill_price": self.exit_price,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "contract_multiplier": self.contract_multiplier,
            "notional": notional,
            "fill_ids": list(self.fill_ids),
            "broker": "paper_exec",
            "account_id": "PAPER_EXEC",
            "is_live": False,
            "tags": tags,
            "pnl_breakdown": breakdown,
            # Gap-4 (v9.1 audit): forwarded TradeSignal/position meta block.
            # setup_family_expectancy_updater reads payload["meta"]["setup_family"]
            # and payload["meta"]["stop_width_usd"] to bucket alpha_intraday_micro
            # trades by ORB / VWAP_RECLAIM / etc. Empty default for strategies
            # that emit no meta — schema_version is unchanged (no formal
            # closed_trade.v1 migration framework exists).
            "meta": sanitized_meta,
        }
        # Only a tainted lot grows an `extra` block — an untainted closed trade's
        # payload keeps the exact shape it had before B2.
        if trust_extra:
            payload["extra"] = trust_extra
        return payload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_side(s: Any) -> str:
    return str(s or "").strip().upper()


def _safe_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


# Statuses we consider trusted enough to feed FIFO matching. Anything else
# (pending, submitted, error, unknown, rejected, cancelled, ...) is skipped.
# INCIDENT-0723: "dry_run" was formerly blessed here — flatten-drill rehearsal
# rows then netted real lots, minted fake trade_history round-trips, and
# false-flatted the position guard (audits/INCIDENT_20260723_DRILL_EXHAUST_
# FALSE_FLAT.md). A dry_run order never traded; it must NEVER enter FIFO
# matching. Only genuine fill statuses are money truth.
_TRUSTED_FILL_STATUSES = frozenset({"filled", "paper_fill"})

# Keywords that — when found in a tag, source string, or extra marker —
# indicate the fill is a placeholder/fallback/synthetic rather than a real
# execution. Matched as case-insensitive substrings.
_PLACEHOLDER_MARKERS = (
    "placeholder",
    "fallback",
    "synthetic",
    "untrusted",
)

# Default fill_price the paper executor falls back to when no live market
# price is available. Real fills land on a live market price; a $100.00
# fill_price paired with a placeholder indicator is the canonical fingerprint
# of a phantom fill (see 2026-05-08 SPY delta incident).
_PLACEHOLDER_FILL_PRICE = 100.0


def _str_has_placeholder_marker(value: Any) -> bool:
    if value is None:
        return False
    s = str(value).strip().lower()
    if not s:
        return False
    return any(m in s for m in _PLACEHOLDER_MARKERS)


def _payload_indicates_placeholder(payload: Dict[str, Any]) -> bool:
    """True if tags / source / extra signal a placeholder or fallback fill."""
    tags = payload.get("tags")
    if isinstance(tags, (list, tuple)):
        for t in tags:
            if _str_has_placeholder_marker(t):
                return True
    if _str_has_placeholder_marker(payload.get("source")):
        return True
    extra = payload.get("extra")
    if isinstance(extra, dict):
        for k in (
            "placeholder",
            "is_placeholder",
            "fallback",
            "is_fallback",
            "synthetic",
            "is_synthetic",
        ):
            if bool(extra.get(k)):
                return True
        if _str_has_placeholder_marker(extra.get("source")):
            return True
        if _str_has_placeholder_marker(extra.get("pnl_untrusted_reason")):
            return True
    return False


def _extract_fill(
    obj: Dict[str, Any],
    *,
    quarantined_fill_ids: Optional[set] = None,
    quarantined_record_hashes: Optional[set] = None,
) -> Optional[Dict[str, Any]]:
    """Pull the inner payload of a FILLS_*.ndjson record into a flat dict.

    Skips records that match the sidecar quarantine (data/fills/quarantine_*.json)
    when *quarantined_fill_ids* / *quarantined_record_hashes* are supplied —
    callers in this module pass a set built once per cycle so per-line lookups
    stay O(1).
    """
    payload = obj.get("payload") if isinstance(obj, dict) else None
    if not isinstance(payload, dict):
        payload = obj if isinstance(obj, dict) else None
    if not isinstance(payload, dict):
        return None

    # Sidecar quarantine: drop pinned poisoned fills before any FIFO matching.
    if quarantined_record_hashes:
        rh = obj.get("record_hash") if isinstance(obj, dict) else None
        if isinstance(rh, str) and rh in quarantined_record_hashes:
            return None
    if quarantined_fill_ids:
        fid = payload.get("fill_id")
        if isinstance(fid, str) and fid in quarantined_fill_ids:
            return None

    if payload.get("reject") is True:
        return None
    status = str(payload.get("status", "")).strip().lower()
    if status in ("rejected", "cancelled", "canceled"):
        return None
    # Status allowlist: any non-empty status that is not blessed as trusted
    # (filled / paper_fill / dry_run) is a pending/error/unknown shape and
    # must not enter FIFO matching.
    if status and status not in _TRUSTED_FILL_STATUSES:
        return None

    # Source allowlist: a source string smelling of placeholder/fallback is
    # rejected outright regardless of fill_price.
    if _str_has_placeholder_marker(payload.get("source")):
        return None

    # Skip untrusted fills (e.g. placeholder fill_price flagged by the
    # paper executor's price-sanity guard). Untrusted fills must not feed
    # into FIFO matching — they would produce phantom realized PnL.
    extra = payload.get("extra") if isinstance(payload, dict) else None
    if isinstance(extra, dict) and bool(extra.get("pnl_untrusted")):
        return None
    if payload.get("pnl_untrusted") is True:
        return None
    tags_list = payload.get("tags") if isinstance(payload, dict) else None
    if isinstance(tags_list, (list, tuple)) and any(
        str(t).strip().lower() == "pnl_untrusted" for t in tags_list
    ):
        return None

    # Placeholder fill_price guard. The 2026-05-08 incident produced 22 fake
    # delta SELL trades where opening fills landed at fill_price=100.0 with
    # extra.expected_price=100.0 — the executor's "no live price" fallback
    # fired, but the upstream price-sanity check could not catch it because
    # price_cache had no SPY entry to compare against. Defense in depth: if
    # fill_price equals the placeholder default AND any placeholder/fallback/
    # synthetic indicator is present, OR the strategy's expected_price also
    # defaulted to 100.0, refuse the fill.
    px_for_guard = _safe_float(payload.get("fill_price"))
    if px_for_guard is not None and abs(px_for_guard - _PLACEHOLDER_FILL_PRICE) < 1e-9:
        if _payload_indicates_placeholder(payload):
            return None
        if isinstance(extra, dict):
            exp = _safe_float(extra.get("expected_price"))
            if exp is not None and abs(exp - _PLACEHOLDER_FILL_PRICE) < 1e-9:
                return None

    fill_id = payload.get("fill_id")
    if not fill_id:
        # Synthesize a deterministic id so we can de-dup
        seed = json.dumps(
            {
                "strategy": payload.get("strategy"),
                "symbol": payload.get("symbol"),
                "side": payload.get("side"),
                "qty": payload.get("quantity"),
                "px": payload.get("fill_price"),
                "ts": payload.get("fill_time_utc") or payload.get("entry_time_utc"),
            },
            sort_keys=True,
            default=str,
        )
        fill_id = "syn_" + hashlib.sha256(seed.encode()).hexdigest()[:24]

    side = _normalize_side(payload.get("side"))
    if side not in ("BUY", "SELL"):
        return None

    qty = _safe_float(payload.get("quantity"))
    px = _safe_float(payload.get("fill_price"))
    if qty is None or px is None or qty <= 0:
        return None

    strategy = str(payload.get("strategy", "unknown")).strip().lower() or "unknown"
    symbol = str(payload.get("symbol", "")).strip().upper()
    if not symbol:
        return None

    ts = (
        payload.get("fill_time_utc")
        or payload.get("entry_time_utc")
        or obj.get("timestamp_utc")
    )

    # Promote options_multiplier and asset_class onto the flat fill dict so
    # _match_fill / _fill_multiplier can apply the per-contract multiplier
    # without re-deriving it from the symbol table. Stocks / futures / crypto
    # records do not carry options_multiplier, so this is a no-op for them.
    options_multiplier: Optional[float] = None
    if isinstance(extra, dict):
        options_multiplier = _coerce_finite_positive(extra.get("options_multiplier"))
    asset_class_top = str(payload.get("asset_class") or "").strip().lower()
    asset_class_extra = ""
    if isinstance(extra, dict):
        asset_class_extra = str(extra.get("asset_class") or "").strip().lower()
    asset_class = asset_class_top or asset_class_extra

    # Gap-4 (v9.1 audit): pull meta from the fill payload so opening lots
    # can stamp setup_family/stop_width_usd before the closing fill arrives.
    # Source precedence (first JSON-safe dict wins):
    #   1. payload["meta"]
    #   2. extra["meta"]
    # Falls back to {} (and trade_closer then consults position_guard at
    # lot-creation time inside _match_fill so writer-side meta forwarding
    # is optional, not required, for the closed_trade.v1 meta block to
    # populate in production).
    fill_meta: Dict[str, Any] = {}
    raw_meta = payload.get("meta")
    if isinstance(raw_meta, dict) and raw_meta:
        fill_meta = _sanitize_meta(raw_meta)
    elif isinstance(extra, dict):
        raw_extra_meta = extra.get("meta")
        if isinstance(raw_extra_meta, dict) and raw_extra_meta:
            fill_meta = _sanitize_meta(raw_extra_meta)

    flat: Dict[str, Any] = {
        "fill_id": str(fill_id),
        "strategy": strategy,
        "symbol": symbol,
        "side": side,
        "quantity": float(qty),
        "fill_price": float(px),
        "ts_utc": str(ts) if ts else None,
        "meta": fill_meta,
    }
    if options_multiplier is not None:
        flat["options_multiplier"] = options_multiplier
    if asset_class:
        flat["asset_class"] = asset_class
    return flat


# ---------------------------------------------------------------------------
# TradeCloser
# ---------------------------------------------------------------------------

EXCLUDED_FROM_ROUTING = {"broker_sync", "manual", "paper_exec", "unknown", ""}


# PFF1 (2026-07-20): the paper_trade_executor emits ONE aggregate SIM fill per
# order (account PAPER_EXEC, source paper_trade_executor). The
# ibkr_paper_fill_harvester ADDITIVELY mirrors that SAME order as the real IBKR
# paper broker's slice fills (account DUR119533, source
# ibkr_paper_fill_harvester, tag ibkr_harvest) — see the harvester docstring:
# "additive — it does not replace the existing paper trade executor". Both land
# in the same FILLS_*.ndjson, so feeding BOTH into FIFO double-books the order.
# On 2026-07-20 the first ACTIVE exit-overlay close (one 273-share SELL) surfaced
# it: the executor aggregate SELL 273 closed the Epoch-3 seed lot (the untrusted
# +625.17 close), while the harvester's 273-in-slices — because the aggregate had
# already flattened the queue — opened phantom SHORT lots that round-tripped the
# re-buys (BUY 5 + BUY 223), fabricating 6 "trusted" closes over ~233 shares and
# netting the gamma queue to zero against a real +228 broker position. This is
# the same double-count behind the ~1.9x guard drift seen since the harvester was
# enabled (~2026-07-08).
#
# Dedup rule (symbol-scoped): a harvester fill is a redundant broker MIRROR for
# any symbol the executor has already booked this day, and is dropped before
# FIFO. Symbol scope (not (strategy,symbol)) is deliberate — the harvester
# re-attributes some mirror slices to "broker_sync" (its resolve_strategy path),
# so keying on strategy would let those broker_sync mirrors survive as a phantom
# lot. A harvester fill for a symbol with NO executor fill is a genuine ORPHAN
# (a broker-side position CHAD never executed through its own pipeline) and is
# preserved so the harvester keeps feeding FIFO for executor-less fills
# (Bug B Fix B, 2026-06-03) and broker-truth adoption is untouched.
_HARVESTER_SOURCES: frozenset = frozenset({"ibkr_paper_fill_harvester"})
_HARVESTER_TAG = "ibkr_harvest"


def _payload_symbol(payload: Dict[str, Any]) -> str:
    """Normalized symbol (upper-cased) for a raw fill payload; '' when absent.

    Mirrors the symbol normalization _extract_fill applies so the mirror-dedup
    set below keys on the same value FIFO uses.
    """
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("symbol", "")).strip().upper()


def _payload_is_harvester(payload: Dict[str, Any]) -> bool:
    """True when the fill was written by the IBKR paper-fill harvester (a broker
    mirror). Either carrier is sufficient: source or the ibkr_harvest tag."""
    if not isinstance(payload, dict):
        return False
    if str(payload.get("source", "")).strip().lower() in _HARVESTER_SOURCES:
        return True
    tags = payload.get("tags")
    if isinstance(tags, (list, tuple)):
        return any(str(t).strip().lower() == _HARVESTER_TAG for t in tags)
    return False


def _payload_is_executor(payload: Dict[str, Any]) -> bool:
    """True when the fill was written by the paper trade executor — the canonical
    single-record-per-order SIM aggregate that owns FIFO."""
    if not isinstance(payload, dict):
        return False
    return str(payload.get("source", "")).strip().lower() == "paper_trade_executor"


class TradeCloser:
    def __init__(
        self,
        fills_dir: pathlib.Path,
        trades_dir: pathlib.Path,
        state_path: pathlib.Path,
        routing_path: Optional[pathlib.Path] = None,
        position_guard_path: Optional[pathlib.Path] = None,
    ) -> None:
        self.fills_dir = pathlib.Path(fills_dir)
        self.trades_dir = pathlib.Path(trades_dir)
        self.state_path = pathlib.Path(state_path)
        self._routing_path = pathlib.Path(routing_path) if routing_path is not None else None
        # Gap-4 (v9.1 audit): optional override of the position_guard.json
        # path used as a fallback source of TradeSignal.meta when the fill
        # record does not carry meta itself. Production default is None →
        # use the canonical runtime path. Tests inject a tmp_path file.
        self._position_guard_path: Optional[pathlib.Path] = (
            pathlib.Path(position_guard_path) if position_guard_path is not None else None
        )
        # queues[(strategy, symbol)] = deque of open lots
        # each lot: {fill_id, side, quantity, fill_price, ts_utc, multiplier, meta}
        self.queues: Dict[Tuple[str, str], Deque[Dict[str, Any]]] = defaultdict(deque)
        self.processed_fill_ids: set = set()

    def _resolve_lot_meta(
        self,
        fill: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Gap-4 (v9.1 audit): pick the meta dict to stamp on a newly
        created opening lot.

        Precedence:
          1. Meta carried on the fill record (set by _extract_fill from
             payload["meta"] or extra["meta"]).
          2. position_guard.json entry for (strategy, symbol) — looked up
             at lot creation time. Production source for alpha_intraday_micro
             until/unless paper_exec_evidence_writer learns to copy meta
             onto the fill record directly.
        Returns {} when both sources are absent / unparseable. Never raises.
        """
        fill_meta = fill.get("meta") if isinstance(fill, dict) else None
        if isinstance(fill_meta, dict) and fill_meta:
            return _sanitize_meta(fill_meta)
        try:
            from chad.core.position_guard import get_open_position_meta
            guard_meta = get_open_position_meta(
                str(fill.get("strategy", "")),
                str(fill.get("symbol", "")),
                path=self._position_guard_path,
            )
        except Exception:  # noqa: BLE001 — position_guard lookup must never break FIFO
            guard_meta = {}
        return _sanitize_meta(guard_meta) if isinstance(guard_meta, dict) else {}

    # ---- state ----------------------------------------------------------

    def load_state(self) -> None:
        # Always seed processed_fill_ids from any existing trade_history files
        # first. If the state file is missing or corrupted, this still prevents
        # a fill_id already consumed in a prior closed_trade from being matched
        # again into a new closed_trade.
        self.seed_processed_from_trade_history()

        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, dict):
            return

        # Union with what we already seeded from history above.
        self.processed_fill_ids |= set(data.get("processed_fill_ids", []) or [])
        self.queues = defaultdict(deque)
        for entry in data.get("queues", []) or []:
            try:
                key = (str(entry["strategy"]), str(entry["symbol"]))
                lots = entry.get("lots", []) or []
                for lot in lots:
                    normalized = dict(lot)
                    if not normalized.get("fill_id"):
                        # Synthesize deterministic id for lots lacking one
                        # (e.g. broker_sync anchor lots rebuilt from a
                        # position snapshot that has no exec_id). Mirrors
                        # _extract_fill's synthesis at line 138-153.
                        seed = json.dumps(
                            {
                                "strategy": key[0],
                                "symbol": key[1],
                                "side": normalized.get("side"),
                                "qty": normalized.get("quantity"),
                                "px": normalized.get("fill_price"),
                                "ts": normalized.get("lot_ts_utc")
                                or normalized.get("ts_utc"),
                            },
                            sort_keys=True,
                            default=str,
                        )
                        normalized["fill_id"] = (
                            "syn_"
                            + hashlib.sha256(seed.encode()).hexdigest()[:24]
                        )
                    self.queues[key].append(normalized)
            except Exception:
                continue

    def save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "processed_fill_ids": sorted(self.processed_fill_ids),
            "queues": [
                {"strategy": k[0], "symbol": k[1], "lots": list(v)}
                for k, v in self.queues.items()
                if v
            ],
            "saved_at_utc": _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None).isoformat() + "Z",
        }
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.state_path)

    # ---- duplicate-fill protection --------------------------------------

    def seed_processed_from_trade_history(self) -> int:
        """
        Seed processed_fill_ids from every fill_id already recorded in any
        existing trade_history_*.ndjson under self.trades_dir.

        This is a defense-in-depth check: if the persisted state file is lost
        or rebuilt, a fill_id that was already consumed in a prior closed_trade
        record must not generate another closed_trade on a re-run.

        Returns the number of fill_ids newly added to the processed set.
        """
        added = 0
        if not self.trades_dir.exists():
            return 0
        try:
            history_files = sorted(self.trades_dir.glob("trade_history_*.ndjson"))
        except Exception:
            return 0
        for hpath in history_files:
            try:
                text = hpath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for raw in text.splitlines():
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = obj.get("payload") if isinstance(obj, dict) else None
                if not isinstance(payload, dict):
                    continue
                if payload.get("schema_version") != "closed_trade.v1":
                    continue
                # Bug B Fix B (consumer defense, 2026-06-03): ignore harvester
                # phantom round-trips. The paper-fill harvester historically
                # wrote a closed_trade.v1 record per OPEN fill (pnl=0, single
                # fill_id, tag ibkr_harvest); seeding those fill_ids marked
                # opens as already-consumed so they never entered the FIFO —
                # the futures-runaway feedback gap. Legitimate FIFO closes
                # always carry [open_id, close_id] (2 ids) and never the tag
                # (validated: 2,432 records, zero crossover on either test).
                tags_field = payload.get("tags")
                if isinstance(tags_field, (list, tuple)) and any(
                    str(t).strip().lower() == "ibkr_harvest" for t in tags_field
                ):
                    continue
                fids = payload.get("fill_ids")
                if not isinstance(fids, (list, tuple)):
                    continue
                if len(fids) < 2:
                    continue
                for fid in fids:
                    if not fid:
                        continue
                    fid_str = str(fid)
                    if fid_str not in self.processed_fill_ids:
                        self.processed_fill_ids.add(fid_str)
                        added += 1
        return added

    # ---- core processing ------------------------------------------------

    def _match_fill(self, fill: Dict[str, Any]) -> List[ClosedTrade]:
        """Apply a single fill to its FIFO queue and emit any closed trades."""
        key = (fill["strategy"], fill["symbol"])
        queue = self.queues[key]
        # The OPEN-side multiplier is what matters for PnL — a BAG opened at
        # options_multiplier=100 must close at 100 regardless of whether the
        # closing fill's metadata happens to drop the field. We therefore
        # stamp the multiplier on the opening lot below, and prefer the
        # lot's stored multiplier on close. The incoming fill's own
        # multiplier acts as a fallback for legacy lots restored from a
        # state file written before this field existed.
        incoming_multiplier = _fill_multiplier(fill)
        # Gap-4: resolve meta once per incoming fill so a single position
        # guard read backs both the same-side append and the flip-residual
        # opening lot paths below. Closing legs do not consume this dict.
        incoming_meta = self._resolve_lot_meta(fill)
        closed: List[ClosedTrade] = []

        remaining = fill["quantity"]
        incoming_side = fill["side"]

        # Empty queue OR same-side as the queue head -> opening / adding
        if not queue or queue[0]["side"] == incoming_side:
            queue.append(
                {
                    "fill_id": fill["fill_id"],
                    "side": incoming_side,
                    "quantity": remaining,
                    "fill_price": fill["fill_price"],
                    "ts_utc": fill["ts_utc"],
                    "multiplier": incoming_multiplier,
                    "meta": incoming_meta,
                }
            )
            return closed

        # Opposite side -> close FIFO against head
        while remaining > 0 and queue and queue[0]["side"] != incoming_side:
            lot = queue[0]
            close_qty = min(lot["quantity"], remaining)

            opening_side = lot["side"]
            entry_price = lot["fill_price"]
            exit_price = fill["fill_price"]
            # Prefer the multiplier stamped on the open-side lot; only fall
            # back to the incoming fill's multiplier when the lot lacks one
            # (legacy state files persisted before lots carried multiplier).
            lot_multiplier = _coerce_finite_positive(lot.get("multiplier"))
            multiplier = lot_multiplier if lot_multiplier is not None else incoming_multiplier
            direction = 1.0 if opening_side == "BUY" else -1.0
            pnl = (exit_price - entry_price) * close_qty * multiplier * direction

            # Gap-4: forward the opening lot's meta onto the ClosedTrade so
            # the closed_trade.v1 payload preserves setup_family / stop_width
            # back to the analytics layer. Legacy lots persisted by older
            # state files lack `meta` — they fall through to {}.
            lot_meta = lot.get("meta")
            ct_meta = _sanitize_meta(lot_meta) if isinstance(lot_meta, dict) else {}

            closed.append(
                ClosedTrade(
                    strategy=fill["strategy"],
                    symbol=fill["symbol"],
                    side=opening_side,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=close_qty,
                    entry_time_utc=str(lot.get("ts_utc") or ""),
                    exit_time_utc=str(fill.get("ts_utc") or ""),
                    pnl=pnl,
                    contract_multiplier=multiplier,
                    fill_ids=[lot["fill_id"], fill["fill_id"]],
                    meta=ct_meta,
                )
            )

            lot["quantity"] -= close_qty
            remaining -= close_qty
            if lot["quantity"] <= 1e-12:
                queue.popleft()

        # Any leftover incoming qty becomes a new opening lot in the
        # opposite direction (flip). Carry incoming_meta so the flip-side
        # lot is also tagged with whatever meta the closing fill provided
        # (or the current position_guard entry for the new side).
        if remaining > 0:
            queue.append(
                {
                    "fill_id": fill["fill_id"],
                    "side": incoming_side,
                    "quantity": remaining,
                    "fill_price": fill["fill_price"],
                    "ts_utc": fill["ts_utc"],
                    "multiplier": incoming_multiplier,
                    "meta": incoming_meta,
                }
            )

        return closed

    def process_fills(self, date_str: str) -> List[ClosedTrade]:
        path = self.fills_dir / f"FILLS_{date_str}.ndjson"
        if not path.exists():
            return []

        # Load sidecar quarantine sets once per call. Failure is non-fatal —
        # the existing placeholder/untrusted checks in _extract_fill remain
        # the primary defense; the sidecar is forensic pinning of known
        # historical pollution.
        quarantined_fill_ids: set = set()
        quarantined_record_hashes: set = set()
        try:
            from chad.analytics.quarantine import get_sidecar_exclusion_sets
            quarantined_fill_ids, quarantined_record_hashes = (
                get_sidecar_exclusion_sets(
                    fills_dir=self.fills_dir,
                    trades_dir=self.trades_dir,
                )
            )
        except Exception:
            quarantined_fill_ids, quarantined_record_hashes = set(), set()

        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

        # PFF1: pre-scan for the symbols the paper_trade_executor has already
        # booked in this day's file. A harvester fill for such a symbol is a
        # redundant broker mirror of the executor's aggregate (the executor is
        # always written first — at submit time — so whenever a harvester slice
        # is present its executor counterpart is too); dropping it below stops
        # the aggregate+slices double-book. Recomputed every call so incremental
        # cycles stay consistent; symbols with no executor fill keep their
        # harvester fills (orphans) and match normally.
        executor_symbols: set = set()
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = obj.get("payload") if isinstance(obj, dict) else None
            if _payload_is_executor(payload):
                sym = _payload_symbol(payload)
                if sym:
                    executor_symbols.add(sym)

        closed_all: List[ClosedTrade] = []
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # PFF1: skip harvester mirrors of executor-booked symbols before FIFO
            # so one order booked from two sources cannot double-count.
            payload = obj.get("payload") if isinstance(obj, dict) else None
            if (
                _payload_is_harvester(payload)
                and _payload_symbol(payload) in executor_symbols
            ):
                continue
            fill = _extract_fill(
                obj,
                quarantined_fill_ids=quarantined_fill_ids,
                quarantined_record_hashes=quarantined_record_hashes,
            )
            if fill is None:
                continue
            if fill["fill_id"] in self.processed_fill_ids:
                continue
            closed_all.extend(self._match_fill(fill))
            self.processed_fill_ids.add(fill["fill_id"])

        return closed_all

    # ---- output ---------------------------------------------------------

    def write_trade_history(
        self, closed_trades: List[ClosedTrade], date_str: str
    ) -> int:
        if not closed_trades:
            return 0
        self.trades_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.trades_dir / f"trade_history_{date_str}.ndjson"

        # Determine starting sequence_id and prev_hash by reading existing tail
        prev_hash = "GENESIS"
        seq = 0
        if out_path.exists():
            try:
                lines = out_path.read_text(
                    encoding="utf-8", errors="ignore"
                ).splitlines()
                for ln in reversed(lines):
                    s = ln.strip()
                    if not s:
                        continue
                    try:
                        last = json.loads(s)
                    except json.JSONDecodeError:
                        continue
                    prev_hash = str(last.get("record_hash") or prev_hash)
                    try:
                        seq = int(last.get("sequence_id") or 0)
                    except Exception:
                        seq = 0
                    break
            except Exception:
                pass

        n = 0
        with out_path.open("a", encoding="utf-8") as fh:
            for ct in closed_trades:
                seq += 1
                payload = ct.to_payload()
                ts = _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None).isoformat() + "Z"
                core = {
                    "payload": payload,
                    "prev_hash": prev_hash,
                    "sequence_id": seq,
                    "timestamp_utc": ts,
                }
                rec_hash = hashlib.sha256(
                    json.dumps(core, sort_keys=True, default=str).encode()
                ).hexdigest()
                core["record_hash"] = rec_hash
                fh.write(json.dumps(core, default=str) + "\n")
                prev_hash = rec_hash
                n += 1
        return n

    # ---- entry points ---------------------------------------------------

    def _route_profits(self, closed: List[ClosedTrade]) -> int:
        """
        Advisory 50/30/20 profit routing for each profitable close.

        Writes to runtime/profit_routing.json as an accounting ledger —
        does NOT transfer capital (single-account paper lane). Failures
        are swallowed: this is ledger sugar on top of the hot path.
        """
        if not closed:
            return 0
        try:
            from chad.risk.profit_router import ProfitRouter  # local to avoid import cycles
            router = ProfitRouter(routing_path=self._routing_path)
        except Exception:  # noqa: BLE001 — router must never break trade_closer
            return 0
        routed = 0
        for ct in closed:
            try:
                if str(ct.strategy or "").lower() in EXCLUDED_FROM_ROUTING:
                    continue  # skip non-strategy bookkeeping labels
                if ct.pnl > 0:
                    router.route_profit(
                        realized_pnl=float(ct.pnl),
                        closing_strategy=str(ct.strategy),
                    )
                    routed += 1
            except Exception:  # noqa: BLE001
                continue
        return routed

    def run_once(self, date_str: Optional[str] = None) -> Dict[str, Any]:
        if date_str is None:
            date_str = _dt.date.today().strftime("%Y%m%d")
        self.load_state()
        closed = self.process_fills(date_str)
        written = self.write_trade_history(closed, date_str)
        routed = self._route_profits(closed)
        self.save_state()
        return {
            "date": date_str,
            "closed_count": len(closed),
            "written": written,
            "total_pnl": sum(c.pnl for c in closed),
            "profits_routed": routed,
        }

    def get_summary(self, date_str: str) -> Dict[str, Any]:
        path = self.trades_dir / f"trade_history_{date_str}.ndjson"
        total_closed = 0
        total_pnl = 0.0
        by_strategy: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"count": 0, "pnl": 0.0}
        )
        if not path.exists():
            return {
                "date": date_str,
                "total_closed": 0,
                "total_pnl": 0.0,
                "by_strategy": {},
            }
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = raw.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            payload = obj.get("payload") if isinstance(obj, dict) else None
            if not isinstance(payload, dict):
                continue
            if payload.get("schema_version") != "closed_trade.v1":
                continue
            try:
                pnl = float(payload.get("pnl", 0.0))
            except (TypeError, ValueError):
                continue
            strat = str(payload.get("strategy", "unknown"))
            total_closed += 1
            total_pnl += pnl
            by_strategy[strat]["count"] += 1
            by_strategy[strat]["pnl"] += pnl
        return {
            "date": date_str,
            "total_closed": total_closed,
            "total_pnl": total_pnl,
            "by_strategy": {k: dict(v) for k, v in by_strategy.items()},
        }


# ---------------------------------------------------------------------------
# CLI entry point (used by chad-trade-closer.service)
# ---------------------------------------------------------------------------

def _default_paths() -> Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    root = pathlib.Path(__file__).resolve().parents[2]
    return (
        root / "data" / "fills",
        root / "data" / "trades",
        root / "runtime" / "trade_closer_state.json",
    )


def main() -> int:
    fills_dir, trades_dir, state_path = _default_paths()
    closer = TradeCloser(
        fills_dir=fills_dir, trades_dir=trades_dir, state_path=state_path
    )
    result = closer.run_once()
    print(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
