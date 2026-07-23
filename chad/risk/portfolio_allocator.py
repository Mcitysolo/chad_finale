"""
chad/risk/portfolio_allocator.py — W5B R1 portfolio allocator, exposure spine.

THE NAMED GAP THIS CLOSES. Nothing in CHAD computes *marginal portfolio risk*.
Per-order caps exist and enforce (policy.py max_total_exposure $500k /
max_symbol_exposure $150k; sizing_config 5% position). A margin-based aggregate
gross exists elsewhere in shadow. But no layer normalizes **every intent plus
every open position into one exposure vector and sums the correlated tickets**.

The live book is the demonstration: eleven tickets, gross == net == every
position long, with two positions (LLY, SPY) already above the ENFORCED $150k
per-symbol cap — because that cap is per-ORDER, and a position accumulated over
several orders never meets it. This module is the layer that can see that.

SCOPE — this file is the exposure spine only (W5B-1): it normalizes a book and
an intent into `ExposureVector`s and accumulates them into a `ProvisionalBook`.
It holds no limits and renders no verdict; that is W5B-2 (`allocator_limits`).
The per-intent observer that calls both is W5B-3.

NON-NEGOTIABLES (PLAN_W5B §0):
  - **It reads; it never re-prices.** positions_truth.json, price_cache.json and
    config/symbol_sectors.json are consumed read-only. No P&L is computed, no
    book is mutated, no intent.quantity is touched.
  - **It never evaluates a close.** Enforced by the caller's placement and by
    `fuse_gate.is_exit_like`; nothing here is on a close path.
  - **No invented numbers.** Every multiplier below cites the table it came from.

CURRENCY (PLAN_W5B §12.2). Exposure is denominated in the instrument's own
currency, which for the entire live book and every price in price_cache is USD.
The only `currency_ok` account equity is CAD and NO FX rate exists anywhere
(portfolio_snapshot: usd_ok=false, usdcad_rate_used=null). So a "gross / equity"
ratio would be a mixed-unit number, wrong by roughly the USDCAD spread. This
module therefore reports **USD notional** and never divides by the CAD equity;
`currency` rides on every vector so a non-USD instrument is visible rather than
silently summed. See `ProvisionalBook.currency_mix`.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

POSITIONS_TRUTH_PATH = REPO_ROOT / "runtime" / "positions_truth.json"
PRICE_CACHE_PATH = REPO_ROOT / "runtime" / "price_cache.json"

UNMAPPED_SECTOR = "unmapped"

# Venues. Only two lanes are wired (PLAN_W5B P7): crypto routes to Kraken,
# everything else to IBKR. There is no venue enum in the codebase to import.
VENUE_IBKR = "IBKR"
VENUE_KRAKEN = "KRAKEN"

# --------------------------------------------------------------------------- #
# D2: allocator-local futures multiplier reference
# --------------------------------------------------------------------------- #
# Point values consolidated READ-ONLY from the three existing, unreconciled
# tables. This does NOT rewrite them — unification is a named follow-up
# (PLAN_W5B §8.3). Where the tables overlap they agree, which is why a local
# read-only copy is safe; the parity is pinned by a test.
#
#   chad/strategies/alpha_futures.py:94-103   DEFAULT_SPECS  (MES MNQ MCL MGC MYM M2K ZN ZB)
#   chad/strategies/omega_macro.py:76-101     OMEGA_MACRO_SPECS (ZN ZB M6E)
#   chad/risk/futures_position_sizer.py:10-15 FUTURES_SPECS  (MES MNQ MCL MGC)
#
# Nine micro roots. There is NO full-size ES/NQ/CL/GC and no M6A/M6B point
# value anywhere in the repo — an intent on one of those yields delta_usd=None
# and a loud reason, never a silent zero.
FUTURES_POINT_VALUES: Dict[str, float] = {
    "MES": 5.0,
    "MNQ": 2.0,
    "MCL": 100.0,
    "MGC": 10.0,
    "MYM": 0.5,
    "M2K": 5.0,
    "ZN": 1000.0,
    "ZB": 1000.0,
    "M6E": 12500.0,
}

# Reason codes. Every null carries one — never a bare None (the W5A rider-R1
# honest-nulls idiom, docs/CONTRACT_W5A_harness_handoff.md §2).
REASON_NO_PRICE = "no_price_available"
REASON_UNMAPPED_FUTURES_ROOT = "unmapped_futures_root"
REASON_NO_QUANTITY = "no_quantity"

# Price provenance, in the order the ladder tries them.
PRICE_SOURCE_LIMIT = "intent_limit_price"
PRICE_SOURCE_EXPECTED = "intent_expected_price"
PRICE_SOURCE_CACHE = "price_cache"
PRICE_SOURCE_AVG_COST = "position_avg_cost"


# --------------------------------------------------------------------------- #
# Asset class / venue derivation
# --------------------------------------------------------------------------- #
# PLAN_W5B P7 caveat A: chad/risk/position_exit_overlay._asset_class has no
# crypto/options branch and omits MYM from its futures set, so MYM misclassifies
# as equity there. The allocator deliberately does NOT inherit that classifier —
# it reads sec_type/secType directly. That upstream defect is a named follow-up,
# not something this module reproduces.

_ETF_HINTS = frozenset(
    {"SPY", "QQQ", "IWM", "IEMG", "VWO", "SH", "PSQ", "SVXY", "UVXY", "VIXY",
     "VXX", "TLT", "GLD", "SIL"}
)


def classify_asset_class(sec_type: Any, symbol: Any = "") -> str:
    """Asset class from the IB security type, which both the intent
    (`sec_type`) and positions_truth (`secType`) carry natively.

    ETF vs EQUITY is a labelling nicety only — both are `mult=1` and both bind
    identically at every limit. The hint set exists so evidence rows read
    honestly, not because anything downstream branches on it.
    """
    st = str(sec_type or "").strip().upper()
    sym = str(symbol or "").strip().upper()
    if st in ("FUT", "FUTURES", "CONTFUT"):
        return "futures"
    if st in ("OPT", "FOP", "OPTIONS"):
        return "options"
    if st in ("CRYPTO", "CRYPTOCURRENCY"):
        return "crypto"
    if st in ("CASH", "FX", "FOREX"):
        return "forex"
    if st in ("STK", "STOCK", "EQUITY"):
        return "etf" if sym in _ETF_HINTS else "equity"
    if not st:
        # Kraken intents carry a pair, not a sec_type.
        if "-" in sym or sym.endswith("USD"):
            return "crypto"
    return "unknown"


def derive_venue(asset_class: str) -> str:
    """Crypto ⇒ Kraken, everything else ⇒ IBKR (P7). Two lanes are wired."""
    return VENUE_KRAKEN if asset_class == "crypto" else VENUE_IBKR


def futures_root(symbol: Any) -> str:
    """Root of a futures symbol. Contract-month suffixes (MESZ5, MES-20261218)
    are stripped so a dated contract maps to its point value."""
    sym = str(symbol or "").strip().upper()
    if not sym:
        return ""
    if sym in FUTURES_POINT_VALUES:
        return sym
    for root in sorted(FUTURES_POINT_VALUES, key=len, reverse=True):
        if sym.startswith(root):
            return root
    return sym


def multiplier_for(asset_class: str, symbol: Any) -> Tuple[Optional[float], Optional[str]]:
    """(multiplier, reason_if_null). Equity/ETF/crypto/forex are implicitly 1.

    An unmapped futures root returns (None, 'unmapped_futures_root') — D2's
    binding rule. A futures ticket whose point value we do not know must never
    contribute a silent 0 to gross; it contributes nothing and says so loudly.
    """
    if asset_class == "futures":
        root = futures_root(symbol)
        pv = FUTURES_POINT_VALUES.get(root)
        if pv is None:
            return None, REASON_UNMAPPED_FUTURES_ROOT
        return pv, None
    if asset_class == "options":
        # Options proxy-route through the underlying today (P5); no real OPT
        # contract is ever built, so no 100x contract multiplier applies. Any
        # options exposure in the book is already the equity-proxy delta.
        return 1.0, None
    return 1.0, None


# --------------------------------------------------------------------------- #
# Read-only loaders
# --------------------------------------------------------------------------- #

def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001 — a missing/corrupt read is not fatal
        return None


def load_price_cache(path: Optional[Path] = None) -> Dict[str, float]:
    """symbol → price from runtime/price_cache.json. Missing/corrupt ⇒ {} —
    every vector then reports price_source=None with REASON_NO_PRICE rather
    than pretending a price existed."""
    obj = _read_json(Path(path) if path is not None else PRICE_CACHE_PATH)
    prices = (obj or {}).get("prices")
    if not isinstance(prices, Mapping):
        return {}
    out: Dict[str, float] = {}
    for k, v in prices.items():
        try:
            out[str(k).strip().upper()] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def load_book_positions(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Open positions from runtime/positions_truth.json (schema
    positions_truth.v1) — the broker-truth book. Zero-quantity rows are dropped;
    a flat position is not exposure."""
    obj = _read_json(Path(path) if path is not None else POSITIONS_TRUTH_PATH)
    rows = (obj or {}).get("positions")
    if not isinstance(rows, list):
        return []
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, Mapping):
            continue
        try:
            qty = float(r.get("position", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if qty == 0.0:
            continue
        out.append(dict(r))
    return out


def load_sector_lookup(path: Optional[Path] = None):
    """Reuse the W4A sector map wholesale (P6) — same file, same inversion, same
    never-trippable `unmapped` fallback. Importing fuse_box's loader rather than
    re-reading the JSON means LC3 and the allocator can never disagree about
    which sector a symbol is in."""
    try:
        from chad.risk.fuse_box import load_sector_map, make_sector_lookup

        return make_sector_lookup(load_sector_map(path))
    except Exception as exc:  # noqa: BLE001 — fail to the unmapped bucket
        LOG.warning("ALLOCATOR_SECTOR_MAP_UNREADABLE err=%s", exc)
        return lambda _sym: UNMAPPED_SECTOR


# --------------------------------------------------------------------------- #
# The exposure vector
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class ExposureVector:
    """One normalized exposure ticket — an open position or a pending intent.

    `delta_usd` is the load-bearing dimension: signed(qty) · price · multiplier.
    Signed means a short is negative, so `net` can cancel and `gross` cannot.

    `beta_weighted_usd` ships at beta=1.0 (D1(a)). No statistical beta-to-SPY
    exists anywhere in the repo (P4: the only numeric beta is an unused
    FMPCompanyProfile field that is never stored). The dimension is present so
    the vector SHAPE is stable for R3, and `beta_source="default_1.0"` says
    plainly that nothing is weighted today. A fabricated beta map would be worse
    than an honest 1.0.
    """

    symbol: str
    side: str                       # BUY | SELL | LONG | SHORT
    strategy: str
    origin: str                     # "position" | "intent"
    quantity: float                 # signed
    delta_usd: Optional[float]
    beta: float
    beta_source: str
    beta_weighted_usd: Optional[float]
    sector: str
    asset_class: str
    venue: str
    currency: str
    multiplier: Optional[float]
    price: Optional[float]
    price_source: Optional[str]
    null_reason: Optional[str] = None

    @property
    def computable(self) -> bool:
        return self.delta_usd is not None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _signed_quantity(qty: Any, side: Any) -> float:
    """Signed quantity. positions_truth already signs (a short is negative);
    an intent carries an unsigned quantity plus a side, so SELL flips it."""
    try:
        q = float(qty or 0.0)
    except (TypeError, ValueError):
        return 0.0
    s = str(side or "").strip().upper()
    if s in ("SELL", "SHORT", "SLD") and q > 0:
        return -q
    return q


def _intent_attr(intent: Any, name: str, default: Any = None) -> Any:
    """Attribute or mapping key — intents are dataclasses in production and
    dicts in tests (the fuse_gate._intent_attr idiom)."""
    v = getattr(intent, name, None)
    if v is None and isinstance(intent, Mapping):
        v = intent.get(name, default)
    return v if v is not None else default


def _build(
    *,
    symbol: str,
    side: str,
    strategy: str,
    origin: str,
    signed_qty: float,
    sec_type: Any,
    currency: str,
    price: Optional[float],
    price_source: Optional[str],
    sector_lookup,
) -> ExposureVector:
    """Shared constructor — both origins funnel here so a position and an
    intent on the same symbol are normalized by exactly one code path."""
    asset_class = classify_asset_class(sec_type, symbol)
    venue = derive_venue(asset_class)
    mult, mult_reason = multiplier_for(asset_class, symbol)

    delta: Optional[float] = None
    null_reason: Optional[str] = mult_reason
    if signed_qty == 0.0:
        null_reason = null_reason or REASON_NO_QUANTITY
    elif price is None:
        null_reason = null_reason or REASON_NO_PRICE
    elif mult is not None:
        delta = signed_qty * float(price) * float(mult)
        null_reason = None

    if delta is None and null_reason is None:
        null_reason = REASON_NO_PRICE

    beta = 1.0
    return ExposureVector(
        symbol=str(symbol or "").strip().upper(),
        side=str(side or "").strip().upper(),
        strategy=str(strategy or "").strip().lower(),
        origin=origin,
        quantity=signed_qty,
        delta_usd=delta,
        beta=beta,
        beta_source="default_1.0",
        beta_weighted_usd=(None if delta is None else delta * beta),
        sector=(sector_lookup(symbol) if sector_lookup else UNMAPPED_SECTOR)
        or UNMAPPED_SECTOR,
        asset_class=asset_class,
        venue=venue,
        currency=str(currency or "").strip().upper() or "USD",
        multiplier=mult,
        price=(float(price) if price is not None else None),
        price_source=price_source,
        null_reason=null_reason,
    )


def vector_from_position(
    position: Mapping[str, Any],
    *,
    prices: Mapping[str, float],
    sector_lookup,
) -> ExposureVector:
    """Normalize one broker-truth position.

    Price ladder: live mark from price_cache, else the position's own avgCost.
    positions_truth carries NO inline mark (P8) — avgCost is entry cost, not a
    mark, so it is the fallback and it is labelled as such in `price_source`.
    A book valued at avgCost is a stale book, and the evidence says which.
    """
    symbol = str(position.get("symbol") or "").strip().upper()
    signed_qty = _signed_quantity(position.get("position"), None)
    price = prices.get(symbol)
    price_source: Optional[str] = PRICE_SOURCE_CACHE if price is not None else None
    if price is None:
        try:
            avg = float(position.get("avgCost"))
            if avg > 0:
                price, price_source = avg, PRICE_SOURCE_AVG_COST
        except (TypeError, ValueError):
            pass
    return _build(
        symbol=symbol,
        side=("LONG" if signed_qty >= 0 else "SHORT"),
        strategy=str(position.get("strategy") or "broker_truth"),
        origin="position",
        signed_qty=signed_qty,
        sec_type=position.get("secType"),
        currency=str(position.get("currency") or "USD"),
        price=price,
        price_source=price_source,
        sector_lookup=sector_lookup,
    )


def vector_from_intent(
    intent: Any,
    *,
    prices: Mapping[str, float],
    sector_lookup,
) -> ExposureVector:
    """Normalize one pending intent (the marginal ticket).

    Price ladder: limit_price → expected_price → price_cache. NOTE (PLAN_W5B
    §11.2): W5A's realized `decision_price_source` ladder is
    `submit_quote.ref_price → expected_price`. The two AGREE on the
    expected_price fallback and DIFFER on the primary leg, because the allocator
    runs upstream of submit and no quote exists yet. A later comparison of
    assumed-vs-realized decision price must account for that rather than
    treating the two numbers as the same.
    """
    symbol = str(_intent_attr(intent, "symbol", "") or "").strip().upper()
    side = str(_intent_attr(intent, "side", "") or "")
    signed_qty = _signed_quantity(_intent_attr(intent, "quantity", 0.0), side)

    price: Optional[float] = None
    price_source: Optional[str] = None
    for attr, src in (
        ("limit_price", PRICE_SOURCE_LIMIT),
        ("expected_price", PRICE_SOURCE_EXPECTED),
    ):
        try:
            v = _intent_attr(intent, attr)
            if v is not None and float(v) > 0:
                price, price_source = float(v), src
                break
        except (TypeError, ValueError):
            continue
    if price is None and symbol in prices:
        price, price_source = prices[symbol], PRICE_SOURCE_CACHE

    return _build(
        symbol=symbol,
        side=side,
        strategy=str(_intent_attr(intent, "strategy", "") or ""),
        origin="intent",
        signed_qty=signed_qty,
        sec_type=_intent_attr(intent, "sec_type"),
        currency=str(_intent_attr(intent, "currency", "USD") or "USD"),
        price=price,
        price_source=price_source,
        sector_lookup=sector_lookup,
    )


# --------------------------------------------------------------------------- #
# The provisional book
# --------------------------------------------------------------------------- #

class ProvisionalBook:
    """Open book at cycle start, plus every entry intent seen so far this cycle.

    Why intents accumulate regardless of verdict: in shadow nothing blocks, so
    the honest counterfactual is the book enforcement WOULD have seen. If a
    would-rejected intent were left out, the second and third correlated tickets
    would each be measured against a book missing the first, and the corpus
    would understate concentration exactly where it matters most.

    BOUND — FINDING W5B-SF1 (standing). This book is open-positions-at-cycle-start
    plus entries. Overlay, crypto-overlay, reconciler and flatten closes never
    traverse stage-3, so their effects land in the NEXT cycle's snapshot. The
    book can therefore be one cycle stale, and a flip — whose closing leg
    bypasses while the executor and reconciler move the position — is the sharp
    case. Shadow evidence supports "this entry would have breached given the
    book as of cycle start"; it does NOT support "gross never exceeded X".
    """

    def __init__(self, base: Iterable[ExposureVector] = ()) -> None:
        self._base: List[ExposureVector] = [v for v in base]
        self._added: List[ExposureVector] = []

    # -- accumulation ------------------------------------------------------ #

    def add_intent(self, vector: ExposureVector) -> None:
        self._added.append(vector)

    @property
    def vectors(self) -> List[ExposureVector]:
        return self._base + self._added

    @property
    def base_vectors(self) -> List[ExposureVector]:
        return list(self._base)

    @property
    def intent_vectors(self) -> List[ExposureVector]:
        return list(self._added)

    # -- aggregates -------------------------------------------------------- #
    # Non-computable vectors (unmapped futures root, no price) are EXCLUDED
    # from every sum and counted separately. Treating an unknown as zero would
    # understate gross silently, which is the failure mode this whole layer
    # exists to prevent.

    @property
    def gross_usd(self) -> float:
        return sum(abs(v.delta_usd) for v in self.vectors if v.computable)

    @property
    def net_usd(self) -> float:
        return sum(v.delta_usd for v in self.vectors if v.computable)

    @property
    def uncomputable(self) -> List[ExposureVector]:
        return [v for v in self.vectors if not v.computable]

    def by_symbol(self) -> Dict[str, float]:
        """Signed net delta per symbol. Position and intent legs on one symbol
        combine here — the per-symbol concentration limit is measured on the
        COMBINED ticket, which is precisely what the per-order cap cannot do."""
        out: Dict[str, float] = {}
        for v in self.vectors:
            if v.computable:
                out[v.symbol] = out.get(v.symbol, 0.0) + v.delta_usd
        return out

    def by_sector(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for v in self.vectors:
            if v.computable:
                out[v.sector] = out.get(v.sector, 0.0) + v.delta_usd
        return out

    def by_venue(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for v in self.vectors:
            if v.computable:
                out[v.venue] = out.get(v.venue, 0.0) + abs(v.delta_usd)
        return out

    def currency_mix(self) -> Dict[str, float]:
        """Gross by instrument currency. A single key ("USD" today) means the
        sums are unit-clean. More than one key means gross is mixing currencies
        with no FX rate to bridge them (§12.2) — the evidence must show it
        rather than silently adding CAD to USD."""
        out: Dict[str, float] = {}
        for v in self.vectors:
            if v.computable:
                out[v.currency] = out.get(v.currency, 0.0) + abs(v.delta_usd)
        return out

    def summary(self) -> Dict[str, Any]:
        return {
            "symbols": len({v.symbol for v in self.vectors}),
            "gross_usd": round(self.gross_usd, 2),
            "net_usd": round(self.net_usd, 2),
            "by_sector": {k: round(v, 2) for k, v in sorted(self.by_sector().items())},
            "by_venue": {k: round(v, 2) for k, v in sorted(self.by_venue().items())},
            "currency_mix": {
                k: round(v, 2) for k, v in sorted(self.currency_mix().items())
            },
            "uncomputable": len(self.uncomputable),
            "uncomputable_reasons": sorted(
                {v.null_reason for v in self.uncomputable if v.null_reason}
            ),
        }


def build_base_book(
    *,
    positions: Optional[Iterable[Mapping[str, Any]]] = None,
    prices: Optional[Mapping[str, float]] = None,
    sector_lookup=None,
    positions_path: Optional[Path] = None,
    price_cache_path: Optional[Path] = None,
) -> ProvisionalBook:
    """Snapshot the open book once per cycle. Explicit arguments win (tests);
    otherwise the live read-only artifacts are used."""
    rows = list(positions) if positions is not None else load_book_positions(positions_path)
    px = dict(prices) if prices is not None else load_price_cache(price_cache_path)
    lookup = sector_lookup if sector_lookup is not None else load_sector_lookup()
    return ProvisionalBook(
        vector_from_position(r, prices=px, sector_lookup=lookup) for r in rows
    )


__all__ = [
    "ExposureVector",
    "ProvisionalBook",
    "FUTURES_POINT_VALUES",
    "VENUE_IBKR",
    "VENUE_KRAKEN",
    "UNMAPPED_SECTOR",
    "build_base_book",
    "classify_asset_class",
    "derive_venue",
    "futures_root",
    "load_book_positions",
    "load_price_cache",
    "load_sector_lookup",
    "multiplier_for",
    "vector_from_intent",
    "vector_from_position",
]
