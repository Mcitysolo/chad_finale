"""chad/validation/trade_log_adapter.py — Phase 6 Stage-2 real-trade ingest seam.

The wall between paper and live money has one job in Stage 2 (SSOT
``docs/CHAD_EDGE_VALIDATION_HARNESS_DESIGN_v1.1.md`` §1.3 / Part 6, Phase 6): feed
REAL paper-trading fills into the *identical* Stage-1 verdict engine (Phases 0-5) so
the harness can render a verdict on actual trades rather than synthetic backtests.
"Only the input adapter differs. One ``ScoringSpine``, no duplicate scoring logic."

This module is that adapter and NOTHING more. It:

  * reads the canonical realized-trade ledger ``data/trades/trade_history_YYYYMMDD.ndjson``
    (the only files matched are the un-suffixed daily ledgers — the ``.scr_reset_bak`` /
    ``.pre_*_bak`` backups are deliberately skipped so a reset-stamped or pre-fix copy of a
    day can never be double-counted);
  * enforces the SSOT trust defenses **fail-closed** — a row that is ``pnl_untrusted``,
    a Kraken ``validate_only`` no-realized-fill, a ``$100`` placeholder, or broker-rejected
    / unconfirmed is *structurally unable* to reach the downstream scorer (see
    :func:`trust_exclusion` and the post-admission self-check in :func:`adapt_records`);
  * maps every ADMITTED row onto the mapping :meth:`chad.validation.cost_model.Trade.from_fill`
    consumes, so the same commission + half-spread + slippage haircut (SSOT §3.5 / S4) is
    charged on real fills as on synthetic Stage-1 trades — this module does NOT itself
    compute costs or a verdict (that stays in the one spine, invoked by ``cli.py``);
  * emits an auditable manifest (input files + sha256, rows read, rows admitted, rows
    excluded by reason, date range) alongside a deterministic, stable-ordered ndjson of the
    admitted canonical trades.

Determinism (SSOT §3.8): identical input files → byte-identical output. Admitted rows are
emitted in a total order (entry time, sequence id, record hash); the manifest lists input
files sorted by name with their sha256. The ONLY non-input-derived value anywhere in the
output is the manifest ``generated_at`` field (overridable via ``--now`` for reproducibility).

Trust-filter provenance (mirrors the live write-path defenses so the offline gate can never
be *more* permissive than the online one):
  * ``$100`` placeholder fingerprint — ``chad/execution/paper_exec_evidence_writer.py:195``
    (``_PLACEHOLDER_FILL_PRICE = 100.0``) and the guard at lines 1922-1958;
  * broker-rejected statuses — ``paper_exec_evidence_writer.py:119`` (``_PAPER_REJECTED_STATUSES``)
    and the demotion at lines 2043-2052 (``pnl_untrusted`` + ``broker_rejected`` tags);
  * canonical trusted fill status — ``paper_exec_evidence_writer.py:135`` (``_STATUS_CANON`` →
    ``paper_fill``), the single value SCR / trade_closer treat as a genuine fill;
  * ``pnl_untrusted`` / ``validate_only`` — the ``payload.extra`` flags every Kraken paper
    and reset-stamped ledger row carries.

Isolation (SSOT §1.2 / §2): standard-library only + the sibling ``chad.validation.cost_model``
enums. It imports NO broker / execution / runtime module and reads the evidence writer only as
TEXT ndjson at run time (never imports it), so the transitive-import-closure isolation test
(``tests/validation/test_isolation.py``) stays green. Its only writes are the two artifacts
under the caller-supplied output directory; it never writes ``runtime/`` or ``ready_for_live``.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from chad.validation.cost_model import InstrumentClass, LiquidityTier

__all__ = [
    "SCHEMA_VERSION",
    "TRUSTED_FILL_STATUSES",
    "REJECTED_STATUSES",
    "PLACEHOLDER_FILL_PRICE",
    "EXCLUSION_REASONS",
    "EXIT_OVERLAY_BOUNDARY",
    "era_of",
    "AdmittedTrade",
    "AdapterManifest",
    "AdapterResult",
    "trust_exclusion",
    "is_quarantined",
    "classify_instrument",
    "is_placeholder_fill",
    "adapt_records",
    "iter_ledger_files",
    "load_quarantine_pins",
    "load_scr_effective",
    "ScrCrosscheckError",
    "load_ndjson",
    "verify_ledger_chain",
    "LedgerChainError",
    "run_adapter",
    "main",
]

SCHEMA_VERSION: str = "stage2_trade_log.v1"

# The single canonical trusted-fill status (PA-EP8): every genuine synchronous fill is
# collapsed to "paper_fill" at the evidence chokepoint; "fill" is the legacy literal.
# A row that carries ANY status outside this set is NOT a confirmed fill (SSOT trust).
TRUSTED_FILL_STATUSES: frozenset[str] = frozenset({"paper_fill", "fill"})

# Statuses the broker did NOT accept — the order never traded, so any realized PnL is
# fictional (paper_exec_evidence_writer.py:119 _PAPER_REJECTED_STATUSES).
REJECTED_STATUSES: frozenset[str] = frozenset({"error", "failed", "rejected", "cancelled"})

# The paper-mode "no live price" placeholder fallback fingerprint
# (paper_exec_evidence_writer.py:195 _PLACEHOLDER_FILL_PRICE).
PLACEHOLDER_FILL_PRICE: float = 100.0

# Futures roots present in CHAD's tradable universe → InstrumentClass.FUT. Reconciled with
# the canonical futures universe (chad/market_data/dynamic_universe_scanner.py:60 = {MES,
# MNQ, MCL, MGC, ZN, ZB, M6E, SIL, MYM, M2K}); ZN/ZB (Treasuries) are included so a Treasury
# fill without an explicit asset_class is not misclassified STK. §1.2 isolation forbids
# importing market_data, so this is a cited local copy (a symbol whose asset_class IS present
# is classified from that first — this set is only the last-resort symbol heuristic). M6A/M6B
# (micro AUD/GBP FX futures) are retained. A ledger symbol may carry a contract month+year
# suffix (e.g. "MESU6" = MES + U[Sep] + 6[2026]) — stripped below.
_FUTURES_ROOTS: frozenset[str] = frozenset(
    {"MES", "MNQ", "MYM", "M2K", "M6E", "M6A", "M6B", "MGC", "MCL", "SIL", "ZN", "ZB"}
)
# Futures month codes (F G H J K M N Q U V X Z) + a 1-2 digit year suffix.
_FUT_SUFFIX_RE = re.compile(r"[FGHJKMNQUVXZ]\d{1,2}$")

# Stable exclusion-reason taxonomy (manifest keys). ``out_of_window`` / ``malformed`` are
# tracked separately from the trust reasons but listed here so the manifest schema is fixed.
EXCLUSION_REASONS: tuple[str, ...] = (
    "placeholder_100",
    "broker_rejected",
    "non_fill_status",
    "validate_only",
    "pnl_untrusted",
    # B2 (FLIP-UNBLOCK): fabricated cost basis (Epoch-3 seed lot).
    "scoring_excluded",
    # W3A-2 (D2): SCR-predicate parity. SCR's effective set excludes operator hand-trades
    # (tag "manual") and synthetic warmup rows (tag "warmup_sim"); the adapter mirrors both so
    # the two scorekeepers exclude the same kinds (see chad/analytics/trade_stats_engine.py
    # _is_manual :107-108 / _is_warmup_sim :146-148).
    "manual",
    "warmup_sim",
    # W3A-2 (D2): futures rows are Bug-B-contaminated (untrusted PnL) pending the Bug-B book
    # disposition; excluded from scoring by default (re-enable with include_futures=True once
    # Bug-B is resolved). SCR excludes futures for the same reason (is_futures_row).
    "futures_bug_b",
    # W3A-2 (D6): a row without schema_version == "closed_trade.v1" (legacy / foreign-writer)
    # is not a canonical closed round-trip → excluded structurally, before the trust gate.
    "non_closed_trade",
    # W2A-1: operator quarantine pin (runtime/quarantine_manifest_*.json). A row whose
    # ``record_hash`` / ``fill_id`` / ``fill_ids`` is pinned in an operator manifest is
    # dropped BEFORE the trust gate — the same authority SCR honours natively
    # (chad.analytics.trade_stats_engine via chad.utils.quarantine.get_exclusion_sets), so
    # both scorekeepers exclude the identical set. This is the ONLY way to scrub a genuine-
    # looking round-trip (e.g. the PFF1 harvester double-book phantoms) without rewriting the
    # hash-chained ledger. Read as TEXT (see :func:`load_quarantine_pins`) — no chad import,
    # so the harness import-closure isolation (tests/validation/test_isolation.py) is preserved.
    "quarantined",
)

_LEDGER_RE = re.compile(r"^trade_history_(\d{8})\.ndjson$")

# The canonical FIFO-closed-round-trip schema (hash-chained by trade_closer.py:
# write_trade_history). This exact value is what the chain recompute (verify_ledger_chain) is
# valid for.
_CLOSED_TRADE_SCHEMA: str = "closed_trade.v1"

# D6 admit gate: the KNOWN closed-lap schema FAMILIES. trade_history carries closed round-trips
# from two writers — closed_trade.v* (chad/execution/trade_closer.py) and paper_trade_result.v*
# (chad/portfolio/ibkr_paper_trade_result_logger.py, used by the Kraken TrustedFillEngine). Both
# are real laps with realized pnl + entry/exit; a row outside these families (raw fills, legacy
# foreign-writer rows, bars/state files) is not a closed lap and is excluded (non_closed_trade).
# Prefix-matched so a future minor version (…v2/v3) is accepted without a code change.
_ADMISSIBLE_SCHEMA_PREFIXES: tuple[str, ...] = ("closed_trade.", "paper_trade_result.")


def _is_closed_lap_schema(schema_version: str) -> bool:
    """True iff ``schema_version`` names a known closed-lap family (D6 admit gate)."""
    return any(schema_version.startswith(p) for p in _ADMISSIBLE_SCHEMA_PREFIXES)

# D4: the FROZEN exit-overlay activation boundary. Laps before it are a DIFFERENT
# data-generating process (pre-overlay reconciliation/adopt closes) than laps on/after it
# (genuine engine-driven overlay closes). position_exit_overlay went SHADOW→ACTIVE here; the
# first ACTIVE close was 2026-07-20T13:50:55Z (docs/ULTRA_CLOSE_AUDIT_2026-07-17.md). The two
# eras are SEPARATE populations and must never be pooled into one headline verdict.
EXIT_OVERLAY_BOUNDARY: str = "2026-07-20"


def era_of(exit_date: Optional[str]) -> str:
    """Sample-regime label for a lap's realization date vs the frozen overlay boundary (D4).

    Returns ``"PRE_OVERLAY"`` (exit date < boundary), ``"POST_OVERLAY"`` (>= boundary), or
    ``"UNKNOWN_ERA"`` when no usable ``YYYY-MM-DD`` date is present. The two eras are distinct
    data-generating processes; a verdict must never silently pool them (the harness reports
    them separately, pooling only as an explicitly-labeled sensitivity view)."""
    if not (isinstance(exit_date, str) and len(exit_date) >= 10 and exit_date[4] == "-"):
        return "UNKNOWN_ERA"
    return "PRE_OVERLAY" if exit_date[:10] < EXIT_OVERLAY_BOUNDARY else "POST_OVERLAY"


def _admitted_exit_date(t: "AdmittedTrade") -> Optional[str]:
    for src in (t.provenance.get("exit_time_utc"), t.provenance.get("entry_time_utc")):
        if isinstance(src, str) and len(src) >= 10 and src[4] == "-" and src[7] == "-":
            return src[:10]
    return None


# --------------------------------------------------------------------------- #
# Small pure helpers.
# --------------------------------------------------------------------------- #
def _truthy(value: Any) -> bool:
    """Interpret a JSON flag as a bool (True only for ``True`` / ``"true"`` / ``1``)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def _finite_number(value: Any) -> Optional[float]:
    """Return ``float(value)`` iff it is a real, finite number, else ``None`` (never raises)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    if f != f or f in (float("inf"), float("-inf")):  # NaN / ±inf
        return None
    return f


def _payload(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """The trade payload — records are ``{"payload": {...}, ...}``; tolerate a bare payload."""
    p = record.get("payload")
    return p if isinstance(p, Mapping) else record


def _extra(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    ex = payload.get("extra")
    return ex if isinstance(ex, Mapping) else {}


def _tags(payload: Mapping[str, Any]) -> list[str]:
    return [str(t).lower() for t in (payload.get("tags") or []) if t is not None]


def _meta(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """The closed_trade.v1 FIFO-lot meta block (B2) — the third trust carrier.

    An Epoch-3 seed lot's fabricated cost basis is marked ``pnl_untrusted`` /
    ``scoring_excluded`` on the opening lot's meta. trade_closer mirrors those
    into ``extra``/``tags`` at write time, but rows written before that mirror
    existed carry the marker on ``meta`` alone — and admitting one injects
    invented alpha into the scorer.
    """
    m = payload.get("meta")
    return m if isinstance(m, Mapping) else {}


def _status(payload: Mapping[str, Any], extra: Mapping[str, Any]) -> Optional[str]:
    """The record's fill status, lowercased — ``None`` when the ledger row carries none.

    The realized-trade ledger predates PA-EP8 and usually has NO status field; such rows
    are judged purely by the pnl_untrusted / validate_only / placeholder flags. When a
    status IS present (evidence-joined or post-EP8), it must be a canonical trusted fill.
    """
    raw = payload.get("status")
    if raw is None:
        raw = extra.get("status")
    if raw is None:
        return None
    return str(raw).strip().lower()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _row_date(payload: Mapping[str, Any], record: Mapping[str, Any]) -> Optional[str]:
    """The row's UTC calendar date ``YYYY-MM-DD`` from entry/exit/record timestamps."""
    for src in (
        payload.get("entry_time_utc"),
        payload.get("exit_time_utc"),
        record.get("timestamp_utc"),
    ):
        if isinstance(src, str) and len(src) >= 10 and src[4] == "-" and src[7] == "-":
            return src[:10]
    return None


def _parse_ts(value: Any) -> Optional[datetime.datetime]:
    """Parse a UTC timestamp tolerating BOTH ``…Z`` and ``…+00:00`` offset forms (real ledger
    rows mix them). Returns ``None`` when unparseable (never raises). A naive value is treated
    as UTC so the walk-forward time axis (cli.py, W3A-4) has a single tz."""
    if not isinstance(value, str) or not value.strip():
        return None
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt


def _hold_hours(payload: Mapping[str, Any]) -> tuple[Optional[float], bool]:
    """``(hold_hours, inverted)`` from the entry/exit timestamps.

    ``inverted`` is ``True`` when ``exit_time_utc`` precedes ``entry_time_utc`` — a
    netting/clock artifact seen on genuine laps (e.g. a real 07-20 gamma close). Such a row is
    KEPT (excluding a real lap over a sub-minute clock skew loses honest data), but the
    inverted magnitude is reported and NOT trusted as a negative hold; ``hold_hours`` is the
    absolute span and the walk-forward axis orders on the later of the two legs."""
    e = _parse_ts(payload.get("entry_time_utc"))
    x = _parse_ts(payload.get("exit_time_utc"))
    if e is None or x is None:
        return (None, False)
    delta_h = (x - e).total_seconds() / 3600.0
    return (abs(delta_h), delta_h < 0.0)


# --------------------------------------------------------------------------- #
# Instrument classification + placeholder fingerprint.
# --------------------------------------------------------------------------- #
def classify_instrument(payload: Mapping[str, Any], extra: Mapping[str, Any]) -> InstrumentClass:
    """Resolve the InstrumentClass from explicit asset_class, broker, then symbol (documented).

    Precedence: an explicit ``asset_class`` wins; else a ``kraken*`` broker ⇒ CRYPTO; else a
    known futures root ⇒ FUT; else a ``*-USD`` / ``*USD`` crypto pair ⇒ CRYPTO; else STK.

    Real ``closed_trade.v1`` rows carry no top-level ``asset_class``; the reliable equity
    signal is ``meta.raw_asset_class`` (e.g. ``"equity"``), so it is added to the explicit
    precedence (W3A-1) — otherwise an equity with a futures-looking symbol could be misclassed.
    """
    meta = _meta(payload)
    ac = str(
        payload.get("asset_class")
        or extra.get("asset_class")
        or meta.get("raw_asset_class")
        or ""
    ).strip().lower()
    if ac in {"crypto", "cryptocurrency", "coin"}:
        return InstrumentClass.CRYPTO
    if ac in {"future", "futures", "fut"}:
        return InstrumentClass.FUT
    if ac in {"stock", "equity", "etf", "stk"}:
        return InstrumentClass.STK

    broker = str(payload.get("broker") or "").strip().lower()
    if "kraken" in broker or "crypto" in broker:
        return InstrumentClass.CRYPTO

    symbol = str(payload.get("symbol") or "").strip().upper()
    base = symbol.split("-")[0].split(".")[0]
    root = _FUT_SUFFIX_RE.sub("", base)  # strip a contract month+year code, if any
    if symbol in _FUTURES_ROOTS or base in _FUTURES_ROOTS or root in _FUTURES_ROOTS:
        return InstrumentClass.FUT
    if symbol.endswith("-USD") or symbol.endswith("USD") or "/" in symbol:
        return InstrumentClass.CRYPTO
    return InstrumentClass.STK


def is_placeholder_fill(payload: Mapping[str, Any], extra: Mapping[str, Any]) -> bool:
    """True iff the row bears the ``$100`` "no live price" placeholder fingerprint.

    Mirrors the live guard (paper_exec_evidence_writer.py:1922-1958): ``fill_price == 100.0``
    AND ``expected_price`` == 100.0 or missing/zero AND an equity-class instrument. Also
    catches the explicit placeholder provenance the guard stamps (``trust_state=PLACEHOLDER``,
    a ``placeholder``/``placeholder_price`` tag, or a ``placeholder*`` untrusted reason) so a
    placeholder is refused even if a future writer forgot the numeric fingerprint. Fail-closed.
    """
    if str(extra.get("trust_state") or "").strip().upper() == "PLACEHOLDER":
        return True
    reason = str(extra.get("pnl_untrusted_reason") or "").strip().lower()
    if reason.startswith("placeholder"):
        return True
    tags = _tags(payload)
    if "placeholder" in tags or "placeholder_price" in tags:
        return True

    fill_price = _finite_number(payload.get("fill_price"))
    if fill_price is None or abs(fill_price - PLACEHOLDER_FILL_PRICE) >= 1e-9:
        return False
    expected = _finite_number(extra.get("expected_price"))
    expected_matches = expected is None or expected == 0.0 or abs(expected - PLACEHOLDER_FILL_PRICE) < 1e-9
    if not expected_matches:
        return False
    inst = classify_instrument(payload, extra)
    return inst is InstrumentClass.STK


# --------------------------------------------------------------------------- #
# The trust gate (fail-closed). Returns the FIRST-matched exclusion reason, or None.
# --------------------------------------------------------------------------- #
def trust_exclusion(record: Mapping[str, Any]) -> Optional[str]:
    """Return the trust-exclusion reason for a fill record, or ``None`` if admissible.

    Precedence (each excluded row is counted once, under its first-matched reason):
    ``placeholder_100`` → ``broker_rejected`` → ``non_fill_status`` → ``validate_only`` →
    ``pnl_untrusted``. A row admitted here has NONE of the SSOT untrust markers; it is the
    only kind :func:`adapt_records` will ever hand downstream.
    """
    payload = _payload(record)
    extra = _extra(payload)
    tags = _tags(payload)

    # 1. $100 placeholder (independent detector; defense-in-depth vs the pnl_untrusted flag).
    if is_placeholder_fill(payload, extra):
        return "placeholder_100"

    # 2/3. Status: broker-rejected, or any non-canonical (unconfirmed) status when present.
    status = _status(payload, extra)
    if status is not None:
        if status in REJECTED_STATUSES:
            return "broker_rejected"
        if status not in TRUSTED_FILL_STATUSES:
            return "non_fill_status"
    if "broker_rejected" in tags:
        return "broker_rejected"

    # 4. Kraken validate-only: the order was accepted for validation but never realized.
    if _truthy(extra.get("validate_only")) or "validate_only" in tags:
        return "validate_only"

    # 5. Any explicitly untrusted PnL (reset-stamped, synthetic-credit, etc.).
    meta = _meta(payload)
    if (
        _truthy(extra.get("pnl_untrusted"))
        or "pnl_untrusted" in tags
        or _truthy(meta.get("pnl_untrusted"))
    ):
        return "pnl_untrusted"

    # 6. B2: scored-basis exclusion. A row whose cost basis was fabricated by the
    # Epoch-3 rebuild (seed lot) rather than paid in the market is arithmetically
    # well-formed but economically fictional. It must never reach the scorer.
    if (
        _truthy(extra.get("scoring_excluded"))
        or "scoring_excluded" in tags
        or _truthy(meta.get("scoring_excluded"))
    ):
        return "scoring_excluded"

    # 7. W3A-2 (D2): SCR-predicate parity — operator hand-trades and synthetic warmup rows are
    # not strategy edge; SCR excludes them from its effective set, so the adapter does too.
    if "manual" in tags or str(payload.get("strategy") or "").strip().lower() == "manual":
        return "manual"
    if "warmup_sim" in tags:
        return "warmup_sim"

    return None


def is_quarantined(
    record: Mapping[str, Any],
    quarantined_hashes: "frozenset[str]",
    quarantined_fill_ids: "frozenset[str]",
) -> bool:
    """True iff *record* is pinned by an operator quarantine manifest.

    Mirrors ``chad.utils.quarantine.is_record_quarantined`` for the manifest-derived
    pins ONLY — top-level ``record_hash``, ``payload.fill_id``, and any element of
    ``payload.fill_ids`` (the derived-closed-trade carrier, which is what catches a
    round-trip built from a pinned opening/closing fill). Reimplemented here as a pure
    stdlib helper rather than imported, so the adapter's import closure stays free of
    ``chad.utils`` (tests/validation/test_isolation.py). It deliberately does NOT
    re-derive ``pnl_untrusted`` — that in-band trust marker is :func:`trust_exclusion`'s
    job; this gate is only the operator's explicit manifest pin.
    """
    if not (quarantined_hashes or quarantined_fill_ids):
        return False
    rh = record.get("record_hash")
    if isinstance(rh, str) and rh in quarantined_hashes:
        return True
    payload = record.get("payload")
    if isinstance(payload, Mapping):
        fid = payload.get("fill_id")
        if isinstance(fid, str) and fid in quarantined_fill_ids:
            return True
        fids = payload.get("fill_ids")
        if isinstance(fids, list):
            for f in fids:
                if isinstance(f, str) and f in quarantined_fill_ids:
                    return True
    return False


# --------------------------------------------------------------------------- #
# Canonical admitted trade.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AdmittedTrade:
    """One admitted real round-trip, mapped onto the cost-model / spine contract.

    ``entry_price`` and ``exit_price`` are BOTH set to the realized ``fill_price`` so the
    two-leg cost haircut (SSOT §3.5) is charged on the traded notional; the authoritative
    realized PnL is carried as ``gross_pnl`` (NOT reconstructed from a fabricated price
    path). ``notional`` is the recorded traded notional (the per-trade return denominator).
    Everything under ``provenance`` is carried for audit and is not consumed by the scorer.
    """

    instrument_class: str
    quantity: float
    fill_price: float
    notional: float
    gross_pnl: float
    liquidity_tier: str
    multiplier: float
    strategy: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_fill_mapping(self) -> dict[str, Any]:
        """The mapping :meth:`chad.validation.cost_model.Trade.from_fill` consumes.

        Prices equal the realized fill (both legs costed on the traded notional); ``pnl`` is
        the authoritative realized PnL. Only admitted trades ever produce this mapping.
        """
        return {
            "instrument_class": self.instrument_class,
            "quantity": self.quantity,
            "entry_price": self.fill_price,
            "exit_price": self.fill_price,
            "liquidity_tier": self.liquidity_tier,
            "multiplier": self.multiplier,
            "pnl": self.gross_pnl,
            "strategy": self.strategy,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument_class": self.instrument_class,
            "quantity": self.quantity,
            "fill_price": self.fill_price,
            "notional": self.notional,
            "gross_pnl": self.gross_pnl,
            "liquidity_tier": self.liquidity_tier,
            "multiplier": self.multiplier,
            "strategy": self.strategy,
            "provenance": dict(self.provenance),
        }


class _MalformedRow(ValueError):
    """A row that cannot become an AdmittedTrade (missing/invalid required field)."""


def _to_admitted(record: Mapping[str, Any], source_file: str) -> AdmittedTrade:
    """Build an :class:`AdmittedTrade` from a TRUSTED record; raise ``_MalformedRow`` if it
    lacks the numeric fields the cost model requires (quantity/fill_price/notional/pnl).

    Never called on an untrusted row (the caller gates on :func:`trust_exclusion` first).
    """
    payload = _payload(record)
    extra = _extra(payload)

    qty = _finite_number(payload.get("quantity") if payload.get("quantity") is not None else payload.get("size"))
    price = _finite_number(payload.get("fill_price"))
    # D3 (W3A-1): feed GROSS pnl to the harness cost model so the harness haircut is the SINGLE
    # cost authority. Real fills today carry gross==net (commission/slippage=0), but PA-EP1
    # fee-modeling is forward-only — once modeled costs land in the ledger, net_pnl != gross_pnl
    # and feeding a net figure on top of the harness haircut would DOUBLE-CHARGE post-EP1 rows.
    # Prefer gross_pnl; fall back to pnl (pre-EP1 rows, where they are equal). The field used is
    # recorded in provenance + tallied in the manifest so the choice is auditable per row.
    gross_raw = payload.get("gross_pnl")
    if _finite_number(gross_raw) is not None:
        pnl = _finite_number(gross_raw)
        pnl_field = "gross_pnl"
    else:
        pnl = _finite_number(payload.get("pnl"))
        pnl_field = "pnl"
    notional = _finite_number(payload.get("notional"))
    if notional is None and qty is not None and price is not None:
        notional = abs(qty * price)

    if qty is None or qty <= 0.0:
        raise _MalformedRow("missing/non-positive quantity")
    if price is None or price <= 0.0:
        raise _MalformedRow("missing/non-positive fill_price")
    if notional is None or notional <= 0.0:
        raise _MalformedRow("missing/non-positive notional")
    if pnl is None:
        raise _MalformedRow("missing/non-finite pnl")

    # multiplier: extra.multiplier → payload.multiplier → payload.contract_multiplier (the key
    # real closed_trade.v1 rows actually carry, W3A-1) → default 1.0.
    mult_raw = extra.get("multiplier")
    if mult_raw is None:
        mult_raw = payload.get("multiplier")
    if mult_raw is None:
        mult_raw = payload.get("contract_multiplier")
    mult = _finite_number(mult_raw)
    if mult is None or mult <= 0.0:
        mult = 1.0
    inst = classify_instrument(payload, extra)
    strategy = str(payload.get("strategy") or "unattributed").strip() or "unattributed"
    hold_h, inverted = _hold_hours(payload)

    provenance = {
        "symbol": payload.get("symbol"),
        "side": payload.get("side"),
        "broker": payload.get("broker"),
        "is_live": payload.get("is_live"),
        "regime": payload.get("regime"),
        "entry_time_utc": payload.get("entry_time_utc"),
        "exit_time_utc": payload.get("exit_time_utc"),
        "execution_id": extra.get("execution_id"),
        "txid": extra.get("txid"),
        "fill_id": record.get("fill_id") or payload.get("fill_id"),
        "fee_model": extra.get("fee_model"),
        "provenance": extra.get("provenance"),
        "slippage_bps": extra.get("slippage_bps"),
        "latency_ms": extra.get("latency_ms"),
        "sequence_id": record.get("sequence_id"),
        "record_hash": record.get("record_hash"),
        "source_file": source_file,
        # W3A-1: schema-mapping provenance.
        "pnl_field": pnl_field,
        "hold_hours": hold_h,
        "inverted_duration": inverted,
    }
    return AdmittedTrade(
        instrument_class=inst.value,
        quantity=qty,
        fill_price=price,
        notional=notional,
        gross_pnl=pnl,
        liquidity_tier=LiquidityTier.MID.value,
        multiplier=mult,
        strategy=strategy,
        provenance=provenance,
    )


# --------------------------------------------------------------------------- #
# Manifest + result.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AdapterManifest:
    """The auditable manifest emitted beside the admitted-trades ndjson (SSOT §3.8)."""

    schema_version: str
    generated_at: str
    since: Optional[str]
    until: Optional[str]
    input_files: list[dict[str, Any]]          # [{path, sha256, rows}]
    rows_read: int
    rows_in_window: int
    out_of_window: int
    malformed: int
    duplicate: int
    admitted: int
    excluded_by_reason: dict[str, int]         # FIXED schema: every EXCLUSION_REASONS key, 0+
    date_range_admitted: Optional[str]         # "min..max" over admitted rows, or None
    strategies_admitted: dict[str, int]
    notes: list[str]
    # CRYPTO-TRUST U1 honesty guard: evidence-class separation so no verdict
    # silently mixes SIMULATED_AGAINST_LIVE_TICKS crypto with broker-confirmed
    # fills. Keyed by fill-level provenance label (unlabeled -> "unspecified")
    # and by instrument_class. Defaults keep any other construction site valid.
    admitted_by_provenance: dict[str, int] = field(default_factory=dict)
    admitted_by_instrument_class: dict[str, int] = field(default_factory=dict)
    # W3A-1 schema-mapping honesty: which pnl field each admitted row was costed from
    # (gross_pnl vs pnl fallback, D3), and how many admitted rows carry an inverted
    # exit<entry timestamp (kept but flagged; the walk-forward axis orders robustly).
    admitted_pnl_field_counts: dict[str, int] = field(default_factory=dict)
    inverted_duration_admitted: int = 0
    # W3A-3 (D2): read-only reconciliation vs SCR's effective set (runtime/scr_state.json).
    # None when the cross-check was not run; a dict surfacing the admitted-vs-effective delta
    # so a divergence between the two scorekeepers is loud, never silent.
    scr_reconciliation: Optional[dict[str, Any]] = None
    # W3A-5 (D4): sample-regime partition at the frozen exit-overlay boundary. pre/post are
    # DIFFERENT populations; stamped on every manifest so a report can never present a pooled
    # cross-era count as if it were one population.
    era_partition: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "since": self.since,
            "until": self.until,
            "input_files": [dict(f) for f in self.input_files],
            "rows_read": self.rows_read,
            "rows_in_window": self.rows_in_window,
            "out_of_window": self.out_of_window,
            "malformed": self.malformed,
            "duplicate": self.duplicate,
            "admitted": self.admitted,
            "excluded_by_reason": dict(self.excluded_by_reason),
            "date_range_admitted": self.date_range_admitted,
            "strategies_admitted": dict(self.strategies_admitted),
            "notes": list(self.notes),
            "admitted_by_provenance": dict(self.admitted_by_provenance),
            "admitted_by_instrument_class": dict(self.admitted_by_instrument_class),
            "admitted_pnl_field_counts": dict(self.admitted_pnl_field_counts),
            "inverted_duration_admitted": self.inverted_duration_admitted,
            "scr_reconciliation": self.scr_reconciliation,
            "era_partition": dict(self.era_partition),
        }


@dataclass(frozen=True)
class AdapterResult:
    """The adapter's output: the admitted canonical trades + the manifest."""

    admitted: list[AdmittedTrade]
    manifest: AdapterManifest

    def fill_mappings(self) -> list[dict[str, Any]]:
        """The downstream (cost-model) mappings for every admitted trade — the ONLY seam
        by which Stage-2 fills reach the scorer, and only trusted rows are ever here."""
        return [t.to_fill_mapping() for t in self.admitted]


# --------------------------------------------------------------------------- #
# The pure core: records → admitted + counters (no filesystem; unit-testable).
# --------------------------------------------------------------------------- #
def _sort_key(t: AdmittedTrade) -> tuple[str, int, str]:
    prov = t.provenance
    seq = prov.get("sequence_id")
    seq_i = seq if isinstance(seq, int) and not isinstance(seq, bool) else -1
    return (
        str(prov.get("entry_time_utc") or ""),
        seq_i,
        str(prov.get("record_hash") or ""),
    )


def adapt_records(
    records: Iterable[tuple[Optional[Mapping[str, Any]], str]],
    *,
    since: Optional[str] = None,
    until: Optional[str] = None,
    quarantined_hashes: "Optional[frozenset[str]]" = None,
    quarantined_fill_ids: "Optional[frozenset[str]]" = None,
    include_futures: bool = False,
) -> tuple[list[AdmittedTrade], dict[str, int]]:
    """Filter + map an iterable of ``(record_or_None, source_file)`` into admitted trades.

    A ``None`` record is a line that failed to parse (counted as ``malformed``). Date window
    ``[since, until]`` (inclusive ``YYYY-MM-DD`` strings; either may be ``None`` for open-ended)
    is applied first — out-of-window rows are neither admitted nor trust-counted. Trust
    exclusions (fail-closed) and malformed-but-in-window rows are tallied. An admitted row
    whose ``record_hash`` was already admitted is dropped as a ``duplicate`` (row-level
    idempotency: the same content hash must not be scored twice, e.g. if it appeared in two
    matched files); rows lacking a hash cannot be de-duplicated and are admitted. Returns the
    stable-sorted admitted list and a counters dict. Never raises on bad data.

    ``quarantined_hashes`` / ``quarantined_fill_ids`` (W2A-1) are the operator manifest pins
    from :func:`load_quarantine_pins`. An in-window row matched by :func:`is_quarantined` is
    dropped as ``excluded:quarantined`` BEFORE the trust gate — the same authority SCR honours
    — so a genuine-looking round-trip pinned by the operator (e.g. a PFF1 double-book phantom)
    cannot reach the scorer even though it carries no in-band untrust marker. Both default to
    ``None`` (no pins), keeping output byte-identical to pre-W2A for every existing caller.

    Structural guarantee: a row reaches ``admitted`` ONLY on the ``trust_exclusion(...) is
    None`` branch (the sole append site) — THIS is what makes the adapter unable to feed an
    untrusted row into the scorer. The post-loop re-run of :func:`trust_exclusion` on each
    admitted record is a secondary tripwire that catches a *non-deterministic gate* (a future
    edit making the gate stateful/random so it admits then excludes the same input); it raises
    rather than shipping such a row. It is an ``assert`` (advisory under ``python -O``); the
    append-gate above is the non-advisory guarantee.
    """
    q_hashes: "frozenset[str]" = quarantined_hashes or frozenset()
    q_fill_ids: "frozenset[str]" = quarantined_fill_ids or frozenset()
    admitted_pairs: list[tuple[AdmittedTrade, Mapping[str, Any]]] = []
    seen_hashes: set[str] = set()
    counters: dict[str, int] = {
        "rows_read": 0,
        "rows_in_window": 0,
        "out_of_window": 0,
        "malformed": 0,
        "duplicate": 0,
        "admitted": 0,
        "inverted_duration": 0,
    }
    for reason in EXCLUSION_REASONS:
        counters[f"excluded:{reason}"] = 0

    for record, source_file in records:
        counters["rows_read"] += 1
        if record is None or not isinstance(record, Mapping):
            counters["malformed"] += 1
            continue

        payload = _payload(record)
        # D6 (W3A-2, corrected W3A-7): exclude rows carrying an EXPLICIT non-lap schema
        # (a raw-fill / bars / state schema pointed at by mistake). A row with NO schema_version
        # is NOT excluded here — legitimate crypto laps written by
        # chad/analytics/trade_result_logger.py carry no schema_version yet ARE real closed laps
        # (verified: the Kraken TrustedFillEngine's realized rows). Those fall through to the
        # trust + malformed gates, which are what actually distinguish a lap from a non-lap on
        # the real ledger (the 2148 no-schema Kraken validate_only rows are caught there, not
        # here). So this gate only fires on a PRESENT, non-closed-lap schema_version.
        sv = str(payload.get("schema_version") or "")
        if sv and not _is_closed_lap_schema(sv):
            counters["excluded:non_closed_trade"] += 1
            continue
        row_date = _row_date(payload, record)
        if (since is not None and (row_date is None or row_date < since)) or (
            until is not None and (row_date is None or row_date > until)
        ):
            counters["out_of_window"] += 1
            continue
        counters["rows_in_window"] += 1

        # W2A-1: operator quarantine pin (manifest record_hash / fill_id) — checked BEFORE
        # the trust gate because a pinned row is genuine-looking (no in-band untrust marker)
        # yet operator-invalidated; this is the only exclusion that would otherwise ADMIT.
        if q_hashes or q_fill_ids:
            if is_quarantined(record, q_hashes, q_fill_ids):
                counters["excluded:quarantined"] += 1
                continue

        reason = trust_exclusion(record)
        if reason is not None:
            counters[f"excluded:{reason}"] += 1
            continue

        try:
            trade = _to_admitted(record, source_file)
        except _MalformedRow:
            counters["malformed"] += 1
            continue

        # D2 (W3A-2): futures rows are Bug-B-contaminated (untrusted PnL) — excluded from
        # scoring by default, mirroring SCR's is_futures_row exclusion. Re-enable only once the
        # Bug-B book disposition lands (include_futures=True).
        if not include_futures and trade.instrument_class == InstrumentClass.FUT.value:
            counters["excluded:futures_bug_b"] += 1
            continue

        rh = record.get("record_hash")
        if isinstance(rh, str) and rh:
            if rh in seen_hashes:
                counters["duplicate"] += 1
                continue
            seen_hashes.add(rh)
        if trade.provenance.get("inverted_duration"):
            counters["inverted_duration"] += 1
        admitted_pairs.append((trade, record))

    admitted_pairs.sort(key=lambda pair: _sort_key(pair[0]))
    admitted = [t for t, _ in admitted_pairs]
    counters["admitted"] = len(admitted)

    # Secondary tripwire (see docstring): a deterministic pure gate re-runs identically, so
    # this only fires if the gate became non-deterministic. The real guarantee is the append
    # gate above (a row is only ever added when trust_exclusion(...) returned None).
    leaked = [rec for _, rec in admitted_pairs if trust_exclusion(rec) is not None]
    assert not leaked, f"trust-filter leak: {len(leaked)} admitted rows are excludable"
    if q_hashes or q_fill_ids:
        q_leaked = [
            rec for _, rec in admitted_pairs if is_quarantined(rec, q_hashes, q_fill_ids)
        ]
        assert not q_leaked, f"quarantine leak: {len(q_leaked)} admitted rows are pinned"

    return admitted, counters


# --------------------------------------------------------------------------- #
# Filesystem I/O.
# --------------------------------------------------------------------------- #
def iter_ledger_files(trades_dir: Path) -> list[Path]:
    """The canonical daily ledgers ``trade_history_YYYYMMDD.ndjson`` under ``trades_dir``.

    Sorted by name (deterministic). Backups (``.scr_reset_bak`` / ``.pre_*_bak`` / any
    suffix) are excluded by the strict regex so a reset-stamped or pre-fix copy of a day can
    never be double-counted alongside the live ledger.
    """
    if not trades_dir.is_dir():
        return []
    return sorted(
        (p for p in trades_dir.iterdir() if p.is_file() and _LEDGER_RE.match(p.name)),
        key=lambda p: p.name,
    )


_MANIFEST_GLOB = "quarantine_manifest_*.json"


def load_quarantine_pins(
    runtime_dir: Optional[Path],
) -> tuple[frozenset[str], frozenset[str], list[str]]:
    """Return ``(record_hashes, fill_ids, consulted_manifest_names)`` from the operator
    quarantine manifests under ``runtime_dir``.

    Reads every ``runtime/quarantine_manifest_*.json`` as TEXT (json.loads) and unions
    ``invalid_trades[].record_hash`` + ``invalid_fills[].fill_id`` — the manifest-only
    subset of ``chad.utils.quarantine.get_exclusion_sets`` (the runtime fills-scan / FIFO-lot
    / sidecar legs are deliberately excluded: they require importing runtime/execution readers
    that would break the harness isolation contract, and the ghost-scrub pins by
    ``record_hash`` regardless). Stdlib-only, no chad import.

    Fail-safe (mirrors :func:`chad.utils.quarantine.get_quarantine_sets`): a missing dir,
    unreadable file, or bad shape contributes nothing rather than raising — a scrub manifest
    a reviewer forgot to fix must never crash the offline gate. ``runtime_dir is None`` (the
    default in :func:`adapt_records`) yields empty sets, so the adapter's behaviour is
    byte-identical to pre-W2A output unless a runtime dir is supplied.
    """
    if runtime_dir is None:
        return frozenset(), frozenset(), []
    rdir = Path(runtime_dir)
    if not rdir.is_dir():
        return frozenset(), frozenset(), []
    hashes: set[str] = set()
    fill_ids: set[str] = set()
    consulted: list[str] = []
    for manifest_path in sorted(rdir.glob(_MANIFEST_GLOB)):
        try:
            doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(doc, Mapping):
            continue
        consulted.append(manifest_path.name)
        for entry in doc.get("invalid_trades") or []:
            if isinstance(entry, Mapping):
                v = entry.get("record_hash")
                if isinstance(v, str) and v:
                    hashes.add(v)
            elif isinstance(entry, str) and entry:
                hashes.add(entry)
        for entry in doc.get("invalid_fills") or []:
            if isinstance(entry, Mapping):
                v = entry.get("fill_id")
                if isinstance(v, str) and v:
                    fill_ids.add(v)
            elif isinstance(entry, str) and entry:
                fill_ids.add(entry)
    return frozenset(hashes), frozenset(fill_ids), consulted


class ScrCrosscheckError(RuntimeError):
    """The SCR reconciliation cross-check was requested but ``runtime/scr_state.json`` is
    absent / unreadable / malformed. The operator granted this read explicitly as READ-only,
    **fail-loud on absence, never written** (D2) — so a missing SCR truth is an error, not a
    silent skip."""


def load_scr_effective(scr_state_path: Optional[Path]) -> tuple[Optional[int], Optional[str]]:
    """Read ``runtime/scr_state.json`` as TEXT (read-only) → ``(effective_trades, ts_utc)``.

    ``scr_state_path is None`` disables the cross-check → ``(None, None)`` (no read). When a
    path IS given it is READ (never written) and a missing / unreadable / malformed file
    raises :class:`ScrCrosscheckError` — fail-loud on absence (D2). This is the same one-way
    text-read shape as :func:`load_quarantine_pins`; no ``chad`` runtime module is imported, so
    the harness import-closure isolation (tests/validation/test_isolation.py) is preserved.
    """
    if scr_state_path is None:
        return (None, None)
    p = Path(scr_state_path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ScrCrosscheckError(f"scr_state.json unreadable at {p}: {exc}") from exc
    if not isinstance(doc, Mapping):
        raise ScrCrosscheckError(f"scr_state.json malformed at {p} (not a JSON object)")
    stats = doc.get("stats")
    eff = stats.get("effective_trades") if isinstance(stats, Mapping) else None
    if not isinstance(eff, int) or isinstance(eff, bool):
        raise ScrCrosscheckError(
            f"scr_state.json at {p} has no integer stats.effective_trades"
        )
    ts = doc.get("ts_utc")
    return (int(eff), str(ts) if ts is not None else None)


def _scr_reconciliation(
    admitted: Sequence[AdmittedTrade],
    excluded_by_reason: Mapping[str, int],
    scr_effective: int,
    scr_ts: Optional[str],
) -> dict[str, Any]:
    """Build the read-only reconciliation block: adapter admissions vs SCR effective set (D2)."""
    pnl_zero = sum(1 for t in admitted if t.gross_pnl == 0.0)
    return {
        "scr_effective_trades": scr_effective,
        "scr_state_ts": scr_ts,
        "adapter_admitted": len(admitted),
        "delta_admitted_minus_scr": len(admitted) - scr_effective,
        # SCR drops pnl==0 laps; the adapter keeps them as legitimate 0-return samples — a
        # KNOWN, reported component of the delta (never silent).
        "adapter_admitted_pnl_zero": pnl_zero,
        "adapter_excluded_by_reason": dict(excluded_by_reason),
        "note": (
            "Two INDEPENDENT scorekeepers. The adapter re-derives admissions from "
            "trade_history over [since, until] with its own fail-closed gate; SCR aggregates "
            "its effective set its own way over its own window. A nonzero delta is EXPECTED "
            "(differing windows; the adapter admits pnl==0 laps SCR drops and excludes "
            "futures/manual SCR also excludes). A LARGE unexplained delta warrants "
            "investigation — surfaced here so the two can never silently diverge."
        ),
    }


def load_ndjson(path: Path) -> Iterable[tuple[Optional[Mapping[str, Any]], str]]:
    """Yield ``(parsed_record_or_None, path_name)`` for every non-blank line in ``path``.

    A line that is not valid JSON (or not a JSON object) yields ``None`` so the caller counts
    it as malformed rather than crashing (SSOT: skip + count, never crash).
    """
    name = path.name
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (ValueError, TypeError):
                yield None, name
                continue
            yield (obj if isinstance(obj, dict) else None), name


class LedgerChainError(RuntimeError):
    """A ledger file's hash chain is broken (linkage / sequence / recomputed record_hash) —
    tamper or corruption. Raised fail-loud by :func:`run_adapter` when ``verify_chain=True``."""


def verify_ledger_chain(path: Path) -> list[str]:
    """Verify one daily ledger's per-file hash chain (D6). Returns a list of issue strings
    (empty ⇒ intact). Read-only; never raises on I/O.

    Each ``trade_history_YYYYMMDD.ndjson`` is its OWN chain starting at ``prev_hash="GENESIS"``
    (chad/execution/trade_closer.py:1011). For every row in file order this checks:
      * **linkage** — ``prev_hash`` equals the previous row's ``record_hash`` (first ==
        ``"GENESIS"``); a break means an inserted / reordered / deleted row;
      * **sequence** — ``sequence_id`` increments by exactly 1;
      * **recomputed record_hash** — for ``closed_trade.v1`` rows only, ``sha256(json.dumps(
        {payload, prev_hash, sequence_id, timestamp_utc}, sort_keys=True, default=str))`` must
        equal the stored ``record_hash`` (verified byte-exact on the real ledgers, 2026-07-22).
        Legacy/foreign-writer rows keep linkage but are NOT hash-recomputed (a different writer
        core), so they never false-positive — matching the ``schema_version`` admit gate.
    """
    issues: list[str] = []
    try:
        raw_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as exc:  # pragma: no cover - unreadable file
        return [f"{path.name}: unreadable ({exc})"]
    prev = "GENESIS"
    prev_seq: Optional[int] = None
    idx = 0
    for line in raw_lines:
        s = line.strip()
        if not s:
            continue
        try:
            rec = json.loads(s)
        except ValueError:
            issues.append(f"{path.name}[{idx}]: unparseable line")
            idx += 1
            continue
        if not isinstance(rec, dict):
            issues.append(f"{path.name}[{idx}]: not a JSON object")
            idx += 1
            continue
        if rec.get("prev_hash") != prev:
            issues.append(
                f"{path.name}[{idx}]: prev_hash {rec.get('prev_hash')!r} != expected {prev!r} "
                "(broken chain — inserted/reordered/deleted row)"
            )
        seq = rec.get("sequence_id")
        if isinstance(seq, int) and not isinstance(seq, bool):
            if prev_seq is not None and seq != prev_seq + 1:
                issues.append(f"{path.name}[{idx}]: sequence_id {seq} != {prev_seq} + 1")
            prev_seq = seq
        payload = rec.get("payload")
        rh = rec.get("record_hash")
        if isinstance(payload, Mapping) and payload.get("schema_version") == _CLOSED_TRADE_SCHEMA:
            core = {
                "payload": payload,
                "prev_hash": rec.get("prev_hash"),
                "sequence_id": rec.get("sequence_id"),
                "timestamp_utc": rec.get("timestamp_utc"),
            }
            recomputed = hashlib.sha256(
                json.dumps(core, sort_keys=True, default=str).encode()
            ).hexdigest()
            if recomputed != rh:
                issues.append(f"{path.name}[{idx}]: record_hash mismatch (tampered payload)")
        prev = rh if isinstance(rh, str) and rh else prev
        idx += 1
    return issues


def _date_range(admitted: Sequence[AdmittedTrade]) -> Optional[str]:
    dates = sorted(
        d for d in (str(t.provenance.get("entry_time_utc") or "")[:10] for t in admitted) if d
    )
    if not dates:
        return None
    return f"{dates[0]}..{dates[-1]}"


def run_adapter(
    *,
    trades_dir: Path,
    since: Optional[str] = None,
    until: Optional[str] = None,
    out_dir: Optional[Path] = None,
    generated_at: Optional[str] = None,
    runtime_dir: Optional[Path] = None,
    include_futures: bool = False,
    verify_chain: bool = False,
    scr_state_path: Optional[Path] = None,
) -> AdapterResult:
    """Read the on-box ledgers, apply the trust gate, and (optionally) write the artifacts.

    Reads every canonical ledger under ``trades_dir`` (read-only), filters to ``[since,
    until]``, and returns an :class:`AdapterResult`. When ``runtime_dir`` is given (W2A-1)
    the operator quarantine manifests under it are read as TEXT and their record_hash /
    fill_id pins drop the matching rows (reason ``quarantined``) — the same authority SCR
    honours, so both scorekeepers exclude the identical set; ``runtime_dir=None`` reads no
    manifest and is byte-identical to pre-W2A. When ``out_dir`` is given, writes
    ``stage2_trades_<since>_<until>.ndjson`` (stable-ordered admitted trades) and
    ``stage2_manifest_<since>_<until>.json`` under it. Deterministic in its inputs (ledgers +
    any consulted manifests, which are listed in the output manifest ``notes`` for audit);
    the only clock value is the manifest ``generated_at`` (defaulted to now, overridable).
    """
    q_hashes, q_fill_ids, consulted_manifests = load_quarantine_pins(runtime_dir)
    files = iter_ledger_files(trades_dir)

    # D6 (W3A-2): fail-loud on a broken hash chain before admitting anything from a
    # tampered/corrupt ledger. Off by default (unit tests use synthetic hashes); the CLIs
    # turn it on for real on-box ledgers.
    if verify_chain:
        chain_issues: list[str] = []
        for fp in files:
            chain_issues.extend(verify_ledger_chain(fp))
        if chain_issues:
            raise LedgerChainError(
                "ledger hash-chain verification FAILED (D6) — refusing to admit from a "
                "tampered/corrupt ledger:\n  " + "\n  ".join(chain_issues[:20])
            )

    input_files: list[dict[str, Any]] = []
    per_file_rows: dict[str, int] = {}

    def _stream() -> Iterable[tuple[Optional[Mapping[str, Any]], str]]:
        for fp in files:
            n = 0
            for rec, name in load_ndjson(fp):
                n += 1
                yield rec, name
            per_file_rows[fp.name] = n

    admitted, counters = adapt_records(
        _stream(),
        since=since,
        until=until,
        quarantined_hashes=q_hashes,
        quarantined_fill_ids=q_fill_ids,
        include_futures=include_futures,
    )

    for fp in files:
        input_files.append(
            {"path": str(fp), "sha256": _sha256_file(fp), "rows": per_file_rows.get(fp.name, 0)}
        )

    # FIXED schema: emit every EXCLUSION_REASONS key (0 when none), so a downstream consumer
    # can rely on the full key set always being present.
    excluded_by_reason = {
        reason: counters[f"excluded:{reason}"] for reason in EXCLUSION_REASONS
    }
    # W3A-3 (D2): read-only reconciliation vs SCR's effective set (fail-loud on absence when a
    # path is supplied). None when the cross-check is disabled (scr_state_path is None).
    scr_reconciliation: Optional[dict[str, Any]] = None
    if scr_state_path is not None:
        scr_eff, scr_ts = load_scr_effective(scr_state_path)
        if scr_eff is not None:
            scr_reconciliation = _scr_reconciliation(
                admitted, excluded_by_reason, scr_eff, scr_ts
            )
    strategies: dict[str, int] = {}
    by_provenance: dict[str, int] = {}
    by_instrument: dict[str, int] = {}
    by_pnl_field: dict[str, int] = {}
    by_era: dict[str, int] = {}
    by_strategy_era: dict[str, int] = {}
    for t in admitted:
        strategies[t.strategy] = strategies.get(t.strategy, 0) + 1
        prov = str(t.provenance.get("provenance") or "unspecified")
        by_provenance[prov] = by_provenance.get(prov, 0) + 1
        by_instrument[t.instrument_class] = by_instrument.get(t.instrument_class, 0) + 1
        pfld = str(t.provenance.get("pnl_field") or "pnl")
        by_pnl_field[pfld] = by_pnl_field.get(pfld, 0) + 1
        era = era_of(_admitted_exit_date(t))
        by_era[era] = by_era.get(era, 0) + 1
        by_strategy_era[f"{t.strategy}|{era}"] = by_strategy_era.get(f"{t.strategy}|{era}", 0) + 1
    era_partition = {
        "boundary": EXIT_OVERLAY_BOUNDARY,
        "counts_by_era": dict(sorted(by_era.items())),
        "counts_by_strategy_era": dict(sorted(by_strategy_era.items())),
        "pooled": False,
        "note": (
            "D4: pre/post exit-overlay eras are DIFFERENT data-generating processes "
            "(pre = reconciliation/adopt closes; post = engine-driven overlay closes). They "
            "are reported SEPARATELY and never pooled into one headline verdict; pooling is "
            "only ever an explicitly-labeled sensitivity view."
        ),
    }

    if generated_at is None:
        generated_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    notes = [
        "Stage-2 real-fill ingest (SSOT §1.3 / Part 6, Phase 6). Admitted rows feed the "
        "identical Stage-1 scoring spine + cost haircut; this adapter computes no verdict.",
        "Trust gate is fail-closed: pnl_untrusted / validate_only / $100-placeholder / "
        "broker-rejected / non-canonical-status rows cannot reach the scorer.",
        "Prices set to the realized fill (two-leg haircut on traded notional); realized PnL "
        "carried authoritatively as gross_pnl, not reconstructed.",
        "Evidence classes are separated (admitted_by_provenance / _by_instrument_class): "
        "SIMULATED_AGAINST_LIVE_TICKS crypto rows are counted apart from broker-confirmed "
        "fills so no verdict silently mixes simulated and broker-confirmed evidence.",
    ]
    if consulted_manifests:
        notes.append(
            "Operator quarantine honoured (W2A-1): pinned record_hash/fill_id rows dropped "
            "as excluded_by_reason['quarantined'] — same authority SCR uses. Manifests "
            "consulted: " + ", ".join(consulted_manifests) + "."
        )
    elif runtime_dir is not None:
        notes.append(
            "Operator quarantine checked (W2A-1): no quarantine_manifest_*.json present under "
            "the runtime dir; nothing pinned."
        )
    manifest = AdapterManifest(
        schema_version=SCHEMA_VERSION,
        generated_at=generated_at,
        since=since,
        until=until,
        input_files=input_files,
        rows_read=counters["rows_read"],
        rows_in_window=counters["rows_in_window"],
        out_of_window=counters["out_of_window"],
        malformed=counters["malformed"],
        duplicate=counters["duplicate"],
        admitted=counters["admitted"],
        excluded_by_reason=excluded_by_reason,
        date_range_admitted=_date_range(admitted),
        strategies_admitted=dict(sorted(strategies.items())),
        notes=notes,
        admitted_by_provenance=dict(sorted(by_provenance.items())),
        admitted_by_instrument_class=dict(sorted(by_instrument.items())),
        admitted_pnl_field_counts=dict(sorted(by_pnl_field.items())),
        inverted_duration_admitted=counters.get("inverted_duration", 0),
        scr_reconciliation=scr_reconciliation,
        era_partition=era_partition,
    )
    result = AdapterResult(admitted=admitted, manifest=manifest)

    if out_dir is not None:
        _write_artifacts(result, out_dir, since=since, until=until)
    return result


def _window_tag(since: Optional[str], until: Optional[str]) -> str:
    return f"{since or 'open'}_{until or 'open'}".replace(":", "-")


def _write_artifacts(
    result: AdapterResult, out_dir: Path, *, since: Optional[str], until: Optional[str]
) -> tuple[Path, Path]:
    """Write the admitted-trades ndjson + manifest json under ``out_dir`` (created if absent)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _window_tag(since, until)
    trades_path = out_dir / f"stage2_trades_{tag}.ndjson"
    manifest_path = out_dir / f"stage2_manifest_{tag}.json"
    with trades_path.open("w", encoding="utf-8") as fh:
        for t in result.admitted:
            fh.write(json.dumps(t.to_dict(), sort_keys=True, ensure_ascii=True) + "\n")
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(result.manifest.to_dict(), fh, sort_keys=True, ensure_ascii=True, indent=2)
        fh.write("\n")
    return trades_path, manifest_path


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def _print_manifest(manifest: AdapterManifest) -> None:
    print("=" * 72)
    print("CHAD Edge-Validation Harness — Stage-2 trade-log adapter (SSOT Part 6, Phase 6)")
    print("=" * 72)
    print(f"window: {manifest.since or '(open)'} .. {manifest.until or '(open)'}")
    print(f"input files: {len(manifest.input_files)}")
    print(
        f"rows_read={manifest.rows_read}  in_window={manifest.rows_in_window}  "
        f"out_of_window={manifest.out_of_window}  malformed={manifest.malformed}  "
        f"duplicate={manifest.duplicate}"
    )
    print(f"ADMITTED (trusted, scorer-ready): {manifest.admitted}")
    if manifest.excluded_by_reason:
        for reason, n in sorted(manifest.excluded_by_reason.items()):
            print(f"   excluded[{reason}] = {n}")
    else:
        print("   excluded: (none in window)")
    if manifest.strategies_admitted:
        print(f"strategies admitted: {manifest.strategies_admitted}")
    print(f"admitted date range: {manifest.date_range_admitted or '(none)'}")
    print("=" * 72)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: ``python -m chad.validation.trade_log_adapter --since DATE [--until DATE]``.

    Reads the on-box ledger read-only, applies the fail-closed trust gate, writes the two
    artifacts, and prints the manifest. Never writes ``runtime/`` or ``ready_for_live``.
    Exit 0 on a completed run (admitted=0 is a valid, honest outcome), 2 on a usage error.
    """
    parser = argparse.ArgumentParser(
        prog="python -m chad.validation.trade_log_adapter",
        description="CHAD Stage-2 real-trade ingest adapter (feeds the Stage-1 verdict engine).",
    )
    parser.add_argument("--since", default=None, help="earliest UTC date to admit (YYYY-MM-DD, inclusive)")
    parser.add_argument("--until", default=None, help="latest UTC date to admit (YYYY-MM-DD, inclusive)")
    parser.add_argument("--trades-dir", default=None, help="ledger dir (default: <repo>/data/trades)")
    parser.add_argument("--out-dir", default=None, help="artifact dir (default: <repo>/edge_reports/stage2)")
    parser.add_argument("--now", default=None, help="override the manifest timestamp (ISO 8601)")
    parser.add_argument("--repo-root", default=None, help="repo root (default: inferred)")
    parser.add_argument("--runtime-dir", default=None,
                        help="runtime dir holding quarantine_manifest_*.json (default: <repo>/runtime)")
    parser.add_argument("--no-quarantine", action="store_true",
                        help="ignore operator quarantine manifests (admit pinned rows; default honours them)")
    parser.add_argument("--include-futures", action="store_true",
                        help="admit futures rows (default excludes them as Bug-B-contaminated, D2)")
    parser.add_argument("--no-verify-chain", action="store_true",
                        help="skip ledger hash-chain verification (default verifies + fails loud, D6)")
    parser.add_argument("--scr-crosscheck", action="store_true",
                        help="reconcile admitted count vs runtime/scr_state.json effective_trades "
                             "(read-only; fails loud if absent, D2)")
    parser.add_argument("--scr-state", default=None,
                        help="scr_state.json path for --scr-crosscheck (default: <repo>/runtime/scr_state.json)")
    args = parser.parse_args(argv)

    for name in ("since", "until"):
        val = getattr(args, name)
        if val is not None and not re.match(r"^\d{4}-\d{2}-\d{2}$", val):
            print(f"error: --{name} must be YYYY-MM-DD, got {val!r}")
            return 2
    if args.since and args.until and args.since > args.until:
        print(f"error: --since ({args.since}) is after --until ({args.until})")
        return 2

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[2]
    trades_dir = Path(args.trades_dir).resolve() if args.trades_dir else repo_root / "data" / "trades"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else repo_root / "edge_reports" / "stage2"
    if args.no_quarantine:
        runtime_dir: Optional[Path] = None
    elif args.runtime_dir:
        runtime_dir = Path(args.runtime_dir).resolve()
    else:
        runtime_dir = repo_root / "runtime"

    if args.scr_crosscheck:
        scr_state_path: Optional[Path] = (
            Path(args.scr_state).resolve() if args.scr_state else repo_root / "runtime" / "scr_state.json"
        )
    else:
        scr_state_path = None

    try:
        result = run_adapter(
            trades_dir=trades_dir,
            since=args.since,
            until=args.until,
            out_dir=out_dir,
            generated_at=args.now,
            runtime_dir=runtime_dir,
            include_futures=args.include_futures,
            verify_chain=not args.no_verify_chain,
            scr_state_path=scr_state_path,
        )
    except LedgerChainError as exc:
        # D6 fail-loud: refuse to emit an artifact from a tampered/corrupt ledger.
        print(f"LEDGER CHAIN VERIFICATION FAILED (D6): {exc}")
        return 2
    except ScrCrosscheckError as exc:
        # D2 fail-loud: the SCR cross-check was requested but SCR truth is absent/unreadable.
        print(f"SCR CROSS-CHECK FAILED (D2): {exc}")
        return 2
    _print_manifest(result.manifest)
    print(f"artifacts written under: {out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
