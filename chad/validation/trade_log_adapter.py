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
    "AdmittedTrade",
    "AdapterManifest",
    "AdapterResult",
    "trust_exclusion",
    "classify_instrument",
    "is_placeholder_fill",
    "adapt_records",
    "iter_ledger_files",
    "load_ndjson",
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
)

_LEDGER_RE = re.compile(r"^trade_history_(\d{8})\.ndjson$")


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


# --------------------------------------------------------------------------- #
# Instrument classification + placeholder fingerprint.
# --------------------------------------------------------------------------- #
def classify_instrument(payload: Mapping[str, Any], extra: Mapping[str, Any]) -> InstrumentClass:
    """Resolve the InstrumentClass from explicit asset_class, broker, then symbol (documented).

    Precedence: an explicit ``asset_class`` wins; else a ``kraken*`` broker ⇒ CRYPTO; else a
    known futures root ⇒ FUT; else a ``*-USD`` / ``*USD`` crypto pair ⇒ CRYPTO; else STK.
    """
    ac = str(payload.get("asset_class") or extra.get("asset_class") or "").strip().lower()
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
    if _truthy(extra.get("pnl_untrusted")) or "pnl_untrusted" in tags:
        return "pnl_untrusted"

    return None


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
    pnl = _finite_number(payload.get("pnl"))
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

    mult = _finite_number(extra.get("multiplier") if extra.get("multiplier") is not None
                          else payload.get("multiplier"))
    if mult is None or mult <= 0.0:
        mult = 1.0
    inst = classify_instrument(payload, extra)
    strategy = str(payload.get("strategy") or "unattributed").strip() or "unattributed"

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

    Structural guarantee: a row reaches ``admitted`` ONLY on the ``trust_exclusion(...) is
    None`` branch (the sole append site) — THIS is what makes the adapter unable to feed an
    untrusted row into the scorer. The post-loop re-run of :func:`trust_exclusion` on each
    admitted record is a secondary tripwire that catches a *non-deterministic gate* (a future
    edit making the gate stateful/random so it admits then excludes the same input); it raises
    rather than shipping such a row. It is an ``assert`` (advisory under ``python -O``); the
    append-gate above is the non-advisory guarantee.
    """
    admitted_pairs: list[tuple[AdmittedTrade, Mapping[str, Any]]] = []
    seen_hashes: set[str] = set()
    counters: dict[str, int] = {
        "rows_read": 0,
        "rows_in_window": 0,
        "out_of_window": 0,
        "malformed": 0,
        "duplicate": 0,
        "admitted": 0,
    }
    for reason in EXCLUSION_REASONS:
        counters[f"excluded:{reason}"] = 0

    for record, source_file in records:
        counters["rows_read"] += 1
        if record is None or not isinstance(record, Mapping):
            counters["malformed"] += 1
            continue

        payload = _payload(record)
        row_date = _row_date(payload, record)
        if (since is not None and (row_date is None or row_date < since)) or (
            until is not None and (row_date is None or row_date > until)
        ):
            counters["out_of_window"] += 1
            continue
        counters["rows_in_window"] += 1

        reason = trust_exclusion(record)
        if reason is not None:
            counters[f"excluded:{reason}"] += 1
            continue

        try:
            trade = _to_admitted(record, source_file)
        except _MalformedRow:
            counters["malformed"] += 1
            continue

        rh = record.get("record_hash")
        if isinstance(rh, str) and rh:
            if rh in seen_hashes:
                counters["duplicate"] += 1
                continue
            seen_hashes.add(rh)
        admitted_pairs.append((trade, record))

    admitted_pairs.sort(key=lambda pair: _sort_key(pair[0]))
    admitted = [t for t, _ in admitted_pairs]
    counters["admitted"] = len(admitted)

    # Secondary tripwire (see docstring): a deterministic pure gate re-runs identically, so
    # this only fires if the gate became non-deterministic. The real guarantee is the append
    # gate above (a row is only ever added when trust_exclusion(...) returned None).
    leaked = [rec for _, rec in admitted_pairs if trust_exclusion(rec) is not None]
    assert not leaked, f"trust-filter leak: {len(leaked)} admitted rows are excludable"

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
) -> AdapterResult:
    """Read the on-box ledgers, apply the trust gate, and (optionally) write the artifacts.

    Reads every canonical ledger under ``trades_dir`` (no ``runtime/`` access, read-only),
    filters to ``[since, until]``, and returns an :class:`AdapterResult`. When ``out_dir`` is
    given, writes ``stage2_trades_<since>_<until>.ndjson`` (stable-ordered admitted trades)
    and ``stage2_manifest_<since>_<until>.json`` under it. Deterministic in its inputs; the
    only clock value is the manifest ``generated_at`` (defaulted to now, overridable).
    """
    files = iter_ledger_files(trades_dir)
    input_files: list[dict[str, Any]] = []
    per_file_rows: dict[str, int] = {}

    def _stream() -> Iterable[tuple[Optional[Mapping[str, Any]], str]]:
        for fp in files:
            n = 0
            for rec, name in load_ndjson(fp):
                n += 1
                yield rec, name
            per_file_rows[fp.name] = n

    admitted, counters = adapt_records(_stream(), since=since, until=until)

    for fp in files:
        input_files.append(
            {"path": str(fp), "sha256": _sha256_file(fp), "rows": per_file_rows.get(fp.name, 0)}
        )

    # FIXED schema: emit every EXCLUSION_REASONS key (0 when none), so a downstream consumer
    # can rely on the full key set always being present.
    excluded_by_reason = {
        reason: counters[f"excluded:{reason}"] for reason in EXCLUSION_REASONS
    }
    strategies: dict[str, int] = {}
    by_provenance: dict[str, int] = {}
    by_instrument: dict[str, int] = {}
    for t in admitted:
        strategies[t.strategy] = strategies.get(t.strategy, 0) + 1
        prov = str(t.provenance.get("provenance") or "unspecified")
        by_provenance[prov] = by_provenance.get(prov, 0) + 1
        by_instrument[t.instrument_class] = by_instrument.get(t.instrument_class, 0) + 1

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

    result = run_adapter(
        trades_dir=trades_dir,
        since=args.since,
        until=args.until,
        out_dir=out_dir,
        generated_at=args.now,
    )
    _print_manifest(result.manifest)
    print(f"artifacts written under: {out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
