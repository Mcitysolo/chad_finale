"""chad/validation/bar_audit.py — Phase 0 forensic bar-corpus data-quality audit.

Read-only, offline, deterministic auditor for the daily bar corpus
(``data/bars/1d/*.json``). It answers a single question per symbol: *is this
series clean enough to backtest on?* — and never modifies a bar file, never
touches the network, a broker, ``runtime/`` state, or any strategy/risk module.

Design intent (SSOT ``docs/CHAD_EDGE_VALIDATION_HARNESS_DESIGN_v1.1.md`` §Phase 0,
§5): every later harness report embeds this structure as its *data-quality
section*, so the output is a reusable, serialisable dataclass tree — not a
one-off print. A ``PASS`` here is a pre-registered minimum for any verdict
stricter than ``INSUFFICIENT_DATA`` (SSOT §4.3).

Per-symbol checks (each contributes findings; symbol status = worst finding):
  1. SCHEMA        — required OHLCV fields present with numeric types; ts parseable.
  2. OHLC sanity   — high>=max(open,close), low<=min(open,close), high>=low, prices>0.
  3. GAPS          — missing trading sessions in the date sequence (weekends/holidays
                     expected); reports count + largest gap.
  4. STALE prints  — runs of identical closes and/or zero-volume bars.
  5. DUPLICATES    — repeated timestamps (hard integrity violation → FAIL).
  6. ADJUSTMENT    — ~2x/0.5x single-day close jumps → SUSPECTED_UNADJUSTED (report only).
  7. FX PROVENANCE — quote-currency ambiguity / CAD-USD mix (SSOT §5). Detection and
                     report only; this auditor never converts. Any conversion elsewhere
                     MUST use :data:`chad.constants.fx.USDCAD_CONVERSION_CONSTANT`.
  8. COVERAGE      — first/last date, bar count; thin history; feed staleness vs corpus.

Severity model: findings are ``WARN`` or ``FAIL``; a symbol with no findings is
``CLEAN``. ``FAIL`` = the series is unsafe to backtest as-is (malformed rows,
OHLC violations, duplicate timestamps, unreadable file). ``WARN`` = usable but
caveated (gaps, stale runs, suspected-unadjusted jumps, thin/stale coverage, FX
provenance).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Final, Mapping, Optional, Sequence

# Canonical FX constant (SSOT §5). ``chad.constants.fx`` is a pure, side-effect-free
# constants module (imports only ``__future__``), so importing it keeps this
# package's transitive closure free of live-loop / broker / runtime deps.
from chad.constants.fx import USDCAD_CONVERSION_CONSTANT

__all__ = [
    "AuditConfig",
    "CorpusAudit",
    "Finding",
    "Status",
    "SymbolAudit",
    "audit_bar_file",
    "audit_corpus",
    "audit_symbol",
    "render_corpus_summary",
    "render_symbol_audit",
]

# --------------------------------------------------------------------------- #
# Schema contract (learned from the real corpus, not assumed).
# --------------------------------------------------------------------------- #
REQUIRED_BAR_KEYS: Final[tuple[str, ...]] = ("open", "high", "low", "close", "volume", "ts_utc")
_PRICE_KEYS: Final[tuple[str, ...]] = ("open", "high", "low", "close")

# Known IBKR forex-future roots whose *price is itself an FX cross-rate* (e.g.
# M6E = Micro EUR/USD). Such a series is not a USD-denominated asset price, so it
# is flagged for provenance even though the quote leg is USD.
_FX_FUTURE_ROOTS: Final[frozenset[str]] = frozenset(
    {"M6E", "M6A", "M6B", "M6C", "M6J", "M6S", "6E", "6A", "6B", "6C", "6J", "6S"}
)
# Tokens that signal a non-USD (CAD) quote leg — none in the current corpus, but
# the check is written so a CAD-quoted symbol added later is caught, not missed.
_CAD_SUFFIXES: Final[tuple[str, ...]] = (".TO", ".V", ".CN", ".NE")


# --------------------------------------------------------------------------- #
# Tunable thresholds — echoed in both the JSON record (`config` block) and the
# rendered summary so the thresholds that produced a verdict are reproducible.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AuditConfig:
    """Deterministic thresholds for the data-quality checks.

    All defaults are conservative; a later harness phase may hash-freeze a chosen
    instance (SSOT §3.2). No default depends on wall-clock time — the audit is a
    pure function of its inputs.
    """

    stale_close_run: int = 5
    """>= this many consecutive identical closes → stale-print WARN."""
    zero_volume_run: int = 5
    """>= this many consecutive zero-volume bars → stale-print WARN."""
    gap_session_threshold: int = 4
    """>= this many missing sessions in one gap → unusual-gap WARN."""
    split_ratio_high: float = 1.9
    """close[i]/close[i-1] >= this → suspected unadjusted split (report only)."""
    split_ratio_low: float = 0.55
    """close[i]/close[i-1] <= this → suspected unadjusted split (report only)."""
    thin_history_bars: int = 90
    """< this many bars → thin-history coverage WARN."""
    coverage_lag_days: int = 10
    """last bar this many calendar days behind the corpus max → stale-feed WARN."""
    max_examples: int = 8
    """cap on example rows carried in a finding's detail (counts stay exact)."""


DEFAULT_CONFIG: Final[AuditConfig] = AuditConfig()


# --------------------------------------------------------------------------- #
# Result model — serialisable, reusable by every later harness report.
# --------------------------------------------------------------------------- #
class Status(Enum):
    """Per-symbol / per-corpus data-quality status. Ordered CLEAN < WARN < FAIL."""

    CLEAN = "CLEAN"
    WARN = "WARN"
    FAIL = "FAIL"


_STATUS_RANK: Final[Mapping[Status, int]] = {Status.CLEAN: 0, Status.WARN: 1, Status.FAIL: 2}


def _worst(a: Status, b: Status) -> Status:
    """Return the more severe of two statuses (FAIL > WARN > CLEAN)."""
    return a if _STATUS_RANK[a] >= _STATUS_RANK[b] else b


@dataclass(frozen=True)
class Finding:
    """One data-quality issue. ``severity`` is always ``WARN`` or ``FAIL``."""

    check: str          # stable check name: "schema" | "ohlc" | "gaps" | ...
    code: str           # stable machine code, e.g. "OHLC_HIGH_LT_MAX_OC"
    severity: Status
    message: str        # human-readable one-liner
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class SymbolAudit:
    """Data-quality verdict for one symbol's bar series."""

    symbol: str
    source: Optional[str]
    timeframe: Optional[str]
    status: Status
    bar_count: int
    first_date: Optional[str]
    last_date: Optional[str]
    quote_currency: str
    currency_provenance: str
    findings: tuple[Finding, ...]
    metrics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "source": self.source,
            "timeframe": self.timeframe,
            "status": self.status.value,
            "bar_count": self.bar_count,
            "first_date": self.first_date,
            "last_date": self.last_date,
            "quote_currency": self.quote_currency,
            "currency_provenance": self.currency_provenance,
            "findings": [f.to_dict() for f in self.findings],
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class CorpusAudit:
    """Corpus-level roll-up over every audited symbol."""

    bars_dir: Optional[str]
    symbol_count: int
    clean: int
    warn: int
    fail: int
    corpus_last_date: Optional[str]
    status_by_symbol: Mapping[str, str]
    symbols: tuple[SymbolAudit, ...]
    corpus_findings: tuple[Finding, ...]
    config: AuditConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "bars_dir": self.bars_dir,
            "symbol_count": self.symbol_count,
            "clean": self.clean,
            "warn": self.warn,
            "fail": self.fail,
            "corpus_last_date": self.corpus_last_date,
            "status_by_symbol": dict(self.status_by_symbol),
            "symbols": [s.to_dict() for s in self.symbols],
            "corpus_findings": [f.to_dict() for f in self.corpus_findings],
            # asdict (not a hand-listed dict) so a later AuditConfig field can
            # never silently drop from the frozen-config record (SSOT §3.2).
            "config": asdict(self.config),
            "usdcad_conversion_constant": USDCAD_CONVERSION_CONSTANT,
        }


# Internal, fully-validated bar (schema + type valid; relationships not yet checked).
@dataclass(frozen=True)
class _ValidBar:
    index: int
    d: date
    open: float
    high: float
    low: float
    close: float
    volume: float


# --------------------------------------------------------------------------- #
# Small pure helpers.
# --------------------------------------------------------------------------- #
def _is_number(x: Any) -> bool:
    """True for a real numeric price/volume value; bool is rejected on purpose."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _parse_date(value: Any) -> Optional[date]:
    """Parse a ``YYYY-MM-DD`` (or ISO datetime) date; ``None`` if unparseable."""
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _sessions_between(prev: date, cur: date, continuous_market: bool) -> int:
    """Count expected trading sessions strictly between two dates (exclusive).

    For a 5-day market, weekend days are not sessions; a Fri→Mon pair therefore
    has zero missing sessions. For a continuous (crypto) market every calendar
    day is a session. Holidays are not modelled, so a single missing session may
    be an ordinary holiday — only a *run* of missing sessions is unusual.
    """
    span = (cur - prev).days
    if span <= 1:
        return 0
    if continuous_market:
        return span - 1
    missing = 0
    day = prev + timedelta(days=1)
    while day < cur:
        if day.weekday() < 5:  # Mon-Fri
            missing += 1
        day += timedelta(days=1)
    return missing


def _is_continuous_market(symbol: str, source: Optional[str]) -> bool:
    """Crypto trades 7 days/week; equities/futures/index do not."""
    s = symbol.upper()
    return source == "kraken" or s.endswith("-USD")


def classify_currency(symbol: str, source: Optional[str]) -> tuple[str, str]:
    """Infer (quote_currency, provenance) — no bar file declares a currency.

    Provenance values:
      * ``explicit_symbol_suffix`` — crypto ``X-USD`` names state their quote leg.
      * ``kraken_usd_pair`` — Kraken bars are USD-quoted.
      * ``index_points_no_currency`` — e.g. VIX is index points, not a currency.
      * ``fx_cross_rate`` — the price IS an FX cross (e.g. M6E EUR/USD); flagged.
      * ``cad_quoted`` — a CAD-quoted listing; flagged as CAD/USD-mixed.
      * ``assumed_usd_undeclared`` — inferred USD; honest but undeclared (corpus note).
    """
    s = symbol.upper()
    if s.endswith("-USD"):
        return ("USD", "explicit_symbol_suffix")
    if source == "kraken":
        return ("USD", "kraken_usd_pair")
    if source == "cboe":
        return ("USD", "index_points_no_currency")
    root = s.split("-")[0].split(".")[0]
    if root in _FX_FUTURE_ROOTS:
        return ("USD", "fx_cross_rate")
    if "CAD" in s or s.endswith(_CAD_SUFFIXES):
        return ("CAD", "cad_quoted")
    return ("USD", "assumed_usd_undeclared")


def _capped(items: Sequence[Any], limit: int) -> list[Any]:
    """Return at most ``limit`` items (for bounded finding detail)."""
    return list(items[:limit])


# --------------------------------------------------------------------------- #
# Individual checks. Each returns (findings, metrics-fragment).
# --------------------------------------------------------------------------- #
def _extract_valid_bars(
    bars: Sequence[Any], config: AuditConfig
) -> tuple[list[_ValidBar], list[Finding], dict[str, Any]]:
    """Check 1 — SCHEMA. Split rows into schema-valid bars and malformed findings."""
    valid: list[_ValidBar] = []
    malformed: list[dict[str, Any]] = []
    for i, row in enumerate(bars):
        if not isinstance(row, dict):
            malformed.append({"index": i, "reason": f"row is {type(row).__name__}, not object"})
            continue
        missing = [k for k in REQUIRED_BAR_KEYS if k not in row]
        if missing:
            malformed.append({"index": i, "reason": f"missing keys {missing}"})
            continue
        bad_types = [k for k in _PRICE_KEYS + ("volume",) if not _is_number(row[k])]
        if bad_types:
            malformed.append(
                {"index": i, "reason": f"non-numeric fields {bad_types}", "ts_utc": row.get("ts_utc")}
            )
            continue
        parsed = _parse_date(row["ts_utc"])
        if parsed is None:
            malformed.append({"index": i, "reason": f"unparseable ts_utc {row['ts_utc']!r}"})
            continue
        valid.append(
            _ValidBar(
                index=i,
                d=parsed,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
        )

    findings: list[Finding] = []
    if malformed:
        findings.append(
            Finding(
                check="schema",
                code="MALFORMED_ROWS",
                severity=Status.FAIL,
                message=f"{len(malformed)} malformed bar row(s)",
                detail={"count": len(malformed), "examples": _capped(malformed, config.max_examples)},
            )
        )
    metrics = {"malformed_count": len(malformed)}
    return valid, findings, metrics


def _check_ohlc(valid: Sequence[_ValidBar], config: AuditConfig) -> tuple[list[Finding], dict[str, Any]]:
    """Check 2 — OHLC sanity + positivity."""
    violations: list[dict[str, Any]] = []
    for b in valid:
        reasons: list[str] = []
        if not (b.open > 0 and b.high > 0 and b.low > 0 and b.close > 0):
            reasons.append("non_positive_price")
        if b.volume < 0:
            reasons.append("negative_volume")
        if b.high < max(b.open, b.close):
            reasons.append("high_lt_max_open_close")
        if b.low > min(b.open, b.close):
            reasons.append("low_gt_min_open_close")
        if b.high < b.low:
            reasons.append("high_lt_low")
        if reasons:
            violations.append(
                {
                    "index": b.index,
                    "ts_utc": b.d.isoformat(),
                    "reasons": reasons,
                    "ohlcv": [b.open, b.high, b.low, b.close, b.volume],
                }
            )
    findings: list[Finding] = []
    if violations:
        findings.append(
            Finding(
                check="ohlc",
                code="OHLC_VIOLATION",
                severity=Status.FAIL,
                message=f"{len(violations)} bar(s) violate OHLC sanity/positivity",
                detail={"count": len(violations), "examples": _capped(violations, config.max_examples)},
            )
        )
    return findings, {"ohlc_violation_count": len(violations)}


def _check_duplicates_and_order(
    valid: Sequence[_ValidBar], config: AuditConfig
) -> tuple[list[Finding], dict[str, Any]]:
    """Check 5 — DUPLICATES (FAIL) + non-monotonic ordering (WARN)."""
    findings: list[Finding] = []
    seen: dict[date, int] = {}
    dups: list[str] = []
    for b in valid:
        if b.d in seen:
            dups.append(b.d.isoformat())
        seen[b.d] = seen.get(b.d, 0) + 1
    if dups:
        findings.append(
            Finding(
                check="duplicates",
                code="DUPLICATE_TIMESTAMP",
                severity=Status.FAIL,
                message=f"{len(dups)} duplicate timestamp(s)",
                detail={"count": len(dups), "examples": _capped(sorted(set(dups)), config.max_examples)},
            )
        )

    non_monotonic = sum(
        1 for i in range(1, len(valid)) if valid[i].d < valid[i - 1].d
    )
    if non_monotonic:
        findings.append(
            Finding(
                check="ordering",
                code="NON_MONOTONIC_TIMESTAMPS",
                severity=Status.WARN,
                message=f"{non_monotonic} out-of-order timestamp step(s); derived checks use sorted order",
                detail={"count": non_monotonic},
            )
        )
    return findings, {"duplicate_count": len(dups), "non_monotonic_count": non_monotonic}


def _check_gaps(
    ordered: Sequence[_ValidBar], continuous: bool, config: AuditConfig
) -> tuple[list[Finding], dict[str, Any]]:
    """Check 3 — GAPS: missing trading sessions across consecutive bars."""
    gaps: list[dict[str, Any]] = []
    largest = 0
    largest_span: Optional[dict[str, Any]] = None
    for i in range(1, len(ordered)):
        prev, cur = ordered[i - 1], ordered[i]
        missing = _sessions_between(prev.d, cur.d, continuous)
        if missing >= 1:
            entry = {
                "from": prev.d.isoformat(),
                "to": cur.d.isoformat(),
                "missing_sessions": missing,
                "calendar_days": (cur.d - prev.d).days,
            }
            gaps.append(entry)
            if missing > largest:
                largest = missing
                largest_span = entry

    findings: list[Finding] = []
    unusual = [g for g in gaps if g["missing_sessions"] >= config.gap_session_threshold]
    if unusual:
        findings.append(
            Finding(
                check="gaps",
                code="UNUSUAL_GAP",
                severity=Status.WARN,
                message=(
                    f"{len(unusual)} gap(s) missing >= {config.gap_session_threshold} sessions; "
                    f"largest = {largest} sessions"
                ),
                detail={
                    "unusual_count": len(unusual),
                    "largest_gap_sessions": largest,
                    "largest_gap_span": largest_span,
                    "examples": _capped(unusual, config.max_examples),
                },
            )
        )
    metrics = {
        "gap_count": len(gaps),
        "largest_gap_sessions": largest,
        "largest_gap_span": largest_span,
    }
    return findings, metrics


def _check_stale(ordered: Sequence[_ValidBar], config: AuditConfig) -> tuple[list[Finding], dict[str, Any]]:
    """Check 4 — STALE PRINTS: runs of identical closes and zero-volume bars."""

    def _max_run(pred_equal_prev: Sequence[bool]) -> tuple[int, int]:
        """Return (max_run_length, end_index_of_that_run) for a boolean 'same as prev' seq."""
        best = 1 if pred_equal_prev else 0
        best_end = 0
        run = 1
        for i, same in enumerate(pred_equal_prev, start=1):
            run = run + 1 if same else 1
            if run > best:
                best, best_end = run, i
        return best, best_end

    findings: list[Finding] = []
    metrics: dict[str, Any] = {}

    # Identical-close runs.
    if len(ordered) >= 1:
        same_close = [ordered[i].close == ordered[i - 1].close for i in range(1, len(ordered))]
        max_close_run, end_idx = _max_run(same_close)
        metrics["max_stale_close_run"] = max_close_run
        if max_close_run >= config.stale_close_run:
            start_idx = end_idx - (max_close_run - 1)
            findings.append(
                Finding(
                    check="stale",
                    code="STALE_CLOSE_RUN",
                    severity=Status.WARN,
                    message=f"{max_close_run} consecutive identical closes",
                    detail={
                        "run_length": max_close_run,
                        "value": ordered[end_idx].close,
                        "from": ordered[start_idx].d.isoformat(),
                        "to": ordered[end_idx].d.isoformat(),
                    },
                )
            )
    else:
        metrics["max_stale_close_run"] = 0

    # Zero-volume runs + total.
    zero_total = sum(1 for b in ordered if b.volume == 0)
    metrics["zero_volume_total"] = zero_total
    is_zero = [b.volume == 0 for b in ordered]
    max_zero_run = 0
    run = 0
    run_end = 0
    for i, z in enumerate(is_zero):
        run = run + 1 if z else 0
        if run > max_zero_run:
            max_zero_run, run_end = run, i
    metrics["max_zero_volume_run"] = max_zero_run
    if max_zero_run >= config.zero_volume_run:
        run_start = run_end - (max_zero_run - 1)
        findings.append(
            Finding(
                check="stale",
                code="ZERO_VOLUME_RUN",
                severity=Status.WARN,
                message=f"{max_zero_run} consecutive zero-volume bars ({zero_total} zero-volume total)",
                detail={
                    "run_length": max_zero_run,
                    "zero_volume_total": zero_total,
                    "from": ordered[run_start].d.isoformat(),
                    "to": ordered[run_end].d.isoformat(),
                },
            )
        )
    return findings, metrics


def _check_splits(ordered: Sequence[_ValidBar], config: AuditConfig) -> tuple[list[Finding], dict[str, Any]]:
    """Check 6 — ADJUSTMENT anomalies: ~2x/0.5x single-day close jumps.

    Report only (``SUSPECTED_UNADJUSTED``); never auto-fix. Vol products and
    crypto can move this much legitimately, so this is always a WARN a human
    adjudicates — the auditor's job is to surface it, not to decide.
    """
    suspected: list[dict[str, Any]] = []
    for i in range(1, len(ordered)):
        prev, cur = ordered[i - 1], ordered[i]
        if prev.close <= 0:
            continue
        ratio = cur.close / prev.close
        if ratio >= config.split_ratio_high or ratio <= config.split_ratio_low:
            suspected.append(
                {
                    "from": prev.d.isoformat(),
                    "to": cur.d.isoformat(),
                    "prev_close": prev.close,
                    "close": cur.close,
                    "ratio": round(ratio, 4),
                }
            )
    findings: list[Finding] = []
    if suspected:
        findings.append(
            Finding(
                check="adjustment",
                code="SUSPECTED_UNADJUSTED",
                severity=Status.WARN,
                message=(
                    f"{len(suspected)} single-day jump(s) outside "
                    f"[{config.split_ratio_low}, {config.split_ratio_high}]x — suspected unadjusted split/dividend"
                ),
                detail={"count": len(suspected), "examples": _capped(suspected, config.max_examples)},
            )
        )
    return findings, {"suspected_split_count": len(suspected)}


def _check_coverage(
    ordered: Sequence[_ValidBar], corpus_last_date: Optional[date], config: AuditConfig
) -> tuple[list[Finding], dict[str, Any]]:
    """Check 8 — COVERAGE: bar count, span, thin history, feed staleness vs corpus."""
    findings: list[Finding] = []
    first_d = ordered[0].d if ordered else None
    last_d = ordered[-1].d if ordered else None
    metrics: dict[str, Any] = {
        "first_date": first_d.isoformat() if first_d else None,
        "last_date": last_d.isoformat() if last_d else None,
        "valid_bar_count": len(ordered),
    }

    if not ordered:
        findings.append(
            Finding(
                check="coverage",
                code="NO_VALID_BARS",
                severity=Status.FAIL,
                message="no schema-valid bars to audit",
                detail={},
            )
        )
        return findings, metrics

    if len(ordered) < config.thin_history_bars:
        findings.append(
            Finding(
                check="coverage",
                code="THIN_HISTORY",
                severity=Status.WARN,
                message=f"only {len(ordered)} bars (< {config.thin_history_bars}); thin for a verdict",
                detail={"bar_count": len(ordered), "threshold": config.thin_history_bars},
            )
        )

    if corpus_last_date is not None and last_d is not None:
        lag = (corpus_last_date - last_d).days
        metrics["coverage_lag_days"] = lag
        if lag >= config.coverage_lag_days:
            findings.append(
                Finding(
                    check="coverage",
                    code="STALE_FEED",
                    severity=Status.WARN,
                    message=f"last bar {last_d.isoformat()} lags corpus by {lag} calendar days",
                    detail={
                        "last_date": last_d.isoformat(),
                        "corpus_last_date": corpus_last_date.isoformat(),
                        "lag_days": lag,
                    },
                )
            )
    return findings, metrics


def _check_fx(symbol: str, source: Optional[str]) -> tuple[list[Finding], dict[str, Any], str, str]:
    """Check 7 — FX PROVENANCE (SSOT §5). Detection/report only; never converts.

    Genuinely mixed cases (FX cross-rate futures, CAD-quoted listings) raise a
    per-symbol WARN. The mass of undeclared-but-inferred-USD symbols does NOT
    raise a per-symbol WARN (that would drown the corpus in noise); instead the
    corpus roll-up records a single provenance note. Per-symbol provenance is
    always carried on the result for a later report to consume.
    """
    quote, provenance = classify_currency(symbol, source)
    findings: list[Finding] = []
    if provenance == "fx_cross_rate":
        findings.append(
            Finding(
                check="fx_provenance",
                code="FX_CROSS_RATE",
                severity=Status.WARN,
                message=(
                    f"{symbol} price is an FX cross-rate, not a USD-denominated asset price; "
                    f"any CAD/USD conversion must use chad.constants.fx.USDCAD_CONVERSION_CONSTANT "
                    f"(={USDCAD_CONVERSION_CONSTANT}), never this series"
                ),
                detail={"quote_currency": quote, "provenance": provenance},
            )
        )
    elif provenance == "cad_quoted":
        findings.append(
            Finding(
                check="fx_provenance",
                code="CAD_QUOTED",
                severity=Status.WARN,
                message=(
                    f"{symbol} appears CAD-quoted (CAD/USD-mixed corpus); conversions must use "
                    f"chad.constants.fx.USDCAD_CONVERSION_CONSTANT (={USDCAD_CONVERSION_CONSTANT})"
                ),
                detail={"quote_currency": quote, "provenance": provenance},
            )
        )
    return findings, {"quote_currency": quote, "currency_provenance": provenance}, quote, provenance


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #
def audit_symbol(
    symbol: str,
    bars: Sequence[Any],
    *,
    source: Optional[str] = None,
    timeframe: Optional[str] = None,
    config: AuditConfig = DEFAULT_CONFIG,
    corpus_last_date: Optional[date] = None,
) -> SymbolAudit:
    """Audit one symbol's already-parsed bar list. Pure; the testable core.

    ``corpus_last_date`` (when provided by :func:`audit_corpus`) enables the
    stale-feed cross-check; standalone it is skipped. No I/O, no mutation.
    """
    findings: list[Finding] = []
    metrics: dict[str, Any] = {}

    # 1 — schema → valid subset.
    valid, f_schema, m_schema = _extract_valid_bars(bars, config)
    findings += f_schema
    metrics.update(m_schema)
    metrics["raw_bar_count"] = len(bars)

    # Derived checks run on a date-sorted copy so out-of-order rows can't corrupt them.
    ordered = sorted(valid, key=lambda b: b.d)

    # 2 — OHLC sanity.
    f_ohlc, m_ohlc = _check_ohlc(valid, config)
    findings += f_ohlc
    metrics.update(m_ohlc)

    # 5 — duplicates + ordering.
    f_dup, m_dup = _check_duplicates_and_order(valid, config)
    findings += f_dup
    metrics.update(m_dup)

    continuous = _is_continuous_market(symbol, source)
    metrics["continuous_market"] = continuous

    # 3 — gaps.
    f_gap, m_gap = _check_gaps(ordered, continuous, config)
    findings += f_gap
    metrics.update(m_gap)

    # 4 — stale prints.
    f_stale, m_stale = _check_stale(ordered, config)
    findings += f_stale
    metrics.update(m_stale)

    # 6 — adjustment anomalies.
    f_split, m_split = _check_splits(ordered, config)
    findings += f_split
    metrics.update(m_split)

    # 8 — coverage / freshness.
    f_cov, m_cov = _check_coverage(ordered, corpus_last_date, config)
    findings += f_cov
    metrics.update(m_cov)

    # 7 — FX provenance.
    f_fx, m_fx, quote, provenance = _check_fx(symbol, source)
    findings += f_fx
    metrics.update(m_fx)

    status = Status.CLEAN
    for f in findings:
        status = _worst(status, f.severity)

    first_date = ordered[0].d.isoformat() if ordered else None
    last_date = ordered[-1].d.isoformat() if ordered else None

    return SymbolAudit(
        symbol=symbol,
        source=source,
        timeframe=timeframe,
        status=status,
        bar_count=len(valid),
        first_date=first_date,
        last_date=last_date,
        quote_currency=quote,
        currency_provenance=provenance,
        findings=tuple(findings),
        metrics=metrics,
    )


def _load_json(path: Path) -> tuple[Optional[dict[str, Any]], Optional[Finding]]:
    """Read + parse a bar JSON file read-only. Returns (payload, load_error_finding)."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, Finding(
            check="io",
            code="UNREADABLE_FILE",
            severity=Status.FAIL,
            message=f"cannot read file: {exc}",
            detail={"path": str(path)},
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, Finding(
            check="io",
            code="INVALID_JSON",
            severity=Status.FAIL,
            message=f"invalid JSON: {exc}",
            detail={"path": str(path)},
        )
    if not isinstance(payload, dict):
        return None, Finding(
            check="io",
            code="UNEXPECTED_TOP_LEVEL",
            severity=Status.FAIL,
            message=f"top-level JSON is {type(payload).__name__}, expected object",
            detail={"path": str(path)},
        )
    return payload, None


def audit_bar_file(
    path: Path | str,
    *,
    config: AuditConfig = DEFAULT_CONFIG,
    corpus_last_date: Optional[date] = None,
) -> SymbolAudit:
    """Audit a single bar JSON file (read-only I/O wrapper around :func:`audit_symbol`)."""
    p = Path(path)
    symbol_from_name = p.stem
    payload, load_error = _load_json(p)
    if load_error is not None:
        return SymbolAudit(
            symbol=symbol_from_name,
            source=None,
            timeframe=None,
            status=Status.FAIL,
            bar_count=0,
            first_date=None,
            last_date=None,
            quote_currency="UNKNOWN",
            currency_provenance="unreadable",
            findings=(load_error,),
            metrics={"raw_bar_count": 0},
        )

    source = payload.get("source") if isinstance(payload.get("source"), str) else None
    timeframe = payload.get("timeframe") if isinstance(payload.get("timeframe"), str) else None
    declared_symbol = payload.get("symbol")
    bars = payload.get("bars")

    extra_findings: list[Finding] = []
    if not isinstance(bars, list):
        return SymbolAudit(
            symbol=symbol_from_name,
            source=source,
            timeframe=timeframe,
            status=Status.FAIL,
            bar_count=0,
            first_date=None,
            last_date=None,
            quote_currency=classify_currency(symbol_from_name, source)[0],
            currency_provenance=classify_currency(symbol_from_name, source)[1],
            findings=(
                Finding(
                    check="schema",
                    code="MISSING_BARS_ARRAY",
                    severity=Status.FAIL,
                    message=f"'bars' is {type(bars).__name__}, expected array",
                    detail={"path": str(p)},
                ),
            ),
            metrics={"raw_bar_count": 0},
        )

    if isinstance(declared_symbol, str) and declared_symbol.upper() != symbol_from_name.upper():
        extra_findings.append(
            Finding(
                check="schema",
                code="SYMBOL_FILENAME_MISMATCH",
                severity=Status.WARN,
                message=f"declared symbol {declared_symbol!r} != filename {symbol_from_name!r}",
                detail={"declared": declared_symbol, "filename": symbol_from_name},
            )
        )

    result = audit_symbol(
        symbol_from_name,
        bars,
        source=source,
        timeframe=timeframe,
        config=config,
        corpus_last_date=corpus_last_date,
    )
    if not extra_findings:
        return result

    merged = extra_findings + list(result.findings)
    status = Status.CLEAN
    for f in merged:
        status = _worst(status, f.severity)
    return SymbolAudit(
        symbol=result.symbol,
        source=result.source,
        timeframe=result.timeframe,
        status=status,
        bar_count=result.bar_count,
        first_date=result.first_date,
        last_date=result.last_date,
        quote_currency=result.quote_currency,
        currency_provenance=result.currency_provenance,
        findings=tuple(merged),
        metrics=result.metrics,
    )


def audit_corpus(
    bars_dir: Path | str,
    *,
    config: AuditConfig = DEFAULT_CONFIG,
) -> CorpusAudit:
    """Audit every ``*.json`` bar file under ``bars_dir`` (read-only, deterministic).

    Two passes: pass 1 finds each symbol's last date to establish the corpus max;
    pass 2 audits each symbol with that max so the stale-feed cross-check is
    meaningful. Files are processed in sorted order for byte-stable output.
    """
    directory = Path(bars_dir)
    files = sorted(directory.glob("*.json"), key=lambda p: p.name)

    # Pass 1 — corpus max last date (cheap; ignores files that fail to load).
    last_dates: list[date] = []
    for p in files:
        pre = audit_bar_file(p, config=config, corpus_last_date=None)
        if pre.last_date is not None:
            parsed = _parse_date(pre.last_date)
            if parsed is not None:
                last_dates.append(parsed)
    corpus_last = max(last_dates) if last_dates else None

    # Pass 2 — full audit with the stale-feed reference.
    symbols: list[SymbolAudit] = []
    for p in files:
        symbols.append(audit_bar_file(p, config=config, corpus_last_date=corpus_last))

    clean = sum(1 for s in symbols if s.status is Status.CLEAN)
    warn = sum(1 for s in symbols if s.status is Status.WARN)
    fail = sum(1 for s in symbols if s.status is Status.FAIL)
    status_by_symbol = {s.symbol: s.status.value for s in symbols}

    corpus_findings: list[Finding] = []
    undeclared = [s.symbol for s in symbols if s.currency_provenance == "assumed_usd_undeclared"]
    if undeclared:
        corpus_findings.append(
            Finding(
                check="fx_provenance",
                code="UNDECLARED_CURRENCY_CORPUS",
                severity=Status.WARN,
                message=(
                    f"{len(undeclared)}/{len(symbols)} symbols carry no explicit quote-currency field; "
                    f"quote currency inferred as USD from source/symbol. Conversions elsewhere must use "
                    f"chad.constants.fx.USDCAD_CONVERSION_CONSTANT (={USDCAD_CONVERSION_CONSTANT})."
                ),
                detail={"count": len(undeclared), "symbols": sorted(undeclared)},
            )
        )
    fx_flagged = [s.symbol for s in symbols if s.currency_provenance in ("fx_cross_rate", "cad_quoted")]
    if fx_flagged:
        corpus_findings.append(
            Finding(
                check="fx_provenance",
                code="FX_PROVENANCE_FLAGGED",
                severity=Status.WARN,
                message=f"{len(fx_flagged)} symbol(s) flagged for FX/CAD provenance",
                detail={"symbols": sorted(fx_flagged)},
            )
        )

    return CorpusAudit(
        bars_dir=str(directory),
        symbol_count=len(symbols),
        clean=clean,
        warn=warn,
        fail=fail,
        corpus_last_date=corpus_last.isoformat() if corpus_last else None,
        status_by_symbol=status_by_symbol,
        symbols=tuple(symbols),
        corpus_findings=tuple(corpus_findings),
        config=config,
    )


# --------------------------------------------------------------------------- #
# Human-readable rendering (the machine result stays the source of truth).
# --------------------------------------------------------------------------- #
def render_symbol_audit(audit: SymbolAudit) -> str:
    """One-symbol human-readable block."""
    lines: list[str] = []
    span = f"{audit.first_date or '?'}..{audit.last_date or '?'}"
    lines.append(
        f"[{audit.status.value:5s}] {audit.symbol:10s} src={audit.source or '?':7s} "
        f"bars={audit.bar_count:4d} {span} quote={audit.quote_currency}/{audit.currency_provenance}"
    )
    for f in audit.findings:
        lines.append(f"    - {f.severity.value:4s} {f.check}/{f.code}: {f.message}")
    return "\n".join(lines)


def render_corpus_summary(audit: CorpusAudit, *, max_symbol_lines: int = 0) -> str:
    """Corpus roll-up: status counts, top findings by code, per-symbol lines.

    ``max_symbol_lines``: 0 = show all symbols; N>0 = show only the first N
    (sorted worst-status first, then symbol) to keep large corpora readable.
    """
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"BAR CORPUS DATA-QUALITY AUDIT — {audit.bars_dir}")
    lines.append("=" * 78)
    lines.append(
        f"symbols={audit.symbol_count}  CLEAN={audit.clean}  WARN={audit.warn}  FAIL={audit.fail}  "
        f"corpus_last_date={audit.corpus_last_date}"
    )
    cfg = audit.config
    lines.append(
        "thresholds: "
        f"stale_close_run={cfg.stale_close_run} zero_volume_run={cfg.zero_volume_run} "
        f"gap_session_threshold={cfg.gap_session_threshold} "
        f"split=[{cfg.split_ratio_low},{cfg.split_ratio_high}]x "
        f"thin_history_bars={cfg.thin_history_bars} coverage_lag_days={cfg.coverage_lag_days}"
    )

    # Top findings aggregated by code.
    code_counts: dict[tuple[str, str], int] = {}
    for s in audit.symbols:
        for f in s.findings:
            key = (f.severity.value, f.code)
            code_counts[key] = code_counts.get(key, 0) + 1
    if code_counts:
        lines.append("")
        lines.append("Top findings (symbols affected, by code):")
        ordered = sorted(code_counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
        for (sev, code), n in ordered:
            lines.append(f"    {sev:4s} {code:26s} {n} symbol(s)")

    if audit.corpus_findings:
        lines.append("")
        lines.append("Corpus-level notes:")
        for f in audit.corpus_findings:
            lines.append(f"    {f.severity.value:4s} {f.code}: {f.message}")

    lines.append("")
    lines.append("Per-symbol:")
    rank = _STATUS_RANK
    ordered_symbols = sorted(audit.symbols, key=lambda s: (-rank[s.status], s.symbol))
    shown = ordered_symbols if max_symbol_lines <= 0 else ordered_symbols[:max_symbol_lines]
    for s in shown:
        lines.append(render_symbol_audit(s))
    if len(shown) < len(ordered_symbols):
        lines.append(f"    ... {len(ordered_symbols) - len(shown)} more symbol(s) omitted")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Minimal read-only CLI: ``python -m chad.validation.bar_audit [bars_dir]``.

    Prints the corpus summary. Does not write any file. (The full harness CLI is
    Phase 5; this is a convenience smoke entry point only.)
    """
    import argparse

    parser = argparse.ArgumentParser(description="Forensic data-quality audit of the daily bar corpus.")
    parser.add_argument("bars_dir", nargs="?", default="data/bars/1d", help="directory of *.json bar files")
    parser.add_argument("--symbol", default=None, help="audit a single bar file (path) instead of the corpus")
    args = parser.parse_args(argv)

    if args.symbol is not None:
        result = audit_bar_file(args.symbol)
        print(render_symbol_audit(result))
        return 0 if result.status is not Status.FAIL else 1

    corpus = audit_corpus(args.bars_dir)
    print(render_corpus_summary(corpus))
    return 0 if corpus.fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
