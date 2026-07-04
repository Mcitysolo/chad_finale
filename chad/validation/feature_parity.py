"""chad/validation/feature_parity.py — Phase 4 per-head feature-parity audit (SSOT §V1).

The FIRST gate of the Stage-1 backtest. Before a single synthetic trade is
generated, this module answers one question for every strategy head: *can the
inputs its live decision consumes be reconstructed from the historical daily-bar
corpus we actually have?* A head that depends on an input unavailable historically
(Polygon news/catalyst, an options-chain cache, a live VIX feed, intraday ticks /
1-minute bars, live account/portfolio state, a runtime state file) is classified
:attr:`ParityStatus.NOT_REPLAYABLE` and is **reported honestly, never scored** —
the design's core defence against a "silent degraded replay" (SSOT
``docs/CHAD_EDGE_VALIDATION_HARNESS_DESIGN_v1.1.md`` §V1 / §4.1).

Why static source analysis, not import-and-introspect
-----------------------------------------------------
The isolation guarantee (SSOT §1.2 / §2, enforced by
``tests/validation/test_isolation.py``) forbids the harness's import closure from
touching ``live_loop``, broker adapters, or any ``runtime/`` reader — and that test
imports *every* submodule of ``chad.validation``, this one included. Importing a
real strategy module to introspect it would drag ``chad.core`` / ``chad.execution``
/ ``runtime`` readers into the closure and fail isolation, and could fire import-time
side effects that reach live state. So this module never imports a strategy: it
reads the strategy's **source text** and classifies it with :mod:`ast` (identifier,
import, call, and non-docstring string-constant inspection). That honours both the
isolation wall and the /goal constraint "inspect read-only, must not be executed".

The honest-default discipline (SSOT Part 0)
-------------------------------------------
Static inspection cannot prove a negative. So the classifier NEVER assumes
"replayable" from absence of evidence:
  * source cannot be read / parsed            → :attr:`ParityStatus.UNKNOWN`
  * NO input family detected at all           → :attr:`ParityStatus.UNKNOWN`
  * any category-(c) UNAVAILABLE input         → :attr:`ParityStatus.NOT_REPLAYABLE`
  * any category-(b) APPROXIMABLE input (no c) → :attr:`ParityStatus.APPROXIMABLE`
  * only category-(a) RECONSTRUCTABLE inputs   → :attr:`ParityStatus.REPLAYABLE`
Only :attr:`ParityStatus.REPLAYABLE` is backtestable by default (see
:func:`is_backtestable`); every other status is skipped by the engine with that
status attached, never faked.

Conservatism / safety asymmetry
-------------------------------
The dangerous error is classifying a head that truly needs unavailable data as
REPLAYABLE (it would then be scored on garbage). The safe error is the reverse
(an honest edge left unproven). The catalog is therefore biased toward *detecting*
category-(c) dependencies, and any head with no clear evidence falls to UNKNOWN
rather than REPLAYABLE. Matching is deliberately collision-resistant: single-word
tokens match whole identifier *word-parts* (so ``scr`` matches an attribute
``scr_state`` but not ``describe``), and compound/path tokens match as substrings
of identifiers and NON-docstring string constants only (a docstring that merely
*mentions* "news" cannot trip a head — verified by test).

Isolation (SSOT §1.2 / §2): pure, offline, deterministic, standard-library only —
:mod:`ast`, :mod:`re`, :mod:`dataclasses`, :mod:`enum`, :mod:`pathlib`. No numpy,
no broker, no ``runtime/`` reader, no live-loop / strategy import.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

__all__ = [
    "InputCategory",
    "ParityStatus",
    "InputSignal",
    "DetectedInput",
    "FeatureParityResult",
    "DEFAULT_SIGNAL_CATALOG",
    "classify_source",
    "classify_head_file",
    "audit_heads",
    "is_backtestable",
    "render_parity_summary",
]


# --------------------------------------------------------------------------- #
# Categories & statuses.
# --------------------------------------------------------------------------- #
class InputCategory(Enum):
    """Reconstructability of one input family from the historical daily-bar corpus.

    Mirrors SSOT §V1's three buckets: (a) reconstructable from historical bars,
    (b) approximable with a declared error, (c) unavailable historically.
    """

    RECONSTRUCTABLE = "reconstructable"  # (a) computable directly from daily bars
    APPROXIMABLE = "approximable"        # (b) reproducible with a declared error
    UNAVAILABLE = "unavailable"          # (c) no historical source in the corpus


class ParityStatus(Enum):
    """Per-head verdict of the feature-parity audit (SSOT §4.1).

    ``REPLAYABLE`` — every detected input is category (a); the head may be
    backtested. ``APPROXIMABLE`` — at least one category-(b) input, none (c); the
    head is replayable only with a declared approximation (out of Phase-4 scope,
    so not backtested by default). ``NOT_REPLAYABLE`` — at least one category-(c)
    input; reported, never scored. ``UNKNOWN`` — inputs could not be determined by
    inspection (unreadable/unparseable source, or no recognised input family); the
    honest default that must never be silently treated as replayable.
    """

    REPLAYABLE = "REPLAYABLE"
    APPROXIMABLE = "APPROXIMABLE"
    NOT_REPLAYABLE = "NOT_REPLAYABLE"
    UNKNOWN = "UNKNOWN"


# --------------------------------------------------------------------------- #
# Signal catalog — one entry per input family the audit knows how to recognise.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InputSignal:
    """A catalog entry: an input family, its category, and how to detect it.

    ``idents`` are single lowercase words matched against identifier *word-parts*
    (whole-word, collision-resistant — ``scr`` hits ``scr_state`` but not
    ``describe``). ``texts`` are substrings matched against the raw identifier
    strings and NON-docstring string constants (for compound / path / filename
    evidence like ``news_intel`` or ``runtime/``). A signal fires if ANY of its
    tokens matches. ``rationale`` explains, for the report, why the family sits in
    its category.
    """

    name: str
    category: InputCategory
    idents: tuple[str, ...]
    texts: tuple[str, ...]
    rationale: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("InputSignal.name must be non-empty")
        if not isinstance(self.category, InputCategory):
            raise ValueError(f"category must be an InputCategory, got {self.category!r}")
        if not self.idents and not self.texts:
            raise ValueError(f"signal {self.name!r} must declare at least one token")
        for tok in self.idents:
            if tok != tok.lower() or not tok or any(ch in tok for ch in " _-/."):
                raise ValueError(
                    f"ident token {tok!r} in {self.name!r} must be a single lowercase word "
                    "(compound/path tokens belong in `texts`)"
                )
        for tok in self.texts:
            if tok != tok.lower() or not tok:
                raise ValueError(f"text token {tok!r} in {self.name!r} must be lowercase/non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "idents": list(self.idents),
            "texts": list(self.texts),
            "rationale": self.rationale,
        }


# The default catalog. Ordered category (a) → (b) → (c); within a category the
# order is stable so report output is deterministic. Tokens were curated against
# CHAD's real strategy surface (a read-only recon of ``chad/strategies/*.py``) and
# deliberately exclude collision-prone bare words (e.g. plain ``open``/``high``/
# ``close``/``delta``/``gamma``/``beta``) that would false-match unrelated code.
DEFAULT_SIGNAL_CATALOG: tuple[InputSignal, ...] = (
    # ---- (a) RECONSTRUCTABLE from historical daily bars --------------------- #
    InputSignal(
        name="daily_bars",
        category=InputCategory.RECONSTRUCTABLE,
        idents=("bars", "bar", "ohlc", "ohlcv", "candle", "candles", "closes", "opens",
                "highs", "lows", "vwap", "prices"),
        texts=("price_history", "close_prices", "daily_bar", "bar_history"),
        rationale="daily OHLCV price history — present in the audited bar corpus",
    ),
    InputSignal(
        name="technical_indicator",
        category=InputCategory.RECONSTRUCTABLE,
        idents=("sma", "ema", "rsi", "macd", "bollinger", "momentum", "roc", "adx",
                "obv", "atr", "zscore"),
        texts=("moving_average", "mean_reversion", "z_score"),
        rationale="an indicator deterministically computable from the bar series",
    ),
    InputSignal(
        name="realized_volatility",
        category=InputCategory.RECONSTRUCTABLE,
        idents=("volatility", "variance", "stdev"),
        texts=("realized_vol", "realized_volatility", "std_dev", "return_std"),
        rationale="realized volatility computable from daily-bar returns",
    ),
    # ---- (b) APPROXIMABLE with a declared error ----------------------------- #
    InputSignal(
        name="regime_label",
        category=InputCategory.APPROXIMABLE,
        idents=("regime", "regimes"),
        texts=("market_regime", "trend_regime", "regime_label"),
        rationale="CHAD's live regime label — approximable by the harness's own "
                  "independent labeler (SSOT §3.4) with a declared divergence error",
    ),
    InputSignal(
        name="correlation_structure",
        category=InputCategory.APPROXIMABLE,
        idents=("correlation", "cointegration", "coint", "covariance"),
        texts=("cross_asset", "half_life", "corr_matrix"),
        rationale="cross-asset correlation/cointegration — approximable from the bar "
                  "cross-section with a window/estimator error",
    ),
    # ---- (c) UNAVAILABLE historically (any → NOT_REPLAYABLE) ----------------- #
    InputSignal(
        name="news_catalyst",
        category=InputCategory.UNAVAILABLE,
        idents=("news", "catalyst", "headline", "headlines", "sentiment"),
        texts=("news_intel", "catalyst_gate", "press_release", "earnings_news"),
        rationale="Polygon-style news/catalyst intel — no historical archive in the corpus",
    ),
    InputSignal(
        name="intraday_microstructure",
        # NOTE: bare ``tick`` is deliberately EXCLUDED — futures contract specs use
        # ``min_tick`` / ``tick_size`` / ``tick_value`` (static instrument metadata, not
        # intraday data), which would false-match. Real intraday reads use ``ctx.ticks``
        # (plural), so ``ticks`` is the true signal and no genuine input is lost.
        category=InputCategory.UNAVAILABLE,
        idents=("ticks", "intraday", "minute", "minutes", "bid", "ask",
                "quote", "quotes", "orderbook", "depth", "microstructure", "orb",
                "sweep", "session"),
        texts=("bars_1m", "1m", "5m", "15m", "minutes_since_open", "order_book",
               "level2", "opening_range", "session_window", "primary_session",
               "last_price", "session_decision"),
        rationale="intraday ticks / minute bars / session-clock state — not "
                  "reconstructable from DAILY bars",
    ),
    InputSignal(
        name="options_iv_greeks",
        category=InputCategory.UNAVAILABLE,
        idents=("option", "options", "chain", "greeks", "greek", "iv", "vega",
                "theta", "strike"),
        texts=("implied_vol", "iv_rank", "put_call", "option_chain", "options_chain",
               "options_greeks", "chain_provider"),
        rationale="options chain / implied-vol / greeks (live daemon cache) — "
                  "unavailable historically",
    ),
    InputSignal(
        name="vix_index_feed",
        category=InputCategory.UNAVAILABLE,
        idents=("vix",),
        texts=("vol_index", "volatility_index", "vix_history", "vix_regime"),
        rationale="a live VIX/volatility-index feed — not in the daily equity bar corpus",
    ),
    InputSignal(
        name="institutional_13f",
        category=InputCategory.UNAVAILABLE,
        idents=("legend", "consensus", "institutional", "whale"),
        texts=("institutional_consensus", "thirteen_f", "13f", "form13f", "sec_13f"),
        rationale="13F / institutional-consensus (legend) data — external quarterly "
                  "feed, no per-bar history",
    ),
    InputSignal(
        name="macro_state",
        category=InputCategory.UNAVAILABLE,
        idents=("macro", "fomc", "cpi", "nfp"),
        texts=("macro_state", "macro_regime", "econ_calendar", "economic_calendar",
               "classify_macro"),
        rationale="macro / economic-calendar state — no historical macro archive in the corpus",
    ),
    InputSignal(
        name="live_account_state",
        category=InputCategory.UNAVAILABLE,
        idents=("portfolio", "drawdown", "equity", "balance", "balances", "holdings", "nav"),
        texts=("buying_power", "account_equity", "kraken_balances"),
        rationale="live account/portfolio state (positions, equity, drawdown) — "
                  "path-dependent on execution, not a daily-bar input",
    ),
    InputSignal(
        name="self_confidence_scr",
        category=InputCategory.UNAVAILABLE,
        idents=("scr",),
        texts=("scr_state", "scr_snapshot", "self_confidence"),
        rationale="SCR self-confidence-rating runtime state — a live governance "
                  "signal with no historical series",
    ),
    InputSignal(
        name="runtime_state_file",
        category=InputCategory.UNAVAILABLE,
        idents=("kraken",),
        texts=("runtime/", "trade_closer_state", "options_chains_cache", "regime_state",
               "relative_strength", "futures_roll_state", "positions_snapshot",
               "position_guard", ".ndjson"),
        rationale="a live ``runtime/`` state file written by the trading loop — "
                  "not reproducible from historical bars",
    ),
)


# --------------------------------------------------------------------------- #
# Detected-input + result records (flat, serialisable, embedded by reports).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DetectedInput:
    """One input family found in a head, with its category, evidence, and rationale."""

    name: str
    category: InputCategory
    evidence: tuple[str, ...]  # the specific tokens that matched (sorted, deduped)
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "evidence": list(self.evidence),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class FeatureParityResult:
    """The reusable per-head feature-parity verdict (SSOT §V1) — the engine's gate.

    ``status`` is the :class:`ParityStatus`. ``inputs`` is the full detected-input
    inventory (all categories, sorted by category then name). ``reasons`` are
    human-readable strings explaining the status. ``analyzed`` is ``False`` when the
    source could not be read/parsed (→ ``UNKNOWN``). ``source_ref`` is the path or a
    caller-supplied tag for provenance. Flat/JSON-serialisable via :meth:`to_dict`.
    """

    head: str
    status: ParityStatus
    inputs: tuple[DetectedInput, ...]
    reasons: tuple[str, ...]
    analyzed: bool
    source_ref: Optional[str]

    def category_counts(self) -> dict[str, int]:
        """Count of detected inputs per category value (all categories keyed)."""
        counts = {c.value: 0 for c in InputCategory}
        for inp in self.inputs:
            counts[inp.category.value] += 1
        return counts

    def unavailable_inputs(self) -> tuple[DetectedInput, ...]:
        """The category-(c) inputs that force NOT_REPLAYABLE (empty if none)."""
        return tuple(i for i in self.inputs if i.category is InputCategory.UNAVAILABLE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "head": self.head,
            "status": self.status.value,
            "inputs": [i.to_dict() for i in self.inputs],
            "reasons": list(self.reasons),
            "analyzed": self.analyzed,
            "source_ref": self.source_ref,
            "category_counts": self.category_counts(),
        }


# --------------------------------------------------------------------------- #
# Source → identifier / string-constant extraction (docstrings excluded).
# --------------------------------------------------------------------------- #
_CAMEL_1 = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CAMEL_2 = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
_NON_ALNUM = re.compile(r"[^0-9A-Za-z]+")


def _split_ident(ident: str) -> list[str]:
    """Split an identifier into lowercase word-parts (camelCase + separators).

    ``"scr_state"`` → ``["scr", "state"]``; ``"barHistory"`` → ``["bar", "history"]``;
    ``"bars_1m"`` → ``["bars", "1m"]``. Used for collision-resistant whole-word
    matching of single-word signal tokens.
    """
    spaced = _CAMEL_2.sub(" ", _CAMEL_1.sub(" ", ident))
    return [p.lower() for p in _NON_ALNUM.split(spaced) if p]


def _docstring_constant_ids(tree: ast.AST) -> set[int]:
    """Return ``id()`` of every Constant node that is a module/class/function docstring.

    A docstring is the first statement of a Module/Class/(Async)FunctionDef body when
    that statement is an ``Expr`` wrapping a ``str`` ``Constant``. Excluding these from
    the string-constant scan is what lets a head merely *mention* "news"/"scr" in prose
    without being misclassified (verified by test).
    """
    doc_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr):
                value = body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    doc_ids.add(id(value))
    return doc_ids


@dataclass(frozen=True)
class _SourceFacts:
    """Extracted, lower-cased evidence from a strategy source (internal)."""

    idents: frozenset[str]       # raw lowercase identifier strings
    words: frozenset[str]        # union of identifier word-parts
    strings: frozenset[str]      # non-docstring string-constant values, lowercased


def _extract_facts(source: str) -> Optional[_SourceFacts]:
    """Parse ``source`` and extract identifiers, word-parts, and non-docstring strings.

    Returns ``None`` when the source cannot be parsed (a syntax error, a null byte,
    or pathological nesting) — the caller maps that to ``UNKNOWN``/``analyzed=False``.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return None

    doc_ids = _docstring_constant_ids(tree)
    idents: set[str] = set()
    strings: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            idents.add(node.attr.lower())
        elif isinstance(node, ast.Name):
            idents.add(node.id.lower())
        elif isinstance(node, ast.arg):
            idents.add(node.arg.lower())
        elif isinstance(node, ast.keyword) and node.arg is not None:
            idents.add(node.arg.lower())
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            idents.add(node.name.lower())
        elif isinstance(node, ast.Import):
            for alias in node.names:
                idents.add(alias.name.lower())
                if alias.asname:
                    idents.add(alias.asname.lower())
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                idents.add(node.module.lower())
            for alias in node.names:
                idents.add(alias.name.lower())
                if alias.asname:
                    idents.add(alias.asname.lower())
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in doc_ids:
                strings.add(node.value.lower())

    words: set[str] = set()
    for ident in idents:
        words.update(_split_ident(ident))

    return _SourceFacts(
        idents=frozenset(idents),
        words=frozenset(words),
        strings=frozenset(strings),
    )


def _signal_evidence(signal: InputSignal, facts: _SourceFacts) -> tuple[str, ...]:
    """Return the sorted, deduped tokens of ``signal`` that matched ``facts`` (empty = no match).

    Single-word ``idents`` match whole identifier word-parts (collision-resistant);
    compound/path ``texts`` match as substrings of raw identifiers or non-docstring
    string constants.
    """
    hits: set[str] = set()
    for tok in signal.idents:
        if tok in facts.words:
            hits.add(tok)
    if signal.texts:
        haystacks = tuple(facts.idents) + tuple(facts.strings)
        for tok in signal.texts:
            if any(tok in hay for hay in haystacks):
                hits.add(tok)
    return tuple(sorted(hits))


# Sort key so the input inventory is deterministic: category (a<b<c) then name.
_CATEGORY_ORDER: dict[InputCategory, int] = {
    InputCategory.RECONSTRUCTABLE: 0,
    InputCategory.APPROXIMABLE: 1,
    InputCategory.UNAVAILABLE: 2,
}


# --------------------------------------------------------------------------- #
# Core classifier — a source string in, a FeatureParityResult out.
# --------------------------------------------------------------------------- #
def classify_source(
    head: str,
    source: str,
    *,
    catalog: Sequence[InputSignal] = DEFAULT_SIGNAL_CATALOG,
    source_ref: Optional[str] = None,
) -> FeatureParityResult:
    """Classify one head from its decision-logic SOURCE (SSOT §V1). The testable core.

    Deterministic and pure — no import of ``source``, no filesystem, no network.
    ``source`` is the raw text of the strategy module (or the relevant decision fn).
    Detection uses :mod:`ast` identifier / import / call / non-docstring string scans
    against ``catalog``.

    Status rules (honest-default, SSOT Part 0): unparseable source → ``UNKNOWN``
    (``analyzed=False``); no input family detected → ``UNKNOWN``; any category-(c)
    input → ``NOT_REPLAYABLE``; else any category-(b) → ``APPROXIMABLE``; else
    (≥1 category-(a), none higher) → ``REPLAYABLE``.
    """
    if not isinstance(head, str) or not head:
        raise ValueError(f"head must be a non-empty str, got {head!r}")
    if not isinstance(source, str):
        raise ValueError(f"source must be a str, got {type(source).__name__}")

    facts = _extract_facts(source)
    if facts is None:
        return FeatureParityResult(
            head=head,
            status=ParityStatus.UNKNOWN,
            inputs=(),
            reasons=("source could not be parsed (syntax error / unreadable); "
                     "inputs undeterminable → UNKNOWN, not assumed replayable",),
            analyzed=False,
            source_ref=source_ref,
        )

    detected: list[DetectedInput] = []
    seen_names: set[str] = set()
    for signal in catalog:
        evidence = _signal_evidence(signal, facts)
        if evidence and signal.name not in seen_names:
            seen_names.add(signal.name)
            detected.append(
                DetectedInput(
                    name=signal.name,
                    category=signal.category,
                    evidence=evidence,
                    rationale=signal.rationale,
                )
            )
    detected.sort(key=lambda d: (_CATEGORY_ORDER[d.category], d.name))
    inputs = tuple(detected)

    unavailable = [d for d in inputs if d.category is InputCategory.UNAVAILABLE]
    approximable = [d for d in inputs if d.category is InputCategory.APPROXIMABLE]
    reconstructable = [d for d in inputs if d.category is InputCategory.RECONSTRUCTABLE]

    if not inputs:
        status = ParityStatus.UNKNOWN
        reasons = (
            "no recognised input family detected by static inspection; "
            "cannot assert the head reads only reconstructable bars → UNKNOWN",
        )
    elif unavailable:
        status = ParityStatus.NOT_REPLAYABLE
        names = ", ".join(f"{d.name}({'/'.join(d.evidence)})" for d in unavailable)
        reasons = (
            f"depends on {len(unavailable)} category-(c) input(s) unavailable "
            f"historically: {names}",
            "reported honestly, NOT scored (SSOT §V1 / §4.1)",
        )
    elif approximable:
        status = ParityStatus.APPROXIMABLE
        names = ", ".join(d.name for d in approximable)
        reasons = (
            f"all inputs reconstructable or approximable; category-(b) input(s) "
            f"present ({names}) require a declared approximation",
            "not backtested by default in Phase 4 (REPLAYABLE-only engine gate)",
        )
    else:
        status = ParityStatus.REPLAYABLE
        names = ", ".join(d.name for d in reconstructable)
        reasons = (
            f"every detected input ({names}) is reconstructable from historical "
            f"daily bars → REPLAYABLE",
        )

    return FeatureParityResult(
        head=head,
        status=status,
        inputs=inputs,
        reasons=reasons,
        analyzed=True,
        source_ref=source_ref,
    )


def classify_head_file(
    head: str,
    path: Path | str,
    *,
    catalog: Sequence[InputSignal] = DEFAULT_SIGNAL_CATALOG,
) -> FeatureParityResult:
    """Read-only wrapper of :func:`classify_source` over a strategy source FILE.

    Reads the file's text (never imports it) and classifies it. An unreadable /
    non-decodable file yields ``UNKNOWN`` (``analyzed=False``) — the honest default,
    never a fabricated pass. ``source_ref`` is set to the resolved path.
    """
    p = Path(path)
    try:
        source = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return FeatureParityResult(
            head=head,
            status=ParityStatus.UNKNOWN,
            inputs=(),
            reasons=(f"source file could not be read ({type(exc).__name__}: {exc}); "
                     "inputs undeterminable → UNKNOWN",),
            analyzed=False,
            source_ref=str(p),
        )
    return classify_source(head, source, catalog=catalog, source_ref=str(p))


def audit_heads(
    head_to_path: Mapping[str, str | Path],
    *,
    catalog: Sequence[InputSignal] = DEFAULT_SIGNAL_CATALOG,
) -> tuple[FeatureParityResult, ...]:
    """Classify a whole map of ``head name → source path`` (sorted by head name).

    Deterministic (results ordered by head name). Each head is classified read-only
    via :func:`classify_head_file`; a missing/unreadable file becomes ``UNKNOWN`` for
    that head rather than aborting the batch.
    """
    results = [
        classify_head_file(head, path, catalog=catalog)
        for head, path in sorted(head_to_path.items(), key=lambda kv: kv[0])
    ]
    return tuple(results)


def is_backtestable(
    result: FeatureParityResult, *, allow_approximable: bool = False
) -> bool:
    """Gate the backtest engine: may this head be replayed on historical bars?

    Default: only :attr:`ParityStatus.REPLAYABLE` is backtestable — the strict
    reading of SSOT §V1 (category-(c) heads NOT scored) and the /goal "replay a
    REPLAYABLE head". ``allow_approximable=True`` additionally admits
    :attr:`ParityStatus.APPROXIMABLE` for callers that supply the declared
    approximation (out of Phase-4 scope). ``NOT_REPLAYABLE`` and ``UNKNOWN`` are
    NEVER backtestable.
    """
    if result.status is ParityStatus.REPLAYABLE:
        return True
    if allow_approximable and result.status is ParityStatus.APPROXIMABLE:
        return True
    return False


# --------------------------------------------------------------------------- #
# Human-readable summary — the honest map of what our data can validate.
# --------------------------------------------------------------------------- #
def render_parity_summary(results: Iterable[FeatureParityResult]) -> str:
    """Render a deterministic, monospace per-head parity table + status tally.

    Columns: head | status | reconstructable / approximable / unavailable counts |
    the category-(c) input names that blocked replay (if any). A ``TOTALS`` line
    tallies each :class:`ParityStatus`. Pure formatting — no I/O.
    """
    rows = list(results)
    header = f"{'HEAD':<26} {'STATUS':<15} {'a/b/c':<9} BLOCKING_UNAVAILABLE_INPUTS"
    lines = [header, "-" * len(header)]
    tally = {s.value: 0 for s in ParityStatus}
    for r in sorted(rows, key=lambda x: x.head):
        tally[r.status.value] += 1
        c = r.category_counts()
        abc = f"{c['reconstructable']}/{c['approximable']}/{c['unavailable']}"
        blockers = ", ".join(i.name for i in r.unavailable_inputs()) or "-"
        lines.append(f"{r.head:<26} {r.status.value:<15} {abc:<9} {blockers}")
    lines.append("-" * len(header))
    tally_str = "  ".join(f"{k}={tally[k]}" for k in (
        ParityStatus.REPLAYABLE.value,
        ParityStatus.APPROXIMABLE.value,
        ParityStatus.NOT_REPLAYABLE.value,
        ParityStatus.UNKNOWN.value,
    ))
    lines.append(f"TOTALS ({len(rows)} heads): {tally_str}")
    return "\n".join(lines)
