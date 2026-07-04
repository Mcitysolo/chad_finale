"""Tests for chad/validation/feature_parity.py — Phase 4 feature-parity gate (SSOT §V1).

Fixture-only: every classification target is a STUB source string (or a temp file)
with known inputs — never a real strategy's internals — so the tests assert the
classifier's contract, not CHAD's live surface. The three headline cases the /goal
names are covered explicitly:
  * a stub reading only bars                       → REPLAYABLE
  * a stub reading news                            → NOT_REPLAYABLE (never scored)
  * a stub whose inputs are undetectable           → UNKNOWN (honest default)
plus the safety properties: docstring mentions cannot trip a head, category-(c)
dominates precedence, and word-part matching is collision-resistant.
"""

from __future__ import annotations

import json

import pytest

from chad.validation.feature_parity import (
    DEFAULT_SIGNAL_CATALOG,
    DetectedInput,
    FeatureParityResult,
    InputCategory,
    InputSignal,
    ParityStatus,
    audit_heads,
    classify_head_file,
    classify_source,
    is_backtestable,
    render_parity_summary,
)

# --------------------------------------------------------------------------- #
# Stub sources with KNOWN inputs (not real strategy code).
# --------------------------------------------------------------------------- #
BARS_ONLY = '''
def decide(window):
    """A pure momentum head."""
    closes = [b.close for b in window.bars]
    fast = sum(closes[-5:]) / 5.0
    slow = sum(closes[-20:]) / 20.0
    return "long" if fast > slow else None
'''

BARS_WITH_INDICATORS = '''
def decide(window):
    prices = window.closes()
    rsi = compute_rsi(prices)
    ema = compute_ema(prices)
    return rsi, ema
'''

NEWS_HEAD = '''
def decide(ctx):
    """Momentum + catalyst gate."""
    if check_catalyst_gate(ctx) and ctx.news_intel:
        return ctx.bars
    return None
'''

APPROXIMABLE_HEAD = '''
def decide(ctx):
    """Regime-aware trend head."""
    bars = ctx.bars
    if ctx.regime == "uptrend":
        return bars
    return None
'''

BARS_PLUS_REGIME_PLUS_NEWS = '''
def decide(ctx):
    return ctx.bars, ctx.regime, ctx.news
'''

UNDETECTABLE_HEAD = '''
def decide(ctx):
    return ctx.compute_thing(ctx.opaque_input, ctx.something_else)
'''

DESCRIBE_TRAP = '''
def describe(ctx):
    """This method describes the strategy. It mentions SCR, news, VIX, and
    options in PROSE only — none are actually read below."""
    prices = ctx.bars
    return prices
'''

RUNTIME_STRING_HEAD = '''
import json
def decide(ctx):
    with open("runtime/scr_state.json") as fh:
        state = json.load(fh)
    return ctx.bars if state else None
'''

INTRADAY_HEAD = '''
def decide(ctx):
    minute_bars = ctx.bars_1m
    last = ctx.ticks["SPY"].price
    return last if minute_bars else None
'''

SYNTAX_ERROR = "def decide(ctx):\n    return ctx.bars ((("


# --------------------------------------------------------------------------- #
# The three headline cases (/goal).
# --------------------------------------------------------------------------- #
def test_bars_only_is_replayable() -> None:
    r = classify_source("bars_only", BARS_ONLY)
    assert r.status is ParityStatus.REPLAYABLE
    assert r.analyzed is True
    assert is_backtestable(r) is True
    names = {i.name for i in r.inputs}
    assert "daily_bars" in names
    # Every detected input is category (a).
    assert all(i.category is InputCategory.RECONSTRUCTABLE for i in r.inputs)
    assert r.unavailable_inputs() == ()


def test_news_head_is_not_replayable_and_never_scored() -> None:
    r = classify_source("newsy", NEWS_HEAD)
    assert r.status is ParityStatus.NOT_REPLAYABLE
    assert is_backtestable(r) is False
    # It DID also read bars, but the category-(c) news dependency dominates.
    blocking = {i.name for i in r.unavailable_inputs()}
    assert "news_catalyst" in blocking
    assert r.category_counts()["unavailable"] >= 1


def test_undetectable_inputs_are_unknown_not_replayable() -> None:
    r = classify_source("opaque", UNDETECTABLE_HEAD)
    assert r.status is ParityStatus.UNKNOWN
    assert r.inputs == ()
    assert is_backtestable(r) is False
    # The honest default: absence of evidence is NOT evidence of bars-only.
    assert any("UNKNOWN" in reason or "cannot assert" in reason for reason in r.reasons)


# --------------------------------------------------------------------------- #
# Safety properties.
# --------------------------------------------------------------------------- #
def test_docstring_mentions_do_not_trip_classification() -> None:
    """A head that only MENTIONS forbidden inputs in prose reads only bars → REPLAYABLE."""
    r = classify_source("describe_trap", DESCRIBE_TRAP)
    assert r.status is ParityStatus.REPLAYABLE
    names = {i.name for i in r.inputs}
    assert names == {"daily_bars"}
    # None of scr/news/vix/options fired from the docstring text.
    assert r.category_counts()["unavailable"] == 0


def test_word_part_matching_is_collision_resistant() -> None:
    """`describe`/`disclosed`/`below`/`allow` must not false-match scr/close/low tokens."""
    src = (
        "def describe(ctx):\n"
        "    disclosed = ctx.allow_below\n"
        "    workflow = ctx.overflow\n"
        "    return disclosed, workflow\n"
    )
    r = classify_source("collide", src)
    # No bar/scr signal should fire from describe/disclosed/below/allow/overflow.
    assert r.status is ParityStatus.UNKNOWN
    assert r.inputs == ()


def test_category_c_dominates_precedence() -> None:
    r = classify_source("mixed", BARS_PLUS_REGIME_PLUS_NEWS)
    assert r.status is ParityStatus.NOT_REPLAYABLE
    c = r.category_counts()
    assert c["reconstructable"] >= 1 and c["approximable"] >= 1 and c["unavailable"] >= 1


def test_approximable_head_not_backtested_by_default() -> None:
    r = classify_source("regimey", APPROXIMABLE_HEAD)
    assert r.status is ParityStatus.APPROXIMABLE
    assert is_backtestable(r) is False
    assert is_backtestable(r, allow_approximable=True) is True
    assert {i.name for i in r.inputs} == {"daily_bars", "regime_label"}


def test_runtime_state_string_literal_is_detected() -> None:
    """A `runtime/…json` string constant (non-docstring) flags live-state reads."""
    r = classify_source("rt", RUNTIME_STRING_HEAD)
    assert r.status is ParityStatus.NOT_REPLAYABLE
    blocking = {i.name for i in r.unavailable_inputs()}
    # Both the scr token and the runtime/ path token should fire.
    assert "runtime_state_file" in blocking or "self_confidence_scr" in blocking


def test_intraday_head_is_not_replayable() -> None:
    r = classify_source("intraday", INTRADAY_HEAD)
    assert r.status is ParityStatus.NOT_REPLAYABLE
    assert "intraday_microstructure" in {i.name for i in r.unavailable_inputs()}


def test_indicators_are_reconstructable() -> None:
    r = classify_source("ind", BARS_WITH_INDICATORS)
    assert r.status is ParityStatus.REPLAYABLE
    assert "technical_indicator" in {i.name for i in r.inputs}


def test_bare_iv_token_is_caught() -> None:
    """A head reading `w.iv` (implied vol) alongside bars is NOT_REPLAYABLE, not REPLAYABLE."""
    r = classify_source("ivhead", "def decide(w):\n    return w.bars, w.iv")
    assert r.status is ParityStatus.NOT_REPLAYABLE
    assert "options_iv_greeks" in {i.name for i in r.unavailable_inputs()}


# --------------------------------------------------------------------------- #
# Unparseable / unreadable source → UNKNOWN (analyzed=False).
# --------------------------------------------------------------------------- #
def test_syntax_error_source_is_unknown() -> None:
    r = classify_source("broken", SYNTAX_ERROR)
    assert r.status is ParityStatus.UNKNOWN
    assert r.analyzed is False
    assert is_backtestable(r) is False


def test_unreadable_file_is_unknown(tmp_path) -> None:
    missing = tmp_path / "does_not_exist.py"
    r = classify_head_file("ghost", missing)
    assert r.status is ParityStatus.UNKNOWN
    assert r.analyzed is False
    assert r.source_ref == str(missing)


def test_classify_head_file_reads_source(tmp_path) -> None:
    f = tmp_path / "head.py"
    f.write_text(BARS_ONLY, encoding="utf-8")
    r = classify_head_file("filehead", f)
    assert r.status is ParityStatus.REPLAYABLE
    assert r.source_ref == str(f)


# --------------------------------------------------------------------------- #
# Batch + rendering + serialisation.
# --------------------------------------------------------------------------- #
def test_audit_heads_is_sorted_and_handles_missing(tmp_path) -> None:
    good = tmp_path / "g.py"
    good.write_text(BARS_ONLY, encoding="utf-8")
    results = audit_heads({"zeta": good, "alpha": tmp_path / "missing.py"})
    assert [r.head for r in results] == ["alpha", "zeta"]  # sorted by head name
    assert results[0].status is ParityStatus.UNKNOWN       # missing file
    assert results[1].status is ParityStatus.REPLAYABLE


def test_render_parity_summary_has_tally() -> None:
    results = [
        classify_source("bars_only", BARS_ONLY),
        classify_source("newsy", NEWS_HEAD),
        classify_source("opaque", UNDETECTABLE_HEAD),
        classify_source("regimey", APPROXIMABLE_HEAD),
    ]
    out = render_parity_summary(results)
    assert "TOTALS (4 heads)" in out
    assert "REPLAYABLE=1" in out
    assert "NOT_REPLAYABLE=1" in out
    assert "UNKNOWN=1" in out
    assert "APPROXIMABLE=1" in out
    # Deterministic: rows are sorted by head name.
    assert out.index("bars_only") < out.index("newsy") < out.index("opaque")


def test_result_to_dict_is_json_serialisable() -> None:
    r = classify_source("mixed", BARS_PLUS_REGIME_PLUS_NEWS)
    d = r.to_dict()
    round_tripped = json.loads(json.dumps(d))
    assert round_tripped["status"] == "NOT_REPLAYABLE"
    assert round_tripped["category_counts"]["unavailable"] >= 1
    assert isinstance(round_tripped["inputs"], list)


# --------------------------------------------------------------------------- #
# Determinism + evidence hygiene.
# --------------------------------------------------------------------------- #
def test_classification_is_deterministic() -> None:
    a = classify_source("h", NEWS_HEAD).to_dict()
    b = classify_source("h", NEWS_HEAD).to_dict()
    assert a == b


def test_evidence_is_sorted_and_deduped() -> None:
    src = "def decide(ctx):\n    return ctx.bars, ctx.bars, ctx.closes, ctx.ohlc"
    r = classify_source("ev", src)
    daily = next(i for i in r.inputs if i.name == "daily_bars")
    assert list(daily.evidence) == sorted(set(daily.evidence))
    assert "bars" in daily.evidence


def test_inputs_sorted_by_category_then_name() -> None:
    r = classify_source("mixed", BARS_PLUS_REGIME_PLUS_NEWS)
    order = [(i.category.value, i.name) for i in r.inputs]
    # reconstructable(0) before approximable(1) before unavailable(2).
    cat_rank = {"reconstructable": 0, "approximable": 1, "unavailable": 2}
    ranks = [cat_rank[c] for c, _ in order]
    assert ranks == sorted(ranks)


# --------------------------------------------------------------------------- #
# is_backtestable matrix.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "status,default,with_approx",
    [
        (ParityStatus.REPLAYABLE, True, True),
        (ParityStatus.APPROXIMABLE, False, True),
        (ParityStatus.NOT_REPLAYABLE, False, False),
        (ParityStatus.UNKNOWN, False, False),
    ],
)
def test_is_backtestable_matrix(status, default, with_approx) -> None:
    r = FeatureParityResult(
        head="h", status=status, inputs=(), reasons=(), analyzed=True, source_ref=None
    )
    assert is_backtestable(r) is default
    assert is_backtestable(r, allow_approximable=True) is with_approx


# --------------------------------------------------------------------------- #
# Catalog integrity + InputSignal validation.
# --------------------------------------------------------------------------- #
def test_default_catalog_is_well_formed() -> None:
    names = [s.name for s in DEFAULT_SIGNAL_CATALOG]
    assert len(names) == len(set(names)), "duplicate signal names in catalog"
    cats = {s.category for s in DEFAULT_SIGNAL_CATALOG}
    # All three categories represented.
    assert cats == {
        InputCategory.RECONSTRUCTABLE,
        InputCategory.APPROXIMABLE,
        InputCategory.UNAVAILABLE,
    }
    # At least one category-(c) signal so heads can be blocked at all.
    assert any(s.category is InputCategory.UNAVAILABLE for s in DEFAULT_SIGNAL_CATALOG)


def test_input_signal_rejects_compound_ident_token() -> None:
    with pytest.raises(ValueError):
        InputSignal(
            name="bad", category=InputCategory.UNAVAILABLE,
            idents=("has_underscore",), texts=(), rationale="x",
        )


def test_input_signal_rejects_no_tokens() -> None:
    with pytest.raises(ValueError):
        InputSignal(
            name="empty", category=InputCategory.RECONSTRUCTABLE,
            idents=(), texts=(), rationale="x",
        )


def test_input_signal_rejects_uppercase_text_token() -> None:
    with pytest.raises(ValueError):
        InputSignal(
            name="up", category=InputCategory.UNAVAILABLE,
            idents=(), texts=("Runtime/",), rationale="x",
        )


def test_custom_catalog_changes_classification() -> None:
    """A caller-supplied catalog is honoured (the classifier is not hard-wired)."""
    only_bars = (
        InputSignal(
            name="daily_bars", category=InputCategory.RECONSTRUCTABLE,
            idents=("bars",), texts=(), rationale="bars",
        ),
    )
    # With a catalog that has no news signal, the news head degrades to REPLAYABLE
    # (it still reads bars) — proving the catalog, not a hidden rule, drives it.
    r = classify_source("newsy", NEWS_HEAD, catalog=only_bars)
    assert r.status is ParityStatus.REPLAYABLE


def test_classify_source_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        classify_source("", BARS_ONLY)
    with pytest.raises(ValueError):
        classify_source("h", 123)  # type: ignore[arg-type]


def test_detected_input_to_dict() -> None:
    di = DetectedInput(
        name="daily_bars", category=InputCategory.RECONSTRUCTABLE,
        evidence=("bars",), rationale="r",
    )
    assert di.to_dict()["category"] == "reconstructable"
