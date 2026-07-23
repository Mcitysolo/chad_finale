"""
W5B-3 — shadow gate, placement, and the three named invariant tests.

  test_w5b_allocator_exits_always_free      — the prime invariant
  test_w5b_allocator_byte_identical_when_off — OFF costs nothing and writes nothing
  test_w5b_shadow_never_blocks               — no should_block-True path exists

Plus: the bypass predicate is the fuse gate's (not a copy), the provisional
book accumulates regardless of verdict, evidence rows are well-formed, the
pytest leak guard fires on a default path, and no failure mode raises upward.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from chad.risk.allocator_limits import (
    ERROR,
    WOULD_APPROVE,
    WOULD_REJECT,
    PortfolioLimits,
)
from chad.risk.allocator_shadow_gate import (
    DEFAULT_EVIDENCE_DIR,
    EVIDENCE_SCHEMA,
    MODE_OFF,
    MODE_SHADOW,
    AllocatorShadowGate,
    allocator_mode,
    append_evidence,
)
from chad.tests.test_w5b_exposure_core import REAL_BOOK, REAL_PRICES


@pytest.fixture()
def sectors():
    from chad.risk.fuse_box import load_sector_map, make_sector_lookup

    return make_sector_lookup(load_sector_map())


@pytest.fixture()
def gate(tmp_path, sectors):
    return AllocatorShadowGate(
        mode=MODE_SHADOW,
        positions=REAL_BOOK,
        prices=REAL_PRICES,
        sector_lookup=sectors,
        limits=PortfolioLimits.load(),
        evidence_dir=tmp_path / "ev",
    )


def _rows(tmp_path) -> list:
    d = tmp_path / "ev"
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.ndjson")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def _entry(symbol="AAPL", qty=10, price=321.01, side="BUY"):
    return {"symbol": symbol, "side": side, "quantity": qty, "sec_type": "STK",
            "limit_price": price, "strategy": "gamma", "meta": {}}


# --------------------------------------------------------------------------- #
# NAMED INVARIANT 1 — exits are always free
# --------------------------------------------------------------------------- #

def test_w5b_allocator_exits_always_free(gate, tmp_path):
    """The allocator never evaluates a close: not an overlay close, not an
    advice-fired close, not a flatten, not a flip, not a protective reduce.

    Asserted two ways: zero verdicts AND zero evidence rows (a row for a close
    would let the corpus be misread as having gated an exit), plus every
    intent byte-identical afterwards.
    """
    closes = [
        # W4B-2 close-provenance stamp (overlay / advice-fired closes)
        {"symbol": "LLY", "side": "SELL", "quantity": 182, "sec_type": "STK",
         "limit_price": 1184.0, "strategy": "gamma", "meta": {"action": "CLOSE"}},
        # explicit EXIT / CLOSE sides
        {"symbol": "SPY", "side": "EXIT", "quantity": 247, "sec_type": "STK",
         "limit_price": 739.42, "strategy": "gamma", "meta": {}},
        {"symbol": "UNH", "side": "CLOSE", "quantity": 240, "sec_type": "STK",
         "limit_price": 422.70, "strategy": "gamma", "meta": {}},
        # flatten-drill shaped intents (INC-0723 salient)
        {"symbol": "IWM", "side": "SELL", "quantity": 200, "sec_type": "STK",
         "limit_price": 291.97, "strategy": "flatten",
         "meta": {"action": "CLOSE", "reason": "liquidation"}},
        # protective reduce / stop-loss
        {"symbol": "BAC", "side": "SELL", "quantity": 213, "sec_type": "STK",
         "limit_price": 61.28, "strategy": "gamma",
         "meta": {"reason": "stop_loss"}},
        {"symbol": "MSFT", "side": "SELL", "quantity": 34, "sec_type": "STK",
         "limit_price": 381.65, "strategy": "gamma",
         "meta": {"tags": ["risk_reduction"]}},
        {"symbol": "V", "side": "SELL", "quantity": 195, "sec_type": "STK",
         "limit_price": 350.76, "strategy": "gamma", "meta": {"reduce": True}},
        {"symbol": "MA", "side": "SELL", "quantity": 10, "sec_type": "STK",
         "limit_price": 530.06, "strategy": "gamma", "meta": {"exit": True}},
    ]
    before = copy.deepcopy(closes)
    gross_before = gate.book.gross_usd

    for intent in closes:
        assert gate.observe(intent) is None
        assert gate.should_block(intent) is False

    assert closes == before, "the allocator must never mutate an intent"
    assert _rows(tmp_path) == [], "a close must produce no evidence row"
    assert gate.counts["evaluated"] == 0
    assert gate.counts["bypassed"] == len(closes)
    assert gate.book.gross_usd == gross_before, "a close must not enter the book"


def test_flatten_all_sentinel_batch_is_free(gate, tmp_path):
    """A whole flatten-all batch, shaped like the W4B CLI's, must pass
    untouched. INC-0723 is why this is its own test."""
    batch = [
        {"symbol": s, "side": "SELL", "quantity": q, "sec_type": "STK",
         "limit_price": 1.0, "strategy": "flatten",
         "meta": {"action": "CLOSE", "reason": "liquidation",
                  "flatten_token": "FLATTEN_ALL"}}
        for s, q in [("AAPL", 12), ("LLY", 182), ("SPY", 247), ("UNH", 240)]
    ]
    for intent in batch:
        assert gate.observe(intent) is None
    assert _rows(tmp_path) == []
    assert gate.counts["evaluated"] == 0


def test_bypass_predicate_is_the_fuse_gates_own(gate):
    """C1: the predicate is IMPORTED from fuse_gate, never re-implemented, so
    the allocator and the fuse can never disagree about what a close is."""
    import chad.risk.fuse_gate as fg

    calls = []
    original = fg.is_exit_like

    def spy(intent):
        calls.append(intent)
        return original(intent)

    fg.is_exit_like = spy
    try:
        gate.observe(_entry())
    finally:
        fg.is_exit_like = original
    assert len(calls) == 1, "observe() must consult fuse_gate.is_exit_like"


def test_bypass_fails_toward_close(gate, tmp_path):
    """`is_exit_like` fails toward True. An intent that raises on attribute
    access must be treated as a close and produce no row — never blocked, and
    never evaluated on uncertainty."""

    class Exploding:
        def __getattr__(self, name):
            raise RuntimeError("boom")

    assert gate.observe(Exploding()) is None
    assert _rows(tmp_path) == []


# --------------------------------------------------------------------------- #
# NAMED INVARIANT 2 — byte-identical when off
# --------------------------------------------------------------------------- #

def test_w5b_allocator_byte_identical_when_off(tmp_path, sectors):
    """OFF must read nothing, write nothing, and evaluate nothing."""
    gate = AllocatorShadowGate(
        mode=MODE_OFF,
        positions=REAL_BOOK,
        prices=REAL_PRICES,
        sector_lookup=sectors,
        evidence_dir=tmp_path / "ev",
    )
    assert gate.active is False
    assert gate.book is None, "OFF must not even snapshot the book"
    assert gate.limits is None, "OFF must not read the limits config"

    for _ in range(5):
        assert gate.observe(_entry()) is None
        assert gate.should_block(_entry()) is False

    assert _rows(tmp_path) == []
    assert gate.counts["evaluated"] == 0


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", MODE_OFF), ("off", MODE_OFF), ("shadow", MODE_SHADOW),
        ("SHADOW", MODE_SHADOW), (" shadow ", MODE_SHADOW),
        ("garbage", MODE_OFF), ("1", MODE_OFF), ("true", MODE_OFF),
    ],
)
def test_mode_parse(raw, expected):
    assert allocator_mode({"CHAD_ALLOCATOR": raw}) == expected


def test_enforce_is_refused_not_honored(caplog):
    """There is no enforce path in W5B. An operator who sets enforce expecting
    enforcement must be told loudly, not silently run in shadow."""
    import logging

    with caplog.at_level(logging.WARNING):
        assert allocator_mode({"CHAD_ALLOCATOR": "enforce"}) == MODE_OFF
    assert "ALLOCATOR_ENFORCE_REFUSED" in caplog.text


def test_no_enforce_string_in_module_modes():
    from chad.risk import allocator_shadow_gate as m

    assert m._VALID_MODES == {"off", "shadow"}
    assert not hasattr(m, "MODE_ENFORCE")


# --------------------------------------------------------------------------- #
# NAMED INVARIANT 3 — shadow never blocks
# --------------------------------------------------------------------------- #

def test_w5b_shadow_never_blocks(gate):
    """should_block() is False for every input, including one that produces a
    WOULD_REJECT — the verdict and the block decision are not connected."""
    rejecting = _entry("LLY", 100, 1184.0)
    verdict = gate.observe(rejecting)
    assert verdict is not None and verdict.verdict == WOULD_REJECT
    assert gate.should_block(rejecting) is False

    for intent in (_entry(), rejecting, _entry("NVDA", 100000, 500.0),
                   {"garbage": True}, None):
        assert gate.should_block(intent) is False


def test_should_block_does_not_consult_the_verdict(gate, monkeypatch):
    """Structural: should_block must not even call the evaluator."""
    import chad.risk.allocator_shadow_gate as m

    def explode(*a, **k):
        raise AssertionError("should_block must not evaluate")

    monkeypatch.setattr(m, "evaluate_marginal", explode)
    assert gate.should_block(_entry()) is False


def test_intent_quantity_never_mutated(gate):
    for symbol, qty, price in (("AAPL", 10, 321.01), ("LLY", 100, 1184.0),
                               ("UNH", 150, 422.70)):
        intent = _entry(symbol, qty, price)
        snapshot = dict(intent)
        gate.observe(intent)
        assert intent == snapshot


# --------------------------------------------------------------------------- #
# Provisional book accumulation
# --------------------------------------------------------------------------- #

def test_rejected_intents_still_enter_the_book(gate):
    """Shadow blocks nothing, so the honest counterfactual includes every
    intent. If a would-rejected ticket were dropped, the next correlated one
    would measure against a book missing it and understate concentration."""
    before = gate.book.gross_usd
    v = gate.observe(_entry("LLY", 100, 1184.0))
    assert v.verdict == WOULD_REJECT
    assert gate.book.gross_usd == pytest.approx(before + 118_400.0)


def test_third_correlated_ticket_sees_the_first_two(gate):
    """The corpus must show the 3rd correlated ticket breaching because of the
    first two — that is the whole point of a provisional book."""
    first = gate.observe(_entry("JNJ", 40, 400.0))     # healthcare +$16k
    second = gate.observe(_entry("JNJ", 40, 400.0))    # +$16k
    assert first.verdict == WOULD_APPROVE
    assert second.verdict == WOULD_APPROVE

    third = gate.observe(_entry("JNJ", 100, 400.0))    # +$40k ⇒ sector over
    assert third.verdict in ("WOULD_REJECT", "WOULD_RESIZE")
    assert third.which_limit == "per_sector"


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #

def test_evidence_row_shape(gate, tmp_path):
    gate.observe(_entry("LLY", 100, 1184.0))
    rows = _rows(tmp_path)
    assert len(rows) == 1
    r = rows[0]
    assert r["schema_version"] == EVIDENCE_SCHEMA
    assert r["mode"] == MODE_SHADOW
    assert r["verdict"] == WOULD_REJECT
    assert r["which_limit"] == "per_symbol"
    assert r["symbol"] == "LLY"
    assert r["ts_utc"].endswith("Z")
    assert r["correlation_basis"] == "static_sector_buckets"
    assert r["join"]["kind"] == "soft_correlation_tuple"
    # FINDING W5B-F1: the IBKR lane mints no execution id.
    assert r["join"]["execution_id"] == ""
    assert r["checks"], "per-dimension arithmetic must be recorded"


def test_evidence_records_no_numeric_correlation(gate, tmp_path):
    """§13.3: W5B computes no rho, so no allocator artifact may carry one."""
    gate.observe(_entry())
    r = _rows(tmp_path)[0]
    assert r["correlation_basis"] == "static_sector_buckets"
    assert not any("corr" in k.lower() and k != "correlation_basis" for k in r)


def test_one_row_per_evaluated_intent(gate, tmp_path):
    for i in range(4):
        gate.observe(_entry("AAPL", 1 + i, 321.01))
    assert len(_rows(tmp_path)) == 4
    assert [r["join"]["cycle_seq"] for r in _rows(tmp_path)] == [1, 2, 3, 4]


def test_evidence_leak_guard_fires_on_default_path(sectors):
    """The six repo-write-guard tests are in the worktree baseline failing set,
    so W5B carries its own guard. It must refuse the real data/ path."""
    with pytest.raises(RuntimeError, match="ALLOCATOR_ERROR"):
        append_evidence([{"x": 1}], evidence_dir=None)
    with pytest.raises(RuntimeError, match="ALLOCATOR_ERROR"):
        append_evidence([{"x": 1}], evidence_dir=DEFAULT_EVIDENCE_DIR)


def test_evidence_write_failure_is_not_fatal(gate, tmp_path, monkeypatch):
    """Evidence is best-effort; a write failure must not kill a cycle."""
    import chad.risk.allocator_shadow_gate as m

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(m, "append_evidence", boom)
    assert gate.observe(_entry()) is not None  # verdict still produced


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #

def test_eval_failure_yields_fail_open_error_verdict(gate, tmp_path, monkeypatch):
    import chad.risk.allocator_shadow_gate as m

    def boom(*a, **k):
        raise ValueError("nope")

    monkeypatch.setattr(m, "evaluate_marginal", boom)
    v = gate.observe(_entry())
    assert v.verdict == ERROR
    assert gate.counts[ERROR] == 1
    assert _rows(tmp_path)[0]["verdict"] == ERROR


def test_construction_failure_is_inert(tmp_path, monkeypatch, sectors):
    import chad.risk.allocator_shadow_gate as m

    def boom(*a, **k):
        raise OSError("no book")

    monkeypatch.setattr(m, "build_base_book", boom)
    gate = AllocatorShadowGate(mode=MODE_SHADOW, prices=REAL_PRICES,
                               sector_lookup=sectors,
                               evidence_dir=tmp_path / "ev")
    assert gate.active is False
    assert gate.construction_error is not None
    assert gate.observe(_entry()) is None
    assert gate.should_block(_entry()) is False


def test_build_cycle_gate_defaults_to_off(monkeypatch):
    from chad.risk.allocator_shadow_gate import build_cycle_gate

    monkeypatch.delenv("CHAD_ALLOCATOR", raising=False)
    gate = build_cycle_gate()
    assert gate is not None
    assert gate.mode == MODE_OFF
    assert gate.active is False


# --------------------------------------------------------------------------- #
# Placement
# --------------------------------------------------------------------------- #

def test_live_loop_placement_is_after_the_fuse_gate():
    """The observer must sit AFTER the fuse gate's block+continue, so a
    fuse-blocked entry never enters the provisional book."""
    src = (Path(__file__).resolve().parents[2] / "chad" / "core" / "live_loop.py").read_text(
        encoding="utf-8"
    )
    fuse_call = src.index("_fuse_gate.should_block(intent)")
    observe_call = src.index("_allocator_gate.observe(intent)")
    assert fuse_call < observe_call

    loop_start = src.index("for intent in intents:")
    assert loop_start < observe_call, "the observer must be inside the per-intent loop"


def test_live_loop_never_branches_on_the_observer():
    """No `if _allocator_gate.observe(...)`, no continue, no quantity write —
    the return value is deliberately unused."""
    src = (Path(__file__).resolve().parents[2] / "chad" / "core" / "live_loop.py").read_text(
        encoding="utf-8"
    )
    i = src.index("_allocator_gate.observe(intent)")
    window = src[i - 200: i + 400]
    assert "if _allocator_gate.observe" not in window
    assert "should_block" not in window
    line = src[src.rindex("\n", 0, i) + 1: src.index("\n", i)].strip()
    assert line == "_allocator_gate.observe(intent)", line
