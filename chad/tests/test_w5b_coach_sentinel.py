"""
W5B-5 — coach streak NOTIFY + sentinel visibility.

The coach half exists to NOT flood: alerting fires only when the same limit
dimension would-rejects N consecutive entries, with a value-free dedupe key
(CTF-T2). The sentinel half makes the allocator heartbeat a watched feed and
pins its schema, so a dead or shape-drifted publisher is loud rather than
silently absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chad.risk.allocator_limits import PortfolioLimits
from chad.risk.allocator_shadow_gate import (
    MODE_OFF,
    MODE_SHADOW,
    AllocatorShadowGate,
    dedupe_identity,
    maybe_notify_reject_streak,
)
from chad.tests.test_w5b_exposure_core import REAL_BOOK, REAL_PRICES

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture()
def sectors():
    from chad.risk.fuse_box import load_sector_map, make_sector_lookup

    return make_sector_lookup(load_sector_map())


@pytest.fixture()
def gate(tmp_path, sectors):
    return AllocatorShadowGate(
        mode=MODE_SHADOW, positions=REAL_BOOK, prices=REAL_PRICES,
        sector_lookup=sectors, limits=PortfolioLimits.load(),
        evidence_dir=tmp_path / "ev",
    )


class _Spy:
    def __init__(self):
        self.calls = []

    def __call__(self, msg, **kw):
        self.calls.append((msg, kw))


def _lly(qty=10):
    """An LLY add — always a per_symbol would-reject (the position is already
    $215k against an enforced $150k cap)."""
    return {"symbol": "LLY", "side": "BUY", "quantity": qty, "sec_type": "STK",
            "limit_price": 1184.0, "strategy": "gamma", "meta": {}}


# --------------------------------------------------------------------------- #
# Streak gating — the anti-flood contract
# --------------------------------------------------------------------------- #

def test_no_notify_below_threshold(gate):
    spy = _Spy()
    for _ in range(2):
        gate.observe(_lly())
    assert maybe_notify_reject_streak(gate, notify_fn=spy) == 0
    assert spy.calls == []


def test_notify_at_threshold(gate):
    spy = _Spy()
    for _ in range(3):
        gate.observe(_lly())
    assert gate.reject_streaks["per_symbol"] == 3
    assert maybe_notify_reject_streak(gate, notify_fn=spy) == 1
    assert len(spy.calls) == 1


def test_one_message_per_dimension_not_per_intent(gate):
    """The R13 flood: ten consecutive would-rejects must be ONE message."""
    spy = _Spy()
    for _ in range(10):
        gate.observe(_lly())
    assert maybe_notify_reject_streak(gate, notify_fn=spy) == 1
    assert len(spy.calls) == 1


def test_approval_breaks_the_streak(gate):
    spy = _Spy()
    for _ in range(3):
        gate.observe(_lly())
    gate.observe({"symbol": "AAPL", "side": "BUY", "quantity": 1,
                  "sec_type": "STK", "limit_price": 321.01,
                  "strategy": "gamma", "meta": {}})
    assert gate.reject_streaks["per_symbol"] == 0
    assert maybe_notify_reject_streak(gate, notify_fn=spy) == 0


def test_bypassed_closes_do_not_affect_the_streak(gate):
    """A close is not evaluated at all, so it can neither build nor break a
    streak — the counters must stay entries-only."""
    spy = _Spy()
    for _ in range(3):
        gate.observe(_lly())
    gate.observe({"symbol": "LLY", "side": "EXIT", "quantity": 182,
                  "sec_type": "STK", "meta": {"action": "CLOSE"}})
    assert gate.reject_streaks["per_symbol"] == 3
    assert maybe_notify_reject_streak(gate, notify_fn=spy) == 1


def test_off_gate_never_notifies(tmp_path, sectors):
    spy = _Spy()
    off = AllocatorShadowGate(mode=MODE_OFF, positions=REAL_BOOK,
                              prices=REAL_PRICES, sector_lookup=sectors,
                              evidence_dir=tmp_path / "ev")
    assert maybe_notify_reject_streak(off, notify_fn=spy) == 0
    assert maybe_notify_reject_streak(None, notify_fn=spy) == 0


# --------------------------------------------------------------------------- #
# Dedupe identity (CTF-T2)
# --------------------------------------------------------------------------- #

def test_dedupe_identity_is_value_free_and_stable():
    """Digits stripped, values never in the key. A key carrying the streak
    count or a dollar amount would change every time the number moved and
    defeat the dedupe — which is how the R13 flood happened."""
    assert dedupe_identity("per_symbol") == "allocator_reject_per_symbol"
    assert dedupe_identity("gross") == dedupe_identity("gross")
    assert dedupe_identity("per_sector3") == dedupe_identity("per_sector99")
    assert dedupe_identity("GROSS") == "allocator_reject_gross"


def test_dedupe_key_used_on_send(gate):
    spy = _Spy()
    for _ in range(3):
        gate.observe(_lly())
    maybe_notify_reject_streak(gate, notify_fn=spy)
    _, kw = spy.calls[0]
    assert kw["dedupe_key"] == "allocator_reject_per_symbol"
    assert kw["raise_on_fail"] is False
    assert kw["severity"] == "warning"


def test_dimensions_have_distinct_dedupe_keys():
    keys = {dedupe_identity(d) for d in
            ("gross", "net", "per_symbol", "per_sector", "venue")}
    assert len(keys) == 5


# --------------------------------------------------------------------------- #
# The coach message
# --------------------------------------------------------------------------- #

def test_coach_message_says_nothing_was_blocked(gate):
    spy = _Spy()
    for _ in range(3):
        gate.observe(_lly())
    maybe_notify_reject_streak(gate, notify_fn=spy)
    msg = spy.calls[0][0]
    assert "Nothing was stopped" in msg or "nothing was blocked" in msg.lower()


def test_coach_message_flags_an_unratified_ceiling():
    """A derived shadow threshold must not read as a ratified limit.

    The caveat rides on the STANDARD context line, which is where the coach
    puts provenance — SIMPLE stays calm and says only that nothing was
    blocked. Both registers are asserted so neither can drift.
    """
    from chad.utils.coach_voice import format_alert

    derived_facts = {
        "dimension": "per_sector", "streak": 3, "cap_usd": 375000,
        "basis": "shadow_derivation_2026-07", "ratified": False,
    }
    standard = format_alert("allocator_reject_streak", derived_facts,
                            mode="STANDARD")
    assert "not a limit that's been signed off" in standard

    ratified = format_alert("allocator_reject_streak", {
        "dimension": "per_symbol", "streak": 3, "cap_usd": 150000,
        "basis": "sourced", "ratified": True,
    }, mode="STANDARD")
    assert "ratified limit" in ratified

    # SIMPLE keeps the calm register but must still disclaim enforcement.
    simple = format_alert("allocator_reject_streak", derived_facts,
                          mode="SIMPLE")
    assert "Nothing was stopped" in simple


def test_numbers_stay_out_of_the_headline():
    """CTF-T2/coach doctrine: counts and dollars live in the PRO line and the
    evidence, never the headline."""
    from chad.utils.coach_voice import format_alert

    simple = format_alert("allocator_reject_streak", {
        "dimension": "gross", "streak": 7, "cap_usd": 750000,
        "basis": "shadow_derivation_2026-07", "ratified": False,
    }, mode="SIMPLE")
    assert "750000" not in simple and "$750,000" not in simple

    pro = format_alert("allocator_reject_streak", {
        "dimension": "gross", "streak": 7, "cap_usd": 750000,
        "basis": "shadow_derivation_2026-07", "ratified": False,
    }, mode="PRO")
    assert "750000" in pro and "streak=7" in pro


def test_notify_failure_is_not_fatal(gate):
    def boom(*a, **k):
        raise OSError("telegram down")

    for _ in range(3):
        gate.observe(_lly())
    assert maybe_notify_reject_streak(gate, notify_fn=boom) == 0  # no raise


def test_template_is_registered():
    from chad.utils.coach_voice import _TEMPLATES

    assert "allocator_reject_streak" in _TEMPLATES


# --------------------------------------------------------------------------- #
# Sentinel visibility
# --------------------------------------------------------------------------- #

def _exterminator():
    return json.loads((REPO / "config" / "exterminator.json").read_text(encoding="utf-8"))


def test_feeds_row_exists_and_matches_the_publisher():
    cfg = _exterminator()
    row = cfg["feeds"]["portfolio_allocator_state"]
    assert row["path"] == "runtime/portfolio_allocator_state.json"
    assert row["ts_field"] == "ts_utc"
    assert row["ttl_verified"] is True

    from chad.risk.allocator_shadow_gate import STATE_TTL_SECONDS

    assert row["ttl_seconds"] == STATE_TTL_SECONDS
    assert row["warn_after_seconds"] == STATE_TTL_SECONDS
    assert row["fail_after_seconds"] > row["warn_after_seconds"]


def test_exs7_pin_matches_the_published_shape(tmp_path):
    """The pin must describe what build_state actually emits — a pin that
    drifts from its publisher is worse than no pin."""
    from chad.risk.allocator_shadow_gate import build_state, publish_state

    cfg = _exterminator()
    pin = cfg["schema_contracts"]["enforced"]["runtime/portfolio_allocator_state.json"]

    p = tmp_path / "state.json"
    publish_state(build_state(None), p)
    payload = json.loads(p.read_text(encoding="utf-8"))

    assert payload["schema_version"] == pin["schema_version"]
    assert payload["schema_version"] in pin["accepts"]
    for key in pin["required_keys"]:
        assert key in payload, f"pin requires {key!r} but build_state omits it"


def test_standing_findings_is_a_required_pinned_key():
    """§13.4: the findings bound what the evidence may claim. A payload that
    lost them would read as an unqualified exposure report, so the pin must
    make their absence a schema break."""
    cfg = _exterminator()
    pin = cfg["schema_contracts"]["enforced"]["runtime/portfolio_allocator_state.json"]
    assert "standing_findings" in pin["required_keys"]


def test_sentinel_check_count_unchanged():
    """W5B adds config rows consumed by the EXISTING EXS1/EXS7 checks. It adds
    no new check, so the EXS1..EXS10 roster must be untouched."""
    from chad.ops import exterminator_sentinel as sentinel_mod

    src = Path(sentinel_mod.__file__).read_text(encoding="utf-8")
    for i in range(1, 11):
        assert f"EXS{i}" in src
    assert "EXS11" not in src


def test_config_is_valid_json_and_w5a_keys_intact():
    """W5B and W5A both write config/exterminator.json. The keys are disjoint;
    this pins that W5B's rows did not disturb Lane A's."""
    cfg = _exterminator()
    assert "clock_health" in cfg
    assert "embedded" in cfg["schema_contracts"]
    assert "implementation_shortfall.v1" in cfg["schema_contracts"]["embedded"]
    assert "mae_mfe.v1" in cfg["schema_contracts"]["embedded"]
