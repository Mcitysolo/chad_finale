"""PA-EP3 — thread the intent idempotency_key into evidence execution_id so
the slippage.v1 and signal_decay records carry a non-empty join key.

Background (Evidence Pipeline survey EP-3):
    chad/core/live_loop.py:2434 constructs the PaperExecEvidence for every
    paper fill but never passes ``execution_id``, so it defaults to "" at
    paper_exec_evidence_writer.py:509. The two downstream consumers —
        - slippage tracker:  paper_exec_evidence_writer.py:847
                             (intent_id=_safe_str(ev.execution_id, ""))
        - signal decay:      paper_exec_evidence_writer.py:866
                             (intent_id=_safe_str(ev.execution_id, ""))
    therefore stamp intent_id="" on 100% of records, orphaning slippage and
    decay evidence from its originating intent.

    The fix is a single line at live_loop.py:2434 — derive execution_id from
    the canonical intent identifier (idempotency_key, trace_id fallback; same
    precedence as routing_gates.py:437-439). Because BOTH consumers read
    ev.execution_id, the one change fixes the slippage AND the signal_decay
    join simultaneously; there is no second edit.

Test layers:
    1. CALL-SITE GUARD (red-before / green-after): inspects the real
       live_loop.run_once source and asserts the PaperExecEvidence(...)
       construction threads execution_id from idempotency_key. This is the
       only layer coupled to the source line; it FAILS before the fix.
    2. CHARACTERIZATION: locks the idempotency_key -> execution_id resolution
       expression (and its trace_id / empty fallbacks).
    3. PROPAGATION (end-to-end): drives the real write_paper_exec_evidence and
       asserts the resolved execution_id lands in both the slippage.v1 ledger
       and the signal_decay ledger — and that the legacy no-thread construction
       yields intent_id="" (documents the bug being fixed).
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

# Importing live_loop triggers a module-level IB connect unless this is set
# (see live_loop.py:114-120). Mirror the gap039 run_once test's guard.
os.environ.setdefault("CHAD_SKIP_IB_CONNECT", "1")

import chad.analytics.signal_decay as decay_mod  # noqa: E402
import chad.analytics.slippage_tracker as slip_mod  # noqa: E402
import chad.core.live_loop as live_loop  # noqa: E402
import chad.execution.paper_exec_evidence_writer as wmod  # noqa: E402
from chad.analytics.signal_decay import SignalDecayRecorder  # noqa: E402
from chad.analytics.slippage_tracker import SlippageTracker  # noqa: E402
from chad.execution.paper_exec_evidence_writer import (  # noqa: E402
    PaperExecEvidence,
    write_paper_exec_evidence,
)

TEST_KEY = "TEST-KEY-123"


# ---------------------------------------------------------------------------
# Shared helper: the canonical resolution expression as applied at the
# live_loop.py:2434 construction site (idempotency_key, trace_id fallback).
# Kept as one function so the characterization and propagation layers exercise
# the *same* expression the source change introduces.
# ---------------------------------------------------------------------------
def _resolve_execution_id(intent) -> str:
    return (
        getattr(intent, "idempotency_key", "")
        or getattr(intent, "trace_id", "")
        or ""
    )


def _build_ev_like_live_loop(intent, *, thread_execution_id: bool) -> PaperExecEvidence:
    """Reproduce the live_loop.py:2434 PaperExecEvidence construction.

    thread_execution_id=False reproduces the pre-fix construction (no
    execution_id kwarg); True reproduces the post-fix construction.
    """
    kwargs = dict(
        symbol="SPY",
        side="BUY",
        quantity=10.0,
        fill_price=400.0,
        expected_price=399.5,
        strategy=getattr(intent, "strategy", "") or "alpha",
        source_strategies=[getattr(intent, "strategy", "") or "alpha"],
        broker="ibkr_paper",
        status="paper_fill",
        asset_class="stock",
        is_live=False,
        fill_time_utc="2026-06-12T00:00:00Z",
        extra={},
    )
    if thread_execution_id:
        kwargs["execution_id"] = _resolve_execution_id(intent)
    return PaperExecEvidence(**kwargs)


# ===========================================================================
# Layer 1 — CALL-SITE GUARD (red-before / green-after)
# ===========================================================================
def test_callsite_threads_execution_id_from_idempotency_key():
    """The real live_loop.run_once PaperExecEvidence(...) construction must
    thread execution_id from the intent idempotency_key.

    This is the mechanical guard the operator requested: it is coupled to the
    actual source line, so it FAILS before the PA-EP3 edit and PASSES after.
    """
    src = inspect.getsource(live_loop.run_once)
    anchor = "PaperExecEvidence("
    assert anchor in src, "live_loop.run_once no longer constructs PaperExecEvidence"
    start = src.index(anchor)
    end = src.index("normalize_paper_fill_evidence(ev)", start)
    ctor = src[start:end]
    assert "execution_id" in ctor, (
        "PA-EP3 regression: live_loop.run_once PaperExecEvidence(...) does not "
        "pass execution_id — slippage.v1 / signal_decay intent_id will be empty"
    )
    assert "idempotency_key" in ctor, (
        "PA-EP3 regression: execution_id is not derived from the intent "
        "idempotency_key at the live_loop construction site"
    )


# ===========================================================================
# Layer 2 — CHARACTERIZATION of the resolution expression
# ===========================================================================
def test_resolution_prefers_idempotency_key():
    intent = SimpleNamespace(idempotency_key=TEST_KEY, trace_id="trace-zzz")
    assert _resolve_execution_id(intent) == TEST_KEY


def test_resolution_falls_back_to_trace_id():
    intent = SimpleNamespace(idempotency_key=None, trace_id="trace-zzz")
    assert _resolve_execution_id(intent) == "trace-zzz"


def test_resolution_empty_when_no_identifier():
    intent = SimpleNamespace(idempotency_key=None)
    assert _resolve_execution_id(intent) == ""


# ===========================================================================
# Layer 3 — PROPAGATION end-to-end through the real writer
# ===========================================================================
@pytest.fixture
def hermetic_sinks(tmp_path: Path, monkeypatch):
    """Redirect every disk sink the writer touches to tmp, and point the
    slippage tracker + signal-decay recorder at tmp ledgers so we can read
    back the intent_id the real propagation code wrote."""
    # Fill / fee / metric output dirs.
    monkeypatch.setattr(wmod, "FILLS_DIR", tmp_path / "fills", raising=True)
    monkeypatch.setattr(wmod, "FEES_DIR", tmp_path / "fees", raising=True)
    monkeypatch.setattr(wmod, "EXEC_METRICS_DIR", tmp_path / "metrics", raising=True)

    # Price cache so normalize_paper_fill_evidence keeps a clean SPY fill.
    cache = tmp_path / "price_cache.json"
    cache.write_text(json.dumps({
        "prices": {"SPY": 400.0},
        "ts_utc": "2026-06-12T00:00:00Z",
        "ttl_seconds": 300,
    }))
    monkeypatch.setattr(wmod, "PRICE_CACHE_PATH", cache, raising=True)

    # Real tracker + recorder, tmp-backed. The writer imports get_default_*
    # at call time, so patching the module attribute is sufficient.
    tracker = SlippageTracker(ledger_dir=tmp_path / "slippage")
    recorder = SignalDecayRecorder(
        ledger_dir=tmp_path / "decay",
        bars_dir=tmp_path / "bars",
    )
    monkeypatch.setattr(slip_mod, "get_default_tracker", lambda: tracker, raising=True)
    monkeypatch.setattr(decay_mod, "get_default_recorder", lambda: recorder, raising=True)

    return SimpleNamespace(tracker=tracker, decay_dir=tmp_path / "decay")


def _read_decay_intent_ids(decay_dir: Path, strategy: str = "alpha"):
    path = decay_dir / f"{strategy}_decay.ndjson"
    if not path.is_file():
        return []
    return [
        json.loads(line)["intent_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_propagation_threaded_execution_id_reaches_both_ledgers(hermetic_sinks):
    """Post-fix construction: idempotency_key flows to slippage.v1.intent_id
    AND signal_decay.intent_id via the real write path."""
    intent = SimpleNamespace(idempotency_key=TEST_KEY, strategy="alpha")
    ev = _build_ev_like_live_loop(intent, thread_execution_id=True)

    write_paper_exec_evidence(ev)

    slip = hermetic_sinks.tracker.read_ledger()
    assert slip, "no slippage record written"
    assert slip[-1]["schema_version"] == "slippage.v1"
    assert slip[-1]["intent_id"] == TEST_KEY

    decay_ids = _read_decay_intent_ids(hermetic_sinks.decay_dir)
    assert decay_ids, "no signal_decay record written"
    assert decay_ids[-1] == TEST_KEY


def test_propagation_legacy_construction_orphans_records(hermetic_sinks):
    """Pre-fix construction (no execution_id kwarg) reproduces the EP-3 bug:
    both ledgers stamp intent_id="" — the orphaning this PA eliminates."""
    intent = SimpleNamespace(idempotency_key=TEST_KEY, strategy="alpha")
    ev = _build_ev_like_live_loop(intent, thread_execution_id=False)

    write_paper_exec_evidence(ev)

    slip = hermetic_sinks.tracker.read_ledger()
    assert slip, "no slippage record written"
    assert slip[-1]["intent_id"] == ""

    decay_ids = _read_decay_intent_ids(hermetic_sinks.decay_dir)
    assert decay_ids, "no signal_decay record written"
    assert decay_ids[-1] == ""
