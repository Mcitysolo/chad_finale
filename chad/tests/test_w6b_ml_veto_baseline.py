"""
chad/tests/test_w6b_ml_veto_baseline.py

W6B-4/5 — durable ML veto shadow history and a stratified baseline.

`config/exterminator.json` asserts "no live rate can be computed or compared".
The second half of that claim (no durable artifact, no counter) is true; the
first half is not — ML_SHADOW lines are fully structured. These tests pin the
parse, the idempotent store, and the two properties that keep the resulting
number honest:

  * stratification, because a window that is 94% one strategy is that
    strategy's rate wearing a portfolio label;
  * sufficiency on BOTH sample count and elapsed span, because the first live
    run gathered 526 rows in 5.26 hours — past any n threshold, and still not
    a baseline.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from chad.analytics import ml_veto_shadow_collector as mc

REAL_LINE = (
    "Jul 24 01:08:44 ip-172-31-8-43 python3[3627817]: "
    "2026-07-24 01:08:44,348 [INFO] ML_SHADOW symbol=GOOGL strategy=gamma "
    "intent_class=entry model_version=xgb_veto_20260510_020007 "
    "manifest_hash=sha256:74afb9e1c8f3e0fe4 loss_prob=0.256 threshold=0.65 "
    "would_veto=False final_action=shadow_only "
    "reason=loss_probability_below_threshold"
)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def test_parses_a_real_production_line():
    row = mc.parse_line(REAL_LINE)
    assert row is not None
    assert row["symbol"] == "GOOGL"
    assert row["strategy"] == "gamma"
    assert row["intent_class"] == "entry"
    assert row["loss_prob"] == 0.256
    assert row["threshold"] == 0.65
    assert row["would_veto"] is False
    assert row["schema_version"] == "ml_veto_shadow.v1"
    assert row["ts_utc"].startswith("2026-07-24T01:08:44")


def test_would_veto_true_is_parsed_as_bool():
    row = mc.parse_line(REAL_LINE.replace("would_veto=False", "would_veto=True"))
    assert row["would_veto"] is True


def test_non_shadow_line_returns_none():
    assert mc.parse_line("some other log line entirely") is None


def test_malformed_numeric_returns_none_rather_than_crashing():
    assert mc.parse_line(REAL_LINE.replace("loss_prob=0.256", "loss_prob=NaNsy")) is None


def test_row_key_is_deterministic_and_discriminating():
    a = mc.parse_line(REAL_LINE)
    b = mc.parse_line(REAL_LINE)
    c = mc.parse_line(REAL_LINE.replace("symbol=GOOGL", "symbol=MSFT"))
    assert a["row_key"] == b["row_key"]
    assert a["row_key"] != c["row_key"]


# --------------------------------------------------------------------------
# Idempotent durable store
# --------------------------------------------------------------------------

def test_append_is_idempotent_across_runs(tmp_path):
    """The collector runs on a timer with overlapping --since windows, so
    re-appending the same journal region must not double-count."""
    store = tmp_path / "shadow.ndjson"
    rows = [mc.parse_line(REAL_LINE)]
    assert mc.append_rows(rows, store) == 1
    assert mc.append_rows(rows, store) == 0
    assert len(mc.load_rows(store)) == 1


def test_append_dedupes_within_a_single_batch(tmp_path):
    store = tmp_path / "shadow.ndjson"
    row = mc.parse_line(REAL_LINE)
    assert mc.append_rows([row, row, row], store) == 1


def test_corrupt_line_in_store_does_not_break_reads(tmp_path):
    store = tmp_path / "shadow.ndjson"
    mc.append_rows([mc.parse_line(REAL_LINE)], store)
    with store.open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    assert len(mc.load_rows(store)) == 1


def test_missing_store_reads_as_empty(tmp_path):
    assert mc.load_rows(tmp_path / "nope.ndjson") == []


# --------------------------------------------------------------------------
# Sufficiency needs span, not just n
# --------------------------------------------------------------------------

def _rows(n, *, span_hours, strategy="gamma", vetoes=0, now=None):
    now = now or datetime(2026, 7, 24, 1, 0, tzinfo=timezone.utc)
    out = []
    for i in range(n):
        offset = timedelta(hours=span_hours * (i / max(1, n - 1))) if n > 1 else timedelta()
        ts = (now - timedelta(hours=span_hours)) + offset
        out.append({
            "ts_utc": ts.isoformat().replace("+00:00", "Z"),
            "strategy": strategy, "intent_class": "entry", "symbol": f"S{i}",
            "loss_prob": 0.2, "threshold": 0.65,
            "would_veto": i < vetoes,
            "model_version": "xgb_veto_20260510_020007",
        })
    return out


def test_many_samples_over_a_short_span_is_not_sufficient():
    """The live case: 526 rows in 5.26 hours. Past any n floor, still one
    market session and 94% one strategy."""
    now = datetime(2026, 7, 24, 1, 0, tzinfo=timezone.utc)
    b = mc.compute_baseline(_rows(526, span_hours=5.26), min_samples=200,
                            min_span_hours=72, now=now)
    assert b["n"] == 526
    assert b["sufficient"] is False
    assert any("observed_span_hours" in r for r in b["insufficient_reasons"])


def test_few_samples_over_a_long_span_is_not_sufficient():
    now = datetime(2026, 7, 24, 1, 0, tzinfo=timezone.utc)
    b = mc.compute_baseline(_rows(20, span_hours=100), min_samples=200,
                            min_span_hours=72, now=now)
    assert b["sufficient"] is False
    assert any("min_samples" in r for r in b["insufficient_reasons"])


def test_both_gates_passing_is_sufficient():
    now = datetime(2026, 7, 24, 1, 0, tzinfo=timezone.utc)
    b = mc.compute_baseline(_rows(400, span_hours=100), min_samples=200,
                            min_span_hours=72, now=now)
    assert b["sufficient"] is True
    assert b["insufficient_reasons"] == []


def test_rate_is_still_reported_when_insufficient():
    """Visibility without authority: the number is published so the countdown
    is watchable, but `sufficient` tells a consumer not to wire it."""
    now = datetime(2026, 7, 24, 1, 0, tzinfo=timezone.utc)
    b = mc.compute_baseline(_rows(50, span_hours=2, vetoes=5), min_samples=200,
                            min_span_hours=72, now=now)
    assert b["veto_rate"] == pytest.approx(0.1)
    assert b["sufficient"] is False


# --------------------------------------------------------------------------
# Stratification
# --------------------------------------------------------------------------

def test_strategy_mix_is_visible_so_a_skewed_window_cannot_hide():
    """Measured live: gamma 493 rows with ZERO vetoes, omega_macro 30 rows with
    the single veto. The headline 0.19% is really 'gamma never vetoes' — only
    the stratification shows that."""
    now = datetime(2026, 7, 24, 1, 0, tzinfo=timezone.utc)
    rows = _rows(493, span_hours=100, strategy="gamma", vetoes=0, now=now)
    rows += _rows(30, span_hours=100, strategy="omega_macro", vetoes=1, now=now)
    b = mc.compute_baseline(rows, min_samples=200, min_span_hours=72, now=now)

    assert b["by_strategy"]["gamma"]["n"] == 493
    assert b["by_strategy"]["gamma"]["veto_rate"] == 0.0
    assert b["by_strategy"]["omega_macro"]["vetoes"] == 1
    assert b["by_strategy"]["omega_macro"]["veto_rate"] == pytest.approx(1 / 30)


def test_intent_class_is_stratified_too():
    now = datetime(2026, 7, 24, 1, 0, tzinfo=timezone.utc)
    b = mc.compute_baseline(_rows(10, span_hours=100, now=now), now=now)
    assert b["by_intent_class"]["entry"]["n"] == 10


def test_caveats_name_the_category_error_explicitly():
    b = mc.compute_baseline([])
    joined = " ".join(b["_caveats"])
    assert "val_veto_rate" in joined
    assert "training-time" in joined


# --------------------------------------------------------------------------
# Windowing and degenerate inputs
# --------------------------------------------------------------------------

def test_rows_outside_the_window_are_excluded():
    now = datetime(2026, 7, 24, 1, 0, tzinfo=timezone.utc)
    old = _rows(5, span_hours=1, now=now - timedelta(days=30))
    new = _rows(5, span_hours=1, now=now)
    b = mc.compute_baseline(old + new, window_hours=168, now=now)
    assert b["n"] == 5
    assert b["rows_total_in_store"] == 10


def test_undated_rows_are_counted_not_silently_dropped():
    b = mc.compute_baseline([{"strategy": "gamma", "would_veto": False}])
    assert b["rows_undated_skipped"] == 1
    assert b["n"] == 0


def test_empty_store_yields_a_null_rate_not_zero():
    """A 0.0 veto rate and 'no data' are different facts; conflating them would
    wire a fabricated baseline."""
    b = mc.compute_baseline([])
    assert b["veto_rate"] is None
    assert b["sufficient"] is False


def test_baseline_carries_schema_and_ttl():
    b = mc.compute_baseline([])
    assert b["schema_version"] == "ml_veto_baseline.v1"
    assert b["ttl_seconds"] > 24 * 3600


def test_multiple_model_versions_are_flagged():
    now = datetime(2026, 7, 24, 1, 0, tzinfo=timezone.utc)
    rows = _rows(5, span_hours=10, now=now)
    rows += [dict(r, model_version="xgb_veto_NEWER") for r in _rows(5, span_hours=10, now=now)]
    b = mc.compute_baseline(rows, now=now)
    assert b["single_model_version"] is False
    assert len(b["model_versions"]) == 2


def test_write_baseline_is_atomic_and_valid_json(tmp_path):
    out = tmp_path / "baseline.json"
    mc.write_baseline(mc.compute_baseline([]), out)
    assert json.loads(out.read_text())["schema_version"] == "ml_veto_baseline.v1"
    assert not list(tmp_path.glob("*.tmp.*")), "temp file must be replaced, not left"


def test_journal_read_failure_degrades_to_empty(monkeypatch):
    """A collector that cannot read the journal must not crash its timer."""
    def _boom(*_a, **_kw):
        raise OSError("journalctl missing")

    monkeypatch.setattr(mc.subprocess, "run", _boom)
    assert mc.read_journal() == []
