"""W4A-9 — operator manual-clear override (scripts/clear_fuse.py mechanism).

A tripped fuse an operator clears reads untripped for the session, honestly
(manually_cleared=True, real streak still shown, previous_* preserved), and the
override auto-expires at the session roll (session-window-keyed).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from chad.risk.fuse_box import (
    BucketSpec,
    TrustedClose,
    evaluate_buckets,
    load_manual_clears,
)

NOW = datetime(2026, 7, 23, 14, 0, 0, tzinfo=timezone.utc)
WINDOW = datetime(2026, 7, 23, tzinfo=timezone.utc)
WINDOW_ISO = "2026-07-23T00:00:00Z"


def _spec():
    return BucketSpec(fuse_id="family:gamma", kind="family",
                      members=frozenset({"gamma"}), consecutive_losers=2)


def _closes(n=3):
    return [
        TrustedClose(strategy="gamma", symbol="PSQ", side="BUY", pnl=-1.0,
                     exit_ts=NOW, regime="unknown", setup_family=None,
                     fill_ids=("a", "b"))
        for _ in range(n)
    ]


def test_manual_clear_forces_untripped():
    rows, events = evaluate_buckets(
        [_spec()], _closes(), now=NOW,
        manual_clears={"family:gamma": {"by": "op"}},
    )
    row = rows[0]
    assert row["tripped"] is False
    assert row["manually_cleared"] is True
    # honest: the real streak is still shown
    assert row["consecutive_losers"] == 3


def test_manual_clear_emits_clear_event_on_transition():
    prior = {"fuses": [{"fuse_id": "family:gamma", "kind": "family",
                        "tripped": True, "tripped_at_utc": "2026-07-23T13:00:00Z"}]}
    rows, events = evaluate_buckets(
        [_spec()], _closes(), prior_state=prior, now=NOW,
        manual_clears={"family:gamma": {"by": "op"}},
    )
    assert [e.event for e in events] == ["clear"]


def test_no_manual_clear_still_trips():
    rows, _ = evaluate_buckets([_spec()], _closes(), now=NOW)
    assert rows[0]["tripped"] is True
    assert "manually_cleared" not in rows[0]


def test_manual_clear_of_untripped_bucket_is_harmless():
    rows, events = evaluate_buckets(
        [_spec()], _closes(1), now=NOW,  # 1 loser < 2 → not tripped
        manual_clears={"family:gamma": {"by": "op"}},
    )
    # not tripped anyway; manual clear path only fires when stats.tripped
    assert rows[0]["tripped"] is False
    assert events == []


# --------------------------------------------------------------------------- #
# load_manual_clears — session-window scoping (auto-expiry)
# --------------------------------------------------------------------------- #

def test_load_clears_matching_window(tmp_path):
    p = tmp_path / "fuse_manual_clears.json"
    p.write_text(json.dumps({
        "cleared": {"family:gamma": {"session_window_start": WINDOW_ISO,
                                     "by": "op"}},
    }))
    out = load_manual_clears(WINDOW, p)
    assert "family:gamma" in out


def test_load_clears_ignores_prior_window(tmp_path):
    """A clear stamped against yesterday's session auto-expires today."""
    p = tmp_path / "fuse_manual_clears.json"
    p.write_text(json.dumps({
        "cleared": {"family:gamma": {"session_window_start": "2026-07-22T00:00:00Z"}},
    }))
    out = load_manual_clears(WINDOW, p)
    assert out == {}


def test_load_clears_missing_file(tmp_path):
    assert load_manual_clears(WINDOW, tmp_path / "nope.json") == {}


def test_load_clears_corrupt_file(tmp_path):
    p = tmp_path / "fuse_manual_clears.json"
    p.write_text("{broken")
    assert load_manual_clears(WINDOW, p) == {}


# --------------------------------------------------------------------------- #
# CLI fail-closed gates (structural)
# --------------------------------------------------------------------------- #

def test_cli_has_fail_closed_gates():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "scripts" / "clear_fuse.py").read_text()
    assert "_check_exec_mode_paper" in src
    assert "_check_scr_safe" in src
    assert "_SAFE_SCR_STATES" in src
    assert "return 2" in src  # refuses with nonzero exit
