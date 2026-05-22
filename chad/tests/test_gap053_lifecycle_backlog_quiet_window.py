"""NEW-GAP-053 — lifecycle_backlog_flag quiet-window policy.

Policy is documented at ops/pending_actions/GAP-053_lifecycle_replay_coverage_policy.md.

Pre-fix `build_trade_lifecycle_state` set `backlog_flag=true` whenever any
of broker_events / fills / fees ledger files had aged past
`CHAD_BROKER_EVENTS_MAX_AGE_SECONDS` (default 900s). That conflated
"no recent fills" (legitimate during cooldown / quiet markets) with
"lifecycle writer fell behind". The post-fix policy uses broker_events
heartbeats as the canonical pipeline-alive signal and only treats stale
fills/fees mtimes as backlog when a non-heartbeat broker event was
observed within the window (i.e. real fill activity that the writer
should have captured).

This file pins every branch of the policy. Safety-critical assertions:
  * gap (any source missing/empty)            => backlog_flag MUST be true
  * stale broker_events (pipeline outage)     => backlog_flag MUST be true
  * recent fill events but stale ledgers      => backlog_flag MUST be true
  * fresh everything                          => backlog_flag MUST be false
  * fresh broker_events + no recent fills +
    stale ledgers (quiet window)              => backlog_flag MUST be false
                                                 AND quiet_window_accepted MUST be true
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from chad.ops import lifecycle_truth_publisher as ltp
from chad.ops.lifecycle_truth_publisher import (
    BrokerEventsEvidence,
    LedgerEvidence,
    build_trade_lifecycle_state,
)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

MAX_AGE_S = 900  # matches the default; tests do not rely on env override.


def _fresh_unix() -> float:
    """Returns a POSIX timestamp that is unambiguously fresh."""
    return (datetime.now(timezone.utc) - timedelta(seconds=5)).timestamp()


def _stale_unix(seconds_old: int = MAX_AGE_S + 600) -> float:
    """Returns a POSIX timestamp that is unambiguously stale (older than the
    configured freshness window)."""
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds_old)).timestamp()


# ---------------------------------------------------------------------------
# Builders for evidence dataclasses + ndjson fixture files
# ---------------------------------------------------------------------------


def _write_broker_events_ndjson(
    tmp_path: Path,
    *,
    heartbeats: int = 5,
    fill_events_in_window: int = 0,
    fill_event_age_seconds: int = 30,
) -> Path:
    """Write a synthetic broker_events ndjson and return its path.

    `fill_events_in_window` controls how many `event_type: "fill"` events are
    written with `ts_utc` inside the freshness window. heartbeats are always
    recent so their type-counting bypass can be exercised."""
    p = tmp_path / "BROKER_EVENTS_IBKR_TEST.ndjson"
    now = datetime.now(timezone.utc)
    lines = []
    # Heartbeats — fresh timestamps so the helper sees them but filters them.
    for i in range(heartbeats):
        ts = (now - timedelta(seconds=10 + i)).isoformat().replace("+00:00", "Z")
        lines.append(json.dumps({
            "event_type": "heartbeat",
            "ts_utc": ts,
            "symbol": "",
            "qty": 0,
            "price": 0,
        }))
    # Fill events in the freshness window.
    for i in range(fill_events_in_window):
        ts = (now - timedelta(seconds=fill_event_age_seconds + i)).isoformat().replace("+00:00", "Z")
        lines.append(json.dumps({
            "event_type": "fill",
            "ts_utc": ts,
            "symbol": "AAPL",
            "qty": 1,
            "price": 100,
        }))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _broker_evidence(
    *,
    exists: bool = True,
    mtime_unix: float | None = None,
    newest_file: str | None = "/tmp/BROKER.ndjson",
    last_ts: str | None = None,
    event_count: int = 50,
) -> BrokerEventsEvidence:
    return BrokerEventsEvidence(
        exists=exists,
        newest_file=newest_file,
        newest_mtime_unix=mtime_unix,
        last_event_ts_utc=last_ts,
        event_count_hint=event_count,
    )


def _ledger_evidence(
    *,
    exists: bool = True,
    mtime_unix: float | None = None,
    newest_file: str | None = "/tmp/LEDGER.ndjson",
    line_count: int = 10,
) -> LedgerEvidence:
    return LedgerEvidence(
        exists=exists,
        newest_file=newest_file,
        newest_mtime_unix=mtime_unix,
        line_count_hint=line_count,
    )


def _build(
    *,
    tmp_path: Path,
    broker_evidence: BrokerEventsEvidence,
    fills_evidence: LedgerEvidence,
    fees_evidence: LedgerEvidence,
) -> dict:
    return build_trade_lifecycle_state(
        repo_root=tmp_path,
        runtime_dir=tmp_path / "runtime",
        data_dir=tmp_path / "data",
        evidence=broker_evidence,
        fills_evidence=fills_evidence,
        fees_evidence=fees_evidence,
    )


# ---------------------------------------------------------------------------
# Safety-critical: gap and broker-events-stale paths must remain blocking
# ---------------------------------------------------------------------------


def test_missing_broker_events_forces_backlog_true(tmp_path: Path) -> None:
    """If broker_events dir is missing entirely, gap_flag must trip and
    backlog_flag MUST stay true regardless of other state."""
    payload = _build(
        tmp_path=tmp_path,
        broker_evidence=_broker_evidence(exists=False, newest_file=None, mtime_unix=None),
        fills_evidence=_ledger_evidence(mtime_unix=_fresh_unix()),
        fees_evidence=_ledger_evidence(mtime_unix=_fresh_unix()),
    )
    assert payload["gap_flag"] is True
    assert payload["backlog_flag"] is True
    assert payload["quiet_window_accepted"] is False


def test_missing_fills_forces_backlog_true(tmp_path: Path) -> None:
    payload = _build(
        tmp_path=tmp_path,
        broker_evidence=_broker_evidence(mtime_unix=_fresh_unix()),
        fills_evidence=_ledger_evidence(exists=False, newest_file=None, mtime_unix=None, line_count=0),
        fees_evidence=_ledger_evidence(mtime_unix=_fresh_unix()),
    )
    assert payload["gap_flag"] is True
    assert payload["backlog_flag"] is True


def test_empty_fees_forces_backlog_true(tmp_path: Path) -> None:
    """File exists but is empty — must trip gap_flag (line_count_hint==0)."""
    payload = _build(
        tmp_path=tmp_path,
        broker_evidence=_broker_evidence(mtime_unix=_fresh_unix()),
        fills_evidence=_ledger_evidence(mtime_unix=_fresh_unix()),
        fees_evidence=_ledger_evidence(line_count=0, mtime_unix=_fresh_unix()),
    )
    assert payload["gap_flag"] is True
    assert payload["backlog_flag"] is True


def test_stale_broker_events_forces_backlog_true(tmp_path: Path) -> None:
    """No heartbeats in window => pipeline broken => backlog_flag MUST be true,
    even if fills/fees ledgers happen to be fresh."""
    payload = _build(
        tmp_path=tmp_path,
        broker_evidence=_broker_evidence(mtime_unix=_stale_unix()),
        fills_evidence=_ledger_evidence(mtime_unix=_fresh_unix()),
        fees_evidence=_ledger_evidence(mtime_unix=_fresh_unix()),
    )
    assert payload["gap_flag"] is False
    assert payload["backlog_flag"] is True
    assert payload["quiet_window_accepted"] is False


# ---------------------------------------------------------------------------
# Writer-outage detection: recent fill events but stale ledgers
# ---------------------------------------------------------------------------


def test_recent_fill_event_with_stale_fills_forces_backlog_true(tmp_path: Path) -> None:
    """Headline GAP-053 anti-regression: if a broker fill arrived in the
    freshness window but the fills ledger is stale, the writer is behind
    and backlog_flag MUST stay true. quiet_window_accepted must be false."""
    be_file = _write_broker_events_ndjson(tmp_path, heartbeats=5, fill_events_in_window=2)
    payload = _build(
        tmp_path=tmp_path,
        broker_evidence=_broker_evidence(
            mtime_unix=_fresh_unix(),
            newest_file=str(be_file),
        ),
        fills_evidence=_ledger_evidence(mtime_unix=_stale_unix()),
        fees_evidence=_ledger_evidence(mtime_unix=_fresh_unix()),
    )
    assert payload["backlog_flag"] is True
    assert payload["evidence"]["had_recent_broker_fill_activity"] is True
    assert payload["evidence"]["broker_events"]["recent_non_heartbeat_count"] >= 1
    assert payload["quiet_window_accepted"] is False


def test_recent_fill_event_with_stale_fees_forces_backlog_true(tmp_path: Path) -> None:
    be_file = _write_broker_events_ndjson(tmp_path, heartbeats=5, fill_events_in_window=1)
    payload = _build(
        tmp_path=tmp_path,
        broker_evidence=_broker_evidence(
            mtime_unix=_fresh_unix(),
            newest_file=str(be_file),
        ),
        fills_evidence=_ledger_evidence(mtime_unix=_fresh_unix()),
        fees_evidence=_ledger_evidence(mtime_unix=_stale_unix()),
    )
    assert payload["backlog_flag"] is True
    assert payload["evidence"]["had_recent_broker_fill_activity"] is True


# ---------------------------------------------------------------------------
# Quiet-window relaxation: the headline POLICY case
# ---------------------------------------------------------------------------


def test_quiet_window_accepts_stale_fills_when_no_recent_fill_activity(tmp_path: Path) -> None:
    """The exact production scenario behind Box 19: paper loop in cooldown,
    broker_events heartbeats fresh, no fill events in the window, fills
    and fees ledger mtimes >15 min old. Policy: backlog_flag=false,
    quiet_window_accepted=true."""
    be_file = _write_broker_events_ndjson(tmp_path, heartbeats=5, fill_events_in_window=0)
    payload = _build(
        tmp_path=tmp_path,
        broker_evidence=_broker_evidence(
            mtime_unix=_fresh_unix(),
            newest_file=str(be_file),
        ),
        fills_evidence=_ledger_evidence(mtime_unix=_stale_unix()),
        fees_evidence=_ledger_evidence(mtime_unix=_stale_unix()),
    )
    assert payload["gap_flag"] is False
    assert payload["backlog_flag"] is False
    assert payload["quiet_window_accepted"] is True
    assert payload["evidence"]["had_recent_broker_fill_activity"] is False
    assert payload["evidence"]["broker_events"]["recent_non_heartbeat_count"] == 0


def test_fully_fresh_yields_backlog_false_without_quiet_window(tmp_path: Path) -> None:
    be_file = _write_broker_events_ndjson(tmp_path, heartbeats=5, fill_events_in_window=2)
    payload = _build(
        tmp_path=tmp_path,
        broker_evidence=_broker_evidence(
            mtime_unix=_fresh_unix(),
            newest_file=str(be_file),
        ),
        fills_evidence=_ledger_evidence(mtime_unix=_fresh_unix()),
        fees_evidence=_ledger_evidence(mtime_unix=_fresh_unix()),
    )
    assert payload["gap_flag"] is False
    assert payload["backlog_flag"] is False
    # Quiet-window flag is only set when we're relaxing stale fills/fees.
    assert payload["quiet_window_accepted"] is False


# ---------------------------------------------------------------------------
# Helper-level pins: _count_recent_non_heartbeats
# ---------------------------------------------------------------------------


def test_helper_counts_only_recent_non_heartbeats(tmp_path: Path) -> None:
    be_file = _write_broker_events_ndjson(
        tmp_path, heartbeats=10, fill_events_in_window=3, fill_event_age_seconds=60
    )
    n = ltp._count_recent_non_heartbeats(str(be_file), max_age_s=MAX_AGE_S)
    assert n == 3


def test_helper_excludes_old_fills(tmp_path: Path) -> None:
    """A fill event with ts_utc OLDER than the window must not count."""
    be_file = _write_broker_events_ndjson(
        tmp_path, heartbeats=5, fill_events_in_window=2, fill_event_age_seconds=MAX_AGE_S + 600
    )
    n = ltp._count_recent_non_heartbeats(str(be_file), max_age_s=MAX_AGE_S)
    assert n == 0


def test_helper_safe_on_missing_file(tmp_path: Path) -> None:
    n = ltp._count_recent_non_heartbeats(
        str(tmp_path / "no_such_file.ndjson"), max_age_s=MAX_AGE_S
    )
    assert n == 0


def test_helper_safe_on_none_path() -> None:
    n = ltp._count_recent_non_heartbeats(None, max_age_s=MAX_AGE_S)
    assert n == 0


# ---------------------------------------------------------------------------
# Regression matrix — backlog_flag cannot silently flip to false
# ---------------------------------------------------------------------------


def test_regression_matrix_no_silent_false_under_failure(tmp_path: Path) -> None:
    """Iterate every degraded state and assert backlog_flag MUST be true.
    The post-fix code can only make backlog stricter under failure, never
    looser."""
    be_file_with_fills = _write_broker_events_ndjson(
        tmp_path, heartbeats=5, fill_events_in_window=2
    )
    quiet_dir = tmp_path / "quiet"
    quiet_dir.mkdir()
    be_file_quiet = _write_broker_events_ndjson(
        quiet_dir, heartbeats=5, fill_events_in_window=0
    )
    cases = [
        # (label, broker_evidence, fills_evidence, fees_evidence)
        (
            "missing broker_events",
            _broker_evidence(exists=False, newest_file=None, mtime_unix=None),
            _ledger_evidence(mtime_unix=_fresh_unix()),
            _ledger_evidence(mtime_unix=_fresh_unix()),
        ),
        (
            "stale broker_events",
            _broker_evidence(mtime_unix=_stale_unix(), newest_file=str(be_file_quiet)),
            _ledger_evidence(mtime_unix=_fresh_unix()),
            _ledger_evidence(mtime_unix=_fresh_unix()),
        ),
        (
            "missing fills",
            _broker_evidence(mtime_unix=_fresh_unix(), newest_file=str(be_file_quiet)),
            _ledger_evidence(exists=False, newest_file=None, mtime_unix=None, line_count=0),
            _ledger_evidence(mtime_unix=_fresh_unix()),
        ),
        (
            "missing fees",
            _broker_evidence(mtime_unix=_fresh_unix(), newest_file=str(be_file_quiet)),
            _ledger_evidence(mtime_unix=_fresh_unix()),
            _ledger_evidence(exists=False, newest_file=None, mtime_unix=None, line_count=0),
        ),
        (
            "recent fill events but stale fills ledger",
            _broker_evidence(mtime_unix=_fresh_unix(), newest_file=str(be_file_with_fills)),
            _ledger_evidence(mtime_unix=_stale_unix()),
            _ledger_evidence(mtime_unix=_fresh_unix()),
        ),
        (
            "recent fill events but stale fees ledger",
            _broker_evidence(mtime_unix=_fresh_unix(), newest_file=str(be_file_with_fills)),
            _ledger_evidence(mtime_unix=_fresh_unix()),
            _ledger_evidence(mtime_unix=_stale_unix()),
        ),
    ]
    for label, be, fills, fees in cases:
        payload = _build(
            tmp_path=tmp_path,
            broker_evidence=be,
            fills_evidence=fills,
            fees_evidence=fees,
        )
        assert payload["backlog_flag"] is True, f"backlog_flag flipped to false for failure case: {label}"
