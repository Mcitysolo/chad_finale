"""PA-EP4 — append-only 1m bar archive.

Covers chad/market_data/bar_archive.BarArchive and the poll_once hook guard:
1. dedup: overlapping appends → each ts_utc archived once
2. normalization (D1): local-labeled aware bar → true UTC; already-UTC unchanged
3. date partitioning: a batch spanning two UTC dates → two date files
4. disk-guard: free < 2 GB (and use >= 90%) → skip, log, return cleanly; recover
5. prune: only files older than retention are unlinked (filename-date compare)
6. hook isolation: an archive that raises does NOT propagate out of poll_once

No IBKR connection or network required — tmp dirs + a monkeypatched
free-space probe make every test deterministic.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from chad.market_data.bar_archive import BarArchive
from chad.market_data.ibkr_bar_provider import Bar

GB = 1024 ** 3


def _bar(ts: str) -> Bar:
    return Bar(ts_utc=ts, open=100.0, high=101.0, low=99.0, close=100.5,
               volume=10.0, symbol="AAPL")


def _healthy_archive(tmp_path, **kw) -> BarArchive:
    """Archive with a probe reporting ample free space (100 GB total / 50 free)."""
    return BarArchive(
        archive_dir=tmp_path / "1m_archive",
        free_space_probe=lambda p: (100 * GB, 50 * GB),
        now_fn=lambda: datetime(2026, 6, 14, tzinfo=timezone.utc),
        **kw,
    )


def _read(tmp_path, sym: str, date: str):
    p = tmp_path / "1m_archive" / sym / f"{date}.ndjson"
    return [ln for ln in p.read_text().splitlines() if ln.strip()]


# 1. dedup ------------------------------------------------------------------

def test_overlapping_appends_dedup_by_ts(tmp_path) -> None:
    arch = _healthy_archive(tmp_path)
    batch1 = [_bar("2026-06-12 19:00:00-04:00"), _bar("2026-06-12 19:01:00-04:00")]
    n1 = arch.append("AAPL", batch1)
    # next cycle: ~full overlap + one genuinely new bar
    batch2 = batch1 + [_bar("2026-06-12 19:02:00-04:00")]
    n2 = arch.append("AAPL", batch2)
    assert n1 == 2
    assert n2 == 1  # only the new bar is appended
    lines = _read(tmp_path, "AAPL", "2026-06-12")
    ts = [json.loads(ln)["ts_utc"] for ln in lines]
    assert len(ts) == 3 and len(set(ts)) == 3


# 2. normalization (D1) -----------------------------------------------------

def test_normalization_local_and_utc(tmp_path) -> None:
    arch = _healthy_archive(tmp_path)
    arch.append("AAPL", [_bar("2026-06-12 19:00:00-04:00")])   # EDT -> 23:00 UTC
    arch.append("AAPL", [_bar("2026-06-12 23:30:00+00:00")])   # already UTC
    ts = sorted(json.loads(ln)["ts_utc"] for ln in _read(tmp_path, "AAPL", "2026-06-12"))
    assert ts == ["2026-06-12 23:00:00+00:00", "2026-06-12 23:30:00+00:00"]


# 3. date partitioning ------------------------------------------------------

def test_batch_spanning_two_utc_dates_writes_two_files(tmp_path) -> None:
    arch = _healthy_archive(tmp_path)
    # 18:00 EST (-05:00) -> 23:00 UTC same day; 19:00 EST -> 00:00 UTC next day
    arch.append("AAPL", [_bar("2026-01-05 18:00:00-05:00"),
                         _bar("2026-01-05 19:00:00-05:00")])
    d1 = _read(tmp_path, "AAPL", "2026-01-05")
    d2 = _read(tmp_path, "AAPL", "2026-01-06")
    assert len(d1) == 1 and len(d2) == 1
    assert json.loads(d1[0])["ts_utc"] == "2026-01-05 23:00:00+00:00"
    assert json.loads(d2[0])["ts_utc"] == "2026-01-06 00:00:00+00:00"


# 4. disk-guard -------------------------------------------------------------

def test_disk_guard_low_free_skips_then_recovers(tmp_path) -> None:
    free = {"val": 1 * GB}  # below the 2 GB floor
    arch = BarArchive(
        archive_dir=tmp_path / "1m_archive",
        free_space_probe=lambda p: (100 * GB, free["val"]),
        now_fn=lambda: datetime(2026, 6, 14, tzinfo=timezone.utc),
    )
    n0 = arch.append("AAPL", [_bar("2026-06-12 19:00:00-04:00")])
    assert n0 == 0
    assert not (tmp_path / "1m_archive" / "AAPL" / "2026-06-12.ndjson").exists()
    # space recovers -> the same bar now archives (seen-set was rolled back)
    free["val"] = 50 * GB
    n1 = arch.append("AAPL", [_bar("2026-06-12 19:00:00-04:00")])
    assert n1 == 1
    assert (tmp_path / "1m_archive" / "AAPL" / "2026-06-12.ndjson").exists()


def test_disk_guard_high_use_pct_skips(tmp_path) -> None:
    # 5 GB free > 2 GB floor, but 95% used >= 90% ceiling -> still skip
    arch = BarArchive(
        archive_dir=tmp_path / "1m_archive",
        free_space_probe=lambda p: (100 * GB, 5 * GB),
        now_fn=lambda: datetime(2026, 6, 14, tzinfo=timezone.utc),
    )
    assert arch.append("AAPL", [_bar("2026-06-12 19:00:00-04:00")]) == 0
    assert not (tmp_path / "1m_archive" / "AAPL").exists()


# 5. prune ------------------------------------------------------------------

def test_prune_removes_only_older_than_retention(tmp_path) -> None:
    arch = _healthy_archive(tmp_path)
    sym_dir = tmp_path / "1m_archive" / "AAPL"
    sym_dir.mkdir(parents=True)
    old = sym_dir / "2025-01-01.ndjson"      # ~530 days before today -> prune
    recent = sym_dir / "2026-06-13.ndjson"   # 1 day before today -> keep
    old.write_text('{"ts_utc":"x"}\n')
    recent.write_text('{"ts_utc":"y"}\n')
    removed = arch._prune("2026-06-14")  # retention 365 -> cutoff 2025-06-14
    assert removed == 1
    assert not old.exists()
    assert recent.exists()


# 6. hook isolation ---------------------------------------------------------

def test_poll_once_hook_swallows_archive_errors() -> None:
    from chad.market_data.ibkr_bar_provider import IBKRBarProvider

    prov = IBKRBarProvider(ib=None, universe=["AAPL"])

    class _Boom:
        def append(self, *a, **k):
            raise RuntimeError("disk exploded")

    prov._archive = _Boom()
    # Must not raise — the live cache write path is unaffected.
    prov._archive_bars_safe("AAPL", [_bar("2026-06-12 19:00:00-04:00")])

    # A missing archive is also a clean no-op.
    prov._archive = None
    prov._archive_bars_safe("AAPL", [_bar("2026-06-12 19:00:00-04:00")])
