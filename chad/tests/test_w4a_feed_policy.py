"""W4A-8 — DQ per-feed policies + the two P18 wired sites.

Fault injection per policy verb: back-dated ts, corrupt bytes, missing file,
future-dated, env-mode override — plus the shadow-is-byte-identical proofs.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from chad.utils.feed_policy import (
    BLOCK_ENTRIES,
    CORRUPT,
    FRESH,
    MISSING,
    STALE,
    FeedPolicies,
    FeedVerdict,
    read_with_policy,
)

REAL_POLICIES = FeedPolicies.load()


def _write_state(path, *, age_seconds=10, ttl=180, schema="scr_state.v1",
                 corrupt=False, extra=None):
    ts = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    if corrupt:
        path.write_text("{broken json")
        return
    obj = {"schema_version": schema, "ts_utc": ts.isoformat().replace("+00:00", "Z"),
           "ttl_seconds": ttl, "state": "CONFIDENT"}
    if extra:
        obj.update(extra)
    path.write_text(json.dumps(obj))


def _read(tmp_path, rel="runtime/scr_state.json", env=None, **write_kw):
    (tmp_path / "runtime").mkdir(parents=True, exist_ok=True)
    abs_path = tmp_path / rel
    _write_state(abs_path, **write_kw)
    return read_with_policy(
        rel, policies=REAL_POLICIES, env=env or {}, repo_root=tmp_path,
        evidence_dir=tmp_path / "ev",
    )


# --------------------------------------------------------------------------- #
# Freshness classification
# --------------------------------------------------------------------------- #

def test_fresh_feed_no_block(tmp_path):
    obj, v = _read(tmp_path, env={"CHAD_DQ_POLICIES": "enforce"}, age_seconds=10)
    assert v.freshness == FRESH
    assert v.should_block is False
    assert obj is not None


def test_stale_feed_classified(tmp_path):
    _, v = _read(tmp_path, env={"CHAD_DQ_POLICIES": "enforce"},
                 age_seconds=400, ttl=180)
    assert v.freshness == STALE
    assert v.policy == BLOCK_ENTRIES


def test_corrupt_feed_classified(tmp_path):
    _, v = _read(tmp_path, env={"CHAD_DQ_POLICIES": "enforce"}, corrupt=True)
    assert v.freshness == CORRUPT
    assert v.policy == BLOCK_ENTRIES


def test_missing_feed_classified(tmp_path):
    (tmp_path / "runtime").mkdir(parents=True, exist_ok=True)
    _, v = read_with_policy(
        "runtime/scr_state.json", policies=REAL_POLICIES,
        env={"CHAD_DQ_POLICIES": "enforce"}, repo_root=tmp_path,
        evidence_dir=tmp_path / "ev",
    )
    assert v.freshness == MISSING
    assert v.should_block is True  # execution_critical block_entries


def test_future_dated_is_fresh(tmp_path):
    """A future ts (clock skew) is within ttl → fresh, not stale."""
    _, v = _read(tmp_path, env={"CHAD_DQ_POLICIES": "enforce"}, age_seconds=-30)
    assert v.freshness == FRESH


# --------------------------------------------------------------------------- #
# Mode: enforce blocks, shadow/off never block
# --------------------------------------------------------------------------- #

def test_enforce_blocks_on_stale(tmp_path):
    _, v = _read(tmp_path, env={"CHAD_DQ_POLICIES": "enforce"},
                 age_seconds=400, ttl=180)
    assert v.should_block is True


def test_shadow_never_blocks(tmp_path):
    _, v = _read(tmp_path, env={"CHAD_DQ_POLICIES": "shadow"},
                 age_seconds=400, ttl=180)
    assert v.freshness == STALE
    assert v.should_block is False  # byte-identical behavior


def test_off_never_blocks(tmp_path):
    _, v = _read(tmp_path, env={}, age_seconds=400, ttl=180)
    assert v.should_block is False


def test_artifact_ttl_wins_over_config(tmp_path):
    """A generous artifact ttl keeps a feed fresh even past the config ttl
    (house rule: artifact self-declared ttl wins)."""
    _, v = _read(tmp_path, env={"CHAD_DQ_POLICIES": "enforce"},
                 age_seconds=250, ttl=600)  # config ttl 180, artifact 600
    assert v.freshness == FRESH


# --------------------------------------------------------------------------- #
# Unknown feed / broken config = fail-safe (never block)
# --------------------------------------------------------------------------- #

def test_unknown_feed_ignores(tmp_path):
    (tmp_path / "runtime").mkdir(parents=True, exist_ok=True)
    p = tmp_path / "runtime" / "mystery.json"
    _write_state(p, age_seconds=9999, ttl=1)
    _, v = read_with_policy("runtime/mystery.json", policies=REAL_POLICIES,
                            env={"CHAD_DQ_POLICIES": "enforce"}, repo_root=tmp_path,
                            evidence_dir=tmp_path / "ev")
    assert v.policy == "ignore"
    assert v.should_block is False


def test_broken_config_disarms(tmp_path):
    """A corrupt feed_policies.json → empty policies → every feed ignores."""
    cfg = tmp_path / "feed_policies.json"
    cfg.write_text("{broken")
    pols = FeedPolicies.load(cfg)
    (tmp_path / "runtime").mkdir(parents=True, exist_ok=True)
    p = tmp_path / "runtime" / "scr_state.json"
    _write_state(p, age_seconds=9999, ttl=1)
    _, v = read_with_policy("runtime/scr_state.json", policies=pols,
                            env={"CHAD_DQ_POLICIES": "enforce"}, repo_root=tmp_path,
                            evidence_dir=tmp_path / "ev")
    assert v.should_block is False


def test_unknown_verb_falls_back_to_ignore(tmp_path):
    pols = FeedPolicies({"runtime/x.json": {"class": "c", "on_stale": "nuke_it"}})
    (tmp_path / "runtime").mkdir(parents=True, exist_ok=True)
    p = tmp_path / "runtime" / "x.json"
    _write_state(p, age_seconds=9999, ttl=1)
    _, v = read_with_policy("runtime/x.json", policies=pols,
                            env={"CHAD_DQ_POLICIES": "enforce"}, repo_root=tmp_path,
                            evidence_dir=tmp_path / "ev")
    assert v.policy == "ignore"
    assert v.should_block is False


# --------------------------------------------------------------------------- #
# degrade_loud (ibkr_status / pnl_state) never blocks — only loud
# --------------------------------------------------------------------------- #

def test_degrade_loud_never_blocks(tmp_path):
    (tmp_path / "runtime").mkdir(parents=True, exist_ok=True)
    p = tmp_path / "runtime" / "ibkr_status.json"
    _write_state(p, age_seconds=9999, ttl=120, schema="ibkr_status")
    _, v = read_with_policy("runtime/ibkr_status.json", policies=REAL_POLICIES,
                            env={"CHAD_DQ_POLICIES": "enforce"}, repo_root=tmp_path,
                            evidence_dir=tmp_path / "ev")
    assert v.freshness == STALE
    assert v.policy == "degrade_loud"
    assert v.should_block is False  # loud only, no behavior change


# --------------------------------------------------------------------------- #
# Real config sanity
# --------------------------------------------------------------------------- #

def test_real_config_scr_is_block_entries():
    cfg = REAL_POLICIES.for_feed("runtime/scr_state.json")
    assert cfg is not None
    assert cfg["on_stale"] == "block_entries"
    assert cfg["class"] == "execution_critical"


def test_real_config_ibkr_status_is_degrade():
    cfg = REAL_POLICIES.for_feed("runtime/ibkr_status.json")
    assert cfg is not None
    assert cfg["on_stale"] == "degrade_loud"


# --------------------------------------------------------------------------- #
# Wired-site structural proofs (P18)
# --------------------------------------------------------------------------- #

def _live_loop_src() -> str:
    # Read the file directly — importing chad.core.live_loop has heavy
    # import-time side effects (IB connect attempts); we only need the text.
    from pathlib import Path

    return (
        Path(__file__).resolve().parent.parent / "core" / "live_loop.py"
    ).read_text()


def test_scr_gate_wired_fail_closed():
    """Site 1: the live_loop SCR gate consults the DQ policy and blocks on a
    dead SCR (DQ_SCR_BLOCK). Structural proof the wiring exists."""
    src = _live_loop_src()
    assert "_dq_scr_block" in src
    assert "DQ_SCR_BLOCK" in src
    assert "read_with_policy" in src


def test_stop_bus_inputs_wired_loud():
    """Site 2: the stop-bus snapshot builder emits DQ loudness for
    ibkr_status / pnl_state when DQ != off."""
    src = _live_loop_src()
    assert "DQ (P18 site 2)" in src
    assert "ibkr_status.json" in src
    assert "pnl_state.json" in src


def test_scr_gate_off_is_byte_identical(tmp_path):
    """With DQ off, a stale SCR yields should_block=False → the live gate's
    _dq_scr_block stays False → no behavior change vs pre-W4A-8."""
    _, v = _read(tmp_path, env={}, age_seconds=99999, ttl=1)
    assert v.should_block is False
