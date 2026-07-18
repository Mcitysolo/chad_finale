"""CRYPTO-EXPLORE-WIRE (UC1-4) — guardrail tests.

Implements the four W-item guarantees of
``ops/pending_actions/UC1_crypto_exploration_mode_2026-07-17.md``:

  W2  CHAD_CRYPTO_EXPLORATION consumption at the regime gate — flag OFF is byte-identical
      stock gating (the matrix is never mutated); flag ON + paper re-admits alpha_crypto in
      ALL regimes, tagged; flag ON + non-paper is FAIL-CLOSED (refuse + loud marker).
  W3  Per-fill live-regime tagging — the trusted-fill engine stamps the regime onto the open
      lot, the FILLS evidence, and (via the lot's ENTRY regime) the realized round-trip's
      trade_history row — replacing the old universal "paper".
  W1  Overlay wiring — the crypto overlay heartbeat emits on OFF / shadow-empty-book paths,
      writes its OWN evidence file (crypto_*.ndjson), and is wired into live_loop BEFORE the
      exploration re-admission can pass a signal (exit-before-exploration ordering).
  $185 The sizing floor that bounds exploration's worst case (min-size SKIP vs BUMP).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from chad.core import kraken_trusted_fill_engine as tfe
from chad.execution.kraken_executor import StrategyTradeIntent
from chad.portfolio import regime_activation as ra
from chad.risk import crypto_exit_overlay as cxo
from chad.risk.position_exit_overlay import load_overlay_config

UTC = timezone.utc
NOW = datetime(2026, 7, 18, 12, 0, 0, tzinfo=UTC)
REPO_ROOT = Path(__file__).resolve().parents[2]
_NOW_EPOCH = 1_800_000_000.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _kintent(strategy="alpha_crypto", pair="SOLUSD", side="buy", vol=0.05, markers=(),
             regime=""):
    return StrategyTradeIntent(
        strategy=strategy, pair=pair, side=side, ordertype="market",
        volume=vol, notional_estimate=vol * 170.0, markers=tuple(markers), regime=regime,
    )


class _RegimeIntent:
    """Duck-typed engine intent carrying an explicit ``regime`` (as the regime gate stamps)."""

    def __init__(self, side="buy", volume=1.0, regime="", markers=()):
        self.strategy = "alpha_crypto"
        self.pair = "SOLUSD"
        self.side = side
        self.volume = volume
        self.ordertype = "market"
        self.markers = markers
        self.idempotency_key = ""
        self.trace_id = ""
        self.regime = regime


# ===========================================================================
# W2 — exploration flag consumption at the regime gate
# ===========================================================================
def test_state_matrix_off_active_refused():
    # unset / falsey -> off
    assert ra.crypto_exploration_state({}) == (False, "off")
    assert ra.crypto_exploration_state({"CHAD_CRYPTO_EXPLORATION": "0"}) == (False, "off")
    assert ra.crypto_exploration_state(
        {"CHAD_CRYPTO_EXPLORATION": "false", "CHAD_EXECUTION_MODE": "paper"}
    ) == (False, "off")
    # on + paper (kraken lane unset/paper) -> active
    assert ra.crypto_exploration_state(
        {"CHAD_CRYPTO_EXPLORATION": "1", "CHAD_EXECUTION_MODE": "paper"}
    ) == (True, "active")
    assert ra.crypto_exploration_state(
        {"CHAD_CRYPTO_EXPLORATION": "1", "CHAD_EXECUTION_MODE": "paper",
         "CHAD_KRAKEN_MODE": "paper_kraken"}
    ) == (True, "active")
    # on + any non-paper (or unset) GLOBAL mode -> refused, fail-closed (axis 1)
    for mode in ("live", "dry_run", "", "LIVE"):
        assert ra.crypto_exploration_state(
            {"CHAD_CRYPTO_EXPLORATION": "1", "CHAD_EXECUTION_MODE": mode}
        ) == (False, "refused_non_paper")
    # SECOND AXIS: global paper but the crypto lane is explicitly LIVE -> refused, fail-closed.
    # CHAD_KRAKEN_MODE=live routes re-admitted alpha_crypto to REAL Kraken orders even under a
    # global paper posture, so a paper CHAD_EXECUTION_MODE is NOT sufficient to arm exploration.
    assert ra.crypto_exploration_state(
        {"CHAD_CRYPTO_EXPLORATION": "1", "CHAD_EXECUTION_MODE": "paper",
         "CHAD_KRAKEN_MODE": "live"}
    ) == (False, "refused_kraken_live")


def test_flag_off_is_byte_identical_stock_gating():
    ac = _kintent(strategy="alpha_crypto")
    other = _kintent(strategy="beta")
    kept = [_kintent(strategy="omega_macro")]
    dropped = [(ac, "regime_strategy_mismatch:regime=ranging;strategy=alpha_crypto"),
               (other, "regime_strategy_mismatch:regime=ranging;strategy=beta")]
    k2, d2, info = ra.apply_crypto_exploration(kept, dropped, "ranging", env={})
    assert info == {"state": "off", "readmitted": 0}
    assert k2 == kept                 # kept unchanged
    assert d2 == dropped              # alpha_crypto STAYS dropped — nothing re-admitted
    # The config matrix itself is never touched by this code path.
    matrix = json.loads((REPO_ROOT / "config" / "regime_activation_matrix.json").read_text())
    assert "alpha_crypto" not in matrix["regimes"]["ranging"], (
        "W2 must NOT mutate the matrix — ranging still excludes alpha_crypto on disk"
    )


def test_flag_on_paper_readmits_and_tags():
    env = {"CHAD_CRYPTO_EXPLORATION": "1", "CHAD_EXECUTION_MODE": "paper"}
    ac = _kintent(strategy="alpha_crypto", markers=("CRYPTO_MIN_SIZE_BUMP",))
    other = _kintent(strategy="beta")
    kept: list = []
    dropped = [(ac, "r1"), (other, "r2")]
    k2, d2, info = ra.apply_crypto_exploration(kept, dropped, "ranging", env=env)
    assert info == {"state": "active", "readmitted": 1}
    assert len(k2) == 1
    tagged = k2[0]
    assert tagged.strategy == "alpha_crypto"
    assert "CRYPTO_EXPLORATION_PASS regime=ranging" in tagged.markers
    assert "exploration=true" in tagged.markers
    assert "CRYPTO_MIN_SIZE_BUMP" in tagged.markers          # pre-existing markers preserved
    assert tagged.regime == "ranging"                        # W3 stamp
    # a non-alpha_crypto drop is untouched
    assert [i.strategy for i, _ in d2] == ["beta"]


def test_flag_on_non_paper_refuses_fail_closed(caplog):
    env = {"CHAD_CRYPTO_EXPLORATION": "1", "CHAD_EXECUTION_MODE": "live"}
    ac = _kintent(strategy="alpha_crypto")
    dropped = [(ac, "r1")]
    with caplog.at_level(logging.ERROR):
        k2, d2, info = ra.apply_crypto_exploration(
            [], dropped, "ranging", env=env, logger=logging.getLogger("cew.test")
        )
    assert info == {"state": "refused_non_paper", "readmitted": 0}
    assert d2 == dropped              # alpha_crypto STAYS dropped — no live exploration EVER
    assert k2 == []
    assert any("CRYPTO_EXPLORATION_REFUSED_NON_PAPER" in r.getMessage() for r in caplog.records)


def test_flag_on_paper_but_kraken_lane_live_refuses_fail_closed(caplog):
    # Global posture is paper, but the crypto lane is explicitly live — exploration must NOT
    # arm, because the re-admitted alpha_crypto intents would route to REAL Kraken orders.
    env = {"CHAD_CRYPTO_EXPLORATION": "1", "CHAD_EXECUTION_MODE": "paper",
           "CHAD_KRAKEN_MODE": "live"}
    ac = _kintent(strategy="alpha_crypto")
    dropped = [(ac, "r1")]
    with caplog.at_level(logging.ERROR):
        k2, d2, info = ra.apply_crypto_exploration(
            [], dropped, "ranging", env=env, logger=logging.getLogger("cew.test")
        )
    assert info == {"state": "refused_kraken_live", "readmitted": 0}
    assert d2 == dropped and k2 == []          # alpha_crypto STAYS dropped
    assert any("CRYPTO_EXPLORATION_REFUSED_NON_PAPER" in r.getMessage() for r in caplog.records)
    assert any("CHAD_KRAKEN_MODE" in r.getMessage() for r in caplog.records)


def test_stamp_intent_regime_is_immutable_replace():
    original = _kintent(strategy="alpha_crypto", regime="")
    stamped = ra.stamp_intent_regime(original, "volatile")
    assert stamped.regime == "volatile"
    assert original.regime == ""      # frozen dataclass: a NEW intent, original untouched


# ===========================================================================
# W3 — per-fill live-regime tagging through the trusted-fill engine
# ===========================================================================
def _engine(tmp_path, touch):
    ev_rows, th_rows = [], []
    eng = tfe.TrustedFillEngine(
        config=tfe.get_kraken_trading_config(),
        tick_source=type("_TS", (), {"get_touch": lambda self, s, *, now_epoch: touch})(),
        book=tfe.RoundTripBook(db_path=tmp_path / "book.sqlite3",
                               now_iso=lambda: "2026-07-18T00:00:00Z"),
        now_fn=lambda: _NOW_EPOCH,
        evidence_writer=lambda kw: (ev_rows.append(kw) or "FILLS.ndjson"),
        trade_history_writer=lambda kw: (th_rows.append(kw) or "th.ndjson"),
    )
    return eng, ev_rows, th_rows


def test_open_fill_stamps_regime_on_lot_and_evidence(tmp_path):
    touch = tfe.Touch(bid=170.0, ask=170.2, last=170.1)
    eng, ev_rows, _ = _engine(tmp_path, touch)
    r = eng.process_intent(_RegimeIntent(side="buy", volume=1.0, regime="ranging"))
    assert r["trusted"] and r["leg"] == "open"
    # FILLS evidence carries the live regime, not the old universal "paper".
    assert ev_rows[0]["regime"] == "ranging"
    # ...and the open lot stores it so the harness can slice by ENTRY regime.
    with sqlite3.connect(tmp_path / "book.sqlite3") as con:
        rows = con.execute(
            "SELECT regime FROM kraken_trusted_lots WHERE strategy='alpha_crypto'"
        ).fetchall()
    assert rows and rows[0][0] == "ranging"


def test_close_roundtrip_carries_entry_regime_into_trade_history(tmp_path):
    touch = tfe.Touch(bid=170.0, ask=170.2, last=170.1)
    eng, ev_rows, th_rows = _engine(tmp_path, touch)
    eng.process_intent(_RegimeIntent(side="buy", volume=1.0, regime="ranging"))   # open @ ranging
    # Close in a DIFFERENT live regime — the round-trip must still be attributed to the ENTRY
    # regime (ranging), read back from the lot, NOT the exit regime and NOT "paper".
    r = eng.process_intent(_RegimeIntent(side="sell", volume=1.0, regime="volatile"))
    assert r["trusted"] and r["leg"] == "close"
    assert len(th_rows) == 1
    assert th_rows[0]["regime"] == "ranging"
    assert th_rows[0]["regime"] != "paper"


def test_regime_tagged_fill_is_still_stage2_admissible():
    from chad.validation.trade_log_adapter import trust_exclusion
    # A trusted, regime-stamped exploration fill carries NO untrust markers, so Stage-2 admits.
    record = {
        "symbol": "SOL-USD", "side": "BUY", "quantity": 1.0, "fill_price": 170.2,
        "status": "paper_fill", "regime": "ranging", "pnl": 0.0,
        "tags": ["kraken_paper", "paper_fill", "trusted_fill", "SIMULATED_AGAINST_LIVE_TICKS",
                 "CRYPTO_EXPLORATION_PASS regime=ranging", "exploration=true"],
        "extra": {"fee_model": "kraken_paper_v1", "provenance": "SIMULATED_AGAINST_LIVE_TICKS",
                  "trust_state": "TRUSTED"},
    }
    assert trust_exclusion(record) is None


def test_missing_regime_falls_back_paper_not_crash(tmp_path):
    # An exit-overlay close intent carries no regime; the open lot had none either -> "paper".
    touch = tfe.Touch(bid=170.0, ask=170.2, last=170.1)
    eng, ev_rows, th_rows = _engine(tmp_path, touch)
    eng.process_intent(_RegimeIntent(side="buy", volume=1.0, regime=""))
    eng.process_intent(_RegimeIntent(side="sell", volume=1.0, regime=""))
    assert ev_rows[0]["regime"] == "paper"      # normalized fallback
    assert th_rows[0]["regime"] == "paper"


# ===========================================================================
# W1 — crypto overlay wiring + heartbeat liveness (off / shadow / empty-book)
# ===========================================================================
def _cfg(mode="shadow"):
    return load_overlay_config({
        "mode": mode, "atr_period": 14, "atr_trail_mult": 3.0,
        "hard_stop_loss_pct": 0.12, "min_bars_for_atr": 16,
        "max_hold_days": {"crypto": 4.0, "default": 4.0},
    })


def _overlay(tmp_path, *, mode, lots, env=None):
    return cxo.CryptoExitOverlay(
        _cfg(mode),
        evidence_path=tmp_path / "evi",
        state_path=tmp_path / "state.json",
        heartbeat_path=tmp_path / "hb.json",
        lots_loader=lambda: lots,
        marks_loader=lambda syms: {},
        bars_loader=lambda syms: {},
        env=env if env is not None else {},
    )


def test_heartbeat_emitted_on_off_path(tmp_path):
    ov = _overlay(tmp_path, mode="off", lots=[])
    ov.run_cycle(now_utc=NOW)
    hb = json.loads((tmp_path / "hb.json").read_text())
    assert hb["schema_version"] == "crypto_exit_overlay_heartbeat.v1"
    assert hb["mode"] == "off" and hb["evaluated"] == 0 and hb["healthy"] is True


def test_heartbeat_emitted_on_shadow_empty_book(tmp_path):
    ov = _overlay(tmp_path, mode="shadow", lots=[])
    ov.run_cycle(now_utc=NOW)
    hb = json.loads((tmp_path / "hb.json").read_text())
    assert hb["mode"] == "shadow" and hb["evaluated"] == 0 and hb["healthy"] is True
    assert hb["ttl_seconds"] == cxo.HEARTBEAT_TTL_SECONDS


def test_shadow_writes_own_crypto_evidence_file(tmp_path):
    # One open SOL lot, fresh mark -> at least one verdict -> evidence lands in crypto_*.ndjson,
    # NEVER the equity lane's exit_overlay_*.ndjson.
    lots = [("alpha_crypto", "SOL-USD", "long", 2.0, 76.6, 0.2, "2026-07-18T00:00:00Z")]
    ov = cxo.CryptoExitOverlay(
        _cfg("shadow"),
        evidence_path=tmp_path / "evi",
        state_path=tmp_path / "state.json",
        heartbeat_path=tmp_path / "hb.json",
        lots_loader=lambda: lots,
        marks_loader=lambda syms: {"SOL-USD": 76.0},
        bars_loader=lambda syms: {},
        env={},
    )
    res = ov.run_cycle(now_utc=NOW)
    assert res.evaluated
    crypto_files = list((tmp_path / "evi").glob("crypto_exit_overlay_*.ndjson"))
    equity_files = list((tmp_path / "evi").glob("exit_overlay_*.ndjson"))
    assert crypto_files, "crypto overlay must write its OWN crypto_*.ndjson evidence file"
    assert not equity_files, "crypto overlay must NOT append to the equity lane's file"
    row = json.loads(crypto_files[0].read_text().splitlines()[0])
    assert row["lane"] == "crypto"


def test_unreadable_book_is_unknown_never_flat(tmp_path):
    # House rule: a loader that RAISES (broker-truth-unreadable) must NOT be treated as flat.
    def _boom():
        raise RuntimeError("lot store unreadable")

    ov = cxo.CryptoExitOverlay(
        _cfg("shadow"),
        evidence_path=tmp_path / "evi", state_path=tmp_path / "state.json",
        heartbeat_path=tmp_path / "hb.json",
        lots_loader=_boom, marks_loader=lambda s: {}, bars_loader=lambda s: {}, env={},
    )
    res = ov.run_cycle(now_utc=NOW)
    assert res.evaluated is False                # error path -> proposes nothing
    hb = json.loads((tmp_path / "hb.json").read_text())
    assert hb["healthy"] is False               # a blind cycle reports UNHEALTHY, not green


# ===========================================================================
# W1 — structural: overlay wired BEFORE exploration can pass a signal
# ===========================================================================
def test_live_loop_wires_crypto_overlay_before_exploration():
    src = (REPO_ROOT / "chad" / "core" / "live_loop.py").read_text()
    assert "from chad.risk.crypto_exit_overlay import build_default_crypto_overlay" in src
    assert "_crypto_exit_overlay.run_cycle()" in src
    i_equity = src.index("build_default_overlay")
    i_crypto = src.index("build_default_crypto_overlay")
    i_metrics = src.index("MarketMetricsPublisher")
    i_explore = src.index("apply_crypto_exploration")
    # equity overlay < crypto overlay < market metrics (i.e. before intent planning), and the
    # crypto EXIT path is wired strictly BEFORE the exploration re-admission (UC1 §0 ordering:
    # exploration that cannot exit is accumulation).
    assert i_equity < i_crypto < i_metrics, "crypto overlay must sit after equity, before metrics"
    assert i_crypto < i_explore, "exit path must be wired BEFORE exploration can pass a signal"


def test_crypto_health_rule_registered_and_blind_check_uses_lot_book():
    from chad.ops import health_monitor_rules as hmr
    import inspect
    src = inspect.getsource(hmr.run_all_rules)
    assert "rule_crypto_exit_overlay_heartbeat" in src
    # the alive-but-blind cross-check reads the Kraken lot book, not the equity snapshot
    rule_src = inspect.getsource(hmr.rule_crypto_exit_overlay_heartbeat)
    assert "crypto_exit_overlay_heartbeat.json" in rule_src
    assert "_kraken_open_lot_qty" in rule_src


# ===========================================================================
# $185 wallet — the sizing floor that bounds exploration's worst case
# ===========================================================================
def test_185_wallet_sub_minimum_is_skipped_not_rounded_up():
    # SOL min is 0.05 (~$8.5 notional). A $1 (0.1x SCR of a $10 target) order is below min AND
    # the min notional exceeds a $1 per-strategy cap -> SKIP, never bumped into a position the
    # wallet cannot carry.
    from chad.execution.kraken_min_size import decide_min_size
    from chad.execution.kraken_trading_config import load_kraken_trading_config
    cfg = load_kraken_trading_config()
    price, pair = 170.0, "SOLUSD"
    mn = cfg.min_volume(pair)
    cv = (10.0 * 0.1) / price                    # $10 target * SCR 0.1 -> $1 -> 0.0059 SOL
    assert cv < mn
    d = decide_min_size(pair=pair, computed_volume=cv, price=price, min_volume=mn,
                        available_notional=185.0, risk_cap_notional=1.0)
    assert d.is_skip and d.marker == "CRYPTO_BELOW_MIN_SKIP"
    assert d.final_volume == 0.0


def test_185_wallet_affordable_min_bumps():
    from chad.execution.kraken_min_size import decide_min_size
    from chad.execution.kraken_trading_config import load_kraken_trading_config
    cfg = load_kraken_trading_config()
    price, pair = 170.0, "SOLUSD"
    mn = cfg.min_volume(pair)
    cv = (50.0 * 0.1) / price                    # $5 -> below min but min ($8.5) affordable
    assert cv < mn
    d = decide_min_size(pair=pair, computed_volume=cv, price=price, min_volume=mn,
                        available_notional=185.0, risk_cap_notional=50.0)
    assert d.is_bump and d.final_volume == mn and d.marker == "CRYPTO_MIN_SIZE_BUMP"
