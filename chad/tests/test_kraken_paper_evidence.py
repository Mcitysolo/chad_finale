"""
chad/tests/test_kraken_paper_evidence.py

GAP: paper_kraken validate-only fills must produce trade_history + FILLS
evidence so SCR / strategy_routing_diagnostics see alpha_crypto activity.

Covers chad.core.kraken_execution.execute_kraken_intents and the
_write_paper_kraken_evidence helper.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_evidence_paths(tmp_path, monkeypatch):
    """Redirect every disk path the patch touches into tmp_path.

    Returns a dict so tests can assert on the resulting files.
    """
    from chad.core import kraken_execution as ke
    from chad.execution import paper_exec_evidence_writer as pew
    from chad.analytics import trade_result_logger as trl
    from chad.execution import idempotency_store as ids

    # CRYPTO-TRUST: these tests validate the LEGACY untrusted _write_paper_
    # kraken_evidence path, which is the kill-switch-off / fallback behavior.
    # Pin the kill-switch OFF so they are deterministic regardless of whether
    # the live WS feed has a fresh runtime/kraken_prices.json touch (which would
    # otherwise activate the trusted engine and write realized rows on close only).
    monkeypatch.setenv("CHAD_KRAKEN_TRUSTED_FILLS", "0")

    fills_dir = tmp_path / "fills"
    fees_dir = tmp_path / "fees"
    metrics_dir = tmp_path / "execution_metrics"
    locks_dir = tmp_path / "locks"
    trades_dir = tmp_path / "trades"
    kraken_fills_dir = tmp_path / "kraken_fills"
    for d in (fills_dir, fees_dir, metrics_dir, locks_dir, trades_dir, kraken_fills_dir):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(pew, "FILLS_DIR", fills_dir)
    monkeypatch.setattr(pew, "FEES_DIR", fees_dir)
    monkeypatch.setattr(pew, "EXEC_METRICS_DIR", metrics_dir)
    monkeypatch.setattr(pew, "LOCKS_DIR", locks_dir)
    monkeypatch.setattr(trl, "TRADE_DIR", trades_dir)

    # Redirect log_kraken_fill to write under tmp_path so tests are hermetic.
    def _patched_log_kraken_fill(payload):
        import time as _time
        ymd = _time.strftime("%Y%m%d", _time.gmtime())
        path = kraken_fills_dir / f"kraken_fills_{ymd}.ndjson"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")

    monkeypatch.setattr(ke, "log_kraken_fill", _patched_log_kraken_fill)

    # Redirect the IdempotencyStore to a tmp_path SQLite so tests can run
    # repeatedly without the dedup table accumulating state.
    db_path = tmp_path / "exec_state_paper.sqlite3"
    store = ids.IdempotencyStore(db_path, table="kraken_paper_evidence")
    # Force fresh singleton.
    monkeypatch.setattr(ke, "_PAPER_EVIDENCE_STORE", store, raising=False)
    # And also stub the lazy getter so any reset path returns this store.
    monkeypatch.setattr(ke, "_get_paper_evidence_store", lambda: store)

    return {
        "fills_dir": fills_dir,
        "trades_dir": trades_dir,
        "kraken_fills_dir": kraken_fills_dir,
        "store": store,
        "db_path": db_path,
    }


def _today_ymd() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d")


def _make_paper_intent():
    """Build a StrategyTradeIntent shaped like what alpha_crypto emits."""
    from chad.execution.kraken_executor import StrategyTradeIntent
    return StrategyTradeIntent(
        strategy="alpha_crypto",
        pair="SOLUSD",
        side="buy",
        ordertype="limit",
        volume=2.5,
        notional_estimate=425.0,
        price=170.0,
    )


def _fake_executor(*, allowed: bool, txids=None):
    """Return a MagicMock executor whose execute_with_risk shape matches the
    real KrakenExecutor (RiskGateResult, TradeResponse|None)."""
    from chad.execution.kraken_executor import RiskGateResult
    from chad.execution.kraken_trade_router import TradeResponse

    rr = RiskGateResult(allowed=allowed, reason="ok" if allowed else "blocked",
                        adjusted_notional=425.0 if allowed else 0.0)
    resp = None
    if allowed:
        resp = TradeResponse(txids=list(txids or []), raw={"validate": True, "descr": {}})
    fake = MagicMock()
    fake.execute_with_risk.return_value = (rr, resp)
    return fake


def _read_ndjson(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# ---------------------------------------------------------------------------
# 1) trade_history_*.ndjson written for paper validate-only success
# ---------------------------------------------------------------------------


def test_paper_kraken_execute_writes_trade_history(
    isolated_evidence_paths, monkeypatch
):
    from chad.core import kraken_execution as ke

    monkeypatch.setattr(ke, "is_kraken_gate_enabled", lambda: True)
    monkeypatch.setattr(ke, "resolve_kraken_mode", lambda: "paper_kraken")
    monkeypatch.setattr(ke, "get_kraken_executor",
                        lambda: _fake_executor(allowed=True, txids=[]))

    intent = _make_paper_intent()
    logger = logging.getLogger("test.paper_kraken.history")
    ke.execute_kraken_intents(logger, [intent])

    trades_path = isolated_evidence_paths["trades_dir"] / f"trade_history_{_today_ymd()}.ndjson"
    rows = _read_ndjson(trades_path)
    assert rows, f"no trade_history written at {trades_path}"
    payload = rows[-1]["payload"]
    assert payload["strategy"] == "alpha_crypto"
    assert payload["broker"] == "kraken_paper"
    assert payload["is_live"] is False
    assert payload["symbol"] == "SOL-USD"
    assert payload["pnl"] == 0.0
    tags = set(payload.get("tags") or [])
    assert "kraken_paper" in tags
    assert "validate_only" in tags
    assert "pnl_untrusted" in tags
    assert payload["extra"]["pnl_untrusted"] is True
    assert payload["extra"]["validate_only"] is True
    assert str(payload["extra"]["txid"]).startswith("PAPER-KRAKEN-")


# ---------------------------------------------------------------------------
# 2) FILLS_*.ndjson written with kraken_paper broker + paper_fill status
# ---------------------------------------------------------------------------


def test_paper_kraken_execute_writes_paper_fill(
    isolated_evidence_paths, monkeypatch
):
    from chad.core import kraken_execution as ke

    monkeypatch.setattr(ke, "is_kraken_gate_enabled", lambda: True)
    monkeypatch.setattr(ke, "resolve_kraken_mode", lambda: "paper_kraken")
    monkeypatch.setattr(ke, "get_kraken_executor",
                        lambda: _fake_executor(allowed=True, txids=[]))

    intent = _make_paper_intent()
    logger = logging.getLogger("test.paper_kraken.fills")
    ke.execute_kraken_intents(logger, [intent])

    fills_path = isolated_evidence_paths["fills_dir"] / f"FILLS_{_today_ymd()}.ndjson"
    rows = _read_ndjson(fills_path)
    assert rows, f"no FILLS evidence written at {fills_path}"
    payload = rows[-1]["payload"]
    assert payload["strategy"] == "alpha_crypto"
    assert payload["asset_class"] == "crypto"
    assert payload["broker"] == "kraken_paper"
    assert payload["status"] == "paper_fill"
    assert payload["side"] == "BUY"
    assert payload["symbol"] == "SOL-USD"
    assert float(payload["fill_price"]) == pytest.approx(170.0, rel=1e-9)
    assert float(payload["quantity"]) == pytest.approx(2.5, rel=1e-9)
    assert payload["is_live"] is False
    tags = set(payload.get("tags") or [])
    assert "kraken_paper" in tags
    assert "validate_only" in tags
    assert "paper_fill" in tags
    extra = payload.get("extra") or {}
    assert extra.get("pnl_untrusted") is True
    assert str(extra.get("synthetic_txid", "")).startswith("PAPER-KRAKEN-")


# ---------------------------------------------------------------------------
# 3) Idempotency within the same UTC minute bucket
# ---------------------------------------------------------------------------


def test_paper_kraken_execute_idempotent_per_minute_bucket(
    isolated_evidence_paths, monkeypatch
):
    from chad.core import kraken_execution as ke

    monkeypatch.setattr(ke, "is_kraken_gate_enabled", lambda: True)
    monkeypatch.setattr(ke, "resolve_kraken_mode", lambda: "paper_kraken")
    monkeypatch.setattr(ke, "get_kraken_executor",
                        lambda: _fake_executor(allowed=True, txids=[]))

    # Pin the UTC minute bucket so re-entry under the same intent still
    # collides on the same idempotency key. Without this, a test that
    # straddles a minute boundary would get two records by design.
    monkeypatch.setattr(ke, "_kraken_paper_minute_bucket", lambda ts=None: "202605081200")

    intent = _make_paper_intent()
    logger = logging.getLogger("test.paper_kraken.dedup")
    ke.execute_kraken_intents(logger, [intent])
    ke.execute_kraken_intents(logger, [intent])
    ke.execute_kraken_intents(logger, [intent])

    trades_path = isolated_evidence_paths["trades_dir"] / f"trade_history_{_today_ymd()}.ndjson"
    fills_path = isolated_evidence_paths["fills_dir"] / f"FILLS_{_today_ymd()}.ndjson"
    trade_rows = _read_ndjson(trades_path)
    fill_rows = _read_ndjson(fills_path)
    assert len(trade_rows) == 1, f"expected 1 trade_history row, got {len(trade_rows)}"
    assert len(fill_rows) == 1, f"expected 1 FILLS row, got {len(fill_rows)}"

    # Advancing the bucket admits a new record.
    monkeypatch.setattr(ke, "_kraken_paper_minute_bucket", lambda ts=None: "202605081201")
    ke.execute_kraken_intents(logger, [intent])
    trade_rows2 = _read_ndjson(trades_path)
    fill_rows2 = _read_ndjson(fills_path)
    assert len(trade_rows2) == 2
    assert len(fill_rows2) == 2


# ---------------------------------------------------------------------------
# 4) Risk-denied intents must NOT produce paper evidence
# ---------------------------------------------------------------------------


def test_paper_kraken_execute_skips_when_risk_denied(
    isolated_evidence_paths, monkeypatch
):
    from chad.core import kraken_execution as ke

    monkeypatch.setattr(ke, "is_kraken_gate_enabled", lambda: True)
    monkeypatch.setattr(ke, "resolve_kraken_mode", lambda: "paper_kraken")
    monkeypatch.setattr(ke, "get_kraken_executor",
                        lambda: _fake_executor(allowed=False, txids=[]))

    intent = _make_paper_intent()
    logger = logging.getLogger("test.paper_kraken.denied")
    ke.execute_kraken_intents(logger, [intent])

    trades_path = isolated_evidence_paths["trades_dir"] / f"trade_history_{_today_ymd()}.ndjson"
    fills_path = isolated_evidence_paths["fills_dir"] / f"FILLS_{_today_ymd()}.ndjson"
    assert _read_ndjson(trades_path) == []
    assert _read_ndjson(fills_path) == []


# ---------------------------------------------------------------------------
# 5) Live mode with real txids must NOT trigger the paper evidence path
# ---------------------------------------------------------------------------


def test_live_kraken_execute_unchanged(
    isolated_evidence_paths, monkeypatch
):
    """When mode=live and the broker returned real txids, the dispatcher
    must not write any kraken_paper FILLS / trade_history records (the
    executor's existing live path is the SSOT for those)."""
    from chad.core import kraken_execution as ke

    monkeypatch.setattr(ke, "is_kraken_gate_enabled", lambda: True)
    monkeypatch.setattr(ke, "resolve_kraken_mode", lambda: "live")
    # Live executor returns a real-looking txid. The executor's own live
    # path would normally log_kraken_trade_event(); here we patch
    # get_kraken_executor to a fake so the executor's internal logging is
    # bypassed and we can assert the dispatcher writes no paper evidence.
    monkeypatch.setattr(
        ke,
        "get_kraken_executor",
        lambda: _fake_executor(allowed=True, txids=["O123-LIVE-TXID"]),
    )

    intent = _make_paper_intent()
    logger = logging.getLogger("test.paper_kraken.live")
    ke.execute_kraken_intents(logger, [intent])

    fills_path = isolated_evidence_paths["fills_dir"] / f"FILLS_{_today_ymd()}.ndjson"
    rows = _read_ndjson(fills_path)
    # No kraken_paper records should appear via the dispatcher.
    paper_rows = [r for r in rows if (r.get("payload") or {}).get("broker") == "kraken_paper"]
    assert paper_rows == [], (
        f"live mode unexpectedly wrote kraken_paper FILLS records: {paper_rows}"
    )

    trades_path = isolated_evidence_paths["trades_dir"] / f"trade_history_{_today_ymd()}.ndjson"
    trade_rows = _read_ndjson(trades_path)
    paper_trades = [
        r for r in trade_rows
        if (r.get("payload") or {}).get("broker") == "kraken_paper"
    ]
    assert paper_trades == []


# ---------------------------------------------------------------------------
# 6) The paper fill is visible to the strategy_routing_diagnostics consumer
# ---------------------------------------------------------------------------


def test_kraken_paper_fill_visible_to_strategy_lane_summary(
    isolated_evidence_paths, monkeypatch, tmp_path
):
    """Once paper evidence is written, build_diagnostics must report
    alpha_crypto with zero_fill_epoch2=False (a fill exists in the active
    epoch) and a non-null last_fill_at."""
    from chad.core import kraken_execution as ke
    from chad.ops import strategy_routing_diagnostics as srd

    # Point the diagnostics scanner at our isolated FILLS dir.
    monkeypatch.setattr(srd, "DATA_DIR", isolated_evidence_paths["fills_dir"].parent)
    # Force epoch_start to a fixed point in the past so any fill we write
    # counts as "in epoch".
    epoch_start = _dt.datetime(2026, 5, 4, tzinfo=_dt.timezone.utc)
    monkeypatch.setattr(srd, "_read_epoch_start", lambda: epoch_start)
    # Skip dynamic_caps / halts I/O to keep the test hermetic.
    monkeypatch.setattr(srd, "_read_dynamic_caps", lambda: {})
    monkeypatch.setattr(srd, "_read_halts", lambda: {})

    monkeypatch.setattr(ke, "is_kraken_gate_enabled", lambda: True)
    monkeypatch.setattr(ke, "resolve_kraken_mode", lambda: "paper_kraken")
    monkeypatch.setattr(ke, "get_kraken_executor",
                        lambda: _fake_executor(allowed=True, txids=[]))

    intent = _make_paper_intent()
    logger = logging.getLogger("test.paper_kraken.diagnostics")
    ke.execute_kraken_intents(logger, [intent])

    diagnostics = srd.build_diagnostics(tracker=None)
    strategies = diagnostics.get("strategies") or {}
    alpha = strategies.get("alpha_crypto")
    assert alpha is not None, "alpha_crypto absent from diagnostics output"
    assert alpha.get("last_fill_at"), (
        f"alpha_crypto.last_fill_at empty after paper evidence write: {alpha}"
    )
    assert alpha.get("zero_fill_epoch2") is False, (
        f"alpha_crypto still flagged zero_fill_epoch2 after paper evidence: {alpha}"
    )


# --------------------------------------------------------------------------- #
# CRYPTO-TRUST U1 — kill-switch routing in execute_kraken_intents
# --------------------------------------------------------------------------- #

def test_paper_kraken_routes_to_trusted_engine_when_enabled(
    isolated_evidence_paths, monkeypatch
):
    from chad.core import kraken_execution as ke

    monkeypatch.setenv("CHAD_KRAKEN_TRUSTED_FILLS", "1")  # override fixture default
    monkeypatch.setattr(ke, "is_kraken_gate_enabled", lambda: True)
    monkeypatch.setattr(ke, "resolve_kraken_mode", lambda: "paper_kraken")
    monkeypatch.setattr(ke, "get_kraken_executor",
                        lambda: _fake_executor(allowed=True, txids=[]))

    calls = {"trusted": 0, "legacy": 0}

    class _FakeEngine:
        def process_intent(self, intent):
            calls["trusted"] += 1
            return {"trusted": True, "reason": "filled"}

    monkeypatch.setattr(ke, "_get_trusted_fill_engine", lambda: _FakeEngine())
    monkeypatch.setattr(ke, "_write_paper_kraken_evidence",
                        lambda *a, **k: calls.__setitem__("legacy", calls["legacy"] + 1))

    ke.execute_kraken_intents(logging.getLogger("t.trusted"), [_make_paper_intent()])
    assert calls["trusted"] == 1
    assert calls["legacy"] == 0  # trusted path used; legacy NOT double-written


def test_paper_kraken_falls_back_to_legacy_when_engine_declines(
    isolated_evidence_paths, monkeypatch
):
    from chad.core import kraken_execution as ke

    monkeypatch.setenv("CHAD_KRAKEN_TRUSTED_FILLS", "1")
    monkeypatch.setattr(ke, "is_kraken_gate_enabled", lambda: True)
    monkeypatch.setattr(ke, "resolve_kraken_mode", lambda: "paper_kraken")
    monkeypatch.setattr(ke, "get_kraken_executor",
                        lambda: _fake_executor(allowed=True, txids=[]))

    calls = {"legacy": 0}

    class _FakeEngine:
        def process_intent(self, intent):
            return {"trusted": False, "reason": "no_fresh_touch"}  # declines

    monkeypatch.setattr(ke, "_get_trusted_fill_engine", lambda: _FakeEngine())
    monkeypatch.setattr(ke, "_write_paper_kraken_evidence",
                        lambda *a, **k: calls.__setitem__("legacy", calls["legacy"] + 1))

    ke.execute_kraken_intents(logging.getLogger("t.fallback"), [_make_paper_intent()])
    assert calls["legacy"] == 1  # engine declined -> legacy fallback (no silent drop)
