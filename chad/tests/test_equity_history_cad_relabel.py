"""
Equity-history CAD relabel + forward-only USD (continuity-safe).

Pins the (b) equity relabel:

* equity_history_publisher writes the broker-native CAD legs under honest
  ``*_cad`` keys, and ADDS forward-only authoritative ``*_usd`` fields sourced
  from the snapshot's (a) fields. The USD fields are fail-closed: null when
  ``usd_ok`` is false, NEVER a CAD fallback.
* The drawdown chain (chad.risk.drawdown_guard) reads the CONTINUOUS CAD series
  — preferring ``total_equity_cad`` and falling back to the legacy
  ``total_equity_usd`` (which historically held the same CAD figure) — so the
  relabel does NOT introduce a phantom drawdown discontinuity there.
* The withdrawal chain (chad.risk.withdrawal_manager) now reads the true-USD v2
  ``total_equity_usd`` column (``usd_ok`` rows only); CAD / pre-v2 rows are
  skipped fail-closed, so the salary HWM is genuine USD, never a CAD peak.

Report-only surface — no execution path is exercised here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from chad.ops import equity_history_publisher as ehp
from chad.risk import drawdown_guard, withdrawal_manager as wm_mod
from chad.risk.withdrawal_manager import DEFAULT_POLICY, compute_authorization


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _write_ndjson(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _run_publisher(tmp_path: Path, snap: Dict[str, Any], monkeypatch) -> Dict[str, Any]:
    """Run equity_history_publisher.main() against an isolated runtime and return
    the single appended record."""
    rt = tmp_path / "runtime"
    rt.mkdir(parents=True, exist_ok=True)
    snap_path = rt / "portfolio_snapshot.json"
    hist_path = rt / "equity_history.ndjson"
    _write_json(snap_path, snap)

    monkeypatch.setattr(ehp, "RUNTIME_DIR", rt)
    monkeypatch.setattr(ehp, "SNAPSHOT_PATH", snap_path)
    monkeypatch.setattr(ehp, "HISTORY_PATH", hist_path)

    rc = ehp.main()
    assert rc == 0, "publisher must succeed"
    lines = [ln for ln in hist_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1, f"expected exactly one appended record, got {len(lines)}"
    return json.loads(lines[0])


# ---------------------------------------------------------------------------
# 1. Publisher: *_cad present (CAD) and *_usd true-USD when usd_ok=true
# ---------------------------------------------------------------------------


def test_publisher_writes_cad_and_true_usd_when_usd_ok(tmp_path, monkeypatch):
    snap = {
        # broker-native CAD legs
        "ibkr_equity": 197000.0,
        "kraken_equity": 300.0,
        "coinbase_equity": 0.0,
        # (a) authoritative USD fields
        "usd_ok": True,
        "total_equity_usd_authoritative": 137250.0,
        "ibkr_equity_usd_display": 137042.0,
    }
    rec = _run_publisher(tmp_path, snap, monkeypatch)

    # Honest CAD series == the snapshot CAD legs.
    assert rec["total_equity_cad"] == pytest.approx(197300.0)
    assert rec["ibkr_equity_cad"] == pytest.approx(197000.0)
    assert rec["kraken_equity_cad"] == pytest.approx(300.0)
    assert rec["coinbase_equity_cad"] == pytest.approx(0.0)

    # Forward-only true USD == the (a) authoritative figures (NOT the CAD sum).
    assert rec["total_equity_usd"] == pytest.approx(137250.0)
    assert rec["ibkr_equity_usd"] == pytest.approx(137042.0)
    assert rec["usd_ok"] is True
    assert rec["schema_version"] == "equity_history.v2"

    # The dishonest part is gone: total_equity_usd must NOT carry the CAD figure.
    assert rec["total_equity_usd"] != pytest.approx(rec["total_equity_cad"])


# ---------------------------------------------------------------------------
# 2. Publisher: usd_ok=false -> *_usd null, *_cad present (fail-closed, never CAD)
# ---------------------------------------------------------------------------


def test_publisher_usd_fields_null_when_usd_ok_false(tmp_path, monkeypatch):
    snap = {
        "ibkr_equity": 197000.0,
        "kraken_equity": 300.0,
        "coinbase_equity": 0.0,
        "usd_ok": False,
        "total_equity_usd_authoritative": None,
        "ibkr_equity_usd_display": None,
    }
    rec = _run_publisher(tmp_path, snap, monkeypatch)

    # CAD series still present and correct.
    assert rec["total_equity_cad"] == pytest.approx(197300.0)
    assert rec["ibkr_equity_cad"] == pytest.approx(197000.0)

    # USD fields fail closed to null — never a CAD fallback.
    assert rec["total_equity_usd"] is None
    assert rec["ibkr_equity_usd"] is None
    assert rec["usd_ok"] is False


def test_publisher_usd_fields_null_when_components_missing_even_if_usd_ok(tmp_path, monkeypatch):
    """Belt-and-suspenders: usd_ok true but the authoritative component absent ->
    still null (never fabricate / fall back to CAD)."""
    snap = {
        "ibkr_equity": 197000.0,
        "kraken_equity": 300.0,
        "coinbase_equity": 0.0,
        "usd_ok": True,
        # components missing entirely
    }
    rec = _run_publisher(tmp_path, snap, monkeypatch)
    assert rec["total_equity_cad"] == pytest.approx(197300.0)
    assert rec["total_equity_usd"] is None
    assert rec["ibkr_equity_usd"] is None


# ---------------------------------------------------------------------------
# 3. Drawdown continuity: CAD->CAD relabel leaves drawdown % unchanged
#    (no phantom drop; true-USD field is never read for the math)
# ---------------------------------------------------------------------------


def test_drawdown_continuity_cad_relabel_no_phantom_drop(tmp_path):
    rt = tmp_path / "runtime"
    rt.mkdir()

    # Legacy v1 series: total_equity_usd actually held the CAD figure.
    legacy = [
        {"date_utc": "2026-04-01", "total_equity_usd": 200000.0},
        {"date_utc": "2026-04-02", "total_equity_usd": 190000.0},
        {"date_utc": "2026-04-03", "total_equity_usd": 180000.0},
    ]
    legacy_path = rt / "equity_history_legacy.ndjson"
    _write_ndjson(legacy_path, legacy)

    # v2 series: SAME CAD figures under total_equity_cad, PLUS a much lower
    # true-USD figure under total_equity_usd. If the chain wrongly read the USD
    # field, the drawdown would change (phantom).
    v2 = [
        {"date_utc": "2026-04-01", "total_equity_cad": 200000.0, "total_equity_usd": 140000.0, "usd_ok": True},
        {"date_utc": "2026-04-02", "total_equity_cad": 190000.0, "total_equity_usd": 133000.0, "usd_ok": True},
        {"date_utc": "2026-04-03", "total_equity_cad": 180000.0, "total_equity_usd": 126000.0, "usd_ok": True},
    ]
    v2_path = rt / "equity_history_v2.ndjson"
    _write_ndjson(v2_path, v2)

    # Current equity comes from the snapshot CAD legs (unchanged by the relabel).
    snap_path = rt / "portfolio_snapshot.json"
    _write_json(snap_path, {"ibkr_equity": 180000.0})

    dd_legacy = drawdown_guard.compute_drawdown(
        equity_history_path=legacy_path, portfolio_snapshot_path=snap_path
    )
    dd_v2 = drawdown_guard.compute_drawdown(
        equity_history_path=v2_path, portfolio_snapshot_path=snap_path
    )

    # Identical HWM and drawdown across the relabel -> no phantom discontinuity.
    assert dd_v2.hwm_cad == dd_legacy.hwm_cad == pytest.approx(200000.0)
    assert dd_v2.current_equity_cad == dd_legacy.current_equity_cad == pytest.approx(180000.0)
    assert dd_v2.drawdown_pct == pytest.approx(dd_legacy.drawdown_pct)
    assert dd_v2.drawdown_pct == pytest.approx((180000.0 - 200000.0) / 200000.0 * 100.0)

    # Explicitly: the chain did NOT read the true-USD field (would have put HWM at
    # ~140k and drawdown near 0%).
    assert dd_v2.hwm_cad != pytest.approx(140000.0)
    assert dd_v2.drawdown_pct < -5.0


def test_drawdown_v2_usd_ok_false_unaffected(tmp_path):
    """usd_ok false -> total_equity_usd is null on v2 rows. The CAD series still
    drives the drawdown; no crash, drawdown computed from *_cad."""
    rt = tmp_path / "runtime"
    rt.mkdir()
    v2 = [
        {"date_utc": "2026-04-01", "total_equity_cad": 200000.0, "total_equity_usd": None, "usd_ok": False},
        {"date_utc": "2026-04-02", "total_equity_cad": 150000.0, "total_equity_usd": None, "usd_ok": False},
    ]
    hist = rt / "equity_history.ndjson"
    _write_ndjson(hist, v2)
    snap = rt / "portfolio_snapshot.json"
    _write_json(snap, {"ibkr_equity": 150000.0})

    dd = drawdown_guard.compute_drawdown(equity_history_path=hist, portfolio_snapshot_path=snap)
    assert dd.status == "ok"
    assert dd.hwm_cad == pytest.approx(200000.0)
    assert dd.current_equity_cad == pytest.approx(150000.0)
    assert dd.drawdown_pct == pytest.approx((150000.0 - 200000.0) / 200000.0 * 100.0)


# ---------------------------------------------------------------------------
# 4. Withdrawal chain reads the continuous CAD series too (no crash on null USD)
# ---------------------------------------------------------------------------


def test_withdrawal_hwm_uses_true_usd_v2_series():
    """compute_authorization HWM must come from the v2 true-USD
    ``total_equity_usd`` column (usd_ok rows only), NOT the CAD peak."""
    pol = dict(DEFAULT_POLICY)
    history = [
        {"ts_utc": "2026-04-01T00:00:00Z", "total_equity_cad": 200000.0, "total_equity_usd": 140000.0, "usd_ok": True},
        {"ts_utc": "2026-04-02T00:00:00Z", "total_equity_cad": 190000.0, "total_equity_usd": 133000.0, "usd_ok": True},
    ]
    out = compute_authorization(
        current_equity=140000.0, history=history, scr_state="CONFIDENT", policy=pol
    )
    # HWM from the true-USD peak (140k), not the CAD peak (200k).
    assert out.high_water_mark_usd == pytest.approx(140000.0)
    assert out.high_water_mark_currency == "USD"
    assert out.high_water_mark_currency_ok is True


def test_withdrawal_no_crash_on_v2_null_usd_rows():
    """v2 rows with usd_ok false carry total_equity_usd=None and are SKIPPED
    (fail-closed). With no usable USD row, HWM falls back to current_equity —
    no crash, and never a CAD value mislabeled as USD."""
    pol = dict(DEFAULT_POLICY)
    history = [
        {"ts_utc": "2026-04-01T00:00:00Z", "total_equity_cad": 200000.0, "total_equity_usd": None, "usd_ok": False},
        {"ts_utc": "2026-04-02T00:00:00Z", "total_equity_cad": 195000.0, "total_equity_usd": None, "usd_ok": False},
    ]
    out = compute_authorization(
        current_equity=195000.0, history=history, scr_state="CONFIDENT", policy=pol
    )
    assert out.high_water_mark_usd == pytest.approx(195000.0)


def test_withdrawal_legacy_v1_rows_skipped_no_crash():
    """Legacy v1 rows (total_equity_usd holding CAD, no usd_ok marker) are now
    SKIPPED for the USD HWM — never treated as USD. With no v2 USD rows the HWM
    fails closed to current_equity; no crash, no CAD-as-USD leak."""
    pol = dict(DEFAULT_POLICY)
    history = [
        {"ts_utc": "2026-04-01T00:00:00Z", "total_equity_usd": 200000.0},
        {"ts_utc": "2026-04-02T00:00:00Z", "total_equity_usd": 190000.0},
    ]
    out = compute_authorization(
        current_equity=190000.0, history=history, scr_state="CONFIDENT", policy=pol
    )
    assert out.high_water_mark_usd == pytest.approx(190000.0)
