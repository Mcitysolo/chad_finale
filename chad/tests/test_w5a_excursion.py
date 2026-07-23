"""W5A-3 — E3 MAE/MFE excursion (watermark extension + close-time sidecar).

R4: no second walker — the overlay's existing per-cycle walk is extended with
SEPARATE hwm/lwm watermark fields (peak/trough, which drive the ATR stop, are
untouched). OFF ⇒ byte-identical anchor state. R1 honest nulls in the metric.
"""

from __future__ import annotations

import json

from chad.analytics.excursion_recorder import (
    build_excursion_row,
    compute_mae_mfe,
    e3_mode,
    latest_bar_hilo,
    record_excursion_at_close,
    update_watermarks,
)


# --------------------------------------------------------------------------- #
# Mode + watermark math
# --------------------------------------------------------------------------- #

def test_mode_tristate():
    assert e3_mode({}) == "off"
    assert e3_mode({"CHAD_E3_EXCURSION": "sidecar"}) == "sidecar"
    assert e3_mode({"CHAD_E3_EXCURSION": "stamp"}) == "stamp"
    assert e3_mode({"CHAD_E3_EXCURSION": "on"}) == "off"  # garbage → off


def test_latest_bar_hilo():
    bars = [{"high": 10, "low": 8}, {"high": 12, "low": 9}]
    assert latest_bar_hilo(bars) == (12.0, 9.0)
    assert latest_bar_hilo([]) == (None, None)


def test_update_watermarks_folds_bar_hilo():
    # prior hwm 11 / lwm 9; point price 10; bar high 13 low 8 -> hwm 13, lwm 8
    wm = update_watermarks({"hwm": 11.0, "lwm": 9.0}, 10.0, 13.0, 8.0)
    assert wm["hwm"] == 13.0 and wm["lwm"] == 8.0
    assert wm["excursion_source"] == "watermark_bar_hilo"


def test_update_watermarks_point_only_when_no_bar():
    wm = update_watermarks(None, 10.0, None, None)
    assert wm["hwm"] == 10.0 and wm["lwm"] == 10.0
    assert wm["excursion_source"] == "watermark_point_only"


# --------------------------------------------------------------------------- #
# MAE/MFE — sign conventions + honest nulls (R1)
# --------------------------------------------------------------------------- #

def test_long_mae_mfe():
    # entry 100, hwm 110 (favorable), lwm 95 (adverse), long, 10 shares
    mm = compute_mae_mfe(entry_price=100, hwm=110, lwm=95, side="BUY",
                         quantity=10, contract_multiplier=1.0)
    assert mm["mfe_pct"] == 0.1 and mm["mfe_usd"] == 100.0
    assert mm["mae_pct"] == -0.05 and mm["mae_usd"] == -50.0


def test_short_mae_mfe():
    # short: favorable = down (lwm), adverse = up (hwm)
    mm = compute_mae_mfe(entry_price=100, hwm=108, lwm=96, side="SELL",
                         quantity=10, contract_multiplier=1.0)
    assert mm["mfe_pct"] == 0.04 and mm["mfe_usd"] == 40.0     # (100-96)/100
    assert mm["mae_pct"] == -0.08 and mm["mae_usd"] == -80.0   # (100-108)/100


def test_mae_mfe_null_without_entry():
    mm = compute_mae_mfe(entry_price=0, hwm=110, lwm=95, side="BUY", quantity=10)
    assert mm["mae_pct"] is None and mm["mfe_pct"] is None
    assert mm["mae_reason"] == "no_entry_price"


def test_mae_mfe_usd_null_without_qty():
    mm = compute_mae_mfe(entry_price=100, hwm=110, lwm=95, side="BUY", quantity=0)
    assert mm["mfe_pct"] == 0.1          # pct still resolved
    assert mm["mfe_usd"] is None         # usd honest-null
    assert mm["mfe_reason"] == "no_quantity_usd_only"


# --------------------------------------------------------------------------- #
# Sidecar row + writer
# --------------------------------------------------------------------------- #

def test_build_excursion_row():
    anchor = {"side": "BUY", "entry_price": 100.0, "hwm": 110.0, "lwm": 95.0,
              "qty": 10.0, "opened_at_utc": "2026-07-23T10:00:00Z",
              "excursion_source": "watermark_bar_hilo"}
    row = build_excursion_row("gamma|PSQ", anchor, lane="equity",
                              closed_detect_utc="2026-07-23T16:00:00Z")
    assert row["schema_version"] == "mae_mfe.v1"
    assert row["strategy"] == "gamma" and row["symbol"] == "PSQ"
    assert row["mfe_usd"] == 100.0 and row["mae_usd"] == -50.0
    assert row["lane"] == "equity"


def test_record_writes_dated_sidecar(tmp_path):
    anchor = {"side": "BUY", "entry_price": 100.0, "hwm": 110.0, "lwm": 95.0,
              "qty": 10.0, "opened_at_utc": "2026-07-23T10:00:00Z"}
    row = record_excursion_at_close("gamma|PSQ", anchor, lane="equity",
                                    evidence_dir=tmp_path,
                                    now_iso="2026-07-23T16:00:00Z")
    assert row is not None
    path = tmp_path / "excursion_20260723.ndjson"
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert rows[0]["mfe_usd"] == 100.0


# --------------------------------------------------------------------------- #
# Overlay walk extension: OFF byte-identical, ON tracks watermarks
# --------------------------------------------------------------------------- #

def _overlay_env():
    """Build a real evaluate_positions call that produces one anchor for a
    long BAC position (mirrors chad/tests/test_position_exit_overlay.py)."""
    from datetime import datetime, timedelta, timezone
    import chad.risk.position_exit_overlay as pxo

    NOW = datetime(2026, 7, 23, 16, tzinfo=timezone.utc)
    cfg = pxo.load_overlay_config({
        "mode": "shadow", "atr_period": 14, "atr_trail_mult": 2.5,
        "hard_stop_loss_pct": 0.08, "min_bars_for_atr": 16,
        "max_hold_days": {"equity": 20.0, "default": 20.0},
    })
    state = {"_version": 1}
    state["gamma|BAC"] = {"open": True, "symbol": "BAC", "side": "BUY",
                          "quantity": 100, "strategy": "gamma",
                          "opened_at": (NOW - timedelta(days=1)).isoformat()}
    state["broker_sync|BAC"] = {"open": False, "symbol": "BAC", "side": "BUY",
                                "quantity": 100, "strategy": "broker_sync"}
    open_positions = {k: v for k, v in state.items()
                      if isinstance(v, dict) and v.get("open") and not k.startswith("_")}
    bars = {"BAC": [{"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0}
                    for _ in range(20)]}
    return pxo, dict(
        open_positions=open_positions, guard_state=state, bars_by_symbol=bars,
        price_by_symbol={"BAC": 100.0}, anchors={}, config=cfg, now_utc=NOW,
    )


def test_walk_off_no_watermark_fields():
    pxo, kw = _overlay_env()
    res = pxo.evaluate_positions(**kw, excursion_mode="off")
    assert res.updated_anchors  # a real anchor was produced
    for anchor in res.updated_anchors.values():
        assert "hwm" not in anchor and "lwm" not in anchor  # byte-identical off


def test_walk_on_tracks_watermarks_separate_from_peak():
    pxo, kw = _overlay_env()
    res = pxo.evaluate_positions(**kw, excursion_mode="sidecar")
    assert res.updated_anchors
    for anchor in res.updated_anchors.values():
        assert "hwm" in anchor and "lwm" in anchor
        # true watermark folds bar high/low (100.5 / 99.5), distinct from the
        # point-sampled peak/trough (100.0) that drive the ATR stop.
        assert anchor["hwm"] == 100.5 and anchor["lwm"] == 99.5
        assert anchor["peak"] == 100.0 and anchor["trough"] == 100.0  # UNCHANGED
        assert anchor["side"] == "BUY" and anchor["qty"] == 100


def test_walk_on_is_observer_only_same_verdicts():
    """The watermark extension must not change any exit verdict (observer)."""
    pxo, kw = _overlay_env()
    off = pxo.evaluate_positions(**kw, excursion_mode="off")
    on = pxo.evaluate_positions(**kw, excursion_mode="sidecar")
    assert [v.verdict for v in off.verdicts] == [v.verdict for v in on.verdicts]
    assert [v.close_qty for v in off.verdicts] == [v.close_qty for v in on.verdicts]
