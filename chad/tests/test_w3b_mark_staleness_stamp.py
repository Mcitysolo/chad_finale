"""W3B-6 — mark-provenance stamps in exit-overlay evidence (exit_overlay.v2).

PA_SIM_MARK_freshness_2026-07-20 documented a 55.3s-stale ref price
($1.88/sh divergence on a 273-sh UNH close) that was invisible in overlay
evidence: both lanes read a mark, used its timestamp for a TTL/freshness
gate, and then DISCARDED it. v2 adds mark_ts_utc / mark_age_s / mark_source
(additive; the stamp makes staleness measurable, it does not shrink it).

Locked properties:
- equity: price_cache marks stamped with the cache's ts_utc + source;
- equity: bar-close fallback labeled bar_close_fallback (never masquerades
  as a live mark);
- crypto: per-symbol kraken tick ts stamped, source kraken_ws_tick;
- backward-compat: plain-dict loaders (all pre-v2 injections) still work,
  fields stay None;
- default loaders return (prices, meta) tuples with honest timestamps.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from chad.risk import position_exit_overlay as pxo

NOW = datetime(2026, 7, 22, 21, 0, 0, tzinfo=timezone.utc)


def _guard(strategy_entries, broker_entries):
    """Minimal guard state: strategy legs + broker_sync mirror legs."""
    state = {"_version": 1}
    for key, (sym, side, qty, _age) in strategy_entries.items():
        state[key] = {"symbol": sym, "side": side, "quantity": float(qty),
                      "open": True, "strategy": key.split("|")[0]}
    for sym, (side, qty) in broker_entries.items():
        state[f"broker_sync|{sym}"] = {"symbol": sym, "side": side,
                                       "quantity": float(qty), "open": True,
                                       "strategy": "broker_sync"}
    return state


def _open_positions(state):
    return {k: v for k, v in state.items()
            if isinstance(v, dict) and not k.startswith("broker_sync|")
            and not k.startswith("_")}


def _cfg():
    return pxo.load_overlay_config({
        "mode": "shadow",
        "atr_period": 14,
        "atr_trail_mult": 2.5,
        "hard_stop_loss_pct": 0.08,
        "min_bars_for_atr": 16,
        "max_hold_days": {"equity": 20.0, "etf": 30.0, "default": 20.0},
    })


def _evaluate(state, prices, *, meta=None, bars=None, anchors=None):
    return pxo.evaluate_positions(
        open_positions=_open_positions(state),
        guard_state=state,
        bars_by_symbol=bars or {},
        price_by_symbol=prices,
        price_meta_by_symbol=meta,
        anchors=anchors or {"gamma|BAC": {"entry_price": 100.0, "peak": 100.0,
                                          "trough": 100.0,
                                          "first_seen_utc": NOW.isoformat()}},
        config=_cfg(),
        now_utc=NOW,
    )


def _one_state():
    return _guard({"gamma|BAC": ("BAC", "BUY", 100, 1)}, {"BAC": ("BUY", 100)})


def _verdict(res, key="gamma|BAC"):
    return next(v for v in res.verdicts if v.position_key == key)


# ---------------------------------------------------------------------------
# equity lane
# ---------------------------------------------------------------------------


def test_schema_version_is_v2():
    assert pxo.EVIDENCE_SCHEMA_VERSION == "exit_overlay.v2"
    res = _evaluate(_one_state(), {"BAC": 101.0})
    d = _verdict(res).to_dict()
    assert d["schema_version"] == "exit_overlay.v2"
    # the three provenance fields always serialize (None when unstamped)
    assert set(("mark_ts_utc", "mark_age_s", "mark_source")) <= set(d)


def test_price_cache_mark_is_stamped_with_age():
    cache_ts = (NOW - timedelta(seconds=55)).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = {"BAC": {"ts_utc": cache_ts, "source": "price_cache"}}
    v = _verdict(_evaluate(_one_state(), {"BAC": 101.0}, meta=meta))
    assert v.mark_source == "price_cache"
    assert v.mark_ts_utc == cache_ts
    # the PA's exact divergence class is now measurable
    assert 54.0 <= v.mark_age_s <= 56.0


def test_plain_dict_loader_backward_compat_fields_none():
    v = _verdict(_evaluate(_one_state(), {"BAC": 101.0}, meta=None))
    assert v.mark_ts_utc is None and v.mark_age_s is None and v.mark_source is None
    assert v.price == 101.0  # behavior unchanged


def test_bar_fallback_is_labeled_and_never_masquerades():
    bar_ts = (NOW - timedelta(hours=22)).strftime("%Y-%m-%dT%H:%M:%SZ")
    bars = {"BAC": [{"close": 99.5, "ts_utc": bar_ts}]}
    # no live price -> falls back to the bar close
    v = _verdict(_evaluate(_one_state(), {}, meta={}, bars=bars))
    assert v.price == 99.5
    assert v.mark_source == "bar_close_fallback"
    assert v.mark_ts_utc == bar_ts
    assert v.mark_age_s > 21 * 3600


def test_skip_no_data_carries_no_stamp():
    res = _evaluate(_one_state(), {}, meta={})
    v = _verdict(res)
    assert v.verdict == "SKIP_NO_DATA"
    assert v.mark_source is None and v.mark_ts_utc is None


def test_default_price_loader_returns_meta(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    cache_ts = (datetime.now(timezone.utc) - timedelta(seconds=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    (runtime / "price_cache.json").write_text(json.dumps({
        "prices": {"BAC": 101.0, "UNH": 423.0},
        "ts_utc": cache_ts,
        "ttl_seconds": 300,
    }), encoding="utf-8")
    loader = pxo._default_price_loader(tmp_path)
    prices, meta = loader(["BAC", "UNH", "MISSING"])
    assert prices == {"BAC": 101.0, "UNH": 423.0}
    assert meta["BAC"] == {"ts_utc": cache_ts, "source": "price_cache"}
    assert meta["UNH"]["ts_utc"] == cache_ts
    assert "MISSING" not in meta


def test_default_price_loader_stale_ttl_still_returns_empty(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    old_ts = (datetime.now(timezone.utc) - timedelta(seconds=600)).strftime("%Y-%m-%dT%H:%M:%SZ")
    (runtime / "price_cache.json").write_text(json.dumps({
        "prices": {"BAC": 101.0}, "ts_utc": old_ts, "ttl_seconds": 300,
    }), encoding="utf-8")
    loader = pxo._default_price_loader(tmp_path)
    assert loader(["BAC"]) == {}  # TTL rejection behavior unchanged (v1 parity)


# ---------------------------------------------------------------------------
# crypto lane
# ---------------------------------------------------------------------------


def _crypto_eval(marks, meta):
    from chad.risk import crypto_exit_overlay as cxo

    snap = cxo.CryptoLotSnapshot(
        strategy="alpha_crypto", symbol="BTC-USD", direction="long",
        qty=0.5, entry_price=60000.0,
        opened_at_utc=(NOW - timedelta(days=1)).isoformat(), lots=1,
    )
    return cxo.evaluate_crypto_positions(
        snapshots=[snap], marks_by_symbol=marks, marks_meta_by_symbol=meta,
        bars_by_symbol={}, anchors={}, config=_cfg(), now_utc=NOW,
    )


def test_crypto_tick_mark_is_stamped():
    tick_ts = (NOW - timedelta(seconds=12)).strftime("%Y-%m-%dT%H:%M:%SZ")
    res = _crypto_eval({"BTC-USD": 61000.0},
                       {"BTC-USD": {"ts_utc": tick_ts, "source": "kraken_ws_tick"}})
    v = next(x for x in res.verdicts if x.symbol == "BTC-USD")
    assert v.mark_source == "kraken_ws_tick"
    assert v.mark_ts_utc == tick_ts
    assert 11.0 <= v.mark_age_s <= 13.0


def test_crypto_plain_mapping_backward_compat():
    res = _crypto_eval({"BTC-USD": 61000.0}, None)
    v = next(x for x in res.verdicts if x.symbol == "BTC-USD")
    assert v.mark_ts_utc is None and v.mark_source is None


def test_crypto_default_marks_loader_returns_meta(tmp_path):
    from chad.risk import crypto_exit_overlay as cxo

    tick_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    prices_path = tmp_path / "kraken_prices.json"
    prices_path.write_text(json.dumps({
        "ts_utc": tick_ts, "ttl_seconds": 30,
        "ticks": {"BTC-USD": {"bid": 60990.0, "ask": 61010.0, "last": 61000.0,
                              "ts_utc": tick_ts}},
    }), encoding="utf-8")
    loader = cxo._default_marks_loader(prices_path, 60.0)
    marks, meta = loader(["BTC-USD"])
    assert marks["BTC-USD"] == 61000.0
    assert meta["BTC-USD"]["source"] == "kraken_ws_tick"
    assert meta["BTC-USD"]["ts_utc"] == tick_ts
