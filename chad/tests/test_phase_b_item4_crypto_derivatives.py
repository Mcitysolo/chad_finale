"""Phase B Item 4 — crypto derivatives publisher and crowding filter tests."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pytest

from chad.market_data import crypto_derivatives_publisher as pub
from chad.utils.crypto_signal_filter import (
    CROWDED_PENALTY,
    LEANING_PENALTY,
    CryptoFilterResult,
    get_crypto_filter,
)


REPO_ROOT = Path("/home/ubuntu/chad_finale")
DEPLOY_DIR = REPO_ROOT / "deploy"


def _ticker(symbol: str, **overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "symbol": symbol,
        "fundingRate": 0.0,
        "openInterest": 1000.0,
        "indexPrice": 100.0,
        "markPrice": 100.0,
        "last": 100.0,
        "bid": 99.9,
        "ask": 100.1,
        "vol24h": 50000.0,
    }
    base.update(overrides)
    return base


def _balanced_tickers() -> List[Dict[str, Any]]:
    return [
        _ticker(
            "PF_XBTUSD",
            fundingRate=0.00005,
            openInterest=2000.0,
            indexPrice=80000.0,
            vol24h=5000.0,
        ),
        _ticker(
            "PF_ETHUSD",
            fundingRate=0.00002,
            openInterest=20000.0,
            indexPrice=2200.0,
            vol24h=30000.0,
        ),
        _ticker(
            "PF_SOLUSD",
            fundingRate=0.00001,
            openInterest=250000.0,
            indexPrice=90.0,
            vol24h=500000.0,
        ),
    ]


# ---------------------------------------------------------------------------
# 1. Publisher payload tests
# ---------------------------------------------------------------------------

def test_1_build_payload_parses_btc_eth_sol_tickers() -> None:
    payload = pub.build_payload(_balanced_tickers())
    syms = payload["symbols"]
    assert set(syms.keys()) == {"BTC-USD", "ETH-USD", "SOL-USD"}
    btc = syms["BTC-USD"]
    assert btc["kraken_symbol"] == "PF_XBTUSD"
    assert btc["data_available"] is True
    assert btc["funding_rate_8h"] == pytest.approx(0.00005)
    assert btc["open_interest_contracts"] == pytest.approx(2000.0)
    assert btc["index_price"] == pytest.approx(80000.0)
    assert btc["open_interest_usd"] == pytest.approx(2000.0 * 80000.0)
    assert payload["schema_version"] == pub.SCHEMA_VERSION
    assert payload["status"] == "ok"
    assert payload["source"]["provider"] == "kraken_futures_public"
    assert payload["source"]["provider_status"] == "real"
    assert payload["ttl_seconds"] == pub.DEFAULT_TTL_SECONDS


def test_2_funding_rate_annualized_equals_8h_times_3_times_365() -> None:
    tickers = [
        _ticker(
            "PF_XBTUSD",
            fundingRate=0.0001,
            openInterest=1000.0,
            indexPrice=70000.0,
        ),
    ]
    payload = pub.build_payload(tickers)
    btc = payload["symbols"]["BTC-USD"]
    assert btc["funding_rate_8h"] == pytest.approx(0.0001)
    assert btc["funding_rate_annualized"] == pytest.approx(0.0001 * 3 * 365)


def test_3_positive_extreme_funding_classifies_long_crowded() -> None:
    tickers = [
        _ticker("PF_XBTUSD", fundingRate=0.001, openInterest=1.0, indexPrice=1.0),
    ]
    payload = pub.build_payload(tickers)
    btc = payload["symbols"]["BTC-USD"]
    assert btc["market_bias"] == "long_crowded"
    assert btc["funding_extreme_long"] is True
    assert btc["funding_extreme_short"] is False
    assert btc["funding_elevated_long"] is False
    assert payload["summary"]["long_crowded_count"] >= 1


def test_4_negative_extreme_funding_classifies_short_crowded() -> None:
    tickers = [
        _ticker("PF_ETHUSD", fundingRate=-0.001, openInterest=1.0, indexPrice=1.0),
    ]
    payload = pub.build_payload(tickers)
    eth = payload["symbols"]["ETH-USD"]
    assert eth["market_bias"] == "short_crowded"
    assert eth["funding_extreme_short"] is True
    assert eth["funding_extreme_long"] is False
    assert payload["summary"]["short_crowded_count"] >= 1


def test_5_elevated_positive_funding_classifies_long_leaning() -> None:
    tickers = [
        _ticker("PF_SOLUSD", fundingRate=0.00015, openInterest=1.0, indexPrice=1.0),
    ]
    payload = pub.build_payload(tickers)
    sol = payload["symbols"]["SOL-USD"]
    assert sol["market_bias"] == "long_leaning"
    assert sol["funding_elevated_long"] is True
    assert sol["funding_extreme_long"] is False


def test_6_balanced_funding_classifies_balanced() -> None:
    tickers = [
        _ticker("PF_XBTUSD", fundingRate=0.00001, openInterest=1.0, indexPrice=1.0),
    ]
    payload = pub.build_payload(tickers)
    btc = payload["symbols"]["BTC-USD"]
    assert btc["market_bias"] == "balanced"
    assert btc["funding_extreme_long"] is False
    assert btc["funding_extreme_short"] is False
    assert btc["funding_elevated_long"] is False


def test_7_missing_mapped_symbol_emits_data_unavailable_unknown_bias() -> None:
    tickers = [
        _ticker("PF_XBTUSD", fundingRate=0.00005, openInterest=1.0, indexPrice=1.0),
    ]
    payload = pub.build_payload(tickers)
    eth = payload["symbols"]["ETH-USD"]
    sol = payload["symbols"]["SOL-USD"]
    assert eth["data_available"] is False
    assert eth["market_bias"] == "unknown"
    assert eth["funding_rate_8h"] is None
    assert sol["data_available"] is False
    assert sol["market_bias"] == "unknown"
    assert payload["status"] == "partial"
    assert payload["source"]["provider_status"] == "partial"
    assert payload["summary"]["symbols_fetched"] == 1


def test_8_oi_change_pct_computed_from_prior_runtime_file(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    prior_path = runtime / "crypto_derivatives.json"
    prior_payload = pub.build_payload(
        [
            _ticker(
                "PF_XBTUSD",
                fundingRate=0.00001,
                openInterest=1000.0,
                indexPrice=80000.0,
            ),
            _ticker(
                "PF_ETHUSD",
                fundingRate=0.00001,
                openInterest=20000.0,
                indexPrice=2200.0,
            ),
            _ticker(
                "PF_SOLUSD",
                fundingRate=0.00001,
                openInterest=250000.0,
                indexPrice=90.0,
            ),
        ]
    )
    prior_path.write_text(json.dumps(prior_payload), encoding="utf-8")

    new_tickers = [
        _ticker(
            "PF_XBTUSD",
            fundingRate=0.00001,
            openInterest=1100.0,
            indexPrice=80000.0,
        ),
        _ticker(
            "PF_ETHUSD",
            fundingRate=0.00001,
            openInterest=20000.0,
            indexPrice=2200.0,
        ),
        _ticker(
            "PF_SOLUSD",
            fundingRate=0.00001,
            openInterest=250000.0,
            indexPrice=90.0,
        ),
    ]
    payload, written = pub.publish(
        runtime_dir=runtime,
        dry_run=False,
        fetcher=lambda: new_tickers,
    )
    assert written is True
    btc = payload["symbols"]["BTC-USD"]
    prior_oi_usd = 1000.0 * 80000.0
    new_oi_usd = 1100.0 * 80000.0
    expected_change = (new_oi_usd - prior_oi_usd) / prior_oi_usd
    assert btc["oi_change_pct"] == pytest.approx(expected_change)


def test_9_publisher_dry_run_with_fake_fetcher_returns_valid_schema(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    payload, written = pub.publish(
        runtime_dir=runtime,
        dry_run=True,
        fetcher=_balanced_tickers,
    )
    assert written is False
    assert not (runtime / "crypto_derivatives.json").exists()
    assert payload["schema_version"] == "crypto_derivatives.v1"
    assert payload["status"] == "ok"
    assert set(payload["symbols"].keys()) == {"BTC-USD", "ETH-USD", "SOL-USD"}
    assert payload["ttl_seconds"] == pub.DEFAULT_TTL_SECONDS


# ---------------------------------------------------------------------------
# 10-15. Filter rules
# ---------------------------------------------------------------------------

def _write_snapshot(runtime: Path, payload: Mapping[str, Any]) -> None:
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "crypto_derivatives.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_10_get_crypto_filter_missing_file_fails_open(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    result = get_crypto_filter("BTC-USD", "BUY", runtime_dir=runtime)
    assert isinstance(result, CryptoFilterResult)
    assert result.confidence_adjustment == 0.0
    assert result.market_bias == "unknown"
    assert result.funding_rate_8h is None
    assert result.funding_extreme is False


def test_11_get_crypto_filter_stale_file_fails_open(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    stale_ts = (datetime.now(timezone.utc) - timedelta(seconds=7200)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    payload = pub.build_payload(
        [_ticker("PF_XBTUSD", fundingRate=0.001, openInterest=1.0, indexPrice=1.0)],
    )
    payload["ts_utc"] = stale_ts
    payload["ttl_seconds"] = 60
    _write_snapshot(runtime, payload)
    result = get_crypto_filter("BTC-USD", "BUY", runtime_dir=runtime)
    assert result.confidence_adjustment == 0.0
    assert result.market_bias == "unknown"


def test_12_get_crypto_filter_buy_long_crowded_returns_minus_020(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    payload = pub.build_payload(
        [_ticker("PF_XBTUSD", fundingRate=0.001, openInterest=1.0, indexPrice=1.0)],
        now_utc=datetime.now(timezone.utc),
    )
    _write_snapshot(runtime, payload)
    result = get_crypto_filter("BTC-USD", "BUY", runtime_dir=runtime)
    assert result.confidence_adjustment == pytest.approx(-CROWDED_PENALTY)
    assert result.market_bias == "long_crowded"
    assert result.funding_extreme is True


def test_13_get_crypto_filter_sell_short_crowded_returns_minus_020(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    payload = pub.build_payload(
        [_ticker("PF_ETHUSD", fundingRate=-0.001, openInterest=1.0, indexPrice=1.0)],
        now_utc=datetime.now(timezone.utc),
    )
    _write_snapshot(runtime, payload)
    result = get_crypto_filter("ETH-USD", "SELL", runtime_dir=runtime)
    assert result.confidence_adjustment == pytest.approx(-CROWDED_PENALTY)
    assert result.market_bias == "short_crowded"
    assert result.funding_extreme is True


def test_14_get_crypto_filter_buy_long_leaning_returns_minus_005(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    payload = pub.build_payload(
        [_ticker("PF_SOLUSD", fundingRate=0.00015, openInterest=1.0, indexPrice=1.0)],
        now_utc=datetime.now(timezone.utc),
    )
    _write_snapshot(runtime, payload)
    result = get_crypto_filter("SOL-USD", "BUY", runtime_dir=runtime)
    assert result.confidence_adjustment == pytest.approx(-LEANING_PENALTY)
    assert result.market_bias == "long_leaning"
    assert result.funding_extreme is False


def test_15_get_crypto_filter_sell_long_crowded_returns_zero(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    payload = pub.build_payload(
        [_ticker("PF_XBTUSD", fundingRate=0.001, openInterest=1.0, indexPrice=1.0)],
        now_utc=datetime.now(timezone.utc),
    )
    _write_snapshot(runtime, payload)
    result = get_crypto_filter("BTC-USD", "SELL", runtime_dir=runtime)
    assert result.confidence_adjustment == 0.0
    assert result.market_bias == "long_crowded"
    assert result.funding_extreme is True


# ---------------------------------------------------------------------------
# 16-17. alpha_crypto integration
# ---------------------------------------------------------------------------

def test_16_alpha_crypto_handler_no_raise_with_missing_runtime_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """alpha_crypto_handler must not raise when no derivatives snapshot exists.

    Verified by pointing the filter's runtime dir at an empty tmp directory
    via monkeypatching the module-level default, then invoking the handler
    with a minimal ctx that produces no signals (no bars). The handler must
    return [] without raising.
    """
    from chad.strategies import alpha_crypto

    # Re-route the default runtime dir used by the filter to an empty path.
    empty_runtime = tmp_path / "runtime_empty"
    empty_runtime.mkdir()
    monkeypatch.setattr(
        "chad.utils.crypto_signal_filter._DEFAULT_RUNTIME_DIR",
        empty_runtime,
        raising=True,
    )

    class _Ctx:
        prices: Dict[str, float] = {}
        bars: Dict[str, list] = {}

    params = alpha_crypto.AlphaCryptoParams(enabled=True)
    # Should not raise even though no crypto_derivatives.json exists.
    out = alpha_crypto.alpha_crypto_handler(_Ctx(), params)
    assert isinstance(out, list)


def test_17_alpha_crypto_signal_meta_contains_crypto_fields_via_static_check() -> (
    None
):
    """Static check: alpha_crypto wires the crowding-filter meta keys into
    the TradeSignal construction. A runtime fixture is impractical because
    the handler requires 22+ daily bars and a non-adverse regime; instead we
    verify the source assigns the four new meta fields adjacent to
    TradeSignal construction.
    """
    src = (REPO_ROOT / "chad/strategies/alpha_crypto.py").read_text(
        encoding="utf-8"
    )
    assert "from chad.utils.crypto_signal_filter import" in src
    assert "get_crypto_filter(symbol, side.value)" in src
    assert '"crypto_market_bias": _crypto_filter.market_bias' in src
    assert '"crypto_funding_rate_8h": _crypto_filter.funding_rate_8h' in src
    assert '"crypto_funding_extreme": _crypto_filter.funding_extreme' in src
    assert "_crypto_filter.confidence_adjustment" in src
    # Confidence adjustment must be applied with the clamp pattern.
    assert "min(0.95, float(confidence) + _crypto_filter.confidence_adjustment)" in src


# ---------------------------------------------------------------------------
# 18. Deploy files
# ---------------------------------------------------------------------------

def test_18_deploy_service_and_timer_files_exist_with_expected_content() -> None:
    service = DEPLOY_DIR / "chad-crypto-derivatives-refresh.service"
    timer = DEPLOY_DIR / "chad-crypto-derivatives-refresh.timer"

    assert service.is_file(), f"missing {service}"
    assert timer.is_file(), f"missing {timer}"

    s = service.read_text(encoding="utf-8")
    assert (
        "ExecStart=/home/ubuntu/chad_finale/venv/bin/python3 -m chad.market_data.crypto_derivatives_publisher"
        in s
    )
    assert "Type=oneshot" in s
    assert "User=ubuntu" in s
    assert "WorkingDirectory=/home/ubuntu/chad_finale" in s

    t = timer.read_text(encoding="utf-8")
    assert "OnUnitActiveSec=300" in t
    assert "OnBootSec=120" in t
    assert "Persistent=true" in t
