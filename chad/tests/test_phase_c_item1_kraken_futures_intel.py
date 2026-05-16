from __future__ import annotations

"""Phase C Item 1A — Kraken Futures public intelligence publisher tests."""

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pytest

from chad.market_data import kraken_futures_intel_publisher as kfip


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ticker(
    symbol: str,
    *,
    funding: float | None = 0.0,
    oi: float | None = 100.0,
    index: float | None = 10.0,
    last: float | None = 10.0,
    bid: float | None = 9.99,
    ask: float | None = 10.01,
    vol24h: float | None = 1000.0,
    volume_quote: float | None = 5000.0,
    change24h: float | None = 1.0,
    mark: float | None = 10.0,
    post_only: bool = False,
    suspended: bool = False,
    funding_prediction: float | None = 0.0,
) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "fundingRate": funding,
        "fundingRatePrediction": funding_prediction,
        "openInterest": oi,
        "indexPrice": index,
        "markPrice": mark,
        "last": last,
        "bid": bid,
        "ask": ask,
        "vol24h": vol24h,
        "volumeQuote": volume_quote,
        "change24h": change24h,
        "postOnly": post_only,
        "suspended": suspended,
    }


def _sample_tickers() -> List[Dict[str, Any]]:
    return [
        _ticker(
            "PF_XBTUSD",
            funding=0.0005,  # > extreme threshold → long_crowded
            oi=1000.0,
            index=50000.0,
            mark=50001.0,
            last=49999.0,
            bid=49998.0,
            ask=50000.0,
            vol24h=200.0,
            volume_quote=10_000_000.0,
        ),
        _ticker(
            "PF_ETHUSD",
            funding=-0.0005,  # < -extreme threshold → short_crowded
            oi=2000.0,
            index=3000.0,
            mark=3001.0,
            last=2999.0,
            bid=2998.0,
            ask=3000.0,
            vol24h=5000.0,
            volume_quote=15_000_000.0,
        ),
        _ticker(
            "PF_SOLUSD",
            funding=0.00015,  # between high and extreme → long_leaning
            oi=5000.0,
            index=150.0,
            mark=150.1,
            last=149.9,
            bid=149.8,
            ask=150.0,
            vol24h=10000.0,
            volume_quote=1_500_000.0,
        ),
        _ticker(
            "PF_DOGEUSD",
            funding=0.0,  # balanced
            oi=10000.0,
            index=0.1,
            mark=0.1,
            last=0.1,
            bid=0.099,
            ask=0.101,
            vol24h=100000.0,
            volume_quote=10_000.0,
        ),
        # A non-perp dated future — must be excluded.
        _ticker(
            "FI_XBTUSD_240927",
            funding=0.0001,
            oi=500.0,
            index=51000.0,
        ),
    ]


# ---------------------------------------------------------------------------
# 1-4: map_perp_symbol
# ---------------------------------------------------------------------------

def test_map_perp_symbol_xbt_to_btc():
    assert kfip.map_perp_symbol("PF_XBTUSD") == "BTC-USD"


def test_map_perp_symbol_eth():
    assert kfip.map_perp_symbol("PF_ETHUSD") == "ETH-USD"


def test_map_perp_symbol_sol():
    assert kfip.map_perp_symbol("PF_SOLUSD") == "SOL-USD"


def test_map_perp_symbol_unknown_returns_none():
    assert kfip.map_perp_symbol("PF_DOGEUSD") is None
    assert kfip.map_perp_symbol("FI_XBTUSD_240927") is None
    assert kfip.map_perp_symbol("") is None


# ---------------------------------------------------------------------------
# 5: parse_base_quote
# ---------------------------------------------------------------------------

def test_parse_base_quote_xbtusd_normalizes_to_btc():
    base, quote = kfip.parse_base_quote("PF_XBTUSD")
    assert base == "BTC"
    assert quote == "USD"


# ---------------------------------------------------------------------------
# 6-9: classify_market_bias
# ---------------------------------------------------------------------------

def test_classify_positive_extreme_long_crowded():
    assert kfip.classify_market_bias(kfip.FUNDING_EXTREME_THRESHOLD * 2) == "long_crowded"


def test_classify_negative_extreme_short_crowded():
    assert (
        kfip.classify_market_bias(-kfip.FUNDING_EXTREME_THRESHOLD * 2) == "short_crowded"
    )


def test_classify_elevated_positive_long_leaning():
    midpoint = (kfip.FUNDING_HIGH_THRESHOLD + kfip.FUNDING_EXTREME_THRESHOLD) / 2.0
    assert kfip.classify_market_bias(midpoint) == "long_leaning"


def test_classify_neutral_balanced_and_unknown():
    assert kfip.classify_market_bias(0.0) == "balanced"
    assert kfip.classify_market_bias(None) == "unknown"


# ---------------------------------------------------------------------------
# 10: build_symbol_record parses key fields
# ---------------------------------------------------------------------------

def test_build_symbol_record_parses_fields():
    t = _ticker(
        "PF_XBTUSD",
        funding=0.0005,
        oi=1000.0,
        index=50000.0,
        mark=50001.0,
        last=49999.0,
        bid=49998.0,
        ask=50000.0,
    )
    rec = kfip.build_symbol_record(t)
    assert rec["kraken_symbol"] == "PF_XBTUSD"
    assert rec["mapped_symbol"] == "BTC-USD"
    assert rec["base"] == "BTC"
    assert rec["quote"] == "USD"
    assert rec["funding_rate_8h"] == pytest.approx(0.0005)
    assert rec["funding_rate_annualized"] == pytest.approx(0.0005 * 3 * 365)
    assert rec["open_interest_contracts"] == pytest.approx(1000.0)
    assert rec["open_interest_usd"] == pytest.approx(1000.0 * 50000.0)
    assert rec["index_price"] == pytest.approx(50000.0)
    assert rec["mark_price"] == pytest.approx(50001.0)
    assert rec["last"] == pytest.approx(49999.0)
    assert rec["bid"] == pytest.approx(49998.0)
    assert rec["ask"] == pytest.approx(50000.0)
    assert rec["spread"] == pytest.approx(2.0)
    assert rec["market_bias"] == "long_crowded"
    assert rec["funding_extreme_long"] is True
    assert rec["data_available"] is True


# ---------------------------------------------------------------------------
# 11: build_payload only includes PF_*
# ---------------------------------------------------------------------------

def test_build_payload_only_includes_perp_symbols():
    payload = kfip.build_payload(_sample_tickers())
    assert payload["schema_version"] == kfip.SCHEMA_VERSION
    for sym in payload["symbols"].keys():
        assert sym.startswith("PF_")
    assert "FI_XBTUSD_240927" not in payload["symbols"]
    assert payload["summary"]["perps_total"] == 4
    assert payload["summary"]["symbols_published"] == 4


# ---------------------------------------------------------------------------
# 12: summary counts long_crowded / short_crowded
# ---------------------------------------------------------------------------

def test_build_payload_summary_counts_crowding():
    payload = kfip.build_payload(_sample_tickers())
    summary = payload["summary"]
    assert summary["long_crowded_count"] == 1
    assert summary["short_crowded_count"] == 1
    assert summary["mapped_symbols_count"] == 3
    assert payload["mapped_symbols"] == {
        "BTC-USD": "PF_XBTUSD",
        "ETH-USD": "PF_ETHUSD",
        "SOL-USD": "PF_SOLUSD",
    }


# ---------------------------------------------------------------------------
# 13: top_open_interest_usd sorted descending
# ---------------------------------------------------------------------------

def test_top_open_interest_usd_sorted_descending():
    payload = kfip.build_payload(_sample_tickers())
    top_oi = payload["summary"]["top_open_interest_usd"]
    assert top_oi, "expected non-empty top OI list"
    values = [pair[1] for pair in top_oi]
    assert values == sorted(values, reverse=True)
    # ETH ($6M OI) > XBT ($50M OI)? XBT = 1000 * 50000 = 50M; ETH = 2000 * 3000 = 6M
    assert top_oi[0][0] == "PF_XBTUSD"


# ---------------------------------------------------------------------------
# 14: top_volume_quote_24h sorted descending
# ---------------------------------------------------------------------------

def test_top_volume_quote_24h_sorted_descending():
    payload = kfip.build_payload(_sample_tickers())
    top_vol = payload["summary"]["top_volume_quote_24h"]
    assert top_vol
    values = [pair[1] for pair in top_vol]
    assert values == sorted(values, reverse=True)
    # ETH volumeQuote=15M > XBT=10M
    assert top_vol[0][0] == "PF_ETHUSD"


# ---------------------------------------------------------------------------
# 15: publish dry-run with fake fetcher returns valid schema, written=False
# ---------------------------------------------------------------------------

def test_publish_dry_run_with_fake_fetcher(tmp_path: Path):
    runtime_dir = tmp_path / "runtime"
    payload, written = kfip.publish(
        runtime_dir,
        dry_run=True,
        fetcher=_sample_tickers,
    )
    assert written is False
    assert payload["schema_version"] == kfip.SCHEMA_VERSION
    assert payload["status"] == "ok"
    assert payload["source"]["provider"] == "kraken_futures_public"
    assert payload["summary"]["symbols_published"] == 4
    assert not (runtime_dir / "kraken_futures_intel.json").exists()


# ---------------------------------------------------------------------------
# 16: fetch failure preserves existing file and returns error payload
# ---------------------------------------------------------------------------

def test_publish_fetch_failure_preserves_existing_file(tmp_path: Path):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    target = runtime_dir / "kraken_futures_intel.json"
    prior = {"schema_version": kfip.SCHEMA_VERSION, "status": "ok", "preserved": True}
    target.write_text(json.dumps(prior), encoding="utf-8")

    def boom() -> List[Mapping[str, Any]]:
        raise RuntimeError("kraken down")

    payload, written = kfip.publish(runtime_dir, fetcher=boom)
    assert written is False
    assert payload["status"] == "error"
    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk == prior


# ---------------------------------------------------------------------------
# 17: history is empty unless include_history=True
# ---------------------------------------------------------------------------

def test_history_empty_when_not_requested():
    payload = kfip.build_payload(_sample_tickers(), include_history=False)
    assert payload["history"] == {}


# ---------------------------------------------------------------------------
# 18: include_history attaches latest funding rate using fake history_fetcher
# ---------------------------------------------------------------------------

def test_include_history_uses_fake_history_fetcher():
    fake_history = {
        "PF_XBTUSD": {
            "rates": [
                {"timestamp": "2026-05-15T00:00:00Z", "fundingRate": 0.1, "relativeFundingRate": 1e-7},
                {"timestamp": "2026-05-15T08:00:00Z", "fundingRate": 0.2, "relativeFundingRate": 2e-7},
            ]
        },
        "PF_ETHUSD": {"rates": []},
    }

    def fetcher(sym: str) -> Mapping[str, Any]:
        return fake_history.get(sym, {})

    payload = kfip.build_payload(
        _sample_tickers(),
        include_history=True,
        history_symbols=["PF_XBTUSD", "PF_ETHUSD"],
        history_fetcher=fetcher,
    )
    history = payload["history"]
    assert "PF_XBTUSD" in history
    assert history["PF_XBTUSD"]["rates_count"] == 2
    assert history["PF_XBTUSD"]["latest"]["fundingRate"] == 0.2
    assert "PF_ETHUSD" in history
    assert history["PF_ETHUSD"]["rates_count"] == 0
    assert history["PF_ETHUSD"]["latest"] is None


# ---------------------------------------------------------------------------
# 19: deploy service/timer files exist and contain expected content
# ---------------------------------------------------------------------------

def test_deploy_files_exist_with_expected_content():
    repo_root = Path(__file__).resolve().parents[2]
    service_path = repo_root / "deploy" / "chad-kraken-futures-intel-refresh.service"
    timer_path = repo_root / "deploy" / "chad-kraken-futures-intel-refresh.timer"
    assert service_path.exists(), f"missing {service_path}"
    assert timer_path.exists(), f"missing {timer_path}"
    service_text = service_path.read_text(encoding="utf-8")
    timer_text = timer_path.read_text(encoding="utf-8")
    assert (
        "ExecStart=/home/ubuntu/chad_finale/venv/bin/python3 -m chad.market_data.kraken_futures_intel_publisher"
        in service_text
    )
    assert "OnUnitActiveSec=300" in timer_text


# ---------------------------------------------------------------------------
# 20: schema shape parses correctly from offline fake data
# ---------------------------------------------------------------------------

def test_payload_schema_shape_from_fake_data():
    payload = kfip.build_payload(_sample_tickers())
    # Required top-level keys
    for key in (
        "schema_version",
        "ts_utc",
        "ttl_seconds",
        "status",
        "source",
        "symbols",
        "mapped_symbols",
        "summary",
        "history",
    ):
        assert key in payload, f"missing {key}"
    # Required source keys
    for key in ("provider", "endpoint", "provider_status"):
        assert key in payload["source"]
    # Required summary keys
    for key in (
        "perps_total",
        "symbols_published",
        "mapped_symbols_count",
        "long_crowded_count",
        "short_crowded_count",
        "suspended_count",
        "post_only_count",
        "top_open_interest_usd",
        "top_volume_quote_24h",
    ):
        assert key in payload["summary"]
    # Required per-symbol record keys
    for sym, rec in payload["symbols"].items():
        for key in (
            "kraken_symbol",
            "mapped_symbol",
            "base",
            "quote",
            "funding_rate_8h",
            "funding_rate_annualized",
            "funding_rate_prediction",
            "open_interest_contracts",
            "open_interest_usd",
            "index_price",
            "mark_price",
            "last",
            "bid",
            "ask",
            "spread",
            "vol_24h",
            "volume_quote_24h",
            "change_24h",
            "post_only",
            "suspended",
            "market_bias",
            "funding_extreme_long",
            "funding_extreme_short",
            "funding_elevated_long",
            "data_available",
        ):
            assert key in rec, f"{sym} missing {key}"
    # JSON-serializable
    json.dumps(payload)
