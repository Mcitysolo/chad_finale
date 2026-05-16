"""Phase C Item 1B — Kraken Futures client scaffold tests.

No live private Kraken Futures network calls. No credentials required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chad.exchanges.kraken_futures_client import (
    KrakenFuturesClient,
    KrakenFuturesCredentials,
    KrakenFuturesOrderRequest,
    KrakenFuturesOrderResult,
    load_credentials_from_env,
)


# ---------------------------------------------------------------------------
# 1. missing credentials -> has_credentials False
# ---------------------------------------------------------------------------

def test_has_credentials_false_when_none() -> None:
    client = KrakenFuturesClient(credentials=None, dry_run=True)
    assert client.has_credentials() is False


# ---------------------------------------------------------------------------
# 2. key loader ignores missing futures keys
# ---------------------------------------------------------------------------

def test_load_credentials_returns_none_when_missing(tmp_path: Path) -> None:
    empty_env: dict[str, str] = {}
    missing_fallback = tmp_path / "no_such_kraken.env"
    creds = load_credentials_from_env(env=empty_env, fallback_path=missing_fallback)
    assert creds is None


def test_load_credentials_ignores_spot_only_env(tmp_path: Path) -> None:
    fallback = tmp_path / "kraken.env"
    fallback.write_text(
        "# spot-only file\nKRAKEN_API_KEY=spotkey\nKRAKEN_API_SECRET=spotsecret\n",
        encoding="utf-8",
    )
    creds = load_credentials_from_env(env={}, fallback_path=fallback)
    assert creds is None


# ---------------------------------------------------------------------------
# 3-7. validate_order rejection cases
# ---------------------------------------------------------------------------

def test_validate_order_rejects_non_pf_symbol() -> None:
    order = KrakenFuturesOrderRequest(
        symbol="BTC-USD", side="buy", order_type="mkt", size=0.01
    )
    err = KrakenFuturesClient.validate_order(order)
    assert err is not None and "invalid_symbol" in err


def test_validate_order_rejects_invalid_side() -> None:
    order = KrakenFuturesOrderRequest(
        symbol="PF_XBTUSD", side="long", order_type="mkt", size=0.01
    )
    err = KrakenFuturesClient.validate_order(order)
    assert err is not None and "invalid_side" in err


def test_validate_order_rejects_invalid_order_type() -> None:
    order = KrakenFuturesOrderRequest(
        symbol="PF_XBTUSD", side="buy", order_type="stop", size=0.01
    )
    err = KrakenFuturesClient.validate_order(order)
    assert err is not None and "invalid_order_type" in err


def test_validate_order_rejects_non_positive_size() -> None:
    order = KrakenFuturesOrderRequest(
        symbol="PF_XBTUSD", side="buy", order_type="mkt", size=0.0
    )
    err = KrakenFuturesClient.validate_order(order)
    assert err is not None and "invalid_size" in err


def test_validate_order_requires_limit_price_for_lmt() -> None:
    order = KrakenFuturesOrderRequest(
        symbol="PF_XBTUSD", side="buy", order_type="lmt", size=0.01, limit_price=None
    )
    err = KrakenFuturesClient.validate_order(order)
    assert err == "limit_price_required_for_lmt"


def test_validate_order_accepts_valid_lmt() -> None:
    order = KrakenFuturesOrderRequest(
        symbol="PF_ETHUSD",
        side="sell",
        order_type="lmt",
        size=0.5,
        limit_price=3000.0,
    )
    assert KrakenFuturesClient.validate_order(order) is None


# ---------------------------------------------------------------------------
# 8. build_order_payload includes core fields
# ---------------------------------------------------------------------------

def test_build_order_payload_includes_core_fields() -> None:
    order = KrakenFuturesOrderRequest(
        symbol="PF_SOLUSD",
        side="buy",
        order_type="lmt",
        size=2.0,
        limit_price=150.0,
        reduce_only=True,
        client_order_id="abc-123",
    )
    payload = KrakenFuturesClient.build_order_payload(order)
    assert payload["symbol"] == "PF_SOLUSD"
    assert payload["side"] == "buy"
    assert payload["orderType"] == "lmt"
    assert payload["size"] == 2.0
    assert payload["limitPrice"] == 150.0
    assert payload["reduceOnly"] is True
    assert payload["cliOrdId"] == "abc-123"


# ---------------------------------------------------------------------------
# 9. submit_order dry-run does not call opener
# ---------------------------------------------------------------------------

class _OpenerSpy:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        return '{"result": "success"}'


def test_submit_order_dry_run_no_network() -> None:
    spy = _OpenerSpy()
    client = KrakenFuturesClient(credentials=None, dry_run=True, opener=spy)
    order = KrakenFuturesOrderRequest(
        symbol="PF_XBTUSD", side="buy", order_type="mkt", size=0.01
    )
    result = client.submit_order(order)
    assert isinstance(result, KrakenFuturesOrderResult)
    assert result.dry_run is True
    assert result.ok is True
    assert result.status == "dry_run_accepted"
    assert spy.calls == []


# ---------------------------------------------------------------------------
# 10. submit_order live without credentials fails closed
# ---------------------------------------------------------------------------

def test_submit_order_live_without_credentials_fails_closed() -> None:
    spy = _OpenerSpy()
    client = KrakenFuturesClient(credentials=None, dry_run=False, opener=spy)
    order = KrakenFuturesOrderRequest(
        symbol="PF_XBTUSD", side="sell", order_type="mkt", size=0.01
    )
    result = client.submit_order(order)
    assert result.ok is False
    assert result.dry_run is False
    assert result.status == "missing_credentials"
    assert result.error == "kraken_futures_credentials_missing"
    assert spy.calls == []


def test_submit_order_validation_failure_returns_error() -> None:
    client = KrakenFuturesClient(credentials=None, dry_run=True)
    bad_order = KrakenFuturesOrderRequest(
        symbol="BTC-USD", side="buy", order_type="mkt", size=0.01
    )
    result = client.submit_order(bad_order)
    assert result.ok is False
    assert result.status == "validation_failed"
    assert result.error is not None and "invalid_symbol" in result.error


# ---------------------------------------------------------------------------
# 11. sign_request returns expected auth header keys
# ---------------------------------------------------------------------------

def test_sign_request_returns_auth_headers() -> None:
    import base64

    fake_secret = base64.b64encode(b"\x01\x02\x03\x04" * 16).decode("ascii")
    creds = KrakenFuturesCredentials(api_key="testkey", api_secret=fake_secret)
    client = KrakenFuturesClient(credentials=creds, dry_run=True)
    headers = client.sign_request(
        path="/sendorder",
        post_data="symbol=PF_XBTUSD&side=buy&orderType=mkt&size=0.01",
        nonce="1700000000000",
    )
    assert "APIKey" in headers
    assert "Authent" in headers
    assert "Nonce" in headers
    assert headers["APIKey"] == "testkey"
    assert headers["Nonce"] == "1700000000000"
    assert isinstance(headers["Authent"], str) and len(headers["Authent"]) > 0


def test_sign_request_without_credentials_raises() -> None:
    client = KrakenFuturesClient(credentials=None, dry_run=True)
    with pytest.raises(RuntimeError):
        client.sign_request(path="/sendorder", post_data="x=1", nonce="1")


# ---------------------------------------------------------------------------
# 12. no live network calls in tests
# ---------------------------------------------------------------------------

def test_no_live_network_in_default_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing the client and validating an order must not touch the network."""
    import urllib.request

    def _explode(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("urllib.request.urlopen called during test")

    monkeypatch.setattr(urllib.request, "urlopen", _explode)

    client = KrakenFuturesClient(credentials=None, dry_run=True)
    order = KrakenFuturesOrderRequest(
        symbol="PF_ETHUSD", side="buy", order_type="lmt", size=0.5, limit_price=3000.0
    )
    payload = KrakenFuturesClient.build_order_payload(order)
    assert payload["symbol"] == "PF_ETHUSD"
    result = client.submit_order(order)
    assert result.dry_run is True
