from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import logging
import os
import ssl
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from http.client import HTTPSConnection, HTTPResponse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger("chad.portfolio.coinbase")


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoinbaseExchangeCredentials:
    """
    Coinbase Exchange API key triple.

    Values are read from environment variables:

      * COINBASE_EXCHANGE_API_KEY
      * COINBASE_EXCHANGE_API_SECRET  (base64-encoded, as provided by Coinbase)
      * COINBASE_EXCHANGE_API_PASSPHRASE
    """

    api_key: str
    api_secret_b64: str
    api_passphrase: str

    @classmethod
    def from_env(cls) -> "CoinbaseExchangeCredentials":
        key = os.getenv("COINBASE_EXCHANGE_API_KEY")
        secret = os.getenv("COINBASE_EXCHANGE_API_SECRET")
        passphrase = os.getenv("COINBASE_EXCHANGE_API_PASSPHRASE")

        missing: List[str] = []
        if not key:
            missing.append("COINBASE_EXCHANGE_API_KEY")
        if not secret:
            missing.append("COINBASE_EXCHANGE_API_SECRET")
        if not passphrase:
            missing.append("COINBASE_EXCHANGE_API_PASSPHRASE")

        if missing:
            raise RuntimeError(
                f"Missing required Coinbase Exchange API env vars: {', '.join(missing)}"
            )

        return cls(api_key=key, api_secret_b64=secret, api_passphrase=passphrase)

    def signing_key(self) -> bytes:
        """
        Return the HMAC signing key: base64-decoded secret bytes, as required by
        Coinbase Exchange REST authentication.
        """
        try:
            return base64.b64decode(self.api_secret_b64)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                "Invalid COINBASE_EXCHANGE_API_SECRET (base64 decode failed)"
            ) from exc


@dataclass(frozen=True)
class CoinbaseClientConfig:
    """
    Low-level HTTP config for Coinbase Exchange + Data API.

    Uses environment variable COINBASE_EXCHANGE_USE_SANDBOX to switch hosts:
      * unset or "0" -> production
      * "1" -> sandbox
    """

    exchange_host: str
    data_host: str
    timeout_seconds: float = 10.0

    @classmethod
    def from_env(cls) -> "CoinbaseClientConfig":
        use_sandbox = os.getenv("COINBASE_EXCHANGE_USE_SANDBOX", "0") == "1"
        if use_sandbox:
            exchange_host = "api-public.sandbox.exchange.coinbase.com"
        else:
            exchange_host = "api.exchange.coinbase.com"

        # Data API (prices/exchange-rates) is on api.coinbase.com
        data_host = "api.coinbase.com"

        return cls(exchange_host=exchange_host, data_host=data_host)


@dataclass(frozen=True)
class CoinbasePortfolioSnapshot:
    """
    Computed Coinbase equity snapshot.

    total_usd:    Sum of all accounts valued in USD.
    per_currency: Map from currency code to its USD valuation.
    """

    total_usd: Decimal
    per_currency: Dict[str, Decimal]


# ---------------------------------------------------------------------------
# HTTP utilities
# ---------------------------------------------------------------------------


class HTTPSJsonClient:
    """
    Minimal HTTPS JSON client with strict TLS verification and explicit timeouts.

    This avoids taking dependencies on external libraries while remaining robust.
    """

    def __init__(self, host: str, timeout_seconds: float) -> None:
        self._host = host
        self._timeout = timeout_seconds
        self._context = ssl.create_default_context()

    def request(
        self,
        method: str,
        path: str,
        *,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[str] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Perform an HTTPS request and parse JSON response.

        Raises RuntimeError on non-2xx status codes or JSON parse failures.
        """
        conn = HTTPSConnection(
            host=self._host,
            timeout=self._timeout,
            context=self._context,
        )
        try:
            req_headers: Dict[str, str] = {
                "User-Agent": "CHAD-PortfolioCollector/1.0",
                "Accept": "application/json",
            }
            if headers:
                req_headers.update(headers)
            if body is not None:
                req_headers.setdefault("Content-Type", "application/json")

            conn.request(method=method, url=path, body=body, headers=req_headers)
            resp: HTTPResponse = conn.getresponse()
            status = resp.status
            raw = resp.read()

            text = raw.decode("utf-8") if raw else ""
            if status < 200 or status >= 300:
                raise RuntimeError(
                    f"HTTPS {method} {self._host}{path} failed: "
                    f"status={status} body={text[:500]}"
                )

            if not text:
                return status, {}

            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Failed to parse JSON from {self._host}{path}: {exc}"
                ) from exc

            return status, data
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Coinbase Exchange client (accounts)
# ---------------------------------------------------------------------------


class CoinbaseExchangeClient:
    """
    Client for Coinbase Exchange REST API (accounts endpoint only).

    Uses API key authentication as documented here:
    https://docs.cdp.coinbase.com/exchange/rest-api/authentication
    """

    def __init__(
        self,
        credentials: CoinbaseExchangeCredentials,
        config: CoinbaseClientConfig,
    ) -> None:
        self._creds = credentials
        self._cfg = config
        self._http = HTTPSJsonClient(
            host=config.exchange_host,
            timeout_seconds=config.timeout_seconds,
        )

    def _signed_headers(self, method: str, request_path: str, body: str = "") -> Dict[str, str]:
        timestamp = str(time.time())
        message = f"{timestamp}{method.upper()}{request_path}{body}"
        key = self._creds.signing_key()

        signature = hmac.new(
            key,
            msg=message.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        signature_b64 = base64.b64encode(signature).decode("ascii")

        return {
            "CB-ACCESS-KEY": self._creds.api_key,
            "CB-ACCESS-PASSPHRASE": self._creds.api_passphrase,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "CB-ACCESS-SIGN": signature_b64,
        }

    def list_accounts(self) -> List[Dict[str, Any]]:
        """
        List all trading accounts for the API key's profile.

        Endpoint: GET /accounts

        Each account object includes:
          * id
          * currency
          * balance
          * available
          * hold
          * profile_id
          * trading_enabled
        """
        path = "/accounts"
        headers = self._signed_headers("GET", path)
        _status, data = self._http.request("GET", path, headers=headers, body=None)

        # Legacy API can return a list or an object with 'data'; support both.
        if isinstance(data, list):
            accounts_raw = data
        elif isinstance(data, dict):
            if "accounts" in data and isinstance(data["accounts"], list):
                accounts_raw = data["accounts"]
            elif "data" in data and isinstance(data["data"], list):
                accounts_raw = data["data"]
            else:
                raise RuntimeError(
                    f"Unexpected /accounts payload structure: keys={list(data.keys())}"
                )
        else:
            raise RuntimeError(
                f"Unexpected /accounts payload type: {type(data).__name__}"
            )

        accounts: List[Dict[str, Any]] = []
        for raw in accounts_raw:
            if not isinstance(raw, dict):
                continue
            currency = str(raw.get("currency", "")).upper()
            balance_str = str(raw.get("balance", "0"))
            try:
                balance = Decimal(balance_str)
            except InvalidOperation:
                continue

            if balance <= 0:
                # Ignore empty accounts
                continue

            accounts.append(
                {
                    "id": raw.get("id"),
                    "currency": currency,
                    "balance": balance,
                }
            )

        logger.info(
            "coinbase.accounts_loaded",
            extra={
                "host": self._cfg.exchange_host,
                "account_count": len(accounts),
            },
        )
        return accounts


# ---------------------------------------------------------------------------
# Coinbase Data API client (spot prices)
# ---------------------------------------------------------------------------


class CoinbaseDataClient:
    """
    Client for Coinbase Data API (spot prices).

    Docs: https://docs.cdp.coinbase.com/coinbase-business/track-apis/prices
    """

    def __init__(self, config: CoinbaseClientConfig) -> None:
        self._cfg = config
        self._http = HTTPSJsonClient(
            host=config.data_host,
            timeout_seconds=config.timeout_seconds,
        )

    def get_spot_price_usd(self, currency: str) -> Optional[Decimal]:
        """
        Return spot price in USD for the given asset symbol.

        Uses:
          GET /v2/prices/:currency_pair/spot

        Example: /v2/prices/BTC-USD/spot
        """
        currency = currency.upper()
        if currency in ("USD", "USDC"):
            return Decimal("1")

        pair = f"{currency}-USD"
        path = f"/v2/prices/{pair}/spot"
        try:
            _status, data = self._http.request("GET", path, headers=None, body=None)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "coinbase.spot_price_failed",
                extra={"currency": currency, "error": str(exc)},
            )
            return None

        try:
            amount_str = str(data["data"]["amount"])
            amount = Decimal(amount_str)
            if amount <= 0:
                return None
            return amount
        except (KeyError, InvalidOperation, TypeError) as exc:
            logger.warning(
                "coinbase.spot_price_invalid",
                extra={"currency": currency, "error": str(exc)},
            )
            return None


# ---------------------------------------------------------------------------
# Portfolio calculation
# ---------------------------------------------------------------------------


class CoinbasePortfolioCalculator:
    """
    Combine Coinbase Exchange account balances with Coinbase Data API prices to
    compute a USD equity snapshot.

    All balances are valued using spot prices in USD.
    """

    def __init__(
        self,
        exchange_client: CoinbaseExchangeClient,
        data_client: CoinbaseDataClient,
    ) -> None:
        self._exchange = exchange_client
        self._data = data_client

    def compute_portfolio(self) -> CoinbasePortfolioSnapshot:
        accounts = self._exchange.list_accounts()
        per_currency: Dict[str, Decimal] = {}
        total_usd = Decimal("0")

        # Cache spot prices by currency to avoid repeated HTTP calls.
        price_cache: Dict[str, Optional[Decimal]] = {}

        for acc in accounts:
            currency = str(acc["currency"]).upper()
            balance: Decimal = acc["balance"]

            if currency == "USD":
                usd_value = balance
            else:
                if currency not in price_cache:
                    price_cache[currency] = self._data.get_spot_price_usd(currency)
                spot = price_cache[currency]
                if spot is None:
                    logger.warning(
                        "coinbase.missing_spot_price",
                        extra={"currency": currency},
                    )
                    continue
                usd_value = balance * spot

            per_currency[currency] = per_currency.get(currency, Decimal("0")) + usd_value
            total_usd += usd_value

        logger.info(
            "coinbase.portfolio_computed",
            extra={
                "total_usd": float(total_usd),
                "currency_count": len(per_currency),
            },
        )
        return CoinbasePortfolioSnapshot(total_usd=total_usd, per_currency=per_currency)


# ---------------------------------------------------------------------------
# Snapshot writer (merge into runtime/portfolio_snapshot.json)
# ---------------------------------------------------------------------------


class PortfolioSnapshotWriter:
    """
    Merge Coinbase equity into runtime/portfolio_snapshot.json.

    Behaviour:
      * If file exists: preserve existing ibkr-equity (supports both ibkr_equity
        and ibkr_equity_usd), update coinbase_equity (canonical).
      * If missing: create new snapshot with ibkr_equity = 0.0, coinbase_equity
        from Coinbase.

    This keeps IBKR and Coinbase collection concerns separate and composable.
    """

    def __init__(self, snapshot_path: Path) -> None:
        self._path = snapshot_path

    @property
    def path(self) -> Path:
        return self._path

    def _load_existing(self) -> Dict[str, Any]:
        if not self._path.is_file():
            return {}
        try:
            text = self._path.read_text(encoding="utf-8")
            if not text.strip():
                return {}
            return json.loads(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "portfolio_snapshot.read_failed",
                extra={"path": str(self._path), "error": str(exc)},
            )
            return {}

    def write_with_coinbase_equity(self, coinbase_equity_usd: Decimal) -> None:
        existing = self._load_existing()

        # Prefer canonical ibkr_equity; fall back to ibkr_equity_usd; else zero.
        if "ibkr_equity" in existing:
            ibkr_raw = existing.get("ibkr_equity", 0.0)
        else:
            ibkr_raw = existing.get("ibkr_equity_usd", 0.0)

        try:
            ibkr_equity_val = float(ibkr_raw)
        except (TypeError, ValueError):
            ibkr_equity_val = 0.0

        new_payload: Dict[str, Any] = {
            "ibkr_equity": ibkr_equity_val,
            "coinbase_equity": float(coinbase_equity_usd),
        }

        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")

        tmp_path.write_text(
            json.dumps(new_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(self._path)

        logger.info(
            "portfolio_snapshot.updated",
            extra={
                "path": str(self._path),
                "ibkr_equity": ibkr_equity_val,
                "coinbase_equity": float(coinbase_equity_usd),
            },
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def default_snapshot_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "runtime" / "portfolio_snapshot.json"


def _configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.setLevel(level)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Coinbase Exchange balances, value them in USD via Coinbase "
            "Data API, and merge into runtime/portfolio_snapshot.json as "
            "coinbase_equity."
        )
    )
    parser.add_argument(
        "--snapshot-path",
        type=str,
        default=None,
        help="Optional override for portfolio_snapshot.json path.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--coinbase-equity",
        type=str,
        default=None,
        help=(
            "Optional override: total Coinbase equity in USD. "
            "If provided, skips API calls and uses this value directly."
        ),
    )
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)

    snapshot_path = (
        Path(args.snapshot_path).expanduser().resolve()
        if args.snapshot_path
        else default_snapshot_path()
    )
    writer = PortfolioSnapshotWriter(snapshot_path)

    # Dummy mode: allow manual override for testing (matches your previous usage).
    if args.coinbase_equity is not None:
        try:
            equity_val = Decimal(args.coinbase_equity)
        except InvalidOperation as exc:
            logger.error(
                "coinbase.override_invalid",
                extra={"value": args.coinbase_equity, "error": str(exc)},
            )
            return 1

        logger.info(
            "coinbase.override_equity_used",
            extra={"coinbase_equity": float(equity_val)},
        )
        writer.write_with_coinbase_equity(coinbase_equity_usd=equity_val)
        return 0

    # Live mode: require credentials and call APIs.
    try:
        creds = CoinbaseExchangeCredentials.from_env()
    except RuntimeError as exc:
        logger.error("coinbase.credentials_missing", extra={"error": str(exc)})
        return 1

    cfg = CoinbaseClientConfig.from_env()
    exchange_client = CoinbaseExchangeClient(credentials=creds, config=cfg)
    data_client = CoinbaseDataClient(config=cfg)
    calculator = CoinbasePortfolioCalculator(exchange_client, data_client)

    try:
        portfolio = calculator.compute_portfolio()
    except Exception as exc:  # noqa: BLE001
        logger.error("coinbase.portfolio_failed", extra={"error": str(exc)})
        return 1

    writer.write_with_coinbase_equity(coinbase_equity_usd=portfolio.total_usd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
