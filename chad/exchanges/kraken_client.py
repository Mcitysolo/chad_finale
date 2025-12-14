from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import requests
from requests import Response, Session


class KrakenAPIError(Exception):
    """Raised when Kraken returns an error array with one or more errors."""


class KrakenConfigError(Exception):
    """Raised when the Kraken client is misconfigured (e.g., missing keys)."""


@dataclass(frozen=True)
class KrakenClientConfig:
    """
    Immutable configuration for KrakenClient.

    Attributes:
        api_key: Public API key obtained from Kraken Pro.
        api_secret: Base64-encoded API private key (secret) from Kraken Pro.
        base_url: Base URL for the Kraken REST API.
    """

    api_key: str
    api_secret: str
    base_url: str = "https://api.kraken.com"

    @classmethod
    def from_env(cls) -> "KrakenClientConfig":
        """
        Build config from environment variables.

        Required environment variables:
            KRAKEN_API_KEY
            KRAKEN_API_SECRET
        """
        key = os.getenv("KRAKEN_API_KEY")
        secret = os.getenv("KRAKEN_API_SECRET")

        missing = []
        if not key:
            missing.append("KRAKEN_API_KEY")
        if not secret:
            missing.append("KRAKEN_API_SECRET")

        if missing:
            raise KrakenConfigError(
                f"Missing Kraken API env vars: {', '.join(missing)}"
            )

        return cls(api_key=key, api_secret=secret)


class KrakenClient:
    """
    Low-level client for the Kraken REST API (Spot / Margin).

    This client handles:
        * HMAC-SHA512 signing for private endpoints.
        * Nonce generation.
        * Error handling and JSON parsing.

    It deliberately exposes only a small, safe surface area that CHAD needs:
        * get_balances()
        * query_open_orders()
        * add_order()
        * cancel_order()
    """

    def __init__(self, config: KrakenClientConfig, session: Optional[Session] = None) -> None:
        self._config = config
        self._session = session or requests.Session()

    # ------------------------------------------------------------------ #
    # Core request machinery                                             #
    # ------------------------------------------------------------------ #

    def _request(
        self,
        method: str,
        path: str,
        *,
        data: Optional[Mapping[str, Any]] = None,
        private: bool = False,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Perform an HTTP request to the Kraken API.

        Args:
            method: "GET" or "POST".
            path: e.g. "/0/public/Time" or "/0/private/Balance".
            data: Optional dict of POST fields.
            private: If True, send signed headers.
            timeout: Request timeout in seconds.

        Returns:
            Parsed JSON response as a dict.

        Raises:
            KrakenAPIError: If Kraken returns error messages.
            requests.RequestException: On network problems.
            ValueError: On JSON parsing issues.
        """
        url = f"{self._config.base_url}{path}"
        headers: Dict[str, str] = {}

        post_data: Dict[str, Any] = {}
        if data:
            post_data.update(data)

        if private:
            nonce = str(int(time.time() * 1000))
            post_data["nonce"] = nonce
            body = self._build_private_body(path, post_data)
            headers.update(
                {
                    "API-Key": self._config.api_key,
                    "API-Sign": body,
                    "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
                }
            )

        if method.upper() != "POST":
            # Kraken private endpoints are POST-only; public ones usually support GET.
            raise ValueError("KrakenClient currently supports POST-only endpoints.")

        resp: Response = self._session.post(
            url,
            data=post_data,
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()

        try:
            payload = resp.json()
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON from Kraken: {resp.text[:200]}") from exc

        # Kraken wraps errors in "error": [] and data in "result": {}
        errors = payload.get("error") or []
        if errors:
            # Kraken errors are strings; join them for now.
            raise KrakenAPIError("; ".join(errors))

        if "result" not in payload:
            raise ValueError(f"Unexpected Kraken payload (no 'result'): {payload}")

        result = payload["result"]
        if not isinstance(result, dict):
            # Some endpoints return list etc., but for our usage we expect dict.
            return {"result": result}

        return result

    def _build_private_body(self, path: str, data: Mapping[str, Any]) -> str:
        """
        Build the API-Sign value for private endpoints.

        According to Kraken docs:
            API-Sign = base64(hmac_sha512(uri_path + sha256(nonce + postdata), decoded_secret))

        Where:
            uri_path is e.g. "/0/private/Balance"
            postdata is URL-encoded query string of fields (including nonce)
        """
        postdata = "&".join(f"{key}={data[key]}" for key in data)  # stable order not required
        sha256 = hashlib.sha256()
        sha256.update((data["nonce"] + postdata).encode("utf-8"))
        sha256_digest = sha256.digest()

        message = path.encode("utf-8") + sha256_digest

        secret_decoded = base64.b64decode(self._config.api_secret)
        hmac_digest = hmac.new(secret_decoded, message, hashlib.sha512).digest()
        return base64.b64encode(hmac_digest).decode("ascii")

    # ------------------------------------------------------------------ #
    # Public wrapper methods                                             #
    # ------------------------------------------------------------------ #

    def get_balances(self) -> Dict[str, float]:
        """
        Fetch account balances for all assets.

        Returns:
            Mapping of asset symbol (e.g., "XXBT", "ZUSD") to float balance.
        """
        result = self._request("POST", "/0/private/Balance", private=True)
        balances: Dict[str, float] = {}
        for asset, amount_str in result.items():
            try:
                balances[asset] = float(amount_str)
            except (TypeError, ValueError):
                continue
        return balances

    def query_open_orders(self) -> Dict[str, Any]:
        """
        Fetch all open orders.

        Returns:
            Kraken 'open' orders structure.
        """
        result = self._request("POST", "/0/private/OpenOrders", private=True)
        # Kraken returns {'open': {...}, 'count': N}
        return result

    def add_order(
        self,
        *,
        pair: str,
        side: str,
        ordertype: str,
        volume: float,
        price: Optional[float] = None,
        timeinforce: Optional[str] = None,
        validate_only: bool = False,
        extra_fields: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Place a new order.

        Args:
            pair: Trading pair (e.g., "XBTUSD" / "XBT/USDT" style depending on Kraken notation).
            side: "buy" or "sell".
            ordertype: e.g. "market" or "limit".
            volume: Amount of base currency to trade.
            price: Limit price (required for limit orders).
            timeinforce: Optional time-in-force (e.g. "GTC", "IOC"), depending on Kraken support.
            validate_only: If True, Kraken validates but does not execute.
            extra_fields: Optional additional fields (e.g., "leverage": "2:1").

        Returns:
            Kraken AddOrder result dict.
        """
        data: Dict[str, Any] = {
            "pair": pair,
            "type": side,
            "ordertype": ordertype,
            "volume": str(volume),
        }
        if price is not None:
            data["price"] = str(price)
        if timeinforce:
            data["timeinforce"] = timeinforce
        if validate_only:
            data["validate"] = True
        if extra_fields:
            data.update(extra_fields)

        result = self._request("POST", "/0/private/AddOrder", data=data, private=True)
        return result

    def cancel_order(self, txid: str) -> Dict[str, Any]:
        """
        Cancel an existing order by transaction ID.

        Returns:
            Kraken CancelOrder result dict.
        """
        data = {"txid": txid}
        result = self._request("POST", "/0/private/CancelOrder", data=data, private=True)
        return result


# --------------------------------------------------------------------------- #
# CLI Self-Test                                                               #
# --------------------------------------------------------------------------- #


def _cli_self_test(client: KrakenClient) -> int:
    """
    Run a minimal self-test:
        * Fetch balances
        * Fetch open orders

    This is safe to run; no trading occurs.
    """
    try:
        balances = client.get_balances()
        open_orders = client.query_open_orders()
    except (KrakenAPIError, KrakenConfigError, requests.RequestException, ValueError) as exc:
        print(f"[SELF-TEST] ERROR: {exc}")
        return 1

    print("[SELF-TEST] Balances:")
    if not balances:
        print("  (No non-zero balances returned)")
    else:
        for asset, amount in sorted(balances.items()):
            print(f"  {asset}: {amount}")

    print("\n[SELF-TEST] Open orders:")
    if not open_orders.get("open"):
        print("  (No open orders)")
    else:
        for txid, order in open_orders["open"].items():
            descr = order.get("descr", {})
            pair = descr.get("pair", "")
            side = descr.get("type", "")
            otype = descr.get("ordertype", "")
            vol = order.get("vol", "")
            print(f"  {txid}: {side} {vol} {pair} ({otype})")

    return 0


def _build_client_from_env() -> KrakenClient:
    cfg = KrakenClientConfig.from_env()
    return KrakenClient(cfg)


def main(argv: Optional[Tuple[str, ...]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "KrakenClient utility.\n"
            "By default, runs a self-test (balances + open orders) using "
            "KRAKEN_API_KEY and KRAKEN_API_SECRET from the environment."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("self-test", help="Run a read-only self test.")

    balance_parser = subparsers.add_parser("balances", help="Print account balances.")
    balance_parser.set_defaults(command="balances")

    args = parser.parse_args(argv)

    client = _build_client_from_env()

    if args.command in (None, "self-test"):
        return _cli_self_test(client)

    if args.command == "balances":
        try:
            balances = client.get_balances()
        except Exception as exc:  # noqa: BLE001
            print(f"Error fetching balances: {exc}")
            return 1
        print("Balances:")
        if not balances:
            print("  (No non-zero balances)")
        else:
            for asset, amount in sorted(balances.items()):
                print(f"  {asset}: {amount}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
