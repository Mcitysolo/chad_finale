#!/usr/bin/env python3
"""
CHAD — Kraken API Client (Production-grade)

This client is used by:
- Kraken portfolio collector (balances snapshot)
- Kraken trade router / executor (AddOrder validate-only + live)

Upgrades included (for weekend productivity)
-------------------------------------------
This file adds private endpoints required to compute real fills / PnL later:
- QueryOrders
- ClosedOrders
- TradesHistory

It also supports public GET endpoints (e.g., AssetPairs) safely.

Safety / Correctness
--------------------
- All private endpoints use Kraken's API-Key / API-Sign HMAC scheme.
- Retries with bounded exponential backoff for transient failures + rate-limit errors.
- Strict error handling: any Kraken "error" list triggers KrakenAPIError.
- POST for private endpoints; GET for public endpoints; both supported.

Environment variables
---------------------
Required:
- KRAKEN_API_KEY
- KRAKEN_API_SECRET

Optional:
- KRAKEN_API_URL (default: https://api.kraken.com)
- KRAKEN_TIMEOUT_SECONDS (default: 15)
- KRAKEN_RETRY_MAX (default: 4)
- KRAKEN_RETRY_BASE_SECONDS (default: 0.6)

"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


# ----------------------------
# Exceptions
# ----------------------------

class KrakenConfigError(RuntimeError):
    pass


class KrakenAPIError(RuntimeError):
    pass


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class KrakenClientConfig:
    api_key: str
    api_secret: str  # base64
    api_url: str = "https://api.kraken.com"
    timeout_seconds: float = 15.0
    retry_max: int = 4
    retry_base_seconds: float = 0.6

    @staticmethod
    def from_env() -> "KrakenClientConfig":
        missing = []
        api_key = (os.environ.get("KRAKEN_API_KEY") or "").strip()
        api_secret = (os.environ.get("KRAKEN_API_SECRET") or "").strip()
        if not api_key:
            missing.append("KRAKEN_API_KEY")
        if not api_secret:
            missing.append("KRAKEN_API_SECRET")
        if missing:
            raise KrakenConfigError(f"Missing Kraken API env vars: {', '.join(missing)}")

        api_url = (os.environ.get("KRAKEN_API_URL") or "https://api.kraken.com").strip()

        def _f(name: str, default: float) -> float:
            v = (os.environ.get(name) or "").strip()
            if not v:
                return default
            try:
                return float(v)
            except Exception:
                return default

        def _i(name: str, default: int) -> int:
            v = (os.environ.get(name) or "").strip()
            if not v:
                return default
            try:
                return int(v)
            except Exception:
                return default

        return KrakenClientConfig(
            api_key=api_key,
            api_secret=api_secret,
            api_url=api_url,
            timeout_seconds=_f("KRAKEN_TIMEOUT_SECONDS", 15.0),
            retry_max=max(0, _i("KRAKEN_RETRY_MAX", 4)),
            retry_base_seconds=max(0.05, _f("KRAKEN_RETRY_BASE_SECONDS", 0.6)),
        )


# ----------------------------
# Client
# ----------------------------

class KrakenClient:
    """
    Minimal Kraken REST client with:
      - public GET endpoints
      - private POST endpoints with signing

    Returns:
      dict with keys: error (list), result (dict)
    Raises:
      KrakenAPIError on Kraken-reported errors or non-JSON failures.
    """

    def __init__(self, cfg: KrakenClientConfig) -> None:
        self._cfg = cfg
        # Pre-decode secret once
        try:
            self._secret_bytes = base64.b64decode(cfg.api_secret)
        except Exception as exc:
            raise KrakenConfigError(f"KRAKEN_API_SECRET is not valid base64: {exc}") from exc

    # --------
    # Signing
    # --------

    def _nonce(self) -> str:
        # Kraken requires increasing nonce. Use ms timestamp.
        return str(int(time.time() * 1000))

    def _sign(self, url_path: str, data: Dict[str, Any]) -> str:
        """
        Kraken API-Sign:
        - sha256(nonce + postdata)
        - HMAC-SHA512(secret, url_path + sha256_digest)
        - base64 encode
        """
        postdata = urllib.parse.urlencode({k: str(v) for k, v in data.items()})
        nonce = str(data.get("nonce", ""))
        sha256_digest = hashlib.sha256((nonce + postdata).encode("utf-8")).digest()
        msg = url_path.encode("utf-8") + sha256_digest
        mac = hmac.new(self._secret_bytes, msg, hashlib.sha512).digest()
        return base64.b64encode(mac).decode("utf-8")

    # -------------
    # HTTP execution
    # -------------

    def _do_http(
        self,
        method: str,
        url_path: str,
        *,
        data: Optional[Dict[str, Any]] = None,
        private: bool = False,
    ) -> Dict[str, Any]:
        base = self._cfg.api_url.rstrip("/")
        url = base + url_path

        headers: Dict[str, str] = {
            "User-Agent": "CHAD-KrakenClient/1.0",
            "Accept": "application/json",
        }

        body: Optional[bytes] = None
        if method.upper() == "GET":
            if data:
                url += "?" + urllib.parse.urlencode({k: str(v) for k, v in data.items()})
            req = urllib.request.Request(url, method="GET", headers=headers)
        else:
            # POST
            post_data = dict(data or {})
            if private:
                post_data["nonce"] = self._nonce()
                headers["API-Key"] = self._cfg.api_key
                headers["API-Sign"] = self._sign(url_path, post_data)
            encoded = urllib.parse.urlencode({k: str(v) for k, v in post_data.items()}).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            body = encoded
            req = urllib.request.Request(url, data=body, method="POST", headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=self._cfg.timeout_seconds) as r:
                raw = r.read().decode("utf-8", errors="replace")
        except Exception as exc:
            raise KrakenAPIError(f"HTTP {method} {url_path} failed: {type(exc).__name__}: {exc}") from exc

        try:
            j = json.loads(raw)
        except Exception as exc:
            raise KrakenAPIError(f"Non-JSON response from Kraken ({url_path}): {raw[:200]}") from exc

        errs = j.get("error") or []
        if errs:
            raise KrakenAPIError("; ".join([str(e) for e in errs]))

        if "result" not in j:
            raise KrakenAPIError(f"Kraken response missing 'result' for {url_path}: {j}")

        return j

    def _request(
        self,
        method: str,
        url_path: str,
        *,
        data: Optional[Dict[str, Any]] = None,
        private: bool = False,
    ) -> Dict[str, Any]:
        """
        Retrying wrapper around _do_http.
        Retries on:
          - transient HTTP/transport failures (raised KrakenAPIError)
          - rate limiting signals in message
        """
        max_attempts = max(1, int(self._cfg.retry_max) + 1)
        base_sleep = float(self._cfg.retry_base_seconds)

        last_exc: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                return self._do_http(method, url_path, data=data, private=private)
            except KrakenAPIError as exc:
                last_exc = exc
                msg = str(exc).lower()
                # Retry only on transient-ish signals
                transient = any(
                    s in msg
                    for s in [
                        "timed out",
                        "timeout",
                        "temporarily",
                        "connection",
                        "bad gateway",
                        "service unavailable",
                        "rate limit",
                        "eapi:rate limit exceeded",
                        "try again",
                    ]
                )
                if attempt >= max_attempts or not transient:
                    raise
                sleep_s = base_sleep * (2 ** (attempt - 1))
                # bounded
                sleep_s = min(sleep_s, 6.0)
                time.sleep(sleep_s)

        # should never hit
        if last_exc:
            raise last_exc
        raise KrakenAPIError("unknown kraken request failure")

    # ----------------------------
    # Public endpoints (GET)
    # ----------------------------

    def asset_pairs(self, *, pair: Optional[str] = None) -> Dict[str, Any]:
        """
        /0/public/AssetPairs
        """
        data = {"pair": pair} if pair else None
        j = self._request("GET", "/0/public/AssetPairs", data=data, private=False)
        return j.get("result") or {}

    def ticker(self, *, pair: str) -> Dict[str, Any]:
        j = self._request("GET", "/0/public/Ticker", data={"pair": pair}, private=False)
        return j.get("result") or {}

    # ----------------------------
    # Private endpoints (POST)
    # ----------------------------

    def balance(self) -> Dict[str, Any]:
        """
        /0/private/Balance
        """
        j = self._request("POST", "/0/private/Balance", data={}, private=True)
        return j.get("result") or {}


    def get_balances(self) -> Dict[str, Any]:
        """Backward-compatible alias for older collectors (calls balance())."""
        return self.balance()
    def open_orders(self) -> Dict[str, Any]:
        """
        /0/private/OpenOrders
        """
        j = self._request("POST", "/0/private/OpenOrders", data={}, private=True)
        return j.get("result") or {}

    def add_order(
        self,
        *,
        pair: str,
        side: str,
        ordertype: str,
        volume: float,
        price: Optional[float] = None,
        validate_only: bool = True,
    ) -> Dict[str, Any]:
        """
        /0/private/AddOrder

        validate_only=True -> validate order only (no execution)
        validate_only=False -> real order
        """
        data: Dict[str, Any] = {
            "pair": pair,
            "type": side,
            "ordertype": ordertype,
            "volume": f"{float(volume):.8f}",
        }
        if price is not None:
            data["price"] = str(float(price))
        if validate_only:
            data["validate"] = "true"

        j = self._request("POST", "/0/private/AddOrder", data=data, private=True)
        return j.get("result") or {}

    def query_orders(self, *, txid: str) -> Dict[str, Any]:
        """
        /0/private/QueryOrders
        """
        j = self._request("POST", "/0/private/QueryOrders", data={"txid": txid}, private=True)
        return j.get("result") or {}

    def closed_orders(
        self,
        *,
        start: Optional[int] = None,
        end: Optional[int] = None,
        ofs: Optional[int] = None,
        closetime: str = "close",
    ) -> Dict[str, Any]:
        """
        /0/private/ClosedOrders
        """
        data: Dict[str, Any] = {"closetime": closetime}
        if start is not None:
            data["start"] = int(start)
        if end is not None:
            data["end"] = int(end)
        if ofs is not None:
            data["ofs"] = int(ofs)

        j = self._request("POST", "/0/private/ClosedOrders", data=data, private=True)
        return j.get("result") or {}

    def trades_history(
        self,
        *,
        start: Optional[int] = None,
        end: Optional[int] = None,
        ofs: Optional[int] = None,
        ttype: str = "all",
    ) -> Dict[str, Any]:
        """
        /0/private/TradesHistory
        """
        data: Dict[str, Any] = {"type": ttype}
        if start is not None:
            data["start"] = int(start)
        if end is not None:
            data["end"] = int(end)
        if ofs is not None:
            data["ofs"] = int(ofs)

        j = self._request("POST", "/0/private/TradesHistory", data=data, private=True)
        return j.get("result") or {}

    # ----------------------------
    # CLI self-test (safe)
    # ----------------------------

    @staticmethod
    def _cli_self_test() -> int:
        cfg = KrakenClientConfig.from_env()
        c = KrakenClient(cfg)
        bal = c.balance()
        # print only keys (no amounts) for safety in CLI
        print("Balance keys:", sorted(bal.keys())[:30])
        return 0


def _build_client_from_env() -> KrakenClient:
    cfg = KrakenClientConfig.from_env()
    return KrakenClient(cfg)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="CHAD KrakenClient self-test")
    p.add_argument("--self-test", action="store_true", help="Run safe private Balance call and print keys.")
    p.add_argument("--assetpairs", action="store_true", help="Fetch public AssetPairs for XXBTZCAD and print ordermin.")
    args = p.parse_args(argv)

    if args.self_test:
        return KrakenClient._cli_self_test()

    if args.assetpairs:
        c = _build_client_from_env()
        ap = c.asset_pairs(pair="XXBTZCAD")
        print(ap)
        if "XXBTZCAD" in ap:
            info = ap["XXBTZCAD"]
            print("ordermin:", info.get("ordermin"), "lot_decimals:", info.get("lot_decimals"), "pair_decimals:", info.get("pair_decimals"))
        return 0

    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

