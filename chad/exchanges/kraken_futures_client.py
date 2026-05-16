"""
CHAD — Kraken Futures REST Client Scaffold (Phase C Item 1B).

Scaffold only. Not authorized for live trading until futures credentials,
endpoint validation, and execution routing approval exist.

Design
------
Low-level Kraken Futures REST client structured to support future authenticated
private trading. Public endpoints are not handled here (the public intel
publisher at ``chad/market_data/kraken_futures_intel_publisher.py`` covers
public ticker/funding data). This client exposes:

    - order payload construction
    - validation guards
    - request-signing structure (Kraken Futures "Authent" scheme)
    - dry-run execution mode (default)
    - fail-closed behavior when private credentials are missing

The client makes NO live private network calls by default. ``submit_order``
returns a dry-run ``KrakenFuturesOrderResult`` unless ``dry_run=False`` AND
credentials are present AND an ``opener`` is wired in by the caller. Even
then, this scaffold is not production-certified — a future authenticated
smoke test against Kraken Futures is required before live use.

Credential names (read from env first, then ``/etc/chad/kraken.env``):
    - KRAKEN_FUTURES_API_KEY
    - KRAKEN_FUTURES_API_SECRET

Endpoints (constants only; not used at import time):
    PUBLIC_BASE_URL  = https://futures.kraken.com/derivatives/api/v3
    PRIVATE_BASE_URL = https://futures.kraken.com/derivatives/api/v3

Signing structure (Kraken Futures "Authent" scheme, per public docs):
    1. Concatenate (post_data + nonce + endpoint_path_segment)
    2. SHA-256 of (1)
    3. base64-decode the API secret
    4. HMAC-SHA-512 with the decoded secret as key, (2) as message
    5. base64-encode the HMAC digest -> ``Authent`` header value
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

PUBLIC_BASE_URL = "https://futures.kraken.com/derivatives/api/v3"
PRIVATE_BASE_URL = "https://futures.kraken.com/derivatives/api/v3"
DEFAULT_TIMEOUT_SECONDS = 8.0

_ENV_API_KEY = "KRAKEN_FUTURES_API_KEY"
_ENV_API_SECRET = "KRAKEN_FUTURES_API_SECRET"
_FALLBACK_ENV_FILE = Path("/etc/chad/kraken.env")

_VALID_SIDES = frozenset({"buy", "sell"})
_VALID_ORDER_TYPES = frozenset({"mkt", "lmt"})

_SEND_ORDER_PATH = "/sendorder"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KrakenFuturesCredentials:
    """Kraken Futures API credential pair. Treat as sensitive."""

    api_key: str
    api_secret: str


@dataclass(frozen=True)
class KrakenFuturesOrderRequest:
    """Validated order request prior to payload construction."""

    symbol: str
    side: str
    order_type: str
    size: float
    limit_price: Optional[float] = None
    reduce_only: bool = False
    client_order_id: Optional[str] = None


@dataclass(frozen=True)
class KrakenFuturesOrderResult:
    """Result envelope for submit_order calls (dry-run or live)."""

    ok: bool
    dry_run: bool
    status: str
    request: Dict[str, Any]
    response: Dict[str, Any]
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Credential loading
# ---------------------------------------------------------------------------

def _strip_quotes(raw: str) -> str:
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1]
    return s.strip()


def _read_env_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError):
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        out[key] = _strip_quotes(value)
    return out


def load_credentials_from_env(
    env: Optional[Dict[str, str]] = None,
    fallback_path: Optional[Path] = None,
) -> Optional[KrakenFuturesCredentials]:
    """Load Kraken Futures credentials. Returns None if either is missing."""
    src = env if env is not None else os.environ
    api_key = _strip_quotes(src.get(_ENV_API_KEY, "") or "")
    api_secret = _strip_quotes(src.get(_ENV_API_SECRET, "") or "")

    if not api_key or not api_secret:
        path = fallback_path if fallback_path is not None else _FALLBACK_ENV_FILE
        if path.exists():
            file_env = _read_env_file(path)
            if not api_key:
                api_key = _strip_quotes(file_env.get(_ENV_API_KEY, "") or "")
            if not api_secret:
                api_secret = _strip_quotes(file_env.get(_ENV_API_SECRET, "") or "")

    if not api_key or not api_secret:
        return None
    return KrakenFuturesCredentials(api_key=api_key, api_secret=api_secret)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class KrakenFuturesClient:
    """Scaffold Kraken Futures REST client.

    Scaffold only. Not authorized for live trading until futures credentials,
    endpoint validation, and execution routing approval exist.
    """

    def __init__(
        self,
        credentials: Optional[KrakenFuturesCredentials] = None,
        dry_run: bool = True,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        opener: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._credentials = credentials
        self._dry_run = bool(dry_run)
        self._timeout = float(timeout)
        self._opener = opener

    # ----- public introspection -----

    def has_credentials(self) -> bool:
        creds = self._credentials
        if creds is None:
            return False
        return bool(creds.api_key) and bool(creds.api_secret)

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    # ----- validation -----

    @staticmethod
    def validate_order(order: KrakenFuturesOrderRequest) -> Optional[str]:
        """Return an error string if invalid, else None."""
        if not isinstance(order.symbol, str) or not order.symbol.startswith("PF_"):
            return f"invalid_symbol: {order.symbol!r} (must start with 'PF_')"
        side = (order.side or "").lower()
        if side not in _VALID_SIDES:
            return f"invalid_side: {order.side!r} (expected one of {sorted(_VALID_SIDES)})"
        order_type = (order.order_type or "").lower()
        if order_type not in _VALID_ORDER_TYPES:
            return f"invalid_order_type: {order.order_type!r} (expected one of {sorted(_VALID_ORDER_TYPES)})"
        try:
            size_val = float(order.size)
        except (TypeError, ValueError):
            return f"invalid_size: {order.size!r}"
        if size_val <= 0.0:
            return f"invalid_size: {order.size!r} (must be > 0)"
        if order_type == "lmt":
            if order.limit_price is None:
                return "limit_price_required_for_lmt"
            try:
                price_val = float(order.limit_price)
            except (TypeError, ValueError):
                return f"invalid_limit_price: {order.limit_price!r}"
            if price_val <= 0.0:
                return f"invalid_limit_price: {order.limit_price!r} (must be > 0)"
        return None

    # ----- payload construction -----

    @staticmethod
    def build_order_payload(order: KrakenFuturesOrderRequest) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "symbol": order.symbol,
            "side": (order.side or "").lower(),
            "orderType": (order.order_type or "").lower(),
            "size": float(order.size),
        }
        if (order.order_type or "").lower() == "lmt" and order.limit_price is not None:
            payload["limitPrice"] = float(order.limit_price)
        if order.reduce_only:
            payload["reduceOnly"] = True
        if order.client_order_id:
            payload["cliOrdId"] = str(order.client_order_id)
        return payload

    # ----- signing -----

    def sign_request(self, path: str, post_data: str, nonce: str) -> Dict[str, str]:
        """Compute the Kraken Futures Authent headers.

        This implements the documented Authent scheme:
            sha256(post_data + nonce + path) ->
            HMAC-SHA-512(base64-decoded secret, sha256_digest) ->
            base64-encoded result
        It is structurally correct but NOT production-certified for live use
        until an authenticated smoke test against Kraken Futures has been run.
        """
        creds = self._credentials
        if creds is None or not creds.api_key or not creds.api_secret:
            raise RuntimeError("kraken_futures_sign_missing_credentials")

        try:
            secret_bytes = base64.b64decode(creds.api_secret)
        except Exception as exc:
            raise RuntimeError(f"kraken_futures_invalid_secret_b64: {exc}") from exc

        # The Authent scheme strips the leading "/derivatives" prefix per
        # Kraken Futures docs; we use the bare path segment passed in here.
        message = (post_data + nonce + path).encode("utf-8")
        sha256_digest = hashlib.sha256(message).digest()
        mac = hmac.new(secret_bytes, sha256_digest, hashlib.sha512).digest()
        authent = base64.b64encode(mac).decode("utf-8")

        return {
            "APIKey": creds.api_key,
            "Authent": authent,
            "Nonce": nonce,
        }

    @staticmethod
    def _generate_nonce() -> str:
        return str(int(time.time() * 1000))

    # ----- submission -----

    def submit_order(self, order: KrakenFuturesOrderRequest) -> KrakenFuturesOrderResult:
        """Validate + (optionally) submit an order.

        Default is dry-run: returns a dry-run result with no network activity.
        Live submission requires dry_run=False AND credentials AND a wired
        opener. Even with those, this scaffold is not production-certified.
        """
        err = self.validate_order(order)
        request_payload = self.build_order_payload(order) if err is None else {
            "symbol": order.symbol,
            "side": order.side,
            "orderType": order.order_type,
            "size": order.size,
        }

        if err is not None:
            return KrakenFuturesOrderResult(
                ok=False,
                dry_run=self._dry_run,
                status="validation_failed",
                request=request_payload,
                response={},
                error=err,
            )

        if self._dry_run:
            return KrakenFuturesOrderResult(
                ok=True,
                dry_run=True,
                status="dry_run_accepted",
                request=request_payload,
                response={"note": "scaffold dry-run; no network call"},
                error=None,
            )

        if not self.has_credentials():
            return KrakenFuturesOrderResult(
                ok=False,
                dry_run=False,
                status="missing_credentials",
                request=request_payload,
                response={},
                error="kraken_futures_credentials_missing",
            )

        if self._opener is None:
            return KrakenFuturesOrderResult(
                ok=False,
                dry_run=False,
                status="no_opener_wired",
                request=request_payload,
                response={},
                error="kraken_futures_live_opener_not_configured",
            )

        nonce = self._generate_nonce()
        post_body = _urlencode_sorted(request_payload)
        try:
            headers = self.sign_request(_SEND_ORDER_PATH, post_body, nonce)
        except Exception as exc:
            return KrakenFuturesOrderResult(
                ok=False,
                dry_run=False,
                status="sign_failed",
                request=request_payload,
                response={},
                error=str(exc),
            )

        url = PRIVATE_BASE_URL + _SEND_ORDER_PATH
        try:
            raw = self._opener(
                url=url,
                method="POST",
                headers=headers,
                body=post_body.encode("utf-8"),
                timeout=self._timeout,
            )
        except Exception as exc:
            return KrakenFuturesOrderResult(
                ok=False,
                dry_run=False,
                status="transport_error",
                request=request_payload,
                response={},
                error=f"{type(exc).__name__}: {exc}",
            )

        try:
            decoded = json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else dict(raw)
            if not isinstance(decoded, dict):
                decoded = {"raw": decoded}
        except Exception as exc:
            return KrakenFuturesOrderResult(
                ok=False,
                dry_run=False,
                status="bad_response",
                request=request_payload,
                response={},
                error=f"{type(exc).__name__}: {exc}",
            )

        ok = bool(decoded.get("result") == "success" or decoded.get("status") == "ok")
        return KrakenFuturesOrderResult(
            ok=ok,
            dry_run=False,
            status="submitted" if ok else "rejected",
            request=request_payload,
            response=decoded,
            error=None if ok else str(decoded.get("error") or decoded.get("status") or "unknown"),
        )


def _urlencode_sorted(payload: Dict[str, Any]) -> str:
    parts = []
    for k in sorted(payload.keys()):
        v = payload[k]
        if isinstance(v, bool):
            v_str = "true" if v else "false"
        else:
            v_str = str(v)
        parts.append(f"{k}={v_str}")
    return "&".join(parts)


__all__ = [
    "PUBLIC_BASE_URL",
    "PRIVATE_BASE_URL",
    "DEFAULT_TIMEOUT_SECONDS",
    "KrakenFuturesCredentials",
    "KrakenFuturesOrderRequest",
    "KrakenFuturesOrderResult",
    "KrakenFuturesClient",
    "load_credentials_from_env",
]
