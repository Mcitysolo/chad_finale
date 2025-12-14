"""
Trade Result Logger (Phase 5 – Foundation)

This module provides a **tamper-evident, append-only trade history log** for CHAD.

Every filled trade (paper or live) should be recorded here by executors
(IBKR, Coinbase, paper engine, etc.), so that Phase 5+ components (Shadow
Confidence Router, ML retrainer, weekly reports) have a single, consistent
source of truth.

Key properties
--------------
* One NDJSON file per day:
    data/trades/trade_history_YYYYMMDD.ndjson

* Each line is a JSON object of the form:
    {
      "timestamp_utc": "...",
      "sequence_id":   <int>,
      "payload":       {... full trade outcome ...},
      "prev_hash":     "<SHA256 of previous record or 'GENESIS'>",
      "record_hash":   "<SHA256(prev_hash | compact_payload_json)>"
    }

* If the file is corrupted, logging does NOT crash the system; it starts a new
  chain for that day (sequence_id resets to 1, prev_hash="GENESIS").

Usage pattern (executors)
-------------------------
Executors should create a `TradeResult` instance and call `log_trade_result(...)`
after a trade is **fully known** (fill + PnL, or at least close enough for
Phase-5 analytics).

Example (inside an executor):

    from chad.analytics.trade_result_logger import TradeResult, log_trade_result

    result = TradeResult(
        strategy="beta",
        symbol="SPY",
        side="BUY",
        quantity=3.0,
        fill_price=678.45,
        notional=2035.35,
        pnl=12.34,
        entry_time_utc="2025-12-01T13:30:00+00:00",
        exit_time_utc="2025-12-01T14:00:00+00:00",
        is_live=False,
        broker="ibkr",
        account_id="DU1234567",
        regime="bull",
        tags=["canary", "beta_core"],
        extra={"order_id": "ABC-123", "reason": "legend_rebalance"},
    )

    log_path = log_trade_result(result)

The **Shadow Confidence Router** and ML retrainer can then read the NDJSON
files under `data/trades/` to compute win-rate, Sharpe, drawdown, etc.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional


LOGGER_NAME = "chad.trade_result_logger"
TRADE_DIR = Path("data") / "trades"
TRADE_PREFIX = "trade_history_"


@dataclass(frozen=True)
class TradeResult:
    """
    Normalized representation of a single completed trade outcome.

    All timestamps must be ISO 8601 UTC strings. For intra-day trades,
    entry_time_utc and exit_time_utc can be equal if the trade is opened
    and closed within the same bar.

    Fields:
        strategy:   Strategy name (e.g., "alpha", "beta", "gamma").
        symbol:     Ticker symbol (e.g., "SPY", "AAPL").
        side:       "BUY" or "SELL" (from the perspective of opening trade).
        quantity:   Filled quantity (positive numeric).
        fill_price: Average fill price.
        notional:   Quantity * fill_price in settlement currency.
        pnl:        Realized PnL for the trade (>= 0 for winners, <= 0 for losers).
        entry_time_utc: ISO 8601 string when position was opened.
        exit_time_utc:  ISO 8601 string when position was closed.
        is_live:    True if trade was placed in a live account, False if paper.
        broker:     "ibkr", "coinbase", "paper", etc.
        account_id: Broker account identifier, if applicable.
        regime:     Market regime label (e.g., "bull", "bear", "volatile"), optional.
        tags:       Arbitrary tags for filtering/aggregation (e.g., ["canary", "hedge"]).
        extra:      Arbitrary JSON-serializable dict with additional metadata
                    (e.g., {"order_id": "...", "reason": "..."}).
    """

    strategy: str
    symbol: str
    side: str
    quantity: float
    fill_price: float
    notional: float
    pnl: float
    entry_time_utc: str
    exit_time_utc: str
    is_live: bool
    broker: str
    account_id: Optional[str] = None
    regime: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TradeRecordEnvelope:
    """
    Tamper-evident envelope for a single trade result.

    Fields:
        timestamp_utc: ISO 8601 timestamp when the record was written.
        sequence_id:   Monotonically increasing integer per-day (best-effort).
        payload:       TradeResult as a normalized dict.
        prev_hash:     SHA-256 of previous record, or "GENESIS".
        record_hash:   SHA-256(prev_hash | compact_payload_json).
    """

    timestamp_utc: str
    sequence_id: int
    payload: Dict[str, Any]
    prev_hash: str
    record_hash: str

    def to_json_line(self) -> str:
        """
        Serialize envelope as a compact JSON line.
        """
        return json.dumps(
            {
                "timestamp_utc": self.timestamp_utc,
                "sequence_id": self.sequence_id,
                "payload": self.payload,
                "prev_hash": self.prev_hash,
                "record_hash": self.record_hash,
            },
            separators=(",", ":"),
            sort_keys=True,
        )


def _get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    return logger


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_last_line(path: Path) -> Optional[str]:
    """
    Read the last non-empty line from the given file, if present.
    """
    if not path.exists():
        return None

    last_line: Optional[str] = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                last_line = stripped
    return last_line


def _compute_next_sequence_id(path: Path) -> int:
    """
    Compute the next sequence_id for today's trade history file.

    If parsing fails for any reason, we return 1 and do NOT raise – logging
    should never take down the system due to historical corruption.
    """
    try:
        last_line = _read_last_line(path)
        if not last_line:
            return 1
        parsed = json.loads(last_line)
        last_id = int(parsed.get("sequence_id", 0))
        return max(1, last_id + 1)
    except Exception:
        return 1


def _load_prev_hash(path: Path) -> str:
    """
    Load record_hash from the last record, or "GENESIS" if none.
    """
    last_line = _read_last_line(path)
    if not last_line:
        return "GENESIS"

    try:
        parsed = json.loads(last_line)
        prev_hash = str(parsed.get("record_hash", "")).strip()
        return prev_hash or "GENESIS"
    except Exception:
        return "GENESIS"


def _normalize_for_json(obj: Any) -> Any:
    """
    Best-effort conversion into JSON-serializable primitives.

    This is mainly here so `extra` and future extensions can safely hold
    richer objects without breaking the logger.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [_normalize_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _normalize_for_json(v) for k, v in obj.items()}

    for attr in ("model_dump", "dict", "__dict__"):
        if hasattr(obj, attr):
            try:
                raw = getattr(obj, attr)()
                if isinstance(raw, dict):
                    return _normalize_for_json(raw)
            except Exception:
                continue

    return repr(obj)


def _build_payload(trade: TradeResult) -> Dict[str, Any]:
    """
    Convert a TradeResult into a JSON-safe dict.
    """
    raw = asdict(trade)

    # Ensure extra fields are also normalized.
    if "extra" in raw:
        raw["extra"] = _normalize_for_json(raw["extra"])

    # Normalize tags to list[str].
    tags = raw.get("tags", [])
    if not isinstance(tags, list):
        tags = [str(tags)]
    raw["tags"] = [str(t) for t in tags]

    return _normalize_for_json(raw)


def log_trade_result(trade: TradeResult) -> Path:
    """
    Append a TradeResult to the tamper-evident trade history log.

    Returns:
        The Path to the NDJSON file that was updated.

    This function is deliberately robust:
        * It ensures the directory exists.
        * It handles corrupt files by starting a new hash chain.
        * It never raises due to logging issues; it logs errors instead.
    """
    logger = _get_logger()
    _ensure_directory(TRADE_DIR)

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    trade_path = TRADE_DIR / f"{TRADE_PREFIX}{today}.ndjson"

    try:
        prev_hash = _load_prev_hash(trade_path)
        sequence_id = _compute_next_sequence_id(trade_path)

        payload = _build_payload(trade)

        # Deterministic JSON for hashing (sorted keys, compact separators).
        compact_payload = json.dumps(
            payload,
            separators=(",", ":"),
            sort_keys=True,
        )
        record_hash = sha256(f"{prev_hash}|{compact_payload}".encode("utf-8")).hexdigest()

        envelope = TradeRecordEnvelope(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            sequence_id=sequence_id,
            payload=payload,
            prev_hash=prev_hash,
            record_hash=record_hash,
        )

        with trade_path.open("a", encoding="utf-8") as handle:
            handle.write(envelope.to_json_line())
            handle.write("\n")

        logger.info(
            "TradeResult logged: strategy=%s symbol=%s side=%s qty=%s pnl=%s file=%s seq=%s",
            trade.strategy,
            trade.symbol,
            trade.side,
            trade.quantity,
            trade.pnl,
            trade_path,
            sequence_id,
        )
    except Exception as exc:  # noqa: BLE001
        # Logging must never break CHAD; we log the error and return.
        logger.exception("Failed to log TradeResult: %s", exc)

    return trade_path
