#!/usr/bin/env python3
"""
CHAD Reddit Sentiment Refresh — scheduled CLI wrapper.

Calls RedditSentimentProvider.get_batch_sentiment across a small
universe and persists runtime/reddit_sentiment.json. Fail-soft stub
on any error so downstream consumers never break.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger("chad.ops.reddit_sentiment_refresh")

RUNTIME_DIR = Path("/home/ubuntu/chad_finale/runtime")
STATE_PATH = RUNTIME_DIR / "reddit_sentiment.json"

UNIVERSE = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "TSLA", "AMD", "META", "BTC"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_stub(reason: str) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "signals": {},
        "cache": {},
        "cache_ts": 0.0,
        "last_updated_utc": _utc_now_iso(),
        "stub_reason": reason,
    }
    STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        from chad.intel.reddit_sentiment import RedditSentimentProvider
    except Exception as exc:
        LOG.warning("Import failed, writing stub: %s", exc)
        _write_stub(f"import_error:{exc}")
        return 0

    try:
        provider = RedditSentimentProvider()
        signals = provider.get_batch_sentiment(UNIVERSE)
        LOG.info("reddit sentiment refresh done symbols=%d", len(signals))
    except Exception as exc:
        LOG.warning("Reddit sentiment fetch failed, writing stub: %s", exc)
        _write_stub(f"fetch_error:{exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
