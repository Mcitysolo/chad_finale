"""Feed watchdog: detect stale runtime data feeds and alert via Telegram.

Runs as a systemd timer every 120s. Checks mtime of critical runtime
files against per-feed TTLs and fires a deduped Telegram alert per
stale feed. Exits 0 always (non-fatal watchdog).
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RUNTIME_DIR = ROOT / "runtime"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WATCHED_FEEDS = [
    {"name": "price_cache",    "file": "price_cache.json",          "ttl": 180},
    {"name": "regime_state",   "file": "regime_state.json",         "ttl": 180},
    {"name": "dynamic_caps",   "file": "dynamic_caps.json",         "ttl": 180},
    {"name": "regime_booster", "file": "regime_booster.json",       "ttl": 180},
    {"name": "kraken_prices",  "file": "kraken_prices.json",        "ttl": 120},
    {"name": "reconciliation", "file": "reconciliation_state.json", "ttl": 480},
    {"name": "event_risk",     "file": "event_risk.json",           "ttl": 2400},
]


def _check_feed(feed):
    path = RUNTIME_DIR / feed["file"]
    if not path.exists():
        return {"name": feed["name"], "ttl": feed["ttl"], "stale": True, "age": None, "missing": True}
    try:
        age = int(time.time() - os.path.getmtime(path))
    except OSError:
        return {"name": feed["name"], "ttl": feed["ttl"], "stale": True, "age": None, "missing": True}
    return {
        "name": feed["name"],
        "ttl": feed["ttl"],
        "stale": age > feed["ttl"],
        "age": age,
        "missing": False,
    }


def _format_line(result):
    if result["missing"]:
        return f"  {result['name']}: MISSING"
    return f"  {result['name']}: {result['age']}s old (TTL={result['ttl']}s)"


def main():
    try:
        from chad.utils.telegram_notify import notify
    except Exception as e:
        notify = None
        print(f"feed_watchdog: notify import failed ({e}); alerts disabled", file=sys.stderr)

    results = [_check_feed(f) for f in WATCHED_FEEDS]
    stale = [r for r in results if r["stale"]]

    if stale and notify is not None:
        for r in stale:
            line = _format_line(r)
            msg = "⚠️ STALE FEED DETECTED\n" + line
            try:
                notify(msg, severity="warning", dedupe_key=f"feed_stale_{r['name']}")
            except Exception as e:
                print(f"feed_watchdog: notify failed for {r['name']}: {e}", file=sys.stderr)

    stale_names = ",".join(r["name"] for r in stale) if stale else "none"
    print(f"feed_watchdog: checked={len(results)} stale={len(stale)} feeds={stale_names}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"feed_watchdog: unexpected error: {e}", file=sys.stderr)
        sys.exit(0)
