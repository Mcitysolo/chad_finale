#!/usr/bin/env python3
"""
CHAD Strategy Intelligence Refresh — scheduled CLI wrapper.

Refreshes runtime/strategy_intelligence.json and the bias cache at
runtime/strategy_intelligence_cache.json by invoking the existing
StrategyIntelligence engine across a small universe.

Fail-soft: writes a neutral stub if the Claude client is unavailable,
so downstream consumers (live_loop _apply_intelligence_bias) never see
an empty or missing cache.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger("chad.ops.strategy_intelligence_refresh")

RUNTIME_DIR = Path("/home/ubuntu/chad_finale/runtime")
OUTPUT_PATH = RUNTIME_DIR / "strategy_intelligence.json"
CACHE_PATH = RUNTIME_DIR / "strategy_intelligence_cache.json"
# IR1 R3: loud-staleness marker. The refresh silently carried ~100-day-stale
# advisory data forward on failure; this state file makes a dead advisory tier
# observable (read by the health monitor -> coach-voiced NOTIFY).
STATE_PATH = RUNTIME_DIR / "intel_refresh_state.json"

UNIVERSE = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "TSLA", "AMD", "META", "BTC"]
STRATEGIES = ["alpha", "gamma", "alpha_futures", "gamma_futures", "gamma_reversion"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_neutral_stub(reason: str) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "confidence": {},
        "regime": {"advisory": {"profile": "normal", "reasoning": reason, "ts_utc": _utc_now_iso()}},
        "last_updated_utc": _utc_now_iso(),
        "stub_reason": reason,
    }
    tmp = CACHE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(CACHE_PATH)


def _read_prev_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_refresh_state(
    *,
    outcome: str,
    real_calls: int,
    fallback_calls: int,
    last_error_class: str = "",
) -> dict:
    """Persist the loud-staleness marker (fail-open — never raises to caller).

    outcome == "OK" when >=1 real provider answered this run; "FAILED" when
    every advisory tier failed (all neutral fallbacks). On FAILED,
    consecutive_failures increments and the INTEL_REFRESH_FAILED marker is set;
    on OK it resets to 0 and stamps last_success_utc.
    """
    prev = _read_prev_state()
    ok = outcome == "OK"
    prev_fail = int(prev.get("consecutive_failures", 0) or 0)
    consecutive_failures = 0 if ok else prev_fail + 1
    last_success_utc = _utc_now_iso() if ok else prev.get("last_success_utc")

    state = {
        "schema_version": "intel_refresh_state.v1",
        "ts_utc": _utc_now_iso(),
        "last_run_utc": _utc_now_iso(),
        "outcome": outcome,
        "marker": None if ok else "INTEL_REFRESH_FAILED",
        "real_calls": int(real_calls),
        "fallback_calls": int(fallback_calls),
        "consecutive_failures": consecutive_failures,
        "last_success_utc": last_success_utc,
        "last_error_class": last_error_class or "",
        "ttl_seconds": 1800,
    }
    try:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STATE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(STATE_PATH)
    except Exception as exc:  # pragma: no cover - fail-open
        LOG.warning("failed to write intel_refresh_state: %s", exc)
    if not ok:
        LOG.warning(
            "INTEL_REFRESH_FAILED consecutive=%d real=%d fallback=%d err_class=%s",
            consecutive_failures, real_calls, fallback_calls, last_error_class,
        )
    return state


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        from chad.intel.claude_client import ClaudeClient
        from chad.intel.strategy_intelligence import StrategyIntelligence
    except Exception as exc:
        LOG.warning("Imports failed, writing neutral stub: %s", exc)
        _write_neutral_stub(f"import_error:{exc}")
        _write_refresh_state(outcome="FAILED", real_calls=0, fallback_calls=0,
                             last_error_class="import_error")
        return 0

    try:
        client = ClaudeClient.load()
    except Exception as exc:
        LOG.warning("Claude client unavailable, writing neutral stub: %s", exc)
        _write_neutral_stub(f"claude_unavailable:{exc}")
        _write_refresh_state(outcome="FAILED", real_calls=0, fallback_calls=0,
                             last_error_class="claude_unavailable")
        return 0

    si = StrategyIntelligence(client, RUNTIME_DIR)

    ok = 0
    err = 0
    for strategy in STRATEGIES:
        try:
            si.get_regime_profile(strategy)
            ok += 1
        except Exception as exc:
            LOG.warning("regime_profile(%s) failed: %s", strategy, exc)
            err += 1

    for sym in UNIVERSE:
        try:
            si.get_confidence_bias(symbol=sym, strategy_name="alpha", base_confidence=0.5)
            ok += 1
        except Exception as exc:
            LOG.warning("confidence_bias(%s) failed: %s", sym, exc)
            err += 1

    # IR1 R3: outcome is driven by whether a REAL provider answered, not by the
    # ok/err loop counters (which count silent neutral fallbacks as "ok").
    stats = {}
    try:
        stats = client.provider_call_stats()
    except Exception:
        stats = {}
    real_calls = int(stats.get("real", 0))
    fallback_calls = int(stats.get("fallback", 0))
    last_error_class = ""
    try:
        last_error_class = client.last_error_class
    except Exception:
        last_error_class = ""
    outcome = "OK" if real_calls > 0 else "FAILED"
    _write_refresh_state(
        outcome=outcome,
        real_calls=real_calls,
        fallback_calls=fallback_calls,
        last_error_class=last_error_class,
    )

    LOG.info(
        "strategy_intelligence refresh done ok=%d err=%d outcome=%s real=%d fallback=%d",
        ok, err, outcome, real_calls, fallback_calls,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
