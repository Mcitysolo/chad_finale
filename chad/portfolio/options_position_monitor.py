#!/usr/bin/env python3
"""
chad/portfolio/options_position_monitor.py

Options position monitor — active enforcement layer for omega_momentum_options.

Watches open options positions every 60 seconds during market hours and
closes them when stop loss, profit target, or time exit is triggered.

Exit conditions:
  - Stop loss:     option lost 25% of entry value
  - Profit target: option gained 50% of entry value
  - Time exit:     3:45 PM ET hard close
  - Expiry:        contract expires today

Uses Black-Scholes synthetic pricing for current option value when
real-time options quotes are unavailable.

CLI: python -m chad.portfolio.options_position_monitor
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path for direct invocation
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from chad.strategies.options_pricing import black_scholes_price, estimate_iv_from_vix

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
FILLS_DIR = DATA_DIR / "fills"
RUNTIME_DIR = REPO_ROOT / "runtime"
PRICE_CACHE_PATH = RUNTIME_DIR / "price_cache.json"
STATE_PATH = RUNTIME_DIR / "options_monitor_state.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRATEGY_NAME = "omega_momentum_options"
VALID_STATUSES = {"PendingSubmit", "paper_fill"}
LOOKBACK_DAYS = 2

# Market hours (UTC, assumes EDT = UTC-4)
ET_OFFSET_HOURS = -4
MARKET_OPEN_UTC_HOUR = 14   # 9:00 AM ET → 14:00 UTC (buffer before 9:30)
MARKET_CLOSE_UTC_HOUR = 21  # 5:00 PM ET → 21:00 UTC

# Defaults from strategy meta
DEFAULT_STOP_LOSS_PCT = 0.25
DEFAULT_TAKE_PROFIT_PCT = 0.50
DEFAULT_TIME_EXIT_ET = "15:45"


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def _load_state() -> Dict[str, Any]:
    try:
        if STATE_PATH.exists():
            return json.loads(STATE_PATH.read_text())
    except Exception:
        pass
    return {"closed_fill_ids": [], "last_run_utc": None, "open_positions": 0}


def _save_state(state: Dict[str, Any]) -> None:
    """Atomic write via tmp + rename."""
    tmp = STATE_PATH.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.rename(STATE_PATH)
    except Exception as exc:
        LOG.error("Failed to save state: %s", exc)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Fills scanning
# ---------------------------------------------------------------------------

def _scan_open_positions(closed_ids: set) -> List[Dict[str, Any]]:
    """
    Scan recent fills for open omega_momentum_options positions.

    Returns list of position dicts with entry details.
    """
    positions: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc)

    for day_offset in range(LOOKBACK_DAYS + 1):
        day = (now - timedelta(days=day_offset)).strftime("%Y%m%d")
        fills_path = FILLS_DIR / f"FILLS_{day}.ndjson"
        if not fills_path.exists():
            continue

        try:
            for line in fills_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    payload = record.get("payload", record)

                    if payload.get("strategy") != STRATEGY_NAME:
                        continue
                    if payload.get("status") not in VALID_STATUSES:
                        continue

                    fill_id = payload.get("fill_id", "")
                    if not fill_id or fill_id in closed_ids:
                        continue

                    # Must be a BUY to be an opening position
                    if payload.get("side") != "BUY":
                        continue

                    fill_price = float(payload.get("fill_price", 0))
                    if fill_price <= 0:
                        continue

                    extra = payload.get("extra", {}) or {}
                    # Strategy meta may be in extra or in tags
                    meta = extra.get("meta", {}) or {}
                    # Also check extra directly for option fields
                    # (the evidence writer may flatten meta into extra)

                    option_right = (
                        meta.get("option_right")
                        or extra.get("option_right")
                        or ""
                    )
                    strike = float(
                        meta.get("strike")
                        or extra.get("strike")
                        or 0
                    )
                    expiry = (
                        meta.get("expiry")
                        or extra.get("expiry")
                        or ""
                    )

                    positions.append({
                        "fill_id": fill_id,
                        "symbol": payload.get("symbol", ""),
                        "option_right": option_right,
                        "strike": strike,
                        "expiry": expiry,
                        "entry_price": fill_price,
                        "quantity": float(payload.get("quantity", 1)),
                        "fill_time": payload.get("fill_time_utc", ""),
                        "stop_loss_pct": float(
                            meta.get("stop_loss_pct", DEFAULT_STOP_LOSS_PCT)
                        ),
                        "take_profit_pct": float(
                            meta.get("take_profit_pct", DEFAULT_TAKE_PROFIT_PCT)
                        ),
                        "time_exit_et": str(
                            meta.get("time_exit_et", DEFAULT_TIME_EXIT_ET)
                        ),
                    })

                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
        except Exception as exc:
            LOG.warning("Error reading fills file %s: %s", fills_path, exc)
            continue

    return positions


# ---------------------------------------------------------------------------
# Price estimation
# ---------------------------------------------------------------------------

def _load_prices() -> Dict[str, float]:
    """Load current prices from price cache."""
    try:
        if not PRICE_CACHE_PATH.exists():
            return {}
        data = json.loads(PRICE_CACHE_PATH.read_text())
        prices = data.get("prices", data)
        if isinstance(prices, dict):
            # prices is nested: first value is the actual prices dict
            first_val = next(iter(prices.values()), None)
            if isinstance(first_val, dict):
                # Flatten: {"prices": {"AAPL": 200, ...}} structure
                return {k: float(v) for k, v in first_val.items() if isinstance(v, (int, float))}
            return {k: float(v) for k, v in prices.items() if isinstance(v, (int, float))}
    except Exception as exc:
        LOG.warning("Failed to load price cache: %s", exc)
    return {}


def _estimate_current_option_price(
    symbol: str,
    strike: float,
    expiry: str,
    option_right: str,
    prices: Dict[str, float],
    vix: float,
) -> Optional[float]:
    """Estimate current option price using Black-Scholes."""
    spot = prices.get(symbol)
    if spot is None or spot <= 0:
        return None
    if strike <= 0:
        return None

    # Calculate remaining DTE
    now = datetime.now(timezone.utc)
    try:
        if len(expiry) == 8:
            exp_date = datetime.strptime(expiry, "%Y%m%d").replace(tzinfo=timezone.utc)
        else:
            exp_date = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None

    dte_days = (exp_date - now).total_seconds() / 86400.0
    if dte_days < 0:
        # Expired — intrinsic value only
        if option_right == "C":
            return max(0.01, spot - strike)
        return max(0.01, strike - spot)

    T = max(0.001, dte_days / 365.0)
    iv = estimate_iv_from_vix(vix, symbol)
    right = option_right if option_right in ("C", "P") else "C"

    return black_scholes_price(spot, strike, T, 0.05, iv, right)


# ---------------------------------------------------------------------------
# Exit condition checks
# ---------------------------------------------------------------------------

def _check_exit(
    position: Dict[str, Any],
    current_price: Optional[float],
    now: datetime,
) -> Optional[str]:
    """
    Check if a position should be closed.

    Returns reason string or None.
    """
    entry_price = position["entry_price"]

    # Expiry check
    expiry = position.get("expiry", "")
    if expiry:
        try:
            if len(expiry) == 8:
                exp_date = datetime.strptime(expiry, "%Y%m%d").replace(tzinfo=timezone.utc)
            else:
                exp_date = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
            if now >= exp_date:
                return "expiry"
        except (ValueError, AttributeError):
            pass

    # Time exit check
    time_exit_et = position.get("time_exit_et", DEFAULT_TIME_EXIT_ET)
    try:
        parts = time_exit_et.split(":")
        exit_hour, exit_min = int(parts[0]), int(parts[1])
        et_hour = (now.hour + ET_OFFSET_HOURS) % 24
        et_min = now.minute
        et_minutes = et_hour * 60 + et_min
        exit_minutes = exit_hour * 60 + exit_min
        # Only trigger during market day (after market open)
        if et_minutes >= exit_minutes and et_hour >= 9:
            return "time_exit"
    except (ValueError, IndexError):
        pass

    # Price-based exits require current price
    if current_price is None or current_price <= 0:
        return None

    # Stop loss
    stop_pct = position.get("stop_loss_pct", DEFAULT_STOP_LOSS_PCT)
    if current_price <= entry_price * (1.0 - stop_pct):
        return "stop_loss"

    # Profit target
    target_pct = position.get("take_profit_pct", DEFAULT_TAKE_PROFIT_PCT)
    if current_price >= entry_price * (1.0 + target_pct):
        return "take_profit"

    return None


# ---------------------------------------------------------------------------
# Close order writing
# ---------------------------------------------------------------------------

def _write_close_fill(
    position: Dict[str, Any],
    reason: str,
    current_price: float,
    now: datetime,
) -> None:
    """Write a close fill record to FILLS_*.ndjson."""
    import hashlib

    day = now.strftime("%Y%m%d")
    fills_path = FILLS_DIR / f"FILLS_{day}.ndjson"
    FILLS_DIR.mkdir(parents=True, exist_ok=True)

    fill_id_raw = (
        f"close|{position['fill_id']}|{reason}|"
        f"{now.isoformat()}|options_monitor"
    )
    close_fill_id = hashlib.sha256(fill_id_raw.encode()).hexdigest()

    payload = {
        "schema_version": "fill.v3",
        "fill_id": close_fill_id,
        "account_id": "PAPER_EXEC",
        "broker": "paper",
        "venue": "paper",
        "symbol": position["symbol"],
        "side": "SELL",  # closing a long position
        "quantity": position["quantity"],
        "fill_price": round(current_price, 2) if current_price else 0.0,
        "order_type": "MKT",
        "status": "paper_fill",
        "strategy": STRATEGY_NAME,
        "source": "options_position_monitor",
        "source_strategies": [STRATEGY_NAME],
        "asset_class": "options",
        "is_live": False,
        "fill_time_utc": now.isoformat().replace("+00:00", "Z"),
        "entry_time_utc": position.get("fill_time", ""),
        "exit_time_utc": now.isoformat().replace("+00:00", "Z"),
        "notional": round(current_price * position["quantity"] * 100, 2) if current_price else 0.0,
        "partial_fill": False,
        "reject": False,
        "plan_now_iso": now.isoformat().replace("+00:00", "Z"),
        "plan_path": "",
        "tags": ["paper", "options", "close", reason, STRATEGY_NAME],
        "extra": {
            "close_reason": reason,
            "entry_fill_id": position["fill_id"],
            "entry_price": position["entry_price"],
            "exit_price": round(current_price, 2) if current_price else 0.0,
            "option_right": position.get("option_right", ""),
            "strike": position.get("strike", 0),
            "expiry": position.get("expiry", ""),
            "pnl_per_share": round(
                (current_price - position["entry_price"]), 2
            ) if current_price else 0.0,
            "pnl_per_contract": round(
                (current_price - position["entry_price"]) * 100, 2
            ) if current_price else 0.0,
            "synthetic_pricing": True,
            "monitor_client_id": 84,
            "source_strategies": [STRATEGY_NAME],
        },
    }

    # Hash chain (simplified — no prev_hash lookup for monitor closes)
    record = {
        "timestamp_utc": now.isoformat().replace("+00:00", "Z"),
        "sequence_id": close_fill_id[:12],
        "prev_hash": "monitor_close",
        "record_hash": hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest(),
        "payload": payload,
    }

    try:
        with open(fills_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        LOG.info(
            "Closed %s %s strike=%.0f reason=%s price=%.2f",
            position["symbol"],
            position.get("option_right", "?"),
            position.get("strike", 0),
            reason,
            current_price or 0,
        )
    except Exception as exc:
        LOG.error("Failed to write close fill: %s", exc)


# ---------------------------------------------------------------------------
# Main monitor
# ---------------------------------------------------------------------------

def run_monitor() -> Dict[str, Any]:
    """
    Run one cycle of the options position monitor.

    Returns dict with results: ok, checked, closed, reasons, open_positions.
    """
    now = datetime.now(timezone.utc)
    result: Dict[str, Any] = {
        "ok": True,
        "checked": 0,
        "closed": 0,
        "reasons": {},
        "open_positions": 0,
        "ts_utc": now.isoformat().replace("+00:00", "Z"),
    }

    try:
        # Load state
        state = _load_state()
        closed_ids = set(state.get("closed_fill_ids", []))

        # Scan for open positions
        positions = _scan_open_positions(closed_ids)
        result["checked"] = len(positions)
        result["open_positions"] = len(positions)

        if not positions:
            state["last_run_utc"] = now.isoformat().replace("+00:00", "Z")
            state["open_positions"] = 0
            _save_state(state)
            return result

        # Load prices
        prices = _load_prices()
        vix = prices.get("VIX") or prices.get("^VIX") or prices.get("VIXY")
        if vix is None:
            vix = 20.0  # fallback

        # Check market hours — allow time exits even outside hours
        et_hour = (now.hour + ET_OFFSET_HOURS) % 24
        is_weekday = now.weekday() < 5
        in_market_hours = (
            is_weekday
            and MARKET_OPEN_UTC_HOUR <= now.hour <= MARKET_CLOSE_UTC_HOUR
        )

        reasons_count: Dict[str, int] = {}
        newly_closed: List[str] = []

        for pos in positions:
            # Estimate current option price
            current_price = _estimate_current_option_price(
                symbol=pos["symbol"],
                strike=pos["strike"],
                expiry=pos["expiry"],
                option_right=pos["option_right"],
                prices=prices,
                vix=vix,
            )

            # Check exit conditions
            reason = _check_exit(pos, current_price, now)

            # Outside market hours, only process time exits and expiry
            if not in_market_hours and reason not in ("time_exit", "expiry"):
                continue

            if reason:
                close_price = current_price if current_price else pos["entry_price"]
                _write_close_fill(pos, reason, close_price, now)
                newly_closed.append(pos["fill_id"])
                reasons_count[reason] = reasons_count.get(reason, 0) + 1

        # Update state
        closed_ids.update(newly_closed)
        state["closed_fill_ids"] = list(closed_ids)
        state["last_run_utc"] = now.isoformat().replace("+00:00", "Z")
        state["open_positions"] = len(positions) - len(newly_closed)
        _save_state(state)

        result["closed"] = len(newly_closed)
        result["reasons"] = reasons_count
        result["open_positions"] = len(positions) - len(newly_closed)

    except Exception as exc:
        LOG.error("Options monitor error: %s", exc, exc_info=True)
        result["ok"] = False
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    result = run_monitor()
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
