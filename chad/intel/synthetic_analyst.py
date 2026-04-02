#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


def _repo_root() -> Path:
    raw = str(os.environ.get("CHAD_ROOT", "")).strip()
    if raw:
        p = Path(raw).expanduser().resolve()
        if (p / "chad").is_dir():
            return p
    return Path(__file__).resolve().parents[2]


def _runtime_dir() -> Path:
    raw = str(os.environ.get("CHAD_RUNTIME_DIR", "")).strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (_repo_root() / "runtime").resolve()


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _load_polygon_api_key() -> str:
    key = str(os.environ.get("POLYGON_API_KEY", "")).strip()
    if key:
        return key

    env_path = Path("/etc/chad/polygon.env")
    if env_path.is_file():
        try:
            for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "POLYGON_API_KEY" and v.strip():
                    return v.strip().strip('"').strip("'")
        except Exception:
            return ""
    return ""


def _load_price(symbol: str) -> Optional[float]:
    obj = _read_json(_runtime_dir() / "price_cache.json")
    prices = obj.get("prices")
    if isinstance(prices, dict):
        v = prices.get(symbol)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _load_position(symbol: str) -> Optional[Dict[str, Any]]:
    obj = _read_json(_runtime_dir() / "positions_snapshot.json")
    positions = obj.get("positions")
    if not isinstance(positions, list):
        return None

    for row in positions:
        if not isinstance(row, dict):
            continue
        if str(row.get("symbol") or "").strip().upper() != symbol:
            continue
        qty = float(row.get("position") or 0.0)
        avg = float(row.get("avgCost") or 0.0)
        return {
            "qty": qty,
            "avg_cost_usd": avg,
            "currency": str(row.get("currency") or "USD").strip().upper(),
        }
    return None


def _load_portfolio_value() -> Optional[float]:
    obj = _read_json(_runtime_dir() / "portfolio_snapshot.json")
    raw = obj.get("ibkr_equity")
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def _load_shadow_state() -> Dict[str, Any]:
    for path in [
        _runtime_dir() / "shadow_state.json",
        _repo_root() / "data" / "shadow" / "shadow_state.json",
    ]:
        obj = _read_json(path)
        if obj:
            return obj
    return {}


def _load_daily_bars(symbol: str) -> List[Dict[str, Any]]:
    path = _repo_root() / "data" / "bars" / "1d" / f"{symbol}.json"
    obj = _read_json(path)
    bars = obj.get("bars")
    return bars if isinstance(bars, list) else []


def _close_from_bar(bar: Dict[str, Any]) -> Optional[float]:
    for key in ("close", "c"):
        val = bar.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _pct_change(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    if float(a) == 0:
        return None
    return ((float(b) - float(a)) / float(a)) * 100.0


def _price_direction_label(chg_5d: Optional[float], chg_20d: Optional[float]) -> str:
    c5 = float(chg_5d or 0.0)
    c20 = float(chg_20d or 0.0)

    if c5 >= 5.0 and c20 >= 8.0:
        return "strong_up"
    if c5 >= 2.0 and c20 >= 4.0:
        return "up"
    if c5 <= -5.0 and c20 <= -8.0:
        return "strong_down"
    if c5 <= -2.0 and c20 <= -4.0:
        return "down"
    return "sideways"


def _holding_balance_label(position_value_usd: Optional[float], portfolio_value_usd: Optional[float], has_position: bool) -> str:
    if not has_position:
        return "none"
    if not isinstance(position_value_usd, (int, float)) or not isinstance(portfolio_value_usd, (int, float)) or portfolio_value_usd <= 0:
        return "held_unknown_size"

    pct = (float(position_value_usd) / float(portfolio_value_usd)) * 100.0
    if pct < 3.0:
        return "small"
    if pct < 10.0:
        return "medium"
    return "heavy"


def _headline_tone(symbol: str, limit: int = 5) -> Dict[str, Any]:
    api_key = _load_polygon_api_key()
    if not api_key:
        return {
            "label": "unknown",
            "headlines_used": [],
            "notes": ["polygon_api_key_missing"],
        }

    try:
        resp = requests.get(
            "https://api.polygon.io/v2/reference/news",
            params={
                "ticker": symbol,
                "limit": max(1, min(int(limit), 5)),
                "order": "desc",
                "sort": "published_utc",
                "apiKey": api_key,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return {
                "label": "unknown",
                "headlines_used": [],
                "notes": [f"polygon_http_{resp.status_code}"],
            }

        payload = resp.json()
        results = payload.get("results")
        if not isinstance(results, list):
            return {
                "label": "unknown",
                "headlines_used": [],
                "notes": ["polygon_results_missing"],
            }

        used: List[Dict[str, Any]] = []
        direct = 0
        broad = 0

        for item in results[: max(1, min(int(limit), 5))]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            published = str(item.get("published_utc") or "").strip()
            tickers = item.get("tickers")
            tickers_up = {str(t).strip().upper() for t in tickers} if isinstance(tickers, list) else set()

            if symbol in tickers_up:
                if len(tickers_up) <= 3:
                    direct += 1
                else:
                    broad += 1

            if title:
                used.append(
                    {
                        "title": title,
                        "published_utc": published,
                        "tickers": sorted(list(tickers_up))[:10],
                    }
                )

        if direct >= 2:
            label = "direct"
        elif direct >= 1:
            label = "somewhat_direct"
        elif broad >= 1:
            label = "broad"
        else:
            label = "weak"

        return {
            "label": label,
            "headlines_used": used[:3],
            "notes": [],
        }
    except Exception as exc:
        return {
            "label": "unknown",
            "headlines_used": [],
            "notes": [f"headline_fetch_error:{type(exc).__name__}"],
        }


def build_synthetic_analyst(symbol: str) -> Dict[str, Any]:
    symbol = str(symbol or "").strip().upper()
    if not symbol:
        return {"error": "missing_symbol"}

    price_now = _load_price(symbol)
    position = _load_position(symbol)
    portfolio_value_usd = _load_portfolio_value()
    shadow = _load_shadow_state()
    bars = _load_daily_bars(symbol)
    news = _headline_tone(symbol, limit=5)

    closes = [_close_from_bar(b) for b in bars if isinstance(b, dict)]
    closes = [c for c in closes if isinstance(c, (int, float))]

    close_5d_ago = closes[-6] if len(closes) >= 6 else None
    close_20d_ago = closes[-21] if len(closes) >= 21 else None
    latest_close = closes[-1] if closes else price_now

    chg_5d = _pct_change(close_5d_ago, latest_close)
    chg_20d = _pct_change(close_20d_ago, latest_close)
    price_direction = _price_direction_label(chg_5d, chg_20d)

    qty = float(position["qty"]) if isinstance(position, dict) and isinstance(position.get("qty"), (int, float)) else 0.0
    avg_cost = float(position["avg_cost_usd"]) if isinstance(position, dict) and isinstance(position.get("avg_cost_usd"), (int, float)) else 0.0
    has_position = isinstance(position, dict)

    position_value_usd: Optional[float] = None
    if has_position:
        ref_px = float(price_now) if isinstance(price_now, (int, float)) else avg_cost
        position_value_usd = abs(qty * ref_px)

    balance_label = _holding_balance_label(position_value_usd, portfolio_value_usd, has_position)
    caution_state = str(shadow.get("state") or "UNKNOWN").strip().upper()

    action = "wait"
    reason_bits: List[str] = []

    if price_direction in {"strong_up", "up"} and news["label"] in {"direct", "somewhat_direct"} and not has_position:
        action = "buy_small" if caution_state == "CAUTIOUS" else "buy"
        reason_bits.append("price_strength")
        reason_bits.append("news_support")
    elif has_position and balance_label in {"medium", "heavy"}:
        if price_direction in {"down", "strong_down"}:
            action = "trim"
            reason_bits.append("already_large")
            reason_bits.append("price_weakness")
        else:
            action = "hold"
            reason_bits.append("already_have_enough")
    elif has_position and balance_label == "small":
        action = "hold" if caution_state == "CAUTIOUS" else "buy_small"
        reason_bits.append("already_started_position")
    else:
        if price_direction == "strong_up":
            action = "buy_small" if caution_state in {"CAUTIOUS", "WARMUP"} else "buy"
            reason_bits.append("strong_price")
        elif price_direction == "up":
            action = "buy_small"
            reason_bits.append("some_price_strength")
        elif price_direction == "sideways":
            action = "wait"
            reason_bits.append("no_clear_push")
        else:
            action = "wait"
            reason_bits.append("price_weak")

    if caution_state == "CAUTIOUS" and action == "buy":
        action = "buy_small"
    if caution_state == "WARMUP" and action in {"buy", "buy_small"}:
        action = "wait"

    # Short-term vs long-term lanes
    short_term = action
    long_term = action

    if symbol in {"NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META"}:
        if long_term == "wait":
            long_term = "buy_small"
        elif long_term == "buy_small":
            long_term = "buy"

    if has_position and balance_label in {"medium", "heavy"}:
        long_term = "hold"

    if price_direction in {"down", "strong_down"} and short_term in {"buy", "buy_small"}:
        short_term = "wait"

    return {
        "symbol": symbol,
        "price_now": price_now,
        "position": position,
        "portfolio_value_usd": portfolio_value_usd,
        "position_value_usd": position_value_usd,
        "holding_balance_label": balance_label,
        "shadow_state": caution_state,
        "recent_move_5d_pct": round(chg_5d, 2) if isinstance(chg_5d, (int, float)) else None,
        "recent_move_20d_pct": round(chg_20d, 2) if isinstance(chg_20d, (int, float)) else None,
        "price_direction": price_direction,
        "headline_tone": news["label"],
        "recent_headlines": news["headlines_used"],
        "decision": action,
        "short_term_decision": short_term,
        "long_term_decision": long_term,
        "decision_reasons": reason_bits,
        "notes": news["notes"],
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Build CHAD synthetic analyst view.")
    ap.add_argument("symbol", help="Ticker symbol, e.g. AAPL")
    args = ap.parse_args()

    out = build_synthetic_analyst(args.symbol)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
