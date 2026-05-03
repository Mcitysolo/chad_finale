"""
CHAD Choppy Regime Detector
============================
Detects low-quality, fakeout-heavy market conditions.
Uses hysteresis to prevent flip-flopping.

Indicators:
    - ADX < 20 (weak trend strength)
    - Direction flip rate (price crossing 5-day MA)
    - Failed breakout count
    - Small loss churn ratio
    - Poor trend follow-through

Hysteresis:
    - 3 consecutive choppy reads to activate
    - 4 consecutive clean reads to deactivate
    - 60-minute minimum hold in choppy state
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("chad.choppy_regime")

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME = REPO_ROOT / "runtime"
DATA = REPO_ROOT / "data"

CONSECUTIVE_READS_TO_ENTER = 3
CONSECUTIVE_READS_TO_EXIT = 4
MIN_CHOPPY_HOLD_MINUTES = 60
CHOPPY_THRESHOLD = 0.55
CLEAN_THRESHOLD = 0.35

ADX_WEAK_THRESHOLD = 20.0
DIRECTION_FLIP_THRESHOLD = 3
FAILED_BREAKOUT_THRESHOLD = 2
SMALL_LOSS_CHURN_THRESHOLD = 0.60
FOLLOWTHROUGH_WEAK_THRESHOLD = 0.40

STATE_PATH = RUNTIME / "choppy_regime_state.json"


def _read_json(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(path))


def _bar_field(b: dict, *keys) -> float:
    for k in keys:
        if k in b and b[k] is not None:
            try:
                return float(b[k])
            except (TypeError, ValueError):
                continue
    return 0.0


def _load_daily_bars(symbol: str, n_days: int = 30) -> List[dict]:
    """Load last n_days of daily bars for symbol."""
    try:
        bar_file = DATA / "bars" / "1d" / f"{symbol}.json"
        if not bar_file.exists():
            return []
        d = json.loads(bar_file.read_text(encoding="utf-8"))
        bars = d if isinstance(d, list) else d.get("bars", [])
        if not isinstance(bars, list):
            return []
        return bars[-n_days:] if len(bars) >= n_days else bars
    except Exception:
        return []


def _compute_adx(bars: List[dict], period: int = 14) -> Optional[float]:
    """Compute ADX from OHLC bars. Returns None if insufficient data."""
    try:
        if len(bars) < period * 2 + 1:
            return None

        highs = [_bar_field(b, "high", "h") for b in bars]
        lows = [_bar_field(b, "low", "l") for b in bars]
        closes = [_bar_field(b, "close", "c") for b in bars]

        if not all(h > 0 for h in highs):
            return None
        if not all(l > 0 for l in lows):
            return None
        if not all(c > 0 for c in closes):
            return None

        tr_list: List[float] = []
        plus_dm: List[float] = []
        minus_dm: List[float] = []
        for i in range(1, len(bars)):
            h, l, c_prev = highs[i], lows[i], closes[i - 1]
            tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
            tr_list.append(tr)
            p_dm = highs[i] - highs[i - 1]
            m_dm = lows[i - 1] - lows[i]
            plus_dm.append(max(p_dm, 0.0) if p_dm > m_dm else 0.0)
            minus_dm.append(max(m_dm, 0.0) if m_dm > p_dm else 0.0)

        def wilder_smooth(data: List[float], p: int) -> List[float]:
            if len(data) < p:
                return []
            result = [sum(data[:p]) / p]
            for i in range(p, len(data)):
                result.append(result[-1] * (p - 1) / p + data[i] * 1.0 / p)
            return result

        atr = wilder_smooth(tr_list, period)
        sm_plus = wilder_smooth(plus_dm, period)
        sm_minus = wilder_smooth(minus_dm, period)
        if not atr or not sm_plus or not sm_minus:
            return None

        p_di = [
            100.0 * (sm_plus[i] / atr[i]) if atr[i] > 0 else 0.0
            for i in range(len(atr))
        ]
        m_di = [
            100.0 * (sm_minus[i] / atr[i]) if atr[i] > 0 else 0.0
            for i in range(len(atr))
        ]
        dx = [
            100.0 * abs(p_di[i] - m_di[i]) / (p_di[i] + m_di[i])
            if (p_di[i] + m_di[i]) > 0 else 0.0
            for i in range(len(atr))
        ]
        adx_values = wilder_smooth(dx, period)
        return float(adx_values[-1]) if adx_values else None

    except Exception as e:
        logger.debug("choppy: adx_compute_failed err=%s", e)
        return None


def _compute_direction_flips(bars: List[dict], window: int = 5) -> int:
    """Count how many times price crosses its 5-day MA in last window days."""
    try:
        if len(bars) < window + 1:
            return 0
        recent = bars[-(window + 1):]
        closes = [_bar_field(b, "close", "c") for b in recent]
        ma = sum(closes[:window]) / window
        flips = 0
        prev_above = closes[0] > ma
        for i in range(1, len(closes)):
            curr_above = closes[i] > ma
            if curr_above != prev_above:
                flips += 1
            prev_above = curr_above
        return flips
    except Exception:
        return 0


def _compute_failed_breakouts(bars: List[dict], window: int = 10) -> int:
    """Count failed breakouts (break 20-day high/low then close back inside)."""
    try:
        if len(bars) < 20 + window:
            return 0
        failures = 0
        start = max(20, len(bars) - window)
        for i in range(start, len(bars)):
            lookback = bars[max(0, i - 20):i]
            if not lookback:
                continue
            prev_high = max(_bar_field(b, "high", "h") for b in lookback)
            prev_low = min(_bar_field(b, "low", "l") for b in lookback)
            curr = bars[i]
            h = _bar_field(curr, "high", "h")
            l = _bar_field(curr, "low", "l")
            c = _bar_field(curr, "close", "c")
            if h > prev_high and c < prev_high:
                failures += 1
            if l < prev_low and c > prev_low:
                failures += 1
        return failures
    except Exception:
        return 0


def _compute_small_loss_churn() -> float:
    """Fraction of today's losses that are small (< 0.3% of notional). 0.0 if no data."""
    try:
        today = datetime.now(timezone.utc).date()
        trade_dir = DATA / "trades"
        if not trade_dir.exists():
            return 0.0
        total_losses = 0
        small_losses = 0

        for fname in sorted(trade_dir.glob("trade_history_*.ndjson"))[-2:]:
            try:
                with open(fname, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        payload = rec.get("payload", rec)
                        if not isinstance(payload, dict):
                            continue
                        exit_time = payload.get("exit_time_utc", "")
                        if not exit_time:
                            continue
                        try:
                            exit_dt = datetime.fromisoformat(
                                str(exit_time).replace("Z", "+00:00")
                            )
                            if exit_dt.date() != today:
                                continue
                        except Exception:
                            continue
                        try:
                            pnl = float(payload.get("pnl", 0) or 0)
                            notional = float(payload.get("notional", 1000) or 1000)
                        except (TypeError, ValueError):
                            continue
                        if pnl < 0:
                            total_losses += 1
                            if abs(pnl) < notional * 0.003:
                                small_losses += 1
            except Exception:
                continue

        if total_losses == 0:
            return 0.0
        return small_losses / total_losses
    except Exception:
        return 0.0


def _compute_followthrough(bars: List[dict], threshold: float = 0.01) -> float:
    """Fraction of 1%+ moves that follow through next day. 0.5 if insufficient data."""
    try:
        if len(bars) < 10:
            return 0.5
        closes = [_bar_field(b, "close", "c") for b in bars]
        total_moves = 0
        followed = 0
        for i in range(1, len(closes) - 1):
            if closes[i - 1] <= 0 or closes[i] <= 0:
                continue
            move = (closes[i] - closes[i - 1]) / closes[i - 1]
            if abs(move) >= threshold:
                total_moves += 1
                next_move = (closes[i + 1] - closes[i]) / closes[i]
                if (move > 0 and next_move > 0) or (move < 0 and next_move < 0):
                    followed += 1
        if total_moves == 0:
            return 0.5
        return followed / total_moves
    except Exception:
        return 0.5


_PROXY_BY_ASSET_CLASS = {
    "equity": "SPY",
    "futures": "MES",   # Micro E-mini S&P as equity-futures proxy
    "crypto": None,     # Crypto is 24/7; choppy concept does not apply
    "options": "SPY",
    "forex": None,      # Not applicable
}


def compute_choppy_score(
    proxy_symbol: str = "SPY",
    asset_class: str = "equity",
) -> Dict:
    """Compute choppy score and indicator breakdown.

    BG13: choppy proxy is now scoped by asset class. Crypto and forex are
    exempt (24/7 / non-applicable); equities and equity-futures map to SPY/MES.
    """
    _proxy = _PROXY_BY_ASSET_CLASS.get(
        str(asset_class or "").lower(), proxy_symbol
    )
    if _proxy is None:
        return {
            "choppy_score": 0.0,
            "is_choppy_raw": False,
            "choppy_exempt": True,
            "asset_class": asset_class,
            "reason": f"{asset_class}_exempt_from_choppy",
            "indicators": {},
            "proxy_symbol": None,
            "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    proxy_symbol = _proxy

    bars = _load_daily_bars(proxy_symbol, n_days=35)
    if not bars:
        bars = _load_daily_bars("MES", n_days=35)
        if bars:
            proxy_symbol = "MES"

    indicators: Dict = {}
    score = 0.0

    adx = _compute_adx(bars)
    adx_weak = adx is not None and adx < ADX_WEAK_THRESHOLD
    indicators["adx"] = round(adx, 2) if adx is not None else None
    indicators["adx_weak"] = bool(adx_weak)
    if adx_weak:
        score += 0.30
    elif adx is None:
        score += 0.15

    flips = _compute_direction_flips(bars)
    flip_high = flips >= DIRECTION_FLIP_THRESHOLD
    indicators["direction_flips_5d"] = int(flips)
    indicators["direction_flip_high"] = bool(flip_high)
    if flip_high:
        score += 0.25

    failures = _compute_failed_breakouts(bars)
    breakout_fail_high = failures >= FAILED_BREAKOUT_THRESHOLD
    indicators["failed_breakouts_10d"] = int(failures)
    indicators["failed_breakouts_high"] = bool(breakout_fail_high)
    if breakout_fail_high:
        score += 0.20

    churn_ratio = _compute_small_loss_churn()
    churn_high = churn_ratio >= SMALL_LOSS_CHURN_THRESHOLD
    indicators["small_loss_churn_ratio"] = round(churn_ratio, 3)
    indicators["churn_high"] = bool(churn_high)
    if churn_high:
        score += 0.15

    followthrough = _compute_followthrough(bars)
    followthrough_weak = followthrough < FOLLOWTHROUGH_WEAK_THRESHOLD
    indicators["trend_followthrough_rate"] = round(followthrough, 3)
    indicators["followthrough_weak"] = bool(followthrough_weak)
    if followthrough_weak:
        score += 0.10

    return {
        "choppy_score": round(score, 3),
        "is_choppy_raw": score >= CHOPPY_THRESHOLD,
        "indicators": indicators,
        "proxy_symbol": proxy_symbol,
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def evaluate_and_persist(state_path: Optional[Path] = None) -> Dict:
    """
    Evaluate choppy regime with hysteresis. Reads prior state, applies
    confirmation rules, writes new state atomically. Returns the new state.
    """
    if state_path is None:
        state_path = STATE_PATH
    prior = _read_json(state_path)
    currently_choppy = bool(prior.get("choppy_active", False))
    consecutive_choppy = int(prior.get("consecutive_choppy_reads", 0) or 0)
    consecutive_clean = int(prior.get("consecutive_clean_reads", 0) or 0)
    entered_at_raw = prior.get("entered_choppy_at_utc", "") or ""
    entered_at: Optional[datetime] = None
    if entered_at_raw:
        try:
            entered_at = datetime.fromisoformat(
                str(entered_at_raw).replace("Z", "+00:00")
            )
        except Exception:
            entered_at = None

    now = datetime.now(timezone.utc)

    result = compute_choppy_score()
    score = float(result["choppy_score"])
    is_choppy_raw = bool(result["is_choppy_raw"])
    is_clean_raw = score < CLEAN_THRESHOLD

    if is_choppy_raw:
        consecutive_choppy += 1
        consecutive_clean = 0
    elif is_clean_raw:
        consecutive_clean += 1
        consecutive_choppy = 0
    else:
        # In the dead-band: don't reinforce either counter.
        pass

    new_choppy = currently_choppy

    if not currently_choppy:
        if consecutive_choppy >= CONSECUTIVE_READS_TO_ENTER:
            new_choppy = True
            entered_at = now
            logger.warning(
                "choppy_regime ACTIVATED score=%.3f consecutive=%d",
                score, consecutive_choppy,
            )
    else:
        min_hold_ok = True
        if entered_at:
            elapsed_min = (now - entered_at).total_seconds() / 60.0
            min_hold_ok = elapsed_min >= MIN_CHOPPY_HOLD_MINUTES

        if consecutive_clean >= CONSECUTIVE_READS_TO_EXIT and min_hold_ok:
            new_choppy = False
            entered_at = None
            logger.info(
                "choppy_regime DEACTIVATED score=%.3f consecutive_clean=%d",
                score, consecutive_clean,
            )
        elif consecutive_clean >= CONSECUTIVE_READS_TO_EXIT and not min_hold_ok:
            logger.debug("choppy_regime: min_hold not met, staying choppy")

    state = {
        "choppy_active": bool(new_choppy),
        "choppy_score": score,
        "consecutive_choppy_reads": int(consecutive_choppy),
        "consecutive_clean_reads": int(consecutive_clean),
        "entered_choppy_at_utc": (
            entered_at.isoformat() if entered_at else None
        ),
        "ts_utc": now.isoformat(),
        "ttl_seconds": 300,
        "indicators": result["indicators"],
        "proxy_symbol": result.get("proxy_symbol"),
        "thresholds": {
            "enter_threshold": CHOPPY_THRESHOLD,
            "exit_threshold": CLEAN_THRESHOLD,
            "consecutive_to_enter": CONSECUTIVE_READS_TO_ENTER,
            "consecutive_to_exit": CONSECUTIVE_READS_TO_EXIT,
            "min_hold_minutes": MIN_CHOPPY_HOLD_MINUTES,
        },
    }

    try:
        _write_json_atomic(state_path, state)
    except Exception as e:
        logger.warning("choppy_regime: state_write_failed err=%s", e)

    return state


def get_choppy_state(
    state_path: Optional[Path] = None,
    asset_class: str = "equity",
) -> Dict:
    """
    Read current choppy regime state without recomputing.
    Fail-open: returns choppy_active=False on any error or staleness.

    BG13: when asset_class is exempt (crypto/forex), short-circuit with an
    inactive overlay so callers can pass the signal's asset_class without
    further branching.
    """
    _ac = str(asset_class or "").lower()
    if _ac in ("crypto", "forex"):
        return {
            "choppy_active": False,
            "choppy_score": 0.0,
            "choppy_exempt": True,
            "asset_class": asset_class,
        }
    if state_path is None:
        state_path = STATE_PATH
    try:
        state = _read_json(state_path)
        if not state:
            return {"choppy_active": False, "choppy_score": 0.0}

        ts = state.get("ts_utc", "")
        if ts:
            try:
                age_s = (
                    datetime.now(timezone.utc)
                    - datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                ).total_seconds()
                if age_s > int(state.get("ttl_seconds", 300) or 300) * 3:
                    return {
                        "choppy_active": False,
                        "choppy_score": 0.0,
                        "stale": True,
                    }
            except Exception:
                pass

        return state
    except Exception:
        return {"choppy_active": False, "choppy_score": 0.0}


def get_choppy_overlay(state_path: Optional[Path] = None) -> Dict:
    """
    Returns the choppy overlay used by the regime classifier and live_loop.

    Shape (active):
        {"active": True, "score": float, "sizing_multiplier": 0.25,
         "confidence_floor_add": 0.15, "block_trend_following": True}

    Shape (inactive):
        {"active": False, "score": float, "sizing_multiplier": 1.0,
         "confidence_floor_add": 0.0, "block_trend_following": False}
    """
    state = get_choppy_state(state_path)
    active = bool(state.get("choppy_active", False))
    score = float(state.get("choppy_score", 0.0) or 0.0)
    if active:
        return {
            "active": True,
            "score": score,
            "sizing_multiplier": 0.25,
            "confidence_floor_add": 0.15,
            "block_trend_following": True,
        }
    return {
        "active": False,
        "score": score,
        "sizing_multiplier": 1.0,
        "confidence_floor_add": 0.0,
        "block_trend_following": False,
    }


__all__ = [
    "STATE_PATH",
    "CHOPPY_THRESHOLD",
    "CLEAN_THRESHOLD",
    "CONSECUTIVE_READS_TO_ENTER",
    "CONSECUTIVE_READS_TO_EXIT",
    "MIN_CHOPPY_HOLD_MINUTES",
    "compute_choppy_score",
    "evaluate_and_persist",
    "get_choppy_state",
    "get_choppy_overlay",
]
