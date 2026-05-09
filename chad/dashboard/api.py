"""CHAD real-time dashboard — FastAPI + WebSocket on port 8765.

Serves a plain-English dashboard that reflects CHAD's current
status, portfolio, open positions, strategy activity, and system
health. All runtime state files are read defensively; missing
files return sensible fallbacks rather than crashing.

v2: adds password auth (HTTP Basic + session cookie) and new
endpoints /api/recent-trades, /api/leaderboard, /api/market.
v3: adds /api/chat — Claude-powered advisory chat panel
(advisory only; no execution, no config mutation).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import secrets
import subprocess
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse

REPO = Path(__file__).resolve().parents[2]
RUNTIME = REPO / "runtime"
DATA = REPO / "data"
STATIC = Path(__file__).resolve().parent / "static"

DASHBOARD_USER = "chad"
_pw = os.environ.get("CHAD_DASHBOARD_PASSWORD")
if not _pw:
    raise RuntimeError(
        "CHAD_DASHBOARD_PASSWORD environment variable is required. "
        "Set it in /etc/chad/dashboard.env (mode 0640 root:chad). "
        "See Step 9 of the overhaul for the expected deployment."
    )
DASHBOARD_PASSWORD = _pw
SESSION_COOKIE = "chad_session"
SESSION_TTL_SECONDS = 24 * 3600
SESSIONS: dict[str, float] = {}

_LOGIN_FAILURES: dict = {}  # ip -> (fail_count, lockout_until_ts)
_LOGIN_MAX_ATTEMPTS = 5
_LOGIN_LOCKOUT_SECONDS = 300  # 5 minutes

STRATEGY_NAMES = {
    "alpha": "Stock Strategy",
    "beta": "Institutional Compounder",
    "beta_trend": "Trend Strategy",
    "gamma": "Swing Strategy",
    "delta": "Growth Strategy",
    "omega": "Hedge Strategy",
    "omega_vol": "Volatility Strategy",
    "omega_macro": "Macro Strategy",
    "alpha_futures": "Futures Strategy",
    "gamma_futures": "Futures Momentum",
    "alpha_crypto": "Crypto Strategy",
    "alpha_options": "Options Strategy",
    "gamma_reversion": "Mean Reversion",
    "delta_pairs": "Pairs Strategy",
}

SERVICES_TO_CHECK = [
    "chad-live-loop.service",
    "chad-ibkr-harvest.service",
    "chad-position-reconciler.service",
    "chad-pnl-refresh.service",
    "chad-reconciliation-publisher.service",
    "chad-price-cache-refresh.service",
    "chad-reddit-sentiment-refresh.service",
    "chad-trends-refresh.service",
    "chad-short-interest-refresh.service",
    "chad-strategy-intelligence-refresh.service",
    "chad-expectancy-tracker.timer",
]


def _load_json(path: Path) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _get_mtime(path: Path):
    try:
        return datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat()
    except Exception:
        return None


def _load_json_with_mtime(path: Path) -> tuple:
    """Load JSON and return (data_dict, mtime_iso_str).
    Returns ({}, None) on any error."""
    try:
        if not path.is_file():
            return {}, None
        mtime = path.stat().st_mtime
        data = json.loads(path.read_text(encoding="utf-8"))
        return data, datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except Exception:
        return {}, None


def _fmt_money(x: float | int | None) -> str:
    if x is None:
        return "—"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.0f}"


def _fmt_money_signed(x: float | int | None) -> str:
    if x is None:
        return "—"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    if x > 0:
        return f"+${x:,.0f}" if abs(x) >= 10 else f"+${x:,.2f}"
    if x < 0:
        return f"-${abs(x):,.0f}" if abs(x) >= 10 else f"-${abs(x):,.2f}"
    return "$0"


def _sharpe_to_score(sharpe: float | None) -> int:
    if sharpe is None:
        return 0
    pts = [(-1.0, 0), (0.0, 20), (0.3, 40), (0.7, 75), (1.0, 100)]
    if sharpe <= pts[0][0]:
        return 0
    if sharpe >= pts[-1][0]:
        return 100
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= sharpe <= x1:
            t = (sharpe - x0) / (x1 - x0) if x1 != x0 else 0
            return int(round(y0 + t * (y1 - y0)))
    return 0


def _mode_from_scr(scr: dict) -> tuple[str, str, str]:
    state = (scr.get("state") or "").upper()
    sizing = scr.get("sizing_factor") or 0.0
    if state in ("WARMUP", "PAUSED"):
        return "Training", "CHAD is building his track record", f"Trading at {int(sizing * 100)}% size"
    if state == "LIVE":
        return "Live", "CHAD is trading with real conviction", f"Trading at {int(sizing * 100)}% size"
    if state == "CAUTIOUS":
        return "Cautious", "CHAD is probing markets carefully", f"Trading at {int(sizing * 100)}% size"
    return state.title() or "Unknown", "Status unavailable", f"Trading at {int(sizing * 100)}% size"


def _next_level(score: int) -> tuple[str, str, int]:
    if score < 30:
        return "Cautious", "Unlocks 25% size at score 30", 30
    if score < 60:
        return "Confident", "Unlocks 50% size at score 60", 60
    if score < 85:
        return "Full Send", "Unlocks 100% size at score 85", 85
    return "Max", "Peak performance", 100


def _vix_label_color(v: float | None) -> tuple[str, str]:
    if v is None:
        return "Unknown", "grey"
    if v < 15:
        return "Calm", "green"
    if v < 20:
        return "Normal", "grey"
    if v < 25:
        return "Cautious", "yellow"
    if v < 30:
        return "Nervous", "orange"
    return "Fearful", "red"


def _vix_label_legacy(v: float | None) -> str:
    label, _ = _vix_label_color(v)
    if v is None:
        return "Unknown"
    return f"{label} ({v:.0f})"


def _market_is_open(now_utc: datetime) -> bool:
    if now_utc.weekday() >= 5:
        return False
    minutes = now_utc.hour * 60 + now_utc.minute
    return 14 * 60 + 30 <= minutes < 21 * 60


# ------------------------- Auth -------------------------

def _session_valid(token: str | None) -> bool:
    if not token:
        return False
    expiry = SESSIONS.get(token)
    if not expiry:
        return False
    if expiry < time.time():
        SESSIONS.pop(token, None)
        return False
    return True


def _check_basic(auth_header: str | None) -> bool:
    if not auth_header or not auth_header.lower().startswith("basic "):
        return False
    try:
        raw = base64.b64decode(auth_header.split(" ", 1)[1]).decode("utf-8", "replace")
        user, _, pw = raw.partition(":")
    except Exception:
        return False
    return secrets.compare_digest(user, DASHBOARD_USER) and secrets.compare_digest(pw, DASHBOARD_PASSWORD)


def _new_session_token() -> str:
    tok = secrets.token_urlsafe(32)
    SESSIONS[tok] = time.time() + SESSION_TTL_SECONDS
    return tok


def _request_authenticated(request: Request) -> bool:
    cookie = request.cookies.get(SESSION_COOKIE)
    if _session_valid(cookie):
        return True
    return _check_basic(request.headers.get("authorization"))


def _ws_authenticated(websocket: WebSocket) -> bool:
    token = websocket.query_params.get("token")
    if token:
        try:
            raw = base64.b64decode(token).decode("utf-8", "replace")
            user, _, pw = raw.partition(":")
            if secrets.compare_digest(user, DASHBOARD_USER) and secrets.compare_digest(pw, DASHBOARD_PASSWORD):
                return True
        except Exception:
            pass
    cookie = websocket.cookies.get(SESSION_COOKIE)
    return _session_valid(cookie)


# ------------------------- State Builder -------------------------

class StateBuilder:
    def __init__(self) -> None:
        self._portfolio_cache: dict | None = None
        self._portfolio_cache_ts: float = 0.0
        self._services_cache: dict | None = None
        self._services_cache_ts: float = 0.0
        self._strategies_cache: list | None = None
        self._strategies_cache_ts: float = 0.0
        self._last_known_positions: list = []
        self._ibkr_connected: bool = False

    def _ibkr_snapshot(self) -> tuple[bool, list]:
        try:
            from ib_async import IB  # type: ignore
        except Exception:
            return False, self._last_known_positions
        ib = IB()
        try:
            ib.connect("127.0.0.1", 4002, clientId=80, timeout=3)
            poss = ib.positions()
            out = []
            for p in poss:
                try:
                    out.append(
                        {
                            "symbol": getattr(p.contract, "symbol", "?"),
                            "position": float(p.position),
                        }
                    )
                except Exception:
                    continue
            self._last_known_positions = out
            self._ibkr_connected = True
            return True, out
        except Exception:
            return False, self._last_known_positions
        finally:
            try:
                ib.disconnect()
            except Exception:
                pass

    def _portfolio(self) -> dict:
        now = time.time()
        if self._portfolio_cache and now - self._portfolio_cache_ts < 30:
            return self._portfolio_cache
        pnl = _load_json(RUNTIME / "pnl_state.json")
        scr = _load_json(RUNTIME / "scr_state.json")
        scr_stats = scr.get("stats") or {}

        equity = pnl.get("account_equity")

        today_raw = pnl.get("realized_pnl")
        try:
            today_realized = float(today_raw) if today_raw is not None else None
        except (TypeError, ValueError):
            today_realized = None

        total_raw = scr_stats.get("total_pnl")
        try:
            total_paper = float(total_raw) if total_raw is not None else None
        except (TypeError, ValueError):
            total_paper = None

        eff_raw = scr_stats.get("effective_trades")
        try:
            effective_trades = int(eff_raw) if eff_raw is not None else None
        except (TypeError, ValueError):
            effective_trades = None

        wr_raw = scr_stats.get("win_rate")
        try:
            win_rate = float(wr_raw) if wr_raw is not None else None
        except (TypeError, ValueError):
            win_rate = None

        sf_raw = scr.get("sizing_factor")
        try:
            sizing_factor = float(sf_raw) if sf_raw is not None else None
        except (TypeError, ValueError):
            sizing_factor = None

        if today_realized is None:
            realized_label = "—"
        elif today_realized > 0:
            realized_label = f"Up ${abs(today_realized):,.0f} realized"
        elif today_realized < 0:
            realized_label = f"Down ${abs(today_realized):,.0f} realized"
        else:
            realized_label = "Flat"

        out = {
            "account_value": equity,
            "account_value_label": _fmt_money(equity),
            "realized_pnl": (round(today_realized, 2) if today_realized is not None else None),
            "realized_pnl_label": realized_label,
            "today_realized_pnl": (round(today_realized, 2) if today_realized is not None else None),
            "today_realized_pnl_label": _fmt_money_signed(today_realized),
            "total_paper_pnl": (round(total_paper, 2) if total_paper is not None else None),
            "total_paper_pnl_label": _fmt_money_signed(total_paper),
            "effective_trades": effective_trades,
            "win_rate": (round(win_rate, 4) if win_rate is not None else None),
            "win_rate_pct_label": (f"{win_rate * 100:.1f}%" if win_rate is not None else "—"),
            "sizing_factor": (round(sizing_factor, 4) if sizing_factor is not None else None),
            "sizing_factor_pct_label": (f"{int(round(sizing_factor * 100))}%" if sizing_factor is not None else "—"),
            "open_trades_pnl": 0.0,
            "note": "Practice money",
        }
        self._portfolio_cache = out
        self._portfolio_cache_ts = now
        return out

    def _open_positions(self) -> list:
        guard = _load_json(RUNTIME / "position_guard.json")
        out = []
        for key, rec in guard.items():
            if not isinstance(rec, dict) or not rec.get("open"):
                continue
            sym = rec.get("symbol") or key.split("|")[-1]
            side = (rec.get("side") or "").upper()
            direction = "Short" if side == "SELL" else "Long"
            plain = "Betting it goes DOWN" if direction == "Short" else "Betting it goes UP"
            strat = rec.get("strategy") or (key.split("|")[0] if "|" in key else "unknown")
            out.append(
                {
                    "symbol": sym,
                    "direction": direction,
                    "direction_plain": plain,
                    "quantity": rec.get("quantity", 0),
                    "entry_price": rec.get("entry_price") or rec.get("fill_price"),
                    "opened_at": rec.get("opened_at") or rec.get("ts_utc"),
                    "strategy": strat,
                    "strategy_plain": STRATEGY_NAMES.get(strat, strat.title()),
                    "is_chad_trade": True,
                }
            )
        return out

    def _chad_status(self, scr: dict) -> dict:
        mode, detail, sizing_label = _mode_from_scr(scr)
        stats = scr.get("stats") or {}
        trades = int(stats.get("effective_trades") or 0)
        win_rate = float(stats.get("win_rate") or 0.0)
        sharpe = stats.get("sharpe_like")
        score = _sharpe_to_score(sharpe)
        next_name, next_detail, next_threshold = _next_level(score)
        progress = int(min(100, round(score * 100 / max(1, next_threshold))))
        warmup_total = 100
        warmup_remaining = max(0, warmup_total - trades) if (scr.get("state") or "").upper() == "WARMUP" else 0
        warmup_progress = int(min(100, round(trades * 100 / warmup_total))) if warmup_total else 100
        sizing_pct = int(round(float(scr.get("sizing_factor") or 0.0) * 100))
        return {
            "mode": mode,
            "mode_detail": detail,
            "sizing_label": sizing_label,
            "sizing_pct": sizing_pct,
            "trades_completed": trades,
            "warmup_total": warmup_total,
            "warmup_remaining": warmup_remaining,
            "warmup_progress_pct": warmup_progress,
            "win_rate_pct": round(win_rate * 100, 1),
            "performance_score": score,
            "progress_to_next_level_pct": progress,
            "next_level": next_name,
            "next_level_detail": next_detail,
        }

    def _strategies(self) -> list:
        now = time.time()
        if self._strategies_cache and now - self._strategies_cache_ts < 60:
            return self._strategies_cache
        exp = _load_json(RUNTIME / "expectancy_state.json")
        caps = _load_json(RUNTIME / "dynamic_caps.json")
        weights = caps.get("normalized_weights") or {}
        strat_stats = exp.get("strategies") or {}

        all_names = set(weights.keys()) | set(strat_stats.keys()) | set(STRATEGY_NAMES.keys())
        out = []
        for name in sorted(all_names):
            s = strat_stats.get(name, {})
            total = int(s.get("total_trades") or 0)
            win_rate = float(s.get("win_rate") or 0.0)
            expectancy = float(s.get("expectancy") or 0.0)
            perf_status = s.get("status", "new" if total < 10 else "underperforming")
            weight = float(weights.get(name) or 0.0)
            signals = int(round(weight * 20)) if weight > 0 else 0
            if signals > 0:
                status_str = "active"
                status_label = f"Watching {signals} opportunities"
            else:
                status_str = "idle"
                status_label = "Waiting for the right moment"
            out.append(
                {
                    "name": STRATEGY_NAMES.get(name, name.title()),
                    "internal_name": name,
                    "signals_now": signals,
                    "status": status_str,
                    "status_label": status_label,
                    "win_rate": round(win_rate, 3),
                    "expectancy": round(expectancy, 2),
                    "total_trades": total,
                    "performance_status": perf_status,
                }
            )
        self._strategies_cache = out
        self._strategies_cache_ts = now
        return out

    def _system_health(self) -> dict:
        now = time.time()
        if self._services_cache and now - self._services_cache_ts < 30:
            return self._services_cache
        ok = 0
        failed = 0
        for svc in SERVICES_TO_CHECK:
            try:
                r = subprocess.run(
                    ["systemctl", "show", svc,
                     "--property=ActiveState,SubState,Type,Result"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                props = {}
                for line in (r.stdout or "").splitlines():
                    if "=" in line:
                        k, _, v = line.partition("=")
                        props[k.strip()] = v.strip()
                active = props.get("ActiveState", "")
                sub = props.get("SubState", "")
                svc_type = props.get("Type", "")
                result = props.get("Result", "")
                is_ok = (
                    active in ("active", "activating", "reloading")
                    or (svc_type == "oneshot" and result == "success")
                    or (active == "inactive" and sub == "dead" and result in ("", "success"))
                )
                if is_ok:
                    ok += 1
                else:
                    failed += 1
            except Exception:
                failed += 1
        recon = _load_json(RUNTIME / "reconciliation_state.json")
        recon_status = recon.get("status", "UNKNOWN")
        all_good = failed == 0 and recon_status == "GREEN"
        summary = (
            "All systems running"
            if all_good
            else f"{failed} issue(s) detected" if failed else "Reconciliation drift"
        )
        full = _load_json(RUNTIME / "full_execution_cycle_last.json")
        last_cycle = full.get("ts_utc") or full.get("finished_utc") or ""
        out = {
            "all_good": all_good,
            "summary": summary,
            "services_ok": ok,
            "services_failed": failed,
            "last_cycle": last_cycle,
            "reconciliation": recon_status,
        }
        self._services_cache = out
        self._services_cache_ts = now
        return out

    def _intelligence(self) -> dict:
        prices = _load_json(RUNTIME / "price_cache.json").get("prices") or {}
        trends = _load_json(RUNTIME / "trends_state.json").get("signals") or {}
        reddit = _load_json(RUNTIME / "reddit_sentiment.json").get("signals") or {}
        kraken = _load_json(RUNTIME / "kraken_prices.json").get("prices") or {}
        vix = prices.get("VIX")
        if vix is None:
            vixy = prices.get("VIXY")
            vix = round(vixy * 0.67, 2) if isinstance(vixy, (int, float)) else None
        btc = prices.get("BTC") or prices.get("BTCUSD") or kraken.get("BTC-USD")
        spy_signal = trends.get("SPY", {}).get("signal", "NEUTRAL").lower()
        top_reddit = None
        best_mentions = -1
        for sym, s in reddit.items():
            mc = s.get("mention_count") or 0
            if mc > best_mentions:
                best_mentions = mc
                top_reddit = sym
        event_risk = _load_json(RUNTIME / "event_risk.json")
        regime = event_risk.get("regime") or "NEUTRAL"
        return {
            "vix": vix,
            "vix_label": _vix_label_legacy(vix),
            "btc_price": btc,
            "spy_trend": spy_signal,
            "top_reddit_mention": top_reddit,
            "market_regime": regime,
        }

    def _business(self) -> dict:
        """
        CHAD-as-a-business framework state for the dashboard:
        phase, tier, authorized salary, regime booster status.
        """
        biz = _load_json(RUNTIME / "business_phase.json") or {}
        tier = _load_json(RUNTIME / "tier_state.json") or {}
        wd = _load_json(RUNTIME / "withdrawal_authorization.json") or {}
        booster = _load_json(RUNTIME / "regime_booster.json") or {}

        try:
            booster_mult = float(booster.get("multiplier", 1.0) or 1.0)
        except (TypeError, ValueError):
            booster_mult = 1.0

        try:
            authorized = float(wd.get("authorized_withdrawal_usd", 0.0) or 0.0)
        except (TypeError, ValueError):
            authorized = 0.0

        try:
            hwm = float(wd.get("high_water_mark_usd", 0.0) or 0.0)
        except (TypeError, ValueError):
            hwm = 0.0

        try:
            growth = float(biz.get("growth_pct_from_seed", 0.0) or 0.0)
        except (TypeError, ValueError):
            growth = 0.0

        return {
            "phase": str(biz.get("phase") or wd.get("phase") or "?"),
            "phase_description": str(biz.get("phase_description") or ""),
            "tier": str(tier.get("tier_name") or "?"),
            "tier_strategies_enabled": len(tier.get("enabled_strategies") or []),
            "authorized_salary_usd": round(authorized, 2),
            "high_water_mark_usd": round(hwm, 2),
            "growth_pct_from_seed": round(growth, 2),
            "regime_booster_active": bool(booster.get("active", False)),
            "regime_booster_multiplier": round(booster_mult, 3),
            "next_phase_requirement": str(biz.get("next_phase_requirement") or ""),
        }

    def build(self) -> dict[str, Any]:
        scr = _load_json(RUNTIME / "scr_state.json")
        connected = self._ibkr_connected
        _source_mtimes = [
            m for m in [
                _get_mtime(RUNTIME / "scr_state.json"),
                _get_mtime(RUNTIME / "dynamic_caps.json"),
                _get_mtime(RUNTIME / "regime_state.json"),
                _get_mtime(RUNTIME / "price_cache.json"),
                _get_mtime(RUNTIME / "reconciliation_state.json"),
            ] if m is not None
        ]
        oldest_source_mtime_utc = min(_source_mtimes) if _source_mtimes else None
        return {
            "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "oldest_source_mtime_utc": oldest_source_mtime_utc,
            "ibkr_connected": connected,
            "chad_status": self._chad_status(scr),
            "portfolio": self._portfolio(),
            "open_positions": self._open_positions(),
            "strategies": self._strategies(),
            "system_health": self._system_health(),
            "intelligence": self._intelligence(),
            "business": self._business(),
        }

    async def refresh_ibkr_async(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._ibkr_snapshot)
        except Exception:
            pass


builder = StateBuilder()
app = FastAPI(title="CHAD Dashboard", version="2.0")


# ------------------------- Pages -------------------------

LOGIN_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>CHAD — Sign in</title>
<style>
:root{--bg:#F7F8FA;--card:#FFFFFF;--border:#E8EAED;--text:#0D1117;--sub:#6B7280;--green:#00C805;--red:#FF3B30;--radius:12px;--shadow:0 4px 12px rgba(0,0,0,.10);}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;background:var(--bg);font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","Segoe UI",sans-serif;color:var(--text);}
.card{width:400px;max-width:calc(100vw - 32px);background:var(--card);border-radius:var(--radius);box-shadow:var(--shadow);padding:40px 32px;border:1px solid var(--border);}
.logo{display:flex;align-items:center;gap:10px;font-weight:800;font-size:28px;letter-spacing:-.5px;}
.dot{width:10px;height:10px;border-radius:50%;background:var(--green);box-shadow:0 0 0 0 rgba(0,200,5,.6);animation:pulse 1.8s infinite;}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(0,200,5,.6)}70%{box-shadow:0 0 0 10px rgba(0,200,5,0)}100%{box-shadow:0 0 0 0 rgba(0,200,5,0)}}
.sub{margin-top:6px;color:var(--sub);font-size:13px;}
form{margin-top:28px;display:flex;flex-direction:column;gap:12px;}
input[type=password]{width:100%;padding:14px 16px;border:1px solid var(--border);border-radius:10px;font-size:15px;outline:none;transition:border .15s, box-shadow .15s;background:var(--card);color:var(--text);}
input[type=password]:focus{border-color:var(--text);box-shadow:0 0 0 3px rgba(13,17,23,.08);}
button{width:100%;padding:14px 16px;border:0;border-radius:10px;background:var(--text);color:#fff;font-size:15px;font-weight:600;cursor:pointer;transition:opacity .12s;}
button:hover{opacity:.92}
.err{color:var(--red);font-size:13px;min-height:18px;}
.foot{margin-top:18px;color:var(--sub);font-size:11px;text-align:center;letter-spacing:.04em;text-transform:uppercase;}
</style></head><body>
<div class="card">
  <div class="logo">CHAD<span class="dot"></span></div>
  <div class="sub">Capital Heuristic Autonomous Deployer</div>
  <form id="f" method="post" action="/login">
    <input type="password" name="password" id="pw" placeholder="Password" autocomplete="current-password" autofocus required/>
    <button type="submit">Sign in</button>
    <div class="err" id="err">__ERR__</div>
  </form>
  <div class="foot">Paper trading · v7.0</div>
</div>
</body></html>
"""


def _login_page(error: str = "") -> HTMLResponse:
    body = LOGIN_HTML.replace("__ERR__", error)
    return HTMLResponse(body, status_code=200 if not error else 401)


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.get("/login")
async def login_get(request: Request) -> HTMLResponse:
    return _login_page()


@app.post("/login")
async def login_post(request: Request) -> Response:
    _now = time.time()
    _client_ip = request.client.host if request.client else "unknown"
    _fail_entry = _LOGIN_FAILURES.get(_client_ip, (0, 0.0))
    if _fail_entry[1] > _now:
        return JSONResponse(
            {"error": "Too many failed attempts. Try again later."},
            status_code=429
        )
    body = await request.body()
    try:
        form = parse_qs(body.decode("utf-8", "replace"))
    except Exception:
        form = {}
    pw_list = form.get("password") or [""]
    pw = pw_list[0] if pw_list else ""
    if not secrets.compare_digest(pw, DASHBOARD_PASSWORD):
        _count = _LOGIN_FAILURES.get(_client_ip, (0, 0.0))[0] + 1
        _lockout = _now + _LOGIN_LOCKOUT_SECONDS if _count >= _LOGIN_MAX_ATTEMPTS else 0.0
        _LOGIN_FAILURES[_client_ip] = (_count, _lockout)
        return _login_page("Incorrect password")
    _LOGIN_FAILURES.pop(_client_ip, None)
    tok = _new_session_token()
    resp = Response(status_code=302, headers={"Location": "/"})
    resp.set_cookie(
        SESSION_COOKIE, tok,
        max_age=SESSION_TTL_SECONDS, httponly=True, samesite="lax", path="/",
    )
    return resp


@app.post("/logout")
@app.get("/logout")
async def logout(request: Request) -> Response:
    tok = request.cookies.get(SESSION_COOKIE)
    if tok:
        SESSIONS.pop(tok, None)
    resp = Response(status_code=302, headers={"Location": "/login"})
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp


@app.get("/")
async def root(request: Request) -> Response:
    if not _request_authenticated(request):
        return _login_page()
    return FileResponse(str(STATIC / "index.html"))


# ------------------------- Auth middleware for /api -------------------------

@app.middleware("http")
async def api_auth(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api/"):
        if not _request_authenticated(request):
            return JSONResponse(
                {"error": "unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="CHAD"'},
            )
    return await call_next(request)


# ------------------------- /api/state -------------------------

@app.get("/api/state")
async def api_state() -> JSONResponse:
    return JSONResponse(builder.build())


# ------------------------- /api/recent-trades -------------------------

_RECENT_TRADES_MAX_LINES_PER_FILE = 500


def _iter_recent_closed_trades(limit: int) -> list[dict]:
    """Prefer closed-trade history (has pnl); fall back to fills.

    D7: each history file is read with a per-file line cap
    (_RECENT_TRADES_MAX_LINES_PER_FILE = 500). Files larger than the cap
    are tail-read via deque(maxlen=500) — newest 500 lines retained,
    older lines discarded. Bounds memory at ~100KB per file regardless
    of history size.
    """
    trades_dir = DATA / "trades"
    fills_dir = DATA / "fills"
    out: list[dict] = []

    files: list[Path] = []
    if trades_dir.exists():
        files = sorted(
            [p for p in trades_dir.iterdir() if p.name.startswith("trade_history_") and p.suffix == ".ndjson"],
            reverse=True,
        )
    records: list[dict] = []
    for f in files[:5]:
        try:
            with open(f, "r") as fh:
                _tail: deque = deque(maxlen=_RECENT_TRADES_MAX_LINES_PER_FILE)
                for line in fh:
                    _tail.append(line)
                lines_to_process = list(_tail)
            for line in lines_to_process:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                p = obj.get("payload") or obj
                if not isinstance(p, dict):
                    continue
                records.append(p)
        except Exception:
            continue

    records.sort(key=lambda r: r.get("exit_time_utc") or r.get("fill_time_utc") or "", reverse=True)

    for p in records[:limit]:
        pnl = p.get("pnl")
        try:
            pnl_f = float(pnl) if pnl is not None else 0.0
        except Exception:
            pnl_f = 0.0
        if abs(pnl_f) < 1e-6:
            outcome = "flat"
        elif pnl_f > 0:
            outcome = "win"
        else:
            outcome = "loss"
        strat = p.get("strategy") or "unknown"
        out.append(
            {
                "symbol": p.get("symbol", "?"),
                "side": (p.get("side") or "").upper(),
                "qty": p.get("quantity"),
                "price": p.get("exit_price") or p.get("fill_price"),
                "strategy": strat,
                "strategy_plain": STRATEGY_NAMES.get(strat, strat.title()),
                "ts_utc": p.get("exit_time_utc") or p.get("fill_time_utc"),
                "pnl": round(pnl_f, 2),
                "pnl_label": _fmt_money_signed(pnl_f),
                "outcome": outcome,
            }
        )

    if out:
        return out

    # Fallback: recent fills (no PnL available)
    if fills_dir.exists():
        files = sorted(
            [p for p in fills_dir.iterdir() if p.name.startswith("FILLS_") and p.suffix == ".ndjson"],
            reverse=True,
        )
        recs: list[dict] = []
        for f in files[:3]:
            try:
                with open(f, "r") as fh:
                    _tail: deque = deque(maxlen=_RECENT_TRADES_MAX_LINES_PER_FILE)
                    for line in fh:
                        _tail.append(line)
                    lines_to_process = list(_tail)
                for line in lines_to_process:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    p = obj.get("payload") or obj
                    if not isinstance(p, dict):
                        continue
                    if p.get("status") != "paper_fill":
                        continue
                    recs.append(p)
            except Exception:
                continue
        recs.sort(key=lambda r: r.get("fill_time_utc") or "", reverse=True)
        for p in recs[:limit]:
            strat = p.get("strategy") or "unknown"
            out.append(
                {
                    "symbol": p.get("symbol", "?"),
                    "side": (p.get("side") or "").upper(),
                    "qty": p.get("quantity"),
                    "price": p.get("fill_price"),
                    "strategy": strat,
                    "strategy_plain": STRATEGY_NAMES.get(strat, strat.title()),
                    "ts_utc": p.get("fill_time_utc"),
                    "pnl": None,
                    "pnl_label": "—",
                    "outcome": "flat",
                }
            )
    return out


@app.get("/api/recent-trades")
async def api_recent_trades() -> JSONResponse:
    return JSONResponse({"trades": _iter_recent_closed_trades(10)})


# ------------------------- /api/leaderboard -------------------------

@app.get("/api/leaderboard")
async def api_leaderboard() -> JSONResponse:
    exp = _load_json(RUNTIME / "expectancy_state.json")
    caps = _load_json(RUNTIME / "dynamic_caps.json")
    weights = caps.get("normalized_weights") or {}
    strats = exp.get("strategies") or {}

    items = []
    for name, s in strats.items():
        total = int(s.get("total_trades") or 0)
        win_rate = float(s.get("win_rate") or 0.0)
        expectancy = float(s.get("expectancy") or 0.0)
        weight = float(weights.get(name) or 0.0)
        signals = int(round(weight * 20)) if weight > 0 else 0
        items.append(
            {
                "name": STRATEGY_NAMES.get(name, name.title()),
                "internal": name,
                "win_rate": round(win_rate, 3),
                "win_rate_pct": f"{int(round(win_rate * 100))}%",
                "total_trades": total,
                "expectancy": round(expectancy, 2),
                "expectancy_label": _fmt_money_signed(expectancy),
                "status": s.get("status", "new"),
                "signals_now": signals,
            }
        )

    ranked = sorted(
        [x for x in items if x["total_trades"] >= 10],
        key=lambda x: (-x["win_rate"], -x["total_trades"]),
    )
    building = sorted(
        [x for x in items if x["total_trades"] < 10],
        key=lambda x: -x["total_trades"],
    )
    for i, x in enumerate(ranked, 1):
        x["rank"] = i
    for x in building:
        x["rank"] = None

    return JSONResponse({"strategies": ranked + building, "ranked_count": len(ranked)})


# ------------------------- /api/market -------------------------

@app.get("/api/market")
async def api_market() -> JSONResponse:
    prices = _load_json(RUNTIME / "price_cache.json").get("prices") or {}
    ticks = _load_json(RUNTIME / "price_cache.json").get("ticks") or {}
    kraken = _load_json(RUNTIME / "kraken_prices.json") or {}
    kraken_prices = kraken.get("prices") or {}
    kraken_ticks = kraken.get("ticks") or {}
    trends = _load_json(RUNTIME / "trends_state.json").get("signals") or {}
    reddit = _load_json(RUNTIME / "reddit_sentiment.json").get("signals") or {}
    event_risk = _load_json(RUNTIME / "event_risk.json")
    regime_state = _load_json(RUNTIME / "regime_state.json")

    vix = prices.get("VIX")
    if vix is None:
        vixy = prices.get("VIXY")
        vix = round(vixy * 0.67, 2) if isinstance(vixy, (int, float)) else None
    vix_label, vix_color = _vix_label_color(vix)

    spy_price = prices.get("SPY")
    spy_change = None
    if isinstance(ticks, dict):
        t = ticks.get("SPY") if isinstance(ticks.get("SPY"), dict) else None
        if t:
            prev = t.get("prev_close") or t.get("open")
            if spy_price and prev:
                try:
                    spy_change = round((float(spy_price) - float(prev)) / float(prev) * 100, 2)
                except Exception:
                    spy_change = None

    btc_price = kraken_prices.get("BTC-USD") or prices.get("BTC")
    btc_change = None
    btc_tick = kraken_ticks.get("BTC-USD") if isinstance(kraken_ticks, dict) else None
    if isinstance(btc_tick, dict):
        prev = btc_tick.get("prev_close") or btc_tick.get("open_24h")
        if btc_price and prev:
            try:
                btc_change = round((float(btc_price) - float(prev)) / float(prev) * 100, 2)
            except Exception:
                btc_change = None

    regime_map = {
        "trending_bull": ("TRENDING BULL", "green"),
        "trending_bear": ("TRENDING BEAR", "red"),
        "volatile":      ("VOLATILE",      "orange"),
        "ranging":       ("RANGING",       "grey"),
        "neutral":       ("NEUTRAL",       "grey"),
    }
    regime_raw = str(regime_state.get("regime") or "neutral").lower()
    regime_label, regime_color = regime_map.get(regime_raw, (regime_raw.upper(), "grey"))

    mentions = sorted(
        [(k, int(v.get("mention_count") or 0)) for k, v in reddit.items() if isinstance(v, dict)],
        key=lambda x: -x[1],
    )
    top_mentions = [k for k, _ in mentions[:3]] or []

    now_utc = datetime.now(timezone.utc)
    is_open = _market_is_open(now_utc)

    return JSONResponse(
        {
            "vix": round(float(vix), 2) if isinstance(vix, (int, float)) else None,
            "vix_label": vix_label,
            "vix_color": vix_color,
            "spy_price": float(spy_price) if isinstance(spy_price, (int, float)) else None,
            "spy_change_pct": spy_change,
            "btc_price": float(btc_price) if isinstance(btc_price, (int, float)) else None,
            "btc_change_pct": btc_change,
            "market_regime": regime_label,
            "regime_color": regime_color,
            "top_mentions": top_mentions,
            "market_open": is_open,
            "market_open_label": "Market Open" if is_open else "Market Closed",
        }
    )


# ------------------------- WebSocket -------------------------

@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    if not _ws_authenticated(websocket):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    try:
        while True:
            state = builder.build()
            await websocket.send_json(state)
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        return
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass


@app.on_event("startup")
async def _startup() -> None:
    async def _bg():
        while True:
            try:
                await builder.refresh_ibkr_async()
            except Exception:
                pass
            await asyncio.sleep(30)

    asyncio.create_task(_bg())


# ------------------------- /api/chat -------------------------
#
# Claude-powered advisory chat. Reuses ClaudeClient for key loading
# (reads /etc/chad/claude.env) and builds a condensed CHAD state
# snapshot that is injected as a user turn before the operator's
# question. The endpoint is strictly advisory: no execution, no
# config mutation, no broker calls. Every failure mode returns a
# graceful fallback reply — the UI never sees a 5xx or raw error.

CHAT_SYSTEM_PROMPT = (
    "You are CHAD — an autonomous trading system explaining itself to its "
    "owner. Your owner is not a trader and does not know any financial "
    "terminology. Speak like you are explaining to a smart friend who has "
    "never traded a day in their life.\n\n"
    "STRICT LANGUAGE RULES — never break these:\n"
    "- No percentages. Say 'about 7 out of 10' instead of '71.7%'\n"
    "- No trading terms. Never say: Sharpe, regime, volatility, PnL, "
    "drawdown, momentum, bull, bear, intraday, basis points, alpha, "
    "beta, or any other finance word\n"
    "- No jargon replacements like 'sizing factor' or 'effective trades' "
    "— instead say 'how aggressively we trade' and 'practice runs'\n"
    "- Dollar amounts are fine. Plain numbers are fine.\n"
    "- Maximum 3 sentences unless they ask for more detail\n"
    "- Be warm, direct, and confident — like a smart friend, not a banker\n\n"
    "PLAIN ENGLISH TRANSLATIONS (always use these):\n"
    "- 'Warmup mode' → 'still in practice mode, not trading at full power yet'\n"
    "- 'trending bull market' → 'markets are going up right now'\n"
    "- 'trending bear market' → 'markets are going down right now'\n"
    "- 'volatile regime' → 'markets are jumpy and unpredictable right now'\n"
    "- 'ranging regime' → 'markets are flat and moving sideways right now'\n"
    "- 'SCR state' → 'how much we trust ourselves to trade right now'\n"
    "- 'sizing factor 0.1' → 'only using 10% of our normal trade size'\n"
    "- 'effective trades' → 'clean practice runs completed'\n"
    "- 'stop bus active' → 'we hit our safety limit and paused everything'\n"
    "- 'Sharpe-like score' → skip it entirely\n"
    "- 'win rate' → 'how often our trades make money'\n"
    "- 'realized PnL' → 'money made or lost today'\n"
    "- 'open positions' → 'trades currently open'\n"
    "- 'active strategies' → 'approaches we are using right now'\n\n"
    "HARD RULES — never violate:\n"
    "- Never say 'you should buy X' or 'sell X'\n"
    "- Never recommend specific trades\n"
    "- Never change any system configuration\n"
    "- Always explain what CHAD is doing in plain English\n"
    "- If asked about something outside CHAD, redirect politely"
)

CHAT_MODEL = "claude-sonnet-4-6"
CHAT_MAX_TOKENS = 400
CHAT_TEMPERATURE = 0.3
CHAT_HISTORY_MAX = 6  # last N messages accepted from client
CHAT_FALLBACK_REPLY = "CHAD is thinking... try again in a moment."

_chat_logger = logging.getLogger("chad.dashboard.chat")
_anthropic_client: Any = None
_anthropic_client_err: str | None = None


def _get_anthropic_client() -> Any:
    """Lazy-init anthropic client, reusing ClaudeClient's key loader."""
    global _anthropic_client, _anthropic_client_err
    if _anthropic_client is not None:
        return _anthropic_client
    try:
        from chad.intel.claude_client import ClaudeClient
        import anthropic
        api_key = ClaudeClient.load_key()
        _anthropic_client = anthropic.Anthropic(api_key=api_key, timeout=20.0)
        return _anthropic_client
    except Exception as exc:
        _anthropic_client_err = f"{type(exc).__name__}: {exc}"
        _chat_logger.warning("chat_client_init_failed err=%s", exc)
        return None


def _chat_context_snapshot() -> dict:
    """Compact CHAD state for Claude (~400 tokens). Summarise, don't dump."""
    scr = _load_json(RUNTIME / "scr_state.json")
    regime = _load_json(RUNTIME / "regime_state.json")
    pnl = _load_json(RUNTIME / "pnl_state.json")
    caps = _load_json(RUNTIME / "dynamic_caps.json")
    stop_bus = _load_json(RUNTIME / "stop_bus.json")
    price_cache = _load_json(RUNTIME / "price_cache.json").get("prices") or {}

    stats = scr.get("stats") or {}
    scr_block = {
        "state": scr.get("state"),
        "sizing_factor": scr.get("sizing_factor"),
        "effective_trades": stats.get("effective_trades"),
        "win_rate": round(float(stats.get("win_rate") or 0.0), 3),
        "sharpe_like": round(float(stats.get("sharpe_like") or 0.0), 3),
        "reasons": (scr.get("reasons") or [])[:2],
    }

    regime_block = {
        "regime": regime.get("regime"),
        "confidence": round(float(regime.get("confidence") or 0.0), 3),
    }

    portfolio_block = {
        "account_equity": pnl.get("account_equity"),
        "realized_pnl_today": pnl.get("realized_pnl"),
        "trade_count_today": pnl.get("trade_count"),
    }

    positions_raw = _load_json(RUNTIME / "position_guard.json")
    open_positions: list[dict] = []
    for key, rec in positions_raw.items():
        if not isinstance(rec, dict) or not rec.get("open"):
            continue
        open_positions.append(
            {
                "symbol": rec.get("symbol") or key.split("|")[-1],
                "side": rec.get("side"),
                "quantity": rec.get("quantity"),
                "strategy": rec.get("strategy"),
            }
        )
    open_positions = open_positions[:3]

    weights = caps.get("normalized_weights") or {}
    active_strategies = sorted(
        [(k, float(v or 0.0)) for k, v in weights.items() if float(v or 0.0) > 0.0],
        key=lambda x: -x[1],
    )[:5]
    active_strategy_names = [STRATEGY_NAMES.get(k, k) for k, _ in active_strategies]

    stop_block = {
        "active": bool(stop_bus.get("active")),
        "reason": stop_bus.get("reason") or "none",
    }

    vix = price_cache.get("VIX")
    if vix is None:
        try:
            bars = _load_json(DATA / "bars" / "1d" / "VIX.json").get("bars") or []
            if bars:
                vix = bars[-1].get("close")
        except Exception:
            vix = None

    # Business framework — phase / tier / salary so the chat can answer
    # "what phase are we in", "when do I get paid", "how many strategies
    # are active right now" without the operator having to dig.
    biz = _load_json(RUNTIME / "business_phase.json") or {}
    tier = _load_json(RUNTIME / "tier_state.json") or {}
    wd = _load_json(RUNTIME / "withdrawal_authorization.json") or {}
    booster = _load_json(RUNTIME / "regime_booster.json") or {}
    business_block = {
        "phase": biz.get("phase") or wd.get("phase"),
        "phase_description": biz.get("phase_description"),
        "tier": tier.get("tier_name"),
        "tier_strategies_enabled": len(tier.get("enabled_strategies") or []),
        "authorized_salary_usd": wd.get("authorized_withdrawal_usd"),
        "growth_pct_from_seed": biz.get("growth_pct_from_seed"),
        "next_phase_requirement": biz.get("next_phase_requirement"),
        "regime_booster_multiplier": booster.get("multiplier"),
        "regime_booster_active": booster.get("active"),
    }

    return {
        "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scr": scr_block,
        "regime": regime_block,
        "portfolio": portfolio_block,
        "open_positions": open_positions,
        "active_strategies": active_strategy_names,
        "stop_bus": stop_block,
        "vix": round(float(vix), 2) if isinstance(vix, (int, float)) else None,
        "business": business_block,
        "mode": "paper",
    }


def _sanitize_history(raw: Any) -> list[dict]:
    """Accept only valid {role, content} dicts; cap to CHAT_HISTORY_MAX."""
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw[-CHAT_HISTORY_MAX:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in ("user", "assistant"):
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        out.append({"role": role, "content": content[:2000]})
    return out


def _chat_call_claude(message: str, history: list[dict]) -> str:
    """Call Claude with context + history + current message. Always returns a str."""
    client = _get_anthropic_client()
    if client is None:
        _chat_logger.warning("chat_client_unavailable err=%s", _anthropic_client_err)
        return CHAT_FALLBACK_REPLY

    try:
        context = _chat_context_snapshot()
    except Exception as exc:
        _chat_logger.warning("chat_context_build_failed err=%s", exc)
        context = {"error": "context unavailable"}

    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                "Current CHAD system state:\n"
                + json.dumps(context, indent=2, default=str)
            ),
        },
        {
            "role": "assistant",
            "content": "Understood. I have the current state. What would you like to know?",
        },
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": message[:2000]})

    try:
        resp = client.messages.create(
            model=CHAT_MODEL,
            max_tokens=CHAT_MAX_TOKENS,
            temperature=CHAT_TEMPERATURE,
            system=CHAT_SYSTEM_PROMPT,
            messages=messages,
        )
        parts = []
        for block in resp.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        text = "\n".join(parts).strip()
        return text or CHAT_FALLBACK_REPLY
    except Exception as exc:
        _chat_logger.warning("chat_api_call_failed err=%s", exc)
        return CHAT_FALLBACK_REPLY


@app.post("/api/chat")
async def api_chat(request: Request) -> JSONResponse:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        body = await request.json()
    except Exception:
        body = {}
    message = (body.get("message") or "").strip() if isinstance(body, dict) else ""
    history = _sanitize_history(body.get("history") if isinstance(body, dict) else None)

    if not message:
        return JSONResponse({"reply": "Ask me anything about the system.", "context_ts": ts})

    loop = asyncio.get_running_loop()
    try:
        reply = await loop.run_in_executor(None, _chat_call_claude, message, history)
    except Exception as exc:
        _chat_logger.warning("chat_executor_failed err=%s", exc)
        reply = CHAT_FALLBACK_REPLY

    return JSONResponse({"reply": reply, "context_ts": ts})


@app.get("/api/chat/clear")
async def api_chat_clear() -> JSONResponse:
    # Server is stateless per request — history lives client-side.
    return JSONResponse({"ok": True})
