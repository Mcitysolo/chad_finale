"""CHAD real-time dashboard — FastAPI + WebSocket on port 8765.

Serves a plain-English dashboard that reflects CHAD's current
status, portfolio, open positions, strategy activity, and system
health. All runtime state files are read defensively; missing
files return sensible fallbacks rather than crashing.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

REPO = Path(__file__).resolve().parents[2]
RUNTIME = REPO / "runtime"
STATIC = Path(__file__).resolve().parent / "static"

STRATEGY_NAMES = {
    "alpha": "Stock Strategy",
    "beta": "Trend Strategy",
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


def _fmt_money(x: float | int | None) -> str:
    if x is None:
        return "—"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.0f}"


def _sharpe_to_score(sharpe: float | None) -> int:
    if sharpe is None:
        return 0
    # Piecewise linear: -1->0, 0->20, 0.3->40, 0.7->75, 1.0+->100
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


def _vix_label(v: float | None) -> str:
    if v is None:
        return "Unknown"
    if v < 15:
        return f"Calm ({v:.0f})"
    if v < 22:
        return f"Steady ({v:.0f})"
    if v < 30:
        return f"Nervous ({v:.0f})"
    return f"Fearful ({v:.0f})"


class StateBuilder:
    """Builds the dashboard state dict. Caches portions that are
    expensive to recompute (IBKR, systemctl)."""

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
        """Best-effort IBKR positions pull. Short timeout; never blocks event loop long."""
        try:
            from ib_insync import IB  # type: ignore
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
        equity = pnl.get("account_equity")
        realized = pnl.get("realized_pnl", 0.0)
        realized_label = (
            f"Up ${abs(realized):,.0f} realized"
            if realized > 0
            else f"Down ${abs(realized):,.0f} realized"
            if realized < 0
            else "Flat"
        )
        out = {
            "account_value": equity,
            "account_value_label": _fmt_money(equity),
            "realized_pnl": round(realized, 2),
            "realized_pnl_label": realized_label,
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
        return {
            "mode": mode,
            "mode_detail": detail,
            "sizing_label": sizing_label,
            "trades_completed": trades,
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
            # Signals now: approximate from allocation weight (>0 means active in rotation)
            weight = float(weights.get(name) or 0.0)
            signals = int(round(weight * 20)) if weight > 0 else 0
            if signals > 0:
                status = "active"
                status_label = f"Watching {signals} opportunities"
            else:
                status = "idle"
                status_label = "Waiting for the right moment"
            out.append(
                {
                    "name": STRATEGY_NAMES.get(name, name.title()),
                    "internal_name": name,
                    "signals_now": signals,
                    "status": status,
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
                    ["systemctl", "is-active", svc],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                state = (r.stdout or "").strip()
                if state in ("active", "activating"):
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
        # VIX not in price cache; try VIXY as proxy or leave unknown
        vix = prices.get("VIX")
        if vix is None:
            # Rough proxy: VIXY ETF ~ 28 when VIX ~ 19
            vixy = prices.get("VIXY")
            vix = round(vixy * 0.67, 2) if isinstance(vixy, (int, float)) else None
        btc = prices.get("BTC") or prices.get("BTCUSD")
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
            "vix_label": _vix_label(vix),
            "btc_price": btc,
            "spy_trend": spy_signal,
            "top_reddit_mention": top_reddit,
            "market_regime": regime,
        }

    def build(self) -> dict[str, Any]:
        scr = _load_json(RUNTIME / "scr_state.json")
        # IBKR snapshot in a background refresh (cheap miss: skip if cache fresh)
        connected = self._ibkr_connected
        return {
            "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ibkr_connected": connected,
            "chad_status": self._chad_status(scr),
            "portfolio": self._portfolio(),
            "open_positions": self._open_positions(),
            "strategies": self._strategies(),
            "system_health": self._system_health(),
            "intelligence": self._intelligence(),
        }

    async def refresh_ibkr_async(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._ibkr_snapshot)
        except Exception:
            pass


builder = StateBuilder()
app = FastAPI(title="CHAD Dashboard", version="1.0")


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.get("/api/state")
async def api_state() -> JSONResponse:
    return JSONResponse(builder.build())


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(str(STATIC / "index.html"))


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
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
