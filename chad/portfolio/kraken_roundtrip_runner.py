#!/usr/bin/env python3
"""
chad/portfolio/kraken_roundtrip_runner.py

CHAD Kraken Roundtrip Runner (24/7 micro execution, production-safe)

Purpose
-------
This runner performs a *small, controlled* Kraken crypto roundtrip to generate
real, traceable TradeResult entries that count toward SCR "effective_trades"
and can be enriched by the Kraken PnL Watcher.

It is designed for:
- wiring verification (end-to-end execution + ledger + enrichment)
- safe micro testing (tiny volume, strict balance floors)
- unattended operation (systemd timer) with concurrency lock + state

Hard Safety Guarantees
----------------------
- Does nothing live unless --live is explicitly provided.
- Enforces CAD_FLOOR and BTC_FLOOR before placing any LIVE order.
- Enforces Kraken minimum volume (ordermin) for the configured pair.
- Single-instance lock prevents overlapping executions.
- Writes no secrets, prints no secrets.

Notes
-----
- This runner uses KrakenExecutor, which logs a TradeResult entry per order.
- The Kraken PnL Watcher should run separately (timer) to enrich those records
  with cost/fee/price and compute realized PnL for sells (FIFO).

Typical usage
-------------
Validate-only (safe):
  python3 -m chad.portfolio.kraken_roundtrip_runner --once

Live canary (one roundtrip):
  python3 -m chad.portfolio.kraken_roundtrip_runner --once --live

Daemon-like loop (N roundtrips, live):
  python3 -m chad.portfolio.kraken_roundtrip_runner --roundtrips 5 --live

Recommended systemd timer cadence:
- Every 5–15 minutes (not every minute) to avoid fee churn.

"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# CHAD internal modules (already in your build)
from chad.execution.kraken_executor import KrakenExecutor, StrategyTradeIntent
from chad.execution.kraken_trade_router import KrakenTradeRouter
from chad.exchanges.kraken_client import KrakenClient, KrakenClientConfig, KrakenAPIError, KrakenConfigError


ROOT = Path("/home/ubuntu/CHAD FINALE")
RUNTIME = ROOT / "runtime"
LOCK_PATH = RUNTIME / "kraken_roundtrip.lock"
STATE_PATH = RUNTIME / "kraken_roundtrip_state.json"

KRAKEN_API_BASE = "https://api.kraken.com"
DEFAULT_PAIR = "XXBTZCAD"
DEFAULT_STRATEGY = "crypto"

# Conservative fee buffer (Kraken fees vary by tier; we use a conservative buffer for safety).
FEE_BUFFER_PCT = 0.01  # 1% buffer on estimated notional
# If Kraken ticker fails, we still allow validate-only, but live requires a price estimate.
PRICE_FALLBACK_CAD = 100_000.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _http_get_json(url: str, timeout_s: float = 10.0) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "CHAD/kraken_roundtrip_runner"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:  # noqa: S310 (we control URL)
        raw = r.read().decode("utf-8", errors="replace").strip()
    if not raw.startswith("{"):
        raise ValueError(f"NOT_JSON_BODY: {raw[:200]!r}")
    j = json.loads(raw)
    errs = j.get("error") or []
    if errs:
        raise ValueError(f"KRAKEN_API_ERROR: {errs}")
    return j


def _kraken_assetpairs(pair: str) -> Dict[str, Any]:
    url = f"{KRAKEN_API_BASE}/0/public/AssetPairs?pair={pair}"
    return _http_get_json(url)


def _kraken_ticker(pair: str) -> Dict[str, Any]:
    url = f"{KRAKEN_API_BASE}/0/public/Ticker?pair={pair}"
    return _http_get_json(url)


def _pair_min_volume(pair: str) -> float:
    j = _kraken_assetpairs(pair)
    info = (j.get("result") or {}).get(pair) or {}
    v = _safe_float(info.get("ordermin"), default=0.0)
    if v <= 0.0:
        raise ValueError(f"Could not read ordermin for pair={pair}: {info}")
    return v


def _pair_mid_price_cad(pair: str) -> float:
    """
    Return a mid-price estimate in CAD for the given pair using Kraken public ticker.

    This is used ONLY to estimate notional for risk gating and balance floors.
    """
    j = _kraken_ticker(pair)
    result = j.get("result") or {}
    # Kraken may return the requested key or an altname key; take first entry.
    if not result:
        raise ValueError(f"No ticker result for pair={pair}")
    info = next(iter(result.values()))
    ask = _safe_float((info.get("a") or [None])[0], default=0.0)
    bid = _safe_float((info.get("b") or [None])[0], default=0.0)
    if ask <= 0.0 and bid <= 0.0:
        raise ValueError(f"Ticker returned invalid bid/ask for pair={pair}: {info}")
    if ask > 0.0 and bid > 0.0:
        return (ask + bid) / 2.0
    return ask if ask > 0.0 else bid


@dataclass(frozen=True)
class Floors:
    cad_floor: float
    btc_floor: float


@dataclass(frozen=True)
class RunConfig:
    pair: str
    strategy: str
    volume: float
    live: bool
    max_roundtrips: int
    sleep_between_legs_s: float
    floors: Floors


@dataclass
class Event:
    ts_utc: str
    event: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunReport:
    ok: bool
    ts_utc: str
    config: Dict[str, Any]
    events: List[Event]
    error_type: Optional[str] = None
    error: Optional[str] = None


def _load_state() -> Dict[str, Any]:
    if not STATE_PATH.is_file():
        return {"last_first_leg": None, "updated_at_utc": None}
    try:
        return json.loads(_read_text(STATE_PATH))
    except Exception:
        return {"last_first_leg": None, "updated_at_utc": None}


def _save_state(state: Dict[str, Any]) -> None:
    state["updated_at_utc"] = _utc_now()
    _write_text_atomic(STATE_PATH, json.dumps(state, indent=2, sort_keys=True) + "\n")


def _acquire_lock() -> Any:
    """
    Best-effort single-instance lock.
    Uses a simple lock file with OS-level advisory lock where available.
    """
    RUNTIME.mkdir(parents=True, exist_ok=True)
    f = open(LOCK_PATH, "a+", encoding="utf-8")  # noqa: PTH123
    try:
        import fcntl  # Linux only

        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception as exc:
        f.close()
        raise RuntimeError(f"LOCK_BUSY: {exc}")
    f.seek(0)
    f.truncate(0)
    f.write(f"pid={os.getpid()} ts_utc={_utc_now()}\n")
    f.flush()
    return f


def _balances(client: KrakenClient) -> Tuple[float, float]:
    """
    Return (btc, cad) balances as floats.
    """
    b = client.balance()
    btc = _safe_float(b.get("XXBT"), default=0.0)  # Kraken uses XXBT for BTC
    cad = _safe_float(b.get("ZCAD"), default=0.0)
    return btc, cad


def _build_executor() -> KrakenExecutor:
    cfg = KrakenClientConfig.from_env()
    client = KrakenClient(cfg)
    router = KrakenTradeRouter(client)
    return KrakenExecutor(router=router)


def _place_order(
    *,
    executor: KrakenExecutor,
    pair: str,
    strategy: str,
    side: str,
    volume: float,
    notional_estimate: float,
    live: bool,
) -> Dict[str, Any]:
    """
    Place/validate a market order via KrakenExecutor.
    Returns a compact dict with risk + response.
    """
    intent = StrategyTradeIntent(
        strategy=strategy,
        pair=pair,
        side=side,
        ordertype="market",
        volume=float(volume),
        notional_estimate=float(notional_estimate),
        price=None,
    )
    risk, resp = executor.execute_with_risk(intent=intent, live=live)

    out: Dict[str, Any] = {
        "side": side,
        "volume": float(volume),
        "notional_estimate": float(notional_estimate),
        "risk_allowed": bool(risk.allowed),
        "risk_reason": str(risk.reason),
        "validate_only": (not live),
        "txids": [],
    }
    if resp is not None:
        out["txids"] = list(resp.txids)
    return out


def _estimated_notional_cad(mid_price: float, volume: float) -> float:
    return float(mid_price * float(volume))


def _can_live_buy(*, cad: float, est_notional: float, floors: Floors) -> bool:
    # Require cad >= floor + est_notional + buffer
    need = floors.cad_floor + est_notional * (1.0 + FEE_BUFFER_PCT)
    return cad >= need


def _can_live_sell(*, btc: float, volume: float, floors: Floors) -> bool:
    # Require btc - volume >= btc_floor
    return (btc - volume) >= floors.btc_floor


def _decide_first_leg(
    *,
    btc: float,
    cad: float,
    volume: float,
    est_notional: float,
    floors: Floors,
    state: Dict[str, Any],
) -> str:
    """
    Decide first leg direction: "buy" or "sell".
    Preference:
      - If can't buy safely but can sell -> sell first
      - If can't sell safely but can buy -> buy first
      - Else alternate with state to avoid bias
    """
    can_buy = _can_live_buy(cad=cad, est_notional=est_notional, floors=floors)
    can_sell = _can_live_sell(btc=btc, volume=volume, floors=floors)

    if not can_buy and can_sell:
        return "sell"
    if not can_sell and can_buy:
        return "buy"

    last = (state.get("last_first_leg") or "").lower()
    if last == "buy":
        return "sell"
    if last == "sell":
        return "buy"

    # Default (deterministic-ish): pick sell if btc is relatively "high", else buy
    return "sell" if btc >= (floors.btc_floor + 2.0 * volume) else "buy"


def run_roundtrip(cfg: RunConfig) -> RunReport:
    ts = _utc_now()
    events: List[Event] = []
    try:
        _lock = _acquire_lock()
        events.append(Event(ts_utc=_utc_now(), event="lock_acquired", data={"lock_path": str(LOCK_PATH)}))

        # Build executor (private client inside)
        executor = _build_executor()
        client = executor._router._client  # noqa: SLF001 (internal reference, stable in this build)

        # Determine Kraken minimum volume and ensure cfg.volume respects it
        min_vol = _pair_min_volume(cfg.pair)
        events.append(Event(ts_utc=_utc_now(), event="pair_min_volume", data={"pair": cfg.pair, "ordermin": min_vol}))

        if cfg.volume + 1e-12 < min_vol:
            raise ValueError(f"volume {cfg.volume} is below Kraken ordermin {min_vol} for pair {cfg.pair}")

        # Price estimate (needed for floors & risk gating)
        try:
            mid = _pair_mid_price_cad(cfg.pair)
        except Exception as exc:
            mid = PRICE_FALLBACK_CAD
            events.append(
                Event(
                    ts_utc=_utc_now(),
                    event="ticker_failed_using_fallback",
                    data={"pair": cfg.pair, "fallback_price_cad": mid, "error": f"{type(exc).__name__}: {exc}"},
                )
            )
        else:
            events.append(Event(ts_utc=_utc_now(), event="ticker_mid_price", data={"pair": cfg.pair, "mid_price_cad": mid}))

        est_notional = _estimated_notional_cad(mid, cfg.volume)
        events.append(Event(ts_utc=_utc_now(), event="notional_estimate", data={"est_notional_cad": est_notional}))

        # Balances
        btc, cad = _balances(client)
        events.append(Event(ts_utc=_utc_now(), event="balances_before", data={"XXBT": btc, "ZCAD": cad}))

        # Decide first leg
        state = _load_state()
        first = _decide_first_leg(btc=btc, cad=cad, volume=cfg.volume, est_notional=est_notional, floors=cfg.floors, state=state)
        second = "buy" if first == "sell" else "sell"
        events.append(Event(ts_utc=_utc_now(), event="roundtrip_plan", data={"first_leg": first, "second_leg": second, "live": cfg.live}))

        # Safety checks for LIVE
        if cfg.live:
            if first == "buy" and not _can_live_buy(cad=cad, est_notional=est_notional, floors=cfg.floors):
                raise RuntimeError(
                    f"LIVE_BLOCKED: insufficient CAD. cad={cad:.4f} need>={cfg.floors.cad_floor + est_notional*(1.0+FEE_BUFFER_PCT):.4f}"
                )
            if first == "sell" and not _can_live_sell(btc=btc, volume=cfg.volume, floors=cfg.floors):
                raise RuntimeError(
                    f"LIVE_BLOCKED: insufficient BTC. btc={btc:.8f} need>={cfg.floors.btc_floor + cfg.volume:.8f}"
                )

        # Place first leg
        r1 = _place_order(
            executor=executor,
            pair=cfg.pair,
            strategy=cfg.strategy,
            side=first,
            volume=cfg.volume,
            notional_estimate=est_notional,
            live=cfg.live,
        )
        events.append(Event(ts_utc=_utc_now(), event="leg1_result", data=r1))
        if not r1.get("risk_allowed", False):
            raise RuntimeError(f"RISK_GATE_BLOCKED leg1: {r1.get('risk_reason')}")

        # Wait between legs (let balances/positions settle)
        if cfg.sleep_between_legs_s > 0:
            time.sleep(cfg.sleep_between_legs_s)

        # Refresh balances after first leg (especially important for LIVE)
        btc2, cad2 = _balances(client)
        events.append(Event(ts_utc=_utc_now(), event="balances_after_leg1", data={"XXBT": btc2, "ZCAD": cad2}))

        # Safety checks for second leg LIVE
        if cfg.live:
            # recompute notional with latest mid for realism (best-effort)
            try:
                mid2 = _pair_mid_price_cad(cfg.pair)
            except Exception:
                mid2 = mid
            est2 = _estimated_notional_cad(mid2, cfg.volume)

            if second == "buy" and not _can_live_buy(cad=cad2, est_notional=est2, floors=cfg.floors):
                raise RuntimeError(
                    f"LIVE_BLOCKED leg2 buy: insufficient CAD. cad={cad2:.4f} need>={cfg.floors.cad_floor + est2*(1.0+FEE_BUFFER_PCT):.4f}"
                )
            if second == "sell" and not _can_live_sell(btc=btc2, volume=cfg.volume, floors=cfg.floors):
                raise RuntimeError(
                    f"LIVE_BLOCKED leg2 sell: insufficient BTC. btc={btc2:.8f} need>={cfg.floors.btc_floor + cfg.volume:.8f}"
                )

        # Place second leg
        # refresh mid estimate lightly to avoid stale notional estimate
        try:
            mid_final = _pair_mid_price_cad(cfg.pair)
        except Exception:
            mid_final = mid
        est_final = _estimated_notional_cad(mid_final, cfg.volume)

        r2 = _place_order(
            executor=executor,
            pair=cfg.pair,
            strategy=cfg.strategy,
            side=second,
            volume=cfg.volume,
            notional_estimate=est_final,
            live=cfg.live,
        )
        events.append(Event(ts_utc=_utc_now(), event="leg2_result", data=r2))
        if not r2.get("risk_allowed", False):
            raise RuntimeError(f"RISK_GATE_BLOCKED leg2: {r2.get('risk_reason')}")

        # Persist alternation state
        state["last_first_leg"] = first
        _save_state(state)
        events.append(Event(ts_utc=_utc_now(), event="state_saved", data={"state_path": str(STATE_PATH), "last_first_leg": first}))

        # Final balances (best-effort)
        btc3, cad3 = _balances(client)
        events.append(Event(ts_utc=_utc_now(), event="balances_after_leg2", data={"XXBT": btc3, "ZCAD": cad3}))

        # Release lock by closing file
        try:
            _lock.close()
        except Exception:
            pass

        return RunReport(
            ok=True,
            ts_utc=ts,
            config=asdict(cfg),
            events=events,
        )

    except (KrakenAPIError, KrakenConfigError) as exc:
        return RunReport(
            ok=False,
            ts_utc=ts,
            config=asdict(cfg),
            events=events + [Event(ts_utc=_utc_now(), event="kraken_error", data={"error": str(exc)})],
            error_type=type(exc).__name__,
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return RunReport(
            ok=False,
            ts_utc=ts,
            config=asdict(cfg),
            events=events + [Event(ts_utc=_utc_now(), event="error", data={"error": f"{type(exc).__name__}: {exc}"})],
            error_type=type(exc).__name__,
            error=str(exc),
        )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CHAD Kraken micro roundtrip runner (safe by default).")
    p.add_argument("--pair", default=DEFAULT_PAIR, help="Kraken pair, e.g. XXBTZCAD.")
    p.add_argument("--strategy", default=DEFAULT_STRATEGY, help="Strategy key used for caps + TradeResult logging (must be in dynamic_caps.json).")
    p.add_argument("--volume", type=float, default=0.00005, help="Base volume per order. Must be >= Kraken ordermin.")
    p.add_argument("--live", action="store_true", help="Actually place live orders. If omitted, validate-only.")
    p.add_argument("--once", action="store_true", help="Run exactly 1 roundtrip.")
    p.add_argument("--roundtrips", type=int, default=1, help="Number of roundtrips to run (ignored if --once is set).")
    p.add_argument("--sleep-between-legs", type=float, default=2.0, help="Sleep seconds between leg1 and leg2.")
    p.add_argument("--cad-floor", type=float, default=20.0, help="Hard CAD floor. Runner refuses LIVE if it would dip below this.")
    p.add_argument("--btc-floor", type=float, default=0.0, help="Hard BTC floor. Runner refuses LIVE if it would dip below this.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    cfg = RunConfig(
        pair=str(args.pair).strip(),
        strategy=str(args.strategy).strip(),
        volume=float(args.volume),
        live=bool(args.live),
        max_roundtrips=1 if bool(args.once) else int(args.roundtrips),
        sleep_between_legs_s=float(args.sleep_between_legs),
        floors=Floors(cad_floor=float(args.cad_floor), btc_floor=float(args.btc_floor)),
    )

    # Small jitter to reduce herd collisions if this is timer-driven.
    time.sleep(random.uniform(0.0, 0.5))

    all_reports: List[Dict[str, Any]] = []
    ok_all = True

    for i in range(cfg.max_roundtrips):
        rep = run_roundtrip(cfg)
        all_reports.append(
            {
                "ok": rep.ok,
                "ts_utc": rep.ts_utc,
                "config": rep.config,
                "error_type": rep.error_type,
                "error": rep.error,
                "events": [dict(ts_utc=e.ts_utc, event=e.event, data=e.data) for e in rep.events],
            }
        )

        print(json.dumps(all_reports[-1], indent=2, sort_keys=True))

        if not rep.ok:
            ok_all = False
            break

        # Small pause between roundtrips (avoid fee-churn burst)
        if i + 1 < cfg.max_roundtrips:
            time.sleep(2.0)

    return 0 if ok_all else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
