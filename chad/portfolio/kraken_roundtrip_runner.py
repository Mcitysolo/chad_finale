#!/usr/bin/env python3
"""
CHAD Kraken Roundtrip Runner — SSOT-safe micro roundtrip harness (validate-only works with zero secrets).

This script is designed to be *impossible* to use as a LiveGate bypass.

Key guarantees
--------------
1) SSOT / LiveGate authority:
   - If --live is requested, LiveGate MUST explicitly return ALLOW_LIVE or we exit cleanly.
   - If LiveGate cannot be imported/evaluated, we fail-closed for --live.

2) Validate-only (default) requires zero Kraken private env vars:
   - Uses only Kraken public endpoints (Ticker, AssetPairs).
   - Produces a deterministic "planned orders" report and writes it to runtime.
   - Does not construct Kraken clients, does not touch secrets.

3) Operational safety:
   - Single-instance lock file using fcntl (Linux) with clear LOCK_BUSY semantics.
   - Atomic writes for state + last report (write tmp then rename).
   - No secrets printed; errors are sanitized and concise.

Execution model
---------------
- validate-only:
    - fetch ordermin + mid price
    - build deterministic "roundtrip plan" (buy then sell OR sell then buy)
    - write runtime/kraken_roundtrip_last_report.json

- live:
    - LiveGate ALLOW_LIVE required
    - requires Kraken private env vars (KRAKEN_API_KEY / KRAKEN_API_SECRET) via CHAD Kraken modules
    - runs a micro roundtrip using KrakenExecutor.execute_with_risk()
    - writes runtime/kraken_roundtrip_last_report.json

Recommended invocation (repo root + venv)
-----------------------------------------
cd "/home/ubuntu/chad_finale"
source venv/bin/activate

Validate-only:
  python3 -m chad.portfolio.kraken_roundtrip_runner --once

Live (only when LiveGate is ALLOW_LIVE):
  python3 -m chad.portfolio.kraken_roundtrip_runner --once --live
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Tuple


# ---------------------------
# Defaults / constants
# ---------------------------

KRAKEN_API_BASE = "https://api.kraken.com"
DEFAULT_PAIR = "XXBTZCAD"
DEFAULT_STRATEGY = "crypto"

HTTP_TIMEOUT_S = 10.0
HTTP_MAX_RETRIES = 3

FEE_BUFFER_PCT = 0.01
PRICE_FALLBACK_CAD = 100_000.0  # fallback is allowed for validate-only, blocked for live by default

LOCK_FILENAME = "kraken_roundtrip.lock"
STATE_FILENAME = "kraken_roundtrip_state.json"
REPORT_FILENAME = "kraken_roundtrip_last_report.json"


# ---------------------------
# Small utilities
# ---------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        j = json.loads(raw)
        return j if isinstance(j, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _backoff_s(attempt: int) -> float:
    base = 0.4 * (2 ** max(0, attempt - 1))
    jitter = random.random() * 0.25
    return float(min(3.0, base + jitter))


def _http_get_json(url: str, *, timeout_s: float = HTTP_TIMEOUT_S) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "CHAD/kraken_roundtrip_runner"})
    last_err: Optional[str] = None

    for i in range(1, HTTP_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as r:  # noqa: S310
                raw = r.read().decode("utf-8", errors="replace").strip()

            if not raw.startswith("{"):
                raise ValueError(f"NOT_JSON_BODY: {raw[:200]!r}")

            j = json.loads(raw)
            if not isinstance(j, dict):
                raise ValueError("NON_OBJECT_JSON")

            errs = j.get("error") or []
            if errs:
                raise ValueError(f"KRAKEN_API_ERROR: {errs}")

            return j

        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            if i < HTTP_MAX_RETRIES:
                time.sleep(_backoff_s(i))
                continue
            raise RuntimeError(f"HTTP_GET_FAILED url={url!r} err={last_err}") from exc

    raise RuntimeError(f"HTTP_GET_FAILED url={url!r} err={last_err}")


# ---------------------------
# Domain objects
# ---------------------------

@dataclass(frozen=True)
class Event:
    ts_utc: str
    event: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Floors:
    cad_floor: float
    btc_floor: float


@dataclass(frozen=True)
class RunConfig:
    pair: str
    strategy: str
    volume: float
    roundtrips: int
    live: bool
    sleep_s: float
    cad_floor: float
    btc_floor: float
    allow_live_with_fallback_price: bool
    runtime_dir: Path


@dataclass(frozen=True)
class RunReport:
    ts_utc: str
    live_requested: bool
    live_allowed: bool
    pair: str
    strategy: str
    volume: float
    roundtrips: int
    events: List[Event]
    orders: List[Dict[str, Any]]
    summary: Dict[str, Any]


# ---------------------------
# Single-instance lock
# ---------------------------

class _FileLock:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._fh = None

    def acquire(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self._path, "a+", encoding="utf-8")  # noqa: PTH123
        try:
            import fcntl  # Linux only
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception as exc:
            fh.close()
            raise RuntimeError(f"LOCK_BUSY: {type(exc).__name__}: {exc}") from exc

        fh.seek(0)
        fh.truncate(0)
        fh.write(f"pid={os.getpid()} ts_utc={_utc_now()}\n")
        fh.flush()
        self._fh = fh

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            import fcntl
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass
        self._fh = None


# ---------------------------
# LiveGate (SSOT authority)
# ---------------------------

@dataclass(frozen=True)
class LiveGateSnapshot:
    mode: str
    reasons: List[str]


def _livegate_snapshot_failclosed() -> LiveGateSnapshot:
    try:
        from chad.core.live_gate import evaluate_live_gate  # type: ignore
    except Exception as exc:
        return LiveGateSnapshot(mode="DENY_ALL", reasons=[f"livegate_import_failed: {type(exc).__name__}"])

    try:
        d = evaluate_live_gate()
        mode = str(getattr(d, "mode", "DENY_ALL")).upper().strip()
        rr = getattr(d, "reasons", None)
        reasons: List[str] = []
        if isinstance(rr, list):
            reasons = [str(x) for x in rr[:50]]
        return LiveGateSnapshot(mode=mode, reasons=reasons)
    except Exception as exc:
        return LiveGateSnapshot(mode="DENY_ALL", reasons=[f"livegate_eval_failed: {type(exc).__name__}: {exc}"])


def _require_allow_live(*, live_requested: bool) -> Tuple[bool, List[str]]:
    if not live_requested:
        return True, ["validate_only: live not requested"]

    snap = _livegate_snapshot_failclosed()
    if snap.mode != "ALLOW_LIVE":
        return False, ["LiveGate denied live execution", f"mode={snap.mode}", *snap.reasons]
    return True, ["LiveGate allow_live", *snap.reasons]


# ---------------------------
# Kraken public market data
# ---------------------------

def _pair_min_volume(pair: str) -> float:
    url = f"{KRAKEN_API_BASE}/0/public/AssetPairs?pair={pair}"
    j = _http_get_json(url)
    res = j.get("result") or {}
    if not isinstance(res, dict) or not res:
        raise RuntimeError("assetpairs_result_empty")
    first = next(iter(res.values()))
    if not isinstance(first, dict):
        raise RuntimeError("assetpairs_bad_shape")
    ordermin = _safe_float(first.get("ordermin"), default=0.0)
    if ordermin <= 0:
        raise RuntimeError("ordermin_missing")
    return float(ordermin)


def _pair_mid_price_cad(pair: str) -> Tuple[float, bool]:
    """
    Returns (mid_price, is_fallback).
    """
    url = f"{KRAKEN_API_BASE}/0/public/Ticker?pair={pair}"
    try:
        j = _http_get_json(url)
        res = j.get("result") or {}
        if not isinstance(res, dict) or not res:
            raise RuntimeError("ticker_result_empty")
        first = next(iter(res.values()))
        if not isinstance(first, dict):
            raise RuntimeError("ticker_bad_shape")
        ask = _safe_float((first.get("a") or [None])[0], default=0.0)
        bid = _safe_float((first.get("b") or [None])[0], default=0.0)
        if ask <= 0 or bid <= 0:
            raise RuntimeError("ticker_missing_bid_ask")
        return float((ask + bid) / 2.0), False
    except Exception:
        return float(PRICE_FALLBACK_CAD), True


# ---------------------------
# Execution DI boundary (live only)
# ---------------------------

@dataclass(frozen=True)
class StrategyTradeIntent:
    strategy: str
    pair: str
    side: str          # "buy" | "sell"
    ordertype: str     # "market"
    volume: float
    notional_estimate: float
    price: Optional[float] = None


class KrakenExecutorPort(Protocol):
    def execute_with_risk(self, *, intent: StrategyTradeIntent, live: bool) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
        ...


class KrakenExecutionError(RuntimeError):
    pass


def _build_executor() -> KrakenExecutorPort:
    """
    Live-only: requires CHAD Kraken modules + private env vars.
    We deliberately do NOT call this on validate-only runs.
    """
    try:
        from chad.execution.kraken_executor import KrakenExecutor as RealExec  # type: ignore
        from chad.execution.kraken_executor import StrategyTradeIntent as RealIntent  # type: ignore
        from chad.execution.kraken_trade_router import KrakenTradeRouter  # type: ignore
        from chad.exchanges.kraken_client import KrakenClient, KrakenClientConfig  # type: ignore
    except Exception as exc:
        raise KrakenExecutionError(f"Kraken modules unavailable: {type(exc).__name__}: {exc}") from exc

    cfg = KrakenClientConfig.from_env()
    client = KrakenClient(cfg)
    router = KrakenTradeRouter(client)
    real = RealExec(router=router)

    class _Adapter:
        def execute_with_risk(self, *, intent: StrategyTradeIntent, live: bool) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
            ri = RealIntent(
                strategy=intent.strategy,
                pair=intent.pair,
                side=intent.side,
                ordertype=intent.ordertype,
                volume=float(intent.volume),
                notional_estimate=float(intent.notional_estimate),
                price=intent.price,
            )
            risk, resp = real.execute_with_risk(intent=ri, live=live)

            risk_m = (
                asdict(risk)
                if hasattr(risk, "__dataclass_fields__")
                else dict(risk.__dict__) if hasattr(risk, "__dict__")
                else {"allowed": getattr(risk, "allowed", False), "reason": getattr(risk, "reason", "")}
            )
            resp_m = (
                asdict(resp)
                if (resp is not None and hasattr(resp, "__dataclass_fields__"))
                else dict(resp.__dict__) if (resp is not None and hasattr(resp, "__dict__"))
                else {"txids": getattr(resp, "txids", [])} if resp is not None
                else {}
            )
            return risk_m, resp_m

    return _Adapter()


def _place_order(
    *,
    executor: KrakenExecutorPort,
    pair: str,
    strategy: str,
    side: str,
    volume: float,
    notional_estimate: float,
    live: bool,
) -> Dict[str, Any]:
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

    allowed = bool(risk.get("allowed", False))
    reason = str(risk.get("reason", ""))

    txids: List[str] = []
    raw = resp.get("txids") if isinstance(resp, Mapping) else None
    if isinstance(raw, list):
        txids = [str(x) for x in raw[:20]]

    return {
        "ts_utc": _utc_now(),
        "pair": pair,
        "strategy": strategy,
        "side": side,
        "volume": float(volume),
        "notional_estimate": float(notional_estimate),
        "validate_only": (not live),
        "risk_allowed": allowed,
        "risk_reason": reason,
        "txids": txids,
    }


# ---------------------------
# Business rules
# ---------------------------

def _estimated_notional(mid_price: float, volume: float) -> float:
    return float(mid_price * float(volume))


def _can_live_buy(*, cad: float, est_notional: float, floors: Floors) -> bool:
    need = floors.cad_floor + est_notional * (1.0 + FEE_BUFFER_PCT)
    return cad >= need


def _can_live_sell(*, btc: float, volume: float, floors: Floors) -> bool:
    return (btc - volume) >= floors.btc_floor


def _decide_first_leg(*, state: Dict[str, Any], floors: Floors, est_notional: float, volume: float) -> str:
    """
    Deterministic leg selection for validate-only (no balance probe):
      - Alternate buy/sell using persisted state, else default to buy.
    Live mode will still respect risk checks inside KrakenExecutor.
    """
    last = str(state.get("last_first_leg", "")).lower().strip()
    if last == "buy":
        return "sell"
    if last == "sell":
        return "buy"
    return "buy"


# ---------------------------
# Core runner
# ---------------------------

def run_roundtrip(cfg: RunConfig) -> RunReport:
    ts0 = _utc_now()
    events: List[Event] = []
    orders: List[Dict[str, Any]] = []

    cfg.runtime_dir.mkdir(parents=True, exist_ok=True)
    lock = _FileLock(cfg.runtime_dir / LOCK_FILENAME)

    state_path = cfg.runtime_dir / STATE_FILENAME
    state = _read_json(state_path)

    floors = Floors(cad_floor=float(cfg.cad_floor), btc_floor=float(cfg.btc_floor))
    events.append(Event(ts_utc=_utc_now(), event="config_loaded", data={
        "pair": cfg.pair,
        "strategy": cfg.strategy,
        "volume": cfg.volume,
        "roundtrips": cfg.roundtrips,
        "live": cfg.live,
        "runtime_dir": str(cfg.runtime_dir),
        "floors": asdict(floors),
    }))

    live_allowed, live_reasons = _require_allow_live(live_requested=cfg.live)
    events.append(Event(ts_utc=_utc_now(), event="livegate_evaluated", data={
        "live_requested": cfg.live,
        "live_allowed": live_allowed,
        "reasons": live_reasons,
    }))

    if cfg.live and not live_allowed:
        report = RunReport(
            ts_utc=ts0,
            live_requested=True,
            live_allowed=False,
            pair=cfg.pair,
            strategy=cfg.strategy,
            volume=cfg.volume,
            roundtrips=cfg.roundtrips,
            events=events,
            orders=[],
            summary={"status": "blocked", "blocked_by": "LiveGate", "reasons": live_reasons},
        )
        _write_text_atomic(cfg.runtime_dir / REPORT_FILENAME, _json_dumps(asdict(report)) + "\n")
        return report

    # Acquire lock only after LiveGate decision so we never lock-block someone just checking status.
    lock.acquire()
    events.append(Event(ts_utc=_utc_now(), event="lock_acquired", data={"path": str(cfg.runtime_dir / LOCK_FILENAME)}))

    try:
        # Public checks
        min_vol = _pair_min_volume(cfg.pair)
        events.append(Event(ts_utc=_utc_now(), event="pair_min_volume", data={"pair": cfg.pair, "ordermin": min_vol}))

        if cfg.volume + 1e-12 < min_vol:
            raise SystemExit(f"volume {cfg.volume} below Kraken ordermin {min_vol} for {cfg.pair}")

        mid, is_fallback = _pair_mid_price_cad(cfg.pair)
        events.append(Event(ts_utc=_utc_now(), event="ticker_mid_price", data={
            "pair": cfg.pair,
            "mid_price_cad": mid,
            "is_fallback": is_fallback,
        }))

        if cfg.live and (not cfg.allow_live_with_fallback_price) and is_fallback:
            raise SystemExit("LIVE_REQUIRES_REAL_PRICE: ticker failed and fallback is disallowed for live")

        est_notional = _estimated_notional(mid, cfg.volume)
        events.append(Event(ts_utc=_utc_now(), event="notional_estimated", data={"mid_price_cad": mid, "est_notional_cad": est_notional}))

        first_leg = _decide_first_leg(state=state, floors=floors, est_notional=est_notional, volume=cfg.volume)
        second_leg = "sell" if first_leg == "buy" else "buy"
        events.append(Event(ts_utc=_utc_now(), event="legs_decided", data={"first": first_leg, "second": second_leg}))

        # Validate-only: no secrets required, just emit a deterministic plan + report.
        if not cfg.live:
            planned = [
                {"side": first_leg, "volume": float(cfg.volume), "notional_estimate": float(est_notional), "validate_only": True},
                {"side": second_leg, "volume": float(cfg.volume), "notional_estimate": float(est_notional), "validate_only": True},
            ]
            events.append(Event(ts_utc=_utc_now(), event="validate_only_plan", data={"planned_orders": planned}))

            state["last_first_leg"] = first_leg
            state["last_run_ts_utc"] = _utc_now()
            _write_text_atomic(state_path, _json_dumps(state) + "\n")

            report = RunReport(
                ts_utc=ts0,
                live_requested=False,
                live_allowed=True,
                pair=cfg.pair,
                strategy=cfg.strategy,
                volume=cfg.volume,
                roundtrips=cfg.roundtrips,
                events=events,
                orders=planned,
                summary={
                    "status": "ok",
                    "mode": "validate_only",
                    "orders_planned": len(planned),
                    "price_used_cad": mid,
                    "price_is_fallback": is_fallback,
                },
            )
            _write_text_atomic(cfg.runtime_dir / REPORT_FILENAME, _json_dumps(asdict(report)) + "\n")
            return report

        # LIVE path: build executor (requires private env vars) and run the roundtrip(s)
        executor = _build_executor()
        events.append(Event(ts_utc=_utc_now(), event="executor_built", data={"live": True}))

        for i in range(int(cfg.roundtrips)):
            events.append(Event(ts_utc=_utc_now(), event="roundtrip_start", data={"i": i + 1, "n": cfg.roundtrips}))

            o1 = _place_order(
                executor=executor,
                pair=cfg.pair,
                strategy=cfg.strategy,
                side=first_leg,
                volume=cfg.volume,
                notional_estimate=est_notional,
                live=True,
            )
            orders.append(o1)
            events.append(Event(ts_utc=_utc_now(), event="order_1", data={"risk_allowed": o1["risk_allowed"], "risk_reason": o1["risk_reason"], "txids": o1["txids"]}))

            o2 = _place_order(
                executor=executor,
                pair=cfg.pair,
                strategy=cfg.strategy,
                side=second_leg,
                volume=cfg.volume,
                notional_estimate=est_notional,
                live=True,
            )
            orders.append(o2)
            events.append(Event(ts_utc=_utc_now(), event="order_2", data={"risk_allowed": o2["risk_allowed"], "risk_reason": o2["risk_reason"], "txids": o2["txids"]}))

            state["last_first_leg"] = first_leg
            state["last_run_ts_utc"] = _utc_now()
            _write_text_atomic(state_path, _json_dumps(state) + "\n")

            if i + 1 < int(cfg.roundtrips):
                time.sleep(float(cfg.sleep_s))

        report = RunReport(
            ts_utc=ts0,
            live_requested=True,
            live_allowed=True,
            pair=cfg.pair,
            strategy=cfg.strategy,
            volume=cfg.volume,
            roundtrips=cfg.roundtrips,
            events=events,
            orders=orders,
            summary={
                "status": "ok",
                "mode": "live",
                "orders": len(orders),
                "price_used_cad": mid,
                "price_is_fallback": is_fallback,
            },
        )
        _write_text_atomic(cfg.runtime_dir / REPORT_FILENAME, _json_dumps(asdict(report)) + "\n")
        return report

    finally:
        lock.release()


# ---------------------------
# CLI
# ---------------------------

def _default_runtime_dir() -> Path:
    env = os.getenv("CHAD_RUNTIME_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()

    # derive repo root from this file location: .../chad/portfolio/kraken_roundtrip_runner.py
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    return (repo_root / "runtime").resolve()


def _parse_args(argv: Optional[List[str]] = None) -> RunConfig:
    ap = argparse.ArgumentParser(description="CHAD Kraken Roundtrip Runner (SSOT-safe, validate-only works without secrets).")
    ap.add_argument("--pair", default=DEFAULT_PAIR)
    ap.add_argument("--strategy", default=DEFAULT_STRATEGY)
    ap.add_argument("--volume", type=float, default=0.0002)
    ap.add_argument("--roundtrips", type=int, default=1)
    ap.add_argument("--once", action="store_true", help="Alias for --roundtrips 1")
    ap.add_argument("--live", action="store_true", help="LIVE orders (requires LiveGate ALLOW_LIVE)")
    ap.add_argument("--sleep-s", type=float, default=2.5)
    ap.add_argument("--cad-floor", type=float, default=25.0)
    ap.add_argument("--btc-floor", type=float, default=0.0005)
    ap.add_argument("--allow-live-with-fallback-price", action="store_true", help="Not recommended")
    ap.add_argument("--runtime-dir", default="", help="Override runtime dir (or use CHAD_RUNTIME_DIR)")

    a = ap.parse_args(argv)

    rt = Path(a.runtime_dir).expanduser().resolve() if str(a.runtime_dir).strip() else _default_runtime_dir()
    roundtrips = 1 if bool(a.once) else int(a.roundtrips)

    vol = float(a.volume)
    if roundtrips <= 0:
        raise SystemExit("roundtrips must be >= 1")
    if not math.isfinite(vol) or vol <= 0:
        raise SystemExit("volume must be finite and > 0")

    sleep_s = float(a.sleep_s)
    if not math.isfinite(sleep_s) or sleep_s < 0:
        raise SystemExit("sleep-s must be finite and >= 0")

    cad_floor = float(a.cad_floor)
    btc_floor = float(a.btc_floor)
    if not math.isfinite(cad_floor) or cad_floor < 0:
        raise SystemExit("cad-floor must be finite and >= 0")
    if not math.isfinite(btc_floor) or btc_floor < 0:
        raise SystemExit("btc-floor must be finite and >= 0")

    return RunConfig(
        pair=str(a.pair).strip(),
        strategy=str(a.strategy).strip(),
        volume=vol,
        roundtrips=roundtrips,
        live=bool(a.live),
        sleep_s=sleep_s,
        cad_floor=cad_floor,
        btc_floor=btc_floor,
        allow_live_with_fallback_price=bool(a.allow_live_with_fallback_price),
        runtime_dir=rt,
    )


def main(argv: Optional[List[str]] = None) -> int:
    try:
        cfg = _parse_args(argv)
        report = run_roundtrip(cfg)
        print(_json_dumps({"ts_utc": report.ts_utc, "summary": report.summary}))
        return 0 if report.summary.get("status") in ("ok", "blocked") else 2
    except SystemExit as exc:
        msg = str(exc)
        if msg:
            print(msg, file=sys.stderr)
        return 2
    except Exception as exc:
        # sanitized error (never prints env vars)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
