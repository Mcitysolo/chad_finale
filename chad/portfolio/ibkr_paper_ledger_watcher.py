#!/usr/bin/env python3
"""
IBKR Paper Ledger Watcher (NO ORDERS)

Purpose
-------
This module runs as a periodic oneshot (systemd timer). It inspects IBKR paper
positions for a target account and writes tamper-evident TradeResult NDJSON
records when a position transitions from OPEN -> CLOSED.

Key properties
--------------
- Read-only with respect to execution: never places orders.
- Resilient IBKR connection (bounded retries + jitter).
- Produces machine-readable run reports every invocation.
- Produces deterministic `writes.details` diagnostics even when no records are written.
- Attempts execution-based trusted PnL (FIFO) from IB executions; otherwise emits
  pnl_untrusted with explicit reasons.

Runtime inputs
--------------
Config JSON (default: /home/ubuntu/CHAD FINALE/runtime/ibkr_paper_ledger.json)
Example:
{
  "enabled": true,
  "default_strategy": "manual",
  "ibkr": { "host": "127.0.0.1", "port": 4002, "client_id": 9013 }
}

State file
----------
We keep lightweight open-state in: /home/ubuntu/CHAD FINALE/runtime/ibkr_paper_ledger_state.json
This file is atomic-written and only stores metadata needed to reconstruct close events.

Outputs
-------
- Run report JSON: /home/ubuntu/CHAD FINALE/reports/ledger/IBKR_PAPER_LEDGER_RUN_YYYYMMDDThhmmssZ.json
- TradeResult NDJSON: /home/ubuntu/CHAD FINALE/data/trades/trade_history_YYYYMMDD.ndjson

"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ib_insync is used for reading IBKR state only.
from ib_insync import IB, Execution, Fill  # type: ignore

# CHAD trade result + logging
try:
    # Expected in your repo per earlier records
    from chad.analytics.trade_result_logger import TradeResult, log_trade_result  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"Missing CHAD TradeResult logger import: {e}") from e


ROOT = Path("/home/ubuntu/CHAD FINALE")
CONFIG_PATH_DEFAULT = ROOT / "runtime" / "ibkr_paper_ledger.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _add_detail(details: list[dict], **kv: Any) -> None:
    kv["ts_utc"] = _iso(_utc_now())
    details.append(kv)


@dataclass(frozen=True)
class LedgerConfig:
    enabled: bool
    default_strategy: str
    ibkr_host: str
    ibkr_port: int
    ibkr_client_id: int

    # Paths
    state_path: Path
    reports_dir: Path
    trades_dir: Path

    # Execution window for trusted PnL fallback (seconds)
    exec_window_seconds: float


def load_config(path: Path) -> LedgerConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))

    enabled = bool(raw.get("enabled", False))
    default_strategy = str(raw.get("default_strategy") or "manual")

    ibkr = raw.get("ibkr") or {}
    host = str(ibkr.get("host") or "127.0.0.1")
    port = int(ibkr.get("port") or 4002)
    client_id = int(ibkr.get("client_id") or 0)
    if client_id <= 0:
        raise ValueError("Config error: ibkr.client_id must be a positive integer")

    # Default: 90 seconds window; can be increased safely.
    exec_window_seconds = float(raw.get("exec_window_seconds") or 90.0)

    return LedgerConfig(
        enabled=enabled,
        default_strategy=default_strategy,
        ibkr_host=host,
        ibkr_port=port,
        ibkr_client_id=client_id,
        state_path=ROOT / "runtime" / "ibkr_paper_ledger_state.json",
        reports_dir=ROOT / "reports" / "ledger",
        trades_dir=ROOT / "data" / "trades",
        exec_window_seconds=exec_window_seconds,
    )


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"open": {}, "last_run_utc": None}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        # If state is corrupted, do not crash: start fresh but keep traceability.
        return {"open": {}, "last_run_utc": None, "state_corrupt": True}


def _state_key(account_id: str, con_id: int, symbol: str, sec_type: str) -> str:
    return f"{account_id}::{con_id}::{symbol}::{sec_type}"


def _connect_ib(host: str, port: int, client_id: int) -> IB:
    """
    Connect to IBKR with bounded retries and jitter.

    Why:
      - IBG can listen on port while API isn't ready (apiStart timeout).
      - This watcher runs on a timer; we prefer bounded retries over long hangs.

    Guarantees:
      - Returns a connected IB instance on success.
      - Raises RuntimeError with concise context on failure.
    """
    max_attempts = 6
    base_sleep = 0.6
    max_sleep = 6.0
    last_err: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        ib = IB()
        try:
            ib.connect(host, port, clientId=client_id, timeout=10)
            if not ib.isConnected():
                raise RuntimeError("connected_socket_false_after_connect")
            return ib
        except Exception as e:
            last_err = e
            try:
                ib.disconnect()
            except Exception:
                pass

            if attempt >= max_attempts:
                raise RuntimeError(
                    f"ibkr_connect_failed after {max_attempts} attempts host={host} port={port} client_id={client_id}: {type(e).__name__}: {e}"
                ) from e

            sleep_s = min(max_sleep, base_sleep * (2 ** (attempt - 1)))
            jitter = sleep_s * 0.25
            sleep_s = max(0.1, sleep_s + (random.random() * 2 - 1) * jitter)
            time.sleep(sleep_s)

    raise RuntimeError(f"ibkr_connect_failed_unreachable host={host} port={port} client_id={client_id}: {last_err}")


def _get_account_id(ib: IB) -> str:
    """
    Determine the account id deterministically.
    Preference order:
      1) managedAccounts() first entry
      2) accountSummary() first entry account field
    """
    accts = list(ib.managedAccounts() or [])
    if accts:
        return str(accts[0])
    summ = ib.accountSummary()
    if summ:
        return str(summ[0].account)
    return ""


def _positions_by_conid(ib: IB) -> Dict[int, dict]:
    """
    Returns mapping conId -> position dict with keys:
      symbol, secType, currency, conId, position, avgCost, exchange
    """
    out: Dict[int, dict] = {}
    for pos in ib.positions():
        c = pos.contract
        con_id = int(getattr(c, "conId", 0) or 0)
        if con_id <= 0:
            continue
        out[con_id] = {
            "symbol": str(getattr(c, "symbol", "") or ""),
            "secType": str(getattr(c, "secType", "") or ""),
            "currency": str(getattr(c, "currency", "") or ""),
            "exchange": str(getattr(c, "exchange", "") or ""),
            "conId": con_id,
            "qty": float(pos.position or 0.0),
            "avg_cost": float(pos.avgCost or 0.0),
        }
    return out


def _coerce_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
        if math.isfinite(x):
            return x
        return None
    except Exception:
        return None


def _fills_in_window(ib: IB, cutoff_utc: datetime) -> list[Fill]:
    """
    Fetch fills known to ib_insync client. Note: ib.fills() returns fills in memory.
    We call reqExecutions to refresh, then use fills list.
    """
    # Refresh executions; this updates fills cache.
    try:
        ib.reqExecutions()
    except Exception:
        # Best effort; do not crash watcher on refresh failure.
        pass

    fills = []
    for f in ib.fills():
        try:
            t = f.time
            # ib_insync fill time may be naive or tz-aware; normalize.
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            if t >= cutoff_utc:
                fills.append(f)
        except Exception:
            continue
    return fills


def _compute_exec_pnl_fifo(
    fills: list[Fill],
    con_id: int,
    opened_at_utc: datetime,
    closed_at_utc: datetime,
) -> Tuple[Optional[float], dict]:
    """
    Compute FIFO PnL from executions for a single conId within [opened_at, closed_at].
    Returns (pnl, details). pnl None if insufficient data.
    """
    # Collect executions
    execs: list[Execution] = []
    for f in fills:
        try:
            if int(getattr(f.contract, "conId", 0) or 0) != int(con_id):
                continue
            t = f.time
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            if opened_at_utc <= t <= closed_at_utc:
                execs.append(f.execution)
        except Exception:
            continue

    if not execs:
        return None, {"included_execs": 0, "reason": "no_executions_in_window"}

    # Build buys and sells
    buys: list[Tuple[float, float]] = []   # (qty, price)
    sells: list[Tuple[float, float]] = []  # (qty, price)
    for e in execs:
        try:
            qty = float(e.shares or 0.0)
            px = float(e.price or 0.0)
            if qty <= 0 or not math.isfinite(qty) or not math.isfinite(px):
                continue
            side = str(e.side or "").upper()
            if side == "BOT":
                buys.append((qty, px))
            elif side == "SLD":
                sells.append((qty, px))
        except Exception:
            continue

    buy_qty = sum(q for q, _ in buys)
    sell_qty = sum(q for q, _ in sells)
    matched_qty = min(buy_qty, sell_qty)

    if matched_qty <= 0:
        return None, {
            "included_execs": len(execs),
            "buy_qty": buy_qty,
            "sell_qty": sell_qty,
            "reason": "no_matched_qty",
        }

    # FIFO match
    pnl = 0.0
    b_i = 0
    s_i = 0
    b_rem = buys[0][0] if buys else 0.0
    s_rem = sells[0][0] if sells else 0.0

    while b_i < len(buys) and s_i < len(sells) and matched_qty > 1e-12:
        b_qty, b_px = buys[b_i]
        s_qty, s_px = sells[s_i]
        b_rem = b_rem if b_rem > 0 else b_qty
        s_rem = s_rem if s_rem > 0 else s_qty
        take = min(b_rem, s_rem, matched_qty)
        pnl += (s_px - b_px) * take
        b_rem -= take
        s_rem -= take
        matched_qty -= take
        if b_rem <= 1e-12:
            b_i += 1
            if b_i < len(buys):
                b_rem = buys[b_i][0]
        if s_rem <= 1e-12:
            s_i += 1
            if s_i < len(sells):
                s_rem = sells[s_i][0]

    # Averages for reporting
    avg_buy = (sum(q * px for q, px in buys) / buy_qty) if buy_qty > 0 else None
    avg_sell = (sum(q * px for q, px in sells) / sell_qty) if sell_qty > 0 else None

    details = {
        "included_execs": len(execs),
        "buy_qty": float(buy_qty),
        "sell_qty": float(sell_qty),
        "matched_qty": float(min(sum(q for q, _ in buys), sum(q for q, _ in sells))),
        "avg_buy": float(avg_buy) if avg_buy is not None else None,
        "avg_sell": float(avg_sell) if avg_sell is not None else None,
        "multiplier": 1.0,
    }
    return float(pnl), details


def run_once(cfg: LedgerConfig) -> dict:
    now = _utc_now()
    report: Dict[str, Any] = {
        "enabled": bool(cfg.enabled),
        "generated_at_utc": _iso(now),
        "writes": {"details": [], "trade_results": 0},
    }
    details = report["writes"]["details"]

    _add_detail(
        details,
        event="watcher_start",
        client_id=int(cfg.ibkr_client_id),
        host=cfg.ibkr_host,
        port=int(cfg.ibkr_port),
        exec_window_seconds=float(cfg.exec_window_seconds),
        state_path=str(cfg.state_path),
    )

    if not cfg.enabled:
        _add_detail(details, event="disabled", reason="config_enabled_false")
        cfg.reports_dir.mkdir(parents=True, exist_ok=True)
        out = cfg.reports_dir / f"IBKR_PAPER_LEDGER_RUN_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
        _atomic_write_json(out, report)
        report["report_path"] = str(out)
        return report

    ib: Optional[IB] = None
    try:
        ib = _connect_ib(cfg.ibkr_host, cfg.ibkr_port, cfg.ibkr_client_id)
        _add_detail(details, event="connect_ok")
        account_id = _get_account_id(ib)
        _add_detail(details, event="account_id", account_id=account_id)

        state = _load_state(cfg.state_path)
        open_state: Dict[str, dict] = dict(state.get("open") or {})

        # Snapshot current positions
        pos_map = _positions_by_conid(ib)
        _add_detail(details, event="positions_snapshot", positions_count=len(pos_map))

        # Detect new opens and closes
        # We track opens by key = account::conId::symbol::secType.
        # When qty != 0, it's open. When previously open and now missing/qty==0, it's close candidate.
        # Note: IB may report zero-qty positions; we treat qty==0 as closed.
        current_open_keys = set()

        for con_id, rec in pos_map.items():
            sym = str(rec.get("symbol") or "")
            sec_type = str(rec.get("secType") or "")
            k = _state_key(account_id, con_id, sym, sec_type)
            qty = float(rec.get("qty") or 0.0)
            if abs(qty) > 1e-12:
                current_open_keys.add(k)
                if k not in open_state:
                    # Record open metadata
                    open_state[k] = {
                        "account_id": account_id,
                        "conId": con_id,
                        "symbol": sym,
                        "secType": sec_type,
                        "currency": str(rec.get("currency") or ""),
                        "exchange": str(rec.get("exchange") or ""),
                        "qty": float(qty),
                        "avg_cost": float(rec.get("avg_cost") or 0.0),
                        "opened_at_utc": _iso(now),
                        "strategy": cfg.default_strategy,
                        "tags": ["ibkr_paper", cfg.default_strategy],
                    }
                    _add_detail(details, event="open_detected", key=k, qty=float(qty), avg_cost=float(rec.get("avg_cost") or 0.0))

        # Close candidates: in open_state but not currently open
        close_keys = [k for k in open_state.keys() if k not in current_open_keys]
        _add_detail(details, event="close_candidates", count=len(close_keys))

        # For trusted PnL, compute fills within a cutoff window
        cutoff = now - timedelta_seconds(cfg.exec_window_seconds)
        fills = _fills_in_window(ib, cutoff)
        _add_detail(details, event="fills_window", cutoff_utc=_iso(cutoff), fills_count=len(fills))

        for k in close_keys:
            rec = open_state.get(k) or {}
            con_id = int(rec.get("conId") or 0)
            sym = str(rec.get("symbol") or "")
            sec_type = str(rec.get("secType") or "")
            cur = str(rec.get("currency") or "")
            opened_at_s = str(rec.get("opened_at_utc") or _iso(now))
            try:
                opened_at = datetime.fromisoformat(opened_at_s.replace("Z", "+00:00"))
                if opened_at.tzinfo is None:
                    opened_at = opened_at.replace(tzinfo=timezone.utc)
            except Exception:
                opened_at = now

            # Attempt execution-based PnL FIFO
            pnl_exec, exec_details = _compute_exec_pnl_fifo(
                fills=fills,
                con_id=con_id,
                opened_at_utc=opened_at,
                closed_at_utc=now,
            )

            pnl_untrusted_reason: Optional[str] = None
            pnl_source: Optional[str] = None
            pnl_logged = 0.0

            extra: Dict[str, Any] = {
                "source": "ibkr_paper_ledger_watcher",
                "currency": cur,
                "conId": con_id,
                "secType": sec_type,
            }

            tags = list(rec.get("tags") or ["ibkr_paper", cfg.default_strategy])
            tags.append("manual") if "manual" not in tags and cfg.default_strategy == "manual" else None

            if pnl_exec is not None and math.isfinite(float(pnl_exec)):
                pnl_logged = float(pnl_exec)
                pnl_source = "executions_fifo"
                extra["exec_pnl_details"] = exec_details
                extra["exec_window_seconds"] = float(cfg.exec_window_seconds)
                extra["exec_cutoff_time_utc"] = _iso(cutoff)
                extra["pnl_source"] = pnl_source
                extra["pnl_untrusted"] = False
            else:
                pnl_logged = 0.0
                pnl_untrusted_reason = exec_details.get("reason") if isinstance(exec_details, dict) else "no_exec_details"
                tags.append("pnl_untrusted")
                extra["pnl_untrusted"] = True
                extra["pnl_untrusted_reason"] = str(pnl_untrusted_reason)

            qty_open = float(rec.get("qty") or 0.0)
            avg_cost = float(rec.get("avg_cost") or 0.0)
            notional = abs(qty_open) * avg_cost
            side = "BUY" if qty_open > 0 else "SELL"

            tr = TradeResult(
                strategy=str(rec.get("strategy", cfg.default_strategy)),
                symbol=sym,
                side=side,
                quantity=abs(qty_open),
                fill_price=avg_cost,
                notional=notional,
                pnl=float(pnl_logged),
                entry_time_utc=str(rec.get("opened_at_utc", _iso(now))),
                exit_time_utc=_iso(now),
                is_live=False,
                broker="ibkr",
                account_id=str(rec.get("account_id", "")) or None,
                regime=None,
                tags=tags,
                extra=extra,
            )

            log_path = log_trade_result(tr)
            report["writes"]["trade_results"] += 1
            report["writes"]["details"].append(
                {
                    "event": "trade_result_written",
                    "symbol": sym,
                    "conId": con_id,
                    "pnl": float(pnl_logged),
                    "pnl_source": pnl_source,
                    "pnl_untrusted_reason": pnl_untrusted_reason,
                    "log_path": str(log_path),
                    "ts_utc": _iso(_utc_now()),
                }
            )

            # Remove closed state
            open_state.pop(k, None)

        _atomic_write_json(cfg.state_path, {"open": open_state, "last_run_utc": _iso(now)})

        if report["writes"]["trade_results"] == 0:
            _add_detail(
                details,
                event="no_trade_results",
                reason="no_close_events_detected_or_no_new_positions_closed",
            )

        cfg.reports_dir.mkdir(parents=True, exist_ok=True)
        out = cfg.reports_dir / f"IBKR_PAPER_LEDGER_RUN_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
        _atomic_write_json(out, report)
        report["report_path"] = str(out)

        _add_detail(details, event="watcher_end", trade_results=int(report["writes"]["trade_results"]))
        return report

    except Exception as e:
        # Always return a report even on failure; systemd will record nonzero exit.
        _add_detail(report["writes"]["details"], event="error", error_type=type(e).__name__, error=str(e))
        cfg.reports_dir.mkdir(parents=True, exist_ok=True)
        out = cfg.reports_dir / f"IBKR_PAPER_LEDGER_RUN_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
        _atomic_write_json(out, report)
        report["report_path"] = str(out)
        raise
    finally:
        try:
            if ib is not None:
                ib.disconnect()
        except Exception:
            pass


def timedelta_seconds(seconds: float) -> datetime:
    # Helper to avoid importing timedelta in older environments; uses epoch arithmetic.
    # Returns utc_now - seconds equivalent is handled by caller by subtracting from now.
    # Here we just provide a sentinel; caller uses `now - timedelta_seconds(...)`.
    # We actually need real timedelta behavior, so we implement a small conversion.
    from datetime import timedelta

    return timedelta(seconds=float(seconds))  # type: ignore[return-value]


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="IBKR paper ledger watcher (no orders).")
    p.add_argument("--config", type=str, default=str(CONFIG_PATH_DEFAULT), help="Path to runtime config JSON.")
    args = p.parse_args(argv)

    cfg = load_config(Path(args.config).expanduser().resolve())
    rep = run_once(cfg)
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
