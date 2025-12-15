#!/usr/bin/env python3
"""
IBKR Paper Ledger Watcher (Production-Safe, Disabled by Default)

Goal
----
Create TradeResult records ONLY when *real* paper-account position lifecycles occur,
without placing orders and without fabricating results.

Safety guarantees
-----------------
- This module NEVER places orders.
- It is DISABLED by default unless explicitly enabled in a runtime config file.
- It only writes TradeResult entries when it can prove:
    - a position was opened (non-zero qty)
    - and later closed (qty returns to zero)
- If disabled, it only emits a preview artifact.

Design
------
We maintain a small state file tracking open positions. On each run:
- read current IBKR positions (paper account)
- detect opens/closes vs prior state
- on close: compute realizedPnL delta via PnLSingle and write TradeResult

This is the “writer” missing from Step 22.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from chad.analytics.trade_result_logger import TradeResult, log_trade_result


# -------------------------------
# Paths / defaults
# -------------------------------

ROOT_DEFAULT = Path("/home/ubuntu/CHAD FINALE")
RUNTIME_DEFAULT = ROOT_DEFAULT / "runtime"
REPORTS_DEFAULT = ROOT_DEFAULT / "reports" / "ledger"

CONFIG_PATH_DEFAULT = RUNTIME_DEFAULT / "ibkr_paper_ledger.json"
STATE_PATH_DEFAULT = RUNTIME_DEFAULT / "ibkr_paper_ledger_state.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


# -------------------------------
# Config
# -------------------------------

@dataclass(frozen=True)
class LedgerConfig:
    enabled: bool
    # Strict allowlist: only trades tagged as CHAD are eligible for SCR later.
    # For now, we still log, but we mark strategy="manual" when unknown.
    default_strategy: str = "manual"

    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 4002
    ibkr_client_id: int = 9003

    # Poll behavior
    pnl_sleep_seconds: float = 0.8  # allow PnLSingle to populate

    # Paths
    state_path: Path = STATE_PATH_DEFAULT
    reports_dir: Path = REPORTS_DEFAULT


def load_config(path: Path = CONFIG_PATH_DEFAULT) -> LedgerConfig:
    """
    Safe-by-default loader:
    - Missing config => enabled=False
    - Invalid config => enabled=False
    """
    if not path.is_file():
        return LedgerConfig(enabled=False)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return LedgerConfig(enabled=False)

    enabled = bool(raw.get("enabled", False))
    default_strategy = str(raw.get("default_strategy", "manual")).strip() or "manual"

    ibkr = raw.get("ibkr", {}) if isinstance(raw.get("ibkr", {}), dict) else {}
    host = str(ibkr.get("host", "127.0.0.1")).strip() or "127.0.0.1"
    port = int(ibkr.get("port", 4002))
    client_id = int(ibkr.get("client_id", 9003))

    return LedgerConfig(
        enabled=enabled,
        default_strategy=default_strategy,
        ibkr_host=host,
        ibkr_port=port,
        ibkr_client_id=client_id,
    )


# -------------------------------
# State
# -------------------------------

def _load_state(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"open": {}}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"open": {}}
    if not isinstance(d, dict):
        return {"open": {}}
    if "open" not in d or not isinstance(d.get("open"), dict):
        d["open"] = {}
    return d


def _state_key(symbol: str, currency: str) -> str:
    return f"{symbol.upper()}::{currency.upper()}"


# -------------------------------
# IBKR helpers (import lazily)
# -------------------------------

def _connect_ib(*, host: str, port: int, client_id: int):
    from ib_insync import IB  # type: ignore[import]
    ib = IB()
    ib.connect(host, port, clientId=client_id, timeout=10)
    return ib


def _get_account_id(ib) -> str:
    accts = list(getattr(ib, "managedAccounts")() or [])
    return str(accts[0]) if accts else ""


def _positions_by_symbol(ib) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Returns mapping:
      (symbol, currency) -> {"qty": float, "avg_cost": float, "conId": int}
    """
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for p in ib.positions():
        contract = getattr(p, "contract", None)
        if contract is None:
            continue
        sym = str(getattr(contract, "symbol", "")).upper()
        cur = str(getattr(contract, "currency", "USD")).upper()
        qty = float(getattr(p, "position", 0.0))
        if sym and qty != 0.0:
            out[(sym, cur)] = {
                "qty": qty,
                "avg_cost": float(getattr(p, "avgCost", 0.0)),
                "conId": int(getattr(contract, "conId", 0) or 0),
            }
    return out


def _req_realized_pnl(ib, *, account_id: str, con_id: int, sleep_s: float) -> float:
    """
    Use reqPnLSingle to read realizedPnL (cumulative for day). Caller must delta it.
    """
    if not account_id or con_id <= 0:
        return 0.0
    pnl = ib.reqPnLSingle(account_id, "", con_id)
    ib.sleep(sleep_s)
    val = getattr(pnl, "realizedPnL", 0.0)
    try:
        return float(val)
    except Exception:
        return 0.0
    finally:
        try:
            ib.cancelPnLSingle(pnl)
        except Exception:
            pass


# -------------------------------
# Core run
# -------------------------------

def run_once(cfg: LedgerConfig) -> Dict[str, Any]:
    now = _utc_now()
    state = _load_state(cfg.state_path)
    open_state: Dict[str, Any] = state.get("open", {})  # key -> info

    # Always produce a small report (even when disabled) for auditability.
    report: Dict[str, Any] = {
        "generated_at_utc": now.isoformat(),
        "enabled": cfg.enabled,
        "writes": {"trade_results": 0, "details": []},
    }

    if not cfg.enabled:
        cfg.reports_dir.mkdir(parents=True, exist_ok=True)
        out = cfg.reports_dir / f"IBKR_PAPER_LEDGER_PREVIEW_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
        _atomic_write_json(out, report)
        report["preview_path"] = str(out)
        return report

    # Enabled path: connect and evaluate real positions.
    ib = _connect_ib(host=cfg.ibkr_host, port=cfg.ibkr_port, client_id=cfg.ibkr_client_id)
    try:
        account_id = _get_account_id(ib)
        positions = _positions_by_symbol(ib)

        # Detect opens
        for (sym, cur), info in positions.items():
            k = _state_key(sym, cur)
            if k in open_state:
                continue
            con_id = int(info.get("conId", 0) or 0)
            baseline_realized = _req_realized_pnl(
                ib, account_id=account_id, con_id=con_id, sleep_s=cfg.pnl_sleep_seconds
            )
            open_state[k] = {
                "symbol": sym,
                "currency": cur,
                "opened_at_utc": now.isoformat(),
                "qty": float(info["qty"]),
                "avg_cost": float(info["avg_cost"]),
                "conId": con_id,
                "baseline_realized_pnl": float(baseline_realized),
                "strategy": cfg.default_strategy,
                "tags": ["ibkr_paper", "manual"],
                "account_id": account_id,
            }

        # Detect closes (items that were open but no longer present)
        still_open_keys = set(_state_key(sym, cur) for (sym, cur) in positions.keys())
        closed_keys = [k for k in list(open_state.keys()) if k not in still_open_keys]

        for k in closed_keys:
            rec = open_state.get(k, {})
            sym = str(rec.get("symbol", "")).upper()
            cur = str(rec.get("currency", "USD")).upper()
            con_id = int(rec.get("conId", 0) or 0)

            baseline = float(rec.get("baseline_realized_pnl", 0.0))
            current_realized = _req_realized_pnl(
                ib, account_id=str(rec.get("account_id", "")), con_id=con_id, sleep_s=cfg.pnl_sleep_seconds
            )
            pnl = float(current_realized - baseline)

            qty = float(rec.get("qty", 0.0))
            avg_cost = float(rec.get("avg_cost", 0.0))
            side = "BUY" if qty > 0 else "SELL"
            notional = abs(qty) * avg_cost

            tr = TradeResult(
                strategy=str(rec.get("strategy", cfg.default_strategy)),
                symbol=sym,
                side=side,
                quantity=abs(qty),
                fill_price=avg_cost,
                notional=notional,
                pnl=pnl,
                entry_time_utc=str(rec.get("opened_at_utc", now.isoformat())),
                exit_time_utc=now.isoformat(),
                is_live=False,
                broker="ibkr",
                account_id=str(rec.get("account_id", "")) or None,
                regime=None,
                tags=list(rec.get("tags", [])),
                extra={
                    "source": "ibkr_paper_ledger_watcher",
                    "currency": cur,
                    "conId": con_id,
                    "baseline_realized_pnl": baseline,
                    "current_realized_pnl": current_realized,
                },
            )

            log_path = log_trade_result(tr)
            report["writes"]["trade_results"] += 1
            report["writes"]["details"].append(
                {"symbol": sym, "pnl": pnl, "log_path": str(log_path)}
            )

            # remove closed
            open_state.pop(k, None)

        # Persist state
        _atomic_write_json(cfg.state_path, {"open": open_state, "last_run_utc": now.isoformat()})

        # Persist run report
        cfg.reports_dir.mkdir(parents=True, exist_ok=True)
        out = cfg.reports_dir / f"IBKR_PAPER_LEDGER_RUN_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
        _atomic_write_json(out, report)
        report["report_path"] = str(out)
        return report

    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


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
