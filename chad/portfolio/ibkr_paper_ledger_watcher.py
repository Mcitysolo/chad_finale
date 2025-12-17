#!/usr/bin/env python3
"""
IBKR Paper Ledger Watcher (Production-Safe, Disabled by Default)

Goal
----
Create tamper-evident TradeResult NDJSON records ONLY when *real* paper-account position
lifecycles occur (open -> close), without placing orders and without fabricating results.

Safety guarantees
-----------------
- This module NEVER places orders.
- DISABLED by default unless explicitly enabled in a runtime config file.
- If disabled, it only emits a preview artifact.
- When enabled, it records open positions into a state file, then on close writes TradeResult.

Critical PnL hygiene (bug fix)
------------------------------
IBKR's reqPnLSingle.realizedPnL can sometimes return an "unset/sentinel" (e.g., ~1.79e308)
or non-finite values. Those MUST NOT be treated as real PnL.

This watcher now:
- Treats non-finite or absurdly large realizedPnL values as invalid.
- If invalid baseline/current realizedPnL is detected:
  - It writes the TradeResult with pnl=0.0
  - Adds tag "pnl_untrusted"
  - Records raw values + reason under TradeResult.extra for later forensic correction

This prevents corrupt PnL from poisoning SCR/analytics while still proving the lifecycle.

Runtime config
--------------
Default path: /home/ubuntu/CHAD FINALE/runtime/ibkr_paper_ledger.json

Example:
{
  "enabled": true,
  "default_strategy": "manual",
  "ibkr": {"host":"127.0.0.1","port":4002,"client_id":9003}
}

Artifacts
---------
- State:   /home/ubuntu/CHAD FINALE/runtime/ibkr_paper_ledger_state.json
- Preview: /home/ubuntu/CHAD FINALE/reports/ledger/IBKR_PAPER_LEDGER_PREVIEW_*.json
- Runs:    /home/ubuntu/CHAD FINALE/reports/ledger/IBKR_PAPER_LEDGER_RUN_*.json
- Ledger:  data/trades/trade_history_YYYYMMDD.ndjson  (via TradeResult logger)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
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


# -------------------------------
# Realized PnL hygiene
# -------------------------------

# Any realizedPnL magnitude above this is treated as invalid/sentinel.
_PNL_ABSURD_ABS_THRESHOLD = 1e100


def _coerce_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v
    except Exception:
        return None


def _is_valid_realized_pnl(v: Optional[float]) -> bool:
    if v is None:
        return False
    if not math.isfinite(v):
        return False
    if abs(v) >= _PNL_ABSURD_ABS_THRESHOLD:
        return False
    return True


def _req_realized_pnl(ib, *, account_id: str, con_id: int, sleep_s: float) -> Optional[float]:
    """
    Use reqPnLSingle to read realizedPnL (cumulative for day). Caller must delta it.

    Returns:
      float realizedPnL if valid, else None.
    """
    if not account_id or con_id <= 0:
        return None

    pnl_obj = ib.reqPnLSingle(account_id, "", con_id)
    try:
        ib.sleep(max(0.0, float(sleep_s)))
        raw = getattr(pnl_obj, "realizedPnL", None)
        v = _coerce_float(raw)
        return v if _is_valid_realized_pnl(v) else None
    finally:
        try:
            ib.cancelPnLSingle(pnl_obj)
        except Exception:
            pass


# -------------------------------
# Core run
# -------------------------------

def run_once(cfg: LedgerConfig) -> Dict[str, Any]:
    now = _utc_now()
    state = _load_state(cfg.state_path)
    open_state: Dict[str, Any] = state.get("open", {})  # key -> info

    # Always produce a report (even when disabled) for auditability.
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
                "baseline_realized_pnl": baseline_realized,  # may be None if invalid
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

            baseline_raw = rec.get("baseline_realized_pnl", None)
            baseline = _coerce_float(baseline_raw)
            baseline_ok = _is_valid_realized_pnl(baseline)

            current_realized = _req_realized_pnl(
                ib, account_id=str(rec.get("account_id", "")), con_id=con_id, sleep_s=cfg.pnl_sleep_seconds
            )
            current_ok = _is_valid_realized_pnl(current_realized)

            pnl_untrusted_reason = ""
            if baseline_ok and current_ok and baseline is not None and current_realized is not None:
                pnl_val = float(current_realized - baseline)
                pnl_ok = _is_valid_realized_pnl(pnl_val)
                if pnl_ok:
                    pnl_logged = pnl_val
                else:
                    pnl_logged = 0.0
                    pnl_untrusted_reason = "computed_pnl_invalid"
            else:
                pnl_logged = 0.0
                if not baseline_ok and not current_ok:
                    pnl_untrusted_reason = "baseline_and_current_invalid"
                elif not baseline_ok:
                    pnl_untrusted_reason = "baseline_invalid"
                else:
                    pnl_untrusted_reason = "current_invalid"

            qty = float(rec.get("qty", 0.0))
            avg_cost = float(rec.get("avg_cost", 0.0))
            side = "BUY" if qty > 0 else "SELL"
            notional = abs(qty) * avg_cost

            tags = list(rec.get("tags", []))
            extra: Dict[str, Any] = {
                "source": "ibkr_paper_ledger_watcher",
                "currency": cur,
                "conId": con_id,
                "baseline_realized_pnl": baseline_raw,
                "current_realized_pnl": current_realized,
            }
            if pnl_untrusted_reason:
                tags.append("pnl_untrusted")
                extra["pnl_untrusted"] = True
                extra["pnl_untrusted_reason"] = pnl_untrusted_reason

            tr = TradeResult(
                strategy=str(rec.get("strategy", cfg.default_strategy)),
                symbol=sym,
                side=side,
                quantity=abs(qty),
                fill_price=avg_cost,
                notional=notional,
                pnl=float(pnl_logged),
                entry_time_utc=str(rec.get("opened_at_utc", now.isoformat())),
                exit_time_utc=now.isoformat(),
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
                {"symbol": sym, "pnl": float(pnl_logged), "log_path": str(log_path), "pnl_untrusted_reason": pnl_untrusted_reason or None}
            )

            open_state.pop(k, None)

        _atomic_write_json(cfg.state_path, {"open": open_state, "last_run_utc": now.isoformat()})

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
