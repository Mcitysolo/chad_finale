#!/usr/bin/env python3
"""
Portfolio Snapshot Publisher.

Refreshes runtime/portfolio_snapshot.json every cycle with live equity
values pulled from IBKR + Kraken. Fixes the long-standing issue where
the snapshot was written once on 2026-04-03 and never refreshed,
causing the dynamic risk allocator to compute caps based on stale
account equity.

CAD-to-USD conversion: IBKR Canadian accounts report NetLiquidation
in CAD. CHAD's risk allocator expects USD. Convert using IBKR's own
FX quote (USD.CAD) so the conversion stays consistent with broker truth.

Run via systemd timer chad-portfolio-snapshot.timer (every 5 min).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LOG = logging.getLogger("chad.ops.portfolio_snapshot_publisher")

REPO_ROOT = Path("/home/ubuntu/chad_finale")
RUNTIME_DIR = REPO_ROOT / "runtime"
OUT_PATH = RUNTIME_DIR / "portfolio_snapshot.json"
KRAKEN_BAL_PATH = RUNTIME_DIR / "kraken_balances.json"

IBKR_HOST = "127.0.0.1"
IBKR_PORT = 4002
IBKR_CLIENT_ID = 84  # dedicated clientId, not colliding with live-loop (99)
IBKR_TIMEOUT_SEC = 15
TTL_SECONDS = 300


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_kraken_usd_equity() -> float:
    """Read existing kraken_balances.json for USD-equivalent."""
    try:
        d = json.loads(KRAKEN_BAL_PATH.read_text(encoding="utf-8"))
        usd_eq = d.get("usd_equivalent") or d.get("usd_eq") or 0.0
        return float(usd_eq)
    except Exception as exc:
        LOG.warning("kraken_balance_read_failed: %s", exc)
        return 0.0


def _ibkr_equity_usd() -> Optional[float]:
    """
    Connect to IBKR readonly, read NetLiquidation, convert CAD→USD if needed.

    Returns USD equity or None on failure.
    """
    from ib_async import IB, Forex

    ib = IB()
    try:
        ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID,
                   readonly=True, timeout=IBKR_TIMEOUT_SEC)

        summary = ib.accountSummary()
        net_liq_value: Optional[float] = None
        currency: str = "USD"
        for item in summary:
            if item.tag == "NetLiquidation":
                net_liq_value = float(item.value)
                currency = (item.currency or "USD").upper()
                break

        if net_liq_value is None:
            LOG.warning("ibkr_equity_no_netliq")
            return None

        if currency == "USD":
            return net_liq_value

        if currency == "CAD":
            # Use IBKR's own FX quote so conversion matches broker truth
            fx_contract = Forex("USDCAD")
            ib.qualifyContracts(fx_contract)
            ticker = ib.reqMktData(fx_contract, "", False, False)
            ib.sleep(2)  # wait for quote
            mid = None
            if ticker.bid and ticker.ask and ticker.bid > 0 and ticker.ask > 0:
                mid = (ticker.bid + ticker.ask) / 2
            elif ticker.last and ticker.last > 0:
                mid = ticker.last
            ib.cancelMktData(fx_contract)

            if mid is None or mid <= 0:
                LOG.warning("usd_cad_quote_unavailable, falling back to 1.40")
                mid = 1.40  # rough fallback

            usd = net_liq_value / mid
            LOG.info("converted CAD %.2f / %.4f = USD %.2f",
                     net_liq_value, mid, usd)
            return usd

        LOG.warning("ibkr_unexpected_currency: %s", currency)
        return net_liq_value  # best effort

    except Exception as exc:
        LOG.warning("ibkr_equity_fetch_failed: %s", exc)
        return None
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    ibkr_equity = _ibkr_equity_usd()
    kraken_equity = _read_kraken_usd_equity()

    if ibkr_equity is None:
        LOG.error("ibkr_equity_fetch_failed_no_write")
        return 1

    payload = {
        "ibkr_equity": float(ibkr_equity),
        "coinbase_equity": 0.0,  # CAD-based, Coinbase not used
        "kraken_equity": float(kraken_equity),
        "ts_utc": _utc_now_iso(),
        "ttl_seconds": TTL_SECONDS,
    }

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(OUT_PATH)

    total = ibkr_equity + kraken_equity
    LOG.info("portfolio_snapshot_published ibkr_usd=%.2f kraken_usd=%.2f total_usd=%.2f",
             ibkr_equity, kraken_equity, total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
