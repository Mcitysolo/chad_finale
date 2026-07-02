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
from typing import Any, Dict, Optional

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

# BOX-034A Inc 3 Step 0b: sane USDCAD band (CAD-per-USD). A live mid outside
# this band is treated as garbage/inverted (e.g. ~0.73 = USD-per-CAD) and is
# rejected -> None. There is NO fallback constant: None means "no live rate",
# which callers turn into fail-closed behaviour (never a currency-tagged fake
# value). The band catches both inverted quotes and outright garbage at source.
USDCAD_BAND_LOW = 1.20
USDCAD_BAND_HIGH = 1.50


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_kraken_usd_equity() -> float:
    """Read existing kraken_balances.json for USD-equivalent.

    Legacy helper retained for back-compat (the Kraken *equity leg* no longer
    reads it — see _read_kraken_cad_equity). Still exposed because downstream
    observability may reference the USD-equivalent figure.
    """
    try:
        d = json.loads(KRAKEN_BAL_PATH.read_text(encoding="utf-8"))
        usd_eq = d.get("usd_equivalent") or d.get("usd_eq") or 0.0
        return float(usd_eq)
    except Exception as exc:
        LOG.warning("kraken_balance_read_failed: %s", exc)
        return 0.0


def _read_kraken_cad_equity(path: Path = KRAKEN_BAL_PATH) -> Optional[float]:
    """Read the NATIVE CAD value of the Kraken account from its snapshot.

    The Kraken account is natively CAD. Prefer the provider-written
    ``cad_equivalent`` (crypto sliver valued via the sanctioned USDCAD constant
    at snapshot time). If that field is absent, reconstruct from ``balances``:
    CAD cash 1:1 (native), crypto skipped (the balance snapshot carries no
    price — the provider's cad_equivalent is the priced path).

    Returns the native CAD float (possibly 0.0 when cad_equivalent is present
    and zero), or None when NEITHER a cad_equivalent NOR any reconstructable
    balance is available -> the caller fails closed. This path NEVER performs a
    USD round-trip and NEVER touches the dark live USDCAD feed.
    """
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        LOG.warning("kraken_cad_read_failed: %s", exc)
        return None
    if not isinstance(d, dict):
        return None

    cad_eq = d.get("cad_equivalent")
    if isinstance(cad_eq, (int, float)) and not isinstance(cad_eq, bool):
        return float(cad_eq)

    # Fallback: reconstruct native CAD from balances (no USD round-trip).
    balances = d.get("balances")
    if not isinstance(balances, dict) or not balances:
        return None
    total = 0.0
    valued_any = False
    for asset, qty in balances.items():
        try:
            qty_f = float(qty)
        except (TypeError, ValueError):
            continue
        if qty_f == 0.0:
            continue
        if str(asset).upper() == "CAD":
            total += qty_f  # native CAD, 1:1
            valued_any = True
        else:
            # Crypto legs carry no price in the balance snapshot; the provider's
            # cad_equivalent is the priced path. Skip (debug) rather than guess.
            LOG.debug("kraken_cad_fallback_skip_unpriced asset=%s qty=%s", asset, qty_f)
    if not valued_any:
        return None
    return float(total)


def _apply_kraken_cad_leg(payload: Dict[str, Any], *, path: Path = KRAKEN_BAL_PATH) -> None:
    """Set the Kraken equity leg to its NATIVE CAD value on ``payload``.

    Honest currency semantics: the value IS CAD, so kraken_equity_currency_ok is
    True whenever a native value is readable — with NO usdcad multiply and NO
    dead-feed dependency. Fail-closed (preserve prior kraken_equity, currency_ok
    False, loud log) only when neither cad_equivalent nor a reconstructable
    balance is available.
    """
    cad = _read_kraken_cad_equity(path)
    if cad is not None:
        payload["kraken_equity"] = float(cad)
        payload["kraken_equity_currency"] = "CAD"
        payload["kraken_equity_currency_ok"] = True
    else:
        # FAIL-CLOSED: preserve the prior kraken_equity (already copied via
        # read-through), flag it untrusted, log loudly. The next cycle with a
        # readable native value self-heals.
        payload["kraken_equity_currency_ok"] = False
        LOG.error(
            "KRAKEN_CAD_UNAVAILABLE: no cad_equivalent and no reconstructable "
            "balance; preserving prior kraken_equity=%s, "
            "kraken_equity_currency_ok=false",
            payload.get("kraken_equity"),
        )


def _validate_usdcad(mid: Optional[float]) -> Optional[float]:
    """
    Return ``mid`` iff it is a sane CAD-per-USD rate in [USDCAD_BAND_LOW,
    USDCAD_BAND_HIGH]; otherwise None.

    NO fallback. An unavailable (None), non-numeric, NaN, inverted (~0.73 =
    USD-per-CAD) or otherwise out-of-band quote yields None — never a fabricated
    constant. None is the unambiguous "no live rate" signal.
    """
    if mid is None:
        return None
    try:
        m = float(mid)
    except (TypeError, ValueError):
        return None
    if m != m:  # NaN
        return None
    if m < USDCAD_BAND_LOW or m > USDCAD_BAND_HIGH:
        return None
    return m


def _fetch_usdcad_mid() -> Optional[float]:
    """
    Connect to IBKR readonly and fetch the live USDCAD mid (CAD per USD).

    Returns the raw mid (bid/ask mid, else last), or None if the quote is
    unavailable. Band validation is the caller's job (_get_live_usdcad_rate).
    """
    from ib_async import IB, Forex

    ib = IB()
    try:
        ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID,
                   readonly=True, timeout=IBKR_TIMEOUT_SEC)
        fx_contract = Forex("USDCAD")
        ib.qualifyContracts(fx_contract)
        ticker = ib.reqMktData(fx_contract, "", False, False)
        ib.sleep(2)  # wait for quote
        mid: Optional[float] = None
        if ticker.bid and ticker.ask and ticker.bid > 0 and ticker.ask > 0:
            mid = (ticker.bid + ticker.ask) / 2
        elif ticker.last and ticker.last > 0:
            mid = ticker.last
        ib.cancelMktData(fx_contract)
        return mid
    except Exception as exc:
        LOG.warning("usdcad_fetch_failed: %s", exc)
        return None
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


def _get_live_usdcad_rate(fetcher=None) -> Optional[float]:
    """
    Single source of the live, validated USDCAD rate (CAD per USD).

    Returns the validated mid when a live quote is available and in-band, else
    None. NO fallback constant is ever returned. ``fetcher`` is injectable for
    testing; production uses _fetch_usdcad_mid.
    """
    if fetcher is None:
        fetcher = _fetch_usdcad_mid
    return _validate_usdcad(fetcher())


def _ibkr_equity_usd(usdcad: Optional[float]) -> Optional[float]:
    """
    Connect to IBKR readonly, read NetLiquidation, convert CAD→USD using the
    supplied live ``usdcad`` (CAD per USD). Used ONLY for the display-only
    ``ibkr_equity_usd_display`` key (no risk/sizing/caps path reads it).

    Returns USD equity, or None on failure. NEVER fabricates a rate: if
    NetLiquidation is CAD and ``usdcad`` is None, returns None so the display
    key stays None rather than carrying a USD figure derived from a fake rate.
    """
    from ib_async import IB

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
            if usdcad is None:
                LOG.warning("ibkr_equity_usd_display_unavailable: no live usdcad rate")
                return None
            usd = net_liq_value / usdcad  # CAD / (CAD per USD) = USD
            LOG.info("converted CAD %.2f / %.4f = USD %.2f",
                     net_liq_value, usdcad, usd)
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

    # BOX-034A Inc 3 Step 0b: one live USDCAD fetch, used for BOTH the IBKR USD
    # display value and the Kraken USD→CAD conversion (consistent rate, no
    # double FX call). None == no usable live rate -> fail-closed downstream.
    usdcad = _get_live_usdcad_rate()
    ibkr_equity_usd = _ibkr_equity_usd(usdcad)

    # BOX-034A §3 (single-writer): the v2 collector
    # (chad/portfolio/ibkr_portfolio_collector_v2.py, chad-ibkr-collector.timer)
    # is the SOLE value-setter of the canonical `ibkr_equity` key, in the
    # broker-native base currency (CAD). This publisher previously also wrote
    # `ibkr_equity` with a USD-converted figure, creating an intermittent
    # CAD<->USD dual-writer race on the same key. We now PRESERVE whatever the
    # collector last wrote (read-through, like the coinbase/merge writers) and
    # record our USD-converted figure under the display-only key
    # `ibkr_equity_usd_display`, which no risk/sizing/caps path reads. Because
    # this file is overwritten wholesale, preserving the existing key (rather
    # than dropping it) avoids a transient window where `ibkr_equity` is absent.
    try:
        existing = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            existing = {}
    except Exception:
        existing = {}

    payload = dict(existing)
    payload["coinbase_equity"] = 0.0  # CAD-based, Coinbase not used

    # Display-only USD figure; never fabricated from a fake rate. None when the
    # live USDCAD rate (or IBKR equity) is unavailable. No consumer reads it.
    payload["ibkr_equity_usd_display"] = (
        float(ibkr_equity_usd) if ibkr_equity_usd is not None else None
    )

    # C-ii native-CAD Kraken leg: the Kraken account is natively CAD. Value it
    # in CAD DIRECTLY from the snapshot's cad_equivalent (crypto sliver priced
    # via the sanctioned USDCAD constant at snapshot time) — NO usdcad multiply,
    # NO dead-feed dependency, NO USD round-trip. This retires the former
    # USD-equivalent -> CAD conversion (BOX-034A Inc 3 Step 0b) for this leg;
    # the IBKR USD display block below is unchanged and still uses the live rate.
    _apply_kraken_cad_leg(payload)

    # --- Authoritative USD equity (additive; display/observability only — no
    # risk/sizing/tier/drawdown consumer reads it). Built ONLY from USD
    # components converted via the SAME validated live USDCAD rate used above:
    #   total = ibkr_equity_usd_display
    #         + (kraken_equity CAD / usdcad)        # CAD per USD -> USD
    #         + coinbase (USD)
    # FAIL-CLOSED: if the live rate is unavailable/invalid, or any component
    # failed to convert (e.g. IBKR NetLiquidation unread -> ibkr display None,
    # or kraken not freshly converted), total_equity_usd_authoritative is None
    # and usd_ok is False. It NEVER falls back to the CAD figure.
    coinbase_usd = float(payload.get("coinbase_equity", 0.0) or 0.0)  # 0.0 (unused)
    kraken_cad_now = payload.get("kraken_equity")
    components_ok = (
        usdcad is not None
        and ibkr_equity_usd is not None
        and payload.get("kraken_equity_currency_ok") is True
        and isinstance(kraken_cad_now, (int, float))
    )
    if components_ok:
        kraken_usd = float(kraken_cad_now) / usdcad  # CAD / (CAD per USD) = USD
        payload["total_equity_usd_authoritative"] = (
            float(ibkr_equity_usd) + kraken_usd + coinbase_usd
        )
        payload["usd_ok"] = True
    else:
        payload["total_equity_usd_authoritative"] = None
        payload["usd_ok"] = False
    payload["usdcad_rate_used"] = float(usdcad) if usdcad is not None else None

    payload["ts_utc"] = _utc_now_iso()
    payload["ttl_seconds"] = TTL_SECONDS

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(OUT_PATH)

    kraken_out = payload.get("kraken_equity")
    total_usd_out = payload.get("total_equity_usd_authoritative")
    LOG.info(
        "portfolio_snapshot_published ibkr_usd_display=%s kraken_cad=%s usdcad=%s "
        "total_usd_authoritative=%s usd_ok=%s",
        ("%.2f" % ibkr_equity_usd) if ibkr_equity_usd is not None else "None",
        ("%.2f" % kraken_out) if isinstance(kraken_out, (int, float)) else kraken_out,
        ("%.4f" % usdcad) if usdcad is not None else "None",
        ("%.2f" % total_usd_out) if isinstance(total_usd_out, (int, float)) else total_usd_out,
        payload.get("usd_ok"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
