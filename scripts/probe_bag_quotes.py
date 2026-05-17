#!/usr/bin/env python3
"""
scripts/probe_bag_quotes.py

Phase D Item 2 Tier 3B — BAG live quote probe (read-only diagnostic).

Safe, no-order-placement helper that builds an ``OptionsSpreadSpec`` from
CLI arguments, attaches either synthetic (``--dry-run-fake``) or
broker-snapshot (``--live-readonly``) quotes, runs
``check_spread_limit_price`` from the Tier 3A offline validator, and prints
a single JSON document describing the result.

Hard safety contract
--------------------
* No order placement. The script never references ``placeOrder``,
  ``submit_intent``, ``IbkrAdapter``, ``chad.execution``, or
  ``chad.strategies``.
* No runtime mutation. Nothing is written to ``runtime/`` or any cache.
* ``ib_async`` is imported only inside the ``--live-readonly`` branch so
  the default ``--dry-run-fake`` mode is hermetic.
* If ``CHAD_EXECUTION_MODE=live`` and ``--live-readonly`` is absent, the
  probe refuses to run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chad.options.quote_check import (  # noqa: E402
    BagComboQuote,
    OptionLegQuote,
    SpreadQuoteCheckInput,
    SpreadQuoteCheckResult,
    check_spread_limit_price,
)
from chad.options.spread_spec import OptionsSpreadSpec  # noqa: E402


_QUOTE_MODES = ("legs", "combo", "both")
_MARKET_DATA_TYPES = ("delayed", "live")
_DEFAULT_CLIENT_ID = 9995


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _safe_optional_float(value: Any) -> Optional[float]:
    """Convert a raw IBKR ticker field to ``Optional[float]``.

    Returns ``None`` for ``None``, non-numeric, NaN, infinite, zero, and
    negative values. Mirrors ``quote_check._positive_float`` so the
    validator's contract is preserved exactly.
    """
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    if out <= 0.0:
        return None
    return out


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Read-only BAG live quote probe (Tier 3B).",
    )
    p.add_argument("--symbol", required=True)
    p.add_argument("--expiry", required=True, help="YYYYMMDD")
    p.add_argument("--long-strike", required=True, type=float)
    p.add_argument("--short-strike", required=True, type=float)
    p.add_argument(
        "--long-right", required=True, choices=["C", "P", "c", "p"]
    )
    p.add_argument(
        "--short-right", required=True, choices=["C", "P", "c", "p"]
    )
    p.add_argument("--limit-price", required=True, type=float)

    p.add_argument(
        "--combo-quote-mode",
        choices=list(_QUOTE_MODES),
        default="legs",
    )
    p.add_argument("--contracts", type=int, default=1)
    p.add_argument("--spread-type", default="CUSTOM")
    p.add_argument("--client-id", type=int, default=_DEFAULT_CLIENT_ID)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=4002)
    p.add_argument("--exchange", default="SMART")
    p.add_argument("--currency", default="USD")
    p.add_argument(
        "--market-data-type",
        choices=list(_MARKET_DATA_TYPES),
        default="delayed",
    )
    p.add_argument("--timeout-seconds", type=float, default=8.0)
    p.add_argument("--max-abs-deviation", type=float, default=0.05)
    p.add_argument("--max-pct-deviation", type=float, default=0.10)

    p.add_argument("--dry-run-fake", action="store_true")
    p.add_argument("--live-readonly", action="store_true")
    return p


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    return _build_argparser().parse_args(argv)


def build_spec_from_args(args: argparse.Namespace) -> OptionsSpreadSpec:
    """Construct a typed ``OptionsSpreadSpec`` from CLI args.

    Lets ``OptionsSpreadSpec.__post_init__`` enforce expiry / strike /
    right / ratio invariants. Raises ``ValueError`` on bad inputs.
    """
    return OptionsSpreadSpec(
        symbol=args.symbol,
        expiry=args.expiry,
        long_strike=float(args.long_strike),
        short_strike=float(args.short_strike),
        long_right=args.long_right,
        short_right=args.short_right,
        exchange=args.exchange,
        currency=args.currency,
        spread_type=args.spread_type,
    )


# --------------------------------------------------------------------------- #
# JSON projection helpers                                                     #
# --------------------------------------------------------------------------- #


def _jsonable_dataclass(obj: Any) -> Optional[dict]:
    if obj is None:
        return None
    if is_dataclass(obj):
        return asdict(obj)
    return None


def _jsonable_spec(spec: OptionsSpreadSpec) -> dict:
    return spec.as_dict()


def _jsonable_quote_check(result: SpreadQuoteCheckResult) -> dict:
    return {
        "ok": bool(result.ok),
        "source": result.source,
        "mid_debit": result.mid_debit,
        "limit_price": result.limit_price,
        "deviation_abs": result.deviation_abs,
        "deviation_pct": result.deviation_pct,
        "max_allowed_deviation": result.max_allowed_deviation,
        "warnings": list(result.warnings),
        "errors": list(result.errors),
    }


# --------------------------------------------------------------------------- #
# Fake quote synthesis                                                        #
# --------------------------------------------------------------------------- #


def _fake_quotes(
    spec: OptionsSpreadSpec,
    limit_price: float,
    mode: str,
) -> Tuple[
    Optional[OptionLegQuote], Optional[OptionLegQuote], Optional[BagComboQuote]
]:
    """Synthesize sane bid/ask quotes whose debit mid equals ``limit_price``.

    The long leg trades around ``limit_price + 0.60``, the short leg
    around ``0.60``, so ``long_mid - short_mid == limit_price`` exactly.
    The combo bid/ask are centered on ``limit_price`` with a 0.05 spread.
    """
    ts = _utc_now_iso()
    long_quote: Optional[OptionLegQuote] = None
    short_quote: Optional[OptionLegQuote] = None
    combo_quote: Optional[BagComboQuote] = None

    if mode in ("legs", "both"):
        long_quote = OptionLegQuote(
            symbol=spec.symbol,
            expiry=spec.expiry,
            strike=float(spec.long_strike),
            right=spec.long_right,
            bid=limit_price + 0.55,
            ask=limit_price + 0.65,
            last=None,
            theo_price=None,
            ts_utc=ts,
            source="fake_synth",
        )
        short_quote = OptionLegQuote(
            symbol=spec.symbol,
            expiry=spec.expiry,
            strike=float(spec.short_strike),
            right=spec.short_right,
            bid=0.55,
            ask=0.65,
            last=None,
            theo_price=None,
            ts_utc=ts,
            source="fake_synth",
        )

    if mode in ("combo", "both"):
        combo_quote = BagComboQuote(
            bid=limit_price - 0.05,
            ask=limit_price + 0.05,
            last=None,
            ts_utc=ts,
            source="fake_synth",
        )

    return long_quote, short_quote, combo_quote


# --------------------------------------------------------------------------- #
# Live-readonly ticker conversion                                             #
# --------------------------------------------------------------------------- #


def _ticker_to_leg_quote(
    spec: OptionsSpreadSpec,
    leg: str,
    ticker: Any,
    source: str,
) -> OptionLegQuote:
    """Project an ``ib_async`` ticker onto an ``OptionLegQuote``.

    NaN / negative / zero IBKR fields are converted to ``None``; the
    validator's contract is preserved. ``leg`` is either ``"long"`` or
    ``"short"`` and selects the matching strike/right from the spec.
    """
    if leg == "long":
        strike = float(spec.long_strike)
        right = spec.long_right
    else:
        strike = float(spec.short_strike)
        right = spec.short_right
    return OptionLegQuote(
        symbol=spec.symbol,
        expiry=spec.expiry,
        strike=strike,
        right=right,
        bid=_safe_optional_float(getattr(ticker, "bid", None)),
        ask=_safe_optional_float(getattr(ticker, "ask", None)),
        last=_safe_optional_float(getattr(ticker, "last", None)),
        theo_price=None,
        ts_utc=_utc_now_iso(),
        source=source,
    )


def _ticker_to_combo_quote(ticker: Any, source: str) -> BagComboQuote:
    """Project an ``ib_async`` BAG ticker onto a ``BagComboQuote``."""
    return BagComboQuote(
        bid=_safe_optional_float(getattr(ticker, "bid", None)),
        ask=_safe_optional_float(getattr(ticker, "ask", None)),
        last=_safe_optional_float(getattr(ticker, "last", None)),
        ts_utc=_utc_now_iso(),
        source=source,
    )


# --------------------------------------------------------------------------- #
# Mode dispatchers                                                            #
# --------------------------------------------------------------------------- #


def run_fake(args: argparse.Namespace) -> dict:
    """Dry-run-fake mode. No IBKR imports, no network."""
    errors: list[str] = []
    try:
        spec = build_spec_from_args(args)
    except ValueError as exc:
        return {
            "ok": False,
            "mode": "dry_run_fake",
            "live_readonly": False,
            "symbol": str(args.symbol),
            "spec": None,
            "quote_mode": args.combo_quote_mode,
            "market_data_type": args.market_data_type,
            "quotes": {"long_leg": None, "short_leg": None, "combo": None},
            "quote_check": None,
            "errors": ["spread_spec_validation_failed", str(exc)],
        }

    long_q, short_q, combo_q = _fake_quotes(
        spec, float(args.limit_price), args.combo_quote_mode
    )

    check_input = SpreadQuoteCheckInput(
        spec=spec,
        limit_price=float(args.limit_price),
        long_quote=long_q,
        short_quote=short_q,
        combo_quote=combo_q,
        max_abs_deviation=float(args.max_abs_deviation),
        max_pct_deviation=float(args.max_pct_deviation),
    )
    result = check_spread_limit_price(check_input)

    return {
        "ok": bool(result.ok),
        "mode": "dry_run_fake",
        "live_readonly": False,
        "symbol": spec.symbol,
        "spec": _jsonable_spec(spec),
        "quote_mode": args.combo_quote_mode,
        "market_data_type": args.market_data_type,
        "quotes": {
            "long_leg": _jsonable_dataclass(long_q),
            "short_leg": _jsonable_dataclass(short_q),
            "combo": _jsonable_dataclass(combo_q),
        },
        "quote_check": _jsonable_quote_check(result),
        "errors": errors,
    }


async def run_live(args: argparse.Namespace) -> dict:
    """Read-only live mode. Imports ``ib_async`` lazily, takes snapshots,
    never places orders, never mutates runtime. ``submit_intent`` /
    ``placeOrder`` are not referenced at any point on this branch.
    """
    errors: list[str] = []
    try:
        spec = build_spec_from_args(args)
    except ValueError as exc:
        return {
            "ok": False,
            "mode": "live_readonly",
            "live_readonly": True,
            "symbol": str(args.symbol),
            "spec": None,
            "quote_mode": args.combo_quote_mode,
            "market_data_type": args.market_data_type,
            "quotes": {"long_leg": None, "short_leg": None, "combo": None},
            "quote_check": None,
            "errors": ["spread_spec_validation_failed", str(exc)],
        }

    # Lazy imports — keep dry-run mode hermetic when ib_async is missing.
    try:
        from ib_async import IB, Contract, ComboLeg, Option  # type: ignore[import]
    except ImportError as exc:
        return {
            "ok": False,
            "mode": "live_readonly",
            "live_readonly": True,
            "symbol": spec.symbol,
            "spec": _jsonable_spec(spec),
            "quote_mode": args.combo_quote_mode,
            "market_data_type": args.market_data_type,
            "quotes": {"long_leg": None, "short_leg": None, "combo": None},
            "quote_check": None,
            "errors": ["ib_async_not_installed", str(exc)],
        }

    mdt_code = 3 if args.market_data_type == "delayed" else 1
    timeout_s = float(args.timeout_seconds)

    long_q: Optional[OptionLegQuote] = None
    short_q: Optional[OptionLegQuote] = None
    combo_q: Optional[BagComboQuote] = None

    ib = IB()
    try:
        await ib.connectAsync(
            args.host, int(args.port), clientId=int(args.client_id)
        )
        try:
            ib.reqMarketDataType(mdt_code)
        except Exception as exc:  # pragma: no cover — defensive
            errors.append(f"reqMarketDataType_failed:{exc}")

        long_opt = Option(
            symbol=spec.symbol,
            lastTradeDateOrContractMonth=spec.expiry,
            strike=float(spec.long_strike),
            right=spec.long_right,
            exchange=args.exchange,
            currency=args.currency,
        )
        short_opt = Option(
            symbol=spec.symbol,
            lastTradeDateOrContractMonth=spec.expiry,
            strike=float(spec.short_strike),
            right=spec.short_right,
            exchange=args.exchange,
            currency=args.currency,
        )

        qualified = ib.qualifyContracts(long_opt, short_opt) or []
        if len(qualified) < 2:
            errors.append("qualify_contracts_failed")
        else:
            long_opt = qualified[0]
            short_opt = qualified[1]
            long_con_id = int(getattr(long_opt, "conId", 0) or 0)
            short_con_id = int(getattr(short_opt, "conId", 0) or 0)
            if long_con_id <= 0 or short_con_id <= 0:
                errors.append("leg_conid_zero")

        source_marker = (
            "ibkr_delayed" if args.market_data_type == "delayed" else "ibkr_live"
        )

        if args.combo_quote_mode in ("legs", "both") and not errors:
            long_ticker = ib.reqMktData(
                long_opt, "", snapshot=True, regulatorySnapshot=False
            )
            short_ticker = ib.reqMktData(
                short_opt, "", snapshot=True, regulatorySnapshot=False
            )
            await _await_ticker(ib, long_ticker, timeout_s)
            await _await_ticker(ib, short_ticker, timeout_s)
            try:
                ib.cancelMktData(long_opt)
                ib.cancelMktData(short_opt)
            except Exception:  # pragma: no cover — defensive
                pass
            long_q = _ticker_to_leg_quote(spec, "long", long_ticker, source_marker)
            short_q = _ticker_to_leg_quote(
                spec, "short", short_ticker, source_marker
            )

        if args.combo_quote_mode in ("combo", "both") and not errors:
            bag = Contract()
            bag.symbol = spec.symbol
            bag.secType = "BAG"
            bag.currency = args.currency
            bag.exchange = args.exchange
            long_leg = ComboLeg()
            long_leg.conId = int(getattr(long_opt, "conId", 0) or 0)
            long_leg.ratio = 1
            long_leg.action = "BUY"
            long_leg.exchange = args.exchange
            short_leg = ComboLeg()
            short_leg.conId = int(getattr(short_opt, "conId", 0) or 0)
            short_leg.ratio = 1
            short_leg.action = "SELL"
            short_leg.exchange = args.exchange
            bag.comboLegs = [long_leg, short_leg]
            combo_ticker = ib.reqMktData(
                bag, "", snapshot=True, regulatorySnapshot=False
            )
            await _await_ticker(ib, combo_ticker, timeout_s)
            try:
                ib.cancelMktData(bag)
            except Exception:  # pragma: no cover — defensive
                pass
            combo_q = _ticker_to_combo_quote(combo_ticker, source_marker)
    finally:
        try:
            ib.disconnect()
        except Exception:  # pragma: no cover — defensive
            pass

    check_input = SpreadQuoteCheckInput(
        spec=spec,
        limit_price=float(args.limit_price),
        long_quote=long_q,
        short_quote=short_q,
        combo_quote=combo_q,
        max_abs_deviation=float(args.max_abs_deviation),
        max_pct_deviation=float(args.max_pct_deviation),
    )
    result = check_spread_limit_price(check_input)

    ok = bool(result.ok) and not errors
    return {
        "ok": ok,
        "mode": "live_readonly",
        "live_readonly": True,
        "symbol": spec.symbol,
        "spec": _jsonable_spec(spec),
        "quote_mode": args.combo_quote_mode,
        "market_data_type": args.market_data_type,
        "quotes": {
            "long_leg": _jsonable_dataclass(long_q),
            "short_leg": _jsonable_dataclass(short_q),
            "combo": _jsonable_dataclass(combo_q),
        },
        "quote_check": _jsonable_quote_check(result),
        "errors": errors,
    }


async def _await_ticker(ib: Any, ticker: Any, timeout_s: float) -> None:
    """Poll a ``reqMktData(snapshot=True)`` ticker for up to ``timeout_s``
    seconds, waiting for bid+ask to populate. Returns silently on timeout —
    the caller projects whatever is present onto a quote dataclass.
    """
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        bid = _safe_optional_float(getattr(ticker, "bid", None))
        ask = _safe_optional_float(getattr(ticker, "ask", None))
        last = _safe_optional_float(getattr(ticker, "last", None))
        if (bid is not None and ask is not None) or last is not None:
            return
        try:
            await asyncio.sleep(0.2)
        except Exception:  # pragma: no cover — defensive
            return


# --------------------------------------------------------------------------- #
# Mode resolution + entrypoint                                                #
# --------------------------------------------------------------------------- #


def main(argv: Optional[list[str]] = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        # argparse already emitted a message on stderr.
        return int(exc.code) if isinstance(exc.code, int) else 2

    if args.dry_run_fake and args.live_readonly:
        print(
            json.dumps(
                {
                    "ok": False,
                    "errors": ["conflicting_mode_flags"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    use_live = bool(args.live_readonly)
    use_fake = bool(args.dry_run_fake) or not use_live

    exec_mode = os.environ.get("CHAD_EXECUTION_MODE", "").strip().lower()
    if exec_mode == "live" and not use_live:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "probe_refused_live_mode_without_live_readonly",
                    "CHAD_EXECUTION_MODE": exec_mode,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    if use_live:
        try:
            payload = asyncio.run(run_live(args))
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "errors": ["live_probe_unexpected_error", str(exc)],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        print(json.dumps(payload, indent=2, sort_keys=True))
        if not payload.get("ok"):
            return 2
        return 0

    # Default: dry-run-fake
    assert use_fake
    payload = run_fake(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload.get("errors"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
