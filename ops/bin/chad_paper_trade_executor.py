#!/usr/bin/env python3

import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from chad.analytics.trade_result_logger import TradeResult, log_trade_result
from chad.execution.paper_exec_evidence_writer import (
    PaperExecEvidence,
    normalize_paper_fill_evidence,
    write_paper_exec_evidence,
)

ROOT = Path("/home/ubuntu/chad_finale")
PLAN_PATH = ROOT / "runtime" / "full_execution_cycle_last.json"
PRICE_CACHE_PATH = ROOT / "runtime" / "price_cache.json"
LEDGER_PATH = ROOT / "runtime" / "ibkr_paper_ledger_state.json"

PRICE_CACHE_TTL_SECONDS = 300

# Strategies whose every fill must trade options (BAG / vertical spread). If
# the plan-stage shape is missing the spread-leg metadata required to safely
# simulate an options fill, the order is refused at this layer — no
# log_trade_result, no write_paper_exec_evidence — so the misclassified
# round-trip cannot enter trade_closer FIFO matching, SCR effective trades,
# or profit_lock as a phantom equity fill. Mirrors the writer-level guard
# in chad/execution/paper_exec_evidence_writer.py (_OPTIONS_ONLY_STRATEGIES)
# so the invariant holds even when downstream normalization is bypassed.
_OPTIONS_ONLY_STRATEGIES = frozenset({"alpha_options", "omega_momentum_options"})
_OPTIONS_ASSET_CLASSES = frozenset({"options"})
_BAG_REQUIRED_META_FIELDS = (
    "long_strike",
    "short_strike",
    "long_right",
    "short_right",
    "expiry",
    "net_debit_estimate",
)

logger = logging.getLogger("chad_paper_trade_executor")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


def utc_iso():
    return datetime.now(timezone.utc).isoformat()


def load_plan():
    if not PLAN_PATH.exists():
        raise RuntimeError("Missing execution plan")
    return json.loads(PLAN_PATH.read_text())


def normalize_strategy(strategy, contributors):
    # Priority:
    # 1. valid strategy
    # 2. valid contributors
    # 3. fallback

    s = (strategy or "").lower().strip()

    if s not in {"", "unknown", "manual", "paper_exec"}:
        return s

    for c in contributors:
        c = (c or "").lower().strip()
        if c not in {"", "unknown", "manual", "paper_exec"}:
            return c

    return "paper_exec"


def load_price_cache():
    """Load price cache. Returns (prices_dict, is_stale) tuple."""
    if not PRICE_CACHE_PATH.exists():
        logger.warning("Price cache not found: %s", PRICE_CACHE_PATH)
        return {}, True

    try:
        raw = json.loads(PRICE_CACHE_PATH.read_text())
    except Exception:
        logger.warning("Failed to parse price cache: %s", PRICE_CACHE_PATH)
        return {}, True

    prices = raw.get("prices", {})
    ts_utc = raw.get("ts_utc", "")
    ttl = raw.get("ttl_seconds", PRICE_CACHE_TTL_SECONDS)

    if not ts_utc:
        logger.warning("Price cache missing ts_utc — treating as stale")
        return prices, True

    try:
        cache_time = datetime.fromisoformat(ts_utc.replace("Z", "+00:00"))
        age_seconds = (datetime.now(timezone.utc) - cache_time).total_seconds()
        if age_seconds > ttl:
            logger.warning(
                "Price cache is stale (age=%.0fs, ttl=%ds)", age_seconds, ttl
            )
            return prices, True
    except Exception:
        logger.warning("Failed to parse price cache timestamp: %s", ts_utc)
        return prices, True

    return prices, False


def load_ledger():
    """Load open positions from ibkr_paper_ledger_state.json.
    Returns dict keyed by symbol -> {avg_cost, qty}."""
    if not LEDGER_PATH.exists():
        logger.warning("Ledger not found: %s", LEDGER_PATH)
        return {}

    try:
        raw = json.loads(LEDGER_PATH.read_text())
    except Exception:
        logger.warning("Failed to parse ledger: %s", LEDGER_PATH)
        return {}

    positions = {}
    for _key, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        symbol = entry.get("symbol")
        avg_cost = entry.get("avg_cost")
        if symbol and avg_cost is not None:
            positions[symbol] = {
                "avg_cost": float(avg_cost),
                "qty": float(entry.get("qty", 0)),
            }
    return positions


def compute_pnl(symbol, side, fill_price, quantity, positions):
    """Compute PnL for a single order.

    Returns float pnl value, or None for new entries (no open position)
    so callers can tag the trade as pnl_untrusted.
    """
    pos = positions.get(symbol)
    if pos is None:
        # New entry — no prior position to compute PnL against
        return None

    avg_cost = pos["avg_cost"]

    side_upper = (side or "BUY").upper()
    if side_upper == "BUY":
        pnl = (fill_price - avg_cost) * quantity
    else:
        pnl = (avg_cost - fill_price) * quantity

    return round(pnl, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-orders", type=int, default=2)
    parser.add_argument("--max-notional-usd", type=float, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = load_plan()
    orders = plan.get("orders", [])

    prices, prices_stale = load_price_cache()
    positions = load_ledger()

    if prices_stale:
        logger.warning("Price cache is stale — PnL will use fill_price vs avg_cost only")

    now = utc_iso()
    wrote = 0

    for idx, o in enumerate(orders[: args.max_orders]):

        symbol = o.get("symbol")
        side = o.get("side", "BUY")
        size = float(o.get("size", 1))

        if not symbol:
            logger.warning("Order %d missing symbol — skipping", idx)
            continue

        # Resolve fill price safely — never default to a magic constant.
        # Order's price is preferred; if missing or non-positive we fall
        # back to runtime/price_cache.json. If neither yields a positive
        # price, the order is rejected (skipped) rather than written as a
        # placeholder $100 trusted fill.
        raw_price = o.get("price")
        try:
            proposed_price = float(raw_price) if raw_price is not None else 0.0
        except (TypeError, ValueError):
            proposed_price = 0.0

        try:
            cached_price = float(prices.get(symbol, 0.0) or 0.0)
        except (TypeError, ValueError):
            cached_price = 0.0

        if proposed_price <= 0.0:
            if cached_price > 0.0:
                price = cached_price
                logger.info(
                    "Order %d %s: price missing — using price_cache=%.4f",
                    idx, symbol, cached_price,
                )
            else:
                logger.warning(
                    "Order %d %s: no order price and no price_cache entry "
                    "— rejecting placeholder order",
                    idx, symbol,
                )
                continue
        else:
            price = proposed_price
            # Sanity guard: if cache disagrees by >50%, the proposed price
            # is almost certainly a placeholder (e.g. price=100 for SPY when
            # SPY is actually 720). Reject rather than emit a phantom fill.
            if cached_price > 0.0:
                deviation = abs(price - cached_price) / cached_price
                if deviation > 0.50:
                    logger.warning(
                        "Order %d %s: proposed price=%.4f deviates %.0f%% "
                        "from price_cache=%.4f — rejecting placeholder fill",
                        idx, symbol, price, deviation * 100, cached_price,
                    )
                    continue

        notional = price * size

        contributors = o.get("contributors") or []
        strategy = normalize_strategy(o.get("strategy"), contributors)

        # Resolve plan-supplied signal_meta once: used both for the
        # options-only guard below AND forwarded into ev.extra later so
        # the writer's BAG paper-fill simulator can rebuild leg / debit
        # information for valid alpha_options vertical spreads.
        order_metadata = o.get("metadata") if isinstance(o.get("metadata"), dict) else {}
        signal_meta_raw = order_metadata.get("signal_meta") if isinstance(order_metadata, dict) else None
        signal_meta = signal_meta_raw if isinstance(signal_meta_raw, dict) else {}

        # --- OPTIONS-ONLY STRATEGY GUARD ---
        # alpha_options / omega_momentum_options orders may only proceed
        # when the plan order carries a complete BAG/COMBO signal_meta:
        # sec_type=BAG|COMBO, all leg fields, net_debit_estimate, expiry,
        # AND an options-class intent. If any piece is missing the order
        # has been silently downgraded upstream (typically to a SPY/STK
        # /etf shape) and writing it would produce a misclassified
        # equity round-trip. We refuse the order here — before
        # log_trade_result and before write_paper_exec_evidence — and
        # log "unsupported_options_combo" with the strategy name so an
        # operator can grep the diagnostic in the executor logs.
        strategy_norm = (strategy or "").strip().lower()
        order_asset_class = (o.get("asset_class") or "").strip().lower()
        if strategy_norm in _OPTIONS_ONLY_STRATEGIES:
            sec_type_meta = str(signal_meta.get("sec_type") or "").strip().upper()
            required_ac_meta = str(signal_meta.get("required_asset_class") or "").strip().lower()
            has_bag_marker = sec_type_meta in ("BAG", "COMBO")
            leg_meta_complete = all(
                signal_meta.get(k) not in (None, "")
                for k in _BAG_REQUIRED_META_FIELDS
            )
            options_intent = (
                required_ac_meta in _OPTIONS_ASSET_CLASSES
                or order_asset_class in _OPTIONS_ASSET_CLASSES
            )
            if not (has_bag_marker and leg_meta_complete and options_intent):
                logger.warning(
                    "Order %d %s: unsupported_options_combo strategy=%s "
                    "asset_class=%s sec_type=%s — skipping (no complete "
                    "BAG metadata to safely simulate options fill)",
                    idx, symbol, strategy_norm,
                    order_asset_class or "<empty>",
                    sec_type_meta or "<empty>",
                )
                continue

        # --- PNL COMPUTATION ---
        pnl_untrusted = False
        try:
            pnl = compute_pnl(symbol, side, price, size, positions)
            if pnl is None:
                pnl = 0.0
                pnl_untrusted = True
        except Exception:
            logger.warning("Failed to compute PnL for %s — marking untrusted", symbol, exc_info=True)
            pnl = 0.0
            pnl_untrusted = True

        # --- TRADE RESULT ---
        trade_tags = ["paper", "filled", strategy]
        trade_extra = {
            "plan_path": str(PLAN_PATH),
            "source": "paper_trade_executor",
            "contributors": contributors,
        }
        if pnl_untrusted:
            trade_tags.append("pnl_untrusted")
            trade_extra["pnl_untrusted"] = True
            trade_extra["pnl_untrusted_reason"] = "new_entry_no_prior_position"

        tr = TradeResult(
            strategy=strategy,
            symbol=symbol,
            side=side,
            quantity=size,
            fill_price=price,
            notional=notional,
            pnl=pnl,
            entry_time_utc=now,
            exit_time_utc=now,
            is_live=False,
            broker="paper_exec",
            account_id="PAPER_EXEC",
            regime="paper",
            tags=trade_tags,
            extra=trade_extra,
        )

        if not args.dry_run:
            path = log_trade_result(tr)

            # --- EVIDENCE ---
            # Forward signal_meta from the plan into ev.extra so the BAG
            # paper-fill simulator (in paper_exec_evidence_writer) can
            # rebuild the spread debit / leg structure for alpha_options
            # vertical spreads. Without this, the writer's underlying-price
            # cache lookup stamps a SPY-underlying fill_price (~$720) onto
            # a record whose true cost basis is the per-contract net debit.
            extra: dict = {"plan_path": str(PLAN_PATH)}
            if signal_meta:
                extra.update(signal_meta)

            ev = PaperExecEvidence(
                strategy=strategy,
                source_strategies=contributors,
                symbol=symbol,
                side=side,
                quantity=size,
                fill_price=price,
                notional=notional,
                broker="paper_exec",
                venue="paper_exec",
                account_id="PAPER_EXEC",
                asset_class=o.get("asset_class", ""),
                fill_time_utc=now,
                entry_time_utc=now,
                exit_time_utc=now,
                tags=["paper", "filled", strategy],
                extra=extra,
            )
            try:
                normalize_paper_fill_evidence(ev)
                write_paper_exec_evidence(ev)
            except ValueError as norm_err:
                logger.warning(
                    "Skipping evidence write for %s — normalization rejected: %s",
                    symbol, norm_err,
                )
                continue

            wrote += 1
            print(f"WROTE {symbol} strategy={strategy} pnl={pnl}")

    print(json.dumps({"ok": True, "wrote": wrote}))


if __name__ == "__main__":
    main()
