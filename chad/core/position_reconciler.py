#!/usr/bin/env python3
"""
chad/core/position_reconciler.py

Close open positions when the net strategy signal direction flips.

Runs every live cycle BEFORE the execution pipeline's netting step so
that a BUY signal on a symbol with an open SELL position produces an
explicit CLOSE intent that cannot be netted away by other strategies.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

LOG = logging.getLogger("chad.core.position_reconciler")

# Pre-existing paper account positions that were not opened by CHAD.
# These are skipped entirely so they never trigger reconciler closes
# nor show up as reconciliation mismatches.
KNOWN_NON_CHAD_SYMBOLS: frozenset = frozenset({"AAPL", "MSFT"})

# GAP-001: unify reconciler exclusions onto the publisher's canonical SSOT
# (config/reconciliation_exclusions.json, loaded once at publisher import).
# Reuses the publisher's already-resolved module constants so there is no
# parallel config reader and no runtime/reconciliation_state.json round-trip.
try:
    from chad.ops.reconciliation_publisher import (
        KNOWN_NON_CHAD_SYMBOLS as _PUBLISHER_NON_CHAD,
        EXCLUSION_POLICY as _PUBLISHER_EXCLUSION_POLICY,
    )
    _EFFECTIVE_NON_CHAD_SYMBOLS = frozenset(
        str(s).upper()
        for s in (
            set(KNOWN_NON_CHAD_SYMBOLS)
            | set(_PUBLISHER_NON_CHAD)
            | set(_PUBLISHER_EXCLUSION_POLICY.keys())
        )
    )
    _EXCLUSION_SOURCE = "unified_publisher_config"
except Exception as _exc:  # pragma: no cover - defensive
    LOG.warning(
        "RECONCILER_EXCLUSION_IMPORT_FAILED source=local_floor err=%s",
        _exc,
    )
    _EFFECTIVE_NON_CHAD_SYMBOLS = frozenset(
        str(s).upper() for s in KNOWN_NON_CHAD_SYMBOLS
    )
    _EXCLUSION_SOURCE = "local_floor_fallback"

_PRICE_CACHE_PATH = Path("/home/ubuntu/chad_finale/runtime/price_cache.json")

# PR-02b: synthesized close-fill price resolver.
# - tier 1 walks recent broker-confirmed fills (data/fills/FILLS_*.ndjson)
# - tier 2 reads runtime/price_cache.json with ts_utc/ttl_seconds freshness
# - if both miss, apply_close_intents abstains (no $100 placeholder emission)
_FILLS_DIR = Path("/home/ubuntu/chad_finale/data/fills")
_BROKER_FILL_SCAN_DAYS = 2
_PRICE_CACHE_TTL_DEFAULT_SECONDS = 300
_REJECTED_FILL_STATUSES = frozenset(
    {"rejected", "error", "failed", "cancelled", "inactive"}
)


def _iter_strategy_signal(routed_signals: Iterable) -> Iterable:
    """
    Yield (strategy, signal) pairs regardless of whether routed_signals
    is a flat list of TradeSignal objects or a list of (strategy, signal)
    tuples. TradeSignal objects carry a .strategy attribute of their own.
    """
    for item in routed_signals or []:
        if isinstance(item, tuple) and len(item) == 2:
            yield item[0], item[1]
        else:
            yield getattr(item, "strategy", None), item


def _signal_size(sig: Any) -> float:
    for attr in ("size", "qty", "quantity", "net_size"):
        v = getattr(sig, attr, None)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return 1.0


def reconcile_positions_with_signals(
    open_positions: Dict[str, dict],
    routed_signals: List,
    prices: Optional[Dict[str, float]] = None,
) -> List[dict]:
    """
    Compare each open position against the net signal direction for its
    symbol. When they conflict, emit a close intent.

    Returns a list of close-intent dicts (not ExecutionIntents). These
    are consumed by apply_close_intents() and bypass the netting layer.
    """
    close_intents: List[dict] = []

    if not open_positions:
        return close_intents

    try:
        pairs = list(_iter_strategy_signal(routed_signals))
    except Exception as exc:  # noqa: BLE001
        LOG.warning("reconciler: failed to iterate routed_signals (%s)", exc)
        return close_intents

    for position_key, position in open_positions.items():
        try:
            if not isinstance(position, dict) or not position.get("open"):
                continue

            symbol = position.get("symbol")
            open_side = str(position.get("side", "")).upper()
            if not symbol or open_side not in ("BUY", "SELL"):
                continue

            if str(symbol).upper() in _EFFECTIVE_NON_CHAD_SYMBOLS:
                continue

            # ISSUE-29 fix: skip partial_attribution_residual entries.
            # These are broker_sync lots being gradually claimed by strategies
            # via position_guard._reduce_or_close_broker_sync (ISSUE-56 v2).
            # They are NOT orphans — they will be consumed over subsequent
            # cycles as attribution completes. Flap-closing them destroys
            # the residual that represents unclaimed broker_truth.
            position_meta = position.get("meta") or position.get("extra") or {}
            if (
                position_meta.get("partial_attribution_residual")
                or position.get("source") == "partial_attribution_residual"
                or position.get("closed_by") == "partial_attribution_residual"
            ):
                LOG.debug(
                    "RECONCILER_SKIP partial_attribution_residual entry %s — "
                    "not an orphan, awaiting strategy attribution",
                    position_key,
                )
                continue

            # ISSUE-29 extension: when a strategy entry exists alongside an
            # OPEN broker_sync residual for the same symbol, attribution is
            # in progress. Flap-closing the strategy side leaves the residual
            # alone and the publisher sees chad < broker — looks like drift.
            strategy_name = str(position.get("strategy", "") or "")
            if strategy_name and strategy_name != "broker_sync":
                paired_bs = open_positions.get(f"broker_sync|{symbol}")
                if (
                    isinstance(paired_bs, dict)
                    and paired_bs.get("open")
                    and paired_bs.get("closed_by") == "partial_attribution_residual"
                ):
                    LOG.debug(
                        "RECONCILER_SKIP strategy entry %s — paired broker_sync "
                        "is partial_attribution_residual, attribution in progress",
                        position_key,
                    )
                    continue

            buy_size = 0.0
            sell_size = 0.0
            for _strat, sig in pairs:
                if getattr(sig, "symbol", None) != symbol:
                    continue
                side_str = str(getattr(sig, "side", "")).upper()
                if "BUY" in side_str:
                    buy_size += _signal_size(sig)
                elif "SELL" in side_str:
                    sell_size += _signal_size(sig)

            if buy_size > sell_size:
                net_direction = "BUY"
            elif sell_size > buy_size:
                net_direction = "SELL"
            else:
                net_direction = None

            if net_direction and net_direction != open_side:
                close_side = "BUY" if open_side == "SELL" else "SELL"
                strategy = (
                    position_key.split("|")[0] if "|" in position_key else "unknown"
                )
                close_intents.append({
                    "symbol": symbol,
                    "action": "CLOSE",
                    "open_side": open_side,
                    "close_side": close_side,
                    "quantity": float(position.get("quantity", 0.0) or 0.0),
                    "reason": f"reconciler_flip_{open_side}_to_{net_direction}",
                    "position_key": position_key,
                    "strategy": strategy,
                })
        except Exception as exc:  # noqa: BLE001
            LOG.warning(
                "reconciler: error evaluating position %s: %s", position_key, exc
            )
            continue

    return close_intents


def _load_price(symbol: str) -> float:
    try:
        data = json.loads(_PRICE_CACHE_PATH.read_text(encoding="utf-8"))
        return float((data.get("prices") or {}).get(symbol, 0.0) or 0.0)
    except Exception:
        return 0.0


def _load_fresh_cache_price(symbol: str) -> float:
    """PR-02b tier-2: read price_cache.json with TTL freshness check.

    Returns 0.0 on missing file, parse error, missing/unparseable ts_utc,
    age > ttl_seconds, or missing/non-positive symbol entry. The freshness
    guard prevents reuse of a stale snapshot from before a price feed drop.
    """
    try:
        data = json.loads(_PRICE_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return 0.0
    ts_utc = str(data.get("ts_utc") or "").strip()
    if not ts_utc:
        return 0.0
    try:
        cache_time = datetime.fromisoformat(ts_utc.replace("Z", "+00:00"))
        if cache_time.tzinfo is None:
            cache_time = cache_time.replace(tzinfo=timezone.utc)
        age_s = (datetime.now(timezone.utc) - cache_time).total_seconds()
    except Exception:
        return 0.0
    try:
        ttl_s = float(data.get("ttl_seconds") or _PRICE_CACHE_TTL_DEFAULT_SECONDS)
    except (TypeError, ValueError):
        ttl_s = float(_PRICE_CACHE_TTL_DEFAULT_SECONDS)
    if age_s > ttl_s:
        return 0.0
    try:
        return float((data.get("prices") or {}).get(symbol, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _load_recent_broker_fill_price(symbol: str) -> float:
    """PR-02b tier-1: walk the last N days of data/fills/FILLS_*.ndjson
    for the most-recent non-rejected, positively-priced, not-pnl_untrusted
    fill matching ``symbol``. Returns 0.0 if none found.

    Scanning is bounded to the two most-recent daily files so the cost is
    O(today + yesterday) per close intent — well below one reconciler cycle.
    """
    sym_norm = str(symbol or "").strip().upper()
    if not sym_norm:
        return 0.0
    try:
        files = sorted(_FILLS_DIR.glob("FILLS_*.ndjson"))[-_BROKER_FILL_SCAN_DAYS:]
    except Exception:
        return 0.0
    best_ts = ""
    best_price = 0.0
    for f in reversed(files):
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for line in reversed(text.splitlines()):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            payload = row.get("payload") if isinstance(row, dict) else None
            p = payload if isinstance(payload, dict) else (row if isinstance(row, dict) else {})
            if str(p.get("symbol", "")).strip().upper() != sym_norm:
                continue
            if bool(p.get("reject", False)):
                continue
            status_norm = str(p.get("status", "")).strip().lower()
            if status_norm in _REJECTED_FILL_STATUSES:
                continue
            extra = p.get("extra") if isinstance(p.get("extra"), dict) else {}
            if bool(extra.get("pnl_untrusted", False)):
                continue
            try:
                fp = float(p.get("fill_price", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if fp <= 0.0:
                continue
            ts = str(p.get("fill_time_utc") or p.get("entry_time_utc") or "")
            if ts >= best_ts:
                best_ts = ts
                best_price = fp
        if best_price > 0.0:
            return best_price
    return best_price


def _resolve_close_fill_price(symbol: str) -> float:
    """PR-02b cascade: broker-confirmed fill → fresh price_cache → abstain (0.0).

    Callers MUST treat a 0.0 result as "no usable price available — abstain
    from synthesizing a close fill" rather than falling back to a magic
    constant (100.0). The legacy ``_load_price`` is intentionally not
    consulted here — its missing freshness check is the root cause of the
    pre-PR-02b reconciler $100 placeholder emissions on IWM/SPY.
    """
    px = _load_recent_broker_fill_price(symbol)
    if px > 0.0:
        return px
    px = _load_fresh_cache_price(symbol)
    if px > 0.0:
        return px
    return 0.0


def _close_intent_to_ibkr(close: dict):
    """
    Wrap a close-intent dict in a minimal object that matches the
    ExecutionIntent interface expected by IbkrAdapter.submit_strategy_trade_intents.

    GAP-037: futures symbols (MES/MNQ/MCL/MGC/M6E/...) MUST be wrapped with
    sec_type="FUT" and intent.meta.contract_month so the adapter resolves to a
    Future(symbol, lastTradeDateOrContractMonth=..., exchange=spec.exchange).
    Without this branch the adapter falls through to STK/SMART and IBKR returns
    Error 200 ("No security definition has been found for the request") for
    every futures-close attempt.
    """
    from chad.types import AssetClass
    # Lazy imports: keep module load light; avoid pulling the full IBKR adapter
    # at import time for callers that only build close-intent dicts.
    from chad.execution.ibkr_adapter import resolve_asset_class
    from chad.market_data.futures_contract_resolver import (
        SOURCE_NAME as _FUTURES_RESOLVER_SOURCE,
        resolve_contract_month,
    )

    symbol = str(close.get("symbol", "") or "").strip().upper()
    asset_class_str = resolve_asset_class(symbol) if symbol else "equity"

    meta: Dict[str, Any] = {}
    if asset_class_str == "futures":
        sec_type = "FUT"
        asset_class_enum = AssetClass.FUTURES
        contract_month = resolve_contract_month(symbol)
        if contract_month:
            meta["contract_month"] = contract_month
            meta["contract_month_source"] = _FUTURES_RESOLVER_SOURCE
            meta["contract_month_resolved_at_utc"] = datetime.now(
                timezone.utc
            ).isoformat()
        else:
            LOG.warning(
                "RECONCILER_CLOSE_FUTURES_UNRESOLVED symbol=%s — "
                "adapter will reject; check futures_contract_resolver schedule",
                symbol,
            )
    elif asset_class_str == "etf":
        sec_type = "STK"
        asset_class_enum = AssetClass.ETF
    else:
        # equity / forex / crypto / options fall back to equity-shaped STK
        # since the reconciler historically only sees STK and FUT positions.
        sec_type = "STK"
        asset_class_enum = AssetClass.EQUITY

    class _ReconcilerIntent:
        __slots__ = (
            "symbol", "side", "quantity", "sec_type", "asset_class",
            "strategy", "order_type", "confidence", "meta",
            "exchange", "currency",
        )

        def __init__(self, c: dict):
            self.symbol = c["symbol"]
            self.side = c["close_side"]
            self.quantity = float(c.get("quantity", 0.0) or 0.0)
            self.sec_type = sec_type
            self.asset_class = asset_class_enum
            self.strategy = c.get("strategy", "reconciler")
            self.order_type = "MKT"
            self.confidence = 1.0
            self.meta = dict(meta)
            # Adapter resolves FUT exchange/currency from its FuturesContractSpec
            # registry; STK falls through to SMART/USD defaults in
            # _intent_from_trade_intent. Both branches tolerate empty fields.
            self.exchange = ""
            self.currency = ""

    return _ReconcilerIntent(close)


def apply_close_intents(close_intents: List[dict], paper_adapter: Any) -> None:
    """
    Submit reconciler close intents directly through the IBKR paper
    adapter. Writes paper fill evidence just like the main live loop.
    Never raises — each intent is isolated so one failure cannot stop
    the rest of the cycle.
    """
    if not close_intents:
        return

    from chad.execution.paper_exec_evidence_writer import (
        PaperExecEvidence,
        StrategyAttributionError,
        normalize_paper_fill_evidence,
        write_paper_exec_evidence,
    )
    from chad.types import AssetClass

    for close in close_intents:
        symbol = close.get("symbol")
        # GAP-001 corrected-scope chokepoint guard (Phase-48):
        # operator-excluded symbols never receive close intents from ANY
        # caller of apply_close_intents. Idempotent for the reconciler
        # path (already filtered upstream at line 115); corrective for
        # the regime_reduction path which has no upstream filter.
        if symbol and str(symbol).upper() in _EFFECTIVE_NON_CHAD_SYMBOLS:
            LOG.warning(
                "APPLY_CLOSE_INTENTS_SKIP_EXCLUDED symbol=%s strategy=%s "
                "position_key=%s reason=%s source=%s",
                symbol,
                close.get("strategy"),
                close.get("position_key"),
                close.get("reason"),
                _EXCLUSION_SOURCE,
            )
            continue
        close_side = close.get("close_side")
        reason = close.get("reason", "reconciler_flip")
        qty = float(close.get("quantity", 0.0) or 0.0)

        if qty <= 0:
            LOG.warning(
                "RECONCILER_CLOSE_SKIP symbol=%s side=%s reason=%s qty_non_positive=%s",
                symbol, close_side, reason, qty,
            )
            continue

        LOG.info(
            "RECONCILER_CLOSE symbol=%s side=%s reason=%s qty=%s",
            symbol, close_side, reason, qty,
        )

        try:
            intent_obj = _close_intent_to_ibkr(close)
            submitted = paper_adapter.submit_strategy_trade_intents([intent_obj])
            # ISSUE-29 (extended): collect post-normalize fill evidence so
            # the guard mutation below requires POSITIVE confirmation
            # (status=filled/paper_fill, fill_id present, not pending,
            # not pnl_untrusted) — not just absence-of-rejection.
            confirmed_fills: List[dict] = []
            for order in submitted:
                # PR-02b: never emit a synthesized close-fill at the legacy
                # $100 placeholder fallback. Resolve a real price via the
                # broker-fill → fresh-cache cascade; if neither tier yields
                # a positive price, abstain (skip the evidence write) and
                # let the next reconciler cycle retry. Guard mutation
                # already requires positive confirmation (ISSUE-29 ext.),
                # so skipping the write keeps the guard open by design.
                fill_price = _resolve_close_fill_price(order.symbol)
                if fill_price <= 0.0:
                    LOG.warning(
                        "RECONCILER_CLOSE_ABSTAIN_NO_PRICE symbol=%s side=%s "
                        "qty=%s reason=no_broker_fill_and_no_fresh_price_cache "
                        "— evidence write skipped; guard remains open",
                        order.symbol, order.side, getattr(order, "quantity", "?"),
                    )
                    continue
                try:
                    # Calibration fix (2026-04-22 per Audit-O): thread
                    # expected_price through reconciler close evidence too.
                    # close intents may not carry limit_price; fall back to
                    # the loaded fill_price so slippage is zero rather than
                    # silently missing.
                    _expected_px_close = float(getattr(intent_obj, "expected_price", 0.0) or 0.0)
                    if _expected_px_close <= 0.0:
                        _lp_close = getattr(intent_obj, "limit_price", None)
                        try:
                            _expected_px_close = float(_lp_close) if _lp_close is not None else fill_price
                        except (TypeError, ValueError):
                            _expected_px_close = fill_price
                    ev = PaperExecEvidence(
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.quantity,
                        fill_price=fill_price,
                        expected_price=_expected_px_close,
                        strategy=close.get("strategy", "reconciler") or "reconciler",
                        source_strategies=[close.get("strategy", "reconciler") or "reconciler"],
                        broker="ibkr_paper",
                        status=order.status,
                        asset_class=getattr(order, "asset_class", AssetClass.EQUITY),
                        is_live=False,
                        fill_time_utc=(
                            order.submitted_at.isoformat()
                            if getattr(order, "submitted_at", None)
                            else datetime.now(timezone.utc).isoformat()
                        ),
                    )
                    # Translate raw IBKR statuses (PendingSubmit/error) to
                    # paper_fill, resolve asset_class, and back-fill fill_price
                    # from price_cache.json with futures symbol normalization.
                    # write_paper_exec_evidence runs the same normalizer as a
                    # safety net; calling it here lets us catch and skip a
                    # ValueError before the hash-chained write happens.
                    normalize_paper_fill_evidence(ev)
                    paths = write_paper_exec_evidence(ev)
                    # Capture post-normalize, post-write evidence so the
                    # guard mutation can verify positive confirmation.
                    _post_status = str(getattr(ev, "status", "") or "").strip().lower()
                    confirmed_fills.append({
                        "fill_id": str(paths.get("fill_id", "")) if isinstance(paths, dict) else "",
                        "status": _post_status,
                        "pnl_untrusted": bool(getattr(ev, "pnl_untrusted", False)),
                        "reject": bool(getattr(ev, "reject", False)),
                        "fills_path": (paths.get("fills_path", "") if isinstance(paths, dict) else ""),
                    })
                except StrategyAttributionError as attr_err:
                    LOG.warning("reconciler evidence attribution failed: %s", attr_err)
                except ValueError as norm_err:
                    LOG.warning(
                        "reconciler evidence normalization rejected fill "
                        "(no resolvable price): %s", norm_err,
                    )
                except Exception as ev_err:  # noqa: BLE001
                    LOG.warning("reconciler evidence write failed: %s", ev_err)

            # ISSUE-29 fix: do NOT mutate the position guard until we
            # have evidence the broker accepted the close. The earlier
            # behavior set state[pk]["open"] = False as soon as
            # submit_strategy_trade_intents returned, even when the
            # adapter had appended an error/cancelled SubmittedOrder.
            # That created phantom-closed positions that the next
            # reconciler cycle could not re-detect.
            _broker_rejected = False
            _reject_status = ""
            for _ord in submitted:
                _status = str(getattr(_ord, "status", "") or "").lower()
                if _status in ("error", "cancelled", "inactive", "reject", "rejected"):
                    _broker_rejected = True
                    _reject_status = _status
                    break

            # ISSUE-29 (extended): positive fill confirmation required.
            # Even when the broker did not return an explicit reject
            # (e.g. PendingSubmit / unknown / no fills written), the
            # guard MUST stay open until at least one confirmed fill
            # evidence record exists for this close intent.
            from chad.core.position_guard import is_fill_confirmed  # type: ignore
            _fill_confirmed = bool(confirmed_fills) and all(
                is_fill_confirmed(f) for f in confirmed_fills
            )

            pk = close.get("position_key")
            if _broker_rejected:
                LOG.warning(
                    "ISSUE29_GUARD_SKIP: broker rejected close intent for "
                    "position_key=%s status=%s — guard NOT mutated to "
                    "avoid phantom close",
                    pk, _reject_status,
                )
            elif not _fill_confirmed:
                LOG.warning(
                    "ISSUE29_GUARD_SKIP: close intent for position_key=%s "
                    "lacks confirmed fill evidence (fills=%s) — guard NOT "
                    "mutated; will retry on next cycle",
                    pk, confirmed_fills,
                )
            else:
                try:
                    from chad.core.position_guard import (  # type: ignore
                        _load_state,
                        write_position_guard,
                    )
                    state = _load_state()
                    if pk and pk in state:
                        state[pk]["open"] = False
                        state[pk]["last_state"] = "CLOSED"
                        state[pk]["closed_by"] = "position_reconciler"
                        state[pk]["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                        # Stamp the fill_id of the confirming evidence so
                        # downstream auditors (reconciliation_publisher)
                        # can trace which fill closed which guard entry.
                        _first_fill_id = (
                            confirmed_fills[0].get("fill_id", "")
                            if confirmed_fills else ""
                        )
                        if _first_fill_id:
                            state[pk]["closed_fill_id"] = _first_fill_id
                        write_position_guard(state)
                except Exception as gs_err:  # noqa: BLE001
                    LOG.warning("reconciler guard update failed: %s", gs_err)

        except Exception as exc:  # noqa: BLE001
            LOG.error(
                "RECONCILER_CLOSE_FAILED symbol=%s side=%s reason=%s err=%s",
                symbol, close_side, reason, exc,
            )


def load_open_positions() -> Dict[str, dict]:
    """Helper — load position_guard.json directly (no side effects)."""
    try:
        from chad.core.position_guard import STATE_PATH
        if not STATE_PATH.is_file():
            return {}
        raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        return {k: v for k, v in raw.items() if isinstance(v, dict) and v.get("open")}
    except Exception:
        return {}
