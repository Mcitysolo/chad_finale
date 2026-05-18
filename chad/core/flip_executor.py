"""
chad/core/flip_executor.py

BG11 — FLIP_ALLOWED close-first / open-second enforcement.

When the Net Exposure Gate emits a FLIP_ALLOWED decision, the executor
MUST first close the conflicting open position and only allow the new
flipped entry through if the close is broker-confirmed. If the close
fails, errors, is rejected, cancelled, or cannot be verified, the
flipped entry is dropped and position_guard is left untouched — no
false-flat state is ever written.

This mirrors the close-with-broker-confirmation pattern already used by
chad/core/position_reconciler.apply_close_intents (ISSUE-29). State
mutation always remains downstream of confirmed broker evidence.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from chad.core.position_guard import _load_state, save_state
# GAP-001 corrected-scope (Phase-48): surgical import of the unified
# operator-exclusion SSOT so flip-close-first also honours it. Preserves
# leaf-dependency direction (position_reconciler does not import here).
from chad.core.position_reconciler import (
    _EFFECTIVE_NON_CHAD_SYMBOLS,
    _EXCLUSION_SOURCE,
)

LOG = logging.getLogger("chad.core.flip_executor")

_REJECTED_STATUSES = {
    "error", "cancelled", "canceled", "inactive",
    "reject", "rejected", "pendingsubmit", "pendingcancel",
    "presubmitted", "unknown", "",
}


def _signal_id(sig: Any) -> str:
    side = getattr(sig, "side", "?")
    side_val = getattr(side, "value", side)
    return (
        f"{getattr(sig, 'strategy', '?')}|"
        f"{getattr(sig, 'symbol', '?')}|"
        f"{side_val}"
    )


def _decision_action_value(decision) -> str:
    action = getattr(decision, "action", None)
    return str(getattr(action, "value", action))


def _build_flip_close_intent(decision, position: Dict[str, Any]):
    """Minimal IBKR-shaped close intent — same shape as
    position_reconciler._close_intent_to_ibkr."""
    from chad.types import AssetClass

    symbol = decision.flip_close_symbol
    open_side = str(decision.flip_close_side or "").upper()
    close_side = "SELL" if open_side == "BUY" else "BUY"
    quantity = float(position.get("quantity", 0.0) or 0.0)

    class _FlipCloseIntent:
        __slots__ = (
            "symbol", "side", "quantity", "sec_type", "asset_class",
            "strategy", "order_type", "confidence",
        )

        def __init__(self) -> None:
            self.symbol = symbol
            self.side = close_side
            self.quantity = quantity
            self.sec_type = "STK"
            self.asset_class = AssetClass.EQUITY
            self.strategy = decision.flip_close_strategy
            self.order_type = "MKT"
            self.confidence = 1.0

    return _FlipCloseIntent()


def _close_confirmed(submitted) -> Tuple[bool, str]:
    """Returns (confirmed, reason). Confirmed iff every order has a
    non-rejected, recognised broker status."""
    if not submitted:
        return False, "no_submitted_orders"
    for ord_obj in submitted:
        status = str(getattr(ord_obj, "status", "") or "").lower()
        if status in _REJECTED_STATUSES:
            return False, f"broker_status={status or 'empty'}"
    return True, "broker_confirmed"


def _matches_flipped_signal(sig: Any, decision) -> bool:
    """The new flipped signal carries decision.strategy + decision.symbol
    on the OPPOSITE side of decision.flip_close_side."""
    sig_strategy = getattr(sig, "strategy", None)
    sig_strategy = getattr(sig_strategy, "value", sig_strategy)
    sig_symbol = getattr(sig, "symbol", None)
    sig_side = getattr(sig, "side", None)
    sig_side = getattr(sig_side, "value", sig_side)
    if str(sig_strategy) != str(decision.strategy):
        return False
    if str(sig_symbol) != str(decision.symbol):
        return False
    if str(sig_side).upper() == str(decision.flip_close_side).upper():
        return False
    return True


def enforce_flip_close_first(
    routed_signals: List[Any],
    gate_decisions: List[Any],
    paper_adapter: Any,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """
    Two-phase flip execution:
      Phase A — submit close intent for the conflicting position.
      Phase B — keep the flipped signal only if Phase A returns
                a non-rejected broker status.

    Returns (filtered_signals, audit_rows). audit_rows are the structured
    BG11 events suitable for any downstream NDJSON writer.

    Position guard is mutated to closed only on broker-confirmed status —
    same gate as position_reconciler.apply_close_intents.
    """
    audit: List[Dict[str, Any]] = []

    if not gate_decisions:
        return list(routed_signals or []), audit

    flip_decisions = [
        d for d in gate_decisions
        if _decision_action_value(d) == "FLIP_ALLOWED"
        and getattr(d, "flip_close_symbol", None)
    ]
    if not flip_decisions:
        return list(routed_signals or []), audit

    open_state = _load_state()
    blocked_ids: set = set()
    confirmed_decisions: List[Any] = []

    for decision in flip_decisions:
        symbol = decision.flip_close_symbol
        # GAP-001 corrected-scope chokepoint guard (Phase-48): operator-
        # excluded symbols are never closed by CHAD via the BG11 flip
        # path either. Drops the matching flipped signal to preserve the
        # operator-exclusion invariant (CHAD must not open or close
        # positions in these symbols at all).
        if symbol and str(symbol).upper() in _EFFECTIVE_NON_CHAD_SYMBOLS:
            ts = datetime.now(timezone.utc).isoformat()
            audit.append({
                "ts_utc": ts,
                "event": "BG11_FLIP_SKIP_EXCLUDED",
                "symbol": symbol,
                "existing_strategy": decision.flip_close_strategy,
                "new_strategy": decision.strategy,
                "source": _EXCLUSION_SOURCE,
                "result": "flipped_signal_blocked",
            })
            LOG.warning(
                "BG11_FLIP_SKIP_EXCLUDED symbol=%s existing=%s new=%s "
                "source=%s — operator-excluded SSOT; close NOT submitted, "
                "flipped signal dropped",
                symbol, decision.flip_close_strategy, decision.strategy,
                _EXCLUSION_SOURCE,
            )
            for sig in routed_signals or []:
                if _matches_flipped_signal(sig, decision):
                    blocked_ids.add(id(sig))
            continue

        existing_strategy = decision.flip_close_strategy
        position_key = f"{existing_strategy}|{symbol}"
        position = open_state.get(position_key) or {}
        ts = datetime.now(timezone.utc).isoformat()

        if not position or not position.get("open"):
            audit.append({
                "ts_utc": ts,
                "event": "BG11_FLIP_NO_OPEN_POSITION",
                "symbol": symbol,
                "existing_strategy": existing_strategy,
                "new_strategy": decision.strategy,
                "result": "allowed_no_close_needed",
            })
            LOG.info(
                "BG11_FLIP_NO_OPEN_POSITION symbol=%s existing=%s new=%s "
                "— flip allowed without close",
                symbol, existing_strategy, decision.strategy,
            )
            continue

        try:
            intent_obj = _build_flip_close_intent(decision, position)
            submitted = paper_adapter.submit_strategy_trade_intents(
                [intent_obj]
            )
        except Exception as exc:  # noqa: BLE001
            LOG.error(
                "BG11_FLIP_BLOCKED_CLOSE_NOT_CONFIRMED symbol=%s "
                "existing_strategy=%s err=%s",
                symbol, existing_strategy, exc,
            )
            for sig in routed_signals or []:
                if _matches_flipped_signal(sig, decision):
                    blocked_ids.add(id(sig))
            audit.append({
                "ts_utc": ts,
                "event": "BG11_FLIP_BLOCKED_CLOSE_NOT_CONFIRMED",
                "symbol": symbol,
                "existing_strategy": existing_strategy,
                "new_strategy": decision.strategy,
                "reason": f"close_submit_exception:{type(exc).__name__}",
            })
            continue

        confirmed, reason = _close_confirmed(submitted)
        if not confirmed:
            LOG.warning(
                "BG11_FLIP_BLOCKED_CLOSE_NOT_CONFIRMED symbol=%s "
                "existing_strategy=%s new_strategy=%s reason=%s",
                symbol, existing_strategy, decision.strategy, reason,
            )
            for sig in routed_signals or []:
                if _matches_flipped_signal(sig, decision):
                    blocked_ids.add(id(sig))
            audit.append({
                "ts_utc": ts,
                "event": "BG11_FLIP_BLOCKED_CLOSE_NOT_CONFIRMED",
                "symbol": symbol,
                "existing_strategy": existing_strategy,
                "new_strategy": decision.strategy,
                "reason": reason,
            })
            continue

        confirmed_decisions.append(decision)
        try:
            if position_key in open_state:
                open_state[position_key]["open"] = False
                open_state[position_key]["last_state"] = "CLOSED"
                open_state[position_key]["closed_by"] = "flip_executor"
                open_state[position_key]["updated_at_utc"] = ts
            save_state(open_state)
        except Exception as ge:  # noqa: BLE001
            LOG.warning("flip_executor: guard update failed: %s", ge)

        try:
            from chad.execution.paper_exec_evidence_writer import (
                PaperExecEvidence,
                normalize_paper_fill_evidence,
                write_paper_exec_evidence,
            )
            from chad.types import AssetClass
            for ord_obj in submitted:
                limit_px = getattr(ord_obj, "limit_price", None)
                fill_px = float(limit_px) if limit_px is not None else 0.0
                ev = PaperExecEvidence(
                    symbol=ord_obj.symbol,
                    side=ord_obj.side,
                    quantity=ord_obj.quantity,
                    fill_price=fill_px,
                    expected_price=fill_px,
                    strategy=existing_strategy or "flip_executor",
                    source_strategies=[existing_strategy or "flip_executor"],
                    broker="ibkr_paper",
                    status=ord_obj.status,
                    asset_class=getattr(
                        ord_obj, "asset_class", AssetClass.EQUITY
                    ),
                    is_live=False,
                    fill_time_utc=(
                        ord_obj.submitted_at.isoformat()
                        if getattr(ord_obj, "submitted_at", None)
                        else ts
                    ),
                )
                normalize_paper_fill_evidence(ev)
                write_paper_exec_evidence(ev)
        except Exception as ev_err:  # noqa: BLE001
            LOG.debug(
                "flip_executor evidence write skipped: %s", ev_err
            )

        LOG.info(
            "BG11_FLIP_CLOSE_CONFIRMED symbol=%s existing_strategy=%s "
            "new_strategy=%s — flipped entry permitted",
            symbol, existing_strategy, decision.strategy,
        )
        audit.append({
            "ts_utc": ts,
            "event": "BG11_FLIP_CLOSE_CONFIRMED",
            "symbol": symbol,
            "existing_strategy": existing_strategy,
            "new_strategy": decision.strategy,
            "reason": reason,
        })

    if not blocked_ids:
        return list(routed_signals or []), audit

    filtered: List[Any] = []
    for sig in routed_signals or []:
        if id(sig) in blocked_ids:
            LOG.warning(
                "BG11_FLIP_DROPPED signal=%s — close was not confirmed",
                _signal_id(sig),
            )
            continue
        filtered.append(sig)
    return filtered, audit
