#!/usr/bin/env python3
"""
scripts/flatten_futures_oneshot.py

One-shot operator tool to FLATTEN all open futures positions on the IBKR paper
account. Mandatory PREVIEW + typed CONFIRM before any order is placed.

This is a STANDALONE tool. It deliberately does NOT import or route through
CHAD's gated execution path (no live_loop, no ibkr_adapter order routing, no
env gates, no SCR/LiveGate). It opens a clean direct ib_async connection and
closes positions the same way a manual TWS "close position" would — i.e. it
intentionally bypasses CHAD's futures-disable gate, because this is an operator
action, not a strategy submission.

Closing orders are derived FROM THE LIVE BROKER POSITION (conId / localSymbol /
expiry come from the broker, never hardcoded): action = SELL if position > 0
else BUY, quantity = abs(position), order type = Market.

Usage (operator, interactive):
    python3 scripts/flatten_futures_oneshot.py

Safety:
- If the connection fails, it prints the error and exits — placing nothing.
- It lists every futures position and requires the operator to type FLATTEN
  exactly. Any other input aborts and places nothing.
- The clientId is a dedicated constant (7715) so this tool never collides with
  CHAD's running services. Change CLIENT_ID below if 7715 is ever in use.

The pure logic (order derivation, preview formatting, the confirm gate, the
orchestration with injectable I/O) is unit-tested in
chad/tests/test_flatten_futures_oneshot.py WITHOUT any network or ib_async — all
ib_async imports here are lazy (inside the connect/place functions) so importing
this module never requires a broker or the ib_async package.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

# --- Dedicated, easily-changed connection constants -------------------------
HOST = "127.0.0.1"
PORT = 4002          # IB Gateway, paper
CLIENT_ID = 7715     # dedicated to this tool — NOT 83 (orchestrator family),
                     # NOT 9034. Change here if 7715 is ever in use.

CONFIRM_TOKEN = "FLATTEN"
EXPECTED_FUT_COUNT = 3            # the expected open-futures count; off-by-N → loud warn
DEFAULT_FILL_TIMEOUT_S = 30.0
DEFAULT_POLL_S = 1.0

# orderStatus values that mean "stop waiting" for a placed order
_TERMINAL_STATUSES = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}


# ---------------------------------------------------------------------------
# Pure, testable logic (no I/O, no ib_async)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClosingOrder:
    """A market order that closes one live futures position. Derived entirely
    from the broker position — nothing here is hardcoded."""
    symbol: str
    action: str            # "SELL" (close a long) or "BUY" (close a short)
    quantity: int          # abs(position) contracts
    position: float        # signed live position (for the preview)
    con_id: int            # broker contract id of the exact position
    local_symbol: str
    contract_month: str


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def is_futures(pos: Any) -> bool:
    """True iff the position's contract secType is FUT."""
    c = getattr(pos, "contract", None)
    return str(getattr(c, "secType", "") or "").upper() == "FUT"


def derive_closing_order(pos: Any) -> Optional[ClosingOrder]:
    """Derive the closing order from a live position.

    long  +142 -> SELL 142
    short  -19 -> BUY  19
    flat/0     -> None (nothing to close)

    The contract identity (conId / localSymbol / expiry month) is taken from
    the broker position, never reconstructed.
    """
    qty = _safe_float(getattr(pos, "position", 0.0), 0.0)
    contracts = int(abs(qty))
    if qty == 0 or contracts <= 0:
        return None
    c = getattr(pos, "contract", None)
    action = "SELL" if qty > 0 else "BUY"
    return ClosingOrder(
        symbol=str(getattr(c, "symbol", "") or "").upper(),
        action=action,
        quantity=contracts,
        position=qty,
        con_id=int(getattr(c, "conId", 0) or 0),
        local_symbol=str(getattr(c, "localSymbol", "") or ""),
        contract_month=str(getattr(c, "lastTradeDateOrContractMonth", "") or ""),
    )


def format_preview_table(orders: List[ClosingOrder]) -> str:
    """Render the one-row-per-futures-position preview table."""
    lines = ["=== FUTURES FLATTEN PREVIEW ==="]
    lines.append(
        f"{'SYMBOL':<8} {'MONTH':<10} {'POSITION':>10} {'ACTION':<6} {'QTY':>6} {'CONID':>12}"
    )
    lines.append("-" * 58)
    for co in orders:
        lines.append(
            f"{co.symbol:<8} {co.contract_month:<10} {co.position:>10.0f} "
            f"{co.action:<6} {co.quantity:>6d} {co.con_id:>12d}"
        )
    lines.append(f"({len(orders)} closing order(s) to place)")
    return "\n".join(lines)


def confirm_gate(input_fn: Callable[[str], str] = input) -> bool:
    """Return True ONLY when the operator types the confirm token exactly.

    Surrounding whitespace is stripped; the token comparison is case-sensitive.
    EOF / Ctrl-C are treated as a refusal.
    """
    prompt = f"\nType {CONFIRM_TOKEN} exactly to place these closing orders (anything else aborts): "
    try:
        resp = input_fn(prompt)
    except (EOFError, KeyboardInterrupt):
        return False
    return str(resp).strip() == CONFIRM_TOKEN


# ---------------------------------------------------------------------------
# Exchange resolution (fixes IBKR Warning 321 "Missing order exchange")
# ---------------------------------------------------------------------------

# Known fallback exchanges for CHAD's micro-futures universe. Used ONLY when
# the broker / qualifyContracts leaves the contract's exchange blank. NEVER
# used to guess for an unknown symbol — those orders are skipped, not placed.
FUTURES_EXCHANGE_MAP = {"M6E": "CME", "M2K": "CME", "MCL": "NYMEX"}


def resolve_exchange(
    ib: Any,
    contract: Any,
    *,
    out: Callable[[str], None] = print,
) -> Optional[str]:
    """Resolve a futures contract's order exchange BEFORE submit (fixes the
    live ValidationError "Warning 321 ... Missing order exchange").

    ib.positions() returns contracts with a conId but a blank exchange; IBKR
    then rejects the closing order. We resolve in three steps:

      1. ib.qualifyContracts(contract) — fills the full spec (incl. exchange)
         from the conId.
      2. If the exchange is STILL blank, fall back to FUTURES_EXCHANGE_MAP
         keyed by symbol.
      3. If it is STILL unresolved (unknown symbol / blank), return None — the
         caller MUST skip the order. We never submit with a missing or guessed
         exchange.

    Returns the resolved exchange string, or None if it cannot be resolved.
    The resolved exchange is written back onto ``contract.exchange``.
    """
    qf = getattr(ib, "qualifyContracts", None)
    if callable(qf):
        try:
            qf(contract)
        except Exception as exc:
            out(
                f"[flatten] qualifyContracts failed for "
                f"{getattr(contract, 'symbol', '?')} "
                f"(conId={getattr(contract, 'conId', '?')}): {exc}"
            )
    exchange = str(getattr(contract, "exchange", "") or "").strip()
    if exchange:
        return exchange
    symbol = str(getattr(contract, "symbol", "") or "").upper()
    mapped = FUTURES_EXCHANGE_MAP.get(symbol)
    if mapped:
        try:
            contract.exchange = mapped
        except Exception:
            pass
        out(
            f"[flatten] exchange for {symbol} "
            f"(conId={getattr(contract, 'conId', '?')}) was blank after qualify "
            f"— using mapped exchange {mapped}"
        )
        return mapped
    return None


# ---------------------------------------------------------------------------
# Order placement (lazy ib_async; injectable for tests)
# ---------------------------------------------------------------------------

def _default_place_and_wait(
    ib: Any,
    contract: Any,
    action: str,
    quantity: int,
    *,
    timeout_s: float = DEFAULT_FILL_TIMEOUT_S,
    poll_s: float = DEFAULT_POLL_S,
    out: Callable[[str], None] = print,
) -> Any:
    """Place a Market closing order on the live contract and wait for a
    terminal status. Returns the ib_async Trade."""
    from ib_async import MarketOrder  # lazy: no import at module load

    order = MarketOrder(action, int(quantity))
    trade = ib.placeOrder(contract, order)
    waited = 0.0
    while waited < timeout_s:
        ib.sleep(poll_s)
        waited += poll_s
        status = str(getattr(getattr(trade, "orderStatus", None), "status", "") or "")
        if status in _TERMINAL_STATUSES:
            break
    return trade


# ---------------------------------------------------------------------------
# Orchestration (injectable I/O so it is fully testable offline)
# ---------------------------------------------------------------------------

def flatten_futures(
    ib: Any,
    *,
    input_fn: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
    place_order_fn: Callable[..., Any] = _default_place_and_wait,
    expected_count: int = EXPECTED_FUT_COUNT,
    timeout_s: float = DEFAULT_FILL_TIMEOUT_S,
) -> str:
    """Preview -> confirm -> flatten -> re-verify.

    Returns one of: "NOTHING", "ABORTED", "FLAT", "INCOMPLETE".
    Places NOTHING unless the operator types the confirm token exactly.
    """
    positions = list(ib.positions() or [])
    futs = [p for p in positions if is_futures(p)]
    if not futs:
        out("No open futures positions — nothing to flatten")
        return "NOTHING"

    closings: List[Tuple[Any, ClosingOrder]] = []
    for p in futs:
        co = derive_closing_order(p)
        if co is not None:
            closings.append((p, co))

    # PREVIEW
    out(format_preview_table([co for _p, co in closings]))
    if len(futs) != expected_count:
        out(
            f"!!! WARNING: expected exactly {expected_count} open futures positions "
            f"but found {len(futs)} — review every row above before confirming !!!"
        )

    # CONFIRM
    if not confirm_gate(input_fn):
        out("Aborted — no orders placed.")
        return "ABORTED"

    # PLACE
    any_skipped = False
    for p, co in closings:
        contract = getattr(p, "contract", None)
        # Resolve the exchange BEFORE submit — ib.positions() returns blank
        # exchanges, which IBKR rejects with Warning 321. Never submit blank.
        exchange = resolve_exchange(ib, contract, out=out)
        if not exchange:
            any_skipped = True
            out(
                f"[flatten] !!! SKIPPED {co.symbol} (conId={co.con_id}): could not "
                f"resolve an exchange — refusing to submit a closing order with a "
                f"missing/blank exchange (would be IBKR Warning 321). "
                f"Position left OPEN."
            )
            continue
        out(
            f"[flatten] placing {co.action} {co.quantity} {co.symbol} "
            f"{co.contract_month} (conId={co.con_id}) on {exchange}"
        )
        try:
            trade = place_order_fn(
                ib, contract, co.action, co.quantity, timeout_s=timeout_s, out=out
            )
            st = getattr(trade, "orderStatus", None)
            out(
                f"[flatten] result {co.symbol}: status={getattr(st, 'status', '?')} "
                f"filled={getattr(st, 'filled', '?')} avgPrice={getattr(st, 'avgFillPrice', '?')}"
            )
        except Exception as exc:  # one bad order must not abort the rest
            out(f"[flatten] ORDER ERROR {co.symbol} (conId={co.con_id}): {exc}")
            any_skipped = True

    # RE-VERIFY
    remaining = [p for p in (list(ib.positions() or [])) if is_futures(p)]
    if not remaining and not any_skipped:
        out("FINAL STATUS: FLAT — no futures positions remain")
        return "FLAT"
    out("FINAL STATUS: INCOMPLETE — futures positions still open or orders skipped:")
    if any_skipped:
        out(
            "  NOTE: one or more closing orders were SKIPPED (exchange unresolved) "
            "— those positions were left OPEN"
        )
    for p in remaining:
        c = getattr(p, "contract", None)
        out(
            f"  {getattr(c, 'symbol', '?')} "
            f"{getattr(c, 'lastTradeDateOrContractMonth', '?')} "
            f"position={getattr(p, 'position', '?')}"
        )
    return "INCOMPLETE"


# ---------------------------------------------------------------------------
# Connection + entrypoint (lazy ib_async)
# ---------------------------------------------------------------------------

def _connect(
    host: str = HOST,
    port: int = PORT,
    client_id: int = CLIENT_ID,
    out: Callable[[str], None] = print,
) -> Optional[Any]:
    """Direct ib_async connection. Returns a connected IB or None on failure."""
    from ib_async import IB  # lazy: no import at module load

    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=10)
    except Exception as exc:
        out(f"[flatten] CONNECT FAILED ({host}:{port} clientId={client_id}): {exc}")
        return None
    if not ib.isConnected():
        out(f"[flatten] CONNECT FAILED ({host}:{port} clientId={client_id}): not connected")
        return None
    out(f"[flatten] connected {host}:{port} clientId={client_id}")
    return ib


def main(argv: Optional[List[str]] = None) -> int:
    ib = _connect()
    if ib is None:
        return 1
    try:
        status = flatten_futures(ib)
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass
    return 0 if status in {"FLAT", "NOTHING", "ABORTED"} else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
