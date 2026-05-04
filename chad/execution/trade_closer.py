"""
Trade Closer — FIFO position matcher.

Reads paper fills from data/fills/FILLS_{date}.ndjson, matches opening
legs against closing legs FIFO per (strategy, symbol), computes realized
PnL using contract multipliers, and writes closed_trade.v1 records to
data/trades/trade_history_{date}.ndjson so downstream consumers (daily
report, SCR learner, profit lock) see real PnL.

State (open FIFO queues + already-processed fill_ids) is persisted to
runtime/trade_closer_state.json so partial days survive restarts.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import pathlib
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Contract multipliers (mirror chad/execution/ibkr_adapter.py FuturesContractSpec)
# ---------------------------------------------------------------------------

CONTRACT_MULTIPLIERS: Dict[str, float] = {
    # Micro futures
    "MES": 5.0,
    "MNQ": 2.0,
    "MCL": 100.0,
    "MGC": 10.0,
    # ETFs ($1 per share)
    "SPY": 1.0,
    "QQQ": 1.0,
    "GLD": 1.0,
    "TLT": 1.0,
    "IWM": 1.0,
    # Crypto (per coin)
    "BTC-USD": 1.0,
    "ETH-USD": 1.0,
    "SOL-USD": 1.0,
    # Inverse / vol ETFs
    "SVXY": 1.0,
    "UVXY": 1.0,
    "SH": 1.0,
    "PSQ": 1.0,
    # Larger futures
    "ZN": 1000.0,
    "ZB": 1000.0,
    "M6E": 12500.0,
    "SIL": 1000.0,
}
DEFAULT_MULTIPLIER = 1.0


def get_multiplier(symbol: str) -> float:
    if not symbol:
        return DEFAULT_MULTIPLIER
    return CONTRACT_MULTIPLIERS.get(symbol.upper(), DEFAULT_MULTIPLIER)


# ---------------------------------------------------------------------------
# ClosedTrade dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ClosedTrade:
    strategy: str
    symbol: str
    side: str  # direction of the OPENING leg: BUY (long) or SELL (short)
    entry_price: float
    exit_price: float
    quantity: float
    entry_time_utc: str
    exit_time_utc: str
    pnl: float
    contract_multiplier: float
    fill_ids: List[str]
    schema: str = "closed_trade.v1"

    def to_payload(self) -> Dict[str, Any]:
        # Local import: keeps trade_closer's import surface narrow and avoids
        # any analytics→execution import cycle if it ever surfaces.
        from chad.analytics.pnl_breakdown import build_pnl_breakdown

        notional = self.entry_price * self.quantity * self.contract_multiplier
        gross = float(self.pnl)
        # Per-fill cost enrichment is not yet wired into FIFO matching, so
        # commission/slippage/fees are *unknown* — not zero. The breakdown
        # carries cost_basis_status="unavailable" so downstream readers see
        # the uncertainty explicitly instead of inheriting a silent default.
        breakdown = build_pnl_breakdown(
            gross_price_pnl=gross,
            entry_price=self.entry_price,
            exit_price=self.exit_price,
            quantity=self.quantity,
            contract_multiplier=self.contract_multiplier,
            commission=None,
            fees=None,
            slippage=None,
            source="paper_exec",
            cost_basis_status="unavailable",
        )
        net = float(breakdown["net_pnl"])
        # DS07: pnl is deprecated alias for net_pnl; use net_pnl.
        # Top-level commission/slippage are kept as numeric for backwards
        # compatibility with old report tooling that pre-dates pnl_breakdown.
        # When costs are unknown we keep them at 0.0 here ONLY to preserve
        # the existing top-level shape — the canonical truth lives in
        # pnl_breakdown.cost_basis_status / pnl_breakdown.commission etc.
        commission_top = breakdown["commission"] if breakdown["commission"] is not None else 0.0
        slippage_top = breakdown["slippage"] if breakdown["slippage"] is not None else 0.0
        fees_top = breakdown["fees"]  # None preserved; legacy field already nullable
        return {
            "schema_version": self.schema,
            "strategy": self.strategy,
            "symbol": self.symbol,
            "side": self.side,
            "pnl": round(float(self.pnl), 4),
            "gross_pnl": round(gross, 4),
            "commission": commission_top,
            "slippage": slippage_top,
            "fees": fees_top,
            "net_pnl": round(net, 4),
            "entry_time_utc": self.entry_time_utc,
            "exit_time_utc": self.exit_time_utc,
            "fill_price": self.exit_price,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "contract_multiplier": self.contract_multiplier,
            "notional": notional,
            "fill_ids": list(self.fill_ids),
            "broker": "paper_exec",
            "account_id": "PAPER_EXEC",
            "is_live": False,
            "tags": ["paper", "closed", self.strategy],
            "pnl_breakdown": breakdown,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_side(s: Any) -> str:
    return str(s or "").strip().upper()


def _safe_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def _extract_fill(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pull the inner payload of a FILLS_*.ndjson record into a flat dict."""
    payload = obj.get("payload") if isinstance(obj, dict) else None
    if not isinstance(payload, dict):
        payload = obj if isinstance(obj, dict) else None
    if not isinstance(payload, dict):
        return None

    if payload.get("reject") is True:
        return None
    status = str(payload.get("status", "")).lower()
    if status in ("rejected", "cancelled", "canceled"):
        return None

    # Skip untrusted fills (e.g. placeholder fill_price flagged by the
    # paper executor's price-sanity guard). Untrusted fills must not feed
    # into FIFO matching — they would produce phantom realized PnL.
    extra = payload.get("extra") if isinstance(payload, dict) else None
    if isinstance(extra, dict) and bool(extra.get("pnl_untrusted")):
        return None
    if payload.get("pnl_untrusted") is True:
        return None
    tags_list = payload.get("tags") if isinstance(payload, dict) else None
    if isinstance(tags_list, (list, tuple)) and any(
        str(t).strip().lower() == "pnl_untrusted" for t in tags_list
    ):
        return None

    fill_id = payload.get("fill_id")
    if not fill_id:
        # Synthesize a deterministic id so we can de-dup
        seed = json.dumps(
            {
                "strategy": payload.get("strategy"),
                "symbol": payload.get("symbol"),
                "side": payload.get("side"),
                "qty": payload.get("quantity"),
                "px": payload.get("fill_price"),
                "ts": payload.get("fill_time_utc") or payload.get("entry_time_utc"),
            },
            sort_keys=True,
            default=str,
        )
        fill_id = "syn_" + hashlib.sha256(seed.encode()).hexdigest()[:24]

    side = _normalize_side(payload.get("side"))
    if side not in ("BUY", "SELL"):
        return None

    qty = _safe_float(payload.get("quantity"))
    px = _safe_float(payload.get("fill_price"))
    if qty is None or px is None or qty <= 0:
        return None

    strategy = str(payload.get("strategy", "unknown")).strip().lower() or "unknown"
    symbol = str(payload.get("symbol", "")).strip().upper()
    if not symbol:
        return None

    ts = (
        payload.get("fill_time_utc")
        or payload.get("entry_time_utc")
        or obj.get("timestamp_utc")
    )

    return {
        "fill_id": str(fill_id),
        "strategy": strategy,
        "symbol": symbol,
        "side": side,
        "quantity": float(qty),
        "fill_price": float(px),
        "ts_utc": str(ts) if ts else None,
    }


# ---------------------------------------------------------------------------
# TradeCloser
# ---------------------------------------------------------------------------

EXCLUDED_FROM_ROUTING = {"broker_sync", "manual", "paper_exec", "unknown", ""}


class TradeCloser:
    def __init__(
        self,
        fills_dir: pathlib.Path,
        trades_dir: pathlib.Path,
        state_path: pathlib.Path,
        routing_path: Optional[pathlib.Path] = None,
    ) -> None:
        self.fills_dir = pathlib.Path(fills_dir)
        self.trades_dir = pathlib.Path(trades_dir)
        self.state_path = pathlib.Path(state_path)
        self._routing_path = pathlib.Path(routing_path) if routing_path is not None else None
        # queues[(strategy, symbol)] = deque of open lots
        # each lot: {fill_id, side, quantity, fill_price, ts_utc}
        self.queues: Dict[Tuple[str, str], Deque[Dict[str, Any]]] = defaultdict(deque)
        self.processed_fill_ids: set = set()

    # ---- state ----------------------------------------------------------

    def load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, dict):
            return

        self.processed_fill_ids = set(data.get("processed_fill_ids", []) or [])
        self.queues = defaultdict(deque)
        for entry in data.get("queues", []) or []:
            try:
                key = (str(entry["strategy"]), str(entry["symbol"]))
                lots = entry.get("lots", []) or []
                for lot in lots:
                    normalized = dict(lot)
                    if not normalized.get("fill_id"):
                        # Synthesize deterministic id for lots lacking one
                        # (e.g. broker_sync anchor lots rebuilt from a
                        # position snapshot that has no exec_id). Mirrors
                        # _extract_fill's synthesis at line 138-153.
                        seed = json.dumps(
                            {
                                "strategy": key[0],
                                "symbol": key[1],
                                "side": normalized.get("side"),
                                "qty": normalized.get("quantity"),
                                "px": normalized.get("fill_price"),
                                "ts": normalized.get("lot_ts_utc")
                                or normalized.get("ts_utc"),
                            },
                            sort_keys=True,
                            default=str,
                        )
                        normalized["fill_id"] = (
                            "syn_"
                            + hashlib.sha256(seed.encode()).hexdigest()[:24]
                        )
                    self.queues[key].append(normalized)
            except Exception:
                continue

    def save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "processed_fill_ids": sorted(self.processed_fill_ids),
            "queues": [
                {"strategy": k[0], "symbol": k[1], "lots": list(v)}
                for k, v in self.queues.items()
                if v
            ],
            "saved_at_utc": _dt.datetime.utcnow().isoformat() + "Z",
        }
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.state_path)

    # ---- core processing ------------------------------------------------

    def _match_fill(self, fill: Dict[str, Any]) -> List[ClosedTrade]:
        """Apply a single fill to its FIFO queue and emit any closed trades."""
        key = (fill["strategy"], fill["symbol"])
        queue = self.queues[key]
        multiplier = get_multiplier(fill["symbol"])
        closed: List[ClosedTrade] = []

        remaining = fill["quantity"]
        incoming_side = fill["side"]

        # Empty queue OR same-side as the queue head -> opening / adding
        if not queue or queue[0]["side"] == incoming_side:
            queue.append(
                {
                    "fill_id": fill["fill_id"],
                    "side": incoming_side,
                    "quantity": remaining,
                    "fill_price": fill["fill_price"],
                    "ts_utc": fill["ts_utc"],
                }
            )
            return closed

        # Opposite side -> close FIFO against head
        while remaining > 0 and queue and queue[0]["side"] != incoming_side:
            lot = queue[0]
            close_qty = min(lot["quantity"], remaining)

            opening_side = lot["side"]
            entry_price = lot["fill_price"]
            exit_price = fill["fill_price"]
            direction = 1.0 if opening_side == "BUY" else -1.0
            pnl = (exit_price - entry_price) * close_qty * multiplier * direction

            closed.append(
                ClosedTrade(
                    strategy=fill["strategy"],
                    symbol=fill["symbol"],
                    side=opening_side,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=close_qty,
                    entry_time_utc=str(lot.get("ts_utc") or ""),
                    exit_time_utc=str(fill.get("ts_utc") or ""),
                    pnl=pnl,
                    contract_multiplier=multiplier,
                    fill_ids=[lot["fill_id"], fill["fill_id"]],
                )
            )

            lot["quantity"] -= close_qty
            remaining -= close_qty
            if lot["quantity"] <= 1e-12:
                queue.popleft()

        # Any leftover incoming qty becomes a new opening lot in the
        # opposite direction (flip).
        if remaining > 0:
            queue.append(
                {
                    "fill_id": fill["fill_id"],
                    "side": incoming_side,
                    "quantity": remaining,
                    "fill_price": fill["fill_price"],
                    "ts_utc": fill["ts_utc"],
                }
            )

        return closed

    def process_fills(self, date_str: str) -> List[ClosedTrade]:
        path = self.fills_dir / f"FILLS_{date_str}.ndjson"
        if not path.exists():
            return []

        closed_all: List[ClosedTrade] = []
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            fill = _extract_fill(obj)
            if fill is None:
                continue
            if fill["fill_id"] in self.processed_fill_ids:
                continue
            closed_all.extend(self._match_fill(fill))
            self.processed_fill_ids.add(fill["fill_id"])

        return closed_all

    # ---- output ---------------------------------------------------------

    def write_trade_history(
        self, closed_trades: List[ClosedTrade], date_str: str
    ) -> int:
        if not closed_trades:
            return 0
        self.trades_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.trades_dir / f"trade_history_{date_str}.ndjson"

        # Determine starting sequence_id and prev_hash by reading existing tail
        prev_hash = "GENESIS"
        seq = 0
        if out_path.exists():
            try:
                lines = out_path.read_text(
                    encoding="utf-8", errors="ignore"
                ).splitlines()
                for ln in reversed(lines):
                    s = ln.strip()
                    if not s:
                        continue
                    try:
                        last = json.loads(s)
                    except json.JSONDecodeError:
                        continue
                    prev_hash = str(last.get("record_hash") or prev_hash)
                    try:
                        seq = int(last.get("sequence_id") or 0)
                    except Exception:
                        seq = 0
                    break
            except Exception:
                pass

        n = 0
        with out_path.open("a", encoding="utf-8") as fh:
            for ct in closed_trades:
                seq += 1
                payload = ct.to_payload()
                ts = _dt.datetime.utcnow().isoformat() + "Z"
                core = {
                    "payload": payload,
                    "prev_hash": prev_hash,
                    "sequence_id": seq,
                    "timestamp_utc": ts,
                }
                rec_hash = hashlib.sha256(
                    json.dumps(core, sort_keys=True, default=str).encode()
                ).hexdigest()
                core["record_hash"] = rec_hash
                fh.write(json.dumps(core, default=str) + "\n")
                prev_hash = rec_hash
                n += 1
        return n

    # ---- entry points ---------------------------------------------------

    def _route_profits(self, closed: List[ClosedTrade]) -> int:
        """
        Advisory 50/30/20 profit routing for each profitable close.

        Writes to runtime/profit_routing.json as an accounting ledger —
        does NOT transfer capital (single-account paper lane). Failures
        are swallowed: this is ledger sugar on top of the hot path.
        """
        if not closed:
            return 0
        try:
            from chad.risk.profit_router import ProfitRouter  # local to avoid import cycles
            router = ProfitRouter(routing_path=self._routing_path)
        except Exception:  # noqa: BLE001 — router must never break trade_closer
            return 0
        routed = 0
        for ct in closed:
            try:
                if str(ct.strategy or "").lower() in EXCLUDED_FROM_ROUTING:
                    continue  # skip non-strategy bookkeeping labels
                if ct.pnl > 0:
                    router.route_profit(
                        realized_pnl=float(ct.pnl),
                        closing_strategy=str(ct.strategy),
                    )
                    routed += 1
            except Exception:  # noqa: BLE001
                continue
        return routed

    def run_once(self, date_str: Optional[str] = None) -> Dict[str, Any]:
        if date_str is None:
            date_str = _dt.date.today().strftime("%Y%m%d")
        self.load_state()
        closed = self.process_fills(date_str)
        written = self.write_trade_history(closed, date_str)
        routed = self._route_profits(closed)
        self.save_state()
        return {
            "date": date_str,
            "closed_count": len(closed),
            "written": written,
            "total_pnl": sum(c.pnl for c in closed),
            "profits_routed": routed,
        }

    def get_summary(self, date_str: str) -> Dict[str, Any]:
        path = self.trades_dir / f"trade_history_{date_str}.ndjson"
        total_closed = 0
        total_pnl = 0.0
        by_strategy: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"count": 0, "pnl": 0.0}
        )
        if not path.exists():
            return {
                "date": date_str,
                "total_closed": 0,
                "total_pnl": 0.0,
                "by_strategy": {},
            }
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = raw.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            payload = obj.get("payload") if isinstance(obj, dict) else None
            if not isinstance(payload, dict):
                continue
            if payload.get("schema_version") != "closed_trade.v1":
                continue
            try:
                pnl = float(payload.get("pnl", 0.0))
            except (TypeError, ValueError):
                continue
            strat = str(payload.get("strategy", "unknown"))
            total_closed += 1
            total_pnl += pnl
            by_strategy[strat]["count"] += 1
            by_strategy[strat]["pnl"] += pnl
        return {
            "date": date_str,
            "total_closed": total_closed,
            "total_pnl": total_pnl,
            "by_strategy": {k: dict(v) for k, v in by_strategy.items()},
        }


# ---------------------------------------------------------------------------
# CLI entry point (used by chad-trade-closer.service)
# ---------------------------------------------------------------------------

def _default_paths() -> Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    root = pathlib.Path(__file__).resolve().parents[2]
    return (
        root / "data" / "fills",
        root / "data" / "trades",
        root / "runtime" / "trade_closer_state.json",
    )


def main() -> int:
    fills_dir, trades_dir, state_path = _default_paths()
    closer = TradeCloser(
        fills_dir=fills_dir, trades_dir=trades_dir, state_path=state_path
    )
    result = closer.run_once()
    print(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
