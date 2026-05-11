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


import math as _math  # local import keeps the module-level surface tidy


def _coerce_finite_positive(v: Any) -> Optional[float]:
    """Return float(v) iff it's a finite, strictly positive number.

    Anything that fails float() (str garbage, None, dicts), or that parses to
    NaN / +inf / -inf / 0 / negative, is rejected so the caller can fall back.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not _math.isfinite(f):
        return None
    if f <= 0.0:
        return None
    return f


def _fill_multiplier(fill: Dict[str, Any]) -> float:
    """Pick the multiplier to use when computing closed-trade PnL.

    Selection rule (first match wins):
      1. If the fill carries a finite-positive ``options_multiplier`` value
         (either at the top level — which is what _extract_fill promotes —
         or under ``extra.options_multiplier``), use it. This is the alpha
         options BAG case where the writer stamped 100x on the record.
      2. If asset_class == "options" AND extra.options_multiplier exists,
         use it (covers fills where the top-level promotion was bypassed
         but the structured ``extra`` is intact).
      3. Otherwise fall back to the per-symbol table via get_multiplier().

    Stocks / futures / crypto are untouched: they don't carry
    options_multiplier, so they hit step 3 unchanged.
    """
    if not isinstance(fill, dict):
        return DEFAULT_MULTIPLIER

    # Step 1: flat top-level options_multiplier (promoted by _extract_fill).
    direct = _coerce_finite_positive(fill.get("options_multiplier"))
    if direct is not None:
        return direct

    # Step 2: structured extra path. Defensive against missing/garbage extra.
    extra = fill.get("extra")
    if isinstance(extra, dict):
        nested = _coerce_finite_positive(extra.get("options_multiplier"))
        if nested is not None:
            asset_class = str(fill.get("asset_class") or "").strip().lower()
            extra_ac = str(extra.get("asset_class") or "").strip().lower()
            if asset_class == "options" or extra_ac == "options":
                return nested
            # No asset_class signal anywhere — still trust the explicit
            # options_multiplier value the writer placed in extra.
            return nested

    # Step 3: fall back to the per-symbol table.
    return get_multiplier(fill.get("symbol", ""))


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


# Statuses we consider trusted enough to feed FIFO matching. Anything else
# (pending, submitted, error, unknown, rejected, cancelled, ...) is skipped.
_TRUSTED_FILL_STATUSES = frozenset({"filled", "paper_fill", "dry_run"})

# Keywords that — when found in a tag, source string, or extra marker —
# indicate the fill is a placeholder/fallback/synthetic rather than a real
# execution. Matched as case-insensitive substrings.
_PLACEHOLDER_MARKERS = (
    "placeholder",
    "fallback",
    "synthetic",
    "untrusted",
)

# Default fill_price the paper executor falls back to when no live market
# price is available. Real fills land on a live market price; a $100.00
# fill_price paired with a placeholder indicator is the canonical fingerprint
# of a phantom fill (see 2026-05-08 SPY delta incident).
_PLACEHOLDER_FILL_PRICE = 100.0


def _str_has_placeholder_marker(value: Any) -> bool:
    if value is None:
        return False
    s = str(value).strip().lower()
    if not s:
        return False
    return any(m in s for m in _PLACEHOLDER_MARKERS)


def _payload_indicates_placeholder(payload: Dict[str, Any]) -> bool:
    """True if tags / source / extra signal a placeholder or fallback fill."""
    tags = payload.get("tags")
    if isinstance(tags, (list, tuple)):
        for t in tags:
            if _str_has_placeholder_marker(t):
                return True
    if _str_has_placeholder_marker(payload.get("source")):
        return True
    extra = payload.get("extra")
    if isinstance(extra, dict):
        for k in (
            "placeholder",
            "is_placeholder",
            "fallback",
            "is_fallback",
            "synthetic",
            "is_synthetic",
        ):
            if bool(extra.get(k)):
                return True
        if _str_has_placeholder_marker(extra.get("source")):
            return True
        if _str_has_placeholder_marker(extra.get("pnl_untrusted_reason")):
            return True
    return False


def _extract_fill(
    obj: Dict[str, Any],
    *,
    quarantined_fill_ids: Optional[set] = None,
    quarantined_record_hashes: Optional[set] = None,
) -> Optional[Dict[str, Any]]:
    """Pull the inner payload of a FILLS_*.ndjson record into a flat dict.

    Skips records that match the sidecar quarantine (data/fills/quarantine_*.json)
    when *quarantined_fill_ids* / *quarantined_record_hashes* are supplied —
    callers in this module pass a set built once per cycle so per-line lookups
    stay O(1).
    """
    payload = obj.get("payload") if isinstance(obj, dict) else None
    if not isinstance(payload, dict):
        payload = obj if isinstance(obj, dict) else None
    if not isinstance(payload, dict):
        return None

    # Sidecar quarantine: drop pinned poisoned fills before any FIFO matching.
    if quarantined_record_hashes:
        rh = obj.get("record_hash") if isinstance(obj, dict) else None
        if isinstance(rh, str) and rh in quarantined_record_hashes:
            return None
    if quarantined_fill_ids:
        fid = payload.get("fill_id")
        if isinstance(fid, str) and fid in quarantined_fill_ids:
            return None

    if payload.get("reject") is True:
        return None
    status = str(payload.get("status", "")).strip().lower()
    if status in ("rejected", "cancelled", "canceled"):
        return None
    # Status allowlist: any non-empty status that is not blessed as trusted
    # (filled / paper_fill / dry_run) is a pending/error/unknown shape and
    # must not enter FIFO matching.
    if status and status not in _TRUSTED_FILL_STATUSES:
        return None

    # Source allowlist: a source string smelling of placeholder/fallback is
    # rejected outright regardless of fill_price.
    if _str_has_placeholder_marker(payload.get("source")):
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

    # Placeholder fill_price guard. The 2026-05-08 incident produced 22 fake
    # delta SELL trades where opening fills landed at fill_price=100.0 with
    # extra.expected_price=100.0 — the executor's "no live price" fallback
    # fired, but the upstream price-sanity check could not catch it because
    # price_cache had no SPY entry to compare against. Defense in depth: if
    # fill_price equals the placeholder default AND any placeholder/fallback/
    # synthetic indicator is present, OR the strategy's expected_price also
    # defaulted to 100.0, refuse the fill.
    px_for_guard = _safe_float(payload.get("fill_price"))
    if px_for_guard is not None and abs(px_for_guard - _PLACEHOLDER_FILL_PRICE) < 1e-9:
        if _payload_indicates_placeholder(payload):
            return None
        if isinstance(extra, dict):
            exp = _safe_float(extra.get("expected_price"))
            if exp is not None and abs(exp - _PLACEHOLDER_FILL_PRICE) < 1e-9:
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

    # Promote options_multiplier and asset_class onto the flat fill dict so
    # _match_fill / _fill_multiplier can apply the per-contract multiplier
    # without re-deriving it from the symbol table. Stocks / futures / crypto
    # records do not carry options_multiplier, so this is a no-op for them.
    options_multiplier: Optional[float] = None
    if isinstance(extra, dict):
        options_multiplier = _coerce_finite_positive(extra.get("options_multiplier"))
    asset_class_top = str(payload.get("asset_class") or "").strip().lower()
    asset_class_extra = ""
    if isinstance(extra, dict):
        asset_class_extra = str(extra.get("asset_class") or "").strip().lower()
    asset_class = asset_class_top or asset_class_extra

    flat: Dict[str, Any] = {
        "fill_id": str(fill_id),
        "strategy": strategy,
        "symbol": symbol,
        "side": side,
        "quantity": float(qty),
        "fill_price": float(px),
        "ts_utc": str(ts) if ts else None,
    }
    if options_multiplier is not None:
        flat["options_multiplier"] = options_multiplier
    if asset_class:
        flat["asset_class"] = asset_class
    return flat


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
        # Always seed processed_fill_ids from any existing trade_history files
        # first. If the state file is missing or corrupted, this still prevents
        # a fill_id already consumed in a prior closed_trade from being matched
        # again into a new closed_trade.
        self.seed_processed_from_trade_history()

        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, dict):
            return

        # Union with what we already seeded from history above.
        self.processed_fill_ids |= set(data.get("processed_fill_ids", []) or [])
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
            "saved_at_utc": _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None).isoformat() + "Z",
        }
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.state_path)

    # ---- duplicate-fill protection --------------------------------------

    def seed_processed_from_trade_history(self) -> int:
        """
        Seed processed_fill_ids from every fill_id already recorded in any
        existing trade_history_*.ndjson under self.trades_dir.

        This is a defense-in-depth check: if the persisted state file is lost
        or rebuilt, a fill_id that was already consumed in a prior closed_trade
        record must not generate another closed_trade on a re-run.

        Returns the number of fill_ids newly added to the processed set.
        """
        added = 0
        if not self.trades_dir.exists():
            return 0
        try:
            history_files = sorted(self.trades_dir.glob("trade_history_*.ndjson"))
        except Exception:
            return 0
        for hpath in history_files:
            try:
                text = hpath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for raw in text.splitlines():
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = obj.get("payload") if isinstance(obj, dict) else None
                if not isinstance(payload, dict):
                    continue
                if payload.get("schema_version") != "closed_trade.v1":
                    continue
                fids = payload.get("fill_ids")
                if not isinstance(fids, (list, tuple)):
                    continue
                for fid in fids:
                    if not fid:
                        continue
                    fid_str = str(fid)
                    if fid_str not in self.processed_fill_ids:
                        self.processed_fill_ids.add(fid_str)
                        added += 1
        return added

    # ---- core processing ------------------------------------------------

    def _match_fill(self, fill: Dict[str, Any]) -> List[ClosedTrade]:
        """Apply a single fill to its FIFO queue and emit any closed trades."""
        key = (fill["strategy"], fill["symbol"])
        queue = self.queues[key]
        # The OPEN-side multiplier is what matters for PnL — a BAG opened at
        # options_multiplier=100 must close at 100 regardless of whether the
        # closing fill's metadata happens to drop the field. We therefore
        # stamp the multiplier on the opening lot below, and prefer the
        # lot's stored multiplier on close. The incoming fill's own
        # multiplier acts as a fallback for legacy lots restored from a
        # state file written before this field existed.
        incoming_multiplier = _fill_multiplier(fill)
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
                    "multiplier": incoming_multiplier,
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
            # Prefer the multiplier stamped on the open-side lot; only fall
            # back to the incoming fill's multiplier when the lot lacks one
            # (legacy state files persisted before lots carried multiplier).
            lot_multiplier = _coerce_finite_positive(lot.get("multiplier"))
            multiplier = lot_multiplier if lot_multiplier is not None else incoming_multiplier
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
                    "multiplier": incoming_multiplier,
                }
            )

        return closed

    def process_fills(self, date_str: str) -> List[ClosedTrade]:
        path = self.fills_dir / f"FILLS_{date_str}.ndjson"
        if not path.exists():
            return []

        # Load sidecar quarantine sets once per call. Failure is non-fatal —
        # the existing placeholder/untrusted checks in _extract_fill remain
        # the primary defense; the sidecar is forensic pinning of known
        # historical pollution.
        quarantined_fill_ids: set = set()
        quarantined_record_hashes: set = set()
        try:
            from chad.analytics.quarantine import get_sidecar_exclusion_sets
            quarantined_fill_ids, quarantined_record_hashes = (
                get_sidecar_exclusion_sets(
                    fills_dir=self.fills_dir,
                    trades_dir=self.trades_dir,
                )
            )
        except Exception:
            quarantined_fill_ids, quarantined_record_hashes = set(), set()

        closed_all: List[ClosedTrade] = []
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            fill = _extract_fill(
                obj,
                quarantined_fill_ids=quarantined_fill_ids,
                quarantined_record_hashes=quarantined_record_hashes,
            )
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
                ts = _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None).isoformat() + "Z"
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
