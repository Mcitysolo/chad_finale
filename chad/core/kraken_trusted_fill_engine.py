"""
Kraken TRUSTED paper-fill engine (U1 / CRYPTO-TRUST).

Replaces `validate_only` as the paper EVIDENCE product for the Kraken lane
(validate_only stays as the pre-check in kraken_executor). A trusted fill is:

  * marked against the LIVE Kraken WS tick feed (runtime/kraken_prices.json,
    ticks[symbol].{bid,ask,last}; writer chad/market_data/kraken_ws_client.py:301)
    with a documented slippage model (half-spread + bps impact floor);
  * charged the real Kraken taker fee (config/kraken_trading.json);
  * FIFO round-trip matched so entries + exits realize PnL on close;
  * emitted into the SAME evidence pipeline as IBKR fills with
    fee_model=kraken_paper_v1 and provenance=SIMULATED_AGAINST_LIVE_TICKS,
    carrying NO validate_only / pnl_untrusted flags so Stage-2 ADMITS it.

Fail-closed honesty: a stale/missing touch (age > max_tick_age_seconds) mints
NO trusted fill — the caller falls back to the legacy untrusted evidence row
rather than fabricate a PnL against a dead tape.

Pure core (Touch/slippage/simulate/FIFO book) is stdlib-only and injectable, so
the whole $185 lifecycle is testable with no live feed / broker.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from chad.execution.kraken_trading_config import (
    KrakenTradingConfig,
    get_kraken_trading_config,
)

LOGGER = logging.getLogger("chad.core.kraken_trusted_fill_engine")

# Trust labels (single source of truth; also mirrored near paper_exec_evidence
# _FEE_MODEL_TAG). Collision-checked: neither exists elsewhere in the tree.
FEE_MODEL_KRAKEN_PAPER_V1 = "kraken_paper_v1"
PROVENANCE_SIMULATED_LIVE_TICKS = "SIMULATED_AGAINST_LIVE_TICKS"

_REPO_ROOT = Path("/home/ubuntu/chad_finale")
_EPS = 1e-12

# Kraken REST altname -> CHAD canonical (SOLUSD -> SOL-USD).
_PAIR_TO_CANONICAL: Dict[str, str] = {
    "XBTUSD": "BTC-USD", "ETHUSD": "ETH-USD", "SOLUSD": "SOL-USD",
    "XBTCAD": "BTC-CAD", "ETHCAD": "ETH-CAD",
}


def pair_to_canonical(pair: str) -> str:
    key = (pair or "").strip().upper()
    if key in _PAIR_TO_CANONICAL:
        return _PAIR_TO_CANONICAL[key]
    try:  # fall back to the shared mapper for any pair not in the local table
        from chad.portfolio.kraken_trade_result_logger import _canonical_symbol_for_pair
        return _canonical_symbol_for_pair(pair)
    except Exception:
        return key


# ---------------------------------------------------------------------------
# Live-tick touch
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Touch:
    bid: float
    ask: float
    last: float
    ts_utc: str = ""

    @property
    def mid(self) -> float:
        if self.bid > 0.0 and self.ask > 0.0:
            return (self.bid + self.ask) / 2.0
        return self.last


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _parse_epoch_from_iso(ts: str) -> Optional[float]:
    if not ts:
        return None
    try:
        s = ts.strip().replace("Z", "+00:00")
        from datetime import datetime
        return datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError):
        return None


def read_touch_from_prices(
    prices_obj: Any,
    symbol: str,
    *,
    now_epoch: float,
    max_age_seconds: float,
) -> Optional[Touch]:
    """Extract a fresh Touch for `symbol` from a parsed kraken_prices.json object.

    Fail-closed: returns None on any missing/malformed/stale data.
    """
    if not isinstance(prices_obj, dict):
        return None
    ticks = prices_obj.get("ticks")
    if not isinstance(ticks, dict):
        return None
    row = ticks.get(symbol)
    if not isinstance(row, dict):
        return None
    bid = _num(row.get("bid"))
    ask = _num(row.get("ask"))
    last = _num(row.get("last"))
    if bid is None or ask is None or bid <= 0.0 or ask <= 0.0 or ask < bid:
        # last-only fallback is not trustworthy for a spread-crossing fill.
        return None
    ts_utc = str(row.get("ts_utc") or prices_obj.get("ts_utc") or "")
    # Freshness gate.
    ts_epoch = _parse_epoch_from_iso(ts_utc)
    if ts_epoch is None:
        return None
    age = now_epoch - ts_epoch
    if age < -1.0 or age > float(max_age_seconds):
        return None
    return Touch(bid=bid, ask=ask, last=(last if last is not None else (bid + ask) / 2.0), ts_utc=ts_utc)


class FileTickSource:
    """Reads runtime/kraken_prices.json (cross-process WS snapshot). Read-only."""

    def __init__(self, path: Optional[Path] = None, *, max_age_seconds: float = 30.0) -> None:
        self._path = Path(path) if path is not None else (_REPO_ROOT / "runtime" / "kraken_prices.json")
        self._max_age = float(max_age_seconds)

    def get_touch(self, symbol: str, *, now_epoch: float) -> Optional[Touch]:
        try:
            obj = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return read_touch_from_prices(obj, symbol, now_epoch=now_epoch, max_age_seconds=self._max_age)


# ---------------------------------------------------------------------------
# Slippage + fee -> a trusted fill
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrustedFill:
    strategy: str
    pair: str
    symbol: str
    side: str            # "buy" | "sell"
    qty: float
    fill_price: float
    notional: float
    fee: float
    slippage_bps: float
    mid: float
    touch_ts_utc: str
    fee_model: str = FEE_MODEL_KRAKEN_PAPER_V1
    provenance: str = PROVENANCE_SIMULATED_LIVE_TICKS
    # CRYPTO-EXPLORE-WIRE W3: the live regime this fill was produced in. Stamped onto the
    # opening lot so the harness can slice crypto edge by the regime that was live at ENTRY
    # (the exploration regime of interest). Default "" keeps every legacy construction site
    # unchanged; a blank regime records nothing new.
    regime: str = ""


def model_fill_price(side: str, touch: Touch, impact_floor_bps: float) -> Tuple[float, float]:
    """Marketable fill price + slippage_bps vs mid.

    buy  = mid*(1 + half_spread_frac + impact_frac)  (= ask + mid*impact_frac)
    sell = mid*(1 - half_spread_frac - impact_frac)
    """
    mid = touch.mid
    if mid <= 0.0:
        return 0.0, 0.0
    half_spread_frac = max(0.0, (touch.ask - touch.bid) / 2.0 / mid)
    impact_frac = max(0.0, float(impact_floor_bps) / 1e4)
    cost_frac = half_spread_frac + impact_frac
    s = (side or "").strip().lower()
    if s == "buy":
        fill = mid * (1.0 + cost_frac)
    else:
        fill = mid * (1.0 - cost_frac)
    return fill, cost_frac * 1e4


def simulate_fill(
    *,
    strategy: str,
    pair: str,
    symbol: str,
    side: str,
    qty: float,
    touch: Touch,
    config: KrakenTradingConfig,
    regime: str = "",
) -> TrustedFill:
    fill_price, slippage_bps = model_fill_price(side, touch, config.slippage_impact_floor_bps)
    notional = fill_price * float(qty)
    fee = config.taker_fee(notional, pair)
    return TrustedFill(
        strategy=str(strategy), pair=str(pair), symbol=str(symbol),
        side=(side or "").strip().lower(), qty=float(qty),
        fill_price=fill_price, notional=notional, fee=fee,
        slippage_bps=slippage_bps, mid=touch.mid, touch_ts_utc=touch.ts_utc,
        regime=str(regime or ""),
    )


# ---------------------------------------------------------------------------
# FIFO round-trip lifecycle book (persisted in the consolidated paper store)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RealizedRoundTrip:
    strategy: str
    symbol: str
    direction: str       # "long" | "short"
    qty: float
    entry_price: float
    exit_price: float
    entry_fee: float
    exit_fee: float
    realized_pnl: float
    opened_at_utc: str
    closed_at_utc: str
    # CRYPTO-EXPLORE-WIRE W3: the regime that was live when the CLOSED lot was OPENED, read
    # back from the lot book on FIFO match. This is what _build_trade_history_kwargs stamps
    # onto the round-trip's `regime` field (replacing the old hardcoded "paper"), so a
    # round-trip is attributable to its ENTRY regime — the falsifiable slice exploration exists
    # to produce. Default "" for legacy round-trips minted before the column existed.
    entry_regime: str = ""


def _default_book_db_path() -> Path:
    # Consolidated Kraken-paper store (store #4). NOT the dead exec_state.sqlite3.
    return _REPO_ROOT / "runtime" / "exec_state_paper.sqlite3"


class RoundTripBook:
    """FIFO lot book. One direction per (strategy, symbol) at a time; an opposing
    fill FIFO-closes lots (realizing PnL net of both legs' fees) and flips the
    residual. Persisted as table `kraken_trusted_lots` in the SAME sqlite store
    the paper lane already uses for idempotency (GAP-021 consolidation)."""

    _TABLE = "kraken_trusted_lots"

    def __init__(self, db_path: Optional[Path] = None, *, now_iso: Optional[Callable[[], str]] = None) -> None:
        self._db_path = Path(db_path) if db_path is not None else _default_book_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._now_iso = now_iso or (lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self._ensure()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self._db_path), timeout=5.0, isolation_level=None)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        return con

    def _ensure(self) -> None:
        with contextlib.closing(self._connect()) as con:
            con.execute(
                f"""CREATE TABLE IF NOT EXISTS {self._TABLE} (
                    strategy TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    qty_remaining REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_fee_per_unit REAL NOT NULL,
                    opened_at_utc TEXT NOT NULL
                )"""
            )
            # CRYPTO-EXPLORE-WIRE W3: additive, idempotent regime column so every open lot
            # records the live regime it was opened in. Backward-compatible — pre-existing rows
            # default to '' and every reader in-tree selects explicit columns (the crypto
            # overlay's _default_lots_loader, record()'s own SELECTs, open_qty), so appending a
            # column at the end breaks nothing. ALTER-if-missing keeps a running store safe.
            cols = {r[1] for r in con.execute(f"PRAGMA table_info({self._TABLE})").fetchall()}
            if "regime" not in cols:
                con.execute(
                    f"ALTER TABLE {self._TABLE} ADD COLUMN regime TEXT NOT NULL DEFAULT ''"
                )

    def record(self, fill: TrustedFill) -> List[RealizedRoundTrip]:
        """Apply a fill; return realized round-trips (empty on a pure open)."""
        strategy, symbol = fill.strategy, fill.symbol
        qty = float(fill.qty)
        if qty <= 0.0:
            return []
        fill_opens_long = (fill.side == "buy")
        now = self._now_iso()
        realized: List[RealizedRoundTrip] = []

        fill_regime = str(getattr(fill, "regime", "") or "")
        with contextlib.closing(self._connect()) as con:
            con.execute("BEGIN IMMEDIATE;")
            rows = con.execute(
                f"SELECT rowid, direction, qty_remaining, entry_price, entry_fee_per_unit, "
                f"opened_at_utc, regime "
                f"FROM {self._TABLE} WHERE strategy=? AND symbol=? ORDER BY rowid ASC",
                (strategy, symbol),
            ).fetchall()

            book_dir = rows[0][1] if rows else None  # 'long'|'short'|None
            fill_dir = "long" if fill_opens_long else "short"

            if not rows or book_dir == fill_dir:
                # OPEN (same direction or flat): push a lot, stamping the entry regime (W3).
                con.execute(
                    f"INSERT INTO {self._TABLE} (strategy, symbol, direction, qty_remaining, "
                    f"entry_price, entry_fee_per_unit, opened_at_utc, regime) VALUES (?,?,?,?,?,?,?,?)",
                    (strategy, symbol, fill_dir, qty, fill.fill_price, fill.fee / qty, now,
                     fill_regime),
                )
                con.execute("COMMIT;")
                return []

            # CLOSE (opposite direction): FIFO-match; exit fee allocated pro-rata. The matched
            # lot's stored regime (its ENTRY regime) rides onto the realized round-trip (W3).
            remaining = qty
            for rowid, direction, qty_rem, entry_price, entry_fpu, opened_at, lot_regime in rows:
                if remaining <= _EPS:
                    break
                matched = min(float(qty_rem), remaining)
                entry_fee_alloc = float(entry_fpu) * matched
                exit_fee_alloc = fill.fee * (matched / qty)
                if direction == "long":
                    gross = (fill.fill_price - float(entry_price)) * matched
                else:
                    gross = (float(entry_price) - fill.fill_price) * matched
                realized.append(RealizedRoundTrip(
                    strategy=strategy, symbol=symbol, direction=direction, qty=matched,
                    entry_price=float(entry_price), exit_price=fill.fill_price,
                    entry_fee=entry_fee_alloc, exit_fee=exit_fee_alloc,
                    realized_pnl=gross - entry_fee_alloc - exit_fee_alloc,
                    opened_at_utc=str(opened_at), closed_at_utc=now,
                    entry_regime=str(lot_regime or ""),
                ))
                new_rem = float(qty_rem) - matched
                if new_rem <= _EPS:
                    con.execute(f"DELETE FROM {self._TABLE} WHERE rowid=?", (rowid,))
                else:
                    con.execute(
                        f"UPDATE {self._TABLE} SET qty_remaining=? WHERE rowid=?",
                        (new_rem, rowid),
                    )
                remaining -= matched

            # FLIP: residual opens a new lot in the fill's own direction (its regime = this fill's).
            if remaining > _EPS:
                con.execute(
                    f"INSERT INTO {self._TABLE} (strategy, symbol, direction, qty_remaining, "
                    f"entry_price, entry_fee_per_unit, opened_at_utc, regime) VALUES (?,?,?,?,?,?,?,?)",
                    (strategy, symbol, fill_dir, remaining, fill.fill_price, fill.fee / qty, now,
                     fill_regime),
                )
            con.execute("COMMIT;")
        return realized

    def open_qty(self, strategy: str, symbol: str) -> float:
        with contextlib.closing(self._connect()) as con:
            row = con.execute(
                f"SELECT COALESCE(SUM(qty_remaining),0) FROM {self._TABLE} WHERE strategy=? AND symbol=?",
                (strategy, symbol),
            ).fetchone()
        return float(row[0]) if row else 0.0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _synthetic_execution_id(strategy: str, pair: str, side: str, qty: float, minute_bucket: str) -> str:
    raw = "|".join([strategy, pair, side, f"{qty:.10f}", minute_bucket])
    return "KPT-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _intent_regime(intent: Any) -> str:
    """The live regime carried on an intent (CRYPTO-EXPLORE-WIRE W3).

    Prefers the explicit ``regime`` field stamped at the regime gate; falls back to the
    signal-derived ``regime_state``. Returns "" when neither is populated (e.g. an
    exit-overlay close intent) — the caller records no regime for that fill, and the closing
    round-trip carries the ENTRY regime read back from the lot book instead.
    """
    for attr in ("regime", "regime_state"):
        v = getattr(intent, attr, None)
        if v:
            s = str(v).strip()
            if s:
                return s
    return ""


class TrustedFillEngine:
    """Ties tick source + FIFO book + evidence writers. All deps injectable."""

    def __init__(
        self,
        *,
        config: Optional[KrakenTradingConfig] = None,
        tick_source: Optional[Any] = None,
        book: Optional[RoundTripBook] = None,
        now_fn: Optional[Callable[[], float]] = None,
        dedup: Optional[Callable[[str], bool]] = None,
        evidence_writer: Optional[Callable[..., Any]] = None,
        trade_history_writer: Optional[Callable[..., Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._config = config or get_kraken_trading_config()
        self._tick_source = tick_source or FileTickSource(max_age_seconds=self._config.max_tick_age_seconds)
        self._book = book if book is not None else RoundTripBook()
        self._now_fn = now_fn or time.time
        self._dedup = dedup  # None => no dedup (every call recorded)
        self._evidence_writer = evidence_writer
        self._trade_history_writer = trade_history_writer
        self._log = logger or LOGGER

    # -- lazy default writers (real pipeline) --
    def _write_evidence(self, ev_kwargs: Dict[str, Any]) -> str:
        if self._evidence_writer is not None:
            return str(self._evidence_writer(ev_kwargs) or "")
        from chad.execution.paper_exec_evidence_writer import (
            PaperExecEvidence,
            write_paper_exec_evidence,
        )
        meta = write_paper_exec_evidence(PaperExecEvidence(**ev_kwargs))
        return str((meta or {}).get("fills_path", "") or "")

    def _write_trade_history(self, tr_kwargs: Dict[str, Any]) -> str:
        if self._trade_history_writer is not None:
            return str(self._trade_history_writer(tr_kwargs) or "")
        from chad.analytics.trade_result_logger import TradeResult, log_trade_result
        return str(log_trade_result(TradeResult(**tr_kwargs)) or "")

    def process_intent(self, intent: Any) -> Dict[str, Any]:
        pair = str(getattr(intent, "pair", "") or "")
        symbol = pair_to_canonical(pair)
        side = str(getattr(intent, "side", "") or "").strip().lower()
        try:
            qty = float(getattr(intent, "volume", 0.0) or 0.0)
        except (TypeError, ValueError):
            qty = 0.0
        if side not in ("buy", "sell") or qty <= 0.0:
            return {"trusted": False, "reason": "bad_intent"}

        now = float(self._now_fn())
        minute_bucket = time.strftime("%Y%m%d%H%M", time.gmtime(now))
        exec_id = (
            str(getattr(intent, "idempotency_key", "") or "")
            or str(getattr(intent, "trace_id", "") or "")
            or _synthetic_execution_id(str(getattr(intent, "strategy", "") or ""), pair, side, qty, minute_bucket)
        )

        touch = self._tick_source.get_touch(symbol, now_epoch=now)
        if touch is None:
            # Fail-closed: no fresh tape -> no trusted fill; caller falls back.
            return {"trusted": False, "reason": "no_fresh_touch"}

        if self._dedup is not None:
            dedup_key = _synthetic_execution_id(
                str(getattr(intent, "strategy", "") or ""), pair, side, qty, minute_bucket
            )
            try:
                if not self._dedup(dedup_key):
                    return {"trusted": True, "reason": "dedup", "execution_id": exec_id}
            except Exception as exc:  # dedup store hiccup -> do not double-count the book
                self._log.warning("KRAKEN_TRUSTED_DEDUP_FAILED: %s", exc)
                return {"trusted": False, "reason": "dedup_error"}

        strategy = str(getattr(intent, "strategy", "alpha_crypto") or "alpha_crypto")
        # CRYPTO-EXPLORE-WIRE W3: forward the live regime the intent was gated in (option (a):
        # threaded onto the intent at the regime gate, NO hidden I/O in this pure-ish core).
        # An exit-overlay close intent carries no regime — the round-trip inherits its ENTRY
        # regime from the closed lot instead, which is the more meaningful slice anyway.
        regime = _intent_regime(intent)
        fill = simulate_fill(
            strategy=strategy, pair=pair, symbol=symbol, side=side, qty=qty,
            touch=touch, config=self._config, regime=regime,
        )
        extra_markers = tuple(str(m) for m in (getattr(intent, "markers", ()) or ()) if str(m))

        realized = self._book.record(fill)
        leg = "close" if realized else "open"
        realized_pnl_sum = sum(rt.realized_pnl for rt in realized) if realized else None

        fills_path = self._write_evidence(
            self._build_evidence_kwargs(fill, intent, leg=leg, execution_id=exec_id,
                                        realized_pnl=realized_pnl_sum, extra_markers=extra_markers)
        )
        th_paths: List[str] = []
        for rt in realized:
            th_paths.append(self._write_trade_history(
                self._build_trade_history_kwargs(rt, execution_id=exec_id, extra_markers=extra_markers)
            ))

        self._log.info(
            "KRAKEN_TRUSTED_FILL execution_id=%s strategy=%s symbol=%s side=%s qty=%s "
            "fill_price=%.6f fee=%.6f slippage_bps=%.3f leg=%s realized=%d pnl=%s",
            exec_id, strategy, symbol, side, qty, fill.fill_price, fill.fee,
            fill.slippage_bps, leg, len(realized),
            (f"{realized_pnl_sum:.6f}" if realized_pnl_sum is not None else "n/a"),
        )
        return {
            "trusted": True, "reason": "filled", "execution_id": exec_id,
            "fill_price": fill.fill_price, "fee": fill.fee, "slippage_bps": fill.slippage_bps,
            "notional": fill.notional, "leg": leg,
            "realized": [rt.__dict__ for rt in realized],
            "fills_path": fills_path, "trade_history_paths": th_paths,
        }

    # -- payload builders --
    def _trust_tags(self, extra_markers: Tuple[str, ...]) -> List[str]:
        # DELIBERATELY excludes validate_only / pnl_untrusted so Stage-2 admits.
        return ["kraken_paper", "paper_fill", "trusted_fill",
                PROVENANCE_SIMULATED_LIVE_TICKS] + list(extra_markers)

    def _build_evidence_kwargs(
        self, fill: TrustedFill, intent: Any, *, leg: str, execution_id: str,
        realized_pnl: Optional[float], extra_markers: Tuple[str, ...],
    ) -> Dict[str, Any]:
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._now_fn()))
        return dict(
            symbol=fill.symbol,
            side=fill.side.upper(),
            quantity=fill.qty,
            fill_price=fill.fill_price,
            notional=fill.notional,
            strategy=fill.strategy,
            broker="kraken_paper",
            venue="kraken_paper",
            account_id="KRAKEN_PAPER",
            is_live=False,
            asset_class="crypto",
            # CRYPTO-EXPLORE-WIRE W3: the live regime this fill was produced in (was universally
            # "paper" before — every fill row is now regime-sliceable at the fill level). "" is
            # normalized to "paper" by PaperExecEvidence's own default.
            regime=(fill.regime or "paper"),
            order_type=str(getattr(intent, "ordertype", "market") or "market"),
            status="paper_fill",
            fill_time_utc=now_iso,
            expected_price=fill.mid,
            fee_amount=fill.fee,          # >0 -> suppresses ibkr_fixed_v1 re-stamp
            fee_currency="USD",
            pnl=(float(realized_pnl) if realized_pnl is not None else 0.0),
            slippage_bps=fill.slippage_bps,
            execution_id=execution_id,
            tags=self._trust_tags(extra_markers),
            extra={
                "fee_model": FEE_MODEL_KRAKEN_PAPER_V1,
                "provenance": PROVENANCE_SIMULATED_LIVE_TICKS,
                "trust_state": "TRUSTED",
                "_fee_modeled": True,
                "leg": leg,
                "pair": fill.pair,
                "mid": fill.mid,
                "slippage_bps": fill.slippage_bps,
                "touch_ts_utc": fill.touch_ts_utc,
                "markers": list(extra_markers),
            },
            source="kraken_trusted_fill_engine",
        )

    def _build_trade_history_kwargs(
        self, rt: RealizedRoundTrip, *, execution_id: str, extra_markers: Tuple[str, ...],
    ) -> Dict[str, Any]:
        return dict(
            strategy=rt.strategy,
            symbol=rt.symbol,
            side=("BUY" if rt.direction == "long" else "SELL"),
            quantity=rt.qty,
            fill_price=rt.exit_price,
            notional=rt.exit_price * rt.qty,
            pnl=rt.realized_pnl,
            entry_time_utc=rt.opened_at_utc,
            exit_time_utc=rt.closed_at_utc,
            is_live=False,
            broker="kraken_paper",
            account_id="KRAKEN_PAPER",
            # CRYPTO-EXPLORE-WIRE W3: the round-trip is attributed to the regime that was live
            # when the CLOSED lot was OPENED (read back from the lot book on FIFO match), NOT
            # the old hardcoded "paper". This is the falsifiable edge-by-regime slice. Legacy
            # lots minted before the regime column fall back to "paper".
            regime=(rt.entry_regime or "paper"),
            tags=self._trust_tags(extra_markers),
            extra={
                "fee_model": FEE_MODEL_KRAKEN_PAPER_V1,
                "provenance": PROVENANCE_SIMULATED_LIVE_TICKS,
                "execution_id": execution_id,
                "direction": rt.direction,
                "entry_price": rt.entry_price,
                "exit_price": rt.exit_price,
                "entry_fee": rt.entry_fee,
                "exit_fee": rt.exit_fee,
                "markers": list(extra_markers),
            },
        )
