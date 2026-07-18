"""chad/risk/crypto_exit_overlay.py — position-aware CRYPTO (Kraken) exit overlay (SHADOW-first).

UC1/U2-A. The Kraken-lane sibling of ``chad.risk.position_exit_overlay``.

Why this exists
---------------
The exit audit (``ops/pending_actions/EXIT_AUDIT_equity_roundtrip_close_2026-07-13.md``)
established that CHAD accumulates inventory it never deterministically exits. That disease is
not equity-specific: ``alpha_crypto`` opens lots in the Kraken trusted-fill book and nothing
closes them on a risk rule. This overlay is the crypto exit — the same three OR'd conditions
(hard-stop -> ATR-trailing -> max-hold, first hit wins), evaluated against the Kraken FIFO lot
store, proposing **reduce-only** closes.

What is REUSED (never forked)
-----------------------------
- ``position_exit_overlay.evaluate_exit_conditions`` — the one pure exit kernel. Both lanes
  share it, so crypto cannot drift from equity semantics.
- ``position_exit_overlay._atr`` — Wilder ATR over ``data/bars/1d/{SYMBOL}.json`` (BTC/ETH/
  SOL-USD bars already exist).
- ``ExitOverlayVerdict`` / ``ExitOverlayResult`` / ``load_overlay_config`` / ``resolve_mode``
  and the ``exit_overlay.v1`` evidence schema.
- ``kraken_trusted_fill_engine`` for the mark (``read_touch_from_prices`` over
  ``runtime/kraken_prices.json``, fail-closed on stale/crossed) and, in ACTIVE, for execution
  (``TrustedFillEngine.process_intent`` — real taker fee, FIFO realization, Stage-2-admissible
  evidence).

Three ways the crypto lane is DELIBERATELY better than the equity lane
----------------------------------------------------------------------
1. **Real entry price.** The equity guard carries no cost basis, so the equity overlay anchors
   ``entry_price`` to the first price it happens to observe. The Kraken lot book carries a real
   ``entry_price`` per lot, so the crypto hard-stop is measured from a genuine, fee-inclusive
   volume-weighted basis — an economically meaningful stop.
2. **Real age.** ``opened_at_utc`` per lot is the true open time, so max-hold measures the real
   holding period rather than "time since a state file was last rebuilt".
3. **Anchors MERGE, never replace.** The equity overlay's ``_save_anchors`` rewrites the state
   file with only the keys seen this cycle, so one skipped/blind cycle silently destroys a
   position's ``peak`` and re-seeds it at spot (live-proven twice on ``gamma|UNH``; see the
   ULTRA-CLOSE audit F1). This lane merges and prunes only on a *confirmed-flat* book, so a
   transient blind cycle cannot erase the trail it depends on.

Safety
------
- Mode ``off|shadow|active`` from ``config/crypto_exit_overlay.json``; env kill-switch
  ``CHAD_CRYPTO_EXIT_OVERLAY`` overrides. Default **shadow** (evaluates + evidences, closes
  NOTHING).
- **Reduce-only is enforced here because the book will not enforce it.**
  ``RoundTripBook.record`` FIFO-matches an opposing fill and then *flips the residual into a new
  opposite lot* — an oversized SELL silently opens a short. That is the INCIDENT-0713 TLT
  flip/oversell mechanism. Every close is clamped to ``book.open_qty(strategy, symbol)`` at
  evaluation AND re-clamped against a freshly-read book immediately before dispatch.
- Fail-closed on a missing/stale/crossed tick (``SKIP_NO_DATA``) and on an empty book
  (``SKIP_UNCONFIRMED``).
- The evaluation core is stdlib-only and pure; chad imports are lazy, inside the runner.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from chad.risk.position_exit_overlay import (
    EVIDENCE_SCHEMA_VERSION,
    MODE_ACTIVE,
    MODE_OFF,
    ExitOverlayConfig,
    ExitOverlayConfigError,
    ExitOverlayResult,
    ExitOverlayVerdict,
    _atr,
    _f,
    _iso,
    _parse_iso,
    evaluate_exit_conditions,
    load_overlay_config,
    resolve_mode,
)

__all__ = [
    "CRYPTO_KILL_SWITCH_ENV",
    "CRYPTO_ASSET_CLASS",
    "CryptoLotSnapshot",
    "aggregate_lots",
    "evaluate_crypto_positions",
    "CryptoExitOverlay",
    "build_default_crypto_overlay",
]

LOGGER = logging.getLogger("chad.risk.crypto_exit_overlay")

CRYPTO_KILL_SWITCH_ENV = "CHAD_CRYPTO_EXIT_OVERLAY"
CRYPTO_ASSET_CLASS = "crypto"
CRYPTO_CONFIG_SCHEMA_VERSION = "crypto_exit_overlay_config.v1"

MARKER_SHADOW = "CRYPTO_EXIT_OVERLAY_SHADOW"
MARKER_SKIP = "CRYPTO_EXIT_OVERLAY_SKIP"
MARKER_ERROR = "CRYPTO_EXIT_OVERLAY_ERROR"
MARKER_ACTIVE = "CRYPTO_EXIT_OVERLAY_ACTIVE_CLOSE"
MARKER_HEARTBEAT = "CRYPTO_EXIT_OVERLAY_HEARTBEAT"

HEARTBEAT_SCHEMA_VERSION = "crypto_exit_overlay_heartbeat.v1"
# Separate heartbeat file from the equity lane on purpose: two overlays sharing one heartbeat
# path would clobber each other's liveness proof, and "crypto stalled" must stay distinguishable
# from "equity stalled" (the XOV-2345 lesson — a dead watcher must never look healthy).
HEARTBEAT_TTL_SECONDS = 900
_DEFAULT_HEARTBEAT_RELPATH = ("runtime", "crypto_exit_overlay_heartbeat.json")
_DEFAULT_STATE_RELPATH = ("runtime", "crypto_exit_overlay_state.json")
_DEFAULT_EVIDENCE_SUBDIR = ("data", "exit_overlay")

# Crypto marks must be live. 24/7 tape with no RTH gate ⇒ a stale tick is never "the market is
# closed", it is always a broken feed. Fail closed.
_DEFAULT_MAX_TICK_AGE_SECONDS = 60.0


# --------------------------------------------------------------------------- #
# Lot aggregation (pure)
# --------------------------------------------------------------------------- #
class CryptoLotSnapshot:
    """One (strategy, symbol) aggregate of the Kraken FIFO lot book."""

    __slots__ = ("strategy", "symbol", "direction", "qty", "entry_price", "opened_at_utc", "lots")

    def __init__(
        self,
        *,
        strategy: str,
        symbol: str,
        direction: str,
        qty: float,
        entry_price: float,
        opened_at_utc: str,
        lots: int,
    ) -> None:
        self.strategy = strategy
        self.symbol = symbol
        self.direction = direction
        self.qty = qty
        self.entry_price = entry_price
        self.opened_at_utc = opened_at_utc
        self.lots = lots

    @property
    def key(self) -> str:
        return f"{self.strategy}|{self.symbol}"

    @property
    def side(self) -> str:
        """Guard-style side: a long book is a BUY position, a short book a SELL position."""
        return "BUY" if self.direction == "long" else "SELL"


def aggregate_lots(rows: Sequence[Sequence[Any]]) -> List[CryptoLotSnapshot]:
    """Aggregate raw ``kraken_trusted_lots`` rows into per-(strategy, symbol) snapshots.

    Row shape (rowid ASC == FIFO order):
      ``(strategy, symbol, direction, qty_remaining, entry_price, entry_fee_per_unit, opened_at_utc)``

    ``entry_price`` is the **fee-inclusive volume-weighted** basis across the open lots — the
    real cost basis, which is what a hard stop must measure from. ``opened_at_utc`` is the
    EARLIEST open (the oldest surviving lot), so max-hold measures the true holding period of
    the position rather than of its most recent add.
    """
    buckets: Dict[Tuple[str, str], List[Sequence[Any]]] = {}
    for r in rows:
        if len(r) < 7:
            continue
        strategy, symbol = str(r[0] or ""), str(r[1] or "")
        if not strategy or not symbol:
            continue
        if _f(r[3]) <= 0.0:
            continue
        buckets.setdefault((strategy, symbol), []).append(r)

    out: List[CryptoLotSnapshot] = []
    for (strategy, symbol), lots in sorted(buckets.items()):
        qty = sum(_f(l[3]) for l in lots)
        if qty <= 0.0:
            continue
        # Fee-inclusive VWAP basis: (price + entry_fee_per_unit) weighted by remaining qty.
        notional = sum(_f(l[3]) * (_f(l[4]) + _f(l[5])) for l in lots)
        entry_price = notional / qty if qty > 0.0 else 0.0
        opened = min(str(l[6] or "") for l in lots)
        directions = {str(l[2] or "") for l in lots}
        # A well-formed book holds one direction per (strategy, symbol). If it somehow holds
        # both, refuse to guess — emit the majority-qty direction and let the reduce-only
        # clamp bound the damage.
        if len(directions) == 1:
            direction = directions.pop()
        else:
            by_dir: Dict[str, float] = {}
            for l in lots:
                by_dir[str(l[2] or "")] = by_dir.get(str(l[2] or ""), 0.0) + _f(l[3])
            direction = max(by_dir.items(), key=lambda kv: kv[1])[0]
        out.append(CryptoLotSnapshot(
            strategy=strategy, symbol=symbol, direction=direction, qty=qty,
            entry_price=entry_price, opened_at_utc=opened, lots=len(lots),
        ))
    return out


def _age_days_from(opened_at_utc: str, now: datetime) -> Optional[float]:
    opened = _parse_iso(opened_at_utc)
    if opened is None:
        return None
    return max(0.0, (now - opened).total_seconds() / 86400.0)


# --------------------------------------------------------------------------- #
# Pure evaluation core
# --------------------------------------------------------------------------- #
def evaluate_crypto_positions(
    *,
    snapshots: Sequence[CryptoLotSnapshot],
    marks_by_symbol: Mapping[str, float],
    bars_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
    anchors: Mapping[str, Mapping[str, Any]],
    config: ExitOverlayConfig,
    now_utc: datetime,
) -> ExitOverlayResult:
    """Deterministic core: lot snapshots + marks + bars + anchors -> verdicts + close intents.

    Pure. No I/O, no mutation of inputs. ``updated_anchors`` carries ONLY the keys evaluated
    this cycle; the runner MERGES them (it never replaces the file wholesale — see module
    docstring point 3).
    """
    ts = _iso(now_utc)
    verdicts: List[ExitOverlayVerdict] = []
    close_intents: List[Dict[str, Any]] = []
    updated_anchors: Dict[str, Dict[str, Any]] = {}

    for snap in snapshots:
        key, symbol, side = snap.key, snap.symbol, snap.side
        max_hold = config.max_hold_for(CRYPTO_ASSET_CLASS)

        def _mk(verdict: str, reason: str, **kw: Any) -> ExitOverlayVerdict:
            base: Dict[str, Any] = dict(
                verdict=verdict, reason=reason, position_key=key, symbol=symbol,
                strategy=snap.strategy, side=side, asset_class=CRYPTO_ASSET_CLASS,
                open_qty=snap.qty, close_qty=0.0, price=None, atr=None, atr_stop=None,
                entry_price=snap.entry_price, hard_stop_price=None, peak=None, trough=None,
                age_days=None, max_hold_days=max_hold,
                # The lot book IS the position truth for this lane (Kraken paper has no
                # separate broker leg), so "broker-confirmed" == open lot qty.
                broker_confirmed_qty=snap.qty, ts_utc=ts,
            )
            base.update(kw)
            return ExitOverlayVerdict(**base)

        if snap.qty <= 0.0:
            verdicts.append(_mk("SKIP_UNCONFIRMED", "no_open_lots", broker_confirmed_qty=0.0))
            continue

        price = _f(marks_by_symbol.get(symbol))
        if price <= 0.0:
            # Fail-closed: 24/7 tape ⇒ a missing mark is a broken feed, never a closed market.
            verdicts.append(_mk("SKIP_NO_DATA", "no_fresh_mark"))
            continue

        prior = anchors.get(key) if isinstance(anchors.get(key), Mapping) else None
        peak = max(_f(prior.get("peak")) if prior else 0.0, price)
        trough_raw = _f(prior.get("trough")) if prior else 0.0
        trough = min(trough_raw, price) if trough_raw > 0.0 else price
        first_seen = (prior.get("first_seen_utc") if prior else None) or ts
        updated_anchors[key] = {
            # entry_price is NOT anchored here — it comes from the lot book every cycle, so it
            # cannot be fabricated or lost. Stored only for observability.
            "entry_price": snap.entry_price,
            "peak": peak,
            "trough": trough,
            "first_seen_utc": first_seen,
            "last_seen_utc": ts,
            "opened_at_utc": snap.opened_at_utc,
        }

        bars = bars_by_symbol.get(symbol) or []
        atr = _atr(bars, config.atr_period) if len(list(bars)) >= config.min_bars_for_atr else None
        age = _age_days_from(snap.opened_at_utc, now_utc)

        fired, hard_stop_price, atr_stop = evaluate_exit_conditions(
            side=side, price=price, entry_price=snap.entry_price, peak=peak, trough=trough,
            atr=atr, age_days=age, max_hold_days=max_hold, config=config,
        )

        common = dict(
            price=price, atr=atr, atr_stop=atr_stop, entry_price=snap.entry_price,
            hard_stop_price=hard_stop_price, peak=peak, trough=trough, age_days=age,
            broker_confirmed_qty=snap.qty,
        )
        if fired is None:
            verdicts.append(_mk("HOLD", "no_condition_met", **common))
            continue

        # Reduce-only: close exactly the open lot qty, never more. The lot book IS the position
        # truth for this lane, so the propose-step qty cannot exceed it by construction; the
        # load-bearing clamp is the re-read in reduce_only_reclamp_crypto immediately before
        # dispatch, which is where a concurrent fill could otherwise cause an over-sell. The
        # book FLIPS a residual into an opposite lot, so an unclamped close silently reverses
        # the position (INCIDENT-0713).
        close_qty = snap.qty
        common["close_qty"] = close_qty
        verdicts.append(_mk("WOULD_CLOSE", fired, **common))
        close_intents.append({
            "symbol": symbol,
            "action": "CLOSE",
            "open_side": side,
            "close_side": "SELL" if side == "BUY" else "BUY",
            "quantity": float(close_qty),
            "reason": f"crypto_exit_overlay_{fired}",
            "position_key": key,
            "strategy": snap.strategy,
            "asset_class": CRYPTO_ASSET_CLASS,
        })

    return ExitOverlayResult(
        verdicts=verdicts,
        close_intents=close_intents,
        updated_anchors=updated_anchors,
        evaluated=True,
    )


# --------------------------------------------------------------------------- #
# Reduce-only re-clamp (pure) — the last line before dispatch
# --------------------------------------------------------------------------- #
def reduce_only_reclamp_crypto(
    close_intents: Sequence[Mapping[str, Any]],
    open_qty_by_key: Mapping[str, float],
) -> List[Dict[str, Any]]:
    """Re-clamp each close against a FRESHLY-read lot book immediately before dispatch.

    Drops any intent whose book has since gone flat. This is not belt-and-braces: because
    ``RoundTripBook.record`` flips the residual, dispatching a close for qty > open_qty does not
    error — it opens a short. INCIDENT-0713 (TLT) is exactly that failure in the equity lane.
    """
    out: List[Dict[str, Any]] = []
    for c in close_intents:
        key = str(c.get("position_key") or "")
        held = _f(open_qty_by_key.get(key))
        if held <= 0.0:
            LOGGER.warning(
                "%s reclamp_drop key=%s reason=book_now_flat", MARKER_ERROR, key,
            )
            continue
        qty = min(_f(c.get("quantity")), held)
        if qty <= 0.0:
            continue
        d = dict(c)
        d["quantity"] = float(qty)
        out.append(d)
    return out


# --------------------------------------------------------------------------- #
# Dispatch intent (duck-typed for TrustedFillEngine.process_intent)
# --------------------------------------------------------------------------- #
class _CryptoCloseIntent:
    """Minimal duck-typed intent matching what ``TrustedFillEngine.process_intent`` reads."""

    __slots__ = ("pair", "side", "volume", "strategy", "ordertype", "markers",
                 "idempotency_key", "trace_id", "reduce_only")

    def __init__(
        self, *, pair: str, side: str, volume: float, strategy: str, reason: str,
        idempotency_key: str,
    ) -> None:
        self.pair = pair
        self.side = side
        self.volume = float(volume)
        self.strategy = strategy
        self.ordertype = "market"
        self.markers = (MARKER_ACTIVE, reason)
        self.idempotency_key = idempotency_key
        self.trace_id = idempotency_key
        self.reduce_only = True


def _canonical_to_pair(symbol: str) -> str:
    """Canonical symbol -> the Kraken pair form ``process_intent`` round-trips.

    Inverts the engine's OWN ``_PAIR_TO_CANONICAL`` table rather than string-munging, because
    the mapping is not mechanical: ``BTC-USD`` must be dispatched as ``XBTUSD`` (Kraken's XBT
    naming), and a slash form (``SOL/USD``) does NOT round-trip — ``pair_to_canonical`` passes
    it through unchanged, so the fill would be booked under symbol ``SOL/USD`` while the
    overlay reads lots under ``SOL-USD``: the close would silently open a NEW position instead
    of reducing the old one. Verified against the live table:
    ``{XBTUSD: BTC-USD, ETHUSD: ETH-USD, SOLUSD: SOL-USD, ...}``.
    """
    try:
        from chad.core.kraken_trusted_fill_engine import _PAIR_TO_CANONICAL

        for pair, canonical in _PAIR_TO_CANONICAL.items():
            if canonical == symbol:
                return pair
    except Exception:  # noqa: BLE001 - fall through to the identity form
        pass
    # The canonical form itself round-trips via the shared mapper fallback.
    return symbol


def _close_idempotency_key(strategy: str, symbol: str, side: str, qty: float) -> str:
    """Deterministic from (strategy, symbol, side, qty) — NO timestamp, deliberately.

    Mirrors ``ibkr_adapter._stable_idempotency_payload``: a timestamped key would mint a fresh
    order every cycle (the overlay re-proposes until the book actually moves), so collision IS
    the safety property. The engine's own per-minute dedup bucket is the second layer.
    """
    return f"crypto_exit|{strategy}|{symbol}|{side}|{qty:.8f}"


# --------------------------------------------------------------------------- #
# Runner (impure)
# --------------------------------------------------------------------------- #
LotsLoader = Callable[[], Sequence[Sequence[Any]]]
MarksLoader = Callable[[Sequence[str]], Mapping[str, float]]
BarsLoader = Callable[[Sequence[str]], Mapping[str, Sequence[Mapping[str, Any]]]]
OpenQtyLoader = Callable[[str, str], float]


class CryptoExitOverlay:
    """Wires the pure crypto core to the lot book, the Kraken tape, evidence, anchor
    persistence, and (only in ACTIVE) the trusted-fill engine. ``run_cycle`` never raises."""

    def __init__(
        self,
        config: ExitOverlayConfig,
        *,
        evidence_path: Optional[Path],
        state_path: Optional[Path],
        lots_loader: LotsLoader,
        marks_loader: MarksLoader,
        bars_loader: BarsLoader,
        open_qty_loader: Optional[OpenQtyLoader] = None,
        engine_factory: Optional[Callable[[], Any]] = None,
        heartbeat_path: Optional[Path] = None,
        env: Optional[Mapping[str, str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._config = config
        self._evidence_path = evidence_path
        self._state_path = state_path
        self._heartbeat_path = heartbeat_path
        self._lots_loader = lots_loader
        self._marks_loader = marks_loader
        self._bars_loader = bars_loader
        self._open_qty_loader = open_qty_loader
        self._engine_factory = engine_factory
        self._env = os.environ if env is None else env
        self._log = logger or LOGGER

    @property
    def mode(self) -> str:
        return resolve_mode(self._config.mode, self._env, env_var=CRYPTO_KILL_SWITCH_ENV)

    def run_cycle(self, *, now_utc: Optional[datetime] = None) -> ExitOverlayResult:
        mode = self.mode
        now = now_utc or datetime.now(timezone.utc)
        if mode == MODE_OFF:
            self._safe_heartbeat(mode, now, evaluated=0, would_close=0, healthy=True)
            return ExitOverlayResult(evaluated=False)
        try:
            rows = self._lots_loader() or []
            snapshots = aggregate_lots(rows)
            symbols = sorted({s.symbol for s in snapshots})
            marks = self._marks_loader(symbols) if symbols else {}
            bars = self._bars_loader(symbols) if symbols else {}
            anchors = self._load_anchors()
            result = evaluate_crypto_positions(
                snapshots=snapshots, marks_by_symbol=marks, bars_by_symbol=bars,
                anchors=anchors, config=self._config, now_utc=now,
            )
        except Exception as exc:  # noqa: BLE001 - never fatal; submits nothing on error
            self._safe_error(exc)
            self._safe_heartbeat(mode, now, evaluated=0, would_close=0, healthy=False)
            return ExitOverlayResult(evaluated=False)

        for v in result.verdicts:
            self._safe_marker(v, mode)
            self._safe_write_evidence(v)
        # MERGE (never replace): a blind cycle must not erase a live position's trail.
        # A book that reads back EMPTY is treated as blind, NOT as confirmed-flat: an
        # all-positions-vanished read is exactly the shape of the equity lane's live-proven
        # false-flat (ULTRA-CLOSE F1), and pruning on it would reproduce the very wipe this
        # lane exists to avoid. Pruning requires a book that returned SOME position and simply
        # no longer lists this key — i.e. a real, observed close.
        self._save_anchors(
            result.updated_anchors,
            live_keys={s.key for s in snapshots} if snapshots else None,
        )
        self._safe_heartbeat(
            mode, now, evaluated=len(result.verdicts),
            would_close=len(result.close_intents), healthy=True,
        )

        if mode == MODE_ACTIVE and result.close_intents:
            self._submit_active(result.close_intents)
        return result

    # -- active submit -------------------------------------------------------- #
    def _submit_active(self, close_intents: List[Dict[str, Any]]) -> None:
        try:
            if self._open_qty_loader is not None:
                fresh = {
                    str(c.get("position_key")): self._open_qty_loader(
                        str(c.get("strategy") or ""), str(c.get("symbol") or "")
                    )
                    for c in close_intents
                }
                intents = reduce_only_reclamp_crypto(close_intents, fresh)
            else:
                intents = list(close_intents)
            if not intents:
                return
            engine = self._engine_factory() if self._engine_factory is not None else None
            if engine is None:
                self._log.error("%s no_engine (submitting nothing)", MARKER_ERROR)
                return
            for c in intents:
                symbol = str(c.get("symbol") or "")
                strategy = str(c.get("strategy") or "")
                side = str(c.get("close_side") or "").lower()
                qty = float(c.get("quantity") or 0.0)
                self._log.info(
                    "%s symbol=%s strategy=%s side=%s qty=%s reason=%s",
                    MARKER_ACTIVE, symbol, strategy, side, qty, c.get("reason"),
                )
                intent = _CryptoCloseIntent(
                    pair=_canonical_to_pair(symbol), side=side, volume=qty,
                    strategy=strategy, reason=str(c.get("reason") or ""),
                    idempotency_key=_close_idempotency_key(strategy, symbol, side, qty),
                )
                res = engine.process_intent(intent)
                if not (isinstance(res, Mapping) and res.get("trusted")):
                    self._log.warning(
                        "%s untrusted_close symbol=%s reason=%s (no fill booked)",
                        MARKER_ERROR, symbol,
                        (res or {}).get("reason") if isinstance(res, Mapping) else "no_result",
                    )
        except Exception as exc:  # noqa: BLE001 - a submit error must not stop the loop
            self._safe_error(exc)

    # -- anchors -------------------------------------------------------------- #
    def _load_anchors(self) -> Dict[str, Dict[str, Any]]:
        if self._state_path is None or not self._state_path.is_file():
            return {}
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
            anchors = raw.get("anchors") if isinstance(raw, dict) else None
            return anchors if isinstance(anchors, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    def _save_anchors(
        self, updated: Mapping[str, Dict[str, Any]], *, live_keys: Optional[set] = None
    ) -> None:
        """MERGE this cycle's anchors over the persisted set; prune only CONFIRMED-flat keys.

        The equity lane replaces the file with only what it saw this cycle, so a single blind
        cycle (stale mark, empty book read, transient loader failure) silently destroys the
        accumulated ``peak`` and re-seeds the trail at spot. That is live-proven (ULTRA-CLOSE
        F1). Here a key is dropped ONLY when the lot book was successfully read and no longer
        holds it — i.e. the position genuinely closed — never merely because it was not
        evaluated.
        """
        if self._state_path is None:
            return
        try:
            merged = self._load_anchors()
            merged.update({k: dict(v) for k, v in updated.items()})
            if live_keys is not None:
                for k in [k for k in merged if k not in live_keys]:
                    merged.pop(k, None)
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
            payload = {
                "schema_version": "crypto_exit_overlay_state.v1",
                "anchors": merged,
            }
            tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self._state_path)
        except Exception:  # noqa: BLE001 - anchor persistence is best-effort
            pass

    # -- observability -------------------------------------------------------- #
    def _safe_marker(self, verdict: ExitOverlayVerdict, mode: str) -> None:
        try:
            marker = MARKER_SKIP if verdict.verdict.startswith("SKIP") else MARKER_SHADOW
            self._log.info(
                "%s verdict=%s reason=%s symbol=%s strategy=%s side=%s close_qty=%s "
                "price=%s atr_stop=%s hard_stop=%s age=%s mode=%s",
                marker, verdict.verdict, verdict.reason, verdict.symbol or "-",
                verdict.strategy or "-", verdict.side or "-", verdict.close_qty,
                verdict.price, verdict.atr_stop, verdict.hard_stop_price,
                verdict.age_days, mode,
                extra={"crypto_exit_overlay": verdict.to_dict()},
            )
        except Exception:  # noqa: BLE001
            pass

    def _safe_error(self, exc: Exception) -> None:
        try:
            self._log.error(
                "%s detail=%s: %s (submitting nothing)", MARKER_ERROR, type(exc).__name__, exc
            )
        except Exception:  # noqa: BLE001
            pass

    def _safe_heartbeat(
        self, mode: str, now: datetime, *, evaluated: int, would_close: int, healthy: bool
    ) -> None:
        try:
            self._log.info(
                "%s evaluated=%d mode=%s would_close=%d healthy=%s",
                MARKER_HEARTBEAT, evaluated, mode, would_close, healthy,
            )
        except Exception:  # noqa: BLE001
            pass
        if self._heartbeat_path is None:
            return
        try:
            payload = {
                "schema_version": HEARTBEAT_SCHEMA_VERSION,
                "ts_utc": _iso(now),
                "ttl_seconds": HEARTBEAT_TTL_SECONDS,
                "mode": mode,
                "evaluated": int(evaluated),
                "would_close": int(would_close),
                "healthy": bool(healthy),
            }
            self._heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._heartbeat_path.with_suffix(self._heartbeat_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self._heartbeat_path)
        except Exception:  # noqa: BLE001
            pass

    def _safe_write_evidence(self, verdict: ExitOverlayVerdict) -> None:
        path = self._resolve_evidence_path(verdict)
        if path is None:
            return
        try:
            row = verdict.to_dict()
            # Lane tag: crypto verdicts write their OWN file (crypto_*.ndjson, W1) in the shared
            # data/exit_overlay directory. The tag is belt-and-braces so a reader that globs the
            # whole directory can still separate lanes without inferring from symbol.
            row["lane"] = "crypto"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
        except Exception:  # noqa: BLE001
            pass

    def _resolve_evidence_path(self, verdict: ExitOverlayVerdict) -> Optional[Path]:
        if self._evidence_path is None:
            return None
        base = self._evidence_path
        if base.suffix == ".ndjson":
            return base
        # CRYPTO-EXPLORE-WIRE W1: OWN evidence file (data/exit_overlay/crypto_*.ndjson), never
        # the equity lane's exit_overlay_*.ndjson. Same directory, distinct filename — so a
        # reader never has to disambiguate two lanes appending to one file, and the crypto
        # trail cannot be interleaved with (or mistaken for) the equity trail.
        day = verdict.ts_utc[:10].replace("-", "") if verdict.ts_utc else "19700101"
        return base / f"crypto_exit_overlay_{day}.ndjson"


# --------------------------------------------------------------------------- #
# Default loaders + factory
# --------------------------------------------------------------------------- #
def default_evidence_dir(repo_root: Path) -> Path:
    return repo_root.joinpath(*_DEFAULT_EVIDENCE_SUBDIR)


def default_heartbeat_path(repo_root: Path) -> Path:
    return repo_root.joinpath(*_DEFAULT_HEARTBEAT_RELPATH)


def _default_state_path(repo_root: Path) -> Path:
    return repo_root.joinpath(*_DEFAULT_STATE_RELPATH)


def _default_lots_loader(book: Any) -> LotsLoader:
    def _load() -> Sequence[Sequence[Any]]:
        import contextlib
        import sqlite3

        db = book._db_path  # noqa: SLF001 - the book owns the path; read-only reuse
        with contextlib.closing(sqlite3.connect(str(db), timeout=5.0)) as con:
            return con.execute(
                "SELECT strategy, symbol, direction, qty_remaining, entry_price, "
                "entry_fee_per_unit, opened_at_utc FROM kraken_trusted_lots ORDER BY rowid ASC"
            ).fetchall()

    return _load


def _default_marks_loader(prices_path: Path, max_age_seconds: float) -> MarksLoader:
    def _load(symbols: Sequence[str]) -> Mapping[str, float]:
        from chad.core.kraken_trusted_fill_engine import read_touch_from_prices

        try:
            obj = json.loads(prices_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        now = time.time()
        out: Dict[str, float] = {}
        for sym in symbols:
            touch = read_touch_from_prices(
                obj, sym, now_epoch=now, max_age_seconds=max_age_seconds
            )
            if touch is not None:
                out[sym] = float(touch.mid)
        return out

    return _load


def _default_bars_loader(repo_root: Path) -> BarsLoader:
    def _load(symbols: Sequence[str]) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        out: Dict[str, Sequence[Mapping[str, Any]]] = {}
        for sym in symbols:
            p = repo_root / "data" / "bars" / "1d" / f"{sym}.json"
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            bars = raw.get("bars") if isinstance(raw, dict) else raw
            if isinstance(bars, list):
                out[sym] = [b for b in bars if isinstance(b, Mapping)]
        return out

    return _load


def _under_active_pytest() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ or "pytest" in os.environ.get("_", "")


def build_default_crypto_overlay(
    *,
    repo_root: Path,
    config_path: Optional[Path] = None,
    evidence_dir: Optional[Path] = None,
    state_path: Optional[Path] = None,
    heartbeat_path: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional["CryptoExitOverlay"]:
    """Build the production crypto overlay from ``config/crypto_exit_overlay.json``.

    FAIL-OPEN: any config problem logs ``CRYPTO_EXIT_OVERLAY_ERROR`` and returns ``None`` (not
    wired → the trading loop is byte-identical). Mirrors ``build_default_overlay``'s pytest
    leak guard so a test can never compose the real evidence/state paths.
    """
    log = logger or LOGGER
    cfg_path = config_path or (repo_root / "config" / "crypto_exit_overlay.json")
    try:
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        config = load_overlay_config(payload)
    except (ExitOverlayConfigError, OSError, ValueError) as exc:
        log.error(
            "%s config_load_failed path=%s detail=%s: %s (fail-open: overlay NOT wired)",
            MARKER_ERROR, cfg_path, type(exc).__name__, exc,
        )
        return None

    if evidence_dir is not None:
        ev: Optional[Path] = evidence_dir
    elif _under_active_pytest():
        raise RuntimeError(
            f"{MARKER_ERROR} build_default_crypto_overlay: evidence_dir is REQUIRED-explicit "
            f"under pytest — pass evidence_dir=tmp_path (never the real "
            f"{default_evidence_dir(repo_root)}). [crypto-exit-overlay test-write leak guard]"
        )
    else:
        ev = default_evidence_dir(repo_root)

    if state_path is not None:
        st: Optional[Path] = state_path
    elif _under_active_pytest():
        raise RuntimeError(
            f"{MARKER_ERROR} build_default_crypto_overlay: state_path is REQUIRED-explicit "
            f"under pytest — pass state_path=tmp_path/... (never the real "
            f"{_default_state_path(repo_root)})."
        )
    else:
        st = _default_state_path(repo_root)

    if heartbeat_path is not None:
        hb: Optional[Path] = heartbeat_path
    elif _under_active_pytest():
        raise RuntimeError(
            f"{MARKER_ERROR} build_default_crypto_overlay: heartbeat_path is REQUIRED-explicit "
            f"under pytest — pass heartbeat_path=tmp_path/... (never the real "
            f"{default_heartbeat_path(repo_root)})."
        )
    else:
        hb = default_heartbeat_path(repo_root)

    from chad.core.kraken_trusted_fill_engine import RoundTripBook, TrustedFillEngine

    book = RoundTripBook()
    max_age = float(payload.get("max_tick_age_seconds", _DEFAULT_MAX_TICK_AGE_SECONDS))
    return CryptoExitOverlay(
        config,
        evidence_path=ev,
        state_path=st,
        heartbeat_path=hb,
        lots_loader=_default_lots_loader(book),
        marks_loader=_default_marks_loader(repo_root / "runtime" / "kraken_prices.json", max_age),
        bars_loader=_default_bars_loader(repo_root),
        open_qty_loader=book.open_qty,
        engine_factory=lambda: TrustedFillEngine(book=book),
        env=env,
        logger=log,
    )
