"""
tier_instrument_gate.py — Tier-aware instrument allowlist gate.

Pipeline-level enforcement that blocks raw TradeSignals whose symbol is
not in the active tier's `allowed_instruments` list as published by
TierManager in runtime/tier_state.json.

Closes Gap-1 of the v9.1 post-implementation audit: alpha_crypto (and
any future strategy with its own internal universe) emits signals for
BTC-USD/ETH-USD/SOL-USD irrespective of tier. Without this gate a
STARTER account (allowed_instruments=["MES","MNQ","SPY","QQQ","IWM",
"BTC-USD"]) could receive ETH-USD or SOL-USD signals.

Contract:
- Reads runtime/tier_state.json once per filter_signals() call. No
  cross-cycle caching.
- Fail-open on any read/parse error: all signals pass, fail_open=True.
- Null/empty/wildcard allowed_instruments lists fail-open (pass all).
- Only blocks when allowed_instruments is a non-empty explicit list
  AND the signal symbol is not in that list.
- Writes runtime/tier_instrument_gate_state.json atomically after every
  call (even on fail-open or empty input).

No broker / adapter / live-execution / strategy imports.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from chad.market_data.price_cache_refresh import normalize_symbol
from chad.types import TradeSignal

LOG = logging.getLogger("chad.execution.tier_instrument_gate")

_STATE_FILENAME = "tier_instrument_gate_state.json"
_TIER_STATE_FILENAME = "tier_state.json"
_SCHEMA_VERSION = "tier_instrument_gate.v1"
_TTL_SECONDS = 300
_BLOCK_REASON = "INSTRUMENT_NOT_IN_TIER_ALLOWLIST"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _signal_strategy_label(sig: TradeSignal) -> str:
    """Return a stable string label for a TradeSignal's strategy field
    regardless of whether it carries a StrategyName enum or raw string."""
    strat = getattr(sig, "strategy", None)
    if strat is None:
        return "?"
    val = getattr(strat, "value", None)
    if isinstance(val, str) and val:
        return val
    return str(strat)


def _signal_symbol_label(sig: TradeSignal) -> str:
    """Return the raw symbol string from the signal for logging."""
    return str(getattr(sig, "symbol", "") or "")


class TierInstrumentGate:
    """Block TradeSignals whose symbol is not in the active tier's
    allowed_instruments list.

    The gate fails open on missing / unreadable / malformed tier state so
    that publisher outages cannot stop the trading pipeline. It writes a
    per-cycle audit artifact to runtime/tier_instrument_gate_state.json.
    """

    def __init__(self, runtime_dir: Path) -> None:
        self._runtime_dir: Path = Path(runtime_dir)
        self._tier_state_path: Path = self._runtime_dir / _TIER_STATE_FILENAME
        self._state_out_path: Path = self._runtime_dir / _STATE_FILENAME

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter_signals(
        self,
        signals: List[TradeSignal],
    ) -> Tuple[List[TradeSignal], List[TradeSignal]]:
        """Return (allowed_signals, blocked_signals).

        Fail-open semantics: any failure to read or parse tier_state.json
        causes all input signals to be returned as allowed with a WARNING
        logged. State file is always written.
        """
        sigs: List[TradeSignal] = list(signals or [])

        tier_name: Optional[str] = None
        allowed_instruments: Optional[Sequence[Any]] = None
        fail_open: bool = False
        fail_open_reason: Optional[str] = None

        try:
            raw = self._tier_state_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            fail_open = True
            fail_open_reason = "tier_state_missing"
            LOG.warning(
                "tier_instrument_gate fail_open reason=%s path=%s — allowing all %d signals",
                fail_open_reason,
                self._tier_state_path,
                len(sigs),
            )
        except OSError as exc:
            fail_open = True
            fail_open_reason = f"tier_state_unreadable:{exc}"
            LOG.warning(
                "tier_instrument_gate fail_open reason=%s — allowing all %d signals",
                fail_open_reason,
                len(sigs),
            )
        else:
            try:
                state = json.loads(raw)
            except (ValueError, TypeError) as exc:
                fail_open = True
                fail_open_reason = f"tier_state_malformed_json:{exc}"
                LOG.warning(
                    "tier_instrument_gate fail_open reason=%s — allowing all %d signals",
                    fail_open_reason,
                    len(sigs),
                )
                state = None
            if isinstance(state, dict):
                tier_name = state.get("tier_name")
                allowed_instruments = state.get("allowed_instruments")
            elif not fail_open:
                fail_open = True
                fail_open_reason = "tier_state_not_object"
                LOG.warning(
                    "tier_instrument_gate fail_open reason=%s — allowing all %d signals",
                    fail_open_reason,
                    len(sigs),
                )

        # SCALE always passes unconditionally — explicit operator-visible
        # short-circuit even though SCALE's allowed_instruments=["*"]
        # already triggers the wildcard branch in _is_allowed.
        scale_short_circuit = (
            not fail_open and isinstance(tier_name, str) and tier_name == "SCALE"
        )

        # Normalise the allowlist to a list of strings (or None to mean
        # "no allowlist enforcement").
        allowlist: Optional[List[str]]
        if fail_open or scale_short_circuit:
            allowlist = None
        else:
            allowlist = self._coerce_allowlist(allowed_instruments)

        allowed: List[TradeSignal] = []
        blocked: List[TradeSignal] = []

        if allowlist is None:
            allowed = list(sigs)
        else:
            for sig in sigs:
                symbol = _signal_symbol_label(sig)
                if self._is_allowed(symbol, allowlist):
                    allowed.append(sig)
                else:
                    blocked.append(sig)
                    LOG.warning(
                        "%s strategy=%s symbol=%s tier=%s",
                        _BLOCK_REASON,
                        _signal_strategy_label(sig),
                        symbol,
                        tier_name or "?",
                    )

        self._write_state(
            tier_name=tier_name,
            fail_open=fail_open,
            fail_open_reason=fail_open_reason,
            allowed=allowed,
            blocked=blocked,
            total=len(sigs),
        )

        return allowed, blocked

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_allowlist(raw: Any) -> Optional[List[str]]:
        """Normalise the allowed_instruments field from tier_state.json.

        Returns None to signal "no enforcement" (null / missing / empty /
        contains '*'). Returns a list of normalised symbols otherwise."""
        if raw is None:
            return None
        if not isinstance(raw, list):
            # Defensive: malformed type → no enforcement (fail-open at
            # the field level, not the gate level).
            LOG.warning(
                "tier_instrument_gate allowed_instruments_not_list type=%s — passing all",
                type(raw).__name__,
            )
            return None
        if len(raw) == 0:
            return None
        items: List[str] = []
        for entry in raw:
            if not isinstance(entry, str):
                continue
            s = entry.strip()
            if not s:
                continue
            if s == "*":
                return None
            items.append(normalize_symbol(s))
        if not items:
            return None
        return items

    def _is_allowed(
        self,
        symbol: str,
        allowed_instruments: List[str],
    ) -> bool:
        """Return True if `symbol` is permitted under `allowed_instruments`.

        Rules (also enforced in _coerce_allowlist, repeated here per the
        Gap-1 spec so the method is self-contained for unit tests):
        - '*' in allowed_instruments → True
        - empty list → True
        - symbol in list → True
        - symbol not in list → False
        """
        if not allowed_instruments:
            return True
        if "*" in allowed_instruments:
            return True
        norm = normalize_symbol(symbol)
        return norm in {normalize_symbol(s) for s in allowed_instruments}

    def _write_state(
        self,
        *,
        tier_name: Optional[str],
        fail_open: bool,
        fail_open_reason: Optional[str],
        allowed: Sequence[TradeSignal],
        blocked: Sequence[TradeSignal],
        total: int,
    ) -> None:
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "ts_utc": _utc_now_iso(),
            "ttl_seconds": _TTL_SECONDS,
            "tier": tier_name,
            "fail_open": bool(fail_open),
            "fail_open_reason": fail_open_reason,
            "signals_evaluated": int(total),
            "signals_allowed": int(len(allowed)),
            "signals_blocked": int(len(blocked)),
            "blocked_symbols": sorted({
                _signal_symbol_label(s) for s in blocked if _signal_symbol_label(s)
            }),
            "blocked_strategies": sorted({
                _signal_strategy_label(s) for s in blocked if _signal_strategy_label(s)
            }),
            "reason": _BLOCK_REASON if blocked else None,
        }

        try:
            self._runtime_dir.mkdir(parents=True, exist_ok=True)
            self._atomic_write_json(self._state_out_path, payload)
        except OSError as exc:
            # Must never crash the pipeline — log and continue.
            LOG.warning(
                "tier_instrument_gate state_write_failed path=%s err=%s",
                self._state_out_path,
                exc,
            )

    @staticmethod
    def _atomic_write_json(path: Path, payload: Any) -> None:
        """Write JSON to `path` via a temp file + os.replace for atomicity."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=path.name + ".",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise


__all__ = ["TierInstrumentGate"]
