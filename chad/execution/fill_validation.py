"""Shared fill-validation predicate (HISTORICAL-PLACEHOLDER-1).

Centralised detection of *trusted-fake* placeholder fills — historical rows
in ``data/fills/FILLS_*.ndjson`` that pre-date the PR-02 / PR-02b /
paper_exec_evidence_writer defense-in-depth and would otherwise be silently
consumed by any reader that does not honour the SCR exclusion contract.

Public API
----------
- ``is_trusted_fake_placeholder(row, *, signals=None) -> bool``
- ``classify_placeholder(row) -> ClassifierResult`` (richer, audit-friendly)
- ``DEFAULT_SIGNALS`` — tuple of detector names in priority order

Defense-in-depth: detection must NOT rely on a single fragile field.
Each detector function takes the payload (dict) and returns a non-empty
string identifier if it fires, or None if it does not.

This module never mutates the input row. It is import-safe (no I/O).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

# Known canonical fingerprints of the historical 14-row incident.
_HISTORICAL_PATTERN_FILL_PRICE = 100.0
_HISTORICAL_PATTERN_STRATEGY = "delta"
_HISTORICAL_PATTERN_SYMBOL = "SPY"


def _payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """If the row is a wrapped ledger record ({payload: {...}, ...}), return
    the inner payload; otherwise return the row itself. Pure function.
    """
    if isinstance(row, Mapping):
        p = row.get("payload")
        if isinstance(p, Mapping):
            return p
    return row if isinstance(row, Mapping) else {}


def _tags_set(payload: Mapping[str, Any]) -> set[str]:
    tags = payload.get("tags") or []
    if isinstance(tags, (list, tuple)):
        return {str(t).strip().lower() for t in tags if t is not None}
    return set()


# --- detectors --------------------------------------------------------------

def detect_explicit_trusted_fake(payload: Mapping[str, Any]) -> str | None:
    """True if any explicit trusted-fake marker is present."""
    if payload.get("trusted_fake") is True:
        return "explicit_trusted_fake_marker"
    extra = payload.get("extra") or {}
    if isinstance(extra, Mapping):
        if extra.get("trusted_fake") is True:
            return "extra.trusted_fake_marker"
        if extra.get("placeholder") is True:
            return "extra.placeholder_marker"
    return None


def detect_placeholder_marker(payload: Mapping[str, Any]) -> str | None:
    """Tag or extra-field marker."""
    tags = _tags_set(payload)
    if "placeholder" in tags or "broker_rejected" in tags or "pnl_untrusted" in tags:
        return "tag_marker"
    extra = payload.get("extra") or {}
    if isinstance(extra, Mapping):
        if "placeholder_fill_price" in extra:
            return "extra.placeholder_fill_price"
    return None


def detect_delta_spy_100_pattern(payload: Mapping[str, Any]) -> str | None:
    """The historical 14-row signature: delta SELL SPY @ $100 with no
    rejection flag. This is the canonical fingerprint of the 2026-05-03
    P0-1 incident.
    """
    try:
        fp = float(payload.get("fill_price"))
    except (TypeError, ValueError):
        return None
    if fp != _HISTORICAL_PATTERN_FILL_PRICE:
        return None
    sym = str(payload.get("symbol") or "").strip().upper()
    strat = str(payload.get("strategy") or "").strip().lower()
    if sym != _HISTORICAL_PATTERN_SYMBOL or strat != _HISTORICAL_PATTERN_STRATEGY:
        return None
    if bool(payload.get("is_live", False)):
        return None
    return "delta_spy_100_pattern"


def detect_pnl_untrusted_flag(payload: Mapping[str, Any]) -> str | None:
    if bool(payload.get("pnl_untrusted")):
        return "pnl_untrusted_flag"
    extra = payload.get("extra") or {}
    if isinstance(extra, Mapping) and bool(extra.get("pnl_untrusted")):
        return "extra.pnl_untrusted_flag"
    return None


def detect_broker_reject(payload: Mapping[str, Any]) -> str | None:
    if bool(payload.get("reject")):
        return "reject_flag"
    status = str(payload.get("status") or "").lower()
    if status in {"rejected", "broker_rejected"}:
        return "status_rejected"
    return None


def detect_synthetic_expected_price(payload: Mapping[str, Any]) -> str | None:
    """Heuristic: if extra.expected_price == 100.0 AND fill_price == 100.0
    AND the row carries no real broker fill receipt fields, this is a
    synthetic placeholder. Conservative — requires both signals.
    """
    extra = payload.get("extra") or {}
    if not isinstance(extra, Mapping):
        return None
    try:
        exp = float(extra.get("expected_price"))
        fp = float(payload.get("fill_price"))
    except (TypeError, ValueError):
        return None
    if exp == 100.0 and fp == 100.0 and str(payload.get("order_type") or "") == "SIM":
        return "synthetic_expected_price"
    return None


# Default chain: order matters only for the "first signal" return value.
Detector = Callable[[Mapping[str, Any]], "str | None"]

DEFAULT_SIGNALS: tuple[str, ...] = (
    "explicit_trusted_fake",
    "placeholder_marker",
    "pnl_untrusted_flag",
    "broker_reject",
    "delta_spy_100_pattern",
    "synthetic_expected_price",
)

_DETECTOR_REGISTRY: dict[str, Detector] = {
    "explicit_trusted_fake": detect_explicit_trusted_fake,
    "placeholder_marker": detect_placeholder_marker,
    "pnl_untrusted_flag": detect_pnl_untrusted_flag,
    "broker_reject": detect_broker_reject,
    "delta_spy_100_pattern": detect_delta_spy_100_pattern,
    "synthetic_expected_price": detect_synthetic_expected_price,
}


@dataclass
class ClassifierResult:
    is_placeholder: bool
    signals_fired: list[str]
    payload_summary: dict[str, Any]


def classify_placeholder(
    row: Mapping[str, Any],
    *,
    signals: Iterable[str] = DEFAULT_SIGNALS,
) -> ClassifierResult:
    """Classify a single row. Runs every requested detector and returns the
    full set that fired. Use this for audit reports.
    """
    payload = _payload(row)
    fired: list[str] = []
    for name in signals:
        det = _DETECTOR_REGISTRY.get(name)
        if det is None:
            continue
        result = det(payload)
        if result:
            fired.append(result)
    return ClassifierResult(
        is_placeholder=bool(fired),
        signals_fired=fired,
        payload_summary={
            "fill_price": payload.get("fill_price"),
            "symbol": payload.get("symbol"),
            "strategy": payload.get("strategy"),
            "side": payload.get("side"),
            "is_live": payload.get("is_live"),
            "status": payload.get("status"),
            "fill_id": payload.get("fill_id"),
        },
    )


def is_trusted_fake_placeholder(
    row: Mapping[str, Any],
    *,
    signals: Iterable[str] = DEFAULT_SIGNALS,
) -> bool:
    """Fast-path boolean predicate for readers.

    A trusted-fake placeholder is any row that any of the configured signals
    flags as suspect. Callers should exclude such rows from any "trusted"
    code path (SCR effective_trades, PnL aggregation, replay, backtest,
    report, dashboard).
    """
    payload = _payload(row)
    for name in signals:
        det = _DETECTOR_REGISTRY.get(name)
        if det is None:
            continue
        if det(payload):
            return True
    return False
