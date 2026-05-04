#!/usr/bin/env python3
"""
CHAD — Trade Stats Engine (Production, Phase 10 Parity)

This module computes deterministic, audit-friendly performance statistics from CHAD’s
append-only trade ledgers under data/trades/.

It is designed to be:
- Deterministic and replayable (same inputs -> same outputs)
- Fail-safe (never raises for malformed rows; skips safely)
- Test-harness friendly (supports pytest monkeypatch.chdir(tmp_path))
- Production fast (bounded scans, streaming parse, optional enrichment dedupe)

Key invariants
--------------
1) "total_trades" counts ONLY trades that pass finite checks (pnl + notional finite).
   Rows with non-finite pnl/notional are excluded entirely and tracked as excluded_nonfinite.
2) "effective_trades" counts the subset of total_trades that are eligible for performance scoring:
   - NOT manual
   - NOT pnl_untrusted
   - NOT unfilled IBKR paper placeholders
3) Kraken enrichment ledger support:
   - data/trades/trade_history_enriched.ndjson may contain enriched records
   - If raw + enriched exist for same txid, enriched wins and raw is excluded from effective stats

Repo root resolution
--------------------
Resolved at CALL-TIME (not import time), so tests that monkeypatch.chdir(...) work.

Priority:
  1) CHAD_REPO_ROOT env (operator override)
  2) current working directory if it contains data/trades (test harness support)
  3) file-based fallback (source tree)

Public API
----------
- load_and_compute(max_trades=5000, days_back=60, include_paper=True, include_live=True) -> dict

Output keys (stable)
--------------------
total_trades, effective_trades,
excluded_manual, excluded_untrusted, excluded_nonfinite,
live_trades, paper_trades,
win_rate, total_pnl, sharpe_like, max_drawdown
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------
# Repo root resolution (call-time)
# -----------------------------


def _resolve_repo_root() -> Path:
    env = os.environ.get("CHAD_REPO_ROOT", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            return p.resolve()

    cwd = Path.cwd()
    if (cwd / "data" / "trades").is_dir():
        return cwd.resolve()

    # Fallback to source tree root
    return Path(__file__).resolve().parents[2]


def _data_trades_dir() -> Path:
    return _resolve_repo_root() / "data" / "trades"


def _enrich_ledger_path() -> Path:
    return _data_trades_dir() / "trade_history_enriched.ndjson"


# -----------------------------
# Time + numeric helpers
# -----------------------------


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _is_manual(tags: Sequence[str]) -> bool:
    return any(str(t).strip().lower() == "manual" for t in (tags or []))


def _is_untrusted(tags: Sequence[str], extra: Dict[str, Any]) -> bool:
    if any(str(t).strip().lower() == "pnl_untrusted" for t in (tags or [])):
        return True
    try:
        if bool((extra or {}).get("pnl_untrusted")):
            return True
    except Exception:
        return True
    return False




def _is_warmup_sim(tags: Sequence[str]) -> bool:
    # Synthetic warmup trades must never influence SCR confidence gating.
    return any(str(t).strip().lower() == "warmup_sim" for t in (tags or []))


# Strategy names recognised by the tag-based attribution recovery below.
# Priority order: more specific names first so e.g. "delta_pairs" wins over
# the prefix "delta" if both happen to appear in the same tag list.
_KNOWN_STRATEGY_NAMES_PRIORITY: Tuple[str, ...] = (
    "alpha_futures", "gamma_futures", "alpha_options",
    "alpha_crypto", "alpha_forex", "gamma_reversion",
    "omega_macro", "omega_vol", "delta_pairs",
    "alpha", "beta", "gamma", "omega", "delta",
)


def _recover_strategy_from_tags(strategy: str, tags: Sequence[str]) -> str:
    """
    Tag-based strategy attribution recovery for legacy "paper_exec" rows.

    Historical fills written before the P0-4 evidence-writer hardening were
    bucketed under the literal label "paper_exec" in the ``strategy`` field
    even though their tag list still carried the real strategy name (e.g.
    ``["paper", "filled", "paper_exec", "etf", "delta"]``). This helper is
    a *read-side* recovery: it inspects the tag list and substitutes the
    first matching known strategy name. The on-disk evidence file is never
    mutated, preserving the hash chain.

    Behaviour:
      - If ``strategy`` is anything other than literal "paper_exec",
        the original value is returned unchanged.
      - If ``strategy`` is "paper_exec" and a known strategy name appears
        in ``tags`` (case-insensitive), the first match in priority order
        is returned.
      - If no match is found, the original "paper_exec" label is preserved
        so downstream filters can still treat the row as untrusted.
    """
    if str(strategy or "").strip().lower() != "paper_exec":
        return strategy
    tag_set = {str(t).strip().lower() for t in (tags or []) if t}
    for cand in _KNOWN_STRATEGY_NAMES_PRIORITY:
        if cand in tag_set:
            return cand
    return strategy
def _kraken_txid(payload: Dict[str, Any]) -> Optional[str]:
    # best-effort extraction; different pipelines may store txid under different keys
    for k in ("txid", "kraken_txid", "order_txid", "trade_txid"):
        v = payload.get(k)
        if v:
            s = str(v).strip()
            if s:
                return s
    extra = payload.get("extra") or {}
    if isinstance(extra, dict):
        for k in ("txid", "kraken_txid", "order_txid", "trade_txid"):
            v = extra.get(k)
            if v:
                s = str(v).strip()
                if s:
                    return s
    return None


def _compute_max_drawdown(equity_curve: Sequence[float]) -> float:
    """
    equity_curve is cumulative pnl series (or equity). Returns max drawdown in same units.
    Drawdown is negative or zero.
    """
    if not equity_curve:
        return 0.0
    peak = float(equity_curve[0])
    max_dd = 0.0
    for x in equity_curve:
        v = float(x)
        if v > peak:
            peak = v
        dd = v - peak
        if dd < max_dd:
            max_dd = dd
    return float(max_dd)


def _compute_sharpe_like(pnl_series: Sequence[float]) -> float:
    """
    Sharpe-like metric: mean(pnl) / std(pnl) * sqrt(N)
    Deterministic and bounded.
    """
    n = len(pnl_series)
    if n <= 1:
        return 0.0
    mean = sum(float(x) for x in pnl_series) / float(n)
    var = sum((float(x) - mean) ** 2 for x in pnl_series) / float(n - 1)
    std = math.sqrt(var) if var > 0.0 else 0.0
    if std <= 0.0 or not math.isfinite(std):
        return 0.0
    return float((mean / std) * math.sqrt(float(n)))


# -----------------------------
# Trade row model
# -----------------------------


@dataclass(frozen=True)
class ParsedTrade:
    broker: str
    is_live: bool
    strategy: str
    symbol: str
    pnl: float
    notional: float
    tags: List[str]
    extra: Dict[str, Any]
    source: str  # "raw" or "enriched"
    txid: Optional[str] = None


def _is_unfilled_ibkr_paper(t: ParsedTrade) -> bool:
    """
    IBKR paper lane may include intent records. We only want filled ones as performance samples.
    Filled proof is detected via:
      - tag "filled", OR
      - payload fill_price > 0, OR
      - extra.fill_price_used / extra.filled_quantity_used > 0
    """
    try:
        if str(t.broker).strip().lower() != "ibkr":
            return False
        if bool(t.is_live):
            return False
        tags = [str(x).strip().lower() for x in (t.tags or [])]
        if "filled" in tags:
            return False

        # We don’t have fill_price on ParsedTrade directly; attempt from extra if present
        extra = t.extra or {}
        if not isinstance(extra, dict):
            extra = {}

        fp2 = _safe_float(extra.get("fill_price_used"), default=0.0)
        fq2 = _safe_float(extra.get("filled_quantity_used"), default=0.0)
        if fp2 > 0.0 or fq2 > 0.0:
            return False

        # If neither proof exists, treat as unfilled placeholder.
        return True
    except Exception:
        # Fail-safe: do not block scoring if something is malformed; treat as unfilled.
        return True


# -----------------------------
# Ledger iteration + parsing
# -----------------------------


def _iter_day_trade_files(days_back: int) -> List[Path]:
    """
    Return newest-first list of trade_history_YYYYMMDD.ndjson files within days_back window.
    If none found in that window, fall back to any available trade_history_*.ndjson files.
    """
    out: List[Path] = []
    base = _data_trades_dir()
    now0 = _utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
    for i in range(max(0, int(days_back)) + 1):
        ymd = (now0 - timedelta(days=i)).strftime("%Y%m%d")
        p = base / f"trade_history_{ymd}.ndjson"
        if p.is_file():
            out.append(p)

    if not out:
        out = sorted(base.glob("trade_history_*.ndjson"), reverse=True)

    # Newest-first (important for max_trades cap semantics)
    return sorted(out, reverse=True)


def _iter_ndjson_lines(path: Path) -> Iterable[str]:
    """
    Streaming NDJSON line iterator. Never raises; yields only non-empty lines.
    """
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                s = ln.strip()
                if s:
                    yield s
    except Exception:
        return


def _parse_record(line: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(line)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _parse_ledger(
    *,
    path: Path,
    source: str,
    include_paper: bool,
    include_live: bool,
    max_trades: int,
    counters: Dict[str, int],
    invalid_fill_ids: Optional[set] = None,
    invalid_trade_hashes: Optional[set] = None,
) -> List[ParsedTrade]:
    """
    Parse a ledger file into ParsedTrade rows with strict numeric hygiene.

    counters mutated:
      - excluded_nonfinite increments for skipped records
      - excluded_quarantined increments for records in runtime/
        quarantine_manifest_*.json (matched by record_hash or fill_id).
    """
    out: List[ParsedTrade] = []
    if max_trades <= 0:
        return out

    bad_fills = invalid_fill_ids or set()
    bad_hashes = invalid_trade_hashes or set()

    for ln in _iter_ndjson_lines(path):
        rec = _parse_record(ln)
        if rec is None:
            continue

        # Quarantine awareness: skip records explicitly flagged in
        # runtime/quarantine_manifest_*.json. Matched on top-level
        # record_hash (trades) or payload.fill_id (fills).
        rh = rec.get("record_hash")
        if isinstance(rh, str) and rh in bad_hashes:
            counters["excluded_quarantined"] = counters.get("excluded_quarantined", 0) + 1
            continue
        _payload_for_fill = rec.get("payload")
        if isinstance(_payload_for_fill, dict):
            fid = _payload_for_fill.get("fill_id")
            if isinstance(fid, str) and fid in bad_fills:
                counters["excluded_quarantined"] = counters.get("excluded_quarantined", 0) + 1
                continue

        payload = rec.get("payload") or {}
        if not isinstance(payload, dict):
            continue

        broker = str(payload.get("broker") or "").strip()
        if not broker:
            continue

        is_live = bool(payload.get("is_live", False))
        if is_live and not include_live:
            continue
        if (not is_live) and not include_paper:
            continue

        pnl = _safe_float(payload.get("pnl"), default=float("nan"))
        notional = _safe_float(payload.get("notional"), default=float("nan"))
        if not math.isfinite(notional):
            entry_price = _safe_float(payload.get("entry_price"), default=float("nan"))
            qty = _safe_float(payload.get("quantity"), default=float("nan"))
            mult = _safe_float(payload.get("contract_multiplier"), default=1.0)
            if math.isfinite(entry_price) and math.isfinite(qty):
                notional = entry_price * qty * mult
        if not (math.isfinite(pnl) and math.isfinite(notional)):
            counters["excluded_nonfinite"] += 1
            continue

        tags = list(payload.get("tags") or [])
        extra = dict(payload.get("extra") or {}) if isinstance(payload.get("extra") or {}, dict) else {}

        # Tag-based attribution recovery: legacy "paper_exec" rows carry the
        # real strategy name in their tag list. Recover at read time so SCR
        # sees true attribution without mutating hash-chained evidence files.
        raw_strategy = str(payload.get("strategy") or "")
        recovered_strategy = _recover_strategy_from_tags(raw_strategy, tags)

        pt = ParsedTrade(
            broker=broker,
            is_live=is_live,
            strategy=recovered_strategy,
            symbol=str(payload.get("symbol") or ""),
            pnl=float(pnl),
            notional=float(notional),
            tags=[str(x) for x in tags],
            extra=extra,
            source=source,
            txid=_kraken_txid(payload) if broker.strip().lower() == "kraken" else None,
        )
        out.append(pt)

        if len(out) >= max_trades:
            break

    return out


def _load_all_trades(
    *,
    days_back: int,
    include_paper: bool,
    include_live: bool,
    max_trades: int,
) -> Tuple[List[ParsedTrade], Dict[str, int]]:
    """
    Load raw + optional enriched trades, bounded by max_trades overall.
    Returns (trades, counters) where counters includes excluded_nonfinite
    and excluded_quarantined.
    """
    counters: Dict[str, int] = {"excluded_nonfinite": 0, "excluded_quarantined": 0}

    trades: List[ParsedTrade] = []
    remaining = max(0, int(max_trades))
    if remaining <= 0:
        return trades, counters

    # Quarantine awareness: load IDs once per call so the same exclusion
    # set applies across the raw daily ledgers and the enrichment ledger.
    try:
        from chad.utils.quarantine import get_quarantine_sets
        invalid_fill_ids, invalid_trade_hashes = get_quarantine_sets(
            runtime_dir=_resolve_repo_root() / "runtime",
        )
    except Exception:
        invalid_fill_ids, invalid_trade_hashes = set(), set()

    # Raw daily ledgers (newest-first)
    for f in _iter_day_trade_files(days_back):
        if remaining <= 0:
            break
        rows = _parse_ledger(
            path=f,
            source="raw",
            include_paper=include_paper,
            include_live=include_live,
            max_trades=remaining,
            counters=counters,
            invalid_fill_ids=invalid_fill_ids,
            invalid_trade_hashes=invalid_trade_hashes,
        )
        trades.extend(rows)
        remaining -= len(rows)

    # Enrichment ledger (optional)
    enrich = _enrich_ledger_path()
    if remaining > 0 and enrich.is_file():
        rows = _parse_ledger(
            path=enrich,
            source="enriched",
            include_paper=include_paper,
            include_live=include_live,
            max_trades=remaining,
            counters=counters,
            invalid_fill_ids=invalid_fill_ids,
            invalid_trade_hashes=invalid_trade_hashes,
        )
        trades.extend(rows)

    return trades, counters


def _dedupe_kraken(trades: List[ParsedTrade]) -> List[ParsedTrade]:
    """
    If both raw+enriched exist for same kraken txid, keep enriched only.
    """
    enriched_by_txid: Dict[str, ParsedTrade] = {}
    raw_by_txid: Dict[str, ParsedTrade] = {}
    others: List[ParsedTrade] = []

    for t in trades:
        if t.broker.strip().lower() != "kraken" or not t.txid:
            others.append(t)
            continue
        if t.source == "enriched":
            enriched_by_txid[t.txid] = t
        else:
            raw_by_txid[t.txid] = t

    kept: List[ParsedTrade] = []
    for txid, t_en in enriched_by_txid.items():
        kept.append(t_en)
        raw_by_txid.pop(txid, None)

    kept.extend(raw_by_txid.values())
    kept.extend(others)

    # Preserve stable order: newest-first semantics matter, but we don’t have per-row timestamps here.
    # Keep original list order by filtering:
    kept_set = set(id(x) for x in kept)
    return [t for t in trades if id(t) in kept_set]


# -----------------------------
# Public API
# -----------------------------


def load_and_compute(
    *,
    max_trades: int = 5000,
    days_back: int = 60,
    include_paper: bool = True,
    include_live: bool = True,
) -> Dict[str, Any]:
    """
    Compute performance stats from ledgers with deterministic hygiene.

    max_trades:
      Upper bound on number of finite trades loaded across all sources (raw + enriched).
      Newest-first: we load recent days first, and stop when max_trades reached.

    days_back:
      Day window for raw daily ledgers. If none exist, fall back to any available ledgers.

    include_paper/include_live:
      Lane filters.
    """
    max_trades_i = max(0, int(max_trades))
    days_back_i = max(0, int(days_back))

    # Load + dedupe
    trades, counters = _load_all_trades(
        days_back=days_back_i,
        include_paper=bool(include_paper),
        include_live=bool(include_live),
        max_trades=max_trades_i if max_trades_i > 0 else 0,
    )
    trades = _dedupe_kraken(trades)

    # Aggregate counters (finite-only trades are "total_trades")
    total_trades = len(trades)
    excluded_nonfinite = int(counters.get("excluded_nonfinite") or 0)
    excluded_quarantined = int(counters.get("excluded_quarantined") or 0)

    # Lane counts (finite-only)
    live_trades = sum(1 for t in trades if bool(t.is_live))
    paper_trades = total_trades - live_trades

    excluded_manual = 0
    excluded_untrusted = 0

    # Effective sample selection
    effective: List[ParsedTrade] = []
    for t in trades:
        tags = t.tags or []
        extra = t.extra or {}
        if _is_warmup_sim(tags):
            # Treat as untrusted-equivalent: excluded from effective performance sample
            excluded_untrusted += 1
            continue
        if _is_manual(tags):
            excluded_manual += 1
            continue
        if _is_untrusted(tags, extra):
            excluded_untrusted += 1
            continue
        if _is_unfilled_ibkr_paper(t):
            # Count as untrusted for performance purposes? keep separate? treat as untrusted-equivalent
            excluded_untrusted += 1
            continue
        effective.append(t)

    # Metrics
    # total_pnl is defined over ALL finite trades (total_trades), not just effective.
    # (Tests and ops reports expect this.)
    total_pnl = float(sum(float(t.pnl) for t in trades)) if trades else 0.0

    # Performance metrics used for trust/SCR are based on effective trades only.
    # Exclude PnL=0 trades: these are open positions or unmatched entry fills,
    # not closed trades. Counting them dilutes win_rate (they look like losses).
    excluded_pnl_zero = sum(1 for t in effective if float(t.pnl) == 0.0)
    pnls = [float(t.pnl) for t in effective if float(t.pnl) != 0.0]
    effective_trades = len(pnls)

    # Win rate: pnl > 0 (denominator is closed trades only)
    wins = sum(1 for p in pnls if p > 0.0)
    win_rate = float(wins) / float(effective_trades) if effective_trades > 0 else 0.0

    # Equity curve = cumulative pnl
    equity: List[float] = []
    c = 0.0
    for p in pnls:
        c += float(p)
        equity.append(c)

    max_drawdown = float(_compute_max_drawdown(equity)) if equity else 0.0
    sharpe_like = float(_compute_sharpe_like(pnls)) if pnls else 0.0

    return {
        "total_trades": int(total_trades),
        "effective_trades": int(effective_trades),
        "excluded_manual": int(excluded_manual),
        # Quarantined records are excluded BEFORE the parse step so they
        # never appear in trades/effective; surface the count via the
        # existing untrusted bucket so dashboards/SCR see the full
        # exclusion picture, and also expose the discrete counter for
        # operator forensics.
        "excluded_untrusted": int(excluded_untrusted + excluded_quarantined),
        "excluded_quarantined": int(excluded_quarantined),
        "excluded_nonfinite": int(excluded_nonfinite),
        "excluded_pnl_zero": int(excluded_pnl_zero),
        "live_trades": int(live_trades),
        "paper_trades": int(paper_trades),
        "win_rate": float(win_rate),
        "total_pnl": float(total_pnl),
        "max_drawdown": float(max_drawdown),
        "sharpe_like": float(sharpe_like),
    }


if __name__ == "__main__":
    # Minimal CLI: prints stats JSON for operator inspection (no side effects).
    stats = load_and_compute(max_trades=5000, days_back=60, include_paper=True, include_live=True)
    print(json.dumps(stats, indent=2, sort_keys=True))
