#!/usr/bin/env python3
"""
CHAD Context Builder — Institutional-Grade (Strategy Input Plane)

Deterministic, audit-friendly MarketContext construction for:
- Phase-3 StrategyEngine
- Phase-7 DRY_RUN full_execution_cycle (expects ContextBuilder.build() result object)

Truth sources (local-only, no network):
- runtime/positions_snapshot.json  (positions)
- runtime/price_cache.json         (prices/ticks)
- legend file (auto-detected)

Cash policy (explicit, fail-closed)
-----------------------------------
We do NOT guess broker cash.

- Default cash = 0.0 (fail-closed)
- Optional operator override via environment variables:
    CHAD_CASH_OVERRIDE            (float)
    CHAD_CASH_OVERRIDE_REASON     (string, required if override provided)

If CHAD_CASH_OVERRIDE is set but invalid, cash remains 0.0 and evidence records error.

This allows strategies like Beta (min_cash=10k) to run ONLY when you explicitly
authorize a cash value for the strategy layer.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from chad.types import (
    AssetClass,
    LegendConsensus,
    MarketContext,
    MarketTick,
    PortfolioSnapshot,
    Position,
)


@dataclass(frozen=True)
class ContextBuildEvidence:
    now_utc: str
    runtime_dir: str

    positions_path: str
    positions_mtime_utc: Optional[str]
    positions_count: int
    positions_error: Optional[str]

    legend_path: str
    legend_mtime_utc: Optional[str]
    legend_symbols: int
    legend_error: Optional[str]

    ticks_source: str
    ticks_count: int
    ticks_error: Optional[str]

    cash_source: str
    cash_value: float
    cash_error: Optional[str]


@dataclass(frozen=True)
class ContextBuildResult:
    context: MarketContext
    prices: Mapping[str, float]
    current_symbol_notional: Mapping[str, float]
    current_total_notional: float


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runtime_dir() -> Path:
    root = _repo_root()
    return Path(os.environ.get("CHAD_RUNTIME_DIR", str(root / "runtime"))).resolve()


def _stat_mtime_utc(p: Path) -> Optional[str]:
    try:
        st = p.stat()
        return _iso(datetime.fromtimestamp(st.st_mtime, tz=timezone.utc))
    except Exception:
        return None


def _load_positions(runtime_dir: Path) -> Tuple[Mapping[str, Position], int, Optional[str], Optional[str]]:
    p = runtime_dir / "positions_snapshot.json"
    mtime = _stat_mtime_utc(p)

    if not p.is_file():
        return {}, 0, mtime, "missing_positions_snapshot"

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}, 0, mtime, "positions_snapshot_not_dict"

        pos_map = raw.get("positions_by_conid")
        if not isinstance(pos_map, dict):
            return {}, 0, mtime, "positions_by_conid_missing_or_invalid"

        out: Dict[str, Position] = {}
        for _conid, rec in pos_map.items():
            if not isinstance(rec, dict):
                continue

            sym = str(rec.get("symbol") or "").strip().upper()
            if not sym:
                continue

            qty = float(rec.get("qty") or 0.0)
            avg = float(rec.get("avg_cost") or 0.0)
            sec_type = str(rec.get("secType") or "STK").strip().upper()
            asset_class = AssetClass.ETF if sec_type == "ETF" else AssetClass.EQUITY

            out[sym] = Position(symbol=sym, asset_class=asset_class, quantity=qty, avg_price=avg)

        return out, len(out), mtime, None

    except Exception as exc:
        return {}, 0, mtime, f"{type(exc).__name__}: {exc}"


def _load_legend(runtime_dir: Path, now: datetime) -> Tuple[Optional[LegendConsensus], str, Optional[str], int, Optional[str]]:
    root = _repo_root()
    candidates = [
        runtime_dir / "legend_top_stocks.json",
        root / "data" / "legend_top_stocks.json",
        root / "data" / "legend" / "legend_top_stocks.json",
    ]

    chosen: Optional[Path] = None
    for c in candidates:
        if c.is_file():
            chosen = c
            break

    if chosen is None:
        return None, "missing", None, 0, "legend_file_missing"

    mtime = _stat_mtime_utc(chosen)

    try:
        raw = json.loads(chosen.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None, str(chosen), mtime, 0, "legend_not_dict"

        weights = raw.get("weights")
        if not isinstance(weights, dict):
            return None, str(chosen), mtime, 0, "legend_weights_missing"

        norm: Dict[str, float] = {}
        for k, v in weights.items():
            sym = str(k).strip().upper()
            if not sym:
                continue
            try:
                w = float(v)
            except Exception:
                continue
            if w > 0.0:
                norm[sym] = w

        if not norm:
            return None, str(chosen), mtime, 0, "legend_weights_empty"

        as_of = now
        as_of_raw = raw.get("as_of")
        if isinstance(as_of_raw, str) and as_of_raw.strip():
            try:
                as_of = datetime.fromisoformat(as_of_raw.replace("Z", "+00:00"))
                if as_of.tzinfo is None:
                    as_of = as_of.replace(tzinfo=timezone.utc)
            except Exception:
                as_of = now

        return LegendConsensus(as_of=as_of, weights=norm), str(chosen), mtime, len(norm), None

    except Exception as exc:
        return None, str(chosen), mtime, 0, f"{type(exc).__name__}: {exc}"


def _load_ticks(runtime_dir: Path, now: datetime) -> Tuple[Mapping[str, MarketTick], str, int, Optional[str]]:
    cache = runtime_dir / "price_cache.json"
    if not cache.is_file():
        return {}, "runtime/price_cache.json (missing)", 0, "missing_price_cache"

    try:
        raw = json.loads(cache.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}, str(cache), 0, "price_cache_not_dict"

        # supports {"prices": {...}} OR flat dict OR nested dict
        obj = raw.get("prices") if isinstance(raw.get("prices"), dict) else raw
        if not isinstance(obj, dict):
            return {}, str(cache), 0, "price_cache_prices_not_dict"

        out: Dict[str, MarketTick] = {}
        for sym, rec in obj.items():
            s = str(sym).strip().upper()
            if not s:
                continue

            if isinstance(rec, dict):
                price = float(rec.get("price") or 0.0)
            else:
                price = float(rec or 0.0)

            if price <= 0.0:
                continue

            out[s] = MarketTick(
                symbol=s,
                price=float(price),
                size=0.0,
                exchange=None,
                timestamp=now,
                source="price_cache",
            )

        return out, str(cache), len(out), None

    except Exception as exc:
        return {}, str(cache), 0, f"{type(exc).__name__}: {exc}"


def _resolve_cash_override() -> Tuple[float, str, Optional[str]]:
    """
    Returns (cash_value, cash_source, cash_error).
    """
    raw = os.environ.get("CHAD_CASH_OVERRIDE")
    if raw is None or not str(raw).strip():
        return 0.0, "fail_closed_default_0", None

    reason = str(os.environ.get("CHAD_CASH_OVERRIDE_REASON") or "").strip()
    if not reason:
        return 0.0, "fail_closed_default_0", "override_missing_reason"

    try:
        val = float(raw)
        if not (val >= 0.0):
            return 0.0, "fail_closed_default_0", "override_negative"
        return float(val), f"override:{reason}", None
    except Exception as exc:
        return 0.0, "fail_closed_default_0", f"override_invalid:{type(exc).__name__}"


def build_market_context(now: Optional[datetime] = None) -> Tuple[MarketContext, ContextBuildEvidence]:
    now_dt = now or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)

    runtime = _runtime_dir()

    positions, pos_count, pos_mtime, pos_err = _load_positions(runtime)
    legend, legend_path, legend_mtime, legend_count, legend_err = _load_legend(runtime, now_dt)
    ticks, ticks_src, ticks_count, ticks_err = _load_ticks(runtime, now_dt)

    cash_value, cash_source, cash_error = _resolve_cash_override()

    portfolio = PortfolioSnapshot(timestamp=now_dt, cash=float(cash_value), positions=positions)
    ctx = MarketContext(now=now_dt, ticks=ticks, legend=legend, portfolio=portfolio)

    ev = ContextBuildEvidence(
        now_utc=_iso(now_dt),
        runtime_dir=str(runtime),
        positions_path=str(runtime / "positions_snapshot.json"),
        positions_mtime_utc=pos_mtime,
        positions_count=pos_count,
        positions_error=pos_err,
        legend_path=legend_path,
        legend_mtime_utc=legend_mtime,
        legend_symbols=legend_count,
        legend_error=legend_err,
        ticks_source=ticks_src,
        ticks_count=ticks_count,
        ticks_error=ticks_err,
        cash_source=cash_source,
        cash_value=float(cash_value),
        cash_error=cash_error,
    )

    return ctx, ev


class ContextBuilder:
    def build(self, now: Optional[datetime] = None) -> ContextBuildResult:
        ctx, _ev = build_market_context(now=now)

        prices: Dict[str, float] = {}
        for sym, tick in ctx.ticks.items():
            try:
                px = float(tick.price)
            except Exception:
                continue
            if px > 0.0:
                prices[str(sym).strip().upper()] = px

        current_symbol_notional: Dict[str, float] = {s: 0.0 for s in prices.keys()}
        current_total_notional = 0.0

        return ContextBuildResult(
            context=ctx,
            prices=prices,
            current_symbol_notional=current_symbol_notional,
            current_total_notional=float(current_total_notional),
        )


if __name__ == "__main__":
    ctx, ev = build_market_context()
    print("now:", _iso(ctx.now))
    print("ticks:", len(ctx.ticks), "symbols:", sorted(ctx.ticks.keys()))
    print("legend:", 0 if ctx.legend is None else len(ctx.legend.weights))
    print("positions:", len(ctx.portfolio.positions), "symbols:", sorted(ctx.portfolio.positions.keys()))
    print("cash:", ctx.portfolio.cash, "cash_source:", ev.cash_source, "cash_error:", ev.cash_error)
    print("evidence:", json.dumps(dataclasses.asdict(ev), indent=2, sort_keys=True))
