"""
chad/risk/portfolio_var.py

GAP-015A: report-only portfolio Value-at-Risk.

This module computes a parametric (variance-covariance) 1-day Value-at-Risk
estimate for the current paper book using ONLY local files:

  * positions  -> runtime/positions_truth.json (or positions_snapshot.json fallback)
  * prices     -> runtime/price_cache.json
  * daily bars -> data/bars/1d/<SYMBOL>.json (per-symbol)

It performs no broker calls, no network calls, no signal suppression and
no writes by itself. It is consumed by ``ops/var_publisher.py`` which wraps
the report into ``runtime/var_state.json`` (schema ``var_state.v1``).

Method
------
For each held symbol with at least ``MIN_BARS_FOR_VAR`` daily closes:

    log_return_t = ln(close_t / close_{t-1})
    sigma        = stdev(log_returns)              # population
    exposure_usd = abs(qty) * px * multiplier(sec_type)

Independent-asset parametric VaR (square-root-of-sum-of-squares):

    portfolio_sigma_usd = sqrt( sum( (exposure_i * sigma_i)^2 ) )
    var_95_1day_usd     = z(0.95) * portfolio_sigma_usd
    var_99_1day_usd     = z(0.99) * portfolio_sigma_usd

Symbols with insufficient bars or unknown prices are recorded under
``symbols_missing_data`` and excluded from the aggregate. If no symbols are
usable the result is reported as ``status="insufficient_data"`` with zero VaR
values (the publisher then writes a valid but empty schema).

This module never enforces trading behavior. Callers MUST treat the report as
diagnostic only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = REPO_ROOT / "runtime"
DATA_BARS_1D_DIR = REPO_ROOT / "data" / "bars" / "1d"

POSITIONS_TRUTH_PATH = RUNTIME_DIR / "positions_truth.json"
POSITIONS_SNAPSHOT_PATH = RUNTIME_DIR / "positions_snapshot.json"
PRICE_CACHE_PATH = RUNTIME_DIR / "price_cache.json"
PORTFOLIO_SNAPSHOT_PATH = RUNTIME_DIR / "portfolio_snapshot.json"
PNL_STATE_PATH = RUNTIME_DIR / "pnl_state.json"

MIN_BARS_FOR_VAR = 30
MAX_BARS_LOOKBACK = 252
Z_95 = 1.6448536269514722
Z_99 = 2.3263478740408408

# Conservative futures multipliers ($/index point or $/unit). Symbols not in
# the table are still scored using multiplier=1.0 if their secType is STK/None,
# otherwise they are flagged as missing_data.
FUTURES_MULTIPLIERS: Dict[str, float] = {
    "ES": 50.0,
    "MES": 5.0,
    "NQ": 20.0,
    "MNQ": 2.0,
    "YM": 5.0,
    "MYM": 0.5,
    "RTY": 50.0,
    "M2K": 5.0,
    "GC": 100.0,
    "MGC": 10.0,
    "SI": 5000.0,
    "SIL": 1000.0,
    "CL": 1000.0,
    "MCL": 100.0,
    "ZB": 1000.0,
    "ZN": 1000.0,
    "M6E": 12500.0,
}


@dataclass(frozen=True)
class SymbolExposure:
    symbol: str
    quantity: float
    price_usd: float
    multiplier: float
    sec_type: str
    exposure_usd: float
    daily_sigma: float
    bars_used: int


@dataclass(frozen=True)
class VarReport:
    status: str
    method: str
    portfolio_equity_usd: float
    var_95_1day_usd: float
    var_99_1day_usd: float
    symbols_used: List[str]
    symbols_missing_data: List[str]
    notes: List[str]
    symbol_count: int
    var_pct_of_equity: float
    confidence_levels: Tuple[float, ...] = (0.95, 0.99)
    exposures: List[SymbolExposure] = field(default_factory=list)


def _safe_load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_positions(
    positions_truth_path: Optional[Path] = None,
    positions_snapshot_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Return (positions, source_label). Prefers positions_truth.json."""
    truth_path = positions_truth_path or POSITIONS_TRUTH_PATH
    snap_path = positions_snapshot_path or POSITIONS_SNAPSHOT_PATH

    obj = _safe_load_json(truth_path)
    if isinstance(obj, dict):
        positions = obj.get("positions")
        if isinstance(positions, list):
            return list(positions), "positions_truth.json"

    obj = _safe_load_json(snap_path)
    if isinstance(obj, dict):
        positions = obj.get("positions")
        if isinstance(positions, list):
            return list(positions), "positions_snapshot.json"

    return [], "none"


def _load_price_cache(path: Optional[Path] = None) -> Dict[str, float]:
    obj = _safe_load_json(path or PRICE_CACHE_PATH)
    if not isinstance(obj, dict):
        return {}
    prices = obj.get("prices") if isinstance(obj.get("prices"), dict) else obj
    out: Dict[str, float] = {}
    if isinstance(prices, dict):
        for k, v in prices.items():
            try:
                f = float(v)
                if math.isfinite(f) and f > 0:
                    out[str(k).upper()] = f
            except Exception:
                continue
    return out


def _load_portfolio_equity(
    portfolio_snapshot_path: Optional[Path] = None,
    pnl_state_path: Optional[Path] = None,
) -> float:
    snap = _safe_load_json(portfolio_snapshot_path or PORTFOLIO_SNAPSHOT_PATH)
    if isinstance(snap, dict):
        total = 0.0
        for k in ("ibkr_equity", "coinbase_equity", "kraken_equity"):
            try:
                total += float(snap.get(k, 0.0) or 0.0)
            except Exception:
                continue
        if total > 0:
            return total

    pnl = _safe_load_json(pnl_state_path or PNL_STATE_PATH)
    if isinstance(pnl, dict):
        try:
            v = float(pnl.get("account_equity", 0.0) or 0.0)
            if v > 0:
                return v
        except Exception:
            pass
    return 0.0


def _load_daily_closes(
    symbol: str,
    bars_dir: Optional[Path] = None,
    max_bars: int = MAX_BARS_LOOKBACK,
) -> List[float]:
    base = bars_dir or DATA_BARS_1D_DIR
    path = base / f"{symbol}.json"
    obj = _safe_load_json(path)
    if not isinstance(obj, dict):
        return []
    bars = obj.get("bars")
    if not isinstance(bars, list):
        return []
    closes: List[float] = []
    for row in bars:
        if not isinstance(row, dict):
            continue
        try:
            c = float(row.get("close"))
        except Exception:
            continue
        if math.isfinite(c) and c > 0:
            closes.append(c)
    if max_bars > 0 and len(closes) > max_bars:
        closes = closes[-max_bars:]
    return closes


def _log_returns(closes: List[float]) -> List[float]:
    out: List[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        cur = closes[i]
        if prev <= 0 or cur <= 0:
            continue
        try:
            r = math.log(cur / prev)
        except Exception:
            continue
        if math.isfinite(r):
            out.append(r)
    return out


def _stdev_population(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(max(0.0, var))


def _multiplier_for(symbol: str, sec_type: str) -> Optional[float]:
    """Return contract multiplier or None if symbol cannot be valued."""
    s = (symbol or "").upper()
    st = (sec_type or "").upper()
    if st == "FUT":
        return FUTURES_MULTIPLIERS.get(s)
    if st in ("", "STK", "ETF", "STOCK"):
        return 1.0
    if st == "CASH":
        return None
    if st in ("OPT", "FOP"):
        return None
    return FUTURES_MULTIPLIERS.get(s, 1.0)


def compute_portfolio_var(
    *,
    positions_truth_path: Optional[Path] = None,
    positions_snapshot_path: Optional[Path] = None,
    price_cache_path: Optional[Path] = None,
    portfolio_snapshot_path: Optional[Path] = None,
    pnl_state_path: Optional[Path] = None,
    bars_dir: Optional[Path] = None,
    min_bars: int = MIN_BARS_FOR_VAR,
) -> VarReport:
    """Compute report-only parametric portfolio VaR. Never raises on missing data."""
    notes: List[str] = []
    positions, pos_source = _load_positions(
        positions_truth_path=positions_truth_path,
        positions_snapshot_path=positions_snapshot_path,
    )
    notes.append(f"positions_source={pos_source}")

    prices = _load_price_cache(price_cache_path)
    equity_usd = _load_portfolio_equity(
        portfolio_snapshot_path=portfolio_snapshot_path,
        pnl_state_path=pnl_state_path,
    )

    if not positions:
        return VarReport(
            status="insufficient_data",
            method="parametric",
            portfolio_equity_usd=equity_usd,
            var_95_1day_usd=0.0,
            var_99_1day_usd=0.0,
            symbols_used=[],
            symbols_missing_data=[],
            notes=notes + ["no_positions_available"],
            symbol_count=0,
            var_pct_of_equity=0.0,
            exposures=[],
        )

    exposures: List[SymbolExposure] = []
    missing: List[str] = []

    for row in positions:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        try:
            qty = float(row.get("position", 0.0) or 0.0)
        except Exception:
            qty = 0.0
        if qty == 0.0:
            continue
        sec_type = str(row.get("secType") or row.get("sec_type") or "").upper()
        mult = _multiplier_for(symbol, sec_type)
        if mult is None:
            missing.append(f"{symbol}:unsupported_sec_type[{sec_type or 'unknown'}]")
            continue
        px = prices.get(symbol)
        if px is None or px <= 0:
            missing.append(f"{symbol}:missing_price")
            continue
        closes = _load_daily_closes(symbol, bars_dir=bars_dir)
        if len(closes) < min_bars:
            missing.append(f"{symbol}:insufficient_bars[{len(closes)}<{min_bars}]")
            continue
        rets = _log_returns(closes)
        sigma = _stdev_population(rets)
        if not math.isfinite(sigma) or sigma <= 0:
            missing.append(f"{symbol}:degenerate_sigma")
            continue
        exposure_usd = abs(qty) * px * mult
        exposures.append(
            SymbolExposure(
                symbol=symbol,
                quantity=qty,
                price_usd=px,
                multiplier=mult,
                sec_type=sec_type,
                exposure_usd=exposure_usd,
                daily_sigma=sigma,
                bars_used=len(closes),
            )
        )

    if not exposures:
        return VarReport(
            status="insufficient_data",
            method="parametric",
            portfolio_equity_usd=equity_usd,
            var_95_1day_usd=0.0,
            var_99_1day_usd=0.0,
            symbols_used=[],
            symbols_missing_data=missing,
            notes=notes + ["no_symbols_with_sufficient_data"],
            symbol_count=0,
            var_pct_of_equity=0.0,
            exposures=[],
        )

    sigma_dollar_sq = sum((e.exposure_usd * e.daily_sigma) ** 2 for e in exposures)
    sigma_dollar = math.sqrt(max(0.0, sigma_dollar_sq))
    var_95 = Z_95 * sigma_dollar
    var_99 = Z_99 * sigma_dollar

    pct = (var_95 / equity_usd * 100.0) if equity_usd > 0 else 0.0

    return VarReport(
        status="ok",
        method="parametric",
        portfolio_equity_usd=equity_usd,
        var_95_1day_usd=var_95,
        var_99_1day_usd=var_99,
        symbols_used=[e.symbol for e in exposures],
        symbols_missing_data=missing,
        notes=notes + [f"independent_asset_assumption", f"min_bars={min_bars}"],
        symbol_count=len(exposures),
        var_pct_of_equity=pct,
        exposures=exposures,
    )


def report_to_state_dict(report: VarReport, ts_utc: str, ttl_seconds: int = 3600) -> Dict[str, Any]:
    """Render a VarReport into the var_state.v1 schema."""
    return {
        "schema_version": "var_state.v1",
        "ts_utc": ts_utc,
        "ttl_seconds": int(ttl_seconds),
        "status": report.status,
        "method": report.method,
        "confidence_levels": list(report.confidence_levels),
        "var_95_1day_usd": round(float(report.var_95_1day_usd), 2),
        "var_99_1day_usd": round(float(report.var_99_1day_usd), 2),
        "portfolio_equity_usd": round(float(report.portfolio_equity_usd), 2),
        "var_pct_of_equity": round(float(report.var_pct_of_equity), 4),
        "symbol_count": int(report.symbol_count),
        "symbols_used": list(report.symbols_used),
        "symbols_missing_data": list(report.symbols_missing_data),
        "enforcement_active": False,
        "notes": list(report.notes),
    }


__all__ = [
    "compute_portfolio_var",
    "report_to_state_dict",
    "VarReport",
    "SymbolExposure",
    "Z_95",
    "Z_99",
    "MIN_BARS_FOR_VAR",
]
