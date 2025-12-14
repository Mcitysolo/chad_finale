#!/usr/bin/env python3
"""
legend_top_stocks.py

TEAM CHAD — SEC 13F “Legend” static top-stocks generator.

Freeze spec (Phase-2 — Data & Feeds)
------------------------------------
- Location: /home/ubuntu/chad_finale/utils/legend_top_stocks.py
- Inputs: static fallback Q1 2025 13F-style weights
          for Buffett, Dalio, Grantham.
- Process:
    * Combine legend portfolios
    * Normalize combined weights
    * Select top 25 symbols
    * Renormalize top 25 to sum to 1.0
- Outputs:
    1) /home/ubuntu/chad_finale/data/legend_top_stocks.json
       {
         "as_of": "YYYY-MM-DD",
         "legends": ["buffett", "dalio", "grantham"],
         "weights": { "AAPL": 0.15, ... }
       }

    2) /home/ubuntu/chad_finale/data/legend/top_stocks.csv
       symbol,weight
       AAPL,0.15
       ...

This script is intentionally deterministic and offline – it does NOT
hit EDGAR, WhaleWisdom or any external APIs. Upstream data refresh
can be wired later; for now this is the static freeze-aligned fallback.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple


# -----------------------------------------------------------------------------
# Static fallback portfolios (approximate Q1 2025 style)
# -----------------------------------------------------------------------------
# NOTE: These are engineered as a *structural* fallback: diversified,
# concentrated, and plausible. They are not meant to be exact replicas
# of actual 13F filings and can be updated in a later “Legend Tracker”
# phase when live 13F ingestion is wired.
#
# Each sub-dict is per-legend, weights summing to ~1.0.


BUFFETT_PORTFOLIO: Dict[str, float] = {
    "AAPL": 0.45,
    "BAC": 0.10,
    "KO": 0.07,
    "AXP": 0.07,
    "CVX": 0.06,
    "OXY": 0.05,
    "KHC": 0.04,
    "MCO": 0.03,
    "C": 0.03,
    "VRSN": 0.02,
    "DVA": 0.02,
    "NU": 0.02,
    "JPM": 0.02,
    "V": 0.01,
    "MA": 0.01,
}


DALIO_PORTFOLIO: Dict[str, float] = {
    # Bridgewater-style macro / ETF tilt
    "SPY": 0.20,
    "QQQ": 0.10,
    "IEMG": 0.10,
    "VWO": 0.10,
    "GLD": 0.08,
    "IEF": 0.07,
    "TLT": 0.07,
    "LQD": 0.06,
    "EFA": 0.05,
    "IWM": 0.05,
    "TIP": 0.04,
    "DBC": 0.04,
    "FXI": 0.04,
}


GRANTHAM_PORTFOLIO: Dict[str, float] = {
    # GMO-style value / quality tilts (illustrative)
    "MSFT": 0.12,
    "GOOGL": 0.10,
    "NVDA": 0.08,
    "UNH": 0.07,
    "JNJ": 0.07,
    "PG": 0.06,
    "PEP": 0.06,
    "XOM": 0.06,
    "MRK": 0.05,
    "HD": 0.05,
    "COST": 0.05,
    "LLY": 0.05,
    "LIN": 0.04,
    "ADBE": 0.04,
    "AVGO": 0.04,
}


@dataclass
class LegendConfig:
    name: str
    weights: Dict[str, float]


LEGENDS: List[LegendConfig] = [
    LegendConfig("buffett", BUFFETT_PORTFOLIO),
    LegendConfig("dalio", DALIO_PORTFOLIO),
    LegendConfig("grantham", GRANTHAM_PORTFOLIO),
]


# -----------------------------------------------------------------------------
# Core math helpers
# -----------------------------------------------------------------------------
def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(w for w in weights.values() if w > 0)
    if total <= 0:
        raise ValueError("Cannot normalize empty or non-positive weights")
    return {sym: w / total for sym, w in weights.items() if w > 0}


def _combine_legends(
    legends: List[LegendConfig],
) -> Dict[str, float]:
    combined: Dict[str, float] = {}
    if not legends:
        raise ValueError("No legends configured")

    per_legend_weight = 1.0 / len(legends)

    for legend in legends:
        normalized = _normalize(legend.weights)
        for sym, w in normalized.items():
            combined[sym] = combined.get(sym, 0.0) + per_legend_weight * w

    return combined


def _top_n(weights: Dict[str, float], n: int) -> Dict[str, float]:
    items: List[Tuple[str, float]] = sorted(
        weights.items(), key=lambda kv: kv[1], reverse=True
    )
    trimmed = items[:n]
    return {sym: w for sym, w in trimmed}


# -----------------------------------------------------------------------------
# IO paths (root is parent of utils/)
# -----------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT / "data"
LEGEND_DIR: Path = DATA_DIR / "legend"
JSON_PATH: Path = DATA_DIR / "legend_top_stocks.json"
CSV_PATH: Path = LEGEND_DIR / "top_stocks.csv"


def generate_legend_files(max_symbols: int = 25) -> None:
    """
    Main entrypoint: generate legend_top_stocks.json and legend/top_stocks.csv.
    """
    LEGEND_DIR.mkdir(parents=True, exist_ok=True)

    combined = _combine_legends(LEGENDS)
    top = _top_n(combined, max_symbols)
    top = _normalize(top)  # renormalize top N

    # JSON output
    payload = {
        "as_of": date.today().isoformat(),
        "legends": [l.name for l in LEGENDS],
        "weights": top,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    # CSV output
    lines = ["symbol,weight"]
    # stable ordering (largest first)
    for sym, w in sorted(top.items(), key=lambda kv: kv[1], reverse=True):
        # round to 6dp for readability
        lines.append(f"{sym},{w:.6f}")

    CSV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Sanity check: weights should sum to ~1.0
    s = sum(top.values())
    if not math.isclose(s, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError(f"Legend weights do not sum to 1.0 (got {s})")


def main() -> None:
    generate_legend_files(max_symbols=25)


if __name__ == "__main__":
    main()
