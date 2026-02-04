from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Protocol, Tuple
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class MacroSnapshot:
    ts_utc: str
    ttl_seconds: int
    risk_label: str  # risk_on | risk_off | neutral | unknown
    yields: Dict[str, float]        # optional
    curve_slope_10y_2y: float       # optional
    notes: str


class MacroProvider(Protocol):
    def fetch(self) -> MacroSnapshot: ...


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        if v != v or v in (float("inf"), float("-inf")):
            return 0.0
        return v
    except Exception:
        return 0.0


class FREDStLouisFedProvider:
    """
    Minimal public-data provider using St. Louis Fed FRED *without API key* via CSV endpoints.
    If this ever becomes unavailable/rate-limited, we fail closed to unknown.

    Data (best effort):
      - 2y yield: DGS2
      - 10y yield: DGS10

    NOTE: This is a bootstrap provider; swap later behind the same interface.
    """
    BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

    def _fetch_series_last(self, series: str) -> Tuple[bool, float]:
        url = self.BASE.format(series=series)
        req = Request(url, headers={"User-Agent": "chad-macro/1.0"})
        with urlopen(req, timeout=6) as resp:
            raw = resp.read().decode("utf-8", errors="replace").splitlines()

        # Expect header: DATE,VALUE then rows
        # Take last non-empty numeric
        for line in reversed(raw):
            if not line or line.startswith("DATE"):
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            val = parts[1].strip()
            if val == "." or val == "":
                continue
            return True, _safe_float(val)
        return False, 0.0

    def fetch(self) -> MacroSnapshot:
        ts = _utc_now_iso()
        ttl = 1800  # 30 min default

        ok2, y2 = self._fetch_series_last("DGS2")
        ok10, y10 = self._fetch_series_last("DGS10")

        yields: Dict[str, float] = {}
        if ok2:
            yields["us_2y"] = y2
        if ok10:
            yields["us_10y"] = y10

        slope = 0.0
        if ok2 and ok10:
            slope = float(y10 - y2)

        # Simple risk label heuristic (placeholder but deterministic):
        # If slope is strongly negative -> risk_off, positive -> neutral/risk_on
        risk = "unknown"
        if ok2 and ok10:
            if slope < -0.25:
                risk = "risk_off"
            elif slope > 0.25:
                risk = "risk_on"
            else:
                risk = "neutral"

        notes = "bootstrap_provider=fed_fred_csv; summary only; no secrets"
        return MacroSnapshot(
            ts_utc=ts,
            ttl_seconds=ttl,
            risk_label=risk,
            yields=yields,
            curve_slope_10y_2y=slope,
            notes=notes,
        )


def snapshot_to_dict(s: MacroSnapshot) -> Dict[str, Any]:
    return {
        "ts_utc": s.ts_utc,
        "ttl_seconds": int(s.ttl_seconds),
        "yields": dict(s.yields),
        "curve_slope_10y_2y": float(s.curve_slope_10y_2y),
        "risk_label": s.risk_label,
        "notes": s.notes,
        "schema_version": "macro_state.v1",
    }
