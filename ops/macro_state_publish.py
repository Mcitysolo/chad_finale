#!/usr/bin/env python3
"""
CHAD Market Radar — macro_state publisher (Production)

File: ops/macro_state_publish.py

Outputs:
  /home/ubuntu/chad_finale/runtime/macro_state.json

SSOT contract (v4.2):
  runtime/macro_state.json must exist, be valid JSON, include ts_utc + ttl_seconds,
  and be safe to read while the system runs (atomic write). :contentReference[oaicite:1]{index=1}

Design Goals
------------
- Always writes VALID single JSON object (never concatenated, never literal '\\n')
- Atomic write: tmp -> fsync(file) -> rename -> fsync(dir)
- Fail-closed: if provider fetch fails, write risk_label="unknown"
- Provider interface: swap data source without touching publisher logic
- Async fetch: concurrent sensor retrieval (fast, resilient)
- Cache-aware: if existing macro_state.json is fresh, optionally no-op
- systemd-safe: no interactive prompts, deterministic exit behavior

No broker calls. No secrets are written to runtime.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import urllib.request


# -----------------------------
# Constants / Paths
# -----------------------------

DEFAULT_RUNTIME_DIR = Path("/home/ubuntu/chad_finale/runtime")
OUT_PATH = Path(os.environ.get("CHAD_RUNTIME_DIR", str(DEFAULT_RUNTIME_DIR))) / "macro_state.json"

# TTL (seconds). SSOT suggests 1800 as a sane baseline for macro. :contentReference[oaicite:2]{index=2}
TTL_SECONDS = int(os.environ.get("CHAD_MACRO_TTL_SECONDS", "1800"))

# If true, publisher will skip network work if current file is fresh.
CACHE_RESPECT_TTL = os.environ.get("CHAD_MACRO_RESPECT_TTL", "1").strip().lower() in ("1", "true", "yes", "on")

# Network controls
HTTP_TIMEOUT_S = float(os.environ.get("CHAD_HTTP_TIMEOUT_S", "6.0"))
# Per-FRED-request timeout (longer than generic to tolerate slow CSV serves)
FRED_TIMEOUT_S = float(os.environ.get("CHAD_FRED_TIMEOUT_S", "15.0"))
# FRED's anti-bot layer silently times out Mozilla-style and bespoke
# product UAs (probed empirically) but lets through plain "curl/*" and
# "python-urllib". This was the root cause of the 2026-04-03 TimeoutError
# freeze — overridable via env.
HTTP_UA = os.environ.get("CHAD_HTTP_USER_AGENT", "curl/7.88.1")

# FRED public CSV endpoints (no key). If FRED changes policy, we fail closed.
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

# Additional FRED series for richer macro context (item 1 enrichment).
# Keys: series id; values: short description.
EXTRA_FRED_SERIES: Dict[str, str] = {
    "UNRATE": "unemployment_rate_pct",
    "CPIAUCSL": "cpi_all_urban_index",
    "BAMLH0A0HYM2": "high_yield_spread_pct",
    "T10Y2Y": "treasury_10y_2y_spread_pct",
}

# Composite-risk thresholds
CREDIT_STRESS_THRESHOLD = 4.0          # high-yield spread > 4.0 (400bps)
INFLATION_ELEVATED_THRESHOLD = 3.5     # CPI YoY > 3.5%
RECESSION_UNEMPLOYMENT_THRESHOLD = 5.0 # UNRATE > 5.0%


# -----------------------------
# Data model
# -----------------------------

@dataclass(frozen=True)
class MacroState:
    ts_utc: str
    ttl_seconds: int
    risk_label: str  # legacy: risk_on | risk_off | neutral | unknown
    yields: Dict[str, float]  # optional keys: us_2y, us_10y
    curve_slope_10y_2y: float
    notes: str
    source: Dict[str, Any]
    # Composite (multi-factor) extension — additive, does not replace risk_label
    indicators: Dict[str, float] = dataclasses.field(default_factory=dict)
    risk_flags: Dict[str, bool] = dataclasses.field(default_factory=dict)
    composite_risk_label: str = "unknown"  # high_risk|elevated|moderate|low_risk|unknown
    schema_version: str = "macro_state.v1"


class MacroProvider(Protocol):
    async def fetch(self) -> MacroState: ...


# -----------------------------
# Helpers (safe)
# -----------------------------

def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        if v != v or v in (float("inf"), float("-inf")):
            return 0.0
        return v
    except Exception:
        return 0.0


def sha256_hex(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def canonical_json_bytes(obj: Dict[str, Any]) -> bytes:
    # Deterministic JSON for hashing + stable diffs
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> str:
    """
    Atomic JSON write (crash-safe):
      tmp -> fsync(file) -> os.replace -> fsync(dir)
    Returns sha256 over canonical json bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(obj)
    digest = sha256_hex(data)

    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)

    # best-effort directory fsync for durability
    try:
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        pass

    return digest


def read_existing_if_fresh(path: Path) -> Optional[Dict[str, Any]]:
    """
    If file exists and ts_utc + ttl_seconds indicate it is still fresh, return it.
    Otherwise return None. Never raises.
    """
    try:
        if not path.is_file():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return None
        ts = str(obj.get("ts_utc") or "")
        ttl = int(obj.get("ttl_seconds") or 0)
        if not ts or ttl <= 0:
            return None

        # parse ts_utc (YYYY-MM-DDTHH:MM:SSZ)
        # This is stable format we produce; fail-safe on parse errors.
        try:
            # manual parse to avoid dateutil dependency
            y = int(ts[0:4])
            mo = int(ts[5:7])
            d = int(ts[8:10])
            hh = int(ts[11:13])
            mm = int(ts[14:16])
            ss = int(ts[17:19])
            # convert to epoch (UTC) using time.gmtime-like tuple
            epoch = int(time.mktime((y, mo, d, hh, mm, ss, 0, 0, 0))) - time.timezone
        except Exception:
            return None

        now = int(time.time())
        if now <= epoch + ttl:
            return obj
        return None
    except Exception:
        return None


def _fred_parse_last_value(csv_text: str) -> Tuple[bool, float]:
    """
    Parse the last valid numeric row from FRED csv.
    Expected header: DATE,VALUE
    """
    lines = csv_text.splitlines()
    for line in reversed(lines):
        if not line or line.startswith("DATE"):
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        v = parts[1].strip()
        if v in ("", "."):
            continue
        return True, safe_float(v)
    return False, 0.0


def _fred_parse_recent(csv_text: str, n: int = 24) -> List[Tuple[str, float]]:
    """
    Parse up to n most-recent valid (date, value) rows from FRED csv.
    Order: oldest -> newest. Skips null markers ('.', '').
    """
    lines = csv_text.splitlines()
    out: List[Tuple[str, float]] = []
    for line in reversed(lines):
        if not line or line.startswith("DATE"):
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        v = parts[1].strip()
        if v in ("", "."):
            continue
        out.append((parts[0].strip(), safe_float(v)))
        if len(out) >= n:
            break
    out.reverse()
    return out


async def _http_get_text(url: str, *, timeout: Optional[float] = None) -> str:
    """
    Async wrapper around urllib.request (threaded) to avoid adding new deps.
    """
    eff_timeout = timeout if timeout is not None else HTTP_TIMEOUT_S

    def _blocking() -> str:
        req = urllib.request.Request(url, headers={"User-Agent": HTTP_UA})
        with urllib.request.urlopen(req, timeout=eff_timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")

    return await asyncio.to_thread(_blocking)


async def _fetch_series_safe(series: str) -> Tuple[str, Optional[str]]:
    """
    Fetch a single FRED series CSV with FRED_TIMEOUT_S. Returns (series, csv|None).
    Never raises — failures return (series, None).
    """
    url = FRED_CSV.format(series=series)
    try:
        text = await _http_get_text(url, timeout=FRED_TIMEOUT_S)
        return series, text
    except Exception:
        return series, None


def _compute_cpi_yoy(rows: List[Tuple[str, float]]) -> Optional[float]:
    """
    Compute YoY % change from monthly CPI rows (oldest->newest). Need >=13 rows.
    Returns None if insufficient data or invalid.
    """
    if len(rows) < 13:
        return None
    latest = rows[-1][1]
    year_ago = rows[-13][1]
    if year_ago <= 0:
        return None
    return (latest / year_ago - 1.0) * 100.0


def _composite_risk_label(flags: Dict[str, bool]) -> str:
    inv = bool(flags.get("yield_curve_inverted"))
    credit = bool(flags.get("credit_stress"))
    inflation = bool(flags.get("inflation_elevated"))
    recession = bool(flags.get("recession_risk"))
    n_flags = sum(1 for v in (inv, credit, inflation, recession) if v)

    if inv and (credit or recession):
        return "high_risk"
    if n_flags >= 2:
        return "elevated"
    if n_flags == 1:
        return "moderate"
    return "low_risk"


# -----------------------------
# Provider implementations
# -----------------------------

class FredYieldProvider:
    """
    Macro provider using public FRED CSV for yields plus enrichment series.

    Core: DGS2, DGS10 — yield-curve slope drives legacy risk_label
          (risk_on|risk_off|neutral) for backward compatibility.

    Enrichment: UNRATE, CPIAUCSL, BAMLH0A0HYM2, T10Y2Y — drive a multi-factor
    composite_risk_label (high_risk|elevated|moderate|low_risk) and risk_flags.

    Each series is fetched concurrently with FRED_TIMEOUT_S per request and
    individual failures do not abort the batch.
    """

    async def fetch(self) -> MacroState:
        ts = utc_now_iso()

        all_series = ["DGS2", "DGS10"] + list(EXTRA_FRED_SERIES.keys())
        results = await asyncio.gather(
            *[_fetch_series_safe(s) for s in all_series],
            return_exceptions=False,
        )
        # Map series_id -> csv text or None
        csvs: Dict[str, Optional[str]] = {sid: text for sid, text in results}

        fetched: List[str] = [s for s, t in csvs.items() if t is not None]
        failed: List[str] = [s for s, t in csvs.items() if t is None]

        # ---- Core yields / slope -----------------------------------------
        ok2, y2 = _fred_parse_last_value(csvs.get("DGS2") or "")
        ok10, y10 = _fred_parse_last_value(csvs.get("DGS10") or "")

        yields: Dict[str, float] = {}
        if ok2:
            yields["us_2y"] = float(y2)
        if ok10:
            yields["us_10y"] = float(y10)

        slope = 0.0
        legacy_risk = "unknown"
        if ok2 and ok10:
            slope = float(y10 - y2)
            if slope < -0.25:
                legacy_risk = "risk_off"
            elif slope > 0.25:
                legacy_risk = "risk_on"
            else:
                legacy_risk = "neutral"

        # ---- Enrichment indicators ---------------------------------------
        indicators: Dict[str, float] = {}

        # UNRATE — most recent monthly value
        if csvs.get("UNRATE"):
            ok, v = _fred_parse_last_value(csvs["UNRATE"])
            if ok:
                indicators["unemployment_rate_pct"] = float(v)

        # BAMLH0A0HYM2 — most recent daily HY OAS (already %)
        if csvs.get("BAMLH0A0HYM2"):
            ok, v = _fred_parse_last_value(csvs["BAMLH0A0HYM2"])
            if ok:
                indicators["high_yield_spread_pct"] = float(v)

        # T10Y2Y — most recent daily spread (already %)
        if csvs.get("T10Y2Y"):
            ok, v = _fred_parse_last_value(csvs["T10Y2Y"])
            if ok:
                indicators["treasury_10y_2y_spread_pct"] = float(v)

        # CPIAUCSL — most recent index level + YoY %
        cpi_yoy: Optional[float] = None
        if csvs.get("CPIAUCSL"):
            rows = _fred_parse_recent(csvs["CPIAUCSL"], n=24)
            if rows:
                indicators["cpi_all_urban_index"] = float(rows[-1][1])
                cpi_yoy = _compute_cpi_yoy(rows)
                if cpi_yoy is not None:
                    indicators["cpi_yoy_pct"] = float(cpi_yoy)

        # ---- Composite flags ---------------------------------------------
        flags: Dict[str, bool] = {}

        # yield_curve_inverted: prefer T10Y2Y direct read, else slope from DGS
        t10y2y = indicators.get("treasury_10y_2y_spread_pct")
        if t10y2y is not None:
            flags["yield_curve_inverted"] = t10y2y < 0.0
        elif ok2 and ok10:
            flags["yield_curve_inverted"] = slope < 0.0

        # credit_stress
        hy = indicators.get("high_yield_spread_pct")
        if hy is not None:
            flags["credit_stress"] = hy > CREDIT_STRESS_THRESHOLD

        # inflation_elevated
        if cpi_yoy is not None:
            flags["inflation_elevated"] = cpi_yoy > INFLATION_ELEVATED_THRESHOLD

        # recession_risk
        unrate = indicators.get("unemployment_rate_pct")
        if unrate is not None:
            flags["recession_risk"] = unrate > RECESSION_UNEMPLOYMENT_THRESHOLD

        # Composite label requires at least one flag computed; otherwise unknown.
        composite = _composite_risk_label(flags) if flags else "unknown"

        notes_bits = ["provider=fed_fred_csv"]
        if failed:
            notes_bits.append("failed=" + ",".join(failed))

        return MacroState(
            ts_utc=ts,
            ttl_seconds=TTL_SECONDS,
            risk_label=legacy_risk,
            yields=yields,
            curve_slope_10y_2y=slope,
            notes="; ".join(notes_bits),
            source={
                "provider": "FredYieldProvider",
                "series": ["DGS2", "DGS10"] + list(EXTRA_FRED_SERIES.keys()),
                "fetched": fetched,
                "failed": failed,
            },
            indicators=indicators,
            risk_flags=flags,
            composite_risk_label=composite,
        )


# -----------------------------
# Publisher / Orchestration
# -----------------------------

def state_to_dict(s: MacroState) -> Dict[str, Any]:
    return {
        "ts_utc": s.ts_utc,
        "ttl_seconds": int(s.ttl_seconds),
        "yields": dict(s.yields),
        "curve_slope_10y_2y": float(s.curve_slope_10y_2y),
        "risk_label": s.risk_label,
        "indicators": dict(s.indicators),
        "risk_flags": dict(s.risk_flags),
        "composite_risk_label": s.composite_risk_label,
        "notes": s.notes,
        "source": dict(s.source),
        "schema_version": s.schema_version,
    }


async def build_state(provider: MacroProvider) -> MacroState:
    return await provider.fetch()


def fail_closed_state(ts: str, *, reason: str) -> Dict[str, Any]:
    # DS06: shape must mirror state_to_dict() exactly so consumers see one
    # canonical schema regardless of whether the publisher succeeded.
    return {
        "ts_utc": ts,
        "ttl_seconds": int(TTL_SECONDS),
        "yields": {},
        "curve_slope_10y_2y": 0.0,
        "risk_label": "unknown",
        "indicators": {},
        "risk_flags": {},
        "composite_risk_label": "unknown",
        "notes": reason,
        "source": {"provider": "error"},
        "schema_version": "macro_state.v1",
    }


def _read_existing_any(path: Path) -> Optional[Dict[str, Any]]:
    """Read existing macro_state.json regardless of freshness. Never raises."""
    try:
        if not path.is_file():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


async def main_async() -> int:
    ts = utc_now_iso()

    if CACHE_RESPECT_TTL:
        existing = read_existing_if_fresh(OUT_PATH)
        if existing is not None:
            # No-op, but still exit 0 (healthy).
            print(json.dumps({"ok": True, "out": str(OUT_PATH), "ts_utc": existing.get("ts_utc"), "cached": True}, sort_keys=True))
            return 0

    provider: MacroProvider = FredYieldProvider()

    try:
        st = await build_state(provider)
        # If every series failed, prefer to preserve any prior usable data
        # rather than overwriting with an empty state.
        if not st.yields and not st.indicators:
            prior = _read_existing_any(OUT_PATH)
            prior_has_data = (
                isinstance(prior, dict)
                and (prior.get("yields") or prior.get("indicators"))
            )
            if prior_has_data:
                payload = dict(prior)
                payload["ts_utc"] = ts
                payload["notes"] = (
                    "fetch_failed_all_series; preserved_prior_indicators"
                )
                src = dict(payload.get("source") or {})
                src["last_refresh_failed_utc"] = ts
                payload["source"] = src
                digest = atomic_write_json(OUT_PATH, payload)
                print(json.dumps({"ok": False, "out": str(OUT_PATH), "ts_utc": ts, "sha256": digest, "preserved_prior": True}, sort_keys=True))
                return 0
        payload = state_to_dict(st)
        digest = atomic_write_json(OUT_PATH, payload)
        print(json.dumps({"ok": True, "out": str(OUT_PATH), "ts_utc": st.ts_utc, "sha256": digest, "cached": False}, sort_keys=True))
        return 0
    except Exception as exc:
        # Fall back to existing valid macro_state if available
        prior = _read_existing_any(OUT_PATH)
        prior_has_data = (
            isinstance(prior, dict)
            and (prior.get("yields") or prior.get("indicators"))
        )
        if prior_has_data:
            payload = dict(prior)
            payload["ts_utc"] = ts
            payload["notes"] = f"publish_error:{type(exc).__name__}; preserved_prior"
            src = dict(payload.get("source") or {})
            src["last_refresh_failed_utc"] = ts
            payload["source"] = src
            digest = atomic_write_json(OUT_PATH, payload)
            print(json.dumps({"ok": False, "out": str(OUT_PATH), "ts_utc": ts, "sha256": digest, "error": str(exc), "preserved_prior": True}, sort_keys=True))
            return 0
        payload = fail_closed_state(ts, reason=f"publish_error:{type(exc).__name__}")
        digest = atomic_write_json(OUT_PATH, payload)
        # Fail-soft (exit 0) so timer doesn't become a restart storm. Runtime file still indicates unknown.
        print(json.dumps({"ok": False, "out": str(OUT_PATH), "ts_utc": ts, "sha256": digest, "error": str(exc)}, sort_keys=True))
        return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())

