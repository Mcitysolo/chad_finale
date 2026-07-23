"""W4B-1 (J16): exit-advice recorder — D4-dropped urges become typed records.

The ctx-positions D4 guardrail (``context_positions.filter_overlay_owned_exits``)
drops every strategy equity/ETF SELL in ON mode so the ACTIVE exit overlay stays
the sole equity/ETF exit authority. Before W4B those dropped urges were discarded
objects — only the integer ``exits_filtered`` survived (heartbeat). This module
records each dropped urge as an ``exit_advice.v1`` row so zero information is
discarded; W4B-3+ lets the overlay evaluate the fresh rows as ADVICE under its
own rules (never as a second sell authority).

Modes (env ``CHAD_EXIT_ADVICE``; default off; garbage -> off):
  off     -> no rows are written; the state heartbeat still stamps ``mode=off``
             (R25 doctrine: "installed but off" must be distinguishable from
             "dead") — writes are best-effort and never affect the cycle.
  record  -> rows appended to ``data/exit_advice/exit_advice_YYYYMMDD.ndjson``;
             the overlay computes ADVICE_WOULD_CLOSE evidence only (W4B-3).
  consume -> as record, plus the overlay may act on advice (W4B-4; the flip to
             consume is a SEPARATE operator GO gated on corpus review — D6 rider).

House rules honored:
  - evidence under ``data/`` never ``runtime/`` (repo write-guard; the
    ctx_positions_shadow precedent); liveness state under ``runtime/``.
  - paths injectable (arg > env > cwd default). Under pytest, default-path
    writes are SKIPPED unless the caller passed an explicit path — existing
    suites monkeypatch ``record_cycle`` but cannot know about this recorder,
    so it must not leak repo writes into them (G3C-HF).
  - observer-only: ``record_dropped_urges`` never raises; a recorder failure
    never breaks a trading cycle.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

LOG = logging.getLogger(__name__)

SCHEMA_VERSION = "exit_advice.v1"
STATE_SCHEMA_VERSION = "exit_advice_state.v1"

MODE_ENV = "CHAD_EXIT_ADVICE"
_VALID_MODES = ("off", "record", "consume")

_EVIDENCE_DIR_ENV = "CHAD_EXIT_ADVICE_EVIDENCE_DIR"
_STATE_PATH_ENV = "CHAD_EXIT_ADVICE_STATE_PATH"

# Heartbeat cadence is the live-loop cycle (~72s); 300s = ~4 cycles of grace,
# matching the drawdown_state/price_cache tier of runtime artifacts.
STATE_TTL_SECONDS = 300

# Sites are the three D4 filter call sites (PLAN_W4B §3.2).
SITE_ROUTER = "router"
SITE_PIPELINE = "pipeline"
SITE_FULL_CYCLE = "full_cycle"

# Cap the per-day tuple rollup in the state file so a pathological cycle can
# never grow the heartbeat without bound (values live in evidence rows).
_STATE_KEY_CAP = 50

# W4B-3: advice consumption tunables (config/exit_advice.json — D7: the frozen
# overlay config is never amended; advice policy lives in its own file).
CONFIG_SCHEMA_VERSION = "exit_advice_config.v1"
_CONFIG_PATH_ENV = "CHAD_EXIT_ADVICE_CONFIG_PATH"
# ~2 live-loop cycles (~72s each) + grace: stale advice must expire fast —
# a strategy that has stopped urging is no longer advising.
ADVICE_TTL_SECONDS_DEFAULT = 180
MIN_CONFIDENCE_DEFAULT = 0.0


def resolve_exit_advice_mode(env: Optional[Mapping[str, str]] = None) -> str:
    """``off | record | consume``; anything else (or unset) -> ``off``."""
    raw = str((env if env is not None else os.environ).get(MODE_ENV, "") or "")
    raw = raw.strip().lower()
    return raw if raw in _VALID_MODES else "off"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _under_active_pytest() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


def _evidence_dir(explicit: Optional[Any]) -> Optional[Path]:
    if explicit is not None:
        return Path(explicit)
    env = os.getenv(_EVIDENCE_DIR_ENV)
    if env:
        return Path(env)
    if _under_active_pytest():  # G3C-HF: never default into the repo under pytest
        return None
    return Path.cwd() / "data" / "exit_advice"


def _state_path(explicit: Optional[Any]) -> Optional[Path]:
    if explicit is not None:
        return Path(explicit)
    env = os.getenv(_STATE_PATH_ENV)
    if env:
        return Path(env)
    if _under_active_pytest():
        return None
    return Path.cwd() / "runtime" / "exit_advice_state.json"


def _default_excluded_symbols() -> Optional[frozenset]:
    """Operator-exclusion SSOT (same source the overlay uses). ``None`` = unknown
    (never guess an empty exclusion set from an import failure)."""
    try:
        from chad.core.position_reconciler import _EFFECTIVE_NON_CHAD_SYMBOLS
        return frozenset(_EFFECTIVE_NON_CHAD_SYMBOLS)
    except Exception:  # pragma: no cover - defensive
        return None


def _strategy_of(sig: Any) -> Optional[str]:
    """Strategy identity: TradeSignal.strategy or RoutedSignal.source_strategies[0]."""
    strat = getattr(sig, "strategy", None)
    if strat is not None:
        return str(getattr(strat, "value", strat))
    sources = getattr(sig, "source_strategies", None) or ()
    if sources:
        return str(getattr(sources[0], "value", sources[0]))
    return None


def _source_strategies_of(sig: Any) -> List[str]:
    sources = getattr(sig, "source_strategies", None) or ()
    return [str(getattr(s, "value", s)) for s in sources]


def _size_of(sig: Any) -> Optional[float]:
    for attr in ("size", "net_size"):
        val = getattr(sig, attr, None)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
    return None


def _position_open(view: Any, symbol: str) -> Optional[bool]:
    """True/False from a KNOWN PositionsView; ``None`` when unknowable."""
    if not symbol or view is None or not getattr(view, "known", False):
        return None
    positions = getattr(view, "positions", None) or {}
    return symbol in positions


def row_from_signal(
    sig: Any,
    *,
    site: str,
    mode: str,
    now: datetime,
    view: Any = None,
    excluded_symbols: Optional[frozenset] = None,
) -> Dict[str, Any]:
    """One ``exit_advice.v1`` row for a D4-dropped urge. Pure; never raises on
    a well-formed signal-ish object (missing attrs degrade to ``None``)."""
    symbol = str(getattr(sig, "symbol", "") or "").strip().upper()
    side = getattr(sig, "side", None)
    meta = getattr(sig, "meta", None) or {}
    asset_class = getattr(sig, "asset_class", None)
    confidence = getattr(sig, "confidence", None)
    excluded: Optional[bool]
    if excluded_symbols is None:
        excluded = None
    else:
        excluded = symbol in excluded_symbols
    return {
        "schema_version": SCHEMA_VERSION,
        "ts_utc": _iso(now),
        "site": site,
        "mode": mode,
        "strategy": _strategy_of(sig),
        "source_strategies": _source_strategies_of(sig),
        "symbol": symbol,
        "side": str(getattr(side, "value", None) or getattr(side, "name", None) or side or "").upper(),
        "size": _size_of(sig),
        "confidence": (float(confidence) if confidence is not None else None),
        "asset_class": str(getattr(asset_class, "value", asset_class) or "") or None,
        "reason": (str(meta.get("reason")) if isinstance(meta, Mapping) and meta.get("reason") else None),
        "excluded": excluded,
        "position_open": _position_open(view, symbol),
    }


def _append_rows(directory: Path, rows: List[Mapping[str, Any]], now: datetime) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fname = directory / f"exit_advice_{now.astimezone(timezone.utc).strftime('%Y%m%d')}.ndjson"
    with fname.open("a", encoding="utf-8") as fh:
        for record in rows:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


def _load_state(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            obj = json.load(fh)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _write_state(
    path: Path,
    *,
    mode: str,
    site: str,
    rows: List[Mapping[str, Any]],
    now: datetime,
) -> None:
    """Atomic heartbeat + per-day rollup. Read-modify-write so the counters
    survive across cycles; day rollover resets. Values live in evidence rows;
    the state carries identities + counts only (CTF-T2)."""
    prev = _load_state(path)
    today = now.astimezone(timezone.utc).strftime("%Y-%m-%d")
    prev_today = prev.get("today") if isinstance(prev.get("today"), dict) else {}
    if prev_today.get("date") != today:
        prev_today = {"date": today, "recorded": 0, "excluded": 0, "by_key": {}}
    by_key = dict(prev_today.get("by_key") or {})
    excluded_n = int(prev_today.get("excluded") or 0)
    for row in rows:
        key = f"{row.get('strategy')}|{row.get('symbol')}|{row.get('side')}"
        if key in by_key or len(by_key) < _STATE_KEY_CAP:
            by_key[key] = int(by_key.get(key, 0)) + 1
        if row.get("excluded"):
            excluded_n += 1
    state = {
        "schema_version": STATE_SCHEMA_VERSION,
        "ts_utc": _iso(now),
        "ttl_seconds": STATE_TTL_SECONDS,
        "mode": mode,
        "last_site": site,
        "last_count": len(rows),
        "today": {
            "date": today,
            "recorded": int(prev_today.get("recorded") or 0) + len(rows),
            "excluded": excluded_n,
            "by_key": by_key,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def record_dropped_urges(
    dropped: Optional[List[Any]],
    *,
    site: str,
    mode: Optional[str] = None,
    view: Any = None,
    excluded_symbols: Optional[frozenset] = None,
    evidence_dir: Optional[Any] = None,
    state_path: Optional[Any] = None,
    logger: Any = None,
    now: Optional[datetime] = None,
) -> int:
    """Record D4-dropped urges as advice rows. Returns rows recorded (0 in off).

    Never raises. In ``off`` mode no rows are built, but the state heartbeat is
    still stamped (mode-off liveness). Under pytest, default-path writes are
    skipped unless the caller injected explicit paths (G3C-HF)."""
    try:
        mode = mode if mode in _VALID_MODES else resolve_exit_advice_mode()
        now = now or _utcnow()
        rows: List[Dict[str, Any]] = []
        if mode != "off" and dropped:
            if excluded_symbols is None:
                excluded_symbols = _default_excluded_symbols()
            for sig in dropped:
                try:
                    rows.append(row_from_signal(
                        sig, site=site, mode=mode, now=now,
                        view=view, excluded_symbols=excluded_symbols,
                    ))
                except Exception:  # pragma: no cover - one bad object never stops the rest
                    continue
        if rows:
            directory = _evidence_dir(evidence_dir)
            if directory is not None:
                try:
                    _append_rows(directory, rows, now)
                except Exception as exc:  # best-effort evidence
                    (logger or LOG).warning("exit_advice evidence write failed (non-fatal): %s", exc)
        sp = _state_path(state_path)
        if sp is not None:
            try:
                _write_state(sp, mode=mode, site=site, rows=rows, now=now)
            except Exception as exc:  # best-effort liveness
                (logger or LOG).warning("exit_advice state write failed (non-fatal): %s", exc)
        if rows and logger is not None:
            logger.info("EXIT_ADVICE_RECORDED site=%s n=%d mode=%s", site, len(rows), mode)
        return len(rows)
    except Exception as exc:  # pragma: no cover - absolute backstop
        (logger or LOG).warning("exit_advice recorder failed (non-fatal): %s", exc)
        return 0


# --------------------------------------------------------------------------- #
# W4B-3: aggregation — recorded rows -> fresh, consumable advice per symbol
# --------------------------------------------------------------------------- #

def _config_path(explicit: Optional[Any]) -> Optional[Path]:
    if explicit is not None:
        return Path(explicit)
    env = os.getenv(_CONFIG_PATH_ENV)
    if env:
        return Path(env)
    return Path.cwd() / "config" / "exit_advice.json"


def load_advice_config(path: Optional[Any] = None) -> Dict[str, Any]:
    """Advice policy tunables. Missing/corrupt file -> safe defaults (the
    policy must be prewritten, but its absence must never invent behavior
    beyond the conservative defaults)."""
    defaults = {
        "advice_ttl_seconds": ADVICE_TTL_SECONDS_DEFAULT,
        "min_confidence": MIN_CONFIDENCE_DEFAULT,
    }
    p = _config_path(path)
    if p is None:
        return defaults
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return defaults
        out = dict(defaults)
        if isinstance(obj.get("advice_ttl_seconds"), (int, float)) and obj["advice_ttl_seconds"] > 0:
            out["advice_ttl_seconds"] = float(obj["advice_ttl_seconds"])
        if isinstance(obj.get("min_confidence"), (int, float)):
            out["min_confidence"] = float(obj["min_confidence"])
        return out
    except Exception:
        return defaults


def _parse_row_ts(raw: Any) -> Optional[datetime]:
    try:
        return datetime.strptime(str(raw), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _iter_recent_rows(directory: Path, now: datetime):
    """Yield rows from today's (and, near midnight, yesterday's) evidence file.
    Corrupt lines are skipped — evidence is append-only best-effort."""
    from datetime import timedelta
    for day in (now, now - timedelta(days=1)):
        fname = directory / f"exit_advice_{day.astimezone(timezone.utc).strftime('%Y%m%d')}.ndjson"
        try:
            with fname.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        yield obj
        except FileNotFoundError:
            continue
        except Exception:
            continue


def load_advice_by_symbol(
    *,
    now: Optional[datetime] = None,
    evidence_dir: Optional[Any] = None,
    excluded_symbols: Optional[frozenset] = None,
    config_path: Optional[Any] = None,
) -> Dict[str, Dict[str, Any]]:
    """Aggregate recorded urges into ``{SYMBOL: advice}`` of FRESH advice.

    Consumability filters (each one a hard wall):
      - row is a SELL urge (advice means "exit a long"; BUYs are never dropped
        by D4, so no advice exists for shorts by construction);
      - row is fresh: ``now - ts_utc <= advice_ttl_seconds``;
      - row confidence >= min_confidence (None -> not consumable);
      - symbol NOT operator-excluded (``excluded_symbols`` wall here, the
        row's own ``excluded`` flag as belt-and-braces, and the overlay's
        SKIP_EXCLUDED + apply_close_intents backstops downstream).

    Advice shape: ``{"symbol", "strategies": [..], "latest_ts_utc",
    "max_confidence", "reasons": [..], "count_fresh"}``. Never raises;
    unreadable evidence -> {}.
    """
    try:
        now = now or _utcnow()
        directory = _evidence_dir(evidence_dir)
        if directory is None:
            return {}
        cfg = load_advice_config(config_path)
        ttl = float(cfg["advice_ttl_seconds"])
        min_conf = float(cfg["min_confidence"])
        if excluded_symbols is None:
            excluded_symbols = _default_excluded_symbols() or frozenset()
        out: Dict[str, Dict[str, Any]] = {}
        for row in _iter_recent_rows(directory, now):
            if str(row.get("side", "")).upper() != "SELL":
                continue
            symbol = str(row.get("symbol", "") or "").upper()
            if not symbol or symbol in excluded_symbols or row.get("excluded"):
                continue
            ts = _parse_row_ts(row.get("ts_utc"))
            if ts is None or (now - ts).total_seconds() > ttl or ts > now:
                continue
            conf = row.get("confidence")
            if conf is None or float(conf) < min_conf:
                continue
            strategy = row.get("strategy")
            if not strategy:
                continue
            entry = out.setdefault(symbol, {
                "symbol": symbol, "strategies": set(), "latest_ts_utc": row.get("ts_utc"),
                "max_confidence": float(conf), "reasons": set(), "count_fresh": 0,
            })
            entry["strategies"].add(str(strategy))
            entry["count_fresh"] += 1
            entry["max_confidence"] = max(entry["max_confidence"], float(conf))
            if row.get("ts_utc") and str(row["ts_utc"]) > str(entry["latest_ts_utc"]):
                entry["latest_ts_utc"] = row["ts_utc"]
            if row.get("reason"):
                entry["reasons"].add(str(row["reason"]))
        for entry in out.values():
            entry["strategies"] = sorted(entry["strategies"])
            entry["reasons"] = sorted(entry["reasons"])
        return out
    except Exception:  # pragma: no cover - aggregation must never break a cycle
        return {}
