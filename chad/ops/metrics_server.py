#!/usr/bin/env python3
"""
chad/ops/metrics_server.py

CHAD Metrics Server — Production-Grade Prometheus Exporter (Phase 6/7/11)

Serves:
  - /metrics (Prometheus text format 0.0.4)
  - /health

Computes paper performance rollups from append-only ledgers:
  - data/trades/trade_history_YYYYMMDD.ndjson

RAW paper rollup (NOT SCR-aligned — diverges intentionally):
  - source: data/trades/trade_history_*.ndjson, days_back=7
  - excludes paper_sim artifacts
  - excludes live trades (is_live=True)
  - excludes non-finite pnl
  - excludes untrusted pnl rows ("pnl_untrusted" tags/flags)
  - excludes legacy alpha_crypto rows with pnl==0.0 (treated as unknown realized outcome)
  - does NOT consult runtime/quarantine_manifest_*.json
  - does NOT apply the Epoch-2 cutoff
  - DIVERGES from SCR canonical. For canonical SCR PnL truth use the
    chad_scr_* metrics (chad_scr_total_pnl, chad_scr_win_rate, etc.).

Why exclude alpha_crypto pnl==0.0?
  Historically, Kraken/crypto logging produced many rows with pnl=0.0 without
  realized outcome calculation. Treating these as trusted destroys win_rate.
  Until real PnL enrichment exists, we mark them untrusted for metrics.

Test/compat requirements:
  - MetricLine
  - _escape_label
  - _normalize_trade_record
  - _paper_rollup_metrics returns List[MetricLine]
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


# -----------------------------
# Paths / Defaults
# -----------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_TRADES = REPO_ROOT / "data" / "trades"
RUNTIME_DIR = REPO_ROOT / "runtime"
RUNTIME_SCR_PATH = RUNTIME_DIR / "scr_state.json"
RUNTIME_VAR_STATE_PATH = RUNTIME_DIR / "var_state.json"
RUNTIME_DRAWDOWN_STATE_PATH = RUNTIME_DIR / "drawdown_state.json"

DEFAULT_HOST = os.environ.get("CHAD_METRICS_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("CHAD_METRICS_PORT", "9620"))
DEFAULT_DAYS_BACK = int(os.environ.get("CHAD_METRICS_DAYS_BACK", "7"))
DEFAULT_MAX_TRADES = int(os.environ.get("CHAD_METRICS_MAX_TRADES", "5000"))

TRADE_FILE_GLOB = "trade_history_*.ndjson"


# -----------------------------
# Prometheus primitives
# -----------------------------

@dataclass(frozen=True)
class MetricLine:
    name: str
    labels: Dict[str, str]
    value: float

    def render(self) -> str:
        if self.labels:
            lbl = ",".join(f'{k}="{_escape_label(v)}"' for k, v in sorted(self.labels.items()))
            return f"{self.name}{{{lbl}}} {self.value}"
        return f"{self.name} {self.value}"


def _escape_label(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _finite_or_zero(x: float) -> float:
    return x if math.isfinite(x) else 0.0


# -----------------------------
# JSON helpers
# -----------------------------

def _safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    s = line.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.is_file():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _coerce_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


# -----------------------------
# Trade normalization (compat)
# -----------------------------

def _normalize_trade_record(rec: dict) -> dict | None:
    """
    Normalize a trade ledger envelope into a metrics-friendly payload record.

    Contract:
      - Return None for paper_sim-tagged records.
      - Return None if payload missing/invalid.
      - Otherwise return payload dict.
    """
    try:
        payload = rec.get("payload") if isinstance(rec, dict) else None
        if not isinstance(payload, dict):
            return None
        tags = payload.get("tags", [])
        if isinstance(tags, list) and any(str(x).lower() == "paper_sim" for x in tags):
            return None
        return payload
    except Exception:
        return None


def _is_live_trade(payload: Dict[str, Any]) -> bool:
    return payload.get("is_live") is True


def _is_untrusted_pnl(payload: Dict[str, Any]) -> bool:
    """
    True if payload should be excluded from the RAW paper-rollup stats
    (NOT SCR-aligned — does NOT consult runtime/quarantine_manifest_*.json;
    DIVERGES from SCR canonical. For SCR truth use chad_scr_* metrics).
    """
    try:
        # Tag/flag-based untrusted
        tags = payload.get("tags") or []
        tagged = isinstance(tags, list) and any(str(x).lower() == "pnl_untrusted" for x in tags)
        flagged = bool(payload.get("pnl_untrusted", False))
        extra = payload.get("extra") or {}
        extra_flag = isinstance(extra, dict) and extra.get("pnl_untrusted") is True
        if tagged or flagged or extra_flag:
            return True

        # Backward-compat safety: legacy alpha_crypto rows logged with pnl=0.0 represent unknown realized outcomes.
        strat = str(payload.get("strategy", "")).strip().lower()
        if strat == "alpha_crypto":
            try:
                pnl = float(payload.get("pnl", 0.0))
            except Exception:
                pnl = 0.0
            if pnl == 0.0:
                return True

        return False
    except Exception:
        return False


def _extract_strategy(payload: Dict[str, Any]) -> str:
    s = payload.get("strategy")
    if isinstance(s, str) and s.strip():
        return s.strip().lower()
    return "unknown"


def _extract_pnl(payload: Dict[str, Any]) -> Optional[float]:
    try:
        return float(payload.get("pnl", 0.0))
    except Exception:
        return None


# -----------------------------
# Paper trade loading
# -----------------------------

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iter_trade_files(days_back: int) -> List[Path]:
    if not DATA_TRADES.exists():
        return []
    files = sorted(DATA_TRADES.glob(TRADE_FILE_GLOB), reverse=True)
    if not files:
        return []
    cutoff = _utc_now().date()
    keep: List[Path] = []
    for f in files:
        try:
            ymd = f.name.split("_", 2)[2].split(".", 1)[0]
            dt = datetime.strptime(ymd, "%Y%m%d").date()
            delta = (cutoff - dt).days
            if 0 <= delta <= max(days_back, 0):
                keep.append(f)
        except Exception:
            keep.append(f)
    return keep


def load_recent_paper_trades(
    *,
    days_back: int,
    max_trades: int,
    trades_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    root = trades_dir if trades_dir is not None else DATA_TRADES
    if not root.exists():
        return out

    files = sorted(root.glob(TRADE_FILE_GLOB), reverse=True) if trades_dir is not None else _iter_trade_files(days_back)

    for f in files:
        try:
            lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue

        for line in lines:
            rec = _safe_json_loads(line)
            if not rec:
                continue

            payload = _normalize_trade_record(rec)
            if payload is None:
                continue

            if _is_live_trade(payload):
                continue

            out.append(payload)
            if len(out) >= max_trades:
                return out

    return out


# -----------------------------
# Performance math
# -----------------------------

def _compute_sharpe_like(pnls: List[float]) -> float:
    if not pnls:
        return 0.0
    mu = sum(pnls) / float(len(pnls))
    var = sum((x - mu) ** 2 for x in pnls) / float(len(pnls))
    sd = math.sqrt(var) if var > 0.0 else 0.0
    if sd <= 0.0 or not math.isfinite(sd):
        return 0.0
    return float(mu / sd)


def _compute_max_drawdown(pnls: List[float]) -> float:
    if not pnls:
        return 0.0
    peak = 0.0
    cur = 0.0
    max_dd = 0.0
    for x in pnls:
        cur += float(x)
        if cur > peak:
            peak = cur
        dd = peak - cur
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


# -----------------------------
# Rollups (tests rely on this)
# -----------------------------

def _paper_rollup_metrics(
    trades_dir: Path | None = None,
    *,
    days_back: int = DEFAULT_DAYS_BACK,
    max_trades: int = DEFAULT_MAX_TRADES,
) -> List[MetricLine]:
    """
    Return MetricLine list for paper rollups (tests rely on this).

    RAW paper-rollup set == finite pnl AND trusted pnl (NOT SCR-aligned;
    does NOT consult runtime/quarantine_manifest_*.json — DIVERGES from
    SCR canonical. For SCR truth use chad_scr_* metrics).
    """
    trades_raw = load_recent_paper_trades(days_back=days_back, max_trades=max_trades, trades_dir=trades_dir)

    total_trades_raw = float(len(trades_raw))
    untrusted_count = float(sum(1.0 for t in trades_raw if _is_untrusted_pnl(t)))

    pnls: List[float] = []
    pnl_nonfinite_count = 0.0

    for t in trades_raw:
        if _is_untrusted_pnl(t):
            continue
        pnl = _extract_pnl(t)
        if pnl is None or not math.isfinite(pnl):
            pnl_nonfinite_count += 1.0
            continue
        pnls.append(float(pnl))

    total_trades = float(len(pnls))
    winners = float(sum(1 for p in pnls if p > 0.0))
    losers = float(sum(1 for p in pnls if p < 0.0))
    win_rate = float(winners / total_trades) if total_trades > 0 else 0.0
    total_pnl = float(sum(pnls)) if pnls else 0.0
    avg_pnl = float(total_pnl / total_trades) if total_trades > 0 else 0.0

    lines: List[MetricLine] = []
    lines.append(MetricLine("chad_paper_trades_total_raw", {}, _finite_or_zero(total_trades_raw)))
    lines.append(MetricLine("chad_paper_trades_total", {}, _finite_or_zero(total_trades)))
    lines.append(MetricLine("chad_paper_pnl_nonfinite_count", {}, _finite_or_zero(pnl_nonfinite_count)))
    lines.append(MetricLine("chad_paper_pnl_untrusted_count", {}, _finite_or_zero(untrusted_count)))
    lines.append(MetricLine("chad_paper_win_rate", {}, _finite_or_zero(win_rate)))
    lines.append(MetricLine("chad_paper_total_pnl", {}, _finite_or_zero(total_pnl)))
    lines.append(MetricLine("chad_paper_avg_pnl", {}, _finite_or_zero(avg_pnl)))
    lines.append(MetricLine("chad_paper_sharpe_like", {}, _finite_or_zero(_compute_sharpe_like(pnls))))
    lines.append(MetricLine("chad_paper_max_drawdown", {}, _finite_or_zero(_compute_max_drawdown(pnls))))
    lines.append(MetricLine("chad_paper_winners", {}, _finite_or_zero(winners)))
    lines.append(MetricLine("chad_paper_losers", {}, _finite_or_zero(losers)))
    return lines


def _paper_strategy_lines(trades: List[Dict[str, Any]]) -> List[MetricLine]:
    buckets: Dict[str, List[float]] = {}
    for t in trades:
        if _is_untrusted_pnl(t):
            continue
        pnl = _extract_pnl(t)
        if pnl is None or not math.isfinite(pnl):
            continue
        strat = _extract_strategy(t)
        buckets.setdefault(strat, []).append(float(pnl))

    lines: List[MetricLine] = []
    for strat, pnls in sorted(buckets.items()):
        total = float(len(pnls))
        wins = float(sum(1 for p in pnls if p > 0.0))
        win_rate = float(wins / total) if total > 0 else 0.0
        total_pnl = float(sum(pnls)) if pnls else 0.0

        lines.append(MetricLine("chad_paper_strategy_trades_total", {"strategy": strat}, _finite_or_zero(total)))
        lines.append(MetricLine("chad_paper_strategy_total_pnl", {"strategy": strat}, _finite_or_zero(total_pnl)))
        lines.append(MetricLine("chad_paper_strategy_win_rate", {"strategy": strat}, _finite_or_zero(win_rate)))

    return lines


# -----------------------------
# SCR snapshot (best-effort)
# -----------------------------

def _load_scr_snapshot() -> Dict[str, Any]:
    obj = _read_json_file(RUNTIME_SCR_PATH)
    return obj if isinstance(obj, dict) else {}


def _scr_lines() -> List[MetricLine]:
    scr = _load_scr_snapshot()
    stats = scr.get("stats") if isinstance(scr.get("stats"), dict) else {}
    if not isinstance(stats, dict):
        stats = {}

    effective = float(_coerce_float(stats.get("effective_trades", 0), 0.0))
    excluded_manual = float(_coerce_float(stats.get("excluded_manual", 0), 0.0))
    excluded_untrusted = float(_coerce_float(stats.get("excluded_untrusted", 0), 0.0))
    excluded_nonfinite = float(_coerce_float(stats.get("excluded_nonfinite", 0), 0.0))

    # Canonical SCR PnL truth (mirrored from runtime/scr_state.json::stats).
    # Post-Epoch-2 cutoff, quarantine-manifest filtered, 60-day window.
    total_pnl = float(_coerce_float(stats.get("total_pnl", 0.0), 0.0))
    win_rate = float(_coerce_float(stats.get("win_rate", 0.0), 0.0))
    sharpe_like = float(_coerce_float(stats.get("sharpe_like", 0.0), 0.0))
    max_dd = float(_coerce_float(stats.get("max_drawdown", 0.0), 0.0))

    state = str(scr.get("state", "UNKNOWN")).upper()
    paper_only = bool(scr.get("paper_only", True))
    sizing_factor = float(_coerce_float(scr.get("sizing_factor", 0.0), 0.0))

    out: List[MetricLine] = []
    out.append(MetricLine("chad_scr_effective_trades", {}, _finite_or_zero(effective)))
    out.append(MetricLine("chad_scr_excluded_manual", {}, _finite_or_zero(excluded_manual)))
    out.append(MetricLine("chad_scr_excluded_untrusted", {}, _finite_or_zero(excluded_untrusted)))
    out.append(MetricLine("chad_scr_excluded_nonfinite", {}, _finite_or_zero(excluded_nonfinite)))
    out.append(MetricLine("chad_scr_sizing_factor", {}, _finite_or_zero(sizing_factor)))
    out.append(MetricLine("chad_scr_paper_only", {}, 1.0 if paper_only else 0.0))
    out.append(MetricLine("chad_scr_total_pnl", {}, _finite_or_zero(total_pnl)))
    out.append(MetricLine("chad_scr_win_rate", {}, _finite_or_zero(win_rate)))
    out.append(MetricLine("chad_scr_sharpe_like", {}, _finite_or_zero(sharpe_like)))
    out.append(MetricLine("chad_scr_max_drawdown", {}, _finite_or_zero(max_dd)))

    for st in ("WARMUP", "CAUTIOUS", "CONFIDENT", "PAUSED", "UNKNOWN"):
        out.append(MetricLine("chad_scr_state", {"state": st}, 1.0 if state == st else 0.0))

    return out


# -----------------------------
# Rendering
# -----------------------------

def _render_prometheus(metrics: Iterable[MetricLine]) -> str:
    lines: List[str] = []

    def ht(help_line: str, type_line: str) -> None:
        lines.append(help_line)
        lines.append(type_line)

    ht("# HELP chad_metrics_server_up Metrics server is running (1).", "# TYPE chad_metrics_server_up gauge")
    ht("# HELP chad_metrics_generated_at_unix Generation timestamp (unix).", "# TYPE chad_metrics_generated_at_unix gauge")

    ht("# HELP chad_paper_trades_total Paper trades in raw-paper-rollup set (finite + trusted; NOT SCR-aligned — does NOT consult runtime/quarantine_manifest_*.json; DIVERGES from SCR canonical; for SCR truth use chad_scr_* metrics).", "# TYPE chad_paper_trades_total gauge")
    ht("# HELP chad_paper_win_rate Win rate in raw-paper-rollup set (NOT SCR-aligned; does NOT consult runtime/quarantine_manifest_*.json; DIVERGES from SCR canonical; for SCR truth use chad_scr_win_rate).", "# TYPE chad_paper_win_rate gauge")
    ht("# HELP chad_paper_sharpe_like Mean/Std on raw-paper-rollup pnl series (NOT SCR-aligned; does NOT consult runtime/quarantine_manifest_*.json; DIVERGES from SCR canonical; for SCR truth use chad_scr_sharpe_like).", "# TYPE chad_paper_sharpe_like gauge")
    ht("# HELP chad_paper_pnl_untrusted_count Trades flagged pnl_untrusted (excluded from raw-paper-rollup; this is NOT the same set as SCR excluded_untrusted — SCR additionally drops runtime/quarantine_manifest_*.json entries).", "# TYPE chad_paper_pnl_untrusted_count gauge")

    ht("# HELP chad_scr_effective_trades Trades included for SCR performance metrics.", "# TYPE chad_scr_effective_trades gauge")
    ht("# HELP chad_scr_state One-hot SCR state label.", "# TYPE chad_scr_state gauge")
    ht("# HELP chad_scr_total_pnl Canonical SCR effective total PnL (runtime/scr_state.json::stats.total_pnl; Epoch-2 cutoff; quarantine-manifest filtered; 60d window). Authoritative PnL truth.", "# TYPE chad_scr_total_pnl gauge")
    ht("# HELP chad_scr_win_rate Canonical SCR win rate (runtime/scr_state.json::stats.win_rate; Epoch-2 cutoff; quarantine-manifest filtered; 60d window).", "# TYPE chad_scr_win_rate gauge")
    ht("# HELP chad_scr_sharpe_like Canonical SCR sharpe-like ratio (runtime/scr_state.json::stats.sharpe_like; Epoch-2 cutoff; quarantine-manifest filtered; 60d window).", "# TYPE chad_scr_sharpe_like gauge")
    ht("# HELP chad_scr_max_drawdown Canonical SCR max drawdown (runtime/scr_state.json::stats.max_drawdown; Epoch-2 cutoff; quarantine-manifest filtered; 60d window).", "# TYPE chad_scr_max_drawdown gauge")

    now_dt = _utc_now()
    base = [
        MetricLine("chad_metrics_server_up", {}, 1.0),
        MetricLine("chad_metrics_generated_at_unix", {}, float(now_dt.timestamp())),
    ]

    rendered: List[str] = []
    for m in base:
        rendered.append(m.render())
    for m in metrics:
        rendered.append(m.render())

    rendered = sorted(rendered)
    lines.extend(rendered)
    return "\n".join(lines) + "\n"


def _redis_lines() -> List[MetricLine]:
    """Redis state bus health metrics (best-effort)."""
    out: List[MetricLine] = []
    try:
        from chad.core.state_bus import get_state_bus
        bus = get_state_bus()
        connected = bus.is_connected()
        out.append(MetricLine("chad_redis_connected", {}, 1.0 if connected else 0.0))
        out.append(MetricLine("chad_redis_ping_ms", {}, _finite_or_zero(bus.ping_ms())))
        out.append(MetricLine("chad_redis_memory_used_mb", {}, _finite_or_zero(bus.memory_used_mb())))
        out.append(MetricLine("chad_redis_messages_published", {}, float(bus.messages_published)))
        out.append(MetricLine("chad_redis_subscriber_count", {}, float(bus.subscriber_count)))
    except Exception:
        out.append(MetricLine("chad_redis_connected", {}, 0.0))
    return out


def _var_drawdown_lines() -> List[MetricLine]:
    """Report-only VaR + drawdown gauges sourced from runtime state files (GAP-015A/016A)."""
    out: List[MetricLine] = []

    var_obj = _read_json_file(RUNTIME_VAR_STATE_PATH) or {}
    var_status = str(var_obj.get("status") or "missing").lower()
    var_95 = _coerce_float(var_obj.get("var_95_1day_usd", 0.0), 0.0)
    var_99 = _coerce_float(var_obj.get("var_99_1day_usd", 0.0), 0.0)
    var_pct = _coerce_float(var_obj.get("var_pct_of_equity", 0.0), 0.0)
    var_symbols = _coerce_float(var_obj.get("symbol_count", 0.0), 0.0)
    out.append(MetricLine("chad_var_95_1day_usd", {}, _finite_or_zero(var_95)))
    out.append(MetricLine("chad_var_99_1day_usd", {}, _finite_or_zero(var_99)))
    out.append(MetricLine("chad_var_pct_of_equity", {}, _finite_or_zero(var_pct)))
    out.append(MetricLine("chad_var_symbol_count", {}, _finite_or_zero(var_symbols)))
    out.append(MetricLine("chad_var_status_ok", {}, 1.0 if var_status == "ok" else 0.0))

    dd_obj = _read_json_file(RUNTIME_DRAWDOWN_STATE_PATH) or {}
    dd_status = str(dd_obj.get("status") or "missing").lower()
    dd_pct = _coerce_float(dd_obj.get("drawdown_pct", 0.0), 0.0)
    dd_threshold = _coerce_float(dd_obj.get("halt_threshold_pct", 0.0), 0.0)
    dd_halt = bool(dd_obj.get("halt", False))
    dd_enforce = bool(dd_obj.get("enforcement_active", False))
    out.append(MetricLine("chad_drawdown_pct", {}, _finite_or_zero(dd_pct)))
    out.append(MetricLine("chad_drawdown_halt_threshold_pct", {}, _finite_or_zero(dd_threshold)))
    out.append(MetricLine("chad_drawdown_halt_active", {}, 1.0 if dd_halt else 0.0))
    out.append(MetricLine("chad_drawdown_enforcement_active", {}, 1.0 if dd_enforce else 0.0))
    out.append(MetricLine("chad_drawdown_status_ok", {}, 1.0 if dd_status == "ok" else 0.0))

    return out


def collect_metrics(*, days_back: int, max_trades: int) -> List[MetricLine]:
    out: List[MetricLine] = []
    out.extend(_paper_rollup_metrics(None, days_back=days_back, max_trades=max_trades))
    trades_raw = load_recent_paper_trades(days_back=days_back, max_trades=max_trades, trades_dir=None)
    out.extend(_paper_strategy_lines(trades_raw))
    out.extend(_scr_lines())
    out.extend(_redis_lines())
    out.extend(_var_drawdown_lines())
    return out


# -----------------------------
# HTTP server
# -----------------------------

class MetricsHandler(BaseHTTPRequestHandler):
    server_version = "CHADMetrics/1.0"

    def _send(self, code: int, body: str, content_type: str = "text/plain; version=0.0.4") -> None:
        b = body.encode("utf-8", errors="replace")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self) -> None:  # noqa: N802
        try:
            if self.path == "/health":
                self._send(200, "ok\n", "text/plain")
                return
            if self.path.startswith("/metrics"):
                metrics = collect_metrics(days_back=DEFAULT_DAYS_BACK, max_trades=DEFAULT_MAX_TRADES)
                self._send(200, _render_prometheus(metrics), "text/plain; version=0.0.4")
                return
            self._send(404, "not_found\n", "text/plain")
        except Exception:
            self._send(500, "error\n", "text/plain")

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> int:
    srv = ThreadingHTTPServer((DEFAULT_HOST, DEFAULT_PORT), MetricsHandler)
    try:
        srv.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            srv.server_close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
