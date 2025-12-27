"""
chad/ops/metrics_server.py

CHAD Prometheus Metrics Server (Phase 9)

Goals (production-safe):
- Serve Prometheus metrics on /metrics (default :9620).
- Never crash the process due to malformed ledgers or missing runtime files.
- Expose:
  * Paper trade rollups (RAW vs LEAN counts aligned with SCR behavior)
  * IBKR health/runtime status metrics
  * SCR-effective trade counters (effective_trades + excluded_*)
  * Optional SCR state + sizing factor

Notes:
- "RAW" = all parsed paper records in the rollup window (may include NaN/Inf pnl rows).
- "LEAN" = finite-PnL paper records only (aligns with trade_stats_engine + SCR gating).
"""

from __future__ import annotations

import json
import math
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------
# Constants / Paths
# -----------------------------

ROOT: Path = Path(__file__).resolve().parents[2]
DATA_TRADES: Path = ROOT / "data" / "trades"
RUNTIME_DIR: Path = ROOT / "runtime"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9620

ROLLUP_DAYS_BACK_DEFAULT = 30
ROLLUP_MAX_TRADES_DEFAULT = 5000
CACHE_TTL_SECONDS_DEFAULT = 5.0


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        v = int(str(raw).strip())
        return v if v > 0 else default
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        v = float(str(raw).strip())
        return v if math.isfinite(v) and v > 0 else default
    except Exception:
        return default


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# -----------------------------
# Metric formatting
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


# -----------------------------
# Trade parsing helpers
# -----------------------------

def _safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _coerce_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x
    except Exception:
        return default


def _isfinite(x: float) -> bool:
    return math.isfinite(x)


def _iter_trade_files(days_back: int) -> List[Path]:
    """
    Return newest-first list of trade_history_YYYYMMDD.ndjson files within days_back window.
    We keep it simple: lexicographic dates in filename.
    """
    if not DATA_TRADES.exists():
        return []
    files = sorted(DATA_TRADES.glob("trade_history_*.ndjson"), reverse=True)
    if not files:
        return []
    # Light filtering by date in filename if possible
    cutoff = _utc_now().date()
    keep: List[Path] = []
    for f in files:
        name = f.name
        # trade_history_YYYYMMDD.ndjson
        try:
            ymd = name.split("_")[2].split(".")[0]
            dt = datetime.strptime(ymd, "%Y%m%d").date()
            delta = (cutoff - dt).days
            if 0 <= delta <= max(days_back, 0):
                keep.append(f)
        except Exception:
            # If filename doesn't parse, keep it (defensive)
            keep.append(f)
    return keep


def _load_recent_paper_trades(*, days_back: int, max_trades: int) -> List[Dict[str, Any]]:
    """
    Load paper trades (payload.is_live == False) from newest files first.
    Returns list of payload dicts (not full record wrapper) up to max_trades.
    """
    out: List[Dict[str, Any]] = []
    for f in _iter_trade_files(days_back):
        try:
            lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        # iterate newest-last? NDJSON is append-only; we want most recent entries first
        for line in reversed(lines):
            obj = _safe_json_loads(line)
            if not obj:
                continue
            payload = obj.get("payload")
            if not isinstance(payload, dict):
                continue
            is_live = bool(payload.get("is_live", False))
            if is_live:
                continue
            out.append(payload)
            if len(out) >= max_trades:
                return out
    return out


def _compute_max_drawdown(pnl_series: List[float]) -> float:
    """
    Compute max drawdown over cumulative pnl curve.
    Returns a non-positive number (<= 0). 0 means no drawdown.
    """
    if not pnl_series:
        return 0.0
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnl_series:
        equity += float(p)
        peak = max(peak, equity)
        dd = equity - peak
        if dd < max_dd:
            max_dd = dd
    return float(max_dd)


def _compute_sharpe_like(pnl_series: List[float]) -> float:
    """
    Sharpe-like = mean(pnl)/stdev(pnl). If stdev ~0, return 0.
    """
    n = len(pnl_series)
    if n < 2:
        return 0.0
    mean = float(sum(pnl_series)) / float(n)
    var = 0.0
    for p in pnl_series:
        var += (float(p) - mean) ** 2
    sd = math.sqrt(var / float(n - 1))
    if sd <= 1e-12:
        return 0.0
    val = mean / sd
    return float(val) if math.isfinite(val) else 0.0


# -----------------------------
# SCR helpers (authoritative)
# -----------------------------

def _load_scr_stats() -> Dict[str, Any]:
    """
    Pull authoritative SCR stats from trade_stats_engine.
    """
    try:
        from chad.analytics.trade_stats_engine import load_and_compute  # local import
        return load_and_compute(
            max_trades=5000,
            days_back=60,
            include_paper=True,
            include_live=True,
        )
    except Exception:
        return {}


def _load_scr_state(scr_stats: Dict[str, Any]) -> Tuple[str, float, bool]:
    """
    Evaluate SCR band from stats using shadow_confidence_router.
    Returns (state, sizing_factor, paper_only).
    """
    try:
        from chad.analytics.shadow_confidence_router import evaluate_confidence  # local import
        shadow = evaluate_confidence(scr_stats)
        return str(shadow.state), float(shadow.sizing_factor), bool(shadow.paper_only)
    except Exception:
        return "UNKNOWN", 0.0, True


# -----------------------------
# IBKR runtime helpers
# -----------------------------

def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _ibkr_status_metrics(now: datetime) -> List[MetricLine]:
    """
    Reads runtime/ibkr_status.json if present.
    Expected shape (best-effort):
      ok: bool
      consecutive_failures: int
      last_ok_at: unix ts (float) OR last_ok_iso: str
    """
    lines: List[MetricLine] = []
    st = _read_json_file(RUNTIME_DIR / "ibkr_status.json") or {}

    ok = 1.0 if bool(st.get("ok", False)) else 0.0
    failures = float(_coerce_float(st.get("consecutive_failures", 0), 0.0))

    # Age since last_ok_at
    last_ok_at = st.get("last_ok_at")
    age = -1.0
    try:
        if last_ok_at is not None:
            ts = float(last_ok_at)
            if math.isfinite(ts) and ts > 0:
                age = max(0.0, now.timestamp() - ts)
    except Exception:
        age = -1.0

    lines.append(MetricLine("chad_ibkr_ok", {}, float(ok)))
    lines.append(MetricLine("chad_ibkr_consecutive_failures", {}, float(failures)))
    lines.append(MetricLine("chad_ibkr_last_ok_age_seconds", {}, float(age)))
    return lines


def _ibkr_paper_ledger_enabled() -> float:
    cfg = _read_json_file(RUNTIME_DIR / "ibkr_paper_ledger.json") or {}
    return 1.0 if bool(cfg.get("enabled", False)) else 0.0


# -----------------------------
# Metrics computation (cached)
# -----------------------------

@dataclass(frozen=True)
class MetricsSnapshot:
    generated_at_utc: str
    body: str


class MetricsCache:
    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = float(ttl_seconds)
        self._lock = threading.Lock()
        self._last_ts = 0.0
        self._last: Optional[MetricsSnapshot] = None

    def get_or_compute(self) -> MetricsSnapshot:
        now = time.time()
        with self._lock:
            if self._last is not None and (now - self._last_ts) <= self._ttl:
                return self._last

        snap = _compute_metrics_snapshot()

        with self._lock:
            self._last_ts = time.time()
            self._last = snap
            return snap


def _compute_metrics_snapshot() -> MetricsSnapshot:
    now_dt = _utc_now()

    days_back = _env_int("CHAD_ROLLUP_DAYS_BACK", ROLLUP_DAYS_BACK_DEFAULT)
    max_trades = _env_int("CHAD_ROLLUP_MAX_TRADES", ROLLUP_MAX_TRADES_DEFAULT)

    # Paper trades rollup (paper-only)
    trades_raw = _load_recent_paper_trades(days_back=days_back, max_trades=max_trades)

    # RAW: all parsed paper trades in-window (may include NaN/Inf pnl).
    total_trades_raw = float(len(trades_raw))

    # Extract PnL list (best-effort), keep parallel indexing with trades_raw.
    raw_pnls: List[float] = []
    for t in trades_raw:
        try:
            raw_pnls.append(float(t.get("pnl", 0.0)))
        except Exception:
            raw_pnls.append(float("nan"))

    pnl_series: List[float] = [p for p in raw_pnls if math.isfinite(p)]
    pnl_nonfinite_count = float(len(raw_pnls) - len(pnl_series))

    finite_trades: List[Dict[str, Any]] = [t for (t, p) in zip(trades_raw, raw_pnls) if math.isfinite(p)]

    # LEAN: finite-PnL only (align with SCR).
    total_trades = float(len(pnl_series))
    winners = float(sum(1 for p in pnl_series if p > 0.0))
    losers = float(sum(1 for p in pnl_series if p < 0.0))
    win_rate = float(winners / total_trades) if total_trades > 0 else 0.0

    total_pnl = float(sum(pnl_series)) if pnl_series else 0.0
    sharpe_like = _compute_sharpe_like(pnl_series)
    max_dd = _compute_max_drawdown(pnl_series)

    # Notional and untrusted counts computed on the LEAN set.
    total_notional = 0.0
    untrusted_count = 0.0
    for t in finite_trades:
        try:
            total_notional += float(t.get("notional", 0.0))
        except Exception:
            pass
        tags = t.get("tags") or []
        if isinstance(tags, list) and any(str(x).lower() == "pnl_untrusted" for x in tags):
            untrusted_count += 1.0

    # Build metrics lines
    lines: List[str] = []
    out: List[MetricLine] = []

    # Process-level
    out.append(MetricLine("chad_metrics_server_up", {}, 1.0))
    out.append(MetricLine("chad_metrics_generated_at_unix", {}, float(now_dt.timestamp())))

    # IBKR / runtime
    out.append(MetricLine("chad_ibkr_paper_ledger_enabled", {}, float(_ibkr_paper_ledger_enabled())))
    out.extend(_ibkr_status_metrics(now_dt))

    # Paper rollups
    out.append(MetricLine("chad_paper_trades_total_raw", {}, float(total_trades_raw)))
    out.append(MetricLine("chad_paper_trades_total", {}, float(total_trades)))
    out.append(MetricLine("chad_paper_pnl_nonfinite_count", {}, float(pnl_nonfinite_count)))

    out.append(MetricLine("chad_paper_win_rate", {}, float(win_rate)))
    out.append(MetricLine("chad_paper_total_pnl", {}, float(total_pnl)))
    out.append(MetricLine("chad_paper_sharpe_like", {}, float(sharpe_like)))
    out.append(MetricLine("chad_paper_max_drawdown", {}, float(max_dd)))
    out.append(MetricLine("chad_paper_total_notional", {}, float(total_notional)))
    out.append(MetricLine("chad_paper_pnl_untrusted_count", {}, float(untrusted_count)))
    out.append(MetricLine("chad_paper_winners", {}, float(winners)))
    out.append(MetricLine("chad_paper_losers", {}, float(losers)))

    # SCR-effective trade counters (authoritative; matches /live-gate)
    scr = _load_scr_stats()
    out.append(MetricLine("chad_scr_effective_trades", {}, float(scr.get("effective_trades", 0))))
    out.append(MetricLine("chad_scr_excluded_manual", {}, float(scr.get("excluded_manual", 0))))
    out.append(MetricLine("chad_scr_excluded_untrusted", {}, float(scr.get("excluded_untrusted", 0))))
    out.append(MetricLine("chad_scr_excluded_nonfinite", {}, float(scr.get("excluded_nonfinite", 0))))

    # Optional SCR band/state
    scr_state, scr_sizing, scr_paper_only = _load_scr_state(scr)
    out.append(MetricLine("chad_scr_sizing_factor", {}, float(scr_sizing)))
    out.append(MetricLine("chad_scr_paper_only", {}, 1.0 if scr_paper_only else 0.0))
    # one-hot state series
    for st in ("WARMUP", "CAUTIOUS", "CONFIDENT", "PAUSED", "UNKNOWN"):
        out.append(MetricLine("chad_scr_state", {"state": st}, 1.0 if scr_state.upper() == st else 0.0))

    # HELP/TYPE header (minimal, stable)
    lines.append("# HELP chad_metrics_server_up Metrics server is running (1).")
    lines.append("# TYPE chad_metrics_server_up gauge")
    lines.append("# HELP chad_metrics_generated_at_unix UNIX timestamp when metrics were generated.")
    lines.append("# TYPE chad_metrics_generated_at_unix gauge")

    lines.append("# HELP chad_paper_trades_total_raw Paper trades parsed in rollup window (RAW).")
    lines.append("# TYPE chad_paper_trades_total_raw gauge")
    lines.append("# HELP chad_paper_trades_total Paper trades with finite PnL only (LEAN, aligns with SCR).")
    lines.append("# TYPE chad_paper_trades_total gauge")
    lines.append("# HELP chad_paper_pnl_nonfinite_count Count of paper trades excluded due to non-finite PnL.")
    lines.append("# TYPE chad_paper_pnl_nonfinite_count gauge")

    lines.append("# HELP chad_scr_effective_trades Count of trades included for SCR performance metrics.")
    lines.append("# TYPE chad_scr_effective_trades gauge")

    # Render metric lines
    for m in out:
        # Prometheus expects finite numbers. If anything non-finite sneaks in, clamp to 0.
        v = float(m.value)
        if not math.isfinite(v):
            v = 0.0
        lines.append(MetricLine(m.name, m.labels, v).render())

    body = "\n".join(lines) + "\n"
    return MetricsSnapshot(generated_at_utc=now_dt.isoformat(), body=body)


# -----------------------------
# HTTP server
# -----------------------------

_CACHE = MetricsCache(ttl_seconds=_env_float("CHAD_METRICS_CACHE_TTL_SECONDS", CACHE_TTL_SECONDS_DEFAULT))


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        try:
            if self.path in ("/", "/health"):
                self._send(200, b"ok\n", content_type="text/plain; charset=utf-8")
                return
            if self.path.startswith("/metrics"):
                snap = _CACHE.get_or_compute()
                self._send(200, snap.body.encode("utf-8"), content_type="text/plain; version=0.0.4; charset=utf-8")
                return
            self._send(404, b"not found\n", content_type="text/plain; charset=utf-8")
        except Exception:
            # Never crash the server on handler errors.
            self._send(500, b"internal error\n", content_type="text/plain; charset=utf-8")

    def log_message(self, fmt: str, *args: Any) -> None:
        # quiet: systemd/journald already logs service lifecycle; keep handler silent
        return

    def _send(self, code: int, payload: bytes, *, content_type: str) -> None:
        self.send_response(int(code))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def serve_forever(host: str, port: int) -> None:
    srv = ThreadingHTTPServer((host, int(port)), _Handler)

    stop = threading.Event()

    def _sig_handler(signum: int, frame: Any) -> None:  # noqa: ARG001
        stop.set()
        try:
            srv.shutdown()
        except Exception:
            pass

    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT, _sig_handler)

    try:
        srv.serve_forever(poll_interval=0.25)
    finally:
        try:
            srv.server_close()
        except Exception:
            pass


def main(argv: Optional[List[str]] = None) -> int:
    host = os.environ.get("CHAD_METRICS_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST
    port = _env_int("CHAD_METRICS_PORT", DEFAULT_PORT)

    # Allow simple CLI override without adding argparse dependency here.
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) == 2:
        host = str(argv[0])
        try:
            port = int(argv[1])
        except Exception:
            port = DEFAULT_PORT

    serve_forever(host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
