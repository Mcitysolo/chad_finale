#!/usr/bin/env python3
"""
chad/ops/metrics_server.py

CHAD Prometheus Metrics Server (Phase 9A)

Production goals (non-negotiable)
- Serve Prometheus metrics on /metrics (default :9620).
- Serve a fast health endpoint on /healthz (200 OK) for uptime probes.
- Read-only observability: NEVER places orders, NEVER modifies trading config/state.
- Fail-safe: /metrics must keep working even if runtime JSON is missing/corrupt,
  ledgers contain bad rows (NaN/Inf), or optional imports fail.
- Deterministic output: stable metric names + labels + HELP/TYPE lines.

Endpoints
- /metrics   Prometheus text format (version=0.0.4)
- /healthz   200 OK "ok\n" (liveness)
- /health    alias of /healthz
- /-/ready   200 if metrics can be generated now, else 503 (best-effort readiness)

Config (env)
- CHAD_METRICS_HOST (default 0.0.0.0)
- CHAD_METRICS_PORT (default 9620)
- CHAD_METRICS_CACHE_TTL_SECONDS (default 5.0)
- CHAD_ROLLUP_DAYS_BACK (default 30)
- CHAD_ROLLUP_MAX_TRADES (default 5000)

Data sources (best-effort)
- data/trades/trade_history_YYYYMMDD.ndjson  (paper trade outcomes)
- runtime/ibkr_status.json                   (IBKR watchdog)
- runtime/ibkr_paper_ledger.json             (paper ledger enabled flag)
- runtime/scr_stats.json                     (SCR trade stats and exclusions)

This file is intentionally stdlib-only.
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


# -----------------------------
# Env helpers
# -----------------------------


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


def _finite_or_zero(x: float) -> float:
    return x if math.isfinite(x) else 0.0


# -----------------------------
# JSON / parsing helpers
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
        x = float(v)
        return x
    except Exception:
        return default


# -----------------------------
# Trade rollups (paper)
# -----------------------------


def _iter_trade_files(days_back: int) -> List[Path]:
    """
    Return newest-first list of trade_history_YYYYMMDD.ndjson files within days_back window.
    """
    if not DATA_TRADES.exists():
        return []
    files = sorted(DATA_TRADES.glob("trade_history_*.ndjson"), reverse=True)
    if not files:
        return []
    cutoff = _utc_now().date()
    keep: List[Path] = []
    for f in files:
        name = f.name
        try:
            ymd = name.split("_", 2)[2].split(".", 1)[0]
            dt = datetime.strptime(ymd, "%Y%m%d").date()
            delta = (cutoff - dt).days
            if 0 <= delta <= max(days_back, 0):
                keep.append(f)
        except Exception:
            # Defensive: if filename doesn't parse, keep it.
            keep.append(f)
    return keep


def _load_recent_paper_trades(*, days_back: int, max_trades: int) -> List[Dict[str, Any]]:
    """
    Load paper trade payloads from newest files first. Returns list of payload dicts.
    This intentionally accepts imperfect records and filters later.
    """
    out: List[Dict[str, Any]] = []
    for f in _iter_trade_files(days_back):
        try:
            lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        # We read from top to bottom; order doesn't matter for rollup totals.
        for line in lines:
            rec = _safe_json_loads(line)
            if not rec:
                continue
            payload = rec.get("payload") or {}
            if not isinstance(payload, dict):
                continue
            # Paper-only: best-effort check. Many CHAD logs use is_live or live flags.
            is_live = payload.get("is_live")
            if is_live is True:
                continue
            out.append(payload)
            if len(out) >= max_trades:
                return out
    return out


def _compute_sharpe_like(pnls: List[float]) -> float:
    # Simple, stable: mean / stddev (population). No annualization; this is "sharpe-like".
    if not pnls:
        return 0.0
    mu = sum(pnls) / float(len(pnls))
    var = sum((x - mu) ** 2 for x in pnls) / float(len(pnls))
    sd = math.sqrt(var) if var > 0.0 else 0.0
    if sd <= 0.0 or not math.isfinite(sd):
        return 0.0
    return float(mu / sd)


def _compute_max_drawdown(pnls: List[float]) -> float:
    # Max peak-to-trough on cumulative pnl.
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


def _paper_rollup_metrics(*, days_back: int, max_trades: int) -> List[MetricLine]:
    now_dt = _utc_now()
    trades_raw = _load_recent_paper_trades(days_back=days_back, max_trades=max_trades)

    # RAW: all parsed records (may include NaN/Inf pnl rows).
    total_trades_raw = float(len(trades_raw))

    # Extract PnL list (best-effort); keep parallel indexing with trades_raw.
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
            n = float(t.get("notional", 0.0))
        except Exception:
            n = 0.0
        if math.isfinite(n):
            total_notional += n

        # "pnl_untrusted" can be in tags or boolean flag.
        tags = t.get("tags") or []
        tagged = isinstance(tags, list) and any(str(x).lower() == "pnl_untrusted" for x in tags)
        flagged = bool(t.get("pnl_untrusted", False))
        if tagged or flagged:
            untrusted_count += 1.0

    lines: List[MetricLine] = []
    lines.append(MetricLine("chad_paper_trades_total_raw", {}, _finite_or_zero(total_trades_raw)))
    lines.append(MetricLine("chad_paper_trades_total", {}, _finite_or_zero(total_trades)))
    lines.append(MetricLine("chad_paper_pnl_nonfinite_count", {}, _finite_or_zero(pnl_nonfinite_count)))

    lines.append(MetricLine("chad_paper_win_rate", {}, _finite_or_zero(win_rate)))
    lines.append(MetricLine("chad_paper_total_pnl", {}, _finite_or_zero(total_pnl)))
    lines.append(MetricLine("chad_paper_sharpe_like", {}, _finite_or_zero(sharpe_like)))
    lines.append(MetricLine("chad_paper_max_drawdown", {}, _finite_or_zero(max_dd)))
    lines.append(MetricLine("chad_paper_total_notional", {}, _finite_or_zero(total_notional)))
    lines.append(MetricLine("chad_paper_pnl_untrusted_count", {}, _finite_or_zero(untrusted_count)))
    lines.append(MetricLine("chad_paper_winners", {}, _finite_or_zero(winners)))
    lines.append(MetricLine("chad_paper_losers", {}, _finite_or_zero(losers)))

    # Per-strategy rollups (LEAN set)
    by_strategy_total: Dict[str, int] = {}
    by_strategy_pnls: Dict[str, List[float]] = {}
    for t, p in zip(trades_raw, raw_pnls):
        if not math.isfinite(p):
            continue
        strat = str(t.get("strategy", "")).strip().lower() or "unknown"
        by_strategy_total[strat] = by_strategy_total.get(strat, 0) + 1
        by_strategy_pnls.setdefault(strat, []).append(float(p))

    for strat in sorted(by_strategy_total.keys()):
        pnls = by_strategy_pnls.get(strat, [])
        st_total = float(by_strategy_total[strat])
        st_total_pnl = float(sum(pnls)) if pnls else 0.0
        st_winners = float(sum(1 for x in pnls if x > 0.0))
        st_win_rate = float(st_winners / st_total) if st_total > 0 else 0.0
        lines.append(MetricLine("chad_paper_strategy_trades_total", {"strategy": strat}, _finite_or_zero(st_total)))
        lines.append(MetricLine("chad_paper_strategy_total_pnl", {"strategy": strat}, _finite_or_zero(st_total_pnl)))
        lines.append(MetricLine("chad_paper_strategy_win_rate", {"strategy": strat}, _finite_or_zero(st_win_rate)))

    return lines


# -----------------------------
# IBKR runtime helpers
# -----------------------------


def _ibkr_status_metrics(now: datetime) -> List[MetricLine]:
    """
    Reads runtime/ibkr_status.json if present.
    Expected shape (best-effort):
      ok: bool
      consecutive_failures: int
      last_ok_at: unix ts (float)
    """
    lines: List[MetricLine] = []
    st = _read_json_file(RUNTIME_DIR / "ibkr_status.json") or {}

    ok = 1.0 if bool(st.get("ok", False)) else 0.0
    failures = float(_coerce_float(st.get("consecutive_failures", 0), 0.0))

    last_ok_at = st.get("last_ok_at")
    age = -1.0
    try:
        if last_ok_at is not None:
            ts = float(last_ok_at)
            if math.isfinite(ts) and ts > 0:
                age = max(0.0, now.timestamp() - ts)
    except Exception:
        age = -1.0

    lines.append(MetricLine("chad_ibkr_ok", {}, _finite_or_zero(ok)))
    lines.append(MetricLine("chad_ibkr_consecutive_failures", {}, _finite_or_zero(failures)))
    lines.append(MetricLine("chad_ibkr_last_ok_age_seconds", {}, _finite_or_zero(age)))
    return lines


def _ibkr_paper_ledger_enabled() -> float:
    cfg = _read_json_file(RUNTIME_DIR / "ibkr_paper_ledger.json") or {}
    return 1.0 if bool(cfg.get("enabled", False)) else 0.0


# -----------------------------
# SCR helpers (best-effort)
# -----------------------------


def _load_scr_stats() -> Dict[str, Any]:
    # Prefer runtime/scr_stats.json if available, else fall back to empty dict.
    return _read_json_file(RUNTIME_DIR / "scr_stats.json") or {}


def _load_scr_state(scr_stats: Dict[str, Any]) -> Tuple[str, float, bool]:
    """
    Evaluate SCR band from stats using shadow_confidence_router (if available).
    Returns (state, sizing_factor, paper_only). Best-effort; never raises.
    """
    try:
        from chad.analytics.shadow_confidence_router import evaluate_confidence  # local import

        shadow = evaluate_confidence(scr_stats)
        return str(shadow.state), float(shadow.sizing_factor), bool(shadow.paper_only)
    except Exception:
        return "UNKNOWN", 0.0, True


# -----------------------------
# Metrics snapshot + caching
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


_CACHE = MetricsCache(ttl_seconds=_env_float("CHAD_METRICS_CACHE_TTL_SECONDS", CACHE_TTL_SECONDS_DEFAULT))


def _render_prometheus(metrics: Iterable[MetricLine]) -> str:
    # Minimal stable HELP/TYPE lines for core metrics, plus dynamic per-strategy.
    lines: List[str] = []

    def ht(help_line: str, typ: str) -> None:
        lines.append(help_line)
        lines.append(typ)

    ht("# HELP chad_metrics_server_up Metrics server is running (1).", "# TYPE chad_metrics_server_up gauge")
    ht("# HELP chad_metrics_generated_at_unix UNIX timestamp when metrics were generated.", "# TYPE chad_metrics_generated_at_unix gauge")

    ht("# HELP chad_ibkr_paper_ledger_enabled 1 if ibkr_paper_ledger.json enabled is true.", "# TYPE chad_ibkr_paper_ledger_enabled gauge")
    ht("# HELP chad_ibkr_ok 1 if IBKR watchdog ok==true.", "# TYPE chad_ibkr_ok gauge")
    ht("# HELP chad_ibkr_consecutive_failures Consecutive IBKR failures.", "# TYPE chad_ibkr_consecutive_failures gauge")
    ht("# HELP chad_ibkr_last_ok_age_seconds Seconds since last_ok_at (or -1 if unknown).", "# TYPE chad_ibkr_last_ok_age_seconds gauge")

    ht("# HELP chad_paper_trades_total_raw Paper trades parsed in rollup window (RAW).", "# TYPE chad_paper_trades_total_raw gauge")
    ht("# HELP chad_paper_trades_total Paper trades with finite PnL only (LEAN, aligns with SCR).", "# TYPE chad_paper_trades_total gauge")
    ht("# HELP chad_paper_pnl_nonfinite_count Count of paper trades excluded due to non-finite PnL.", "# TYPE chad_paper_pnl_nonfinite_count gauge")

    ht("# HELP chad_scr_effective_trades Count of trades included for SCR performance metrics.", "# TYPE chad_scr_effective_trades gauge")
    ht("# HELP chad_scr_excluded_manual Count of trades excluded manually by operator.", "# TYPE chad_scr_excluded_manual gauge")
    ht("# HELP chad_scr_excluded_untrusted Count of trades excluded due to untrusted PnL tags.", "# TYPE chad_scr_excluded_untrusted gauge")
    ht("# HELP chad_scr_excluded_nonfinite Count of trades excluded due to non-finite PnL.", "# TYPE chad_scr_excluded_nonfinite gauge")
    ht("# HELP chad_scr_sizing_factor SCR sizing factor (0..1).", "# TYPE chad_scr_sizing_factor gauge")
    ht("# HELP chad_scr_paper_only 1 if SCR is paper-only.", "# TYPE chad_scr_paper_only gauge")
    ht("# HELP chad_scr_state One-hot SCR state label.", "# TYPE chad_scr_state gauge")

    # Strategy rollups are dynamic; HELP/TYPE lines once.
    ht("# HELP chad_paper_strategy_trades_total Per-strategy paper trades (LEAN).", "# TYPE chad_paper_strategy_trades_total gauge")
    ht("# HELP chad_paper_strategy_total_pnl Per-strategy total PnL (LEAN).", "# TYPE chad_paper_strategy_total_pnl gauge")
    ht("# HELP chad_paper_strategy_win_rate Per-strategy win rate (LEAN).", "# TYPE chad_paper_strategy_win_rate gauge")

    # Render metric lines
    for m in metrics:
        v = _finite_or_zero(float(m.value))
        lines.append(MetricLine(m.name, m.labels, v).render())

    return "\n".join(lines) + "\n"


def _compute_metrics_snapshot() -> MetricsSnapshot:
    now_dt = _utc_now()

    days_back = _env_int("CHAD_ROLLUP_DAYS_BACK", ROLLUP_DAYS_BACK_DEFAULT)
    max_trades = _env_int("CHAD_ROLLUP_MAX_TRADES", ROLLUP_MAX_TRADES_DEFAULT)

    out: List[MetricLine] = []
    out.append(MetricLine("chad_metrics_server_up", {}, 1.0))
    out.append(MetricLine("chad_metrics_generated_at_unix", {}, float(now_dt.timestamp())))

    out.append(MetricLine("chad_ibkr_paper_ledger_enabled", {}, float(_ibkr_paper_ledger_enabled())))
    out.extend(_ibkr_status_metrics(now_dt))

    out.extend(_paper_rollup_metrics(days_back=days_back, max_trades=max_trades))

    # SCR trade counters + state (best-effort)
    scr = _load_scr_stats()
    out.append(MetricLine("chad_scr_effective_trades", {}, float(_coerce_float(scr.get("effective_trades", 0), 0.0))))
    out.append(MetricLine("chad_scr_excluded_manual", {}, float(_coerce_float(scr.get("excluded_manual", 0), 0.0))))
    out.append(MetricLine("chad_scr_excluded_untrusted", {}, float(_coerce_float(scr.get("excluded_untrusted", 0), 0.0))))
    out.append(MetricLine("chad_scr_excluded_nonfinite", {}, float(_coerce_float(scr.get("excluded_nonfinite", 0), 0.0))))

    scr_state, scr_sizing, scr_paper_only = _load_scr_state(scr)
    out.append(MetricLine("chad_scr_sizing_factor", {}, _finite_or_zero(float(scr_sizing))))
    out.append(MetricLine("chad_scr_paper_only", {}, 1.0 if bool(scr_paper_only) else 0.0))
    for st in ("WARMUP", "CAUTIOUS", "CONFIDENT", "PAUSED", "UNKNOWN"):
        out.append(MetricLine("chad_scr_state", {"state": st}, 1.0 if scr_state.upper() == st else 0.0))

    body = _render_prometheus(out)
    return MetricsSnapshot(generated_at_utc=now_dt.isoformat(), body=body)


# -----------------------------
# HTTP server
# -----------------------------


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        try:
            # Liveness
            if self.path == "/healthz" or self.path.startswith("/healthz?") or self.path == "/health" or self.path.startswith("/health?") or self.path == "/":
                self._send(200, b"ok\n", content_type="text/plain; charset=utf-8")
                return

            # Readiness (best-effort): can we compute metrics right now?
            if self.path == "/-/ready" or self.path.startswith("/-/ready?"):
                try:
                    _ = _CACHE.get_or_compute()
                    self._send(200, b"ready\n", content_type="text/plain; charset=utf-8")
                except Exception:
                    self._send(503, b"not ready\n", content_type="text/plain; charset=utf-8")
                return

            # Metrics
            if self.path.startswith("/metrics"):
                snap = _CACHE.get_or_compute()
                self._send(
                    200,
                    snap.body.encode("utf-8"),
                    content_type="text/plain; version=0.0.4; charset=utf-8",
                )
                return

            self._send(404, b"not found\n", content_type="text/plain; charset=utf-8")
        except Exception:
            # Never crash the server on handler errors.
            try:
                self._send(500, b"internal error\n", content_type="text/plain; charset=utf-8")
            except Exception:
                pass

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
