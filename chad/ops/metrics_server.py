#!/usr/bin/env python3
"""
CHAD Metrics Server (Phase 9+)

Exposes Prometheus-compatible metrics on /metrics (default :9620).

Non-negotiable guarantees
-------------------------
- Read-only observability: NEVER places orders, NEVER modifies trading config/state.
- Fail-safe: /metrics must keep working even if:
  * runtime JSON is missing/corrupt
  * trade ledgers contain bad rows (NaN/inf)
  * systemd queries fail
- Deterministic output: stable metric names, labels, and HELP/TYPE blocks.
- Cross-environment: works under systemd, Docker, or CLI, regardless of cwd.

Key features
------------
1) Systemd unit health metrics (active/failed).
2) Runtime artifact freshness metrics.
3) Portfolio snapshot + caps consistency metrics.
4) IBKR watchdog summary (from runtime/ibkr_status.json).
5) Trade ledger counts (active/quarantine).
6) Paper performance rollups (paper-only) with robust NaN/inf handling:
   - Emits totals, win rate, total/avg pnl (finite-only), sharpe-like, max drawdown,
     total notional, untrusted count, nonfinite pnl count, last trade age.
   - Per-strategy totals/win-rate.

Config via environment variables
--------------------------------
- CHAD_METRICS_HOST (default 0.0.0.0)
- CHAD_METRICS_PORT (default 9620)
- CHAD_ROOT_DIR      (default: /home/ubuntu/CHAD FINALE if exists, else cwd)
- CHAD_REPO_DIR      (default: /home/ubuntu/chad_finale if exists, else cwd)
- CHAD_TRADES_DIR    (optional override for trade history directory)
- CHAD_ROLLUP_MAX_TRADES (default 500)
- CHAD_ROLLUP_DAYS_BACK  (default 30)
- CHAD_ROLLUP_TTL_SECONDS (default 15) cache TTL for rollup computation

Endpoints
---------
- /metrics : Prometheus text format
- /healthz : 200 if server loop is alive
- /-/ready : 200 if /metrics can be generated (best-effort)

Run
---
python -m chad.ops.metrics_server
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from statistics import mean, pstdev
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

try:
    from http.server import ThreadingHTTPServer as _HTTPServer
except Exception:  # pragma: no cover
    from http.server import HTTPServer as _HTTPServer  # type: ignore[assignment]


# -------------------------
# Constants / configuration
# -------------------------

DEFAULT_ROOT_CANDIDATES = [
    Path("/home/ubuntu/CHAD FINALE"),
    Path("/home/ubuntu/CHAD_FINALE"),
]
DEFAULT_REPO_CANDIDATES = [
    Path("/home/ubuntu/chad_finale"),
    Path("/home/ubuntu/CHAD FINALE"),
]

LISTEN_HOST_DEFAULT = "0.0.0.0"
LISTEN_PORT_DEFAULT = 9620

ROLLUP_MAX_TRADES_DEFAULT = 500
ROLLUP_DAYS_BACK_DEFAULT = 30
ROLLUP_TTL_SECONDS_DEFAULT = 15.0


# -------------------------
# Metric line rendering
# -------------------------

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


def _escape_label(v: str) -> str:
    return v.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


# -------------------------
# Env/path resolution
# -------------------------

def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _resolve_root_dir() -> Path:
    raw = os.environ.get("CHAD_ROOT_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    for c in DEFAULT_ROOT_CANDIDATES:
        if c.exists():
            return c
    return Path.cwd().resolve()


def _resolve_repo_dir() -> Path:
    raw = os.environ.get("CHAD_REPO_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    for c in DEFAULT_REPO_CANDIDATES:
        if c.exists():
            return c
    return Path.cwd().resolve()


def _resolve_trades_dir(root_dir: Path, repo_dir: Path) -> Path:
    raw = os.environ.get("CHAD_TRADES_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()

    # Prefer repo-relative (matches logger default Path("data")/"trades" under repo cwd).
    repo_trades = repo_dir / "data" / "trades"
    if repo_trades.exists():
        return repo_trades

    root_trades = root_dir / "data" / "trades"
    if root_trades.exists():
        return root_trades

    # Last resort: relative to current working dir
    return (Path.cwd() / "data" / "trades").resolve()


# -------------------------
# IO helpers (fail-safe)
# -------------------------

def _now_epoch() -> float:
    return time.time()


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _file_age_seconds(path: Path, now: float) -> Optional[float]:
    try:
        st = path.stat()
        return float(max(0.0, now - st.st_mtime))
    except Exception:
        return None


def _run(cmd: Sequence[str], timeout_s: float = 2.0) -> Tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        out = (p.stdout or "").strip()
        err = (p.stderr or "").strip()
        if p.returncode != 0 and err:
            return p.returncode, f"{out}\n{err}".strip()
        return p.returncode, out
    except Exception as e:
        return 255, str(e)


def _systemd_is_active(unit: str) -> Optional[bool]:
    rc, out = _run(["systemctl", "is-active", unit])
    if rc != 0 and out == "":
        return None
    return out.strip() == "active"


def _systemd_is_failed(unit: str) -> Optional[bool]:
    rc, out = _run(["systemctl", "is-failed", unit])
    if rc != 0 and out == "":
        return None
    return out.strip() == "failed"


# -------------------------
# Trade ledger parsing/rollup
# -------------------------

def _parse_iso_utc(ts: str) -> Optional[datetime]:
    s = (ts or "").strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _iter_trade_files(trades_dir: Path, days_back: int) -> List[Path]:
    # Prefer date-pattern files if present, else fall back to glob.
    now = datetime.now(timezone.utc)
    candidates: List[Path] = []
    for i in range(days_back + 1):
        day = now - timedelta(days=i)
        stamp = day.strftime("%Y%m%d")
        p = trades_dir / f"trade_history_{stamp}.ndjson"
        if p.exists():
            candidates.append(p)

    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates

    # Fallback: any matching file in dir
    try:
        gl = list(trades_dir.glob("trade_history_*.ndjson"))
        gl.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return gl
    except Exception:
        return []


def _normalize_trade_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Normalize a trade record line (enveloped) into a dict used for rollups.
    Mirrors trade_stats_engine exclusion semantics.
    """
    payload = record.get("payload", {}) if isinstance(record.get("payload", {}), dict) else {}
    tags = payload.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    tags_norm = [str(x).strip().lower() for x in tags]

    extra = payload.get("extra") or {}
    source = ""
    if isinstance(extra, dict):
        source = str(extra.get("source", "")).strip().lower()

    # Exclusions (must match safety intent)
    if "paper_sim" in tags_norm:
        return None
    if ("warmup" in tags_norm) and (source == "ibkr_execution_runner"):
        return None

    try:
        strategy = str(payload["strategy"])
        symbol = str(payload["symbol"])
        pnl = float(payload["pnl"])
        notional = float(payload["notional"])
        is_live = bool(payload.get("is_live", False))
        exit_time = _parse_iso_utc(str(payload.get("exit_time_utc", ""))) or datetime.now(timezone.utc)
    except Exception:
        return None

    pnl_untrusted = False
    if "pnl_untrusted" in tags_norm:
        pnl_untrusted = True
    if isinstance(extra, dict) and bool(extra.get("pnl_untrusted", False)):
        pnl_untrusted = True

    return {
        "strategy": strategy,
        "symbol": symbol,
        "pnl": pnl,
        "notional": notional,
        "is_live": is_live,
        "exit_time": exit_time,
        "pnl_untrusted": pnl_untrusted,
    }


def _load_recent_paper_trades(trades_dir: Path, *, max_trades: int, days_back: int) -> List[Dict[str, Any]]:
    trades: List[Dict[str, Any]] = []
    if not trades_dir.exists():
        return trades

    files = _iter_trade_files(trades_dir, days_back=days_back)
    for f in files:
        try:
            for line in f.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except Exception:
                    continue
                norm = _normalize_trade_record(rec)
                if norm is None:
                    continue
                if bool(norm.get("is_live", False)):
                    continue  # paper-only rollup
                trades.append(norm)
        except Exception:
            continue

    trades.sort(key=lambda t: t.get("exit_time") or datetime.now(timezone.utc))
    if len(trades) > max_trades:
        trades = trades[-max_trades:]
    return trades


def _compute_max_drawdown(pnl_series: List[float]) -> float:
    clean = [float(x) for x in pnl_series if isinstance(x, (int, float)) and math.isfinite(float(x))]
    if not clean:
        return 0.0
    equity: List[float] = []
    total = 0.0
    for p in clean:
        total += p
        equity.append(total)
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = v - peak
        if dd < max_dd:
            max_dd = dd
    return float(max_dd)


def _compute_sharpe_like(pnl_series: List[float]) -> float:
    clean = [float(x) for x in pnl_series if isinstance(x, (int, float)) and math.isfinite(float(x))]
    if len(clean) < 2:
        return 0.0
    mu = mean(clean)
    sigma = pstdev(clean)
    if sigma <= 0:
        return 0.0
    return float(mu / sigma)


@dataclass
class RollupSnapshot:
    computed_at_epoch: float
    metrics: List[MetricLine]


class RollupCache:
    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = max(1.0, float(ttl_seconds))
        self._lock = Lock()
        self._snap: Optional[RollupSnapshot] = None

    def get_or_compute(self, compute_fn) -> List[MetricLine]:
        now = _now_epoch()
        with self._lock:
            if self._snap and (now - self._snap.computed_at_epoch) <= self._ttl:
                return list(self._snap.metrics)
        metrics = compute_fn()
        with self._lock:
            self._snap = RollupSnapshot(computed_at_epoch=now, metrics=list(metrics))
        return list(metrics)


# -------------------------
# Collector
# -------------------------

def _iter_units() -> List[str]:
    return [
        "chad-metrics.service",
        "chad-paper-shadow-runner.timer",
        "chad-paper-shadow-runner.service",
        "chad-ibkr-paper-ledger-watcher.timer",
        "chad-ibkr-paper-ledger-watcher.service",
        "chad-ibkr-health.timer",
        "chad-ibkr-watchdog.timer",
        "chad-ibkr-collector.timer",
        "chad-ibgateway.service",
    ]


def _iter_runtime_files(runtime_dir: Path) -> List[Path]:
    return [
        runtime_dir / "portfolio_snapshot.json",
        runtime_dir / "dynamic_caps.json",
        runtime_dir / "ibkr_status.json",
        runtime_dir / "paper_shadow.json",
        runtime_dir / "ibkr_paper_ledger.json",
    ]


def _paper_rollup_metrics(trades_dir: Path) -> List[MetricLine]:
    now_dt = datetime.now(timezone.utc)
    max_trades = _env_int("CHAD_ROLLUP_MAX_TRADES", ROLLUP_MAX_TRADES_DEFAULT)
    days_back = _env_int("CHAD_ROLLUP_DAYS_BACK", ROLLUP_DAYS_BACK_DEFAULT)

    trades = _load_recent_paper_trades(trades_dir, max_trades=max_trades, days_back=days_back)

    total_trades = len(trades)
    winners = sum(1 for t in trades if float(t["pnl"]) > 0)
    losers = sum(1 for t in trades if float(t["pnl"]) < 0)

    # Sanitize PnL for sums/means/sharpe/dd
    raw_pnls = [float(t["pnl"]) for t in trades]
    pnl_series = [p for p in raw_pnls if math.isfinite(p)]
    pnl_nonfinite_count = float(len(raw_pnls) - len(pnl_series))

    total_pnl = float(sum(pnl_series)) if pnl_series else 0.0
    avg_pnl = float(total_pnl / len(pnl_series)) if pnl_series else 0.0
    win_rate = float(winners / total_trades) if total_trades > 0 else 0.0

    sharpe_like = _compute_sharpe_like(pnl_series)
    max_dd = _compute_max_drawdown(pnl_series)

    total_notional = float(sum(float(t["notional"]) for t in trades)) if trades else 0.0
    pnl_untrusted_count = float(sum(1 for t in trades if bool(t.get("pnl_untrusted", False))))

    last_age = 0.0
    if trades:
        last_exit = trades[-1].get("exit_time")
        if isinstance(last_exit, datetime):
            last_age = float(max(0.0, (now_dt - last_exit).total_seconds()))

    lines: List[MetricLine] = []
    lines.append(MetricLine("chad_paper_trades_total", {}, float(total_trades)))
    lines.append(MetricLine("chad_paper_trades_winners", {}, float(winners)))
    lines.append(MetricLine("chad_paper_trades_losers", {}, float(losers)))
    lines.append(MetricLine("chad_paper_win_rate", {}, float(win_rate)))
    lines.append(MetricLine("chad_paper_total_pnl", {}, float(total_pnl)))
    lines.append(MetricLine("chad_paper_avg_pnl", {}, float(avg_pnl)))
    lines.append(MetricLine("chad_paper_sharpe_like", {}, float(sharpe_like)))
    lines.append(MetricLine("chad_paper_max_drawdown", {}, float(max_dd)))
    lines.append(MetricLine("chad_paper_total_notional", {}, float(total_notional)))
    lines.append(MetricLine("chad_paper_pnl_untrusted_count", {}, float(pnl_untrusted_count)))
    lines.append(MetricLine("chad_paper_pnl_nonfinite_count", {}, float(pnl_nonfinite_count)))
    lines.append(MetricLine("chad_paper_last_trade_age_seconds", {}, float(last_age)))

    # Per-strategy (finite-only sums)
    by_strat: Dict[str, List[Dict[str, Any]]] = {}
    for t in trades:
        by_strat.setdefault(str(t["strategy"]), []).append(t)

    for strat, items in sorted(by_strat.items(), key=lambda kv: kv[0].lower()):
        s_total = len(items)
        s_raw = [float(x["pnl"]) for x in items]
        s_clean = [p for p in s_raw if math.isfinite(p)]
        s_total_pnl = float(sum(s_clean)) if s_clean else 0.0
        s_winners = sum(1 for x in items if float(x["pnl"]) > 0)
        s_win_rate = float(s_winners / s_total) if s_total > 0 else 0.0
        lines.append(MetricLine("chad_paper_strategy_trades_total", {"strategy": strat}, float(s_total)))
        lines.append(MetricLine("chad_paper_strategy_total_pnl", {"strategy": strat}, float(s_total_pnl)))
        lines.append(MetricLine("chad_paper_strategy_win_rate", {"strategy": strat}, float(s_win_rate)))

    return lines


def _collect_metrics(root_dir: Path, repo_dir: Path, trades_dir: Path, rollup_cache: RollupCache) -> List[MetricLine]:
    runtime_dir = root_dir / "runtime"
    lines: List[MetricLine] = []
    now = _now_epoch()

    # systemd unit health
    for unit in _iter_units():
        active = _systemd_is_active(unit)
        failed = _systemd_is_failed(unit)
        if active is not None:
            lines.append(MetricLine("chad_systemd_unit_active", {"unit": unit}, 1.0 if active else 0.0))
        if failed is not None:
            lines.append(MetricLine("chad_systemd_unit_failed", {"unit": unit}, 1.0 if failed else 0.0))

    # runtime file ages
    for p in _iter_runtime_files(runtime_dir):
        age = _file_age_seconds(p, now)
        if age is not None:
            lines.append(MetricLine("chad_runtime_file_age_seconds", {"file": str(p)}, float(age)))

    # portfolio totals & caps
    snap = _safe_read_json(runtime_dir / "portfolio_snapshot.json") or {}
    caps = _safe_read_json(runtime_dir / "dynamic_caps.json") or {}
    ibkr_eq = float(snap.get("ibkr_equity", 0.0) or 0.0)
    cb_eq = float(snap.get("coinbase_equity", 0.0) or 0.0)
    kr_eq = float(snap.get("kraken_equity", 0.0) or 0.0)
    total_eq = ibkr_eq + cb_eq + kr_eq
    caps_total = float(caps.get("total_equity", 0.0) or 0.0) if caps else 0.0
    match = 1.0 if abs(total_eq - caps_total) < 1e-6 else 0.0
    lines.append(MetricLine("chad_portfolio_total_equity", {}, float(total_eq)))
    lines.append(MetricLine("chad_dynamic_caps_total_equity", {}, float(caps_total)))
    lines.append(MetricLine("chad_snapshot_caps_match", {}, float(match)))

    # paper lane flags
    ps = _safe_read_json(runtime_dir / "paper_shadow.json") or {}
    lines.append(MetricLine("chad_paper_shadow_enabled", {}, 1.0 if bool(ps.get("enabled", False)) else 0.0))
    pl = _safe_read_json(runtime_dir / "ibkr_paper_ledger.json") or {}
    lines.append(MetricLine("chad_ibkr_paper_ledger_enabled", {}, 1.0 if bool(pl.get("enabled", False)) else 0.0))

    # trade ledgers count
    active_count = 0
    quarantine_count = 0
    try:
        if trades_dir.exists():
            active_count = len(list(trades_dir.glob("trade_history_*.ndjson")))
            q = trades_dir / "_quarantine"
            if q.exists():
                quarantine_count = len(list(q.glob("trade_history_*.ndjson")))
    except Exception:
        pass
    lines.append(MetricLine("chad_trade_ledgers_active_count", {}, float(active_count)))
    lines.append(MetricLine("chad_trade_ledgers_quarantine_count", {}, float(quarantine_count)))

    # IBKR watchdog summary
    ibs = _safe_read_json(runtime_dir / "ibkr_status.json") or {}
    ok = 1.0 if bool(ibs.get("ok", False)) else 0.0
    failures = float(ibs.get("consecutive_failures", 0) or 0)
    last_ok_at = float(ibs.get("last_ok_at", 0) or 0)
    age_ok = float(max(0.0, now - last_ok_at)) if last_ok_at > 0 else float("inf")
    lines.append(MetricLine("chad_ibkr_ok", {}, ok))
    lines.append(MetricLine("chad_ibkr_consecutive_failures", {}, failures))
    # "inf" is valid in Prometheus, but keep it finite if unknown
    lines.append(MetricLine("chad_ibkr_last_ok_age_seconds", {}, age_ok if math.isfinite(age_ok) else 9.9e15))

    # rollups (cached)
    def compute_rollup():
        return _paper_rollup_metrics(trades_dir)

    try:
        lines.extend(rollup_cache.get_or_compute(compute_rollup))
    except Exception:
        # never break /metrics
        pass

    # server up
    lines.append(MetricLine("chad_metrics_server_up", {}, 1.0))
    return lines


def _render_prometheus(lines: Iterable[MetricLine]) -> str:
    out: List[str] = []
    # Core
    out.append("# HELP chad_metrics_server_up Metrics server is running.")
    out.append("# TYPE chad_metrics_server_up gauge")
    out.append("# HELP chad_systemd_unit_active Whether a systemd unit is active.")
    out.append("# TYPE chad_systemd_unit_active gauge")
    out.append("# HELP chad_systemd_unit_failed Whether a systemd unit is failed.")
    out.append("# TYPE chad_systemd_unit_failed gauge")
    out.append("# HELP chad_runtime_file_age_seconds Age of runtime artifact file in seconds.")
    out.append("# TYPE chad_runtime_file_age_seconds gauge")
    out.append("# HELP chad_portfolio_total_equity Total equity from portfolio_snapshot.json.")
    out.append("# TYPE chad_portfolio_total_equity gauge")
    out.append("# HELP chad_dynamic_caps_total_equity Total equity from dynamic_caps.json.")
    out.append("# TYPE chad_dynamic_caps_total_equity gauge")
    out.append("# HELP chad_snapshot_caps_match 1 if portfolio total equals dynamic caps total within tolerance.")
    out.append("# TYPE chad_snapshot_caps_match gauge")
    out.append("# HELP chad_paper_shadow_enabled 1 if paper_shadow.json enabled is true.")
    out.append("# TYPE chad_paper_shadow_enabled gauge")
    out.append("# HELP chad_ibkr_paper_ledger_enabled 1 if ibkr_paper_ledger.json enabled is true.")
    out.append("# TYPE chad_ibkr_paper_ledger_enabled gauge")
    out.append("# HELP chad_trade_ledgers_active_count Count of active trade_history_*.ndjson files.")
    out.append("# TYPE chad_trade_ledgers_active_count gauge")
    out.append("# HELP chad_trade_ledgers_quarantine_count Count of quarantined trade_history files.")
    out.append("# TYPE chad_trade_ledgers_quarantine_count gauge")
    out.append("# HELP chad_ibkr_ok 1 if IBKR watchdog status ok is true.")
    out.append("# TYPE chad_ibkr_ok gauge")
    out.append("# HELP chad_ibkr_consecutive_failures Consecutive IBKR healthcheck failures.")
    out.append("# TYPE chad_ibkr_consecutive_failures gauge")
    out.append("# HELP chad_ibkr_last_ok_age_seconds Seconds since last ok healthcheck (large if unknown).")
    out.append("# TYPE chad_ibkr_last_ok_age_seconds gauge")

    # Rollups
    out.append("# HELP chad_paper_trades_total Total number of paper trades in rollup window.")
    out.append("# TYPE chad_paper_trades_total gauge")
    out.append("# HELP chad_paper_win_rate Win-rate over paper trades in rollup window.")
    out.append("# TYPE chad_paper_win_rate gauge")
    out.append("# HELP chad_paper_total_pnl Total finite PnL over paper trades in rollup window.")
    out.append("# TYPE chad_paper_total_pnl gauge")
    out.append("# HELP chad_paper_avg_pnl Average finite PnL over paper trades in rollup window.")
    out.append("# TYPE chad_paper_avg_pnl gauge")
    out.append("# HELP chad_paper_sharpe_like Sharpe-like ratio over finite PnL series.")
    out.append("# TYPE chad_paper_sharpe_like gauge")
    out.append("# HELP chad_paper_max_drawdown Max drawdown over finite cumulative PnL series.")
    out.append("# TYPE chad_paper_max_drawdown gauge")
    out.append("# HELP chad_paper_total_notional Total notional over paper trades in rollup window.")
    out.append("# TYPE chad_paper_total_notional gauge")
    out.append("# HELP chad_paper_pnl_untrusted_count Count of paper trades tagged pnl_untrusted.")
    out.append("# TYPE chad_paper_pnl_untrusted_count gauge")
    out.append("# HELP chad_paper_pnl_nonfinite_count Count of paper trades whose pnl is NaN/inf (excluded from sums).")
    out.append("# TYPE chad_paper_pnl_nonfinite_count gauge")
    out.append("# HELP chad_paper_last_trade_age_seconds Seconds since most recent paper trade exit.")
    out.append("# TYPE chad_paper_last_trade_age_seconds gauge")

    out.append("# HELP chad_paper_strategy_trades_total Paper trades per strategy in rollup window.")
    out.append("# TYPE chad_paper_strategy_trades_total gauge")
    out.append("# HELP chad_paper_strategy_total_pnl Finite total PnL per strategy in rollup window.")
    out.append("# TYPE chad_paper_strategy_total_pnl gauge")
    out.append("# HELP chad_paper_strategy_win_rate Win-rate per strategy in rollup window.")
    out.append("# TYPE chad_paper_strategy_win_rate gauge")

    for ln in lines:
        out.append(ln.render())
    out.append("")
    return "\n".join(out)


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._server_state = kwargs.pop("server_state")
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/healthz":
            body = b"ok\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path in ("/-/ready", "/readyz"):
            try:
                _ = self._server_state.render_metrics()
                body = b"ready\n"
                self.send_response(200)
            except Exception:
                body = b"not ready\n"
                self.send_response(503)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path not in ("/metrics", "/"):
            self.send_response(404)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"not found\n")
            return

        try:
            body = self._server_state.render_metrics().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            msg = f"metrics error: {e}\n".encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


class MetricsServerState:
    def __init__(self, root_dir: Path, repo_dir: Path, trades_dir: Path, rollup_cache: RollupCache) -> None:
        self._root_dir = root_dir
        self._repo_dir = repo_dir
        self._trades_dir = trades_dir
        self._rollup_cache = rollup_cache

    def render_metrics(self) -> str:
        lines = _collect_metrics(self._root_dir, self._repo_dir, self._trades_dir, self._rollup_cache)
        return _render_prometheus(lines)


def main() -> int:
    root_dir = _resolve_root_dir()
    repo_dir = _resolve_repo_dir()
    trades_dir = _resolve_trades_dir(root_dir, repo_dir)

    host = os.environ.get("CHAD_METRICS_HOST", LISTEN_HOST_DEFAULT).strip() or LISTEN_HOST_DEFAULT
    port_s = os.environ.get("CHAD_METRICS_PORT", str(LISTEN_PORT_DEFAULT)).strip()
    try:
        port = int(port_s)
    except Exception:
        port = LISTEN_PORT_DEFAULT

    ttl = _env_float("CHAD_ROLLUP_TTL_SECONDS", ROLLUP_TTL_SECONDS_DEFAULT)
    rollup_cache = RollupCache(ttl_seconds=ttl)
    state = MetricsServerState(root_dir=root_dir, repo_dir=repo_dir, trades_dir=trades_dir, rollup_cache=rollup_cache)

    def handler(*args: Any, **kwargs: Any):
        return _Handler(*args, server_state=state, **kwargs)

    httpd = _HTTPServer((host, port), handler)
    print(f"[metrics_server] cwd={Path.cwd().resolve()}")
    print(f"[metrics_server] root_dir={root_dir} repo_dir={repo_dir} trades_dir={trades_dir}")
    print(f"[metrics_server] listening on http://{host}:{port}/metrics")
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
