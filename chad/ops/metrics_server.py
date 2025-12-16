#!/usr/bin/env python3
"""
CHAD Metrics Server (Phase 9)

Exposes a minimal Prometheus-compatible /metrics endpoint on 0.0.0.0:9620.

Design goals
------------
- Read-only observability: NEVER places orders, NEVER modifies trading config.
- Safe-by-default: if runtime artifacts are missing/corrupt, metrics still render.
- Low dependencies: uses stdlib + existing CHAD paths (no extra pip installs).
- Deterministic output: stable metric names + labels.

What we export (initial baseline)
---------------------------------
1) Timer/Service health (systemd):
   - chad_systemd_unit_active{unit="..."} 0|1
   - chad_systemd_unit_failed{unit="..."} 0|1
2) Runtime artifact freshness (seconds since mtime):
   - chad_runtime_file_age_seconds{file="..."} <float>
3) Portfolio snapshot + caps totals:
   - chad_portfolio_total_equity <float>
   - chad_dynamic_caps_total_equity <float>
   - chad_snapshot_caps_match 0|1
4) Paper lanes (preview-only right now):
   - chad_paper_shadow_enabled 0|1
   - chad_ibkr_paper_ledger_enabled 0|1
   - chad_trade_ledgers_active_count <int>
   - chad_trade_ledgers_quarantine_count <int>

This is Phase 9 foundation. Alerting rules and richer metrics come next.

Run
---
python -m chad.ops.metrics_server

Systemd service will be added in the next step.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

ROOT_DEFAULT = Path("/home/ubuntu/CHAD FINALE")
RUNTIME_DEFAULT = ROOT_DEFAULT / "runtime"
DATA_TRADES_DIR = ROOT_DEFAULT / "data" / "trades"

LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 9620


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


def _now() -> float:
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
    # "failed" or "inactive" are distinct; we specifically flag failed.
    rc, out = _run(["systemctl", "is-failed", unit])
    if rc != 0 and out == "":
        return None
    return out.strip() == "failed"


def _iter_units() -> List[str]:
    # Units we know we care about right now (from Phase 8/9 work).
    return [
        "chad-paper-shadow-runner.timer",
        "chad-paper-shadow-runner.service",
        "chad-ibkr-paper-ledger-watcher.timer",
        "chad-ibkr-paper-ledger-watcher.service",
        "chad-ibkr-health.timer",
        "chad-ibkr-watchdog.timer",
        "chad-ibkr-collector.timer",
    ]


def _iter_runtime_files() -> List[Path]:
    # Runtime artifacts that should exist if CHAD is healthy.
    return [
        RUNTIME_DEFAULT / "portfolio_snapshot.json",
        RUNTIME_DEFAULT / "dynamic_caps.json",
        RUNTIME_DEFAULT / "ibkr_status.json",
        RUNTIME_DEFAULT / "paper_shadow.json",
        RUNTIME_DEFAULT / "ibkr_paper_ledger.json",
    ]


def _collect_metrics() -> List[MetricLine]:
    now = _now()
    lines: List[MetricLine] = []

    # --- systemd health ---
    for unit in _iter_units():
        active = _systemd_is_active(unit)
        failed = _systemd_is_failed(unit)
        if active is not None:
            lines.append(MetricLine("chad_systemd_unit_active", {"unit": unit}, 1.0 if active else 0.0))
        if failed is not None:
            lines.append(MetricLine("chad_systemd_unit_failed", {"unit": unit}, 1.0 if failed else 0.0))

    # --- runtime file ages ---
    for p in _iter_runtime_files():
        age = _file_age_seconds(p, now)
        if age is not None:
            lines.append(MetricLine("chad_runtime_file_age_seconds", {"file": str(p)}, float(age)))

    # --- portfolio totals & caps ---
    snap = _safe_read_json(RUNTIME_DEFAULT / "portfolio_snapshot.json") or {}
    caps = _safe_read_json(RUNTIME_DEFAULT / "dynamic_caps.json") or {}

    ibkr_eq = float(snap.get("ibkr_equity", 0.0) or 0.0)
    cb_eq = float(snap.get("coinbase_equity", 0.0) or 0.0)
    kr_eq = float(snap.get("kraken_equity", 0.0) or 0.0)
    total = ibkr_eq + cb_eq + kr_eq

    caps_total = float(caps.get("total_equity", 0.0) or 0.0) if caps else 0.0
    match = 1.0 if abs(total - caps_total) < 1e-6 else 0.0

    lines.append(MetricLine("chad_portfolio_total_equity", {}, float(total)))
    lines.append(MetricLine("chad_dynamic_caps_total_equity", {}, float(caps_total)))
    lines.append(MetricLine("chad_snapshot_caps_match", {}, float(match)))

    # --- paper shadow config flags ---
    ps = _safe_read_json(RUNTIME_DEFAULT / "paper_shadow.json") or {}
    ps_enabled = 1.0 if bool(ps.get("enabled", False)) else 0.0
    lines.append(MetricLine("chad_paper_shadow_enabled", {}, ps_enabled))

    # --- ibkr paper ledger watcher flags ---
    pl = _safe_read_json(RUNTIME_DEFAULT / "ibkr_paper_ledger.json") or {}
    pl_enabled = 1.0 if bool(pl.get("enabled", False)) else 0.0
    lines.append(MetricLine("chad_ibkr_paper_ledger_enabled", {}, pl_enabled))

    # --- trade ledgers (active vs quarantine) ---
    active_count = 0
    quarantine_count = 0
    try:
        if DATA_TRADES_DIR.exists():
            active_count = len(list(DATA_TRADES_DIR.glob("trade_history_*.ndjson")))
            q = DATA_TRADES_DIR / "_quarantine"
            if q.exists():
                quarantine_count = len(list(q.glob("trade_history_*.ndjson")))
    except Exception:
        pass
    lines.append(MetricLine("chad_trade_ledgers_active_count", {}, float(active_count)))
    lines.append(MetricLine("chad_trade_ledgers_quarantine_count", {}, float(quarantine_count)))

    # --- build info ---
    lines.append(MetricLine("chad_metrics_server_up", {}, 1.0))

    return lines


def _render_prometheus(lines: Iterable[MetricLine]) -> str:
    # Emit HELP/TYPE for key metrics (small, stable set).
    out: List[str] = []
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

    for ln in lines:
        out.append(ln.render())
    out.append("")
    return "\n".join(out)


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path not in ("/metrics", "/"):
            self.send_response(404)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"not found\n")
            return

        try:
            body = _render_prometheus(_collect_metrics()).encode("utf-8")
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

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401
        # Silence default http.server access logs; systemd/journal is enough.
        return


def main() -> int:
    host = os.environ.get("CHAD_METRICS_HOST", LISTEN_HOST).strip() or LISTEN_HOST
    port_s = os.environ.get("CHAD_METRICS_PORT", str(LISTEN_PORT)).strip()
    try:
        port = int(port_s)
    except Exception:
        port = LISTEN_PORT

    httpd = HTTPServer((host, port), _Handler)
    print(f"[metrics_server] listening on http://{host}:{port}/metrics")
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
