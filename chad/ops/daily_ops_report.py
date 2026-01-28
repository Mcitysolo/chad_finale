from __future__ import annotations

"""
chad/ops/daily_ops_report.py

CHAD Phase 10 — Daily Ops Report Generator (hardened)

What it produces
----------------
Writes two artifacts to reports/ops/:
1) JSON:  DAILY_OPS_REPORT_YYYYMMDDThhmmssZ.json
2) MD:    DAILY_OPS_REPORT_YYYYMMDDThhmmssZ.md

And updates symlinks:
- DAILY_OPS_REPORT_LATEST.json
- DAILY_OPS_REPORT_LATEST.md

Inputs (read-only, best-effort)
-------------------------------
- Metrics endpoint (http://127.0.0.1:9620/metrics)
- runtime/portfolio_snapshot.json
- runtime/ibkr_status.json
- runtime/full_execution_cycle_last.json
- data/shadow/shadow_state.json
- data/trades/trade_history_YYYYMMDD.ndjson (today; falls back to latest ledger if missing)
- config/symbol_caps.json (optional policy snapshot config)

Safety contract
---------------
- No broker calls
- No config mutation
- No runtime mutation (except writing report artifacts + symlinks)
- No secrets emitted (reads only non-secret artifacts)

CLI
---
python -m chad.ops.daily_ops_report
"""

import json
import math
import os
import re
import socket
import subprocess
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ----------------------------
# Paths / constants
# ----------------------------

ROOT = Path("/home/ubuntu/CHAD FINALE")
RUNTIME = ROOT / "runtime"
DATA = ROOT / "data"
TRADES_DIR = DATA / "trades"
REPORTS_DIR = ROOT / "reports" / "ops"
CONFIG_DIR = ROOT / "config"
SYMBOL_CAPS_CONFIG = CONFIG_DIR / "symbol_caps.json"

METRICS_URL = "http://127.0.0.1:9620/metrics"


# ----------------------------
# Core data models
# ----------------------------

@dataclass(frozen=True)
class InputStatus:
    path: str
    ok: bool
    error: Optional[str]


@dataclass(frozen=True)
class LedgerSummary:
    ledger_path: str
    total_records: int
    alpha_records: int
    beta_records: int
    untrusted_records: int
    total_pnl: float


@dataclass(frozen=True)
class IBKRHealthView:
    ok: Optional[bool]
    latency_ms: Optional[float]
    ttl_seconds: Optional[int]
    ts_utc: Optional[str]


# ----------------------------
# Utility helpers
# ----------------------------

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def today_ymd_utc() -> str:
    return utc_now().strftime("%Y%m%d")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def safe_symlink_latest(target: Path, link: Path) -> None:
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        # relative symlink within same dir
        link.symlink_to(target.name)
    except Exception:
        # never block report generation on symlink errors
        pass


def read_json_file(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        if not path.is_file():
            return None, f"missing:{path}"
        raw = path.read_text(encoding="utf-8", errors="ignore")
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return None, f"invalid_json_object:{path}"
        return obj, None
    except Exception as exc:
        return None, f"read_failed:{path}:{type(exc).__name__}:{exc}"


def clamp_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        if not math.isfinite(f):
            return None
        return f
    except Exception:
        return None


def clamp_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def norm_symbol(v: Any) -> str:
    s = str(v or "").strip().upper()
    return s if s else "UNKNOWN"


def norm_strategy(v: Any) -> str:
    s = str(v or "").strip().lower()
    return s if s else "unknown"


# ----------------------------
# Metrics parsing
# ----------------------------

_METRIC_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(-?\d+(?:\.\d+)?)$")


def fetch_metrics_text(timeout_s: float = 2.0) -> Tuple[Optional[str], Optional[str]]:
    try:
        with urllib.request.urlopen(METRICS_URL, timeout=float(timeout_s)) as r:
            raw = r.read().decode("utf-8", errors="replace")
        return raw, None
    except Exception as exc:
        return None, f"metrics_fetch_failed:{type(exc).__name__}:{exc}"


def parse_metric_no_labels(metrics_text: str, name: str) -> Optional[float]:
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _METRIC_LINE.match(line.strip())
        if not m:
            continue
        mname, labels, value = m.group(1), m.group(2), m.group(3)
        if mname == name and labels is None:
            try:
                return float(value)
            except Exception:
                return None
    return None


def parse_scr_state(metrics_text: str) -> str:
    """
    Determine which SCR one-hot state is 1.0.
    Expects lines like:
      chad_scr_state{state="WARMUP"} 1.0
    """
    best = "UNKNOWN"
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        if not line.startswith("chad_scr_state"):
            continue
        if "state=" not in line:
            continue
        try:
            label = line.split("{", 1)[1].split("}", 1)[0]
            state = label.split("state=", 1)[1].strip().strip('"')
            val = float(line.rsplit(" ", 1)[1])
            if val == 1.0:
                best = state
                break
        except Exception:
            continue
    return best


# ----------------------------
# Ledger discovery + parsing
# ----------------------------

def ledger_path_today_or_latest() -> Path:
    """
    Use today's ledger if present; otherwise fall back to newest available ledger.
    If none exist, return today's expected path (caller will mark missing).
    """
    today = TRADES_DIR / f"trade_history_{today_ymd_utc()}.ndjson"
    if today.is_file():
        return today
    candidates = sorted(TRADES_DIR.glob("trade_history_*.ndjson"))
    if candidates:
        return candidates[-1]
    return today


def iter_ndjson(path: Path) -> Iterable[Dict[str, Any]]:
    # streaming reader; avoids loading entire file
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def ledger_payload(obj: Dict[str, Any]) -> Dict[str, Any]:
    p = obj.get("payload")
    return p if isinstance(p, dict) else obj


def is_untrusted_entry_only(p: Dict[str, Any]) -> bool:
    extra = p.get("extra")
    return isinstance(extra, dict) and extra.get("pnl_untrusted") is True


def summarize_ledger(path: Path) -> Tuple[Optional[LedgerSummary], Optional[str]]:
    if not path.is_file():
        return None, f"missing:{path}"

    total = 0
    alpha = 0
    beta = 0
    untrusted = 0
    pnl_sum = 0.0

    try:
        for obj in iter_ndjson(path):
            p = ledger_payload(obj)
            if not isinstance(p, dict):
                continue

            total += 1
            strat = norm_strategy(p.get("strategy"))
            if strat == "alpha":
                alpha += 1
            elif strat == "beta":
                beta += 1

            # Canonical untrusted flag (your real schema)
            if is_untrusted_entry_only(p):
                untrusted += 1

            pnl = clamp_float(p.get("pnl"))
            if pnl is not None:
                pnl_sum += pnl

        return (
            LedgerSummary(
                ledger_path=str(path),
                total_records=total,
                alpha_records=alpha,
                beta_records=beta,
                untrusted_records=untrusted,
                total_pnl=float(pnl_sum),
            ),
            None,
        )
    except Exception as exc:
        return None, f"ledger_read_failed:{type(exc).__name__}:{exc}"


# ----------------------------
# Symbol cap config + evaluator
# ----------------------------

def load_symbol_caps_config() -> Dict[str, Any]:
    """
    Loads config/symbol_caps.json (no secrets). Fail-safe on any error.
    Expected schema:
      { "enabled": true, "symbols": { "AAPL": {"max_trades_per_day": 200, "max_consecutive_losses": 8}, ... } }
    """
    try:
        if not SYMBOL_CAPS_CONFIG.is_file():
            return {"enabled": False, "symbols": {}}
        obj = json.loads(SYMBOL_CAPS_CONFIG.read_text(encoding="utf-8", errors="ignore"))
        if not isinstance(obj, dict):
            return {"enabled": False, "symbols": {}}
        enabled = bool(obj.get("enabled", False))
        symbols = obj.get("symbols", {})
        if not isinstance(symbols, dict):
            symbols = {}
        return {"enabled": enabled, "symbols": symbols}
    except Exception:
        return {"enabled": False, "symbols": {}}


def run_symbol_cap_evaluator(symbol: str, max_trades_per_day: int, max_consecutive_losses: int) -> Dict[str, Any]:
    """
    Calls the deterministic evaluator module and returns parsed JSON.
    Fail-safe: returns {"ok": False, "error": "..."} on errors.
    """
    try:
        cmd = [
            str(ROOT / "venv" / "bin" / "python"),
            "-m",
            "chad.policy_guards.symbol_trade_cap",
            "--symbol",
            str(symbol),
            "--max-trades-per-day",
            str(int(max_trades_per_day)),
            "--max-consecutive-losses",
            str(int(max_consecutive_losses)),
        ]
        p = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if p.returncode != 0:
            return {"ok": False, "error": f"rc={p.returncode}", "stderr_tail": (p.stderr or "")[-300:]}
        out = (p.stdout or "").strip()
        if not out:
            return {"ok": False, "error": "empty_stdout"}
        return {"ok": True, "decision": json.loads(out)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def build_symbol_caps_snapshot() -> Dict[str, Any]:
    cfg = load_symbol_caps_config()
    if not cfg.get("enabled"):
        return {"enabled": False, "symbols": {}}

    symbols_cfg = cfg.get("symbols") or {}
    if not isinstance(symbols_cfg, dict):
        return {"enabled": True, "symbols": {}}

    out: Dict[str, Any] = {"enabled": True, "symbols": {}}

    for sym, limits in symbols_cfg.items():
        sym_s = norm_symbol(sym)
        if sym_s == "UNKNOWN":
            continue

        max_trades = 200
        max_losses = 8
        if isinstance(limits, dict):
            mt = clamp_int(limits.get("max_trades_per_day"))
            ml = clamp_int(limits.get("max_consecutive_losses"))
            if mt is not None:
                max_trades = mt
            if ml is not None:
                max_losses = ml

        out["symbols"][sym_s] = run_symbol_cap_evaluator(sym_s, max_trades, max_losses)

    return out


# ----------------------------
# IBKR health view (from runtime/ibkr_status.json)
# ----------------------------

def ibkr_health_view(ibkr_status: Optional[Dict[str, Any]]) -> Optional[IBKRHealthView]:
    if not isinstance(ibkr_status, dict):
        return None
    return IBKRHealthView(
        ok=ibkr_status.get("ok") if isinstance(ibkr_status.get("ok"), bool) else None,
        latency_ms=clamp_float(ibkr_status.get("latency_ms")),
        ttl_seconds=clamp_int(ibkr_status.get("ttl_seconds")),
        ts_utc=str(ibkr_status.get("ts_utc")) if ibkr_status.get("ts_utc") is not None else None,
    )


# ----------------------------
# Report builder + renderer
# ----------------------------

def build_report() -> Tuple[Dict[str, Any], str]:
    now = utc_now()
    host = socket.gethostname()

    metrics_text, metrics_err = fetch_metrics_text()
    scr_state = "UNKNOWN"
    paper_trades_total = None
    paper_win_rate = None
    paper_total_pnl = None

    if metrics_text:
        scr_state = parse_scr_state(metrics_text)
        paper_trades_total = parse_metric_no_labels(metrics_text, "chad_paper_trades_total")
        paper_win_rate = parse_metric_no_labels(metrics_text, "chad_paper_win_rate")
        paper_total_pnl = parse_metric_no_labels(metrics_text, "chad_paper_total_pnl")

    portfolio, portfolio_err = read_json_file(RUNTIME / "portfolio_snapshot.json")
    ibkr_status, ibkr_err = read_json_file(RUNTIME / "ibkr_status.json")
    full_cycle, cycle_err = read_json_file(RUNTIME / "full_execution_cycle_last.json")
    shadow, shadow_err = read_json_file(DATA / "shadow" / "shadow_state.json")

    ledger_path = ledger_path_today_or_latest()
    ledger_sum, ledger_err = summarize_ledger(ledger_path)

    # Symbol cap snapshot (config-driven)
    symbol_caps = build_symbol_caps_snapshot()

    report: Dict[str, Any] = {
        "generated_at_utc": iso_utc(now),
        "host": host,
        "phase": "10",
        "scr_state": scr_state,
        "metrics": {
            "ok": bool(metrics_text is not None),
            "error": metrics_err,
            "paper_trades_total": paper_trades_total,
            "paper_win_rate": paper_win_rate,
            "paper_total_pnl": paper_total_pnl,
        },
        "inputs": {
            "portfolio_snapshot": as_input_status(RUNTIME / "portfolio_snapshot.json", portfolio, portfolio_err),
            "ibkr_status": as_input_status(RUNTIME / "ibkr_status.json", ibkr_status, ibkr_err),
            "full_execution_cycle_last": as_input_status(RUNTIME / "full_execution_cycle_last.json", full_cycle, cycle_err),
            "shadow_state": as_input_status(DATA / "shadow" / "shadow_state.json", shadow, shadow_err),
            "ledger_today": {
                "path": str(ledger_path),
                "ok": ledger_sum is not None,
                "error": ledger_err,
            },
        },
        "ledger_summary": None if ledger_sum is None else {
            "total_records": ledger_sum.total_records,
            "alpha_records": ledger_sum.alpha_records,
            "beta_records": ledger_sum.beta_records,
            "untrusted_records": ledger_sum.untrusted_records,
            "total_pnl": ledger_sum.total_pnl,
        },
        "full_cycle_counts": None if not isinstance(full_cycle, dict) else full_cycle.get("counts"),
        "ibkr_health": None if ibkr_status is None else asdict_ibkr(ibkr_health_view(ibkr_status)),
        "symbol_caps": symbol_caps,
    }

    # Markdown render
    md_lines: List[str] = []
    md_lines.append("# CHAD Daily Ops Report (Phase 10)")
    md_lines.append(f"- Generated: `{report['generated_at_utc']}`")
    md_lines.append(f"- Host: `{host}`")
    md_lines.append("")

    md_lines.append("## SCR / Mode")
    md_lines.append(f"- SCR State: **{scr_state}**")
    md_lines.append("")

    md_lines.append("## Metrics (paper)")
    md_lines.append(f"- Paper trades total (LEAN): `{paper_trades_total}`")
    md_lines.append(f"- Paper win rate: `{paper_win_rate}`")
    md_lines.append(f"- Paper total PnL: `{paper_total_pnl}`")
    md_lines.append("")

    md_lines.append("## IBKR Health (runtime)")
    ibv = report.get("ibkr_health") or {}
    md_lines.append(f"- ok: `{ibv.get('ok') if isinstance(ibv, dict) else None}`")
    md_lines.append(f"- latency_ms: `{ibv.get('latency_ms') if isinstance(ibv, dict) else None}`")
    md_lines.append(f"- ttl_seconds: `{ibv.get('ttl_seconds') if isinstance(ibv, dict) else None}`")
    md_lines.append("")

    md_lines.append("## Ledger (latest)")
    if ledger_sum:
        md_lines.append(f"- records: `{ledger_sum.total_records}` (alpha `{ledger_sum.alpha_records}`, beta `{ledger_sum.beta_records}`)")
        md_lines.append(f"- pnl total (finite): `{ledger_sum.total_pnl}`")
        md_lines.append(f"- pnl_untrusted count: `{ledger_sum.untrusted_records}`")
        md_lines.append(f"- ledger_path: `{ledger_sum.ledger_path}`")
    else:
        md_lines.append(f"- error: `{ledger_err}`")
        md_lines.append(f"- ledger_path: `{ledger_path}`")
    md_lines.append("")

    md_lines.append("## Symbol Caps (policy snapshot)")
    scaps = report.get("symbol_caps") or {}
    if isinstance(scaps, dict) and scaps.get("enabled") is True:
        sym_map = scaps.get("symbols") or {}
        if isinstance(sym_map, dict) and sym_map:
            for sym, entry in sorted(sym_map.items()):
                if not isinstance(entry, dict) or entry.get("ok") is not True:
                    md_lines.append(f"- {sym}: symbol_cap_unavailable `{entry}`")
                    continue
                d = entry.get("decision")
                if not isinstance(d, dict):
                    md_lines.append(f"- {sym}: invalid_decision `{entry}`")
                    continue
                md_lines.append(
                    f"- {sym}: allowed=`{d.get('allowed')}` "
                    f"reason=`{d.get('reason_code')}` "
                    f"trades=`{d.get('trades_counted')}` "
                    f"loss_streak=`{d.get('consecutive_losses')}` "
                    f"caps=`{d.get('max_trades_per_day')}/{d.get('max_consecutive_losses')}` "
                    f"ledger=`{d.get('ledger_path')}`"
                )
        else:
            md_lines.append("- (enabled, but no symbols configured)")
    else:
        md_lines.append("- (disabled)")
    md_lines.append("")

    md_lines.append("## Input health")
    inputs = report.get("inputs") or {}
    if isinstance(inputs, dict):
        for k, v in inputs.items():
            if isinstance(v, dict):
                md_lines.append(f"- `{k}`: ok=`{v.get('ok')}` err=`{v.get('error')}` path=`{v.get('path')}`")
    md = "\n".join(md_lines) + "\n"

    return report, md


def as_input_status(path: Path, obj: Optional[Dict[str, Any]], err: Optional[str]) -> Dict[str, Any]:
    return {"path": str(path), "ok": obj is not None, "error": err}


def asdict_ibkr(v: Optional[IBKRHealthView]) -> Optional[Dict[str, Any]]:
    if v is None:
        return None
    return {"ok": v.ok, "latency_ms": v.latency_ms, "ttl_seconds": v.ttl_seconds, "ts_utc": v.ts_utc}


# ----------------------------
# Entrypoint
# ----------------------------

def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    report, md = build_report()

    json_path = REPORTS_DIR / f"DAILY_OPS_REPORT_{stamp}.json"
    md_path = REPORTS_DIR / f"DAILY_OPS_REPORT_{stamp}.md"

    atomic_write_text(json_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    atomic_write_text(md_path, md)

    safe_symlink_latest(json_path, REPORTS_DIR / "DAILY_OPS_REPORT_LATEST.json")
    safe_symlink_latest(md_path, REPORTS_DIR / "DAILY_OPS_REPORT_LATEST.md")

    print(str(json_path))
    print(str(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
