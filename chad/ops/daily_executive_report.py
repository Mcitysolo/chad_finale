"""
CHAD Phase 10 — Daily Executive Report (Ops + Performance)

Goal:
- Produce ONE daily executive report that merges:
  1) Daily Ops Report (health + SCR + metrics + inputs)
  2) Daily Performance Report (trusted realized PnL rollups from ledger)

Outputs:
  reports/ops/DAILY_EXEC_REPORT_<ts>.json
  reports/ops/DAILY_EXEC_REPORT_<ts>.md

Design:
- Calls existing report modules as functions when possible; otherwise falls back to module execution.
- Fails safe: if one sub-report fails, executive report still emits with error context.
- No secrets printed.
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path("/home/ubuntu/CHAD FINALE")
REPORTS_DIR = REPO_ROOT / "reports" / "ops"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def write_md(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content.rstrip() + "\n", encoding="utf-8")
    os.replace(tmp, path)


@dataclass(frozen=True)
class SubReportResult:
    ok: bool
    err: Optional[str]
    json_path: Optional[str]
    md_path: Optional[str]
    payload: Optional[Dict[str, Any]]


def _load_json(path_str: str) -> Dict[str, Any]:
    p = Path(path_str)
    return json.loads(p.read_text(encoding="utf-8", errors="ignore"))


def run_ops_report() -> SubReportResult:
    """
    Attempt to run chad.ops.daily_ops_report.
    Expects it to print:
      <json_path>
      <md_path>
    """
    try:
        from chad.ops import daily_ops_report  # type: ignore

        if hasattr(daily_ops_report, "run"):
            json_path, md_path = daily_ops_report.run()  # type: ignore[attr-defined]
            payload = _load_json(str(json_path))
            return SubReportResult(True, None, str(json_path), str(md_path), payload)

        import subprocess

        proc = subprocess.run(
            [str(REPO_ROOT / "venv" / "bin" / "python"), "-m", "chad.ops.daily_ops_report"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            check=True,
        )
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        if len(lines) < 2:
            return SubReportResult(False, "daily_ops_report produced no paths", None, None, None)
        json_path, md_path = lines[-2], lines[-1]
        payload = _load_json(json_path)
        return SubReportResult(True, None, json_path, md_path, payload)
    except Exception as e:
        return SubReportResult(False, f"{type(e).__name__}: {e}", None, None, None)


def run_perf_report() -> SubReportResult:
    """
    Attempt to run chad.ops.daily_performance_report.
    Expects it to print:
      <json_path>
      <md_path>
    """
    try:
        from chad.ops import daily_performance_report  # type: ignore

        if hasattr(daily_performance_report, "run"):
            json_path, md_path = daily_performance_report.run()  # type: ignore[attr-defined]
            payload = _load_json(str(json_path))
            return SubReportResult(True, None, str(json_path), str(md_path), payload)

        import subprocess

        proc = subprocess.run(
            [str(REPO_ROOT / "venv" / "bin" / "python"), "-m", "chad.ops.daily_performance_report"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            check=True,
        )
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        if len(lines) < 2:
            return SubReportResult(False, "daily_performance_report produced no paths", None, None, None)
        json_path, md_path = lines[-2], lines[-1]
        payload = _load_json(json_path)
        return SubReportResult(True, None, json_path, md_path, payload)
    except Exception as e:
        return SubReportResult(False, f"{type(e).__name__}: {e}", None, None, None)


def _fmt(v: Any) -> str:
    return str(v)


def render_md(exec_payload: Dict[str, Any]) -> str:
    ops = exec_payload.get("ops") or {}
    perf = exec_payload.get("performance") or {}

    lines = []
    lines.append("# CHAD Daily Executive Report (Phase 10)")
    lines.append(f"- Generated: `{exec_payload.get('generated_utc')}`")
    lines.append(f"- Host: `{exec_payload.get('host')}`")
    lines.append("")

    # Ops snapshot
    lines.append("## Ops Snapshot")
    if ops.get("ok") is True and isinstance(ops.get("payload"), dict):
        op = ops["payload"]

        # daily_ops_report schema (source of truth):
        # - scr_state: "WARMUP" | ...
        # - ibkr_health: { ok, latency_ms, ttl_seconds, ... }
        # - metrics: { paper_trades_total, paper_win_rate, paper_total_pnl, ... }
        # - ledger_summary: { total_records, total_pnl, alpha_records, beta_records, untrusted_records }
        lines.append(f"- SCR State: `{_fmt(op.get('scr_state'))}`")

        ib = op.get("ibkr_health") or {}
        if isinstance(ib, dict):
            lines.append(
                f"- IBKR ok: `{_fmt(ib.get('ok'))}` "
                f"latency_ms=`{_fmt(ib.get('latency_ms'))}` "
                f"ttl=`{_fmt(ib.get('ttl_seconds'))}`"
            )

        met = op.get("metrics") or {}
        if isinstance(met, dict):
            lines.append(
                f"- Paper trades: `{_fmt(met.get('paper_trades_total'))}` "
                f"win_rate=`{_fmt(met.get('paper_win_rate'))}` "
                f"pnl=`{_fmt(met.get('paper_total_pnl'))}`"
            )

        led = op.get("ledger_summary") or {}
        if isinstance(led, dict):
            lines.append(
                f"- Ledger today: total_records=`{_fmt(led.get('total_records'))}` "
                f"alpha=`{_fmt(led.get('alpha_records'))}` "
                f"beta=`{_fmt(led.get('beta_records'))}` "
                f"untrusted=`{_fmt(led.get('untrusted_records'))}` "
                f"pnl_total=`{_fmt(led.get('total_pnl'))}`"
            )
    else:
        lines.append(f"- Ops report: FAILED err=`{ops.get('err')}`")
    lines.append("")

    # Performance snapshot
    lines.append("## Performance Snapshot (trusted realized)")
    if perf.get("ok") is True and isinstance(perf.get("payload"), dict):
        pp = perf["payload"]
        overall = pp.get("overall") or {}
        parse = pp.get("parse") or {}
        lines.append(
            f"- Ledger lines: `{_fmt(parse.get('ledger_lines'))}` "
            f"rows_used=`{_fmt(parse.get('rows_used'))}` "
            f"skipped=`{_fmt(parse.get('rows_skipped'))}`"
        )
        if isinstance(overall, dict):
            lines.append(
                f"- Trades: `{_fmt(overall.get('trades'))}` "
                f"win_rate=`{_fmt(overall.get('win_rate'))}` "
                f"pnl_total=`{_fmt(overall.get('pnl_total'))}`"
            )
            bt = overall.get("best_trade") or {}
            wt = overall.get("worst_trade") or {}
            if isinstance(bt, dict) and bt:
                lines.append(
                    f"- Best trade: `{bt.get('strategy')}` `{bt.get('symbol')}` `{bt.get('side')}` pnl=`{bt.get('pnl')}`"
                )
            if isinstance(wt, dict) and wt:
                lines.append(
                    f"- Worst trade: `{wt.get('strategy')}` `{wt.get('symbol')}` `{wt.get('side')}` pnl=`{wt.get('pnl')}`"
                )
        lines.append("")
        lines.append("### By Strategy (PnL)")
        bys = pp.get("by_strategy") or {}
        if isinstance(bys, dict) and bys:
            for strat, summ in sorted(bys.items()):
                if isinstance(summ, dict):
                    lines.append(
                        f"- `{strat}` "
                        f"trades=`{_fmt(summ.get('trades'))}` "
                        f"win_rate=`{_fmt(summ.get('win_rate'))}` "
                        f"pnl=`{_fmt(summ.get('pnl_total'))}`"
                    )
        else:
            lines.append("- (none)")
    else:
        lines.append(f"- Performance report: FAILED err=`{perf.get('err')}`")

    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- Ops JSON: `{ops.get('json_path')}`")
    lines.append(f"- Ops MD: `{ops.get('md_path')}`")
    lines.append(f"- Perf JSON: `{perf.get('json_path')}`")
    lines.append(f"- Perf MD: `{perf.get('md_path')}`")
    lines.append("")

    return "\n".join(lines)


def run() -> Tuple[Path, Path]:
    safe_mkdir(REPORTS_DIR)
    ts = utc_now_compact()

    ops_res = run_ops_report()
    perf_res = run_perf_report()

    payload: Dict[str, Any] = {
        "generated_utc": utc_now_iso(),
        "host": socket.gethostname(),
        "ops": {
            "ok": ops_res.ok,
            "err": ops_res.err,
            "json_path": ops_res.json_path,
            "md_path": ops_res.md_path,
            "payload": ops_res.payload,
        },
        "performance": {
            "ok": perf_res.ok,
            "err": perf_res.err,
            "json_path": perf_res.json_path,
            "md_path": perf_res.md_path,
            "payload": perf_res.payload,
        },
    }

    jpath = REPORTS_DIR / f"DAILY_EXEC_REPORT_{ts}.json"
    mpath = REPORTS_DIR / f"DAILY_EXEC_REPORT_{ts}.md"

    write_json(jpath, payload)
    md_text = render_md(payload)
    write_md(mpath, md_text)

    # Telegram push (Phase 10 operator surface)
    # Uses hardened notifier already used by watchdogs.
    # Fail-safe: report generation must succeed even if Telegram fails.
    try:
        from chad.utils.telegram_notify import notify  # type: ignore

        snippet = "\n".join(md_text.splitlines()[:35]).strip()
        if snippet:
            notify(
                snippet,
                severity="info",
                dedupe_key="daily_exec_report",
                raise_on_fail=False,
            )
    except Exception:
        pass

    print(str(jpath))
    print(str(mpath))
    return jpath, mpath


if __name__ == "__main__":
    run()
