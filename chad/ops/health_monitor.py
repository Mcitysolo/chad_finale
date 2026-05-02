"""
CHAD AI Health Monitor — Main Orchestrator
Three-tier autonomous monitoring and self-healing.

Tier 1: Rule engine (every run, no API cost)
Tier 2: Claude reasoning (market hours, every run)
Tier 3: Auto-remediation (immediate fix + Telegram notify)

Operator always gets Telegram notification of what was done.
5-minute veto window for source code changes (future).
"""
from __future__ import annotations
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

# Add repo to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chad.ops.health_monitor_rules import Finding, run_all_rules
from chad.ops.health_monitor_remediation import execute_remedy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("chad.health_monitor")

RUNTIME = REPO_ROOT / "runtime"
_IS_MARKET_HOURS_APPROX = True  # always run Claude reasoning for now


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _build_system_snapshot() -> str:
    """Build a compact system snapshot for Claude reasoning."""
    lines = [f"CHAD system snapshot at {datetime.now(timezone.utc).isoformat()}"]

    # SCR
    scr = _read_json(RUNTIME / "scr_state.json")
    scr_effective = (scr.get("stats", {}) or {}).get("effective_trades")
    lines.append(f"SCR: state={scr.get('state')} sizing={scr.get('sizing_factor')} "
                 f"effective_trades={scr_effective}")

    # Regime
    regime = _read_json(RUNTIME / "regime_state.json")
    lines.append(f"Regime: {regime.get('regime')} confidence={regime.get('confidence')}")

    # PnL — include trade_count and realized_pnl explicitly,
    # and flag the gap with SCR effective_trades
    pnl = _read_json(RUNTIME / "pnl_state.json")
    raw_trades = pnl.get("trade_count")
    realized = pnl.get("realized_pnl")
    lines.append(f"PnL today: realized={realized} trades={raw_trades}")
    try:
        if raw_trades is not None and scr_effective is not None and \
                int(raw_trades) > 0 and int(scr_effective) > 0 and \
                int(raw_trades) > int(scr_effective) * 2:
            gap = int(raw_trades) - int(scr_effective)
            lines.append(
                f"SCR/raw gap: {gap} trades excluded "
                f"(raw={raw_trades} effective={scr_effective}) — "
                "likely rejected/partial/excluded fills"
            )
    except Exception:
        pass

    # Profit lock
    pl = _read_json(RUNTIME / "profit_lock_state.json")
    lines.append(f"Profit lock: mode={pl.get('mode')} sizing={pl.get('sizing_factor')}")

    # Equity
    snap = _read_json(RUNTIME / "portfolio_snapshot.json")
    lines.append(f"Equity: ibkr={snap.get('ibkr_equity')} kraken={snap.get('kraken_equity')}")

    # Reconciliation
    recon = _read_json(RUNTIME / "reconciliation_state.json")
    lines.append(f"Reconciliation: status={recon.get('status')} worst_diff={recon.get('worst_diff')}")

    # Strategy health — include all scores so Claude can see the full distribution
    health = _read_json(RUNTIME / "strategy_health.json")
    if health:
        strats = health.get("strategies", {}) or {}
        scores = {k: round(float(v.get("health_score", 1.0)), 3)
                  for k, v in strats.items() if isinstance(v, dict)}
        if scores:
            lines.append(f"Strategy health scores: {scores}")
        flagged = [(k, v.get("health_score", 1.0)) for k, v in strats.items()
                   if isinstance(v, dict)
                   and v.get("health_score", 1.0) < 0.5
                   and v.get("sample_count", 0) >= 10]
        if flagged:
            lines.append(f"Low health strategies (sample>=10): {flagged}")
        # Alpha cluster correlated degradation
        alphas_low = [k for k, s in scores.items()
                      if "alpha" in k.lower() and s < 0.5]
        if len(alphas_low) >= 3:
            lines.append(
                f"Alpha cluster degraded: {len(alphas_low)} strategies "
                f"<0.5 health: {alphas_low}"
            )

    # Winner scaling — flag anything <0.7 or >1.3
    ws = _read_json(RUNTIME / "winner_scaling.json")
    multipliers = ws.get("multipliers", {}) or {}
    penalized = {k: v for k, v in multipliers.items() if v < 0.7}
    boosted = {k: v for k, v in multipliers.items() if v > 1.3}
    if penalized:
        lines.append(f"Penalized strategies (<0.7): {penalized}")
    if boosted:
        lines.append(f"Boosted strategies (>1.3): {boosted}")

    # Business phase
    biz = _read_json(RUNTIME / "business_phase.json")
    lines.append(f"Business: phase={biz.get('phase')} growth={biz.get('growth_pct_from_seed')}%")

    # Stop bus
    sb = _read_json(RUNTIME / "stop_bus.json")
    lines.append(f"Stop bus: active={sb.get('active')}")

    # Signal throttle — surface if active so Claude knows churn brake is engaged
    throttle = _read_json(RUNTIME / "signal_throttle.json")
    if throttle.get("active"):
        lines.append(
            f"Signal throttle ACTIVE: max={throttle.get('max_signals_per_cycle')} "
            f"reason={throttle.get('reason')} "
            f"expires={throttle.get('auto_expires_at_utc')}"
        )

    return "\n".join(lines)


def _ask_claude(snapshot: str, rule_findings: List[Finding]) -> str:
    """Ask Claude to reason about the system snapshot."""
    try:
        import urllib.request
        import urllib.error

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return ""

        # Build context of what rules already found
        rule_summary = ""
        if rule_findings:
            rule_summary = "\n\nRule engine already found:\n" + "\n".join(
                f"- [{f.severity}] {f.title}" for f in rule_findings
            )

        prompt = f"""You are CHAD's AI health monitor. Review this system snapshot and identify issues AND for each issue state: (1) severity, (2) exact fix, (3) whether it is safe to auto-apply.

{snapshot}{rule_summary}

Rules:
1. Be specific. Name exact metrics, values, and thresholds.
2. Only flag real issues — not normal operating conditions.
3. Prioritize: CRITICAL (trading broken/halted) > WARNING (degrading) > INFO (watch).
4. If everything looks healthy, say "SYSTEM HEALTHY — no anomalies detected".
5. For each issue: severity → exact fix (command or file change) → safe to auto-apply (yes/no).
6. Max 8 lines total.

Focus on: strategy performance trends, regime stability, SCR trajectory,
equity curve shape, fill rate patterns, anything that looks unusual."""

        payload = json.dumps({
            "model": "claude-sonnet-4-6",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}]
        }).encode()

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            }
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data.get("content", [{}])[0].get("text", "")
    except Exception as e:
        logger.warning("Claude reasoning failed: %s", e)
        return ""


def _notify(message: str, severity: str = "info",
            dedupe_key: str = "") -> None:
    """Send Telegram notification."""
    try:
        from chad.utils.telegram_notify import notify
        notify(message, severity=severity,
               dedupe_key=dedupe_key if dedupe_key else None)
    except Exception as e:
        logger.warning("Telegram notify failed: %s", e)


def run_monitor(dry_run: bool = False) -> None:
    """Main monitor loop — one execution cycle."""
    start = time.time()
    logger.info("health_monitor: starting cycle dry_run=%s", dry_run)

    # ── Tier 1: Rule engine ───────────────────────────────────────────────────
    findings = run_all_rules()
    logger.info("health_monitor: rule_engine found=%d findings", len(findings))

    # ── Tier 2: Claude reasoning ─────────────────────────────────────────────
    claude_analysis = ""
    try:
        snapshot = _build_system_snapshot()
        claude_analysis = _ask_claude(snapshot, findings)
        if claude_analysis:
            logger.info("health_monitor: claude_analysis=%s",
                        claude_analysis[:200])
    except Exception as e:
        logger.warning("health_monitor: claude_reasoning_failed err=%s", e)

    # ── Tier 3: Remediation ───────────────────────────────────────────────────
    remediation_results = []
    for finding in findings:
        if dry_run:
            logger.info("DRY_RUN would fix: [%s] %s via %s",
                        finding.severity, finding.title, finding.remedy_action)
            continue

        result = execute_remedy(finding.remedy_action, finding.remedy_args)
        remediation_results.append((finding, result))
        logger.info("health_monitor: remediated rule=%s result=%s",
                    finding.rule_id, result[:100])

    # ── Telegram notifications ────────────────────────────────────────────────
    if not dry_run:
        # Send one message per critical finding with its remedy result
        for finding, result in remediation_results:
            if finding.remedy_action == "notify":
                # Pure notification — no auto-fix
                msg = (f"{'🚨' if finding.severity == 'CRITICAL' else '⚠️'} "
                       f"HEALTH MONITOR [{finding.severity}]\n"
                       f"{finding.title}\n"
                       f"{finding.description}\n"
                       f"Evidence: {finding.evidence}")
            else:
                # Auto-fixed — report what was done
                msg = (f"🔧 AUTO-FIXED [{finding.severity}]\n"
                       f"{finding.title}\n"
                       f"Action: {result}")
            _notify(msg,
                    severity="critical" if finding.severity == "CRITICAL"
                    else "warning",
                    dedupe_key=f"health_{finding.rule_id}_{finding.title[:30]}")

        # Send Claude analysis if it found something new
        if claude_analysis and "SYSTEM HEALTHY" not in claude_analysis:
            _notify(
                f"🤖 CHAD AI HEALTH ANALYSIS\n{claude_analysis}",
                severity="warning",
                dedupe_key="health_claude_analysis",
            )

    elapsed = time.time() - start
    summary = (f"health_monitor: checked rules={len(findings)} "
               f"findings={len(findings)} "
               f"elapsed={elapsed:.1f}s "
               f"claude={'yes' if claude_analysis else 'no'}")
    logger.info(summary)
    print(summary)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CHAD Health Monitor")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without making changes or sending alerts")
    args = parser.parse_args()
    try:
        run_monitor(dry_run=args.dry_run)
    except Exception as e:
        logger.exception("health_monitor: fatal err=%s", e)
    # Always exit 0 — monitor failure must not cascade into systemd alerts
    sys.exit(0)
