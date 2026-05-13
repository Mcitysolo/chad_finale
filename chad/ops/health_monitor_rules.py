"""
CHAD Health Monitor — Rule Engine (Tier 1)
Fast mechanical checks. No API call. Runs every cycle.
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path("/home/ubuntu/chad_finale")
RUNTIME = REPO_ROOT / "runtime"
DATA = REPO_ROOT / "data"


def is_weekday() -> bool:
    """True Mon–Fri (UTC); False Sat/Sun. Used to gate equity-only rules."""
    return datetime.now(timezone.utc).weekday() < 5

# Feed → publisher service mapping (mirrors remediation.FEED_PUBLISHER_MAP).
# SS01: regime_state and the orchestrator-published feed (dynamic_caps) are
# served by trading engines; their staleness must NEVER trigger an
# auto-restart of the trading engine itself. The remediation map below is
# only consulted for feeds whose publisher is a side-car timer/service.
_FEED_PUBLISHER_MAP = {
    "price_cache.json": "chad-ibkr-price-refresh.timer",
    "regime_state.json": "chad-regime-classifier-refresh.timer",
    "dynamic_caps.json": "chad-orchestrator.service",
    "regime_booster.json": "chad-regime-booster.timer",
    "kraken_prices.json": "chad-kraken-ws.service",
    "reconciliation_state.json": "chad-reconciliation-publisher.timer",
    "choppy_regime_state.json": "chad-choppy-regime.timer",
    "macro_state.json": "chad-macro-state.timer",
}

# SS01: feeds whose staleness must degrade to NOTIFY_ONLY rather than a
# service restart, because the only available publisher is a trading engine.
_FEED_NOTIFY_ONLY = {
    "regime_state.json",
}

@dataclass
class Finding:
    rule_id: str
    severity: str  # CRITICAL / WARNING / INFO
    title: str
    description: str
    remedy_type: str  # SAFE_AUTO / CODE_CHANGE / SERVICE_RESTART / NOTIFY_ONLY
    remedy_action: str  # what to do
    remedy_args: dict = field(default_factory=dict)
    evidence: str = ""

def _age(path: Path) -> Optional[float]:
    """Return file age in seconds, or None if missing."""
    try:
        return time.time() - path.stat().st_mtime
    except Exception:
        return None

def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _service_active(name: str) -> bool:
    try:
        r = subprocess.run(
            ["systemctl", "is-active", name],
            capture_output=True, text=True, timeout=5
        )
        return r.stdout.strip() == "active"
    except Exception:
        return False

def _service_failed(name: str) -> bool:
    try:
        r = subprocess.run(
            ["systemctl", "is-failed", name],
            capture_output=True, text=True, timeout=5
        )
        return r.stdout.strip() == "failed"
    except Exception:
        return False


# ── Rule definitions ──────────────────────────────────────────────────────────

def rule_critical_services(findings: List[Finding]) -> None:
    """R01 — Critical services must be active."""
    critical = [
        "chad-live-loop.service",
        "chad-orchestrator.service",
        "chad-ibgateway.service",
        "chad-kraken-ws.service",
        "chad-telegram-bot.service",
        "chad-dashboard.service",
        "chad-backend.service",
    ]
    for svc in critical:
        if _service_failed(svc):
            findings.append(Finding(
                rule_id="R01",
                severity="CRITICAL",
                title=f"Service FAILED: {svc}",
                description=f"{svc} is in failed state and needs restart.",
                remedy_type="SERVICE_RESTART",
                remedy_action="restart_service",
                remedy_args={"service": svc},
                evidence=f"systemctl is-failed {svc} = failed",
            ))
        elif not _service_active(svc):
            findings.append(Finding(
                rule_id="R01",
                severity="CRITICAL",
                title=f"Service INACTIVE: {svc}",
                description=f"{svc} is not active.",
                remedy_type="SERVICE_RESTART",
                remedy_action="restart_service",
                remedy_args={"service": svc},
                evidence=f"systemctl is-active {svc} = inactive",
            ))


def rule_feed_freshness(findings: List[Finding]) -> None:
    """R02 — Critical feeds must be within TTL."""
    feeds = [
        ("price_cache.json", 180),
        ("regime_state.json", 360),
        ("dynamic_caps.json", 180),
        ("regime_booster.json", 240),
        ("kraken_prices.json", 120),
        ("reconciliation_state.json", 480),
        ("choppy_regime_state.json", 900),
        ("macro_state.json", 7200),
    ]
    for fname, ttl in feeds:
        age = _age(RUNTIME / fname)
        if age is None:
            findings.append(Finding(
                rule_id="R02",
                severity="WARNING",
                title=f"Feed MISSING: {fname}",
                description=f"Runtime file {fname} does not exist.",
                remedy_type="NOTIFY_ONLY",
                remedy_action="notify",
                evidence=f"{fname} not found in {RUNTIME}",
            ))
        elif age > ttl * 2:
            svc = _FEED_PUBLISHER_MAP.get(fname, "")
            # SS01: regime_state staleness alerts but never triggers a
            # trading-engine restart. Operator must investigate manually.
            if fname in _FEED_NOTIFY_ONLY:
                findings.append(Finding(
                    rule_id="R02",
                    severity="WARNING",
                    title=f"Feed STALE (notify): {fname} ({int(age)}s old, TTL={ttl}s)",
                    description=(
                        f"{fname} is {int(age)}s old — more than 2× TTL. "
                        "Auto-restart suppressed: publisher is a trading "
                        "engine. Operator investigation required."
                    ),
                    remedy_type="NOTIFY_ONLY",
                    remedy_action="notify",
                    evidence=f"mtime age={int(age)}s TTL={ttl}s feed={fname}",
                ))
            else:
                findings.append(Finding(
                    rule_id="R02",
                    severity="CRITICAL",
                    title=f"Feed STALE: {fname} ({int(age)}s old, TTL={ttl}s)",
                    description=f"{fname} is {int(age)}s old — more than 2× TTL.",
                    remedy_type="SERVICE_RESTART",
                    remedy_action="restart_feed_publisher",
                    remedy_args={"feed": fname, "age": age, "ttl": ttl,
                                 "service": svc},
                    evidence=f"mtime age={int(age)}s TTL={ttl}s service={svc}",
                ))


def rule_scr_state(findings: List[Finding]) -> None:
    """R03 — SCR must not be PAUSED."""
    d = _read_json(RUNTIME / "scr_state.json")
    state = str(d.get("state", "")).upper()
    if state == "PAUSED":
        findings.append(Finding(
            rule_id="R03",
            severity="CRITICAL",
            title="SCR is PAUSED — trading halted",
            description="SCR state machine is PAUSED. Sizing factor=0. No trades executing.",
            remedy_type="NOTIFY_ONLY",
            remedy_action="notify",
            evidence=f"scr_state.json state={state}",
        ))


def rule_stop_bus(findings: List[Finding]) -> None:
    """R04 — Stop bus must not be active."""
    d = _read_json(RUNTIME / "stop_bus.json")
    if d.get("active"):
        findings.append(Finding(
            rule_id="R04",
            severity="CRITICAL",
            title="STOP BUS ACTIVE — all trading halted",
            description=f"Stop bus is active. Reason: {d.get('reason', 'unknown')}",
            remedy_type="NOTIFY_ONLY",
            remedy_action="notify",
            evidence=f"stop_bus.json active=true reason={d.get('reason')}",
        ))


def rule_reconciliation(findings: List[Finding]) -> None:
    """R05 — Reconciliation must be GREEN."""
    d = _read_json(RUNTIME / "reconciliation_state.json")
    status = str(d.get("status", "")).upper()
    if status == "RED":
        findings.append(Finding(
            rule_id="R05",
            severity="CRITICAL",
            title="Reconciliation RED — position mismatch",
            description=f"Reconciliation is RED. worst_diff={d.get('worst_diff')} mismatches={d.get('mismatches')}",
            remedy_type="NOTIFY_ONLY",
            remedy_action="notify",
            evidence=f"reconciliation_state.json status=RED",
        ))


def rule_profit_lock(findings: List[Finding]) -> None:
    """R06 — Alert on profit lock escalation."""
    d = _read_json(RUNTIME / "profit_lock_state.json")
    mode = str(d.get("mode", "NORMAL")).upper()
    if mode in ("LOCK2", "LOCK3", "HARD_STOP"):
        findings.append(Finding(
            rule_id="R06",
            severity="CRITICAL",
            title=f"Profit Lock: {mode}",
            description=f"Profit lock in {mode} — sizing heavily restricted.",
            remedy_type="NOTIFY_ONLY",
            remedy_action="notify",
            evidence=f"profit_lock_state.json mode={mode}",
        ))


def rule_disk_usage(findings: List[Finding]) -> None:
    """R07 — Disk usage must stay below 75%."""
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        pct = used / total * 100
        if pct > 85:
            findings.append(Finding(
                rule_id="R07",
                severity="CRITICAL",
                title=f"Disk {pct:.1f}% full — CRITICAL",
                description="Disk usage above 85%. System may fail to write runtime files.",
                remedy_type="SAFE_AUTO",
                remedy_action="archive_old_fills",
                remedy_args={"current_pct": pct},
                evidence=f"disk usage={pct:.1f}%",
            ))
        elif pct > 75:
            findings.append(Finding(
                rule_id="R07",
                severity="WARNING",
                title=f"Disk {pct:.1f}% full — WARNING",
                description="Disk usage above 75%. Consider archiving old fill files.",
                remedy_type="SAFE_AUTO",
                remedy_action="archive_old_fills",
                remedy_args={"current_pct": pct},
                evidence=f"disk usage={pct:.1f}%",
            ))
    except Exception:
        pass


def rule_corrupt_runtime_files(findings: List[Finding]) -> None:
    """R08 — Runtime JSON files must not be zero-byte or corrupt."""
    critical_files = [
        "position_guard.json",
        "dynamic_caps.json",
        "scr_state.json",
        "profit_lock_state.json",
    ]
    for fname in critical_files:
        p = RUNTIME / fname
        if not p.exists():
            continue
        if p.stat().st_size == 0:
            findings.append(Finding(
                rule_id="R08",
                severity="CRITICAL",
                title=f"Zero-byte runtime file: {fname}",
                description=f"{fname} is 0 bytes — likely corrupted mid-write.",
                remedy_type="SAFE_AUTO",
                remedy_action="restore_from_backup",
                remedy_args={"file": fname},
                evidence=f"{fname} size=0",
            ))
            continue
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            findings.append(Finding(
                rule_id="R08",
                severity="CRITICAL",
                title=f"Corrupt runtime file: {fname}",
                description=f"{fname} failed JSON parse: {e}",
                remedy_type="SAFE_AUTO",
                remedy_action="restore_from_backup",
                remedy_args={"file": fname},
                evidence=str(e),
            ))


def rule_edge_decay_halts(findings: List[Finding]) -> None:
    """R09 — Notify on any active edge decay halts."""
    p = RUNTIME / "strategy_allocations.json"
    if not p.exists():
        return
    d = _read_json(p)
    allocations = d.get("allocations", {}) if isinstance(d, dict) else {}
    if not isinstance(allocations, dict):
        return
    for strategy, info in allocations.items():
        if isinstance(info, dict) and info.get("halted"):
            findings.append(Finding(
                rule_id="R09",
                severity="WARNING",
                title=f"Edge decay halt active: {strategy}",
                description=f"Strategy {strategy} is halted by edge decay monitor.",
                remedy_type="NOTIFY_ONLY",
                remedy_action="notify",
                evidence=f"strategy_allocations.json allocations.{strategy}.halted=true",
            ))


def rule_high_trade_churn(findings: List[Finding]) -> None:
    """R10 — Detect high trade churn with negative PnL. Weekday-only:
    weekend churn signal is meaningless when equity markets are closed."""
    if not is_weekday():
        return
    d = _read_json(RUNTIME / "pnl_state.json")
    if not d:
        return
    trade_count = int(d.get("trade_count", 0) or 0)
    realized_pnl = float(d.get("realized_pnl", 0.0) or 0.0)
    if trade_count > 300 and realized_pnl < -500:
        findings.append(Finding(
            rule_id="R10",
            severity="CRITICAL",
            title=f"High churn: {trade_count} trades, PnL=${realized_pnl:.2f}",
            description=(
                f"{trade_count} trades today with ${realized_pnl:.2f} realized. "
                "Strategies are churning losses. Signal throttle recommended."
            ),
            remedy_type="SAFE_AUTO",
            remedy_action="write_signal_throttle",
            remedy_args={"trade_count": trade_count, "pnl": realized_pnl},
            evidence="runtime/pnl_state.json",
        ))


def rule_stale_reconciliation_artifact(findings: List[Finding]) -> None:
    """R11 — Detect stale reconciliation strategy entries with penalties."""
    p = RUNTIME / "winner_scaling.json"
    if not p.exists():
        return
    d = _read_json(p)
    multipliers = d.get("multipliers", {})
    now = datetime.now(timezone.utc)
    for strategy, mult in multipliers.items():
        try:
            mult_f = float(mult)
        except Exception:
            continue
        name_lower = strategy.lower()
        if "reconciled" not in name_lower and "reconciled_phase" not in name_lower:
            continue
        if mult_f >= 1.0:
            continue
        # Parse YYYYMMDD date suffix from the name to estimate age
        m = re.search(r"(\d{8})", strategy)
        if not m:
            continue
        try:
            dt = datetime.strptime(m.group(1), "%Y%m%d").replace(tzinfo=timezone.utc)
        except Exception:
            continue
        age_days = (now - dt).total_seconds() / 86400.0
        if age_days <= 7:
            continue
        findings.append(Finding(
            rule_id="R11",
            severity="WARNING",
            title=f"Stale reconciliation artifact: {strategy}",
            description=f"Strategy {strategy} has been penalized for {int(age_days)}+ days.",
            remedy_type="SAFE_AUTO",
            remedy_action="clear_reconciliation_artifact",
            remedy_args={"strategy": strategy},
            evidence="runtime/winner_scaling.json",
        ))


def rule_alpha_cluster_degradation(findings: List[Finding]) -> None:
    """R12 — Detect correlated degradation across alpha-cluster strategies.
    Weekday-only: equity alpha strategies do not trade on weekends, so
    health scores are stale by construction."""
    if not is_weekday():
        return
    p = RUNTIME / "strategy_health.json"
    if not p.exists():
        return
    d = _read_json(p)
    strats = d.get("strategies", {}) or {}
    low_alphas = []
    for name, info in strats.items():
        if not isinstance(info, dict):
            continue
        if "alpha" not in name.lower():
            continue
        score = info.get("health_score")
        sample = int(
            info.get("sample_count",
            info.get("trade_count", 0)) or 0
        )
        if score is None or sample < 10:
            continue
        try:
            if float(score) < 0.5:
                low_alphas.append((name, float(score)))
        except Exception:
            continue
    if len(low_alphas) >= 3:
        details = ", ".join(f"{n}={s:.2f}" for n, s in sorted(low_alphas))
        findings.append(Finding(
            rule_id="R12",
            severity="WARNING",
            title="Alpha cluster correlated degradation",
            description=(
                "Multiple alpha strategies simultaneously below health "
                "threshold — shared signal or data dependency issue. "
                f"Affected: {details}"
            ),
            remedy_type="NOTIFY_ONLY",
            remedy_action="notify",
            evidence="runtime/strategy_health.json",
        ))


def rule_scr_effective_trades_gap(findings: List[Finding]) -> None:
    """R13 — Flag large gaps between raw trade count and SCR effective trades.

    Surfaces the SCR exclusion classification (excluded_untrusted,
    excluded_manual, excluded_nonfinite) when present so the operator can
    confirm the gap is composed of legitimate exclusions (pnl_untrusted,
    rejected/partial fills, manual exclusions) rather than a counting bug.
    """
    pnl = _read_json(RUNTIME / "pnl_state.json")
    scr = _read_json(RUNTIME / "scr_state.json")
    if not pnl or not scr:
        return
    raw_trade_count = int(pnl.get("trade_count", 0) or 0)
    stats = scr.get("stats", {}) or {}
    effective = int(stats.get("effective_trades", 0) or 0)
    if raw_trade_count <= 0 or effective <= 0:
        return
    if raw_trade_count > effective * 2:
        gap = raw_trade_count - effective
        # Surface classification breakdown so the gap can be triaged
        # without manual log inspection.
        cls_keys = (
            "excluded_untrusted",
            "excluded_manual",
            "excluded_nonfinite",
            "excluded_partial",
            "excluded_pnl_zero",
        )
        cls_parts = []
        accounted = 0
        for k in cls_keys:
            try:
                v = int(stats.get(k, 0) or 0)
            except Exception:
                continue
            if v > 0:
                cls_parts.append(f"{k}={v}")
                accounted += v
        unaccounted = max(gap - accounted, 0)
        if unaccounted > 0:
            cls_parts.append(f"unclassified={unaccounted}")
        cls_summary = ", ".join(cls_parts) if cls_parts else "no breakdown"
        findings.append(Finding(
            rule_id="R13",
            severity="INFO",
            title=f"SCR gap: {raw_trade_count} raw vs {effective} effective",
            description=(
                f"{gap} trades excluded from SCR. "
                f"Classification: {cls_summary}. "
                "Check for rejected/partial/excluded fills."
            ),
            remedy_type="NOTIFY_ONLY",
            remedy_action="notify",
            evidence="runtime/scr_state.json + runtime/pnl_state.json",
        ))


def rule_halted_with_unclamped_boost(findings: List[Finding]) -> None:
    """R14 — Halted strategy must have its winner boost clamped.

    Reads the *effective* per-strategy ``winner_factor`` from
    ``dynamic_caps.json`` (the live, clamped value the allocator uses)
    rather than the raw multiplier from ``winner_scaling.json``. The
    orchestrator's halt clamp publishes ``winner_factor<=1.0`` with
    ``halt_clamp_applied=true`` for any halted strategy, so a raw boost
    of 1.5x in winner_scaling.json is *not* a contradiction once the
    effective value is neutral.

    Warns only when:
      - ``dynamic_caps.json`` is missing/empty (cannot verify clamp), OR
      - effective ``winner_factor`` for a halted strategy is > 1.0.
    Stays silent when the clamp is correctly applied.
    """
    alloc = _read_json(RUNTIME / "strategy_allocations.json")
    allocations = alloc.get("allocations", {}) if isinstance(alloc, dict) else {}
    if not isinstance(allocations, dict):
        return
    halted = [
        name for name, info in allocations.items()
        if isinstance(info, dict) and info.get("halted")
    ]
    if not halted:
        return

    caps = _read_json(RUNTIME / "dynamic_caps.json")
    caps_strats = (caps.get("strategies") if isinstance(caps, dict) else None) or {}
    raw_ws = _read_json(RUNTIME / "winner_scaling.json")
    raw_mults = (raw_ws.get("multipliers") if isinstance(raw_ws, dict) else None) or {}

    for strategy in halted:
        try:
            raw_mult = float(raw_mults.get(strategy, 1.0))
        except Exception:
            raw_mult = 1.0
        # Only audit halted strategies that actually carry a raw boost.
        if raw_mult <= 1.0:
            continue

        cap_entry = caps_strats.get(strategy) if isinstance(caps_strats, dict) else None
        if not isinstance(cap_entry, dict) or not caps_strats:
            findings.append(Finding(
                rule_id="R14",
                severity="WARNING",
                title=f"Halt+boost unverified: {strategy}",
                description=(
                    f"Strategy {strategy} is halted with raw winner boost "
                    f"{raw_mult:.2f}x but dynamic_caps.json is missing or "
                    "has no per-strategy entry — cannot verify the halt "
                    "clamp suppressed the boost."
                ),
                remedy_type="NOTIFY_ONLY",
                remedy_action="notify",
                evidence=(
                    f"strategy_allocations.{strategy}.halted=true; "
                    f"winner_scaling.multipliers.{strategy}={raw_mult}; "
                    "dynamic_caps.json missing per-strategy data"
                ),
            ))
            continue

        try:
            eff = float(cap_entry.get("winner_factor", 1.0))
        except Exception:
            eff = 1.0
        clamp_applied = bool(cap_entry.get("halt_clamp_applied", False))
        if eff > 1.0:
            findings.append(Finding(
                rule_id="R14",
                severity="WARNING",
                title=f"Halt+boost contradiction: {strategy}",
                description=(
                    f"Strategy {strategy} is halted but effective "
                    f"winner_factor={eff:.2f}x in dynamic_caps "
                    f"(halt_clamp_applied={clamp_applied}). The boost "
                    "should be clamped to <=1.0 while halted."
                ),
                remedy_type="NOTIFY_ONLY",
                remedy_action="notify",
                evidence=(
                    f"strategy_allocations.{strategy}.halted=true; "
                    f"dynamic_caps.strategies.{strategy}.winner_factor={eff}; "
                    f"halt_clamp_applied={clamp_applied}"
                ),
            ))
        # eff <= 1.0 → halt clamp working as designed; no finding emitted.


def rule_tier_daily_loss_approaching(findings: List[Finding]) -> None:
    """R15 — Daily-loss budget approaching tier limit (MICRO / STARTER).

    Fires when the tier-enforced daily loss budget has burned more than 70%
    of the tier ceiling (i.e. budget_remaining_today_usd is below 30% of
    max_daily_loss_usd). NOTIFY only — no kill-switch and no position
    action; the tier_risk_enforcer remains the sole authority for hard
    stops.
    """
    state = _read_json(RUNTIME / "tier_enforcement_state.json")
    if not state:
        return
    if bool(state.get("daily_loss_limit_hit", False)):
        return
    tier = str(state.get("tier", "") or "").upper()
    if tier not in ("MICRO", "STARTER"):
        return
    max_daily_loss = state.get("max_daily_loss_usd")
    if max_daily_loss is None:
        return
    try:
        max_daily_loss_f = float(max_daily_loss)
    except (TypeError, ValueError):
        return
    if max_daily_loss_f <= 0:
        return
    try:
        budget_remaining = float(state.get("budget_remaining_today_usd", 0.0) or 0.0)
    except (TypeError, ValueError):
        return
    threshold = max_daily_loss_f * 0.30
    if budget_remaining >= threshold:
        return
    findings.append(Finding(
        rule_id="R15",
        severity="WARNING",
        title=f"Daily loss budget approaching limit — tier {tier}",
        description=(
            f"Daily loss budget {budget_remaining:.2f} remaining "
            f"(30% threshold hit) — tier {tier}"
        ),
        remedy_type="NOTIFY_ONLY",
        remedy_action="notify",
        evidence=(
            f"tier_enforcement_state.json tier={tier} "
            f"budget_remaining={budget_remaining:.2f} "
            f"max_daily_loss={max_daily_loss_f:.2f}"
        ),
    ))


def rule_setup_family_skip_rate(findings: List[Finding]) -> None:
    """R16 — Setup family skipping too many entries via stop-too-wide gate.

    Looks at runtime/setup_family_expectancy.json and flags any family
    where skip_count_stop_too_wide is more than 2× the realised trade
    count (with a sample-size floor of 5 trades to avoid noise). Indicates
    the active tier's stop budget may be too tight for the family.
    NOTIFY only.
    """
    expectancy = _read_json(RUNTIME / "setup_family_expectancy.json")
    if not isinstance(expectancy, dict):
        return
    families = expectancy.get("families")
    if isinstance(families, dict):
        family_iter = families.items()
    elif isinstance(expectancy.get("setup_families"), dict):
        family_iter = expectancy["setup_families"].items()
    else:
        # Fall back to scanning top-level dict entries that look like
        # per-family records.
        family_iter = (
            (k, v) for k, v in expectancy.items()
            if isinstance(v, dict) and (
                "trades" in v
                or "skip_count_stop_too_wide" in v
            )
        )
    for family, info in family_iter:
        if not isinstance(info, dict):
            continue
        try:
            trades = int(info.get("trades", 0) or 0)
        except (TypeError, ValueError):
            continue
        try:
            skip_count = int(info.get("skip_count_stop_too_wide", 0) or 0)
        except (TypeError, ValueError):
            continue
        if trades < 5:
            continue
        if skip_count <= trades * 2:
            continue
        findings.append(Finding(
            rule_id="R16",
            severity="WARNING",
            title=f"Setup family skip rate high: {family}",
            description=(
                f"Setup family {family} skip rate high: {skip_count} "
                f"skips vs {trades} trades — stop budget may be too "
                "tight for active tier"
            ),
            remedy_type="NOTIFY_ONLY",
            remedy_action="notify",
            evidence=(
                f"setup_family_expectancy.{family} "
                f"skip_count_stop_too_wide={skip_count} trades={trades}"
            ),
        ))


def run_all_rules() -> List[Finding]:
    """Run all rules and return list of findings."""
    findings: List[Finding] = []
    for fn in [
        rule_critical_services,
        rule_feed_freshness,
        rule_scr_state,
        rule_stop_bus,
        rule_reconciliation,
        rule_profit_lock,
        rule_disk_usage,
        rule_corrupt_runtime_files,
        rule_edge_decay_halts,
        rule_high_trade_churn,
        rule_stale_reconciliation_artifact,
        rule_alpha_cluster_degradation,
        rule_scr_effective_trades_gap,
        rule_halted_with_unclamped_boost,
        rule_tier_daily_loss_approaching,
        rule_setup_family_skip_rate,
    ]:
        try:
            fn(findings)
        except Exception as e:
            pass  # rule failure never crashes the monitor
    return findings
