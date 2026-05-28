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


def rule_options_chain_refresh_health(findings: List[Finding]) -> None:
    """R17 (NEW-GAP-044) — chad-options-chain-refresh.service must succeed
    OR fail loudly. The refresh service persists its failure to
    runtime/options_chains_cache.json via an ``error`` field — but that
    file change alone does NOT page operators. This rule promotes any
    non-empty error field, an empty chains map, or a stale ts_utc into a
    CRITICAL Finding so the health monitor's existing Telegram pipeline
    surfaces it.

    Three branches:
      * file missing                            → INFO (bootstrap window)
      * cache file present + non-empty ``error``→ CRITICAL (loud alert)
      * cache file present + empty chains + no error → CRITICAL (degenerate)
      * cache file present + ts_utc older than 26h on a weekday → WARNING
        (the daily timer is Mon-Fri 12:30 UTC; 26h covers the gap with slack)
    """
    chain_path = RUNTIME / "options_chains_cache.json"
    if not chain_path.is_file():
        findings.append(Finding(
            rule_id="R17",
            severity="INFO",
            title="Options chain cache missing",
            description=(
                "runtime/options_chains_cache.json is absent. Expected to be "
                "written daily by chad-options-chain-refresh.service "
                "(Mon-Fri 12:30 UTC)."
            ),
            remedy_type="NOTIFY_ONLY",
            remedy_action="notify",
            evidence=f"path={chain_path} exists=False",
        ))
        return

    doc = _read_json(chain_path)
    err = doc.get("error")
    chains = doc.get("chains") if isinstance(doc.get("chains"), dict) else {}
    ts_utc_raw = str(doc.get("ts_utc") or "").strip()

    if isinstance(err, str) and err.strip():
        findings.append(Finding(
            rule_id="R17",
            severity="CRITICAL",
            title="Options chain refresh failed",
            description=(
                f"chad-options-chain-refresh.service wrote an error: {err}. "
                "Strategies that depend on the options chain (alpha_options, "
                "omega_momentum_options) cannot generate fresh signals; the "
                "Greeks gate will fall back to defaults."
            ),
            remedy_type="NOTIFY_ONLY",
            remedy_action="notify",
            evidence=f"options_chains_cache.json error={err!r} ts_utc={ts_utc_raw}",
        ))
        return

    if not chains:
        findings.append(Finding(
            rule_id="R17",
            severity="CRITICAL",
            title="Options chain cache empty",
            description=(
                "options_chains_cache.json has no chains and no error field. "
                "Either the refresh service exited 0 with no work, or the "
                "writer corrupted the cache. Either way, options strategies "
                "will not produce signals."
            ),
            remedy_type="NOTIFY_ONLY",
            remedy_action="notify",
            evidence=f"options_chains_cache.json chains_count=0 ts_utc={ts_utc_raw}",
        ))
        return

    if ts_utc_raw:
        try:
            ts = datetime.fromisoformat(ts_utc_raw.replace("Z", "+00:00"))
            age_s = (datetime.now(timezone.utc) - ts).total_seconds()
            if age_s > 26 * 3600 and is_weekday():
                findings.append(Finding(
                    rule_id="R17",
                    severity="WARNING",
                    title="Options chain cache stale",
                    description=(
                        f"options_chains_cache.json ts_utc={ts_utc_raw} is "
                        f"{age_s/3600:.1f}h old (>26h); the daily refresh "
                        "timer did not run successfully today."
                    ),
                    remedy_type="NOTIFY_ONLY",
                    remedy_action="notify",
                    evidence=f"options_chains_cache.json ts_utc={ts_utc_raw} age_s={int(age_s)}",
                ))
        except Exception:
            pass


def rule_ibkr_sustained_latency(findings: List[Finding]) -> None:
    """R19 (STOP-BUS-RECOVERY-1) — promote the sustained-latency counter to
    a loud Telegram alert.

    Reads runtime/ibkr_status.json::consecutive_cycles_above_stop_threshold
    (emitted by chad/ops/ibkr_reliability_tracker.py via the healthcheck).
    When the counter is >= the configured alert-at value (default 5), a
    CRITICAL Finding is emitted so the existing Telegram pipeline surfaces
    the pattern before the stop_bus latches.

    The publisher's own auto-recovery (clean_streak=5) is unchanged; this
    rule is observability only.
    """
    status_path = RUNTIME / "ibkr_status.json"
    if not status_path.is_file():
        return
    try:
        from chad.ops.ibkr_reliability_tracker import (
            should_alert,
            DEFAULT_ALERT_AT_CONSECUTIVE,
        )
    except Exception:
        return
    doc = _read_json(status_path)
    fire, alert = should_alert(doc, alert_at=DEFAULT_ALERT_AT_CONSECUTIVE)
    if not fire:
        return
    findings.append(Finding(
        rule_id="R19",
        severity="CRITICAL",
        title="IBKR sustained latency above stop threshold",
        description=(
            f"avg_latency_ms has been above the stop threshold for "
            f"{alert['consecutive_cycles_above_stop_threshold']} consecutive "
            f"healthcheck cycles (threshold={alert['stop_threshold_ms']:.0f}ms, "
            f"max_in_window={alert['max_latency_observed_in_window']!r}ms). "
            "The publisher's own auto-recovery (clean_streak=5) is still in "
            "effect; this finding is operator-visible defense-in-depth."
        ),
        remedy_type="NOTIFY_ONLY",
        remedy_action="notify",
        evidence=(
            "ibkr_status.json "
            f"consecutive_cycles_above_stop_threshold={alert['consecutive_cycles_above_stop_threshold']} "
            f"recovery_state={alert['current_recovery_state']} "
            f"last_above_at={alert['last_above_threshold_at']} "
            f"last_gateway_churn_at={alert['last_gateway_churn_at']}"
        ),
    ))


def rule_ibkr_gateway_version(findings: List[Finding]) -> None:
    """R20 — surface Gateway version staleness as an audit finding.

    Reads a cached ``runtime/ibkr_gateway_version.json`` if present and fresh
    (< 24h); otherwise invokes ``chad.tools.ibkr_gateway_version_check``
    programmatically against the live install tree. Severity follows the
    tool's classification:

        tool "info"    -> no finding (current; nothing to surface)
        tool "warning" -> WARNING
        tool "stale"   -> CRITICAL
        tool "unknown" -> WARNING (detection failed; operator should look)

    This rule is observability only — it triggers no remediation action.
    """
    cache_path = RUNTIME / "ibkr_gateway_version.json"
    report = None
    age = _age(cache_path)
    if age is not None and age < 86400:
        cached = _read_json(cache_path)
        if cached.get("schema_version") == "ibkr_gateway_version_check.v1":
            report = cached
    if report is None:
        try:
            from chad.tools.ibkr_gateway_version_check import build_report
            report = build_report()
        except Exception:
            return

    comp = report.get("comparison", {})
    severity = comp.get("severity", "unknown")
    installed = report.get("installed", {})
    target = report.get("target", {})

    sev_map = {
        "info": None,
        "warning": "WARNING",
        "stale": "CRITICAL",
        "unknown": "WARNING",
    }
    finding_severity = sev_map.get(severity, "WARNING")
    if finding_severity is None:
        return  # current — nothing to surface

    if severity == "unknown":
        title = "IBKR Gateway version could not be determined"
        description = (
            "chad.tools.ibkr_gateway_version_check could not derive an "
            f"installed build from {installed.get('install_path')!r} "
            f"(detection_error={installed.get('detection_error')!r}). "
            "Operator should verify the Gateway install tree."
        )
    else:
        description = (
            f"Installed IB Gateway build {installed.get('build')} "
            f"({installed.get('display')}) is behind the recommended target "
            f"build {target.get('build')} ({target.get('display')}) by "
            f"{comp.get('build_delta')} builds. Recommendation: "
            f"{report.get('recommendation')}. See "
            "ops/pending_actions/IBKR_GATEWAY_VERSION_UPGRADE_2026-05-28.md."
        )
        title = "IBKR Gateway version is stale" if severity == "stale" \
            else "IBKR Gateway version upgrade recommended"

    findings.append(Finding(
        rule_id="R20",
        severity=finding_severity,
        title=title,
        description=description,
        remedy_type="NOTIFY_ONLY",
        remedy_action="notify",
        evidence=(
            f"installed_build={installed.get('build')} "
            f"display={installed.get('display')} "
            f"detection_source={installed.get('detection_source')} "
            f"install_path={installed.get('install_path')} "
            f"target_build={target.get('build')} severity={severity}"
        ),
    ))


def rule_options_chain_refresh_failure_artifact(findings: List[Finding]) -> None:
    """R17b (OPTIONS-CHAIN-1) — read the dedicated failure artifact emitted by
    chad-options-chain-refresh.service (``_write_failure_artifact`` writes
    ``runtime/options_chain_refresh_failure.json``). A *fresh* failure
    artefact promotes to CRITICAL even when the cache itself is benign
    (e.g. the cache may still hold yesterday's content while today's refresh
    explicitly failed).

      * file missing                                   → silent (no finding)
      * artefact present + age <= 2h                   → CRITICAL
      * artefact present + age > 2h                    → WARNING (stale failure;
        a fresh successful refresh would have cleared it)
    """
    try:
        from chad.market_data.options_chain_freshness import (
            is_failure_artifact_fresh,
        )
    except Exception:
        return
    fresh, details = is_failure_artifact_fresh()
    if not details["failure_artifact_exists"]:
        return
    reason = details.get("failure_artifact_reason") or "unknown"
    age_s = details.get("failure_artifact_age_seconds")
    age_label = f"{int(age_s)}s" if isinstance(age_s, (int, float)) else "unknown_age"
    if fresh:
        findings.append(Finding(
            rule_id="R17b",
            severity="CRITICAL",
            title="Options chain refresh failure artifact is fresh",
            description=(
                "runtime/options_chain_refresh_failure.json is present and "
                f"recent (age={age_label}; blocked_reason={reason}). The most "
                "recent refresh attempt failed; options-routed strategies "
                "must fail-closed via chain_usability() until the next "
                "successful refresh."
            ),
            remedy_type="NOTIFY_ONLY",
            remedy_action="notify",
            evidence=(
                "options_chain_refresh_failure.json "
                f"age_s={age_label} reason={reason}"
            ),
        ))
    else:
        findings.append(Finding(
            rule_id="R17b",
            severity="WARNING",
            title="Options chain refresh failure artifact is stale",
            description=(
                "runtime/options_chain_refresh_failure.json exists but its "
                f"ts_utc is older than 2h (age={age_label}). Either a fresh "
                "successful refresh has not run yet to clear it, or the "
                "refresh writer is no longer emitting failure artefacts on "
                "success."
            ),
            remedy_type="NOTIFY_ONLY",
            remedy_action="notify",
            evidence=(
                "options_chain_refresh_failure.json "
                f"age_s={age_label} reason={reason}"
            ),
        ))


def rule_options_greeks_freshness(findings: List[Finding]) -> None:
    """R18 (NEW-GAP-044) — companion rule that flags silently-stale Greeks.

    runtime/options_greeks.json carries its own ``ttl_seconds`` (default
    90000 ≈ 25h) and a ``source.chain_cache_ts_utc`` lineage. The
    options_greeks_gate fall-back-to-default behaviour is failure-soft on
    purpose; this rule re-surfaces the failure so it isn't invisible:

      * file missing                            → CRITICAL
      * doc ts_utc older than ttl_seconds        → CRITICAL (stale Greeks)
      * status field == "partial" or "failed"    → WARNING
      * symbols map empty                        → WARNING (no usable greeks)
    """
    greeks_path = RUNTIME / "options_greeks.json"
    if not greeks_path.is_file():
        findings.append(Finding(
            rule_id="R18",
            severity="CRITICAL",
            title="Options Greeks file missing",
            description=(
                "runtime/options_greeks.json is absent. options_greeks_gate "
                "will fall back to default delta=±0.5 for every options "
                "lookup."
            ),
            remedy_type="NOTIFY_ONLY",
            remedy_action="notify",
            evidence=f"path={greeks_path} exists=False",
        ))
        return

    doc = _read_json(greeks_path)
    ts_utc_raw = str(doc.get("ts_utc") or "").strip()
    ttl_raw = doc.get("ttl_seconds")
    try:
        ttl_s = int(ttl_raw) if ttl_raw is not None else 90000
        if ttl_s <= 0:
            ttl_s = 90000
    except (TypeError, ValueError):
        ttl_s = 90000

    if ts_utc_raw:
        try:
            ts = datetime.fromisoformat(ts_utc_raw.replace("Z", "+00:00"))
            age_s = (datetime.now(timezone.utc) - ts).total_seconds()
            if age_s > ttl_s:
                findings.append(Finding(
                    rule_id="R18",
                    severity="CRITICAL",
                    title="Options Greeks stale beyond TTL",
                    description=(
                        f"options_greeks.json ts_utc={ts_utc_raw} is "
                        f"{age_s/3600:.1f}h old, exceeding declared "
                        f"ttl_seconds={ttl_s}. options_greeks_gate is now "
                        "silently returning default deltas for every "
                        "lookup."
                    ),
                    remedy_type="NOTIFY_ONLY",
                    remedy_action="notify",
                    evidence=f"options_greeks.json age_s={int(age_s)} ttl_s={ttl_s}",
                ))
                return  # don't double-fire status/symbol warnings on stale.
        except Exception:
            pass

    status = str(doc.get("status") or "").strip().lower()
    if status in ("failed", "error"):
        findings.append(Finding(
            rule_id="R18",
            severity="CRITICAL",
            title=f"Options Greeks publisher reported status={status}",
            description=(
                "options_greeks.json::status reflects a failed publisher run. "
                "Strategies that consult the Greeks gate will receive "
                "default deltas with source='default' instead of computed "
                "values."
            ),
            remedy_type="NOTIFY_ONLY",
            remedy_action="notify",
            evidence=f"options_greeks.json status={status}",
        ))
        return

    if status == "partial":
        findings.append(Finding(
            rule_id="R18",
            severity="WARNING",
            title="Options Greeks publisher partial",
            description=(
                "options_greeks.json::status=partial — at least one expiry or "
                "strike could not be priced. Inspect source.provider_status."
            ),
            remedy_type="NOTIFY_ONLY",
            remedy_action="notify",
            evidence=f"options_greeks.json status=partial",
        ))

    symbols = doc.get("symbols")
    if isinstance(symbols, dict) and not symbols:
        findings.append(Finding(
            rule_id="R18",
            severity="WARNING",
            title="Options Greeks symbols map empty",
            description=(
                "options_greeks.json::symbols is {}. No per-symbol greeks "
                "are available; gate will return defaults for every lookup."
            ),
            remedy_type="NOTIFY_ONLY",
            remedy_action="notify",
            evidence=f"options_greeks.json symbols_count=0",
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
        rule_options_chain_refresh_health,
        rule_options_chain_refresh_failure_artifact,
        rule_options_greeks_freshness,
        rule_ibkr_sustained_latency,
        rule_ibkr_gateway_version,
    ]:
        try:
            fn(findings)
        except Exception as e:
            pass  # rule failure never crashes the monitor
    return findings
