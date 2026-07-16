"""Exterminator Read-Only Sentinel — Stage 1.

SSOT v9.0 §7. Pure read-only scanner. Detects 18 anomaly categories and
emits a single timestamped JSON+Markdown report pair under
``reports/exterminator/``. Does not mutate runtime, does not call brokers,
does not restart services, does not stage or commit. Tests lock these
contracts.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

# ---------------------------------------------------------------------------
# Constants (no I/O performed at import time)
# ---------------------------------------------------------------------------

DEFAULT_REPO_ROOT = Path("/home/ubuntu/chad_finale")
SCHEMA_VERSION = "exterminator_report.v1"

CORE_RUNTIME_FILES: tuple[str, ...] = (
    "scr_state.json",
    "live_readiness.json",
    "epoch_state.json",
    "profit_lock_state.json",
    "stop_bus.json",
    "reconciliation_state.json",
    "regime_state.json",
    "portfolio_snapshot.json",
    "strategy_health.json",
    "winner_scaling.json",
)

TTL_RUNTIME_FILES: tuple[str, ...] = (
    "scr_state.json",
    "profit_lock_state.json",
    "reconciliation_state.json",
    "regime_state.json",
    "choppy_regime_state.json",
    "macro_state.json",
    "event_risk.json",
    # IR1 R3: the advisory-refresh liveness marker (ts_utc + ttl_seconds=1800).
    # NB: we watch intel_refresh_state.json, NOT strategy_intelligence_cache.json
    # — the cache is rewritten with a fresh mtime even on total advisory failure
    # (neutral fallback) and carries no ts_utc/ttl_seconds, so it can never go
    # "stale" and is useless as a sentinel. The state marker DOES go stale if the
    # refresh timer itself dies, which is the failure worth catching here.
    "intel_refresh_state.json",
)

SCHEMA_REQUIRED_FILES: tuple[str, ...] = tuple(
    sorted(
        set(CORE_RUNTIME_FILES)
        | {
            "choppy_regime_state.json",
            "macro_state.json",
            "event_risk.json",
            "withdrawal_authorization.json",
        }
    )
)

PLACEHOLDER_PRICE = 100.0
SUBPROCESS_TIMEOUT_SECONDS = 15

SEVERITY_INFO = "INFO"
SEVERITY_WARNING = "WARNING"
SEVERITY_CRITICAL = "CRITICAL"

# ---------------------------------------------------------------------------
# Finding model
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    id: str
    severity: str
    category: str
    title: str
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)
    auto_fix_allowed: bool = False
    recommended_next_action: str = ""
    requires_operator: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "severity": self.severity,
            "category": self.category,
            "title": self.title,
            "summary": self.summary,
            "evidence": self.evidence,
            "auto_fix_allowed": self.auto_fix_allowed,
            "recommended_next_action": self.recommended_next_action,
            "requires_operator": self.requires_operator,
        }


# ---------------------------------------------------------------------------
# Default subprocess providers (read-only, bounded)
# ---------------------------------------------------------------------------


def default_git_provider(repo_root: Path) -> dict[str, Any]:
    """Return git status without mutating anything. Subprocess is read-only."""
    out: dict[str, Any] = {
        "head": "",
        "branch": "",
        "clean": True,
        "dirty_files": [],
        "tags": [],
        "error": None,
    }
    try:
        out["head"] = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SECONDS, check=False,
        ).stdout.strip()
        out["branch"] = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SECONDS, check=False,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SECONDS, check=False,
        ).stdout
        dirty = [ln for ln in status.splitlines() if ln.strip()]
        out["dirty_files"] = dirty
        out["clean"] = not dirty
        tags_proc = subprocess.run(
            ["git", "-C", str(repo_root), "tag", "--points-at", "HEAD"],
            capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SECONDS, check=False,
        )
        out["tags"] = [t for t in tags_proc.stdout.splitlines() if t.strip()]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as err:
        out["error"] = repr(err)
    return out


def default_systemctl_provider() -> dict[str, Any]:
    """Return list of failed CHAD systemd units. Read-only."""
    out: dict[str, Any] = {"failed_units": [], "error": None}
    try:
        result = subprocess.run(
            ["systemctl", "--failed", "--no-pager", "--no-legend"],
            capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SECONDS, check=False,
        )
        for line in result.stdout.splitlines():
            tokens = line.split()
            if tokens and tokens[0].startswith("chad-") and tokens[0].endswith(".service"):
                out["failed_units"].append(tokens[0])
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as err:
        out["error"] = repr(err)
    return out


def default_fills_provider(runtime_dir: Path) -> dict[str, Any]:
    """Return summary of recent paper fills if a discoverable artifact exists.

    Stage 1 is read-only and does not parse the executor SQLite. If the runtime
    publishes a fills NDJSON or a paper-fill audit JSON we surface it; otherwise
    we return ``available=False`` so the placeholder check downgrades to INFO.
    """
    candidates = (
        runtime_dir / "paper_fills_today.ndjson",
        runtime_dir / "paper_fills_recent.json",
    )
    for path in candidates:
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        rows: list[dict[str, Any]] = []
        if path.suffix == ".ndjson":
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        else:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, list):
                rows = [r for r in payload if isinstance(r, dict)]
            elif isinstance(payload, dict) and isinstance(payload.get("fills"), list):
                rows = [r for r in payload["fills"] if isinstance(r, dict)]
        return {"available": True, "source": str(path), "fills": rows}
    return {"available": False, "source": None, "fills": []}


def default_log_scan_provider(repo_root: Path, pattern: str, max_files: int = 5) -> dict[str, Any]:
    """Count occurrences of ``pattern`` across recent log files. Read-only."""
    out: dict[str, Any] = {"matches": 0, "files_scanned": 0, "samples": []}
    log_dir = repo_root / "logs"
    if not log_dir.exists():
        return out
    try:
        files = sorted(
            (p for p in log_dir.rglob("*.log") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:max_files]
    except OSError:
        return out
    rx = re.compile(pattern)
    for path in files:
        out["files_scanned"] += 1
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if rx.search(line):
                        out["matches"] += 1
                        if len(out["samples"]) < 3:
                            out["samples"].append(line.strip()[:240])
        except OSError:
            continue
    return out


# ---------------------------------------------------------------------------
# Exterminator
# ---------------------------------------------------------------------------


class Exterminator:
    """Stage 1 read-only sentinel. All mutation paths are forbidden."""

    def __init__(
        self,
        repo_root: Path = DEFAULT_REPO_ROOT,
        runtime_dir: Path | None = None,
        reports_dir: Path | None = None,
        git_provider: Callable[[], dict[str, Any]] | None = None,
        systemctl_provider: Callable[[], dict[str, Any]] | None = None,
        fills_provider: Callable[[], dict[str, Any]] | None = None,
        bar_freshness_provider: Callable[[], dict[str, Any]] | None = None,
        ml_shadow_provider: Callable[[], dict[str, Any]] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.runtime_dir = Path(runtime_dir) if runtime_dir else self.repo_root / "runtime"
        self.reports_dir = Path(reports_dir) if reports_dir else self.repo_root / "reports" / "exterminator"
        self.git_provider = git_provider or (lambda: default_git_provider(self.repo_root))
        self.systemctl_provider = systemctl_provider or default_systemctl_provider
        self.fills_provider = fills_provider or (lambda: default_fills_provider(self.runtime_dir))
        self.bar_freshness_provider = bar_freshness_provider or (
            lambda: default_log_scan_provider(self.repo_root, r"bar_stale|data_freshness", max_files=5)
        )
        self.ml_shadow_provider = ml_shadow_provider or (
            lambda: default_log_scan_provider(self.repo_root, r"ML_SHADOW", max_files=5)
        )
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    # ---- helpers -------------------------------------------------------

    def _read_json(self, name: str) -> tuple[dict[str, Any] | None, str | None]:
        path = self.runtime_dir / name
        if not path.exists():
            return None, "missing"
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as err:
            return None, f"unreadable: {err}"
        if not isinstance(data, dict):
            return None, "not_an_object"
        return data, None

    @staticmethod
    def _parse_ts(value: Any) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    # ---- checks --------------------------------------------------------

    def check_failed_services(self) -> list[Finding]:
        info = self.systemctl_provider()
        if info.get("error"):
            return [Finding(
                id="EX001", severity=SEVERITY_WARNING, category="services",
                title="systemctl probe failed",
                summary="Could not enumerate failed CHAD services.",
                evidence={"error": info["error"]},
                recommended_next_action="Run `systemctl --failed` manually and investigate access.",
            )]
        failed = info.get("failed_units") or []
        if failed:
            return [Finding(
                id="EX001", severity=SEVERITY_CRITICAL, category="services",
                title="Failed CHAD systemd units",
                summary=f"{len(failed)} CHAD service unit(s) reporting failed.",
                evidence={"failed_units": failed},
                recommended_next_action="Inspect `journalctl -u <unit>` for each failed unit; do not auto-restart trading engine.",
            )]
        return []

    def check_dirty_git(self, git_info: dict[str, Any]) -> list[Finding]:
        if git_info.get("error"):
            return [Finding(
                id="EX002", severity=SEVERITY_WARNING, category="repo",
                title="git probe failed",
                summary="Could not determine repository state.",
                evidence={"error": git_info["error"]},
                recommended_next_action="Run `git status` manually.",
            )]
        if not git_info.get("clean", True):
            return [Finding(
                id="EX002", severity=SEVERITY_WARNING, category="repo",
                title="Working tree is dirty",
                summary=f"{len(git_info.get('dirty_files', []))} uncommitted entries.",
                evidence={"dirty_files": git_info.get("dirty_files", [])[:50]},
                recommended_next_action="Review pending edits before any tag/commit; do not auto-stage.",
            )]
        return []

    def check_stale_runtime_files(self) -> list[Finding]:
        out: list[Finding] = []
        now = self.clock()
        for name in TTL_RUNTIME_FILES:
            data, err = self._read_json(name)
            if err == "missing" or data is None:
                continue  # missing handled by check_missing_runtime_files / invalid_json
            ts = self._parse_ts(data.get("ts_utc"))
            ttl = data.get("ttl_seconds")
            if ts is None or not isinstance(ttl, (int, float)):
                continue
            age = (now - ts).total_seconds()
            if age > ttl:
                out.append(Finding(
                    id="EX003", severity=SEVERITY_WARNING, category="runtime_freshness",
                    title=f"Stale runtime file: {name}",
                    summary=f"{name} ts_utc is {age:.0f}s old; ttl_seconds={ttl}.",
                    evidence={"file": name, "age_seconds": round(age, 1), "ttl_seconds": ttl, "ts_utc": data.get("ts_utc")},
                    recommended_next_action=f"Investigate publisher of {name}; do not patch the file.",
                ))
        return out

    def check_missing_runtime_files(self) -> list[Finding]:
        out: list[Finding] = []
        for name in CORE_RUNTIME_FILES:
            path = self.runtime_dir / name
            if not path.exists():
                out.append(Finding(
                    id="EX004", severity=SEVERITY_WARNING, category="runtime_completeness",
                    title=f"Missing required runtime file: {name}",
                    summary=f"{name} not present under {self.runtime_dir}.",
                    evidence={"file": name, "expected_path": str(path)},
                    recommended_next_action=f"Verify the publisher service for {name} is running.",
                ))
        return out

    def check_invalid_runtime_json(self) -> list[Finding]:
        out: list[Finding] = []
        for name in CORE_RUNTIME_FILES:
            path = self.runtime_dir / name
            if not path.exists():
                continue
            data, err = self._read_json(name)
            if err and err != "missing":
                severity = SEVERITY_CRITICAL if name in ("scr_state.json", "live_readiness.json", "stop_bus.json", "reconciliation_state.json") else SEVERITY_WARNING
                out.append(Finding(
                    id="EX005", severity=severity, category="runtime_integrity",
                    title=f"Corrupted runtime file: {name}",
                    summary=f"{name} is unreadable or not a JSON object.",
                    evidence={"file": name, "error": err},
                    recommended_next_action=f"Quarantine {name} and re-publish from authoritative source. Do not edit by hand.",
                ))
        return out

    def check_placeholder_fills(self) -> list[Finding]:
        info = self.fills_provider()
        if not info.get("available"):
            return [Finding(
                id="EX006", severity=SEVERITY_INFO, category="fills",
                title="No fills source available to scan",
                summary="No paper fills artifact found at known runtime paths; placeholder scan skipped.",
                evidence={"checked_runtime_dir": str(self.runtime_dir)},
                recommended_next_action="If paper executor publishes fills NDJSON, expose it under runtime/.",
            )]
        placeholders: list[dict[str, Any]] = []
        for fill in info.get("fills", []):
            price = fill.get("fill_price", fill.get("price"))
            try:
                if price is not None and float(price) == PLACEHOLDER_PRICE:
                    placeholders.append({
                        "symbol": fill.get("symbol"),
                        "fill_price": price,
                        "ts_utc": fill.get("ts_utc"),
                        "trade_id": fill.get("trade_id"),
                    })
            except (TypeError, ValueError):
                continue
        if placeholders:
            return [Finding(
                id="EX006", severity=SEVERITY_WARNING, category="fills",
                title="Placeholder $100.0 fill prices detected",
                summary=f"{len(placeholders)} fill(s) with fill_price={PLACEHOLDER_PRICE} (untrusted placeholder).",
                evidence={"placeholder_fills": placeholders[:10], "fills_source": info.get("source")},
                recommended_next_action="Confirm paper executor placeholder rejection (SSOT v9.0 §5 #6) is active; audit how these fills passed.",
            )]
        return []

    def check_untrusted_spike(self, scr: dict[str, Any] | None) -> list[Finding]:
        if not scr:
            return []
        stats = scr.get("stats", {}) if isinstance(scr.get("stats"), dict) else {}
        excluded_untrusted = stats.get("excluded_untrusted") or 0
        excluded_nonfinite = stats.get("excluded_nonfinite") or 0
        excluded_manual = stats.get("excluded_manual") or 0
        total = (excluded_untrusted or 0) + (excluded_nonfinite or 0) + (excluded_manual or 0)
        breakdown = {
            "untrusted": excluded_untrusted,
            "nonfinite": excluded_nonfinite,
            "manual": excluded_manual,
        }
        if total == 0:
            return []
        # Spike heuristic: untrusted exclusions > 5 = WARNING, else INFO (expected safety exclusion).
        severity = SEVERITY_WARNING if (excluded_untrusted or 0) > 5 else SEVERITY_INFO
        title = "Untrusted/excluded paper-fill spike" if severity == SEVERITY_WARNING else "Expected safety exclusions present"
        return [Finding(
            id="EX007", severity=severity, category="paper_fills",
            title=title,
            summary=f"SCR exclusion breakdown: untrusted={excluded_untrusted}, nonfinite={excluded_nonfinite}, manual={excluded_manual}.",
            evidence={"breakdown": breakdown},
            recommended_next_action="Review paper executor logs for the untrusted fill source; do not clear exclusions.",
        )]

    def check_scr_raw_effective_gap(self, scr: dict[str, Any] | None) -> list[Finding]:
        if not scr:
            return []
        stats = scr.get("stats", {}) if isinstance(scr.get("stats"), dict) else {}
        total = stats.get("total_trades")
        effective = stats.get("effective_trades")
        excluded_untrusted = stats.get("excluded_untrusted") or 0
        excluded_nonfinite = stats.get("excluded_nonfinite") or 0
        excluded_manual = stats.get("excluded_manual") or 0
        if not isinstance(total, (int, float)) or not isinstance(effective, (int, float)):
            return []
        gap = int(total) - int(effective)
        if gap <= 0:
            return []
        sum_known = (excluded_untrusted or 0) + (excluded_nonfinite or 0) + (excluded_manual or 0)
        pnl_zero = max(gap - sum_known, 0)
        breakdown = {
            "total_trades": total,
            "effective_trades": effective,
            "gap": gap,
            "excluded_untrusted": excluded_untrusted,
            "excluded_nonfinite": excluded_nonfinite,
            "excluded_manual": excluded_manual,
            "pnl_zero_or_other": pnl_zero,
        }
        # Gap is informational — exclusions are by-design. Elevate only if pnl_zero unexplained > sum_known.
        severity = SEVERITY_INFO if pnl_zero == 0 else SEVERITY_INFO
        return [Finding(
            id="EX008", severity=severity, category="scr",
            title="SCR raw vs effective trade gap",
            summary=f"Total {total} vs effective {effective} (gap={gap}).",
            evidence=breakdown,
            recommended_next_action="Confirm exclusions match expected paper executor safety paths.",
        )]

    def check_reconciliation(self, recon: dict[str, Any] | None) -> list[Finding]:
        if not recon:
            return []
        status = (recon.get("status") or "").upper()
        mismatches = recon.get("mismatches") or []
        drifts = recon.get("drifts") or []
        # PR-09: broker_sync-only advisory entries now live in
        # diagnostic_drifts (drifts[] is reserved for strategy-attributable
        # drifts that MUST trip live_readiness RED). For the EX009 visibility
        # surface, combine both so operators continue to see broker-side
        # advisory entries without losing the GAP-041 fail-closed contract.
        diagnostic_drifts = recon.get("diagnostic_drifts") or []
        advisory_drifts = list(drifts) + list(diagnostic_drifts)
        if status in ("RED", "FAIL", "MISMATCH") or mismatches:
            return [Finding(
                id="EX009C", severity=SEVERITY_CRITICAL, category="reconciliation",
                title="Reconciliation mismatch",
                summary=f"Reconciliation status={status or 'UNKNOWN'}, mismatches={len(mismatches)}.",
                evidence={"status": status, "mismatch_count": len(mismatches), "mismatches": mismatches[:10]},
                recommended_next_action="Halt new entries; investigate broker truth before any further trading.",
            )]
        if status == "GREEN" and advisory_drifts:
            return [Finding(
                id="EX009", severity=SEVERITY_WARNING, category="reconciliation",
                title="Reconciliation GREEN with tracked drifts",
                summary=(
                    f"Status GREEN but {len(advisory_drifts)} drift entries on broker side "
                    f"(strategy-attributable={len(drifts)} diagnostic={len(diagnostic_drifts)})."
                ),
                evidence={
                    "status": status,
                    "drift_count": len(advisory_drifts),
                    "strategy_drift_count": len(drifts),
                    "diagnostic_drift_count": len(diagnostic_drifts),
                    "drifts": advisory_drifts[:10],
                    "worst_diff": recon.get("worst_diff"),
                },
                recommended_next_action="Reconcile or formally exclude each drift before live promotion (SSOT v9.0 §6 #10).",
            )]
        return []

    def check_halt_boost_contradiction(
        self,
        strategy_health: dict[str, Any] | None,
        winner_scaling: dict[str, Any] | None,
    ) -> list[Finding]:
        if not strategy_health or not winner_scaling:
            return []

        halted: set[str] = set()
        sh_strategies = strategy_health.get("strategies") or strategy_health
        if isinstance(sh_strategies, dict):
            for name, payload in sh_strategies.items():
                if not isinstance(payload, dict):
                    continue
                halt_flag = payload.get("edge_decay_halt") or payload.get("halt") or payload.get("halted")
                if halt_flag is True or (isinstance(halt_flag, str) and halt_flag.lower() in ("active", "true", "halted")):
                    halted.add(str(name))

        boosted: dict[str, float] = {}
        ws_section = winner_scaling.get("strategies") or winner_scaling.get("multipliers") or winner_scaling
        if isinstance(ws_section, dict):
            for name, payload in ws_section.items():
                if isinstance(payload, dict):
                    mult = payload.get("multiplier") or payload.get("winner_multiplier")
                elif isinstance(payload, (int, float)):
                    mult = payload
                else:
                    mult = None
                try:
                    if mult is not None and float(mult) > 1.0:
                        boosted[str(name)] = float(mult)
                except (TypeError, ValueError):
                    continue

        contradictions = sorted(halted & boosted.keys())
        if contradictions:
            return [Finding(
                id="EX010", severity=SEVERITY_WARNING, category="contradiction",
                title="Halt + winner-scaling boost contradiction",
                summary=f"Halted strategies receiving multipliers > 1.0×: {', '.join(contradictions)}.",
                evidence={"halted": sorted(halted), "boosts_for_halted": {k: boosted[k] for k in contradictions}},
                recommended_next_action="Audit edge-decay halt; suppress boost for halted strategies. Do not manually clear halts.",
            )]
        return []

    def check_bar_freshness_log_spam(self) -> list[Finding]:
        info = self.bar_freshness_provider()
        matches = info.get("matches", 0)
        if matches > 25:
            return [Finding(
                id="EX011", severity=SEVERITY_WARNING, category="data",
                title="Bar staleness log spam",
                summary=f"{matches} `bar_stale`/`data_freshness` lines across recent logs.",
                evidence=info,
                recommended_next_action="Investigate data feed; confirm 1m bar provider health (SSOT v9.0 §5 #8).",
            )]
        return []

    def check_ml_shadow(self) -> list[Finding]:
        info = self.ml_shadow_provider()
        if info.get("matches", 0) == 0:
            return [Finding(
                id="EX012", severity=SEVERITY_INFO, category="ml",
                title="ML_SHADOW activity not observed in recent logs",
                summary="No ML_SHADOW log entries in the most recent log files.",
                evidence=info,
                recommended_next_action="Confirm ML veto loop is firing in shadow mode (SSOT v9.0 §5 #12).",
            )]
        return [Finding(
            id="EX012", severity=SEVERITY_INFO, category="ml",
            title="ML_SHADOW activity present (shadow-only)",
            summary=f"{info['matches']} ML_SHADOW lines observed across recent logs.",
            evidence=info,
            recommended_next_action="None; promotion to enforcement requires explicit governance.",
        )]

    def check_schema_versions(self) -> list[Finding]:
        out: list[Finding] = []
        for name in SCHEMA_REQUIRED_FILES:
            data, err = self._read_json(name)
            if data is None:
                continue
            if "schema_version" not in data:
                out.append(Finding(
                    id="EX013", severity=SEVERITY_WARNING, category="schema",
                    title=f"schema_version missing on {name}",
                    summary=f"{name} has no schema_version field.",
                    evidence={"file": name},
                    recommended_next_action=f"Update the publisher of {name} to emit schema_version.",
                ))
        return out

    def check_live_readiness(self, live_readiness: dict[str, Any] | None, scr: dict[str, Any] | None) -> list[Finding]:
        if not live_readiness:
            return []
        ready = live_readiness.get("ready_for_live")
        scr_state = (scr or {}).get("state", "UNKNOWN")
        if ready is True and scr_state in ("WARMUP", "PAUSED", "UNKNOWN"):
            return [Finding(
                id="EX014C", severity=SEVERITY_CRITICAL, category="live_gate",
                title="live_readiness=true while SCR is unsafe",
                summary=f"ready_for_live=true but SCR state={scr_state}.",
                evidence={"ready_for_live": ready, "scr_state": scr_state, "requirements_remaining": live_readiness.get("requirements_remaining", [])},
                recommended_next_action="Investigate live_readiness_publish; do not promote to live until SCR is CONFIDENT.",
            )]
        if ready is False:
            return [Finding(
                id="EX014", severity=SEVERITY_INFO, category="live_gate",
                title="live_readiness=false (live not authorized)",
                summary="System remains paper-only; live promotion gate intact.",
                evidence={
                    "ready_for_live": False,
                    "requirements_remaining": live_readiness.get("requirements_remaining", []),
                    "epoch": (live_readiness.get("epoch_metadata") or {}).get("epoch"),
                },
                recommended_next_action="None; expected paper posture per SSOT v9.0.",
            )]
        return []

    def check_stop_bus(self, stop_bus: dict[str, Any] | None) -> list[Finding]:
        if not stop_bus:
            return []
        if stop_bus.get("active") is True:
            return [Finding(
                id="EX015", severity=SEVERITY_CRITICAL, category="halt",
                title="Stop bus is ACTIVE",
                summary=stop_bus.get("reason") or "Trading halt is engaged.",
                evidence=stop_bus,
                recommended_next_action="Investigate trigger before clearing; never clear by hand.",
            )]
        return []

    def check_profit_lock(self, profit_lock: dict[str, Any] | None) -> list[Finding]:
        if not profit_lock:
            return []
        if profit_lock.get("daily_loss_limit_hit") is True or profit_lock.get("stop_new_entries") is True:
            return [Finding(
                id="EX016", severity=SEVERITY_CRITICAL, category="risk",
                title="Profit lock daily-loss hard stop engaged",
                summary=f"mode={profit_lock.get('mode')}, daily_loss={profit_lock.get('daily_loss_today')}.",
                evidence=profit_lock,
                recommended_next_action="Hold new entries; review the loss path before any reset.",
            )]
        if profit_lock.get("profit_lock_active") is True:
            return [Finding(
                id="EX016", severity=SEVERITY_WARNING, category="risk",
                title="Profit lock active",
                summary=f"sizing_factor={profit_lock.get('sizing_factor')}, mode={profit_lock.get('mode')}.",
                evidence=profit_lock,
                recommended_next_action="Confirm profit-lock sizing applied through full execution path.",
            )]
        return []

    def check_strategy_health(self, strategy_health: dict[str, Any] | None) -> list[Finding]:
        if not strategy_health:
            return []
        section = strategy_health.get("strategies") or strategy_health
        if not isinstance(section, dict):
            return []
        out: list[Finding] = []
        for name, payload in section.items():
            if not isinstance(payload, dict):
                continue
            samples = payload.get("samples") or payload.get("sample_count") or payload.get("trades")
            score = payload.get("score") or payload.get("health_score")
            try:
                samples_int = int(samples) if samples is not None else None
            except (TypeError, ValueError):
                samples_int = None
            try:
                score_float = float(score) if score is not None else None
            except (TypeError, ValueError):
                score_float = None
            if samples_int is not None and samples_int < 5:
                out.append(Finding(
                    id="EX017", severity=SEVERITY_INFO, category="strategy_health",
                    title=f"Low-sample strategy: {name}",
                    summary=f"{name} has {samples_int} samples (< 5).",
                    evidence={"strategy": name, "samples": samples_int, "score": score_float},
                    recommended_next_action="Maturity gate active; do not promote allocations on this evidence.",
                ))
            elif samples_int is not None and samples_int >= 30 and score_float is not None and score_float < 0.3:
                out.append(Finding(
                    id="EX017", severity=SEVERITY_WARNING, category="strategy_health",
                    title=f"Mature strategy degradation: {name}",
                    summary=f"{name} has {samples_int} samples and score={score_float}.",
                    evidence={"strategy": name, "samples": samples_int, "score": score_float},
                    recommended_next_action="Review edge-decay rules for this strategy.",
                ))
        return out

    def check_winner_scaling(self, winner_scaling: dict[str, Any] | None, strategy_health: dict[str, Any] | None) -> list[Finding]:
        if not winner_scaling:
            return []
        ts = self._parse_ts(winner_scaling.get("ts_utc") or winner_scaling.get("last_updated"))
        out: list[Finding] = []
        if ts is not None:
            age_hours = (self.clock() - ts).total_seconds() / 3600.0
            if age_hours > 24:
                out.append(Finding(
                    id="EX018", severity=SEVERITY_WARNING, category="winner_scaling",
                    title="Winner scaling state stale",
                    summary=f"winner_scaling.json last updated {age_hours:.1f}h ago.",
                    evidence={"age_hours": round(age_hours, 1)},
                    recommended_next_action="Verify winner scaling publisher cadence.",
                ))
        # aggressive boost detection without halt is informational; halt+boost is EX010 (covered there).
        section = winner_scaling.get("strategies") or winner_scaling.get("multipliers") or winner_scaling
        if isinstance(section, dict):
            aggressive = []
            for name, payload in section.items():
                mult = payload.get("multiplier") if isinstance(payload, dict) else (payload if isinstance(payload, (int, float)) else None)
                try:
                    if mult is not None and float(mult) >= 1.4:
                        aggressive.append({"strategy": name, "multiplier": float(mult)})
                except (TypeError, ValueError):
                    continue
            if aggressive:
                out.append(Finding(
                    id="EX018", severity=SEVERITY_INFO, category="winner_scaling",
                    title="Aggressive winner-scaling multipliers in effect",
                    summary=f"{len(aggressive)} strategy(ies) at ≥1.4× multiplier.",
                    evidence={"aggressive": aggressive[:10]},
                    recommended_next_action="Cross-check aggressive boosts against halt/edge-decay state.",
                ))
        return out

    # ---- assembly ------------------------------------------------------

    def _build_runtime_posture(
        self,
        scr: dict[str, Any] | None,
        live_readiness: dict[str, Any] | None,
        epoch: dict[str, Any] | None,
        profit_lock: dict[str, Any] | None,
        stop_bus: dict[str, Any] | None,
        recon: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "epoch": (epoch or {}).get("active_epoch") or ((live_readiness or {}).get("epoch_metadata") or {}).get("epoch") or "",
            "paper_only": bool((epoch or scr or {}).get("paper_only", True)),
            "live_readiness": bool((live_readiness or {}).get("ready_for_live", False)),
            "scr_state": (scr or {}).get("state") or "",
            "scr_sizing_factor": float((scr or {}).get("sizing_factor", 0.0) or 0.0),
            "profit_lock_mode": (profit_lock or {}).get("mode") or "",
            "stop_bus_active": bool((stop_bus or {}).get("active", False)),
            "reconciliation_status": (recon or {}).get("status") or "",
        }

    def run(self) -> dict[str, Any]:
        # Read core runtime payloads once.
        scr, _ = self._read_json("scr_state.json")
        live_readiness, _ = self._read_json("live_readiness.json")
        epoch, _ = self._read_json("epoch_state.json")
        profit_lock, _ = self._read_json("profit_lock_state.json")
        stop_bus, _ = self._read_json("stop_bus.json")
        recon, _ = self._read_json("reconciliation_state.json")
        strategy_health, _ = self._read_json("strategy_health.json")
        winner_scaling, _ = self._read_json("winner_scaling.json")
        git_info = self.git_provider()

        findings: list[Finding] = []

        def _safe_extend(label: str, gen: Callable[[], list[Finding]]) -> None:
            try:
                findings.extend(gen())
            except Exception as err:  # check failure must not abort the report
                findings.append(Finding(
                    id="EX999", severity=SEVERITY_WARNING, category="sentinel_self",
                    title=f"Sentinel check raised: {label}",
                    summary=f"Check {label!r} raised {type(err).__name__}: {err}",
                    evidence={"check": label, "error": repr(err)},
                    recommended_next_action="Investigate sentinel module; report may be incomplete.",
                ))

        _safe_extend("failed_services", self.check_failed_services)
        _safe_extend("dirty_git", lambda: self.check_dirty_git(git_info))
        _safe_extend("missing_runtime_files", self.check_missing_runtime_files)
        _safe_extend("invalid_runtime_json", self.check_invalid_runtime_json)
        _safe_extend("stale_runtime_files", self.check_stale_runtime_files)
        _safe_extend("placeholder_fills", self.check_placeholder_fills)
        _safe_extend("untrusted_spike", lambda: self.check_untrusted_spike(scr))
        _safe_extend("scr_raw_effective_gap", lambda: self.check_scr_raw_effective_gap(scr))
        _safe_extend("reconciliation", lambda: self.check_reconciliation(recon))
        _safe_extend("halt_boost_contradiction", lambda: self.check_halt_boost_contradiction(strategy_health, winner_scaling))
        _safe_extend("bar_freshness", self.check_bar_freshness_log_spam)
        _safe_extend("ml_shadow", self.check_ml_shadow)
        _safe_extend("schema_versions", self.check_schema_versions)
        _safe_extend("live_readiness", lambda: self.check_live_readiness(live_readiness, scr))
        _safe_extend("stop_bus", lambda: self.check_stop_bus(stop_bus))
        _safe_extend("profit_lock", lambda: self.check_profit_lock(profit_lock))
        _safe_extend("strategy_health", lambda: self.check_strategy_health(strategy_health))
        _safe_extend("winner_scaling", lambda: self.check_winner_scaling(winner_scaling, strategy_health))
        _safe_extend(
            "scr_warmup_info",
            lambda: (
                [Finding(
                    id="EX020", severity=SEVERITY_INFO, category="scr",
                    title="SCR is in WARMUP",
                    summary=f"sizing_factor={(scr or {}).get('sizing_factor')}, effective_trades={((scr or {}).get('stats') or {}).get('effective_trades')}.",
                    evidence={"scr_state": (scr or {}).get("state"), "sizing_factor": (scr or {}).get("sizing_factor")},
                    recommended_next_action="None; expected during paper soak (SSOT v9.0).",
                )]
                if (scr or {}).get("state") == "WARMUP"
                else []
            ),
        )

        counts = {"critical": 0, "warning": 0, "info": 0}
        for f in findings:
            sev = f.severity.lower()
            if sev == "critical":
                counts["critical"] += 1
            elif sev == "warning":
                counts["warning"] += 1
            elif sev == "info":
                counts["info"] += 1

        report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": self.clock().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "mode": "read_only",
            "repo": {
                "head": git_info.get("head", ""),
                "branch": git_info.get("branch", ""),
                "git_clean": bool(git_info.get("clean", True)),
                "dirty_files": list(git_info.get("dirty_files", []) or []),
                "tags": list(git_info.get("tags", []) or []),
            },
            "runtime_posture": self._build_runtime_posture(
                scr, live_readiness, epoch, profit_lock, stop_bus, recon
            ),
            "findings": [f.to_dict() for f in findings],
            "counts": counts,
            "read_only_confirmed": True,
            "runtime_files_modified": [],
            "services_restarted": False,
        }
        return report

    # ---- writers -------------------------------------------------------

    def _format_markdown(self, report: dict[str, Any]) -> str:
        lines: list[str] = []
        lines.append("# CHAD Exterminator Stage 1 — Read-Only Sentinel Report")
        lines.append("")
        lines.append(f"**Generated:** {report['generated_at_utc']}")
        lines.append(f"**Mode:** {report['mode']}")
        lines.append(f"**Schema:** {report['schema_version']}")
        lines.append("")
        lines.append("## Posture Summary")
        lines.append("")
        repo = report["repo"]
        post = report["runtime_posture"]
        lines.append(f"- Repo HEAD: `{repo['head']}`")
        lines.append(f"- Branch: `{repo['branch']}`")
        lines.append(f"- Git clean: {repo['git_clean']}")
        lines.append(f"- Tags at HEAD: {', '.join(repo['tags']) if repo['tags'] else '(none)'}")
        lines.append(f"- Epoch: `{post['epoch']}`")
        lines.append(f"- SCR state: `{post['scr_state']}` (sizing_factor={post['scr_sizing_factor']})")
        lines.append(f"- Live readiness: {post['live_readiness']}")
        lines.append(f"- Profit lock mode: `{post['profit_lock_mode']}`")
        lines.append(f"- Stop bus active: {post['stop_bus_active']}")
        lines.append(f"- Reconciliation: `{post['reconciliation_status']}`")
        lines.append("")
        c = report["counts"]
        lines.append(f"## Findings ({len(report['findings'])}) — CRITICAL={c['critical']} WARNING={c['warning']} INFO={c['info']}")
        lines.append("")
        if report["findings"]:
            lines.append("| ID | Severity | Category | Title |")
            lines.append("|----|----------|----------|-------|")
            for f in report["findings"]:
                title = (f["title"] or "").replace("|", "\\|")
                lines.append(f"| {f['id']} | {f['severity']} | {f['category']} | {title} |")
            lines.append("")
            lines.append("### Detail")
            lines.append("")
            for f in report["findings"]:
                lines.append(f"#### {f['id']} — {f['title']} ({f['severity']})")
                lines.append("")
                lines.append(f"- **Category:** {f['category']}")
                lines.append(f"- **Summary:** {f['summary']}")
                if f["recommended_next_action"]:
                    lines.append(f"- **Next action:** {f['recommended_next_action']}")
                lines.append(f"- **Auto-fix allowed:** {f['auto_fix_allowed']}")
                lines.append(f"- **Requires operator:** {f['requires_operator']}")
                if f["evidence"]:
                    lines.append("- **Evidence:**")
                    lines.append("")
                    lines.append("```json")
                    lines.append(json.dumps(f["evidence"], indent=2, sort_keys=True, default=str))
                    lines.append("```")
                lines.append("")
        else:
            lines.append("_No findings._")
            lines.append("")
        lines.append("## Next Actions")
        lines.append("")
        next_actions = [f["recommended_next_action"] for f in report["findings"] if f.get("recommended_next_action")]
        if next_actions:
            seen: set[str] = set()
            for action in next_actions:
                if action in seen:
                    continue
                seen.add(action)
                lines.append(f"- {action}")
        else:
            lines.append("- No operator action required.")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("**This is a read-only Stage 1 report. No runtime files were modified, no services were restarted, no commits were created.**")
        lines.append("")
        return "\n".join(lines)

    def write_reports(self, report: dict[str, Any]) -> tuple[Path, Path]:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = report["generated_at_utc"].replace("-", "").replace(":", "").replace("Z", "Z")
        # stamp shape: 20260505T123456Z → convert to YYYYMMDD_HHMMSSZ
        try:
            dt = datetime.strptime(report["generated_at_utc"], "%Y-%m-%dT%H:%M:%SZ")
            stamp = dt.strftime("%Y%m%d_%H%M%SZ")
        except ValueError:
            pass
        json_path = self.reports_dir / f"EXTERMINATOR_{stamp}.json"
        md_path = self.reports_dir / f"EXTERMINATOR_{stamp}.md"
        json_tmp = json_path.with_suffix(".json.tmp")
        md_tmp = md_path.with_suffix(".md.tmp")
        json_tmp.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
        md_tmp.write_text(self._format_markdown(report), encoding="utf-8")
        os.replace(json_tmp, json_path)
        os.replace(md_tmp, md_path)
        return json_path, md_path


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> int:
    sentinel = Exterminator()
    report = sentinel.run()
    json_path, md_path = sentinel.write_reports(report)
    counts = report["counts"]
    print(
        f"exterminator: critical={counts['critical']} warning={counts['warning']} info={counts['info']} "
        f"json={json_path} md={md_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
