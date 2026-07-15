"""Exterminator Sentinel — Stage 1 (canonical).

Pure read-only scanner. Runs 8 checks, each returning ok|warn|fail plus the
evidence behind the verdict, and emits::

    runtime/reports/EXTERMINATOR_SENTINEL_LATEST.json   (exterminator_sentinel.v1)
    runtime/reports/EXTERMINATOR_SENTINEL_HISTORY.ndjson (append-only)

Those two paths are the ONLY writes this module is permitted to make. It does
not mutate runtime state, does not call brokers, does not restart or reset
services, does not stage or commit, and never auto-heals anything it finds.
Stage 1 observes; a future Stage-2 PA may PROPOSE fixes, and even then a human
applies them. chad/tests/test_exterminator_sentinel.py locks these contracts.

Supersedes chad/ops/exterminator.py (exterminator_report.v1), which last ran
2026-05-05 against Epoch 2 and has zero production importers. That scanner's
harness was sound and is reused here in spirit; four of its checks were blind
and are rebuilt — see docs/EXTERMINATOR_SENTINEL_INSTALL.md.

THE INDEPENDENT-LEG RULE (check 4, the lesson of XOV-2345)
----------------------------------------------------------
Position truth has three legs, and only one of them is independent:

  reconciliation_state.json  IBKR clientId=83 vs the guard's own broker_sync
                             rows -- and in paper mode it filters to
                             strategy == "broker_sync" (reconciliation_publisher
                             .py:166), so strategy rows are never compared.
  position_guard_drift.json  guard's broker_sync rows vs guard's strategy rows
                             -- BOTH LEGS COME FROM position_guard.json. It
                             compares that file against itself.
  positions_snapshot.json    clientId=99, separate process, own IB(), own timer
                             (ibkr_portfolio_collector_v2.py:125). INDEPENDENT.

When the guard was false-flatted at 23:45 by a dead shared `ib` cache, the first
two legs agreed that everything was fine because they were reading the same
corrupted source. Reconciliation stayed GREEN through a whole-book false-flat.
So check 4 compares the guard against positions_snapshot.json and NEVER against
broker_sync alone, and it explicitly reports LEG DISAGREEMENT when the
same-source legs claim GREEN while the independent leg disagrees.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

DEFAULT_REPO_ROOT = Path("/home/ubuntu/chad_finale")
SCHEMA_VERSION = "exterminator_sentinel.v1"
CONFIG_SCHEMA_VERSION = "exterminator_config.v1"

LATEST_FILENAME = "EXTERMINATOR_SENTINEL_LATEST.json"
HISTORY_FILENAME = "EXTERMINATOR_SENTINEL_HISTORY.ndjson"

STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"
_STATUS_RANK = {STATUS_OK: 0, STATUS_WARN: 1, STATUS_FAIL: 2}

SUBPROCESS_TIMEOUT_SECONDS = 15
QTY_TOLERANCE = 1e-6

# Remediation policy for every finding this module emits. Stage 1 never fixes.
REMEDY_NOTIFY_ONLY = "NOTIFY_ONLY"


def worst_status(statuses: Iterable[str]) -> str:
    """Return the most severe status in ``statuses`` (ok < warn < fail)."""
    out = STATUS_OK
    for status in statuses:
        if _STATUS_RANK.get(status, 0) > _STATUS_RANK[out]:
            out = status
    return out


def stable_identity(text: str) -> str:
    """Strip fluctuating numeric values from ``text`` to build a dedupe identity.

    CTF-T2: the old health dedupe keyed on ``title[:30]``, and several titles
    embed a changing value ("SCR gap 214 raw vs 67 effective"), so every cycle
    minted a new key and re-sent the same alert -- R13 produced 7 dedupe files
    and re-alerted every 5 minutes. The regex drops digit runs NOT preceded by a
    letter, so values go and identifiers stay (M6E, EXS1, alpha_crypto survive).
    Mirrors chad/ops/health_monitor.py:249.
    """
    identity = re.sub(r"(?<![A-Za-z])\d[\d,\.]*", "", text or "")
    return re.sub(r"\s+", " ", identity).strip()


@dataclass
class CheckResult:
    """One check's verdict. ``evidence`` carries every fluctuating value.

    Values belong in evidence, never in ``title`` -- titles feed the dedupe
    identity and must stay stable across cycles (CTF-T2).
    """

    check_id: str
    name: str
    status: str
    title: str
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)
    remedy_type: str = REMEDY_NOTIFY_ONLY

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "name": self.name,
            "status": self.status,
            "title": self.title,
            "summary": self.summary,
            "evidence": self.evidence,
            "remedy_type": self.remedy_type,
        }


# ---------------------------------------------------------------------------
# Read-only providers
# ---------------------------------------------------------------------------


def default_systemctl_provider(query: Sequence[str]) -> dict[str, Any]:
    """Enumerate failed units with a read-only query. Never mutates unit state."""
    out: dict[str, Any] = {"failed_units": [], "error": None, "query": list(query)}
    try:
        result = subprocess.run(
            list(query), capture_output=True, text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS, check=False,
        )
        for line in result.stdout.splitlines():
            tokens = line.split()
            if tokens and tokens[0].endswith((".service", ".timer")):
                out["failed_units"].append(tokens[0])
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as err:
        out["error"] = repr(err)
    return out


def default_git_provider(repo_root: Path) -> dict[str, Any]:
    """Return porcelain status + HEAD. Read-only: no add, commit, tag or push."""
    out: dict[str, Any] = {"head": "", "branch": "", "entries": [], "error": None}
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
        out["entries"] = [ln for ln in status.splitlines() if ln.strip()]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as err:
        out["error"] = repr(err)
    return out


def default_notifier(message: str, dedupe_key: str) -> bool:
    """Send a Telegram alert. Import is local + guarded: alerting is optional.

    ``notify`` raises NotifyError when the bot config is absent (that absence is
    the de-facto kill switch), so every failure mode is swallowed -- a sentinel
    must never fail because it could not talk to Telegram.
    """
    try:
        from chad.utils.telegram_notify import notify

        return bool(notify(message, severity="critical", dedupe_key=dedupe_key))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Sentinel
# ---------------------------------------------------------------------------


class ExterminatorSentinel:
    """Stage 1 read-only sentinel. Every mutation path is forbidden."""

    def __init__(
        self,
        repo_root: Path = DEFAULT_REPO_ROOT,
        runtime_dir: Path | None = None,
        data_dir: Path | None = None,
        reports_dir: Path | None = None,
        config_path: Path | None = None,
        clock: Callable[[], datetime] | None = None,
        systemctl_provider: Callable[[Sequence[str]], dict[str, Any]] | None = None,
        git_provider: Callable[[], dict[str, Any]] | None = None,
        notifier: Callable[[str, str], bool] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.runtime_dir = Path(runtime_dir) if runtime_dir else self.repo_root / "runtime"
        self.data_dir = Path(data_dir) if data_dir else self.repo_root / "data"
        explicit_reports = reports_dir is not None
        self.reports_dir = Path(reports_dir) if reports_dir else self.runtime_dir / "reports"
        self.config_path = Path(config_path) if config_path else self.repo_root / "config" / "exterminator.json"
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.systemctl_provider = systemctl_provider or default_systemctl_provider
        self.git_provider = git_provider or (lambda: default_git_provider(self.repo_root))
        self.notifier = notifier or default_notifier

        # Test-write leak guard, mirroring build_default_overlay
        # (chad/risk/position_exit_overlay.py:964): under pytest the report
        # directory must be explicit, so a test can never append to the real
        # runtime history it is supposed to be inspecting.
        if not explicit_reports and "pytest" in sys.modules:
            raise RuntimeError(
                "ExterminatorSentinel: reports_dir must be explicit under pytest "
                "(refusing to default to the real runtime/reports)"
            )
        self.config = self._load_config()

    # ---- config --------------------------------------------------------

    def _load_config(self) -> dict[str, Any]:
        try:
            with self.config_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    # ---- helpers -------------------------------------------------------

    def _read_json(self, path: Path) -> tuple[Any, str | None]:
        if not path.exists():
            return None, "missing"
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh), None
        except (OSError, json.JSONDecodeError) as err:
            return None, f"unreadable: {err}"

    @staticmethod
    def _parse_ts(value: Any) -> datetime | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    def _age_seconds(self, ts: datetime | None) -> float | None:
        if ts is None:
            return None
        return (self.clock() - ts).total_seconds()

    @staticmethod
    def _last_ndjson_row(path: Path) -> dict[str, Any] | None:
        last: dict[str, Any] | None = None
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        last = row
        except OSError:
            return None
        return last

    @staticmethod
    def _iter_ndjson(path: Path) -> Iterable[dict[str, Any]]:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        yield row
        except OSError:
            return

    def _recent_files(self, pattern: str, days: int) -> list[Path]:
        """Files matching ``pattern`` (repo-relative glob) modified within ``days``."""
        cutoff = self.clock().timestamp() - days * 86400
        out: list[Path] = []
        for name in sorted(glob.glob(str(self.repo_root / pattern))):
            path = Path(name)
            try:
                if path.is_file() and path.stat().st_mtime >= cutoff:
                    out.append(path)
            except OSError:
                continue
        return out

    # ---- check 1: stale feeds -----------------------------------------

    def check_stale_feeds(self) -> CheckResult:
        feeds_cfg = (self.config.get("feeds") or {}) if self.config else {}
        if not feeds_cfg:
            return CheckResult(
                "EXS1", "stale_feeds", STATUS_WARN, "Stale-feed TTL table unavailable",
                "config/exterminator.json declares no feeds; staleness cannot be judged.",
                {"config_path": str(self.config_path)},
            )

        rows: list[dict[str, Any]] = []
        statuses: list[str] = []
        for feed_name, cfg in sorted(feeds_cfg.items()):
            if not isinstance(cfg, dict):
                continue
            path = self.repo_root / str(cfg.get("path", ""))
            ttl_verified = bool(cfg.get("ttl_verified"))
            ttl_source = str(cfg.get("ttl_source") or "operator_verify")
            row: dict[str, Any] = {
                "feed": feed_name,
                "path": str(cfg.get("path", "")),
                "ttl_verified": ttl_verified,
                "ttl_source": ttl_source if ttl_verified else "operator_verify",
            }

            if not path.exists():
                row.update({"status": STATUS_WARN, "reason": "missing"})
                rows.append(row)
                statuses.append(STATUS_WARN)
                continue

            ts_field = str(cfg.get("ts_field") or "ts_utc")
            if str(cfg.get("format")) == "ndjson":
                payload = self._last_ndjson_row(path)
            else:
                payload, _ = self._read_json(path)
            if not isinstance(payload, dict):
                row.update({"status": STATUS_WARN, "reason": "unreadable_or_not_an_object"})
                rows.append(row)
                statuses.append(STATUS_WARN)
                continue

            age = self._age_seconds(self._parse_ts(payload.get(ts_field)))
            if age is None:
                row.update({"status": STATUS_WARN, "reason": f"no_parsable_{ts_field}"})
                rows.append(row)
                statuses.append(STATUS_WARN)
                continue

            # The artifact's own ttl_seconds wins, so this table can never
            # silently drift from the publisher that defines the contract.
            declared = payload.get("ttl_seconds")
            ttl = float(declared) if isinstance(declared, (int, float)) else float(cfg.get("ttl_seconds") or 0)
            if isinstance(declared, (int, float)):
                row["ttl_source"] = "artifact:ttl_seconds" if ttl_verified else "operator_verify"
            # Never warn before the publisher's own contract has elapsed: if the
            # artifact declares a longer TTL than this table assumed, the
            # artifact wins. max() also keeps a config warn_after that is
            # deliberately looser than the TTL (equity_history allows one missed
            # daily publish before warning).
            warn_after = max(float(cfg.get("warn_after_seconds") or 0), ttl)
            fail_after_cfg = cfg.get("fail_after_seconds")
            fail_after = float(fail_after_cfg) if isinstance(fail_after_cfg, (int, float)) else None
            if fail_after is not None:
                # A fail must never precede its own warn.
                fail_after = max(fail_after, warn_after)

            status = STATUS_OK
            if fail_after is not None and ttl_verified and age > fail_after:
                status = STATUS_FAIL
            elif warn_after and age > warn_after:
                status = STATUS_WARN
            row.update({
                "status": status,
                "age_seconds": round(age, 1),
                "ttl_seconds": ttl,
                "warn_after_seconds": warn_after,
                "fail_after_seconds": fail_after,
            })
            if warn_after and age > warn_after:
                row["breach_ratio"] = round(age / warn_after, 2)
            # An unratified TTL may never fail a gate.
            if not ttl_verified and status == STATUS_FAIL:
                row["status"] = STATUS_WARN
                row["capped_at_warn"] = "ttl_source=operator_verify"
                status = STATUS_WARN
            rows.append(row)
            statuses.append(status)

        blind = self._blind_check()
        if blind is not None:
            rows.append(blind)
            statuses.append(blind["status"])

        status = worst_status(statuses)
        bad = [r for r in rows if r.get("status") != STATUS_OK]
        if status == STATUS_OK:
            summary = f"All {len(rows)} tracked feeds are within their declared TTL."
        else:
            summary = "Feeds outside their declared TTL: " + ", ".join(
                str(r.get("feed")) for r in bad
            ) + "."
        return CheckResult(
            "EXS1", "stale_feeds", status,
            "Stale feed detected" if status != STATUS_OK else "Feeds fresh",
            summary,
            {"feeds": rows, "unverified_ttls": [r["feed"] for r in rows if r.get("ttl_source") == "operator_verify"]},
        )

    def _blind_check(self) -> dict[str, Any] | None:
        """R14's blind pattern: fresh-but-watching-nothing is not health.

        A heartbeat cannot prove it isn't blind -- only a second, independently
        published, provably fresh artifact can. If that truth artifact is stale
        or absent we return ok-with-reason rather than alert: unproven input is
        silence, not a fail. Mirrors chad/ops/health_monitor_rules.py:1134.
        """
        cfg = (self.config.get("blind_check") or {}) if self.config else {}
        feeds_cfg = (self.config.get("feeds") or {}) if self.config else {}
        hb_cfg = feeds_cfg.get(str(cfg.get("heartbeat_feed") or ""))
        truth_cfg = feeds_cfg.get(str(cfg.get("truth_feed") or ""))
        if not isinstance(hb_cfg, dict) or not isinstance(truth_cfg, dict):
            return None

        row: dict[str, Any] = {"feed": "exit_overlay_blind_check", "status": STATUS_OK}
        hb, _ = self._read_json(self.repo_root / str(hb_cfg.get("path", "")))
        if not isinstance(hb, dict):
            row["reason"] = "heartbeat_unreadable"
            return row

        # Intentionally-off or actively-evaluating overlays are not blind.
        if str(hb.get("mode") or "") == "off":
            row["reason"] = "overlay_mode_off"
            return row
        try:
            evaluated = int(hb.get("evaluated") or 0)
        except (TypeError, ValueError):
            evaluated = 0
        if evaluated > 0:
            row.update({"reason": "overlay_evaluating", "evaluated": evaluated})
            return row

        truth, _ = self._read_json(self.repo_root / str(truth_cfg.get("path", "")))
        if not isinstance(truth, dict):
            row["reason"] = "truth_feed_unreadable_proves_nothing"
            return row
        truth_age = self._age_seconds(self._parse_ts(truth.get("ts_utc")))
        max_age = float(cfg.get("truth_max_age_seconds") or 900)
        if truth_age is None or truth_age > max_age:
            row.update({"reason": "truth_feed_stale_proves_nothing", "truth_age_seconds": truth_age})
            return row

        rows = truth.get("positions")
        if not isinstance(rows, list):
            row["reason"] = "truth_feed_malformed"
            return row
        sec_type = str(cfg.get("sec_type") or "STK").upper()
        held = [
            r for r in rows
            if isinstance(r, dict)
            and str(r.get("secType") or "").upper() == sec_type
            and abs(float(r.get("position") or 0.0)) > QTY_TOLERANCE
        ]
        if not held:
            row["reason"] = "broker_genuinely_flat"
            return row

        row.update({
            "status": STATUS_FAIL,
            "reason": "overlay_blind",
            "evaluated": evaluated,
            "mode": hb.get("mode"),
            "broker_positions_held": len(held),
            "detail": "heartbeat is fresh and reports evaluated=0 while the independent collector shows held positions",
        })
        return row

    # ---- check 2: placeholder fills -----------------------------------

    def check_placeholder_fills(self) -> CheckResult:
        cfg = (self.config.get("fills") or {}) if self.config else {}
        days = int(cfg.get("scan_days") or 7)
        files = self._recent_files("data/fills/FILLS_*.ndjson", days)
        if not files:
            return CheckResult(
                "EXS2", "placeholder_fills", STATUS_WARN, "No fill ledger to scan",
                "No data/fills/FILLS_*.ndjson modified in the scan window; placeholder scan proved nothing.",
                {"scan_days": days, "scanned_files": []},
            )

        contained: list[dict[str, Any]] = []
        leaked: list[dict[str, Any]] = []
        scanned = 0
        for path in files:
            for record in self._iter_ndjson(path):
                # Real rows nest everything under a `payload` envelope.
                payload = record.get("payload") if isinstance(record.get("payload"), dict) else record
                scanned += 1
                if not self._is_placeholder(payload):
                    continue
                extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
                row = {
                    "file": path.name,
                    "symbol": payload.get("symbol"),
                    "strategy": payload.get("strategy"),
                    "status": payload.get("status"),
                    "fill_id": str(payload.get("fill_id") or "")[:16],
                    "placeholder_fill_price": extra.get("placeholder_fill_price"),
                    "trust_state": extra.get("trust_state"),
                    "reason": extra.get("pnl_untrusted_reason"),
                }
                if self._placeholder_contained(payload):
                    contained.append(row)
                else:
                    leaked.append(row)

        if leaked:
            return CheckResult(
                "EXS2", "placeholder_fills", STATUS_FAIL, "Uncontained placeholder fill",
                "Placeholder-priced fills were written WITHOUT the untrusted/rejected demotion; "
                "the paper_exec_evidence_writer placeholder defense did not fire.",
                {"leaked": leaked[:10], "leaked_count": len(leaked),
                 "contained_count": len(contained), "rows_scanned": scanned,
                 "defense": "chad/execution/paper_exec_evidence_writer.py:1923"},
            )
        if contained:
            return CheckResult(
                "EXS2", "placeholder_fills", STATUS_OK, "Placeholder fills contained",
                "Placeholder-priced fills were found and every one was correctly demoted to "
                "rejected/pnl_untrusted by the evidence-writer defense.",
                {"contained": contained[:10], "contained_count": len(contained),
                 "rows_scanned": scanned, "scanned_files": [p.name for p in files]},
            )
        return CheckResult(
            "EXS2", "placeholder_fills", STATUS_OK, "No placeholder fills",
            "No placeholder-class rows in the recent fill ledger.",
            {"rows_scanned": scanned, "scanned_files": [p.name for p in files]},
        )

    @staticmethod
    def _is_placeholder(payload: dict[str, Any]) -> bool:
        """Detect a placeholder row by its MARKERS, never by fill_price == 100.

        paper_exec_evidence_writer.py:1952 deliberately zeroes the top-level
        numeric fields when it catches a placeholder, preserving the original
        under extra.placeholder_fill_price -- precisely so no consumer can
        misread the row as a real $100 fill. A caught placeholder therefore
        never has fill_price == 100.0, and the old exterminator's equality test
        (exterminator.py:395) could not have matched one.
        """
        extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
        if extra.get("placeholder_fill_price") is not None:
            return True
        if str(extra.get("trust_state") or "").upper() == "PLACEHOLDER":
            return True
        tags = payload.get("tags")
        if isinstance(tags, list) and any(str(t).lower() == "placeholder" for t in tags):
            return True
        return False

    @staticmethod
    def _placeholder_contained(payload: dict[str, Any]) -> bool:
        """A placeholder is contained when it cannot reach a scored path."""
        if str(payload.get("status") or "").lower() == "rejected":
            return True
        if payload.get("reject") is True:
            return True
        return _payload_is_untrusted(payload)

    # ---- check 3: untrusted fills outside permitted stores ------------

    def check_untrusted_fills(self) -> CheckResult:
        cfg = (self.config.get("fills") or {}) if self.config else {}
        days = int(cfg.get("scan_days") or 7)
        permitted = list(cfg.get("permitted_stores") or [])
        scored = list(cfg.get("scored_stores") or [])

        leaks: list[dict[str, Any]] = []
        scanned = 0
        scanned_files: list[str] = []
        for pattern in scored:
            for path in self._recent_files(pattern, days):
                scanned_files.append(path.name)
                for record in self._iter_ndjson(path):
                    payload = record.get("payload") if isinstance(record.get("payload"), dict) else record
                    scanned += 1
                    untrusted = _payload_is_untrusted(payload)
                    validate_only = _payload_is_validate_only(payload)
                    if not (untrusted or validate_only):
                        continue
                    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
                    leaks.append({
                        "file": path.name,
                        "symbol": payload.get("symbol"),
                        "strategy": payload.get("strategy"),
                        "marker": "validate_only" if validate_only else "pnl_untrusted",
                        "reason": extra.get("pnl_untrusted_reason"),
                        "tags": payload.get("tags"),
                        "pnl": payload.get("pnl"),
                        # Rows derived from a symbol close carry no trade_id;
                        # close_key is the only stable handle on them.
                        "row_id": str(
                            payload.get("trade_id")
                            or payload.get("fill_id")
                            or extra.get("close_key")
                            or record.get("record_hash")
                            or ""
                        )[:16],
                    })

        evidence = {
            "scored_stores": scored,
            "permitted_stores": permitted,
            "rows_scanned": scanned,
            "scanned_files": scanned_files[:10],
            "scan_days": days,
        }
        if leaks:
            return CheckResult(
                "EXS3", "untrusted_fills", STATUS_FAIL, "Untrusted rows in a scored store",
                "validate_only / pnl_untrusted rows are sitting in a SCORED store. Presence is not "
                "proof they are being scored -- SCR may still drop them via a different exclusion "
                "bucket (manual/quarantine/futures), and the exclusion counters in scr_state.json "
                "say which. It IS proof the scored store is contaminated: the only thing standing "
                "between these rows and effective_trades is whichever bucket happens to match "
                "first, so a re-attribution could score them.",
                {**evidence, "leaks": leaks[:10], "leak_count": len(leaks),
                 "verify_with": "runtime/scr_state.json::stats.excluded_* counters",
                 "exclusion_order": "warmup -> manual -> validate_only -> untrusted "
                                    "(chad/analytics/trade_stats_engine.py:658)"},
            )
        if not scanned_files:
            return CheckResult(
                "EXS3", "untrusted_fills", STATUS_WARN, "No scored store to scan",
                "No scored trade-history file was modified in the scan window; leak scan proved nothing.",
                evidence,
            )
        return CheckResult(
            "EXS3", "untrusted_fills", STATUS_OK, "Scored stores clean",
            "No untrusted or validate_only rows leaked into a scored store. "
            "Permitted stores are not scanned: they legitimately hold marked rows.",
            evidence,
        )

    # ---- check 4: reconciliation drift (INDEPENDENT-LEG RULE) ---------

    def check_reconciliation_drift(self) -> CheckResult:
        snap, snap_err = self._read_json(self.runtime_dir / "positions_snapshot.json")
        guard, guard_err = self._read_json(self.runtime_dir / "position_guard.json")

        if not isinstance(guard, dict):
            return CheckResult(
                "EXS4", "reconciliation_drift", STATUS_WARN, "Position guard unreadable",
                "position_guard.json could not be read; guard drift cannot be judged.",
                {"error": guard_err},
            )

        # The independent leg must prove itself fresh before it can indict the
        # guard. A stale collector proves nothing -- degrade to warn (blind),
        # never to a false ok.
        if not isinstance(snap, dict):
            return CheckResult(
                "EXS4", "reconciliation_drift", STATUS_WARN, "Independent leg blind",
                "positions_snapshot.json (the independent collector leg) is unreadable, so guard "
                "truth cannot be independently verified. This is exactly the state in which "
                "same-source legs report a false GREEN.",
                {"error": snap_err, "independent_leg": "runtime/positions_snapshot.json"},
            )
        snap_age = self._age_seconds(self._parse_ts(snap.get("ts_utc")))
        snap_ttl = float(snap.get("ttl_seconds") or 300)
        if snap_age is None or snap_age > snap_ttl * 3:
            return CheckResult(
                "EXS4", "reconciliation_drift", STATUS_WARN, "Independent leg blind",
                "positions_snapshot.json is too stale to serve as independent broker truth; "
                "guard drift is unproven either way.",
                {"snapshot_age_seconds": snap_age, "ttl_seconds": snap_ttl,
                 "independent_leg": "runtime/positions_snapshot.json"},
            )

        broker = _aggregate_snapshot(snap)
        mirror = _aggregate_guard_broker_mirror(guard)
        strategy = _aggregate_guard_strategy(guard)

        drift_state, _ = self._read_json(self.runtime_dir / "position_guard_drift.json")
        excluded = set()
        if isinstance(drift_state, dict):
            excluded = {str(s).upper() for s in (drift_state.get("excluded_symbols") or [])}

        actionable: list[dict[str, Any]] = []
        info: list[dict[str, Any]] = []
        for symbol in sorted(set(broker) | set(mirror) | set(strategy)):
            b = broker.get(symbol, 0.0)
            m = mirror.get(symbol, 0.0)
            s = strategy.get(symbol, 0.0)
            mirror_delta = m - b
            strat_delta = s - b
            if abs(mirror_delta) <= QTY_TOLERANCE and abs(strat_delta) <= QTY_TOLERANCE:
                continue
            row = {
                "symbol": symbol,
                "independent_broker_qty": b,
                "guard_broker_mirror_qty": m,
                "guard_strategy_qty": s,
                "mirror_delta": round(mirror_delta, 6),
                "strategy_delta": round(strat_delta, 6),
            }
            # Mixed-ownership symbols blend operator shares with CHAD lots, so
            # a strategy-vs-broker delta there is not like-with-like. The v3
            # detector reports these as informational for the same reason
            # (chad/core/position_guard.py:686); inflating drift here would
            # re-introduce the false-RED class WKF U3 fixed.
            if symbol in excluded:
                row["drift_kind"] = "mixed_ownership_info"
                row["is_excluded"] = True
                info.append(row)
                continue
            if abs(mirror_delta) > QTY_TOLERANCE:
                # THE XOV-2345 SIGNATURE: the guard's own mirror of broker truth
                # disagrees with the independent collector.
                row["drift_kind"] = "mirror_vs_independent_broker"
                row["is_excluded"] = False
                actionable.append(row)
            elif abs(s) > QTY_TOLERANCE and abs(b) <= QTY_TOLERANCE:
                row["drift_kind"] = "phantom_guard_entry"
                row["is_excluded"] = False
                actionable.append(row)
            else:
                row["drift_kind"] = "strategy_attribution_info"
                row["is_excluded"] = True
                info.append(row)

        splits = self._guard_vs_fifo_splits(guard)
        leg_disagreement = self._leg_disagreement(actionable, drift_state)

        evidence: dict[str, Any] = {
            "independent_leg": "runtime/positions_snapshot.json (ibkr_portfolio_collector_v2, clientId=99)",
            "rejected_leg": "broker_sync alone (same-source; blinded us during XOV-2345)",
            "snapshot_age_seconds": round(snap_age, 1),
            "actionable_drifts": actionable[:10],
            "actionable_count": len(actionable),
            "info_drifts": info[:10],
            "info_count": len(info),
            "excluded_symbols": sorted(excluded),
            "guard_vs_fifo_splits": splits[:10],
            "split_count": len(splits),
        }
        if leg_disagreement:
            evidence["leg_disagreement"] = leg_disagreement

        if leg_disagreement:
            return CheckResult(
                "EXS4", "reconciliation_drift", STATUS_FAIL, "Same-source legs claim GREEN, independent leg disagrees",
                "The independent collector contradicts the guard while reconciliation_state and/or "
                "position_guard_drift report no drift. This is the XOV-2345 blindness signature: "
                "legs that share a source agreeing with each other is not evidence.",
                evidence,
            )
        if actionable:
            return CheckResult(
                "EXS4", "reconciliation_drift", STATUS_FAIL, "Guard disagrees with independent broker truth",
                "The position guard disagrees with the independent collector on one or more symbols.",
                evidence,
            )
        if splits:
            return CheckResult(
                "EXS4", "reconciliation_drift", STATUS_WARN, "Guard and FIFO disagree internally",
                "Guard open entries and the trade_closer FIFO queues disagree on quantity.",
                evidence,
            )
        return CheckResult(
            "EXS4", "reconciliation_drift", STATUS_OK, "Guard agrees with independent broker truth",
            "The guard matches the independent collector on every symbol, and the FIFO agrees with the guard.",
            evidence,
        )

    def _guard_vs_fifo_splits(self, guard: dict[str, Any]) -> list[dict[str, Any]]:
        """Flag guard-vs-FIFO internal disagreement, keyed per (strategy, symbol)."""
        tc, _ = self._read_json(self.runtime_dir / "trade_closer_state.json")
        if not isinstance(tc, dict):
            return []
        queues = tc.get("queues")
        if not isinstance(queues, list):
            return []
        fifo: dict[tuple[str, str], float] = {}
        for queue in queues:
            if not isinstance(queue, dict):
                continue
            key = (str(queue.get("strategy") or "").strip(), str(queue.get("symbol") or "").strip().upper())
            total = 0.0
            for lot in queue.get("lots") or []:
                if not isinstance(lot, dict):
                    continue
                try:
                    qty = abs(float(lot.get("quantity") or 0.0))
                except (TypeError, ValueError):
                    continue
                total += -qty if str(lot.get("side") or "").upper() in ("SELL", "SHORT") else qty
            fifo[key] = fifo.get(key, 0.0) + total

        guard_by_key: dict[tuple[str, str], float] = {}
        for key, entry in guard.items():
            if not isinstance(key, str) or key.startswith("_") or not isinstance(entry, dict):
                continue
            if key.startswith("broker_sync|") or entry.get("open") is not True:
                continue
            k = (str(entry.get("strategy") or "").strip(), str(entry.get("symbol") or "").strip().upper())
            guard_by_key[k] = guard_by_key.get(k, 0.0) + _signed_qty(entry)

        splits: list[dict[str, Any]] = []
        for key in sorted(set(fifo) | set(guard_by_key)):
            g, f = guard_by_key.get(key, 0.0), fifo.get(key, 0.0)
            if abs(g - f) > QTY_TOLERANCE:
                splits.append({
                    "strategy": key[0], "symbol": key[1],
                    "guard_qty": g, "fifo_qty": f, "delta": round(g - f, 6),
                })
        return splits

    @staticmethod
    def _leg_disagreement(actionable: list[dict[str, Any]], drift_state: Any) -> dict[str, Any] | None:
        """Detect same-source legs reporting clean while the independent leg indicts."""
        if not actionable:
            return None
        claims: dict[str, Any] = {}
        if isinstance(drift_state, dict):
            try:
                drift_count = int(drift_state.get("drift_count") or 0)
            except (TypeError, ValueError):
                drift_count = 0
            if drift_count == 0:
                claims["position_guard_drift.json"] = {
                    "drift_count": 0,
                    "why_blind": "both of its legs are read from position_guard.json (chad/core/position_guard.py:589)",
                }
        if not claims:
            return None
        return {
            "independent_leg_finds": len(actionable),
            "same_source_legs_claim_clean": claims,
        }

    # ---- check 5: failed services -------------------------------------

    def check_failed_services(self) -> CheckResult:
        cfg = (self.config.get("services") or {}) if self.config else {}
        query = list(cfg.get("query") or ["systemctl", "list-units", "--state=failed", "--no-pager", "--no-legend", "--plain"])
        prefix = str(cfg.get("unit_prefix") or "chad-")
        info = self.systemctl_provider(query)
        if info.get("error"):
            return CheckResult(
                "EXS5", "failed_services", STATUS_WARN, "systemd probe failed",
                "Could not enumerate failed units.",
                {"error": info.get("error"), "query": query},
            )
        units = [u for u in (info.get("failed_units") or []) if str(u).startswith(prefix)]
        others = [u for u in (info.get("failed_units") or []) if not str(u).startswith(prefix)]
        if units:
            return CheckResult(
                "EXS5", "failed_services", STATUS_FAIL, "Failed CHAD units",
                "One or more CHAD systemd units are in the failed state.",
                {"failed_units": units, "non_chad_failed_units": others, "query": query,
                 "operator_action": "inspect journalctl -u <unit>; the sentinel never restarts a unit"},
            )
        return CheckResult(
            "EXS5", "failed_services", STATUS_OK, "No failed CHAD units",
            "No CHAD systemd unit is in the failed state.",
            {"non_chad_failed_units": others, "query": query},
        )

    # ---- check 6: dirty git -------------------------------------------

    def check_dirty_git(self) -> CheckResult:
        cfg = (self.config.get("git") or {}) if self.config else {}
        production = tuple(cfg.get("production_paths") or ())
        allowlist = list(cfg.get("allowlist") or [])
        info = self.git_provider()
        if info.get("error"):
            return CheckResult(
                "EXS6", "dirty_git", STATUS_WARN, "git probe failed",
                "Could not determine repository state.",
                {"error": info.get("error")},
            )

        dirty: list[dict[str, Any]] = []
        allowed: list[dict[str, Any]] = []
        for line in info.get("entries") or []:
            code, _, path = line[:2], line[2:3], line[3:].strip()
            if not path:
                continue
            if code.startswith("??"):
                continue  # untracked files are not tracked-file drift
            entry = {"code": code, "path": path}
            match = _allowlist_match(code, path, allowlist)
            if match is not None:
                entry["allowlisted_reason"] = match
                allowed.append(entry)
                continue
            if production and not path.startswith(production):
                continue
            dirty.append(entry)

        evidence = {
            "head": info.get("head"), "branch": info.get("branch"),
            "production_paths": list(production),
            "dirty": dirty[:20], "dirty_count": len(dirty),
            "allowlisted_count": len(allowed),
            "allowlisted_sample": allowed[:3],
        }
        if dirty:
            return CheckResult(
                "EXS6", "dirty_git", STATUS_WARN, "Uncommitted changes in production paths",
                "Tracked files in production paths have uncommitted modifications; "
                "the running code may not match any commit.",
                evidence,
            )
        return CheckResult(
            "EXS6", "dirty_git", STATUS_OK, "Production paths clean",
            "No uncommitted tracked changes in production paths (allowlisted entries excluded).",
            evidence,
        )

    # ---- check 7: schema breaks ---------------------------------------

    def check_schema_breaks(self) -> CheckResult:
        cfg = (self.config.get("schema_contracts") or {}) if self.config else {}
        enforced = cfg.get("enforced") or {}
        unpinned_known = cfg.get("unpinned_known") or {}

        breaks: list[dict[str, Any]] = []
        checked: list[str] = []
        for rel_path, contract in sorted(enforced.items()):
            if not isinstance(contract, dict):
                continue
            path = self.repo_root / rel_path
            payload, err = self._read_json(path)
            if err == "missing":
                breaks.append({"file": rel_path, "break": "missing"})
                continue
            if not isinstance(payload, dict):
                breaks.append({"file": rel_path, "break": "unreadable_or_not_an_object", "error": err})
                continue
            checked.append(rel_path)
            accepts = [str(a) for a in (contract.get("accepts") or [contract.get("schema_version")])]
            actual = payload.get("schema_version")
            if actual is None:
                breaks.append({"file": rel_path, "break": "schema_version_absent",
                               "expected_one_of": accepts, "pinned_at": contract.get("pinned_at")})
            elif str(actual) not in accepts:
                breaks.append({"file": rel_path, "break": "schema_version_unrecognised",
                               "actual": actual, "expected_one_of": accepts,
                               "pinned_at": contract.get("pinned_at")})
            missing_keys = [k for k in (contract.get("required_keys") or []) if k not in payload]
            if missing_keys:
                breaks.append({"file": rel_path, "break": "required_keys_missing",
                               "missing_keys": missing_keys, "pinned_at": contract.get("pinned_at")})

        evidence = {
            "enforced_contracts": len(enforced),
            "files_checked": checked,
            "breaks": breaks[:10],
            "break_count": len(breaks),
            "unpinned_known": unpinned_known,
            "unpinned_known_count": len(unpinned_known),
        }
        if breaks:
            return CheckResult(
                "EXS7", "schema_breaks", STATUS_FAIL, "Runtime schema contract broken",
                "A runtime JSON contract violates its pinned schema_version or is missing required keys.",
                evidence,
            )
        if unpinned_known:
            return CheckResult(
                "EXS7", "schema_breaks", STATUS_WARN, "Runtime contracts without a pinned schema",
                "All enforced contracts hold. Separately, some runtime contracts have no pinned "
                "schema_version at all and cannot be validated -- a pre-existing gap, reported "
                "rather than silently passed.",
                evidence,
            )
        return CheckResult(
            "EXS7", "schema_breaks", STATUS_OK, "Runtime schema contracts hold",
            "Every enforced runtime contract matches its pinned schema_version and required keys.",
            evidence,
        )

    # ---- check 8: ML anomalies ----------------------------------------

    def check_ml_anomalies(self) -> CheckResult:
        cfg = (self.config.get("ml") or {}) if self.config else {}
        manifest: dict[str, Any] | None = None
        manifest_path: str | None = None
        for rel in cfg.get("manifest_paths") or []:
            payload, _ = self._read_json(self.repo_root / str(rel))
            if isinstance(payload, dict):
                manifest, manifest_path = payload, str(rel)
                break

        if manifest is None:
            return CheckResult(
                "EXS8", "ml_anomalies", STATUS_WARN, "ML manifest absent",
                "No xgb veto manifest found, so neither model version nor veto rate can be established.",
                {"searched": list(cfg.get("manifest_paths") or [])},
            )

        version = manifest.get(str(cfg.get("model_version_key") or "model_version"))
        metrics = manifest.get("metrics") if isinstance(manifest.get("metrics"), dict) else {}
        training_stat = metrics.get(str(cfg.get("training_stat_key") or "val_veto_rate_at_0.65"))
        baseline = cfg.get("baseline_veto_rate")

        evidence: dict[str, Any] = {
            "manifest_path": manifest_path,
            "model_version": version,
            "model_version_present": version is not None,
            "baseline_veto_rate": baseline,
            "baseline_source": cfg.get("baseline_veto_rate_source"),
            "training_stat_only": {cfg.get("training_stat_key"): training_stat},
        }

        # Manifest age is a real, citable signal even without a baseline.
        stale_days = float(cfg.get("manifest_stale_days") or 30)
        trained_ts = self._parse_ts(manifest.get("trained_at_utc") or manifest.get("ts_utc"))
        version_age_days: float | None = None
        if trained_ts is not None:
            age = self._age_seconds(trained_ts)
            version_age_days = round(age / 86400.0, 1) if age is not None else None
        elif isinstance(version, str):
            match = re.search(r"(\d{8})", version)
            if match:
                try:
                    stamp = datetime.strptime(match.group(1), "%Y%m%d").replace(tzinfo=timezone.utc)
                    age = self._age_seconds(stamp)
                    version_age_days = round(age / 86400.0, 1) if age is not None else None
                except ValueError:
                    version_age_days = None
        evidence["model_version_age_days"] = version_age_days
        evidence["manifest_stale_days"] = stale_days
        evidence["manifest_stale_days_source"] = cfg.get("manifest_stale_days_source")

        if version is None:
            return CheckResult(
                "EXS8", "ml_anomalies", STATUS_WARN, "ML model version missing",
                "The xgb veto manifest declares no model_version.",
                evidence,
            )

        # No production veto rate is recorded anywhere -- veto decisions leave
        # only log lines. Comparing against the manifest's training-time
        # val_veto_rate would be a category error, so say so plainly.
        if baseline is None:
            evidence["no_baseline_detail"] = (
                "No production veto-rate baseline exists. Veto decisions emit ML_SHADOW/ML_VETO log "
                "lines only -- no counter, no runtime artifact -- so no live rate can be computed or "
                "compared. The manifest's val_veto_rate is a training-time validation statistic, not "
                "a production baseline."
            )
            if version_age_days is not None and version_age_days > stale_days:
                evidence["manifest_stale"] = True
                return CheckResult(
                    "EXS8", "ml_anomalies", STATUS_WARN, "no_baseline",
                    "No veto-rate baseline exists, so veto drift cannot be detected. The model "
                    "manifest is also older than its own staleness threshold, which would refuse "
                    "enforcement anyway.",
                    evidence,
                )
            return CheckResult(
                "EXS8", "ml_anomalies", STATUS_WARN, "no_baseline",
                "No veto-rate baseline exists, so veto drift cannot be detected. Model version is present.",
                evidence,
            )

        return CheckResult(
            "EXS8", "ml_anomalies", STATUS_OK, "ML veto within baseline",
            "Model version present and a veto-rate baseline is configured.",
            evidence,
        )

    # ---- assembly ------------------------------------------------------

    def run(self) -> dict[str, Any]:
        checks: list[CheckResult] = []

        def _safe(label: str, fn: Callable[[], CheckResult]) -> None:
            try:
                checks.append(fn())
            except Exception as err:  # one bad check must never abort the report
                checks.append(CheckResult(
                    "EXS999", f"sentinel_self:{label}", STATUS_WARN,
                    "Sentinel check raised",
                    f"Check {label!r} raised {type(err).__name__}; the report is incomplete.",
                    {"check": label, "error": repr(err)},
                ))

        _safe("stale_feeds", self.check_stale_feeds)
        _safe("placeholder_fills", self.check_placeholder_fills)
        _safe("untrusted_fills", self.check_untrusted_fills)
        _safe("reconciliation_drift", self.check_reconciliation_drift)
        _safe("failed_services", self.check_failed_services)
        _safe("dirty_git", self.check_dirty_git)
        _safe("schema_breaks", self.check_schema_breaks)
        _safe("ml_anomalies", self.check_ml_anomalies)

        counts = {STATUS_OK: 0, STATUS_WARN: 0, STATUS_FAIL: 0}
        for check in checks:
            if check.status in counts:
                counts[check.status] += 1
        overall = worst_status(c.status for c in checks)

        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": self.clock().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "mode": "read_only",
            "stage": 1,
            "overall_status": overall,
            "counts": counts,
            "checks": [c.to_dict() for c in checks],
            "read_only_confirmed": True,
            "runtime_files_modified": [],
            "services_restarted": False,
            "remedy_type": REMEDY_NOTIFY_ONLY,
        }

    # ---- writers -------------------------------------------------------

    def report_paths(self) -> tuple[Path, Path]:
        return self.reports_dir / LATEST_FILENAME, self.reports_dir / HISTORY_FILENAME

    def write_reports(self, report: dict[str, Any]) -> tuple[Path, Path]:
        """Write the two artifacts this module is permitted to write. Nothing else."""
        latest_path, history_path = self.report_paths()
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        tmp = latest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
        os.replace(tmp, latest_path)

        # History keeps one compact line per run: the verdicts, not the evidence.
        row = {
            "ts_utc": report["generated_at_utc"],
            "schema_version": SCHEMA_VERSION,
            "overall_status": report["overall_status"],
            "counts": report["counts"],
            "checks": {c["check_id"]: c["status"] for c in report["checks"]},
        }
        with history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        return latest_path, history_path

    # ---- notification --------------------------------------------------

    def maybe_notify(self, report: dict[str, Any]) -> bool:
        """Send a coach-voiced Telegram summary ONLY when a check failed."""
        failed = [c for c in report["checks"] if c["status"] == STATUS_FAIL]
        if not failed:
            return False

        # Stable by construction: check_ids never fluctuate, so the key is the
        # same every cycle for the same set of failures (CTF-T2). stable_identity
        # is applied as a belt-and-braces guard on the title text.
        ids = "_".join(sorted(c["check_id"] for c in failed))
        dedupe_key = f"exterminator_sentinel_{stable_identity(ids) or 'fail'}"

        titles = "; ".join(c["title"] for c in failed)
        message = _coach_message(failed, titles)
        return bool(self.notifier(message, dedupe_key))


def _coach_message(failed: list[dict[str, Any]], titles: str) -> str:
    """Format the alert in coach voice, falling back to plain text."""
    try:
        from chad.utils.coach_voice import format_alert

        return format_alert(
            "health_finding",
            {
                "rule_id": ",".join(c["check_id"] for c in failed),
                "severity": "CRITICAL",
                "title": titles,
                "description": failed[0]["summary"],
                "evidence": f"{len(failed)} sentinel check(s) failed",
                "remedy_type": REMEDY_NOTIFY_ONLY,
                "remedy_action": "notify",
            },
        )
    except Exception:
        return f"🚨 Exterminator sentinel: {len(failed)} check(s) failed — {titles}"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _payload_is_untrusted(payload: dict[str, Any]) -> bool:
    """Canonical untrusted predicate, mirroring chad/utils/quarantine.py:123."""
    if not isinstance(payload, dict):
        return False
    if payload.get("pnl_untrusted") is True:
        return True
    extra = payload.get("extra")
    if isinstance(extra, dict) and extra.get("pnl_untrusted") is True:
        return True
    tags = payload.get("tags")
    if isinstance(tags, list) and any(str(t).lower() == "pnl_untrusted" for t in tags):
        return True
    return False


def _payload_is_validate_only(payload: dict[str, Any]) -> bool:
    """Mirrors chad/analytics/trade_stats_engine.py:129."""
    if not isinstance(payload, dict):
        return False
    if payload.get("validate_only") is True:
        return True
    extra = payload.get("extra")
    if isinstance(extra, dict) and extra.get("validate_only") is True:
        return True
    tags = payload.get("tags")
    if isinstance(tags, list) and any(str(t).lower() == "validate_only" for t in tags):
        return True
    return False


def _signed_qty(entry: dict[str, Any]) -> float:
    try:
        qty = abs(float(entry.get("quantity") or 0.0))
    except (TypeError, ValueError):
        return 0.0
    return -qty if str(entry.get("side") or "").upper() in ("SELL", "SHORT") else qty


def _aggregate_snapshot(snap: dict[str, Any]) -> dict[str, float]:
    """Signed positions per symbol from the independent collector."""
    out: dict[str, float] = {}
    for row in snap.get("positions") or []:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        try:
            qty = float(row.get("position") or 0.0)
        except (TypeError, ValueError):
            continue
        out[symbol] = out.get(symbol, 0.0) + qty
    return {k: v for k, v in out.items() if abs(v) > QTY_TOLERANCE}


def _aggregate_guard_broker_mirror(guard: dict[str, Any]) -> dict[str, float]:
    """The guard's recorded mirror of broker truth, per symbol.

    Counts broker_sync rows REGARDLESS of the open flag: the broker-truth
    rebuild writes the last-known broker quantity and marks the row open=False
    (closed_by="strategy_ownership_assumed"), and that quantity IS the mirror.
    Gating on open here is the exact bug the v1 drift detector had
    (chad/core/position_guard.py:580) -- it saw zero broker truth and cried
    broker_truth_missing for symbols the broker plainly held.
    """
    out: dict[str, float] = {}
    for key, entry in guard.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            continue
        if not key.startswith("broker_sync|"):
            continue
        symbol = str(entry.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        out[symbol] = out.get(symbol, 0.0) + _signed_qty(entry)
    return {k: v for k, v in out.items() if abs(v) > QTY_TOLERANCE}


def _aggregate_guard_strategy(guard: dict[str, Any]) -> dict[str, float]:
    """Strategy-attributed OPEN guard quantity per symbol.

    Deliberately NOT summed with the broker_sync mirror: the guard dual-books
    the same shares (gamma|UNH 273 AND broker_sync|UNH 273 for one 273-share
    position), so adding the legs double-counts every mixed symbol and invents
    a 2.0x phantom.
    """
    out: dict[str, float] = {}
    for key, entry in guard.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            continue
        if key.startswith("_") or key.startswith("broker_sync|"):
            continue
        if entry.get("open") is not True:
            continue
        symbol = str(entry.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        out[symbol] = out.get(symbol, 0.0) + _signed_qty(entry)
    return {k: v for k, v in out.items() if abs(v) > QTY_TOLERANCE}


def _allowlist_match(code: str, path: str, allowlist: list[Any]) -> str | None:
    for item in allowlist:
        if not isinstance(item, dict):
            continue
        pattern = str(item.get("pattern") or "")
        if not pattern or pattern not in path:
            continue
        codes = item.get("status_codes")
        if isinstance(codes, list) and codes and not any(code == str(c) or code.strip() == str(c).strip() for c in codes):
            continue
        return str(item.get("reason") or "allowlisted")
    return None


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> int:
    sentinel = ExterminatorSentinel()
    report = sentinel.run()
    latest_path, history_path = sentinel.write_reports(report)
    notified = sentinel.maybe_notify(report)
    counts = report["counts"]
    print(
        f"exterminator_sentinel: overall={report['overall_status']} "
        f"ok={counts['ok']} warn={counts['warn']} fail={counts['fail']} "
        f"notified={notified} latest={latest_path} history={history_path}"
    )
    # Exit code reports the scan verdict, not process health: a sentinel that
    # exits non-zero on a finding would put its own unit into `failed` and trip
    # check 5 on the next run.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
