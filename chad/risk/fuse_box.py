"""
chad/risk/fuse_box.py — W4A fuse box core (LC2/LC3 trip engine + state publisher).

One spine, four fuses (PLAN_W4A §2). This module is the CORE only: trusted-loss
counting, the bucket trip/clear engine, and the sentinel-visible state publisher.
Bucket builders (family/setup — W4A-3; symbol/sector — W4A-4), the per-intent
gate (W4A-5), LC5 drawdown enforcement (W4A-6/7) and DQ policies (W4A-8) layer
on top. Nothing here is wired into live_loop until W4A-5; nothing here ever
blocks a close (fuses gate ENTRIES only — the prime invariant is enforced at
the gate layer by predicate AND structurally by placement, PLAN_W4A §8.1).

Counting doctrine (PLAN_W4A §3 + INCIDENT-0723 inheritance (a), GO record §5):
a fuse counter may only ever move on a PROVEN-trusted closed trade. The engine
re-derives per-bucket session stats from the trusted ledger every cycle (the
stateless GAP-026 pattern — chad/risk/per_strategy_loss_guard.py) and diffs
against the previously published state for edge-triggered trip/clear events.

Trust predicates, in order (each exclusion is tallied under its reason):
  non_closed_trade      — schema_version must start with "closed_trade."
  out_of_window         — exit_time_utc outside [session window start, now]
  quarantined           — chad.utils.quarantine exclusion sets (operator
                          manifests + untrusted-fills scan + seed lots +
                          sidecars), matched on record_hash / payload.fill_id /
                          any element of payload.fill_ids (the
                          trade_stats_engine idiom)
  <trust_exclusion>     — chad.validation.trade_log_adapter.trust_exclusion:
                          placeholder_100 / broker_rejected / non_fill_status /
                          validate_only / pnl_untrusted / scoring_excluded /
                          manual / warmup_sim (SCR-parity predicates)
  strategy_excluded     — strategy ∈ {broker_sync, paper_exec, unknown, ""}
                          (manual is caught upstream by trust_exclusion)
  futures_bug_b         — instrument classifies FUT (Bug-B contamination;
                          adapter/SCR precedent) until the Bug-B disposition
  unverified_provenance — INCIDENT-0723, BY CONSTRUCTION: any cited fill_id
                          that resolves in the session window's FILLS_*.ndjson
                          files with a non-genuine status (∉ {paper_fill,
                          fill, filled}) condemns the row. A drill/rehearsal
                          (status=dry_run) exhaust row can therefore never
                          move a fuse counter even if an upstream writer
                          re-blesses it — the fuse re-verifies provenance
                          itself. Covers both incident shapes: dry_run exit
                          leg (the 8 fake 07-23 rows) and dry_run entry leg
                          with a genuine exit fill (the 13:31:30 PSQ row).

Regime scoping (D2 rider, GO record §2): a row's regime comes from the
forward-only stamp (W4A-2); absent/unrecognised ⇒ "unknown". Unknown rows
count toward GLOBAL bucket legs only — a regime-scoped leg matches exact
stamps and can NEVER count an unknown row, even if a config lists "unknown"
in a regimes scope (validation strips it, and the filter refuses it as a
structural belt).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "fuse_box.json"
DEFAULT_STATE_PATH = REPO_ROOT / "runtime" / "fuse_box_state.json"
DEFAULT_EVIDENCE_DIR = REPO_ROOT / "data" / "fuse_box"
DEFAULT_TRADES_DIR = REPO_ROOT / "data" / "trades"
DEFAULT_FILLS_DIR = REPO_ROOT / "data" / "fills"
DEFAULT_RUNTIME_DIR = REPO_ROOT / "runtime"
EPOCH_STATE_PATH = DEFAULT_RUNTIME_DIR / "epoch_state.json"

STATE_SCHEMA_VERSION = "fuse_box_state.v1"
# Cycle cadence (~60s) × 3 grace — pinned in config/exterminator.json feeds row.
STATE_TTL_SECONDS = 180

# Tri-state mode flags (house convention: garbage → off).
MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_ENFORCE = "enforce"
_VALID_MODES = frozenset({MODE_OFF, MODE_SHADOW, MODE_ENFORCE})

ENV_LC2 = "CHAD_FUSE_LC2"
ENV_LC3 = "CHAD_FUSE_LC3"
ENV_LC5 = "CHAD_FUSE_LC5"
ENV_DQ = "CHAD_DQ_POLICIES"

# INCIDENT-0723 inheritance (a): the fuse box is consumer #7 of the W4B-8f
# exclusion census. Lowercased statuses that count as a genuine executed fill —
# the union of every pinned consumer allowlist (trade_closer:352,
# position_guard:33, trade_log_adapter:95, ibkr_paper_ledger_watcher:315).
# Must stay disjoint from position_reconciler._EVIDENCE_SKIP_FILL_STATUSES —
# pinned in chad/tests/test_w4b8_exhaust_hygiene_sites.py.
GENUINE_FILL_STATUSES = frozenset({"paper_fill", "fill", "filled"})

# Predicate 4 (PLAN_W4A §3): non-strategy attribution rows are not edge
# evidence. "manual" is excluded upstream by trust_exclusion (SCR parity).
EXCLUDED_STRATEGIES = frozenset({"broker_sync", "paper_exec", "unknown", ""})

# Live classifier vocabulary (chad/analytics/regime_classifier.py VALID_REGIMES).
# "unknown" is the D2-rider bucket for unstamped/unrecognised rows.
KNOWN_REGIMES = frozenset(
    {"trending_bull", "trending_bear", "ranging", "volatile", "unknown"}
)

_TRADE_FILE_RE = re.compile(r"^trade_history_(\d{8})\.ndjson$")
_FILLS_FILE_RE = re.compile(r"^FILLS_(\d{8})\.ndjson$")


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #

def fuse_mode(env_var: str, env: Optional[Mapping[str, str]] = None) -> str:
    """Tri-state env parse: off | shadow | enforce; anything else → off."""
    src = env if env is not None else os.environ
    raw = str(src.get(env_var, "")).strip().lower()
    return raw if raw in _VALID_MODES else MODE_OFF


def read_modes(env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    return {
        "lc2": fuse_mode(ENV_LC2, env),
        "lc3": fuse_mode(ENV_LC3, env),
        "lc5": fuse_mode(ENV_LC5, env),
        "dq": fuse_mode(ENV_DQ, env),
    }


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class FuseBoxConfig:
    """Parsed config/fuse_box.json. Missing/corrupt file → safe defaults
    (no families, no thresholds overridden) — a broken config can disarm the
    fuse box (report-only posture) but can never invent a trip."""

    default_consecutive_losers: int
    default_session_net_pnl_usd: Optional[float]
    families: Dict[str, Tuple[str, ...]]
    setup_fuse_strategies: Tuple[str, ...]
    symbol_consecutive_losers: int
    sector_consecutive_losers: int
    lc5_ladder: Dict[str, Any]
    raw: Dict[str, Any]

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "FuseBoxConfig":
        target = Path(path) if path is not None else CONFIG_PATH
        try:
            obj = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(obj, dict):
                obj = {}
        except Exception:
            obj = {}
        defaults = obj.get("defaults") or {}
        try:
            n_default = int(defaults.get("consecutive_losers", 3))
        except (TypeError, ValueError):
            n_default = 3
        pnl_default: Optional[float]
        try:
            v = defaults.get("session_net_pnl_usd")
            pnl_default = float(v) if v is not None else None
        except (TypeError, ValueError):
            pnl_default = None

        families: Dict[str, Tuple[str, ...]] = {}
        fam_raw = obj.get("families") or {}
        if isinstance(fam_raw, dict):
            for name, members in fam_raw.items():
                if isinstance(members, list):
                    families[str(name)] = tuple(
                        str(m).strip().lower() for m in members if str(m).strip()
                    )

        setup_raw = (obj.get("setup_fuses") or {}).get("enabled_strategies") or []
        setup_strategies = tuple(
            str(s).strip().lower() for s in setup_raw if str(s).strip()
        ) if isinstance(setup_raw, list) else ()

        def _n(section: str, fallback: int) -> int:
            try:
                return int((obj.get(section) or {}).get("consecutive_losers", fallback))
            except (TypeError, ValueError):
                return fallback

        ladder = obj.get("lc5_ladder") or {}
        return cls(
            default_consecutive_losers=n_default,
            default_session_net_pnl_usd=pnl_default,
            families=families,
            setup_fuse_strategies=setup_strategies,
            symbol_consecutive_losers=_n("symbol_fuse", n_default),
            sector_consecutive_losers=_n("sector_fuse", n_default),
            lc5_ladder=ladder if isinstance(ladder, dict) else {},
            raw=obj,
        )


# --------------------------------------------------------------------------- #
# Session window (GAP-026 pattern: max(UTC midnight, epoch start))
# --------------------------------------------------------------------------- #

def _parse_iso(v: Any) -> Optional[datetime]:
    if not v:
        return None
    try:
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def read_epoch_start(epoch_state_path: Optional[Path] = None) -> Optional[datetime]:
    target = Path(epoch_state_path) if epoch_state_path is not None else EPOCH_STATE_PATH
    try:
        obj = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None
    return _parse_iso(obj.get("epoch_started_at_utc"))


def session_window_start(
    now: Optional[datetime] = None,
    epoch_state_path: Optional[Path] = None,
) -> datetime:
    """max(UTC midnight today, epoch_started_at_utc) — the GAP-026 window."""
    now_utc = now or datetime.now(timezone.utc)
    midnight = datetime(
        now_utc.year, now_utc.month, now_utc.day, tzinfo=timezone.utc
    )
    epoch_start = read_epoch_start(epoch_state_path)
    if epoch_start is None:
        return midnight
    return max(midnight, epoch_start)


# --------------------------------------------------------------------------- #
# Trusted-close loading
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class TrustedClose:
    strategy: str
    symbol: str
    side: str
    pnl: float
    exit_ts: datetime
    regime: str          # normalized; "unknown" when unstamped/unrecognised
    setup_family: Optional[str]
    fill_ids: Tuple[str, ...]


def _dates_in_window(window_start: datetime, now: datetime) -> List[str]:
    """YYYYMMDD strings from window start through *now* (UTC), inclusive."""
    out: List[str] = []
    day = datetime(
        window_start.year, window_start.month, window_start.day,
        tzinfo=timezone.utc,
    )
    while day.date() <= now.date():
        out.append(day.strftime("%Y%m%d"))
        day = day + timedelta(days=1)
    return out


def load_window_fill_statuses(
    window_start: datetime,
    now: datetime,
    fills_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """Map fill_id → lowercased status from the session window's FILLS files.

    INCIDENT-0723 inheritance (a): this is the provenance substrate. Only
    window-dated files are read (bounded — entry fills older than the window
    are vetted by the quarantine/trust belts instead, and cannot condemn).
    """
    src = Path(fills_dir) if fills_dir is not None else DEFAULT_FILLS_DIR
    out: Dict[str, str] = {}
    if not src.is_dir():
        return out
    wanted = set(_dates_in_window(window_start, now))
    for path in sorted(src.iterdir()):
        m = _FILLS_FILE_RE.match(path.name)
        if not m or m.group(1) not in wanted:
            continue
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            payload = rec.get("payload", rec)
            if not isinstance(payload, Mapping):
                continue
            fid = str(payload.get("fill_id") or "").strip()
            if not fid:
                continue
            status = str(payload.get("status") or "").strip().lower()
            # Last write wins — FILLS rows are append-only; a fill_id that
            # appears twice (harvester double-write, incident D7) keeps the
            # later row's status. Either copy of a genuine fill is genuine.
            out[fid] = status
    return out


def _norm_regime(v: Any) -> str:
    s = str(v or "").strip().lower()
    return s if s in KNOWN_REGIMES and s != "unknown" else "unknown"


def load_trusted_session_closes(
    now: Optional[datetime] = None,
    *,
    trades_dir: Optional[Path] = None,
    fills_dir: Optional[Path] = None,
    runtime_dir: Optional[Path] = None,
    epoch_state_path: Optional[Path] = None,
) -> Tuple[List[TrustedClose], Dict[str, int]]:
    """Load session-window closed trades that pass EVERY trust predicate.

    Returns (trusted closes sorted by exit ts, exclusion tally by reason).
    Fail-safe: unreadable substrate yields an empty list — a fuse without
    evidence stays untripped (counters only ever move on proven data).
    """
    end = now or datetime.now(timezone.utc)
    window_start = session_window_start(end, epoch_state_path)
    src_dir = Path(trades_dir) if trades_dir is not None else DEFAULT_TRADES_DIR
    tally: Dict[str, int] = {}
    closes: List[TrustedClose] = []

    def _count(reason: str) -> None:
        tally[reason] = tally.get(reason, 0) + 1

    if not src_dir.is_dir():
        return closes, tally

    # Quarantine sets — the SCR idiom (loss-guard precedent: failure here must
    # not raise; it degrades to manifest-less sets, never blocks the cycle).
    try:
        from chad.utils.quarantine import get_exclusion_sets, is_record_quarantined
        bad_fills, bad_hashes = get_exclusion_sets(
            runtime_dir=Path(runtime_dir) if runtime_dir is not None else DEFAULT_RUNTIME_DIR,
            fills_dir=Path(fills_dir) if fills_dir is not None else DEFAULT_FILLS_DIR,
            trades_dir=src_dir,
        )
    except Exception:
        bad_fills, bad_hashes = set(), set()
        is_record_quarantined = None  # type: ignore[assignment]

    # SCR-parity trust gate + instrument classifier (import-isolated module).
    from chad.validation.trade_log_adapter import (
        InstrumentClass,
        classify_instrument,
        trust_exclusion,
    )

    fill_statuses = load_window_fill_statuses(window_start, end, fills_dir)

    wanted_dates = set(_dates_in_window(window_start, end))
    for path in sorted(src_dir.iterdir()):
        m = _TRADE_FILE_RE.match(path.name)
        if not m or m.group(1) not in wanted_dates:
            continue
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, Mapping):
                continue
            payload = rec.get("payload", rec)
            if not isinstance(payload, Mapping):
                continue

            # 1. Structural: canonical closed round-trip only.
            schema = str(payload.get("schema_version") or "")
            if not schema.startswith("closed_trade."):
                _count("non_closed_trade")
                continue

            # 2. Window.
            ts = _parse_iso(payload.get("exit_time_utc"))
            if ts is None or ts < window_start or ts > end:
                _count("out_of_window")
                continue

            # 3. Quarantine pins (record_hash / fill_id / fill_ids).
            if is_record_quarantined is not None and is_record_quarantined(
                rec, bad_fills, bad_hashes
            ):
                _count("quarantined")
                continue

            # 4. SCR-parity trust gate.
            reason = trust_exclusion(rec)
            if reason is not None:
                _count(reason)
                continue

            # 5. Attribution.
            strategy = str(payload.get("strategy") or "").strip().lower()
            if strategy in EXCLUDED_STRATEGIES:
                _count("strategy_excluded")
                continue

            # 6. Futures (Bug-B) — adapter/SCR precedent.
            extra = payload.get("extra") if isinstance(payload.get("extra"), Mapping) else {}
            if classify_instrument(payload, extra) is InstrumentClass.FUT:
                _count("futures_bug_b")
                continue

            # 7. INCIDENT-0723: provenance verification, by construction.
            fill_ids = tuple(
                str(f).strip() for f in (payload.get("fill_ids") or []) if str(f).strip()
            )
            condemned = False
            for fid in fill_ids:
                status = fill_statuses.get(fid)
                if status is not None and status not in GENUINE_FILL_STATUSES:
                    condemned = True
                    break
            if condemned:
                _count("unverified_provenance")
                continue

            meta = payload.get("meta") if isinstance(payload.get("meta"), Mapping) else {}
            setup_family = meta.get("setup_family")
            pnl_raw = payload.get("net_pnl", payload.get("pnl"))
            try:
                pnl = float(pnl_raw)
            except (TypeError, ValueError):
                _count("malformed_pnl")
                continue

            closes.append(
                TrustedClose(
                    strategy=strategy,
                    symbol=str(payload.get("symbol") or "").strip().upper(),
                    side=str(payload.get("side") or "").strip().upper(),
                    pnl=pnl,
                    exit_ts=ts,
                    regime=_norm_regime(payload.get("regime")),
                    setup_family=(
                        str(setup_family).strip() if setup_family else None
                    ),
                    fill_ids=fill_ids,
                )
            )

    closes.sort(key=lambda c: c.exit_ts)
    return closes, tally


# --------------------------------------------------------------------------- #
# Buckets + trip/clear engine
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class BucketSpec:
    """One fuse bucket. `members` semantics per kind:
      family — strategy ids; setup — "<strategy>:<setup_family>" pairs;
      symbol — symbols; sector — sector names (resolved via sector_lookup).
    `regimes=None` is the GLOBAL leg (counts every row incl. unknown);
    a non-None frozenset is a regime-scoped leg (exact stamps only — the D2
    rider forbids unknown rows from ever counting here)."""

    fuse_id: str
    kind: str
    members: frozenset
    consecutive_losers: Optional[int] = 3
    session_net_pnl_usd: Optional[float] = None
    regimes: Optional[frozenset] = None

    def matches(
        self,
        close: TrustedClose,
        sector_lookup: Optional[Callable[[str], Optional[str]]] = None,
    ) -> bool:
        if self.kind == "family":
            hit = close.strategy in self.members
        elif self.kind == "setup":
            hit = (
                close.setup_family is not None
                and f"{close.strategy}:{close.setup_family}" in self.members
            )
        elif self.kind == "symbol":
            hit = close.symbol in self.members
        elif self.kind == "sector":
            if sector_lookup is None:
                return False
            sector = sector_lookup(close.symbol)
            hit = sector is not None and sector in self.members
        else:
            return False
        if not hit:
            return False
        if self.regimes is None:
            return True
        # D2 rider: regime-scoped legs match exact stamps only; an unknown
        # row can never count here (structural belt over config validation).
        return close.regime != "unknown" and close.regime in self.regimes


def sanitize_regime_scope(raw: Any) -> Optional[frozenset]:
    """Config → regimes scope. None/empty → GLOBAL leg. "unknown" is stripped
    (D2 rider — a scoped leg may never count unstamped rows) with a warning;
    unrecognised names are dropped. All names stripped → GLOBAL leg (never
    silently produce a leg that can't match anything)."""
    if not raw:
        return None
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return None
    cleaned = set()
    for r in raw:
        s = str(r).strip().lower()
        if s == "unknown":
            LOG.warning(
                "FUSE_CONFIG_REGIME_UNKNOWN_STRIPPED — 'unknown' may not scope "
                "a fuse leg (D2 rider); counting it belongs to the GLOBAL leg"
            )
            continue
        if s in KNOWN_REGIMES:
            cleaned.add(s)
        else:
            LOG.warning("FUSE_CONFIG_REGIME_UNRECOGNISED name=%s dropped", s)
    return frozenset(cleaned) if cleaned else None


@dataclasses.dataclass(frozen=True)
class BucketStats:
    fuse_id: str
    kind: str
    matched: int
    consecutive_losers: int
    session_net_pnl: float
    tripped: bool
    trip_rule: Optional[str]


def compute_bucket_stats(
    spec: BucketSpec,
    closes: Iterable[TrustedClose],
    sector_lookup: Optional[Callable[[str], Optional[str]]] = None,
) -> BucketStats:
    """Session stats for one bucket. A "loser" is a trusted close with
    pnl < 0. pnl == 0 scratches neither extend nor reset the trailing streak
    (edge-decay precedent)."""
    rows = [c for c in closes if spec.matches(c, sector_lookup)]
    net = sum(c.pnl for c in rows)
    streak = 0
    for c in reversed(rows):  # rows arrive exit-ts-sorted
        if c.pnl < 0:
            streak += 1
        elif c.pnl > 0:
            break
        # pnl == 0 → scratch: skip, streak unbroken
    trip_rule: Optional[str] = None
    if spec.consecutive_losers is not None and streak >= spec.consecutive_losers:
        trip_rule = "consecutive_losers"
    elif (
        spec.session_net_pnl_usd is not None
        and rows
        and net <= spec.session_net_pnl_usd
    ):
        trip_rule = "session_net_pnl"
    return BucketStats(
        fuse_id=spec.fuse_id,
        kind=spec.kind,
        matched=len(rows),
        consecutive_losers=streak,
        session_net_pnl=round(net, 4),
        tripped=trip_rule is not None,
        trip_rule=trip_rule,
    )


@dataclasses.dataclass(frozen=True)
class FuseEvent:
    """Edge-triggered trip/clear event (state N-1 → state N diff)."""

    event: str  # "trip" | "clear"
    fuse_id: str
    kind: str
    trip_rule: Optional[str]
    consecutive_losers: int
    session_net_pnl: float


def evaluate_buckets(
    buckets: Iterable[BucketSpec],
    closes: List[TrustedClose],
    *,
    prior_state: Optional[Mapping[str, Any]] = None,
    now: Optional[datetime] = None,
    sector_lookup: Optional[Callable[[str], Optional[str]]] = None,
    manual_clears: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[FuseEvent]]:
    """Evaluate every bucket; return (state fuse rows, edge-triggered events).

    Trip/clear is EDGE-triggered against *prior_state* (the previously
    published fuse_box_state.json): a bucket that stays tripped re-emits
    nothing; `tripped_at_utc` is preserved across cycles while tripped.
    Clearing is automatic at the session roll (counters re-derive empty) —
    the tripped→untripped transition emits the clear event.
    """
    now_utc = now or datetime.now(timezone.utc)
    prior_fuses: Dict[str, Mapping[str, Any]] = {}
    if isinstance(prior_state, Mapping):
        for row in prior_state.get("fuses") or []:
            if isinstance(row, Mapping) and row.get("fuse_id"):
                prior_fuses[str(row["fuse_id"])] = row

    cleared_ids = set(manual_clears or {})
    rows: List[Dict[str, Any]] = []
    events: List[FuseEvent] = []
    seen_ids = set()
    for spec in buckets:
        stats = compute_bucket_stats(spec, closes, sector_lookup)
        seen_ids.add(spec.fuse_id)
        prior = prior_fuses.get(spec.fuse_id)
        was_tripped = bool(prior and prior.get("tripped"))
        tripped_at: Optional[str] = None
        # W4A-9: an operator manual-clear (scripts/clear_fuse.py) for THIS
        # session window forces the bucket untripped even though the ledger
        # still shows the streak — recorded honestly (real streak shown,
        # manually_cleared=True). Emits a one-shot clear event on the
        # tripped→cleared transition.
        manually_cleared = spec.fuse_id in cleared_ids
        if manually_cleared and stats.tripped:
            if was_tripped:
                events.append(
                    FuseEvent(
                        event="clear", fuse_id=spec.fuse_id, kind=spec.kind,
                        trip_rule=None,
                        consecutive_losers=stats.consecutive_losers,
                        session_net_pnl=stats.session_net_pnl,
                    )
                )
            rows.append({
                "fuse_id": spec.fuse_id, "kind": spec.kind, "tripped": False,
                "trip_rule": None, "matched": stats.matched,
                "consecutive_losers": stats.consecutive_losers,
                "session_net_pnl": stats.session_net_pnl,
                "regime_scope": (
                    sorted(spec.regimes) if spec.regimes is not None else None
                ),
                "clears_at": "next_session", "manually_cleared": True,
            })
            continue
        if stats.tripped:
            if was_tripped and prior is not None and prior.get("tripped_at_utc"):
                tripped_at = str(prior["tripped_at_utc"])
            else:
                tripped_at = (
                    now_utc.isoformat().replace("+00:00", "Z")
                )
                events.append(
                    FuseEvent(
                        event="trip",
                        fuse_id=spec.fuse_id,
                        kind=spec.kind,
                        trip_rule=stats.trip_rule,
                        consecutive_losers=stats.consecutive_losers,
                        session_net_pnl=stats.session_net_pnl,
                    )
                )
        elif was_tripped:
            events.append(
                FuseEvent(
                    event="clear",
                    fuse_id=spec.fuse_id,
                    kind=spec.kind,
                    trip_rule=None,
                    consecutive_losers=stats.consecutive_losers,
                    session_net_pnl=stats.session_net_pnl,
                )
            )
        row: Dict[str, Any] = {
            "fuse_id": spec.fuse_id,
            "kind": spec.kind,
            "tripped": stats.tripped,
            "trip_rule": stats.trip_rule,
            "matched": stats.matched,
            "consecutive_losers": stats.consecutive_losers,
            "session_net_pnl": stats.session_net_pnl,
            "regime_scope": (
                sorted(spec.regimes) if spec.regimes is not None else None
            ),
            "clears_at": "next_session",
        }
        if tripped_at is not None:
            row["tripped_at_utc"] = tripped_at
        rows.append(row)

    # A bucket present in prior state but absent from the current spec set
    # (config change) clears silently — by design: no spec, no fuse. Its
    # disappearance is visible in the state diff; no event is invented for a
    # bucket nobody ratified this cycle.
    return rows, events


# --------------------------------------------------------------------------- #
# W4A-3: LC2 bucket builders (family + setup grains, D1)
# --------------------------------------------------------------------------- #

def _bucket_thresholds(
    config: FuseBoxConfig, overrides: Optional[Mapping[str, Any]]
) -> Tuple[Optional[int], Optional[float], Optional[frozenset]]:
    """(consecutive_losers, session_net_pnl_usd, regimes) for one bucket,
    starting from config defaults. A null override disables that leg."""
    n: Optional[int] = config.default_consecutive_losers
    pnl: Optional[float] = config.default_session_net_pnl_usd
    regimes: Optional[frozenset] = None
    if isinstance(overrides, Mapping):
        if "consecutive_losers" in overrides:
            v = overrides.get("consecutive_losers")
            try:
                n = int(v) if v is not None else None
            except (TypeError, ValueError):
                pass
        if "session_net_pnl_usd" in overrides:
            v = overrides.get("session_net_pnl_usd")
            try:
                pnl = float(v) if v is not None else None
            except (TypeError, ValueError):
                pass
        regimes = sanitize_regime_scope(overrides.get("regimes"))
    return n, pnl, regimes


def _warn_family_registry_parity(config: FuseBoxConfig) -> None:
    """Runtime warn leg of the W3B-9 perimeter idiom (hard leg:
    chad/tests/test_w4a_fuse_config_parity.py). Config drift must never
    brick the engine — warn loud, keep going."""
    try:
        from chad.strategy_registry import active_strategy_values

        union = set()
        for members in config.families.values():
            union |= set(members)
        missing = set(active_strategy_values()) - union
        if missing:
            LOG.warning(
                "FUSE_CONFIG_FAMILY_PARITY active strategies without a "
                "family (uncovered by LC2): %s", sorted(missing),
            )
    except Exception:  # noqa: BLE001 — parity warning is best-effort
        pass


def build_lc2_buckets(
    config: FuseBoxConfig,
    closes: Iterable[TrustedClose],
) -> List[BucketSpec]:
    """Family buckets from the config map + setup buckets from the stamps
    OBSERVED in the trusted session closes (no fixed setup list to drift —
    a setup bucket exists exactly when the ledger shows that setup traded
    this session)."""
    _warn_family_registry_parity(config)
    buckets: List[BucketSpec] = []
    fam_overrides = (
        config.raw.get("family_thresholds")
        if isinstance(config.raw.get("family_thresholds"), Mapping)
        else {}
    )
    for name in sorted(config.families):
        members = config.families[name]
        if not members:
            continue
        n, pnl, regimes = _bucket_thresholds(config, fam_overrides.get(name))
        buckets.append(
            BucketSpec(
                fuse_id=f"family:{name}",
                kind="family",
                members=frozenset(members),
                consecutive_losers=n,
                session_net_pnl_usd=pnl,
                regimes=regimes,
            )
        )
    enabled = set(config.setup_fuse_strategies)
    if enabled:
        setup_defaults = (
            config.raw.get("setup_fuse")
            if isinstance(config.raw.get("setup_fuse"), Mapping)
            else None
        )
        n, pnl, regimes = _bucket_thresholds(config, setup_defaults)
        pairs = sorted(
            {
                f"{c.strategy}:{c.setup_family}"
                for c in closes
                if c.strategy in enabled and c.setup_family
            }
        )
        for pair in pairs:
            buckets.append(
                BucketSpec(
                    fuse_id=f"setup:{pair}",
                    kind="setup",
                    members=frozenset({pair}),
                    consecutive_losers=n,
                    session_net_pnl_usd=pnl,
                    regimes=regimes,
                )
            )
    return buckets


# --------------------------------------------------------------------------- #
# W4A-4: LC3 bucket builders (symbol + sector, anti-revenge)
# --------------------------------------------------------------------------- #

SECTOR_MAP_PATH = REPO_ROOT / "config" / "symbol_sectors.json"
UNMAPPED_SECTOR = "unmapped"


def load_sector_map(path: Optional[Path] = None) -> Dict[str, str]:
    """symbol → sector from config/symbol_sectors.json. Missing/corrupt →
    empty map (warn): every symbol then drains into the never-trippable
    unmapped bucket — a broken map can disarm LC3's sector grain but can
    never invent a blockable bucket nobody ratified (PLAN_W4A §4)."""
    target = Path(path) if path is not None else SECTOR_MAP_PATH
    try:
        obj = json.loads(target.read_text(encoding="utf-8"))
        sectors = obj.get("sectors") if isinstance(obj, dict) else None
        if not isinstance(sectors, dict):
            raise ValueError("no sectors object")
    except Exception as exc:  # noqa: BLE001
        LOG.warning("FUSE_SECTOR_MAP_UNREADABLE path=%s err=%s", target, exc)
        return {}
    out: Dict[str, str] = {}
    for sector, symbols in sectors.items():
        if not isinstance(symbols, list):
            continue
        name = str(sector).strip()
        if name == UNMAPPED_SECTOR:
            LOG.warning(
                "FUSE_SECTOR_MAP_RESERVED_NAME 'unmapped' row ignored — that "
                "name is the runtime accounting bucket"
            )
            continue
        for s in symbols:
            out[str(s).strip().upper()] = name
    return out


def make_sector_lookup(
    sector_map: Mapping[str, str],
) -> Callable[[str], Optional[str]]:
    """Lookup with the unmapped fallback — a symbol missing from the map
    lands in the UNMAPPED_SECTOR accounting bucket, never nowhere."""
    def _lookup(symbol: str) -> Optional[str]:
        return sector_map.get(str(symbol).strip().upper(), UNMAPPED_SECTOR)

    return _lookup


def build_lc3_buckets(
    config: FuseBoxConfig,
    closes: Iterable[TrustedClose],
    sector_map: Mapping[str, str],
) -> List[BucketSpec]:
    """Symbol + sector buckets from the symbols OBSERVED in the trusted
    session closes (a bucket exists exactly when that symbol/sector traded
    this session — no static bucket sprawl). The unmapped sector bucket is
    COUNT-ONLY: both trip legs None ⇒ it can never trip (loud in state and
    via the FUSE_SECTOR_UNMAPPED warning instead)."""
    closes = list(closes)
    buckets: List[BucketSpec] = []

    sym_over = (
        config.raw.get("symbol_fuse")
        if isinstance(config.raw.get("symbol_fuse"), Mapping)
        else None
    )
    n_sym, pnl_sym, reg_sym = _bucket_thresholds(config, sym_over)
    for sym in sorted({c.symbol for c in closes if c.symbol}):
        buckets.append(
            BucketSpec(
                fuse_id=f"symbol:{sym}",
                kind="symbol",
                members=frozenset({sym}),
                consecutive_losers=n_sym,
                session_net_pnl_usd=pnl_sym,
                regimes=reg_sym,
            )
        )

    sec_over = (
        config.raw.get("sector_fuse")
        if isinstance(config.raw.get("sector_fuse"), Mapping)
        else None
    )
    n_sec, pnl_sec, reg_sec = _bucket_thresholds(config, sec_over)
    lookup = make_sector_lookup(sector_map)
    observed_sectors = sorted(
        {lookup(c.symbol) or UNMAPPED_SECTOR for c in closes if c.symbol}
    )
    unmapped_syms = sorted(
        {
            c.symbol
            for c in closes
            if c.symbol and lookup(c.symbol) == UNMAPPED_SECTOR
        }
    )
    if unmapped_syms:
        LOG.warning(
            "FUSE_SECTOR_UNMAPPED symbols=%s — counting under the "
            "never-trippable '%s' bucket; add rows to "
            "config/symbol_sectors.json to arm them",
            unmapped_syms, UNMAPPED_SECTOR,
        )
    for sector in observed_sectors:
        is_unmapped = sector == UNMAPPED_SECTOR
        buckets.append(
            BucketSpec(
                fuse_id=f"sector:{sector}",
                kind="sector",
                members=frozenset({sector}),
                # Unmapped: count-only, never trips — a missing map row must
                # not create a blockable bucket nobody ratified.
                consecutive_losers=None if is_unmapped else n_sec,
                session_net_pnl_usd=None if is_unmapped else pnl_sec,
                regimes=None if is_unmapped else reg_sec,
            )
        )
    return buckets


# --------------------------------------------------------------------------- #
# W4A-3: eventing — marker + coach NOTIFY + dedupe-stable identity (§7)
# --------------------------------------------------------------------------- #

_DIGITS_RE = re.compile(r"\d+")


def dedupe_identity(event: str, fuse_id: str) -> str:
    """CTF-T2 stable identity: rule/bucket id + entity only, digits stripped,
    values never in the key (they live in evidence/facts)."""
    ident = _DIGITS_RE.sub("", str(fuse_id).strip().lower())
    return f"fuse_{event}_{ident}"


def emit_fuse_events(
    events: Iterable[FuseEvent],
    *,
    mode: str,
    evidence_dir: Optional[Path] = None,
    notify_fn: Optional[Callable[..., Any]] = None,
    now: Optional[datetime] = None,
) -> int:
    """All four §7 surfaces per trip/clear: grep-able marker line, evidence
    ndjson row, coach-voiced Telegram NOTIFY (sanctioned generic-kind path,
    edge_decay_cleared precedent), dedupe-stable identity. Never raises —
    eventing is presentation; the state write is the record of truth."""
    emitted = 0
    for ev in events:
        try:
            marker = "FUSE_TRIP" if ev.event == "trip" else "FUSE_CLEAR"
            LOG.warning(
                "%s kind=%s bucket=%s rule=%s streak=%d net=%.2f mode=%s",
                marker, ev.kind, ev.fuse_id, ev.trip_rule,
                ev.consecutive_losers, ev.session_net_pnl, mode,
            )
            append_evidence(
                [
                    {
                        "event": ev.event,
                        "fuse_id": ev.fuse_id,
                        "kind": ev.kind,
                        "trip_rule": ev.trip_rule,
                        "consecutive_losers": ev.consecutive_losers,
                        "session_net_pnl": ev.session_net_pnl,
                        "mode": mode,
                    }
                ],
                evidence_dir=evidence_dir,
                now=now,
            )
            facts = {
                "title": (
                    f"Fuse {'tripped' if ev.event == 'trip' else 'cleared'}: "
                    f"{ev.fuse_id}"
                ),
                "fuse_id": ev.fuse_id,
                "kind": ev.kind,
                "rule": ev.trip_rule,
                "mode": mode,
                "summary": (
                    f"The {ev.kind} fuse {ev.fuse_id} "
                    + (
                        "tripped — no new entries for this bucket until the "
                        "next session"
                        if ev.event == "trip"
                        else "cleared — the session window rolled or the "
                        "streak resolved"
                    )
                    + (
                        ". Shadow mode: recorded only, nothing was blocked."
                        if mode == MODE_SHADOW and ev.event == "trip"
                        else "."
                    )
                ),
            }
            msg: Optional[str] = None
            try:
                from chad.utils.coach_voice import format_alert

                rendered = format_alert(f"fuse_{ev.event}", facts)
                if isinstance(rendered, str) and rendered.strip():
                    msg = rendered
            except Exception:  # noqa: BLE001 — presentation is optional
                msg = None
            if not msg:
                msg = facts["summary"]
            send = notify_fn
            if send is None:
                from chad.utils.telegram_notify import notify as send  # type: ignore
            send(
                msg,
                severity="warning",
                dedupe_key=dedupe_identity(ev.event, ev.fuse_id),
                raise_on_fail=False,
            )
            emitted += 1
        except RuntimeError:
            raise  # pytest leak guard must not be swallowed
        except Exception as exc:  # noqa: BLE001 — NOTIFY is best-effort
            LOG.warning(
                "FUSE_EVENT_EMIT_FAILED fuse_id=%s err=%s", ev.fuse_id, exc
            )
    return emitted


# --------------------------------------------------------------------------- #
# State publisher + evidence (heartbeat doctrine: written every cycle incl. OFF)
# --------------------------------------------------------------------------- #

def _under_pytest() -> bool:
    """Margin-gate G3C-HF pattern: PYTEST_CURRENT_TEST is set per running
    test; a test that reaches a default path without an explicit override
    trips the leak guard."""
    return "PYTEST_CURRENT_TEST" in os.environ


def _guard_test_write(path: Path, default: Path, what: str) -> None:
    if _under_pytest() and Path(path) == default:
        raise RuntimeError(
            f"FUSE_BOX_ERROR {what} is REQUIRED-explicit under pytest — pass "
            f"an explicit tmp path (never the real {default}). "
            "[W4A test-write leak guard, margin-gate G3C-HF pattern]"
        )


def build_state(
    *,
    modes: Mapping[str, str],
    fuse_rows: List[Dict[str, Any]],
    counting_tally: Mapping[str, int],
    trusted_count: int,
    regime_unknown_rows: int,
    session_window_start_utc: Optional[datetime],
    epoch_started_at_utc: Optional[datetime],
    lc5: Optional[Mapping[str, Any]] = None,
    dq: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble fuse_box_state.v1. Identities are value-free (CTF-T2) — the
    numbers live in rows/evidence, never in dedupe identities or titles."""
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "modes": dict(modes),
        "session_window_start_utc": (
            session_window_start_utc.isoformat().replace("+00:00", "Z")
            if session_window_start_utc
            else None
        ),
        "epoch_started_at_utc": (
            epoch_started_at_utc.isoformat().replace("+00:00", "Z")
            if epoch_started_at_utc
            else None
        ),
        "counting": {
            "trusted_closes": int(trusted_count),
            "excluded": dict(counting_tally),
            "regime_unknown_rows": int(regime_unknown_rows),
        },
        "fuses": fuse_rows,
        "lc5": dict(lc5) if lc5 else {
            "factor": 1.0,
            "dd_5d_pct": None,
            "dd_20d_pct": None,
            "staleness": None,
            "emergency": False,
        },
        "dq": dict(dq) if dq else {"verdicts": {}},
    }


def publish_state(
    state: Dict[str, Any],
    state_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Atomic ts_utc/ttl-stamped write (runtime_json canon). Heartbeat
    doctrine: call this every evaluator cycle INCLUDING all-modes-off —
    `evaluated=0` must be distinguishable from dead (XOV lesson)."""
    from chad.utils.runtime_json import write_runtime_state_json

    target = Path(state_path) if state_path is not None else DEFAULT_STATE_PATH
    _guard_test_write(target, DEFAULT_STATE_PATH, "state_path")
    return write_runtime_state_json(
        target, state, ttl_seconds=STATE_TTL_SECONDS
    )


def read_prior_state(state_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    from chad.utils.runtime_json import read_json

    target = Path(state_path) if state_path is not None else DEFAULT_STATE_PATH
    return read_json(target)


MANUAL_CLEARS_PATH = REPO_ROOT / "runtime" / "fuse_manual_clears.json"


def load_manual_clears(
    session_window_start_utc: Optional[datetime],
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load operator manual-clears (scripts/clear_fuse.py) valid for the
    CURRENT session window. A clear stamped against a prior window is ignored
    (it auto-expires at the session roll — no cleanup needed). Missing/corrupt
    → {} (a broken override file can never trip a fuse, only fail to clear
    one)."""
    from chad.utils.runtime_json import read_json

    target = Path(path) if path is not None else MANUAL_CLEARS_PATH
    obj = read_json(target)
    if not isinstance(obj, dict):
        return {}
    cleared = obj.get("cleared")
    if not isinstance(cleared, dict):
        return {}
    cur = (
        session_window_start_utc.isoformat().replace("+00:00", "Z")
        if session_window_start_utc else None
    )
    out: Dict[str, Any] = {}
    for fuse_id, meta in cleared.items():
        if not isinstance(meta, Mapping):
            continue
        if cur is not None and str(meta.get("session_window_start")) != cur:
            continue
        out[str(fuse_id)] = dict(meta)
    return out


def append_evidence(
    rows: Iterable[Mapping[str, Any]],
    evidence_dir: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> int:
    """Append evidence rows (trips, clears, would_block, staleness) to
    data/fuse_box/fuse_box_YYYYMMDD.ndjson. Never raises upward from the
    caller's perspective beyond the pytest leak guard — evidence is
    best-effort, the trip decision is not contingent on it."""
    target_dir = Path(evidence_dir) if evidence_dir is not None else DEFAULT_EVIDENCE_DIR
    _guard_test_write(target_dir, DEFAULT_EVIDENCE_DIR, "evidence_dir")
    now_utc = now or datetime.now(timezone.utc)
    written = 0
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"fuse_box_{now_utc.strftime('%Y%m%d')}.ndjson"
        with open(path, "a", encoding="utf-8") as f:
            for row in rows:
                out = dict(row)
                out.setdefault(
                    "ts_utc", now_utc.isoformat().replace("+00:00", "Z")
                )
                f.write(json.dumps(out, sort_keys=True) + "\n")
                written += 1
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001 — evidence is best-effort
        LOG.warning("FUSE_EVIDENCE_WRITE_FAILED err=%s", exc)
    return written


# --------------------------------------------------------------------------- #
# W4A-7: LC5 progressive drawdown sizing (D4/D5)
# --------------------------------------------------------------------------- #

DRAWDOWN_STATE_PATH = REPO_ROOT / "runtime" / "drawdown_state.json"
LC5_DEFAULT_STALE_MAX_SECONDS = 3600
LC5_DEFAULT_EMERGENCY_PCT = -15.0


def _ladder_factor(dd_pct: Optional[float], rungs: Any) -> float:
    """Deepest triggered rung factor for one window. Rungs: [{at_pct<=0,
    factor}]. dd_pct <= at_pct triggers (both negative). Returns 1.0 when the
    window is None or no rung triggers. Malformed rungs are skipped."""
    if dd_pct is None or not isinstance(rungs, list):
        return 1.0
    f = 1.0
    for rung in rungs:
        if not isinstance(rung, Mapping):
            continue
        try:
            at = float(rung.get("at_pct"))
            fac = float(rung.get("factor"))
        except (TypeError, ValueError):
            continue
        if dd_pct <= at:
            f = min(f, fac)
    return f


def _clamp_factor(f: float) -> float:
    """Clamp to (0, 1] — a sizing factor may shrink but never invert or zero
    (0 would be a silent halt; emergency handles halting explicitly)."""
    if f <= 0.0:
        return 0.01
    return min(1.0, f)


def compute_lc5_state(
    now: Optional[datetime] = None,
    *,
    config: Optional[FuseBoxConfig] = None,
    prior_lc5: Optional[Mapping[str, Any]] = None,
    prior_session_window_start: Optional[str] = None,
    session_window_start_utc: Optional[datetime] = None,
    drawdown_state_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Compute the LC5 sizing factor + emergency flag from drawdown_state.v2.

    Returns (lc5_dict, staleness_events). The factor is the min of every
    ladder rung triggered across the 5d/20d windows (depth-gated: a window
    with sample_count < its length never contributes). Staleness (§5.4):
    - fresh: factor from the ladder; worst-rung memory updated.
    - stale/missing/corrupt within stale_max_seconds: HOLD the prior factor
      (never tighten or loosen on unknown data) — loud.
    - stale beyond stale_max_seconds: degrade to the worst rung reached this
      session (never upward).
    Session roll (window start changed) resets the worst-rung memory.
    """
    cfg = config or FuseBoxConfig.load()
    ladder = cfg.lc5_ladder or {}
    stale_max = float(ladder.get("stale_max_seconds", LC5_DEFAULT_STALE_MAX_SECONDS))
    emergency_pct = float(ladder.get("emergency_halt_pct", LC5_DEFAULT_EMERGENCY_PCT))
    now_utc = now or datetime.now(timezone.utc)

    prior = dict(prior_lc5) if isinstance(prior_lc5, Mapping) else {}
    prior_factor = float(prior.get("factor", 1.0) or 1.0)
    # Session-scoped worst (most conservative) factor reached. Reset on roll.
    cur_window = (
        session_window_start_utc.isoformat().replace("+00:00", "Z")
        if session_window_start_utc else None
    )
    rolled = (
        prior_session_window_start is not None
        and cur_window is not None
        and prior_session_window_start != cur_window
    )
    worst_prior = 1.0 if rolled else float(prior.get("worst_factor_session", 1.0) or 1.0)

    events: List[Dict[str, Any]] = []

    from chad.utils.runtime_json import read_runtime_state_json

    path = drawdown_state_path or DRAWDOWN_STATE_PATH
    obj, freshness = read_runtime_state_json(path)

    if obj is None or not freshness.ok:
        # Unknown data: never tighten/loosen. Hold last, unless stale too long.
        reason = getattr(freshness, "reason", "unknown")
        age = getattr(freshness, "age_seconds", None)
        degraded = age is not None and age > stale_max
        factor = _clamp_factor(worst_prior if degraded else prior_factor)
        was_stale = str(prior.get("staleness") or "fresh") != "fresh"
        if not was_stale:
            events.append({
                "event": "lc5_stale",
                "marker": "FUSE_LC5_DRAWDOWN_STALE",
                "reason": reason,
                "held_factor": factor,
                "degraded_to_worst": degraded,
            })
        LOG.warning(
            "FUSE_LC5_DRAWDOWN_STALE reason=%s age=%s held_factor=%.3f "
            "degraded_to_worst=%s", reason, age, factor, degraded,
        )
        lc5 = {
            "factor": round(factor, 4),
            "dd_5d_pct": prior.get("dd_5d_pct"),
            "dd_20d_pct": prior.get("dd_20d_pct"),
            "staleness": reason if reason in {"missing", "stale"} else "stale",
            "emergency": bool(prior.get("emergency", False)),
            "worst_factor_session": round(min(worst_prior, factor), 4),
            "session_window_start": cur_window,
        }
        return lc5, events

    # Fresh data.
    if str(prior.get("staleness") or "fresh") != "fresh":
        events.append({
            "event": "lc5_restored",
            "marker": "FUSE_LC5_DRAWDOWN_RESTORED",
        })
        LOG.warning("FUSE_LC5_DRAWDOWN_RESTORED fresh drawdown_state resumed")

    def _win(dd_key: str, n_key: str, window: int, rungs_key: str) -> Optional[float]:
        n = int(obj.get(n_key, 0) or 0)
        if n < window:
            return None  # depth guard: reports but does not enforce
        v = obj.get(dd_key)
        return float(v) if v is not None else None

    dd5 = _win("dd_5d_pct", "sample_count_5d", 5, "dd_5d")
    dd20 = _win("dd_20d_pct", "sample_count_20d", 20, "dd_20d")
    f5 = _ladder_factor(dd5, ladder.get("dd_5d"))
    f20 = _ladder_factor(dd20, ladder.get("dd_20d"))
    factor = _clamp_factor(min(f5, f20))

    # Emergency (D5): the existing −15% halt boolean OR an enforce-eligible
    # window at/below emergency_halt_pct. Block only — exits stay free.
    emergency = bool(obj.get("halt", False))
    for dd in (dd5, dd20):
        if dd is not None and dd <= emergency_pct:
            emergency = True
    prior_emergency = bool(prior.get("emergency", False))
    if emergency and not prior_emergency:
        events.append({
            "event": "lc5_emergency", "marker": "FUSE_LC5_EMERGENCY",
            "emergency_halt_pct": emergency_pct, "dd_5d_pct": dd5, "dd_20d_pct": dd20,
        })
        LOG.warning("FUSE_LC5_EMERGENCY dd_5d=%s dd_20d=%s pct=%s — new entries "
                    "blocked (exits free)", dd5, dd20, emergency_pct)
    elif prior_emergency and not emergency:
        events.append({"event": "lc5_emergency_cleared",
                       "marker": "FUSE_LC5_EMERGENCY_CLEARED"})

    lc5 = {
        "factor": round(factor, 4),
        "dd_5d_pct": dd5,
        "dd_20d_pct": dd20,
        "staleness": "fresh",
        "emergency": emergency,
        "worst_factor_session": round(min(worst_prior, factor), 4),
        "session_window_start": cur_window,
    }
    return lc5, events


def apply_lc5_factor(quantity: float, factor: float, sec_type: str) -> float:
    """Multiply an ENTRY quantity by the LC5 factor with the SCR-CAUTIOUS
    rounding rules (live_loop.py:2708-2714): FUT rounds to whole contracts
    min 1; equity floors to whole shares min 1. Never > input. Only the
    caller decides this is an entry (exits never reach here)."""
    import math

    try:
        raw = float(quantity)
    except (TypeError, ValueError):
        return quantity
    if factor >= 1.0 or raw <= 0:
        return quantity
    if str(sec_type or "").upper() == "FUT":
        return max(1.0, float(round(raw * factor)))
    return max(1.0, float(math.floor(raw * factor)))


# --------------------------------------------------------------------------- #
# W4A-3: evaluator cycle — the single call live_loop makes (wired at W4A-5)
# --------------------------------------------------------------------------- #

def run_evaluator_cycle(
    now: Optional[datetime] = None,
    *,
    config: Optional[FuseBoxConfig] = None,
    env: Optional[Mapping[str, str]] = None,
    trades_dir: Optional[Path] = None,
    fills_dir: Optional[Path] = None,
    runtime_dir: Optional[Path] = None,
    epoch_state_path: Optional[Path] = None,
    state_path: Optional[Path] = None,
    evidence_dir: Optional[Path] = None,
    notify_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """One evaluator pass: modes → trusted counting → buckets → edge events →
    state heartbeat. Called every live_loop cycle (W4A-5 wiring) inside the
    loss-guard failure-soft envelope — an exception here must never kill the
    cycle, and the state write happens even when every mode is off
    (heartbeat doctrine: evaluated=0 ≠ dead).

    Counting runs when LC2 or LC3 is shadow/enforce; all-off publishes the
    bare heartbeat only (cheap). The GATE (W4A-5) is the only enforcement
    point — this function never blocks anything; it derives and publishes.
    """
    now_utc = now or datetime.now(timezone.utc)
    modes = read_modes(env)
    cfg = config or FuseBoxConfig.load()
    prior = read_prior_state(state_path)

    counting_active = (
        modes["lc2"] != MODE_OFF or modes["lc3"] != MODE_OFF
    )
    window_start = session_window_start(now_utc, epoch_state_path)
    closes: List[TrustedClose] = []
    tally: Dict[str, int] = {}
    fuse_rows: List[Dict[str, Any]] = []
    events: List[FuseEvent] = []
    if counting_active:
        closes, tally = load_trusted_session_closes(
            now_utc,
            trades_dir=trades_dir,
            fills_dir=fills_dir,
            runtime_dir=runtime_dir,
            epoch_state_path=epoch_state_path,
        )
        buckets: List[BucketSpec] = []
        if modes["lc2"] != MODE_OFF:
            buckets.extend(build_lc2_buckets(cfg, closes))
        sector_lookup: Optional[Callable[[str], Optional[str]]] = None
        if modes["lc3"] != MODE_OFF:
            sector_map = load_sector_map()
            sector_lookup = make_sector_lookup(sector_map)
            buckets.extend(build_lc3_buckets(cfg, closes, sector_map))
        fuse_rows, events = evaluate_buckets(
            buckets, closes, prior_state=prior, now=now_utc,
            sector_lookup=sector_lookup,
            manual_clears=load_manual_clears(window_start),
        )
        if events:
            emit_fuse_events(
                events,
                # Mode label for evidence/alerts: enforce iff any counting
                # fuse is enforcing; shadow otherwise.
                mode=(
                    MODE_ENFORCE
                    if MODE_ENFORCE in (modes["lc2"], modes["lc3"])
                    else MODE_SHADOW
                ),
                evidence_dir=evidence_dir,
                notify_fn=notify_fn,
                now=now_utc,
            )

    # W4A-7: LC5 sizing/emergency. Computed when LC5 is shadow/enforce; carried
    # forward (hold-last) from prior state otherwise. Staleness/emergency
    # events surface via the same evidence + coach path.
    prior_lc5 = (prior or {}).get("lc5") if isinstance(prior, Mapping) else None
    lc5_state: Optional[Dict[str, Any]] = prior_lc5 if isinstance(prior_lc5, dict) else None
    if modes["lc5"] != MODE_OFF:
        lc5_state, lc5_events = compute_lc5_state(
            now_utc,
            config=cfg,
            prior_lc5=prior_lc5,
            prior_session_window_start=(
                (prior_lc5 or {}).get("session_window_start")
                if isinstance(prior_lc5, Mapping) else None
            ),
            session_window_start_utc=window_start,
        )
        if lc5_events:
            _emit_lc5_events(
                lc5_events, mode=modes["lc5"],
                evidence_dir=evidence_dir, notify_fn=notify_fn, now=now_utc,
            )

    state = build_state(
        modes=modes,
        fuse_rows=fuse_rows,
        counting_tally=tally,
        trusted_count=len(closes),
        regime_unknown_rows=sum(1 for c in closes if c.regime == "unknown"),
        session_window_start_utc=(
            window_start if (counting_active or modes["lc5"] != MODE_OFF) else None
        ),
        epoch_started_at_utc=read_epoch_start(epoch_state_path),
        lc5=lc5_state,
    )
    return publish_state(state, state_path)


def _emit_lc5_events(
    events: Iterable[Mapping[str, Any]],
    *,
    mode: str,
    evidence_dir: Optional[Path] = None,
    notify_fn: Optional[Callable[..., Any]] = None,
    now: Optional[datetime] = None,
) -> None:
    """LC5 staleness/emergency events: marker (already logged in compute),
    evidence row, coach feed_stale-kind NOTIFY with value-free dedupe. Never
    raises."""
    for ev in events:
        try:
            append_evidence(
                [{**ev, "mode": mode}], evidence_dir=evidence_dir, now=now
            )
            kind = str(ev.get("event") or "lc5")
            is_emerg = "emergency" in kind
            facts = {
                "title": str(ev.get("marker") or f"LC5 {kind}"),
                "summary": (
                    "LC5 drawdown feed is stale — holding the last sizing "
                    "factor, tightening nothing on unknown data."
                    if "stale" in kind else
                    "LC5 drawdown feed recovered — sizing resumes on fresh data."
                    if "restored" in kind else
                    "LC5 emergency drawdown reached — new entries blocked; "
                    "exits, flips and the overlay stay free."
                    if kind == "lc5_emergency" else
                    "LC5 emergency cleared — entry sizing resumes."
                ),
            }
            msg = None
            try:
                from chad.utils.coach_voice import format_alert

                rendered = format_alert("feed_stale" if "stale" in kind else "drawdown", facts)
                if isinstance(rendered, str) and rendered.strip():
                    msg = rendered
            except Exception:  # noqa: BLE001
                msg = None
            if not msg:
                msg = facts["summary"]
            send = notify_fn
            if send is None:
                from chad.utils.telegram_notify import notify as send  # type: ignore
            send(
                msg,
                severity="critical" if is_emerg else "warning",
                dedupe_key=f"fuse_{kind}",
                raise_on_fail=False,
            )
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001
            LOG.warning("FUSE_LC5_EVENT_EMIT_FAILED err=%s", exc)


__all__ = [
    "ENV_DQ",
    "ENV_LC2",
    "ENV_LC3",
    "ENV_LC5",
    "EXCLUDED_STRATEGIES",
    "GENUINE_FILL_STATUSES",
    "KNOWN_REGIMES",
    "MODE_ENFORCE",
    "MODE_OFF",
    "MODE_SHADOW",
    "STATE_SCHEMA_VERSION",
    "STATE_TTL_SECONDS",
    "BucketSpec",
    "BucketStats",
    "FuseBoxConfig",
    "FuseEvent",
    "TrustedClose",
    "UNMAPPED_SECTOR",
    "append_evidence",
    "apply_lc5_factor",
    "build_lc2_buckets",
    "build_lc3_buckets",
    "build_state",
    "compute_bucket_stats",
    "compute_lc5_state",
    "dedupe_identity",
    "emit_fuse_events",
    "evaluate_buckets",
    "fuse_mode",
    "run_evaluator_cycle",
    "load_manual_clears",
    "load_sector_map",
    "load_trusted_session_closes",
    "load_window_fill_statuses",
    "make_sector_lookup",
    "publish_state",
    "read_modes",
    "read_prior_state",
    "sanitize_regime_scope",
    "session_window_start",
]
