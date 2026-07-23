"""
chad/risk/allocator_shadow_gate.py — W5B-3 per-intent allocator observer.

The observer half of the R1 allocator. Constructed once per cycle with an
open-book snapshot; called per intent inside the existing stage-3 loop, beside
the fuse gate. Each call adds the intent's exposure to a running provisional
book and records the would-verdict for that intent's MARGINAL contribution.

**IT BLOCKS NOTHING, AND THERE IS NO CODE PATH BY WHICH IT COULD.**
`should_block()` is hard-coded to return False and takes no branch on the
verdict. The mode flag is `off | shadow` — deliberately NOT the tri-state
`off|shadow|enforce` the fuse box uses, because an `enforce` value here must
not be one edit away from live. R1-enforce is a separate future wave with its
own PA; see FINDING W5B-SF1 below for why enforcement is not merely "flip a
flag" work.

THE PRIME INVARIANT — the allocator never evaluates a close — holds three ways:
  1. Structural (the strong leg): overlay, crypto-overlay, reconciler and
     flatten closes go `apply_close_intents → adapter` direct and never
     traverse stage-3. This observer is entry-scoped by PLACEMENT; it is
     physically not on any close path.
  2. Predicate: `fuse_gate.is_exit_like` — IMPORTED, not re-implemented, so the
     allocator and the fuse can never disagree about what counts as a close.
     A bypassed intent produces no verdict and no evidence row, so the corpus
     is entries-only and cannot be misread as having gated an exit.
  3. Mode: in shadow nothing blocks regardless of verdict.

FINDING W5B-SF1 (standing, PLAN_W5B §13.4). The same structural property that
makes this observer safe also makes it partially blind: the bypassing paths can
move the book without the observer ever seeing the intent that moved it. The
provisional book is therefore open-positions-at-cycle-start plus entries, and
can be one cycle stale — a flip, whose closing leg bypasses while the executor
and reconciler move the position, is the sharp case. Shadow evidence supports
"this entry would have breached given the book as of cycle start"; it does NOT
support "gross never exceeded X". Any future enforce-flip PA opens on this item.
The finding is carried in every heartbeat so the sentinel sees it on the first
report and on every one after.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from chad.risk.allocator_limits import (
    ERROR,
    WOULD_APPROVE,
    WOULD_REJECT,
    WOULD_RESIZE,
    AllocatorVerdict,
    PortfolioLimits,
    evaluate_marginal,
)
from chad.risk.portfolio_allocator import (
    ExposureVector,
    ProvisionalBook,
    build_base_book,
    load_price_cache,
    load_sector_lookup,
    vector_from_intent,
)

LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

ENV_ALLOCATOR = "CHAD_ALLOCATOR"

MODE_OFF = "off"
MODE_SHADOW = "shadow"
# Note the absence of `enforce`. See the module docstring.
_VALID_MODES = frozenset({MODE_OFF, MODE_SHADOW})

EVIDENCE_SCHEMA = "allocator_shadow.v1"
DEFAULT_EVIDENCE_DIR = REPO_ROOT / "data" / "allocator_shadow"

MARKER = "ALLOCATOR_SHADOW"


def allocator_mode(env: Optional[Mapping[str, str]] = None) -> str:
    """`off | shadow`; anything else → off (garbage-to-off, the fuse idiom).

    `enforce` is explicitly rejected AND logged loudly: an operator who sets it
    expecting enforcement must find out here, not discover later that a wave
    which never built an enforce path silently ran in shadow.
    """
    src = env if env is not None else os.environ
    raw = str(src.get(ENV_ALLOCATOR, "")).strip().lower()
    if raw == "enforce":
        LOG.warning(
            "ALLOCATOR_ENFORCE_REFUSED %s=enforce is not implemented — W5B "
            "builds no enforce path (PLAN_W5B §0). Falling back to off.",
            ENV_ALLOCATOR,
        )
        return MODE_OFF
    return raw if raw in _VALID_MODES else MODE_OFF


def _under_pytest() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ


def _guard_test_write(path: Path, default: Path, what: str) -> None:
    """W4A leak guard (fuse_box.py:1066). A test that reaches a real
    runtime/ or data/ path without an explicit override trips this.

    W5B needs its own copy: the six `test_repo_write_guard.py` tests are in the
    worktree baseline failing set (audits/W5B_BASELINE.md), so that suite
    cannot be relied on to catch an evidence-path leak here.
    """
    if _under_pytest() and Path(path) == default:
        raise RuntimeError(
            f"ALLOCATOR_ERROR {what} is REQUIRED-explicit under pytest — pass "
            f"an explicit tmp path (never the real {default}). "
            "[W5B test-write leak guard, W4A pattern]"
        )


def append_evidence(
    rows: Iterable[Mapping[str, Any]],
    evidence_dir: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> int:
    """Append allocator_shadow.v1 rows to
    data/allocator_shadow/allocator_shadow_YYYYMMDD.ndjson.

    Best-effort: evidence never blocks a cycle. The pytest leak guard is the
    one exception that propagates.
    """
    target_dir = Path(evidence_dir) if evidence_dir is not None else DEFAULT_EVIDENCE_DIR
    _guard_test_write(target_dir, DEFAULT_EVIDENCE_DIR, "evidence_dir")
    now_utc = now or datetime.now(timezone.utc)
    written = 0
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"allocator_shadow_{now_utc.strftime('%Y%m%d')}.ndjson"
        with open(path, "a", encoding="utf-8") as f:
            for row in rows:
                out = dict(row)
                out.setdefault("ts_utc", now_utc.isoformat().replace("+00:00", "Z"))
                f.write(json.dumps(out, sort_keys=True) + "\n")
                written += 1
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001 — evidence is best-effort
        LOG.warning("ALLOCATOR_EVIDENCE_WRITE_FAILED err=%s", exc)
    return written


class AllocatorShadowGate:
    """One instance per cycle, reused across intents (the fuse-gate lifecycle).

    The book snapshot is taken ONCE at construction. That is deliberate: it
    makes the counterfactual well-defined ("the book as of cycle start, plus
    the entries seen so far"), and it is also the bound named in W5B-SF1.
    """

    def __init__(
        self,
        *,
        mode: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        book: Optional[ProvisionalBook] = None,
        limits: Optional[PortfolioLimits] = None,
        limits_path: Optional[Path] = None,
        positions: Optional[Iterable[Mapping[str, Any]]] = None,
        prices: Optional[Mapping[str, float]] = None,
        sector_lookup=None,
        evidence_dir: Optional[Path] = None,
        now: Optional[datetime] = None,
    ) -> None:
        self.mode = mode if mode is not None else allocator_mode(env)
        self._evidence_dir = evidence_dir
        self._now = now
        self._cycle_seq = 0

        # Per-cycle counters, consumed by the W5B-4 heartbeat.
        self.counts: Dict[str, int] = {
            "evaluated": 0, "bypassed": 0,
            WOULD_APPROVE: 0, WOULD_RESIZE: 0, WOULD_REJECT: 0, ERROR: 0,
        }
        self.by_limit: Dict[str, int] = {}
        # Consecutive would-rejects on the SAME dimension — the coach streak
        # input (W5B-5). Reset by any non-reject on that dimension.
        self.reject_streaks: Dict[str, int] = {}

        self._prices: Mapping[str, float] = {}
        self._sector_lookup = None
        self.book: Optional[ProvisionalBook] = None
        self.limits: Optional[PortfolioLimits] = None
        self.construction_error: Optional[str] = None

        if self.mode == MODE_OFF:
            return  # OFF pays nothing: no book read, no config read.

        try:
            self._sector_lookup = (
                sector_lookup if sector_lookup is not None else load_sector_lookup()
            )
            self._prices = (
                dict(prices) if prices is not None else load_price_cache()
            )
            self.book = book if book is not None else build_base_book(
                positions=positions,
                prices=self._prices,
                sector_lookup=self._sector_lookup,
            )
            self.limits = limits if limits is not None else PortfolioLimits.load(limits_path)
        except Exception as exc:  # noqa: BLE001 — an observer never kills a cycle
            LOG.warning("ALLOCATOR_CONSTRUCTION_FAILED (inert): %s", exc)
            self.construction_error = str(exc)
            self.book = None
            self.limits = None

    # -- properties -------------------------------------------------------- #

    @property
    def active(self) -> bool:
        return (
            self.mode == MODE_SHADOW
            and self.book is not None
            and self.limits is not None
        )

    # -- the contract ------------------------------------------------------ #

    def should_block(self, intent: Any) -> bool:
        """ALWAYS False. Present so the call site reads like its neighbour the
        fuse gate, and so the shadow contract is a testable assertion rather
        than a claim in a comment. It does not consult the verdict."""
        return False

    def observe(self, intent: Any) -> Optional[AllocatorVerdict]:
        """Evaluate one intent's marginal contribution and record it.

        Returns the verdict, or None when inactive or bypassed. The intent is
        added to the provisional book REGARDLESS of verdict, because shadow
        blocks nothing: the honest counterfactual is the book enforcement would
        have seen. Omitting a would-rejected intent would make the second and
        third correlated tickets each measure against a book missing the first,
        understating concentration exactly where it matters most.

        Never raises. Any internal failure yields a fail-open ERROR verdict and
        the cycle continues.
        """
        if not self.active:
            return None
        try:
            if self._is_bypassed(intent):
                self.counts["bypassed"] += 1
                return None
        except Exception:  # noqa: BLE001 — fail toward "it is a close"
            self.counts["bypassed"] += 1
            return None

        self._cycle_seq += 1
        try:
            vector = vector_from_intent(
                intent, prices=self._prices, sector_lookup=self._sector_lookup
            )
            verdict = evaluate_marginal(vector, self.book, self.limits)
            self.book.add_intent(vector)
        except Exception as exc:  # noqa: BLE001 — fail-open
            LOG.warning("%s_ERROR eval failed (fail-open): %s", MARKER, exc)
            self.counts["evaluated"] += 1
            self.counts[ERROR] += 1
            self._emit_error(intent, str(exc))
            return AllocatorVerdict(
                verdict=ERROR, which_limit=None,
                reason=f"eval_failed:{type(exc).__name__}", checks=(),
            )

        self._record(verdict)
        self._emit(intent, vector, verdict)
        return verdict

    # -- internals --------------------------------------------------------- #

    @staticmethod
    def _is_bypassed(intent: Any) -> bool:
        """Imported from the fuse gate, never re-implemented (PLAN_W5B §12 C1).
        Flips, EXIT/CLOSE sides, protective tags/reasons and W4B-2 close stamps
        all bypass. `is_exit_like` itself fails toward True."""
        from chad.risk.fuse_gate import is_exit_like

        return is_exit_like(intent)

    def _record(self, verdict: AllocatorVerdict) -> None:
        self.counts["evaluated"] += 1
        self.counts[verdict.verdict] = self.counts.get(verdict.verdict, 0) + 1
        dim = verdict.which_limit
        if verdict.verdict == WOULD_REJECT and dim:
            self.by_limit[dim] = self.by_limit.get(dim, 0) + 1
            self.reject_streaks[dim] = self.reject_streaks.get(dim, 0) + 1
        elif dim:
            self.by_limit[dim] = self.by_limit.get(dim, 0) + 1
            self.reject_streaks[dim] = 0
        else:
            # A clean approval breaks every dimension's streak.
            for k in list(self.reject_streaks):
                self.reject_streaks[k] = 0

    def _row(
        self, intent: Any, vector: ExposureVector, verdict: AllocatorVerdict
    ) -> Dict[str, Any]:
        book = self.book
        summary = book.summary() if book is not None else {}
        return {
            "schema_version": EVIDENCE_SCHEMA,
            "mode": self.mode,
            "evaluated": True,
            "verdict": verdict.verdict,
            "which_limit": verdict.which_limit,
            "reason": verdict.reason,
            "breach_by_usd": verdict.breach_by_usd,
            "breach_by_pct": verdict.breach_by_pct,
            "headroom_usd": verdict.headroom_usd,
            "would_resize_to_qty": verdict.would_resize_to_qty,
            # --- the ticket ---
            "symbol": vector.symbol,
            "side": vector.side,
            "strategy": vector.strategy,
            "quantity": vector.quantity,
            "asset_class": vector.asset_class,
            "venue": vector.venue,
            "sector": vector.sector,
            "currency": vector.currency,
            "intent_delta_usd": vector.delta_usd,
            "intent_beta_weighted_usd": vector.beta_weighted_usd,
            "beta_source": vector.beta_source,
            "price": vector.price,
            "price_source": vector.price_source,
            "multiplier": vector.multiplier,
            # --- the provisional book AFTER this intent ---
            "book_gross_usd": summary.get("gross_usd"),
            "book_net_usd": summary.get("net_usd"),
            "book_sector_usd": (book.by_sector().get(vector.sector) if book else None),
            "book_symbol_usd": (book.by_symbol().get(vector.symbol) if book else None),
            "book_uncomputable": summary.get("uncomputable"),
            "currency_mix": summary.get("currency_mix"),
            # --- per-dimension arithmetic ---
            "checks": [c.to_dict() for c in verdict.checks],
            # --- provenance ---
            # Soft correlation tuple, NOT a hard join key. FINDING W5B-F1:
            # StrategyTradeIntent defines neither idempotency_key nor trace_id,
            # so PA-EP3's execution_id is always "" on the IBKR lane. Nothing
            # here fabricates a key the execution layer does not mint.
            "join": {
                "kind": "soft_correlation_tuple",
                "cycle_seq": self._cycle_seq,
                "execution_id": str(
                    getattr(intent, "idempotency_key", "")
                    or getattr(intent, "trace_id", "")
                    or ""
                ),
                "execution_id_note": "empty on the IBKR lane — see FINDING W5B-F1",
            },
            "correlation_basis": "static_sector_buckets",
        }

    def _emit(
        self, intent: Any, vector: ExposureVector, verdict: AllocatorVerdict
    ) -> None:
        try:
            LOG.info(
                "%s verdict=%s which_limit=%s symbol=%s side=%s strategy=%s "
                "delta_usd=%s book_gross=%s mode=%s",
                MARKER, verdict.verdict, verdict.which_limit, vector.symbol,
                vector.side, vector.strategy, vector.delta_usd,
                (self.book.summary().get("gross_usd") if self.book else None),
                self.mode,
            )
            append_evidence(
                [self._row(intent, vector, verdict)],
                evidence_dir=self._evidence_dir,
                now=self._now,
            )
        except RuntimeError:
            raise  # pytest leak guard
        except Exception as exc:  # noqa: BLE001
            LOG.warning("ALLOCATOR_EVIDENCE_FAILED err=%s", exc)

    def _emit_error(self, intent: Any, err: str) -> None:
        try:
            append_evidence(
                [{
                    "schema_version": EVIDENCE_SCHEMA,
                    "mode": self.mode,
                    "evaluated": False,
                    "verdict": ERROR,
                    "which_limit": None,
                    "reason": "eval_failed",
                    "error": err,
                    "symbol": str(getattr(intent, "symbol", "") or ""),
                    "side": str(getattr(intent, "side", "") or ""),
                    "strategy": str(getattr(intent, "strategy", "") or ""),
                }],
                evidence_dir=self._evidence_dir,
                now=self._now,
            )
        except RuntimeError:
            raise
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# W5B-4: heartbeat state
# --------------------------------------------------------------------------- #

STATE_SCHEMA = "allocator_state.v1"
STATE_TTL_SECONDS = 180
DEFAULT_STATE_PATH = REPO_ROOT / "runtime" / "portfolio_allocator_state.json"

# Standing findings, carried in EVERY heartbeat so the sentinel sees them on
# the first report and on every one after — not once in a doc nobody re-reads.
# They bound what the shadow corpus is allowed to claim.
STANDING_FINDINGS = [
    {
        "id": "W5B-SF1",
        "severity": "bounds_evidence",
        "title": "stage-3 bypass makes the allocator partially blind",
        "detail": (
            "Overlay, crypto-overlay, reconciler and flatten closes go "
            "apply_close_intents -> adapter direct and never traverse stage-3. "
            "That is what makes this observer safe, and it is also what makes "
            "it blind: those paths change the book without the observer seeing "
            "the intent that changed it. The provisional book is open "
            "positions at cycle start plus entries, so it can be one cycle "
            "stale. A flip is the sharp case -- its closing leg bypasses while "
            "the executor and reconciler move the position."
        ),
        "bounds": (
            "Shadow evidence supports 'this entry WOULD have breached given "
            "the book as of cycle start'. It does NOT support 'gross never "
            "exceeded X'."
        ),
        "blocks": "heads the agenda of any future enforce-flip PA",
    },
    {
        "id": "W5B-SF2",
        "severity": "bounds_evidence",
        "title": "provisional book is an upper bound on submitted exposure",
        "detail": (
            "The observer sits at the stage-3 chokepoint, upstream of the "
            "cooldown gate, the ML veto and the dynamic risk gate. An intent "
            "counted here can still be suppressed below."
        ),
        "bounds": (
            "The book measures allocation DEMAND at stage-3, not predicted "
            "fills."
        ),
    },
    {
        "id": "W5B-F1",
        "severity": "informational",
        "title": "no execution join key on the IBKR lane",
        "detail": (
            "StrategyTradeIntent defines neither idempotency_key nor "
            "trace_id, so PA-EP3's execution_id resolves to the empty string "
            "for IBKR intents. Allocator rows carry a soft correlation tuple "
            "rather than a fabricated hard key."
        ),
        "bounds": "would-verdict -> realized-cost joins cannot be exact today",
    },
]


def build_state(gate: Optional["AllocatorShadowGate"]) -> Dict[str, Any]:
    """Assemble the allocator_state.v1 payload.

    Called every cycle INCLUDING when the flag is off (fuse-box heartbeat
    doctrine, the XOV lesson): `evaluated=0` must be distinguishable from dead.
    A None or inactive gate yields a well-formed all-off heartbeat.
    """
    mode = gate.mode if gate is not None else MODE_OFF
    state: Dict[str, Any] = {
        "schema_version": STATE_SCHEMA,
        "mode": mode,
        "active": bool(gate is not None and gate.active),
        "cycle": {
            "intents_evaluated": 0, "bypassed": 0,
            "would_approve": 0, "would_resize": 0, "would_reject": 0,
            "errors": 0, "by_limit": {},
        },
        "book": None,
        "correlation": {
            "mode": "static_sector_buckets",
            "rolling_deferred_to": "R2",
            "note": (
                "W5B computes no correlation coefficient. Sector buckets are "
                "the correlation proxy; a per-order reducer already exists at "
                "chad/risk/correlation_monitor.py."
            ),
        },
        "limits": None,
        "standing_findings": STANDING_FINDINGS,
        "construction_error": (gate.construction_error if gate is not None else None),
    }
    if gate is None or not gate.active:
        return state

    state["cycle"] = {
        "intents_evaluated": gate.counts.get("evaluated", 0),
        "bypassed": gate.counts.get("bypassed", 0),
        "would_approve": gate.counts.get(WOULD_APPROVE, 0),
        "would_resize": gate.counts.get(WOULD_RESIZE, 0),
        "would_reject": gate.counts.get(WOULD_REJECT, 0),
        "errors": gate.counts.get(ERROR, 0),
        "by_limit": dict(sorted(gate.by_limit.items())),
    }
    try:
        state["book"] = gate.book.summary()
    except Exception:  # noqa: BLE001
        state["book"] = None
    try:
        limits = gate.limits
        binding: Dict[str, Any] = {}
        for key in ("gross_exposure", "net_exposure", "per_symbol_concentration",
                    "per_sector"):
            cap, binds, basis, ratified = limits.cap(key)
            binding[key] = {
                "cap_usd": cap, "binds": binds, "basis": basis,
                "ratified": ratified,
            }
        state["limits"] = {
            "dimensions": binding,
            "equity_basis": limits.equity_basis,
            "unratified_derived": sorted(
                k for k, v in binding.items()
                if v["binds"] and not v["ratified"]
            ),
        }
    except Exception:  # noqa: BLE001
        state["limits"] = None
    return state


def publish_state(
    state: Dict[str, Any], state_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Atomic ts_utc/ttl-stamped write (runtime_json canon)."""
    from chad.utils.runtime_json import write_runtime_state_json

    target = Path(state_path) if state_path is not None else DEFAULT_STATE_PATH
    _guard_test_write(target, DEFAULT_STATE_PATH, "state_path")
    return write_runtime_state_json(target, state, ttl_seconds=STATE_TTL_SECONDS)


def publish_cycle_state(
    gate: Optional["AllocatorShadowGate"],
    state_path: Optional[Path] = None,
) -> None:
    """Heartbeat for the live loop. Never raises — a heartbeat failure must not
    end a trading cycle."""
    try:
        publish_state(build_state(gate), state_path)
    except RuntimeError:
        raise  # pytest leak guard
    except Exception as exc:  # noqa: BLE001
        LOG.warning("ALLOCATOR_HEARTBEAT_FAILED err=%s", exc)


# --------------------------------------------------------------------------- #
# W5B-5: coach NOTIFY — would-reject STREAKS only
# --------------------------------------------------------------------------- #

import re as _re

_DIGITS_RE = _re.compile(r"\d+")


def dedupe_identity(dimension: str) -> str:
    """CTF-T2 stable identity: rule + entity only, digits stripped, values
    NEVER in the key. A key containing the streak count or a dollar amount
    would change every time the number moved and defeat the dedupe entirely —
    which is exactly how the R13 flood happened."""
    ident = _DIGITS_RE.sub("", str(dimension).strip().lower())
    return f"allocator_reject_{ident}"


def maybe_notify_reject_streak(
    gate: Optional["AllocatorShadowGate"],
    *,
    notify_fn=None,
) -> int:
    """NOTIFY once per dimension that would-rejected `reject_streak_n`
    consecutive entries this cycle.

    NOT per intent — per-intent alerting is precisely the R13 flood CTF-T2
    fixed. One message per dimension per dedupe window, value-free key, count
    and dollars in the evidence and the PRO line only.

    Never raises: alerting is presentation, the evidence file is the record.
    """
    if gate is None or not gate.active:
        return 0
    sent = 0
    try:
        threshold = gate.limits.reject_streak_n
    except Exception:  # noqa: BLE001
        threshold = 3

    for dimension, streak in sorted(gate.reject_streaks.items()):
        if streak < threshold:
            continue
        try:
            cap = binds = basis = ratified = None
            key = {
                "gross": "gross_exposure",
                "net": "net_exposure",
                "per_symbol": "per_symbol_concentration",
                "per_sector": "per_sector",
            }.get(dimension)
            if key:
                cap, binds, basis, ratified = gate.limits.cap(key)
            facts = {
                "dimension": dimension,
                "streak": streak,
                "cap_usd": cap,
                "basis": basis,
                "ratified": bool(ratified),
                "book_gross_usd": (
                    gate.book.summary().get("gross_usd") if gate.book else None
                ),
            }
            msg = None
            try:
                from chad.utils.coach_voice import format_alert

                rendered = format_alert("allocator_reject_streak", facts)
                if isinstance(rendered, str) and rendered.strip():
                    msg = rendered
            except Exception:  # noqa: BLE001 — presentation is optional
                msg = None
            if not msg:
                msg = (
                    f"Allocator (shadow): {streak} consecutive entries would "
                    f"have been rejected on {dimension}. Nothing was blocked."
                )
            send = notify_fn
            if send is None:
                from chad.utils.telegram_notify import notify as send  # type: ignore
            send(
                msg,
                severity="warning",
                dedupe_key=dedupe_identity(dimension),
                raise_on_fail=False,
            )
            sent += 1
        except RuntimeError:
            raise  # pytest leak guard must not be swallowed
        except Exception as exc:  # noqa: BLE001 — NOTIFY is best-effort
            LOG.warning(
                "ALLOCATOR_STREAK_NOTIFY_FAILED dimension=%s err=%s", dimension, exc
            )
    return sent


def build_cycle_gate(
    *, env: Optional[Mapping[str, str]] = None
) -> Optional[AllocatorShadowGate]:
    """Construct the per-cycle observer for the live loop. Returns None on any
    failure — the call site treats None as "no observer this cycle" and the
    cycle is byte-identical to today."""
    try:
        gate = AllocatorShadowGate(env=env)
        return gate
    except Exception as exc:  # noqa: BLE001
        LOG.debug("allocator gate construction failed (non-fatal): %s", exc)
        return None


__all__ = [
    "AllocatorShadowGate",
    "DEFAULT_EVIDENCE_DIR",
    "DEFAULT_STATE_PATH",
    "ENV_ALLOCATOR",
    "EVIDENCE_SCHEMA",
    "MARKER",
    "MODE_OFF",
    "MODE_SHADOW",
    "STANDING_FINDINGS",
    "STATE_SCHEMA",
    "STATE_TTL_SECONDS",
    "allocator_mode",
    "append_evidence",
    "build_cycle_gate",
    "build_state",
    "dedupe_identity",
    "maybe_notify_reject_streak",
    "publish_cycle_state",
    "publish_state",
]
