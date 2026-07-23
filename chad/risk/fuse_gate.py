"""
chad/risk/fuse_gate.py — W4A-5 per-intent fuse gate (IBKR stage-3 + Kraken).

The enforcement half of the fuse box. The evaluator (fuse_box.run_evaluator_cycle,
wired earlier in the same live_loop cycle) DERIVES bucket state and publishes
runtime/fuse_box_state.json; this gate READS that state and decides whether a
single ENTRY intent is blocked — the margin-gate contract exactly
(chad/execution/margin_shadow_gate.py): injectable, ctor-default reads the live
state, should_block() returns False in shadow, marker + dated ndjson evidence.

THE PRIME INVARIANT — a fuse never blocks a close — is enforced three ways here,
belt-and-braces (PLAN_W4A §8.1 / audits/W4A_GO_RECORD.md §4.1):
  1. Structural (the strong leg): overlay/reconciler/flatten closes NEVER reach
     this gate — they go apply_close_intents → adapter direct, bypassing
     live_loop stage-3 and the Kraken risk path. This gate is entry-scoped by
     PLACEMENT; it is physically not on any close path.
  2. Predicate bypass: even for intents that DO reach stage-3, is_exit_like()
     passes anything that is a flip, an EXIT/CLOSE side, a protective tag
     (reduce/hedge/stop_loss/liquidation — the loss-guard's set), or carries a
     W4B-2 close-provenance stamp (meta.action == "CLOSE").
  3. Kraken mirror additionally exempts reduce_only intents and
     crypto_exit|-keyed / overlay-marked intents.

Modes (CHAD_FUSE_LC2 / CHAD_FUSE_LC3, tri-state, default off): a fuse enforces
only when its own flag is `enforce`. Shadow evaluates + emits would_block
evidence, returns block=False (byte-identical behavior). LC2 governs
family/setup buckets; LC3 governs symbol/sector buckets — a tripped bucket
blocks only under its own flag's enforce.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, List, Mapping, Optional

from chad.risk.fuse_box import (
    ENV_LC2,
    ENV_LC3,
    ENV_LC5,
    MODE_ENFORCE,
    MODE_OFF,
    fuse_mode,
    read_prior_state,
)

LOG = logging.getLogger(__name__)

# Kind → owning flag. A bucket kind enforces only when its flag is `enforce`.
_KIND_FLAG = {
    "family": ENV_LC2,
    "setup": ENV_LC2,
    "symbol": ENV_LC3,
    "sector": ENV_LC3,
}

# The loss-guard's protective tag vocabulary (mirrored — the GO record binds
# the fuse bypass to be no stricter against exits than the loss guard).
_PROTECTIVE_TAGS = frozenset(
    {"exit", "close", "reduce", "hedge", "stop_loss", "liquidation",
     "risk_reduction", "protective"}
)
_PROTECTIVE_SIDES = frozenset({"EXIT", "CLOSE"})
_PROTECTIVE_REASONS = frozenset(
    {"max_hold_exit", "stop_loss", "liquidation", "risk_reduction"}
)


@dataclasses.dataclass(frozen=True)
class FuseVerdict:
    blocked: bool
    would_block: bool
    fuse_id: Optional[str]
    kind: Optional[str]
    reason: str
    mode: str


def _intent_attr(intent: Any, name: str, default: Any = None) -> Any:
    v = getattr(intent, name, None)
    if v is None and isinstance(intent, Mapping):
        v = intent.get(name, default)
    return v if v is not None else default


def is_exit_like(intent: Any) -> bool:
    """Predicate bypass (§8.1). True for anything a fuse must never block:
    flips, EXIT/CLOSE sides, protective tags/reasons, and W4B-2 close stamps.
    Fail-SAFE: any error → treat as exit-like (never block on uncertainty
    about whether something is a close)."""
    try:
        # W4B-2 close-provenance stamp (belt over structural bypass).
        meta = _intent_attr(intent, "meta")
        if isinstance(meta, Mapping):
            if str(meta.get("action", "")).strip().upper() == "CLOSE":
                return True
            if str(meta.get("reason", "")).strip().lower() in _PROTECTIVE_REASONS:
                return True
            if meta.get("exit") or meta.get("close") or meta.get("reduce"):
                return True
            mtags = meta.get("tags") or meta.get("signal_tags") or []
            if any(str(t).strip().lower() in _PROTECTIVE_TAGS for t in mtags):
                return True

        side = str(_intent_attr(intent, "side", "") or "").strip().upper()
        if side in _PROTECTIVE_SIDES:
            return True

        tags = _intent_attr(intent, "tags") or ()
        if any(str(t).strip().lower() in _PROTECTIVE_TAGS for t in tags):
            return True

        # Flip signals are exit-and-reverse — the closing leg must pass.
        try:
            from chad.core.position_guard import is_flip_signal

            if is_flip_signal(intent):
                return True
        except Exception:  # noqa: BLE001 — flip detection is best-effort
            pass
    except Exception:  # noqa: BLE001 — fail toward "exit-like" (never block)
        return True
    return False


class FuseGate:
    """Per-intent gate. Reads published fuse_box_state.json (the evaluator's
    output this cycle); no recomputation. One instance per cycle, reused
    across intents (the margin-gate lifecycle)."""

    def __init__(
        self,
        *,
        state: Optional[Mapping[str, Any]] = None,
        state_path: Optional[Path] = None,
        env: Optional[Mapping[str, str]] = None,
        evidence_dir: Optional[Path] = None,
        now: Optional[datetime] = None,
    ) -> None:
        self._env = env if env is not None else os.environ
        self._evidence_dir = evidence_dir
        self._now = now
        self.lc2 = fuse_mode(ENV_LC2, self._env)
        self.lc3 = fuse_mode(ENV_LC3, self._env)
        # Load state once (published earlier this cycle). Explicit state wins
        # (tests); otherwise read the live artifact.
        if state is not None:
            self._state: Mapping[str, Any] = state
        else:
            self._state = read_prior_state(state_path) or {}
        # Index tripped buckets for O(1) intent lookup.
        self._tripped: List[Mapping[str, Any]] = [
            row
            for row in (self._state.get("fuses") or [])
            if isinstance(row, Mapping) and row.get("tripped")
        ]
        # Load config maps ONCE per cycle (not per intent) — only when a fuse
        # is active and something is tripped, so an all-clear cycle pays
        # nothing.
        self._family_of_cache: dict = {}
        self._sector_lookup = None
        if self._tripped and self.any_active:
            try:
                from chad.risk.fuse_box import (
                    FuseBoxConfig,
                    load_sector_map,
                    make_sector_lookup,
                )

                cfg = FuseBoxConfig.load()
                for name, members in cfg.families.items():
                    for m in members:
                        self._family_of_cache[m] = name
                self._sector_lookup = make_sector_lookup(load_sector_map())
            except Exception as exc:  # noqa: BLE001
                LOG.warning("FUSE_GATE_MAP_LOAD_FAILED (fail-open): %s", exc)

        self.lc5 = fuse_mode(ENV_LC5, self._env)
        self._lc5 = self._state.get("lc5") if isinstance(self._state.get("lc5"), Mapping) else {}

    @property
    def any_active(self) -> bool:
        """True iff at least one entry-blocking fuse flag is not off
        (LC2/LC3). LC5 is a separate sizing/emergency axis (lc5_active)."""
        return self.lc2 != MODE_OFF or self.lc3 != MODE_OFF

    @property
    def lc5_active(self) -> bool:
        return self.lc5 != MODE_OFF

    def lc5_factor(self) -> float:
        """The published LC5 sizing factor (1.0 when LC5 is off, state absent,
        or malformed). Enforcement multiplies an ENTRY quantity by this."""
        if self.lc5 != MODE_ENFORCE:
            return 1.0
        try:
            f = float(self._lc5.get("factor", 1.0))
            return f if 0.0 < f <= 1.0 else 1.0
        except (TypeError, ValueError):
            return 1.0

    def lc5_emergency(self) -> bool:
        """True iff LC5 is ENFORCE and the published state flags emergency —
        the caller must block a NEW ENTRY (exits/flips/protectives bypass)."""
        if self.lc5 != MODE_ENFORCE:
            return False
        return bool(self._lc5.get("emergency", False))

    def should_block_lc5_entry(self, intent: Any) -> bool:
        """LC5 emergency entry-block for an IBKR-style intent. Never blocks an
        exit-like intent (the prime invariant holds at −15% too). Fail-open."""
        try:
            if not self.lc5_emergency():
                return False
            if is_exit_like(intent):
                return False
            self._emit(intent, FuseVerdict(
                blocked=True, would_block=True, fuse_id="lc5:emergency",
                kind="lc5_emergency", reason="lc5_emergency", mode=self.lc5,
            ))
            return True
        except Exception as exc:  # noqa: BLE001 — fail-open
            LOG.warning("FUSE_LC5_BLOCK_FAILED (fail-open): %s", exc)
            return False

    def _mode_for_kind(self, kind: str) -> str:
        flag = _KIND_FLAG.get(kind)
        if flag is None:
            return MODE_OFF
        return fuse_mode(flag, self._env)

    def _matching_tripped(
        self, strategy: str, symbol: str, setup: Optional[str]
    ) -> Optional[Mapping[str, Any]]:
        """First tripped bucket whose membership covers (strategy, symbol,
        setup). Regime scope was applied at COUNT time (evaluate_buckets), so a
        tripped row's `tripped` flag already reflects the D2 rider — the gate
        needs identity membership only."""
        family = self._family_of_cache.get(strategy)
        sector = self._sector_lookup(symbol) if self._sector_lookup else None
        for row in self._tripped:
            kind = str(row.get("kind") or "")
            fid = str(row.get("fuse_id") or "")
            if kind == "family" and family and fid == f"family:{family}":
                return row
            if kind == "setup" and setup and fid == f"setup:{strategy}:{setup}":
                return row
            if kind == "symbol" and fid == f"symbol:{symbol}":
                return row
            if kind == "sector" and sector and fid == f"sector:{sector}":
                return row
        return None

    def evaluate_fields(
        self,
        *,
        strategy: str,
        symbol: str,
        setup_family: Optional[str] = None,
        exit_like: bool = False,
    ) -> FuseVerdict:
        """Lane-agnostic core: verdict from explicit fields. IBKR and Kraken
        both funnel here so the tripped-bucket matching is one implementation."""
        if not self.any_active:
            return FuseVerdict(False, False, None, None, "all_off", MODE_OFF)
        if exit_like:
            return FuseVerdict(
                False, False, None, None, "exit_like_never_blocked", MODE_OFF
            )
        strat = str(strategy or "").strip().lower()
        sym = str(symbol or "").strip().upper()
        row = self._matching_tripped(strat, sym, setup_family)
        if row is None:
            return FuseVerdict(False, False, None, None, "no_tripped_bucket",
                               MODE_OFF)
        kind = str(row.get("kind") or "")
        fid = str(row.get("fuse_id") or "")
        mode = self._mode_for_kind(kind)
        return FuseVerdict(
            blocked=(mode == MODE_ENFORCE),
            would_block=True,
            fuse_id=fid,
            kind=kind,
            reason="bucket_tripped",
            mode=mode,
        )

    def evaluate(self, intent: Any) -> FuseVerdict:
        """Verdict for one IBKR-style intent (stage-3). Never blocks an
        exit-like intent; blocks an entry only when a covering bucket is
        tripped AND that bucket's kind is under enforce."""
        if not self.any_active:
            return FuseVerdict(False, False, None, None, "all_off", MODE_OFF)
        if is_exit_like(intent):
            return FuseVerdict(
                False, False, None, None, "exit_like_never_blocked", MODE_OFF
            )
        strategy = str(_intent_attr(intent, "strategy", "") or "").strip().lower()
        symbol = str(_intent_attr(intent, "symbol", "") or "").strip().upper()
        setup = None
        meta = _intent_attr(intent, "meta")
        if isinstance(meta, Mapping) and meta.get("setup_family"):
            setup = str(meta.get("setup_family")).strip()
        row = self._matching_tripped(strategy, symbol, setup)
        if row is None:
            return FuseVerdict(False, False, None, None, "no_tripped_bucket",
                               MODE_OFF)
        kind = str(row.get("kind") or "")
        fid = str(row.get("fuse_id") or "")
        mode = self._mode_for_kind(kind)
        would = True
        blocked = mode == MODE_ENFORCE
        return FuseVerdict(
            blocked=blocked,
            would_block=would,
            fuse_id=fid,
            kind=kind,
            reason="bucket_tripped",
            mode=mode,
        )

    def should_block(self, intent: Any) -> Optional[FuseVerdict]:
        """Margin-gate-shaped hook (IBKR stage-3). Returns a verdict when the
        intent is BLOCKED (enforce + tripped), else None to proceed. Always
        emits would_block evidence in shadow. Fail-open: any error → None
        (proceed); an entry is never blocked on a broken fuse gate."""
        try:
            verdict = self.evaluate(intent)
        except Exception as exc:  # noqa: BLE001 — fail-open
            LOG.warning("FUSE_GATE_EVAL_FAILED (fail-open): %s", exc)
            return None
        if verdict.would_block:
            self._emit(intent, verdict)
        return verdict if verdict.blocked else None

    def should_block_kraken(self, intent: Any) -> Optional[FuseVerdict]:
        """Kraken mirror. Exempts reduce_only intents, overlay-marked closes
        (CRYPTO_EXIT_OVERLAY_ACTIVE_CLOSE), and crypto_exit|-keyed intents —
        belt over the structural bypass (overlay closes don't traverse this
        path). Maps the Kraken intent's strategy + pair into the shared
        fields core. NOTE: the family/setup legs are exact (strategy-keyed);
        crypto symbol/sector legs are best-effort until crypto closes enter
        the trusted closed_trade substrate (they are schema-excluded today),
        so a raw pair is passed as the symbol. Fail-open."""
        try:
            exit_like = _kraken_exit_like(intent)
            strategy = str(getattr(intent, "strategy", "") or "")
            pair = str(getattr(intent, "pair", "") or getattr(intent, "symbol", "") or "")
            verdict = self.evaluate_fields(
                strategy=strategy, symbol=pair, exit_like=exit_like
            )
        except Exception as exc:  # noqa: BLE001 — fail-open
            LOG.warning("FUSE_GATE_KRAKEN_EVAL_FAILED (fail-open): %s", exc)
            return None
        if verdict.would_block:
            self._emit_kraken(intent, verdict)
        return verdict if verdict.blocked else None

    def _emit_kraken(self, intent: Any, verdict: FuseVerdict) -> None:
        try:
            marker = "FUSE_BLOCK" if verdict.blocked else "FUSE_WOULD_BLOCK"
            LOG.warning(
                "%s lane=kraken fuse_id=%s kind=%s pair=%s side=%s strategy=%s mode=%s",
                marker, verdict.fuse_id, verdict.kind,
                getattr(intent, "pair", None), getattr(intent, "side", None),
                getattr(intent, "strategy", None), verdict.mode,
            )
            from chad.risk.fuse_box import append_evidence

            append_evidence(
                [
                    {
                        "event": "block" if verdict.blocked else "would_block",
                        "lane": "kraken",
                        "fuse_id": verdict.fuse_id,
                        "kind": verdict.kind,
                        "mode": verdict.mode,
                        "pair": str(getattr(intent, "pair", "") or ""),
                        "side": str(getattr(intent, "side", "") or ""),
                        "strategy": str(getattr(intent, "strategy", "") or ""),
                    }
                ],
                evidence_dir=self._evidence_dir,
                now=self._now,
            )
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001
            LOG.warning("FUSE_GATE_KRAKEN_EVIDENCE_FAILED err=%s", exc)

    def _emit(self, intent: Any, verdict: FuseVerdict) -> None:
        try:
            marker = "FUSE_BLOCK" if verdict.blocked else "FUSE_WOULD_BLOCK"
            LOG.warning(
                "%s fuse_id=%s kind=%s symbol=%s side=%s strategy=%s mode=%s",
                marker, verdict.fuse_id, verdict.kind,
                _intent_attr(intent, "symbol"), _intent_attr(intent, "side"),
                _intent_attr(intent, "strategy"), verdict.mode,
            )
            from chad.risk.fuse_box import append_evidence

            append_evidence(
                [
                    {
                        "event": "block" if verdict.blocked else "would_block",
                        "fuse_id": verdict.fuse_id,
                        "kind": verdict.kind,
                        "mode": verdict.mode,
                        "symbol": str(_intent_attr(intent, "symbol", "") or ""),
                        "side": str(_intent_attr(intent, "side", "") or ""),
                        "strategy": str(_intent_attr(intent, "strategy", "") or ""),
                    }
                ],
                evidence_dir=self._evidence_dir,
                now=self._now,
            )
        except RuntimeError:
            raise  # pytest leak guard
        except Exception as exc:  # noqa: BLE001 — evidence best-effort
            LOG.warning("FUSE_GATE_EVIDENCE_FAILED err=%s", exc)


_CRYPTO_EXIT_MARKER = "CRYPTO_EXIT_OVERLAY_ACTIVE_CLOSE"


def _kraken_exit_like(intent: Any) -> bool:
    """Kraken close exemptions (fail-safe → exit-like on error)."""
    try:
        if bool(getattr(intent, "reduce_only", False)):
            return True
        markers = getattr(intent, "markers", None) or ()
        if any(str(m) == _CRYPTO_EXIT_MARKER for m in markers):
            return True
        idem = str(getattr(intent, "idempotency_key", "") or "")
        if idem.startswith("crypto_exit|"):
            return True
    except Exception:  # noqa: BLE001 — never block a close on uncertainty
        return True
    return False


__all__ = ["FuseGate", "FuseVerdict", "is_exit_like"]
