#!/usr/bin/env python3
"""
chad/utils/pipeline.py

CHAD Decision Pipeline (Phase 4/11) — production-grade, SSOT-safe.

Goals
-----
- Deterministic orchestration of strategy signals -> policy evaluation -> routing.
- Zero broker/exchange calls (brains stay separated from execution).
- Optional spam-governor that prevents any one strategy (e.g., Beta) from flooding cycles.
- Spam governor is FAIL-SAFE: it can never crash the pipeline; worst case it no-ops.
- When enabled, writes runtime/strategy_spam_governor_state.json every cycle.

Notes
-----
This module intentionally keeps responsibilities narrow:
- Strategies produce TradeSignal objects.
- Policy layer (if provided/enabled) evaluates those signals.
- Router merges/conflicts and produces RoutedSignal objects.
- This module does NOT place orders.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from chad.types import TradeSignal, RoutedSignal  # core intent types
from chad.utils.signal_router import SignalRouter

# Router (must exist in your repo; tests reference routing behavior)
from chad.utils.signal_router import route_signals  # merges/conflicts and returns routed signals

logger = logging.getLogger("chad.pipeline")


# -----------------------------
# Config + result contracts
# -----------------------------

@dataclass(frozen=True)
class PipelineConfig:
    """
    Controls pipeline behavior.
    """
    use_policy: bool = True
    # Spam governor: default OFF; enable via env CHAD_SPAM_GOVERNOR_ENABLED=1
    spam_governor_enabled: bool = False


@dataclass(frozen=True)
class PipelineResult:
    """
    End-to-end snapshot of a single pipeline run.

    - raw_signals: signals emitted directly by strategies (after optional spam governor)
    - evaluated_signals: policy decisions (if policy enabled), otherwise empty list
    - routed_signals: final merged/conflict-resolved signals ready for execution planning
    - meta: small debug metadata (safe to log)
    """
    raw_signals: List[TradeSignal]
    evaluated_signals: List[Dict[str, Any]]
    routed_signals: List[RoutedSignal]
    meta: Dict[str, Any]


# -----------------------------
# Runtime helpers (safe + deterministic)
# -----------------------------

def _runtime_dir() -> Path:
    """
    Resolve runtime directory safely without assumptions.
    Priority:
      1) CHAD_RUNTIME_DIR
      2) CHAD_ROOT/runtime
      3) ./runtime
    """
    env = os.getenv("CHAD_RUNTIME_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    root = os.getenv("CHAD_ROOT", "").strip()
    if root:
        return (Path(root).expanduser().resolve() / "runtime").resolve()
    return (Path.cwd() / "runtime").resolve()


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    """
    Atomic write to avoid partial/corrupt files.
    Never throws (caller decides).
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        tmp.replace(path)
    except Exception:
        # Fail-safe: pipeline must never crash due to telemetry.
        return


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip() == "1"


def _env_int(name: str, default: int) -> int:
    try:
        v = int(os.getenv(name, str(default)).strip())
        return v if v > 0 else default
    except Exception:
        return default


# -----------------------------
# Spam governor (Phase 11 edge-quality guardrail)
# -----------------------------

def _apply_spam_governor(raw: List[TradeSignal]) -> Tuple[List[TradeSignal], Dict[str, Any]]:
    """
    Apply strategy spam governor to raw TradeSignals.

    Contract:
    - Must NEVER crash the pipeline.
    - If the governor module is missing or errors, return raw unchanged.
    - Always returns a report dict (even if empty), so we can persist state every cycle.
    """
    report: Dict[str, Any] = {
        "ts_utc": None,
        "enabled": False,
        "in_signals": len(raw),
        "out_signals": len(raw),
        "dropped_count": 0,
        "limits": {},
        "notes": [],
    }

    if not _env_flag("CHAD_SPAM_GOVERNOR_ENABLED", "0"):
        report["notes"].append("spam_governor_disabled")
        return raw, report

    report["enabled"] = True
    report["ts_utc"] = os.getenv("TS_UTC_OVERRIDE", "") or None  # optional testing hook

    max_per_strategy = _env_int("CHAD_SPAM_MAX_PER_STRATEGY_PER_CYCLE", 6)
    max_per_pair = _env_int("CHAD_SPAM_MAX_PER_STRATEGY_SYMBOL_PER_CYCLE", 2)
    report["limits"] = {
        "max_per_strategy_per_cycle": max_per_strategy,
        "max_per_strategy_symbol_per_cycle": max_per_pair,
    }

    try:
        from chad.policy_guards.strategy_spam_governor import SpamLimits, apply_spam_governor  # type: ignore

        allowed, gov_report = apply_spam_governor(
            raw,
            limits=SpamLimits(
                max_per_strategy_per_cycle=max_per_strategy,
                max_per_strategy_symbol_per_cycle=max_per_pair,
            ),
            runtime_dir=_runtime_dir(),
        )
        # Normalize types for safety
        allowed2 = [s for s in allowed if isinstance(s, TradeSignal)]
        report["out_signals"] = len(allowed2)

        # Merge details from gov_report (but keep small)
        if isinstance(gov_report, dict):
            report["dropped_count"] = int(gov_report.get("dropped_count") or 0)
            report["strategy_counts_out"] = gov_report.get("strategy_counts_out") or {}
        return allowed2, report

    except Exception as exc:
        # Fail-safe: do not block trading logic due to governor failure
        report["notes"].append(f"spam_governor_failed:{type(exc).__name__}")
        return raw, report


def _write_spam_state(report: Dict[str, Any]) -> None:
    """
    Always attempt to write spam governor state when enabled,
    to prove the hook executed (Phase 11 audit requirement).
    """
    if not report.get("enabled", False):
        return
    state_path = _runtime_dir() / "strategy_spam_governor_state.json"
    _atomic_write_json(state_path, report)


# -----------------------------
# Main pipeline
# -----------------------------

class DecisionPipeline:
    """
    Orchestrates a full decision cycle.

    Dependencies are injected:
    - engine: must provide run_cycle(ctx) -> List[TradeSignal]
    - policy: optional; if provided must provide evaluate_signals(...)
    - router: optional; backward-compatible injection point for tests / older callers
    """

    def __init__(
        self,
        engine: Any,
        policy: Any = None,
        router: Optional[SignalRouter] = None,
        *,
        config: PipelineConfig | None = None,
    ) -> None:
        self.engine = engine
        self.policy = policy
        self.router = router or SignalRouter()
        self.config = config or PipelineConfig(use_policy=True, spam_governor_enabled=False)

    def run(
        self,
        ctx: Any,
        *,
        prices: Optional[Mapping[str, float]] = None,
        current_symbol_notional: Optional[Mapping[str, float]] = None,
        current_total_notional: float = 0.0,
    ) -> PipelineResult:
        """
        Backward-compatible alias expected by older tests/callers.
        Extra exposure arguments are accepted for compatibility even if the
        underlying pipeline remains mostly strategy/policy/router focused.
        """
        return self.run_cycle(
            ctx,
            prices=prices,
            current_symbol_notional=current_symbol_notional,
            current_total_notional=current_total_notional,
        )

    def run_cycle(
        self,
        ctx: Any,
        *,
        prices: Optional[Mapping[str, float]] = None,
        current_symbol_notional: Optional[Mapping[str, float]] = None,
        current_total_notional: float = 0.0,
    ) -> PipelineResult:
        # 1) Strategy emission
        raw_signals: List[TradeSignal] = list(self.engine.run_cycle(ctx) or [])
        raw_signals = [s for s in raw_signals if isinstance(s, TradeSignal)]

        # 1b) Spam governor (Phase 11 edge-quality protection)
        governed, spam_report = _apply_spam_governor(raw_signals)
        _write_spam_state(spam_report)
        raw_signals = governed

        # 2) Policy evaluation (optional)
        evaluated: List[Dict[str, Any]] = []
        passed_signals: List[TradeSignal] = raw_signals

        if self.config.use_policy and self.policy is not None:
            try:
                out = self.policy.evaluate_signals(
                    raw_signals,
                    current_symbol_notional=current_symbol_notional or {},
                    current_total_notional=float(current_total_notional or 0.0),
                    prices=prices or {},
                )  # type: ignore

                evaluated = list(out or [])
                allowed_signals: List[TradeSignal] = []

                for item in evaluated:
                    # Backward-compatible dict shape
                    if isinstance(item, dict):
                        allowed = bool(item.get("allowed", item.get("accepted", False)))
                        sig = item.get("signal")
                        adjusted_size = float(item.get("adjusted_size", 0.0) or 0.0)

                        if allowed and isinstance(sig, TradeSignal) and adjusted_size > 0.0:
                            if adjusted_size == float(sig.size):
                                allowed_signals.append(sig)
                            else:
                                allowed_signals.append(
                                    TradeSignal(
                                        strategy=sig.strategy,
                                        symbol=sig.symbol,
                                        side=sig.side,
                                        size=adjusted_size,
                                        confidence=sig.confidence,
                                        asset_class=sig.asset_class,
                                        created_at=sig.created_at,
                                        meta=dict(sig.meta or {}),
                                    )
                                )
                        continue

                    # Native EvaluatedSignal object shape
                    sig = getattr(item, "signal", None)
                    dec = getattr(item, "decision", None)

                    if isinstance(sig, TradeSignal) and dec is not None:
                        accepted = bool(getattr(dec, "accepted", False))
                        adjusted_size = float(getattr(dec, "adjusted_size", 0.0) or 0.0)

                        if accepted and adjusted_size > 0.0:
                            if adjusted_size == float(sig.size):
                                allowed_signals.append(sig)
                            else:
                                allowed_signals.append(
                                    TradeSignal(
                                        strategy=sig.strategy,
                                        symbol=sig.symbol,
                                        side=sig.side,
                                        size=adjusted_size,
                                        confidence=sig.confidence,
                                        asset_class=sig.asset_class,
                                        created_at=sig.created_at,
                                        meta=dict(sig.meta or {}),
                                    )
                                )

                passed_signals = allowed_signals

            except Exception as exc:
                # Fail-open policy is intentional for robustness, but we record why.
                evaluated = [{"allowed": True, "reason": f"policy_error:{type(exc).__name__}"}]
                passed_signals = raw_signals

        # 3) Routing / conflict resolution
        routed: List[RoutedSignal] = []
        try:
            routed = list(self.router.route(passed_signals))
            routed = [r for r in routed if isinstance(r, RoutedSignal)]
        except Exception as exc:
            # Hard fail here would be dangerous; return empty routed signals.
            logger.exception("route_signals failed: %s", exc)
            routed = []

        meta = {
            "raw_signals": len(raw_signals),
            "passed_signals": len(passed_signals),
            "routed_signals": len(routed),
            "spam_governor_enabled": bool(spam_report.get("enabled", False)),
            "spam_dropped": int(spam_report.get("dropped_count") or 0),
        }

        return PipelineResult(
            raw_signals=raw_signals,
            evaluated_signals=evaluated,
            routed_signals=routed,
            meta=meta,
        )
