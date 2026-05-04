"""
CHAD ML Veto Predictor

Loads the trained XGBoost model, validates its manifest, scores trade
intents, and returns a structured shadow decision. Enforcement is
gated by ``CHAD_ML_VETO_ENABLED`` (default OFF). Even when enforcement
is on, the predictor never blocks exits, reductions, hedges, or
liquidations, and never enforces outside the canary strategy
allowlist.

This module is deliberately fail-open on every error path: prediction
failures, missing model, missing manifest, stale manifest, and
malformed inputs all collapse to ``would_veto=False`` /
``final_action="shadow_only"``. A model error must never strand
trade flow.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

logger = logging.getLogger("chad.ml_veto")

REPO_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = REPO_ROOT / "shared" / "models" / "xgb_veto_model.json"
MANIFEST_PATH = REPO_ROOT / "shared" / "models" / "xgb_veto_manifest.json"

# Default threshold above which a positive prediction (P(loss)) means
# "would veto". Tuned during shadow soak.
VETO_THRESHOLD = float(os.environ.get("CHAD_ML_VETO_THRESHOLD", "0.65"))

# Manifest considered stale beyond this many days. Stale → no
# enforcement (shadow logging continues).
MANIFEST_STALE_DAYS = int(os.environ.get("CHAD_ML_VETO_STALE_DAYS", "30"))

# Manifests below this minimum sample count are not promotable to
# enforcement. Mirrors trainer's MIN_TRAINING_SAMPLES so the predictor
# can defend itself against a manually-installed weak model.
MIN_MANIFEST_SAMPLES = int(os.environ.get("CHAD_ML_VETO_MIN_SAMPLES", "100"))

# Final-action sentinels exposed on every shadow decision.
ACTION_SHADOW_ONLY = "shadow_only"
ACTION_VETO = "veto"
ACTION_PASS = "pass"

# Intent classes. Only ENTRY is ever eligible for enforcement; the
# remainder are protective and must always pass to the broker.
INTENT_ENTRY = "entry"
INTENT_EXIT = "exit"
INTENT_REDUCTION = "reduction"
INTENT_HEDGE = "hedge"
INTENT_LIQUIDATION = "liquidation"

# Reason strings emitted on the shadow decision for downstream
# auditing. Stable identifiers — do not reword without updating tests.
REASON_DISABLED = "enforcement_disabled"
REASON_PROTECTIVE = "protective_intent_never_vetoed"
REASON_NOT_IN_CANARY = "strategy_not_in_canary_allowlist"
REASON_NO_MODEL = "model_unavailable"
REASON_NO_MANIFEST = "manifest_missing_or_invalid"
REASON_MANIFEST_STALE = "manifest_stale"
REASON_MANIFEST_LOW_SAMPLES = "manifest_below_min_samples"
REASON_FEATURE_FAIL = "feature_extraction_failed"
REASON_PREDICT_FAIL = "prediction_failed"
REASON_VETO_FIRED = "loss_probability_above_threshold"
REASON_BELOW_THRESHOLD = "loss_probability_below_threshold"

_STRATEGY_MAP = {
    "alpha": 0, "alpha_intraday": 1, "alpha_futures": 2,
    "alpha_options": 3, "alpha_crypto": 4, "beta": 5,
    "beta_trend": 6, "delta": 7, "delta_pairs": 8,
    "gamma": 9, "gamma_futures": 10, "gamma_reversion": 11,
    "omega_vol": 12, "omega_momentum_options": 13,
    "omega_macro": 14, "omega": 15,
}
_REGIME_MAP = {
    "trending_bull": 0, "ranging": 1, "trending_bear": 2,
    "volatile": 3, "unknown": 4,
}

FEATURE_NAMES = [
    "strategy_encoded",
    "regime_encoded",
    "vix_level",
    "hour_of_day",
    "day_of_week",
    "scr_sizing_factor",
    "is_buy",
    "equity_normalized",
    "recent_win_rate",
    "regime_confidence",
]


@dataclass(frozen=True)
class ShadowDecision:
    """Structured ML veto decision suitable for INFO-level shadow
    logging and downstream auditing.

    Fields are the contract Batch 9 consumers (live_loop, audit
    pipelines, evaluation jobs) rely on; renames must be coordinated
    with those consumers.
    """
    model_version: str
    manifest_path: str
    manifest_hash: str
    threshold: float
    features: Dict[str, float]
    prediction: float
    would_veto: bool
    final_action: str  # ACTION_*
    reason: str
    intent_class: str
    enforcement_enabled: bool
    canary_match: bool
    manifest_stale: bool
    manifest_valid: bool
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Manifest + model loading. Cached on first call; reload via
# ``reset_manifest_cache`` from tests when manifest fixtures change.
# ---------------------------------------------------------------------------


_model = None
_model_load_attempted = False
_manifest_cache: Optional[Dict[str, Any]] = None
_manifest_load_attempted = False


def reset_manifest_cache() -> None:
    """Reset cached manifest + model so a subsequent ``score_intent``
    re-reads disk. Tests use this to swap in a fresh manifest."""
    global _model, _model_load_attempted, _manifest_cache, _manifest_load_attempted
    _model = None
    _model_load_attempted = False
    _manifest_cache = None
    _manifest_load_attempted = False


def _load_manifest() -> Optional[Dict[str, Any]]:
    global _manifest_cache, _manifest_load_attempted
    if _manifest_load_attempted:
        return _manifest_cache
    _manifest_load_attempted = True
    try:
        if not MANIFEST_PATH.exists():
            logger.info("ml_veto: manifest missing at %s", MANIFEST_PATH)
            _manifest_cache = None
            return None
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            logger.warning("ml_veto: manifest not a dict — invalid")
            _manifest_cache = None
            return None
        _manifest_cache = data
        return _manifest_cache
    except Exception as exc:  # noqa: BLE001 — fail-open
        logger.warning("ml_veto: manifest_load_failed err=%s", exc)
        _manifest_cache = None
        return None


def _manifest_is_valid(manifest: Optional[Mapping[str, Any]]) -> Tuple[bool, str]:
    """Return (is_valid, reason). A manifest is valid when it carries
    every field a downstream auditor needs to reproduce the model.
    """
    if not isinstance(manifest, Mapping):
        return False, REASON_NO_MANIFEST
    required = (
        "schema_version",
        "model_path",
        "trained_at_utc",
        "training_samples",
        "feature_names",
        "metrics",
    )
    for key in required:
        if key not in manifest:
            return False, REASON_NO_MANIFEST

    metrics = manifest.get("metrics") or {}
    if not isinstance(metrics, Mapping) or not metrics:
        return False, REASON_NO_MANIFEST
    # Accuracy must be present *and* finite — guards against a manually
    # zeroed / NaN'd manifest masquerading as a real one.
    acc = metrics.get("accuracy")
    if acc is None or not isinstance(acc, (int, float)) or not math.isfinite(float(acc)):
        return False, REASON_NO_MANIFEST

    samples = manifest.get("training_samples")
    try:
        n = int(samples)
    except (TypeError, ValueError):
        return False, REASON_NO_MANIFEST
    if n < MIN_MANIFEST_SAMPLES:
        return False, REASON_MANIFEST_LOW_SAMPLES

    return True, ""


def _manifest_is_stale(manifest: Mapping[str, Any]) -> bool:
    ts = manifest.get("trained_at_utc")
    if not isinstance(ts, str) or not ts:
        return True
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return True
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - dt
    return age > timedelta(days=MANIFEST_STALE_DAYS)


def _manifest_hash_short(manifest: Mapping[str, Any]) -> str:
    """Prefer the model_sha256 field when present (Batch 9 manifests).
    Fall back to dataset_hash, then to a sentinel. Keeps shadow lines
    short while still identifiable."""
    for key in ("model_sha256", "dataset_hash"):
        v = manifest.get(key)
        if isinstance(v, str) and v:
            return v[:24]
    return "sha256:unknown"


def _model_version(manifest: Optional[Mapping[str, Any]]) -> str:
    if isinstance(manifest, Mapping):
        v = manifest.get("model_version")
        if isinstance(v, str) and v:
            return v
        ts = manifest.get("trained_at_utc")
        if isinstance(ts, str) and ts:
            return f"xgb_veto_{ts}"
    return "xgb_veto_unknown"


def _load_model():
    global _model, _model_load_attempted
    if _model_load_attempted:
        return _model
    _model_load_attempted = True
    try:
        import xgboost as xgb  # type: ignore[import]
        if not MODEL_PATH.exists():
            logger.info("ml_veto: model not found at %s — veto disabled",
                        MODEL_PATH)
            return None
        bst = xgb.Booster()
        bst.load_model(str(MODEL_PATH))
        logger.info("ml_veto: model loaded from %s", MODEL_PATH)
        _model = bst
        return _model
    except Exception as e:  # noqa: BLE001 — fail-open
        logger.warning("ml_veto: model load failed err=%s — veto disabled", e)
        return None


# ---------------------------------------------------------------------------
# Intent classification + canary gate
# ---------------------------------------------------------------------------


_PROTECTIVE_TAGS = {"exit", "close", "closing", "liquidation",
                    "reduce", "reducing", "reduction",
                    "hedge", "hedging"}


def _intent_meta(intent: Any) -> Mapping[str, Any]:
    meta = getattr(intent, "meta", None)
    return meta if isinstance(meta, Mapping) else {}


def _intent_tags(intent: Any) -> List[str]:
    tags = getattr(intent, "tags", None)
    if isinstance(tags, list):
        return [str(t).strip().lower() for t in tags]
    meta = _intent_meta(intent)
    raw = meta.get("tags") or meta.get("signal_tags") or []
    if isinstance(raw, list):
        return [str(t).strip().lower() for t in raw]
    return []


def classify_intent(intent: Any, ctx: Optional[Mapping[str, Any]] = None) -> str:
    """Classify a trade intent so the predictor can refuse to enforce
    on protective intents.

    Returns one of INTENT_ENTRY / INTENT_EXIT / INTENT_REDUCTION /
    INTENT_HEDGE / INTENT_LIQUIDATION. Heuristics, in order:
      1. action / side strings ``EXIT``, ``CLOSE`` → INTENT_EXIT.
      2. tag set intersection with protective markers (exit, close,
         liquidation, reduce, hedge) → matching INTENT_*.
      3. meta.is_exit / meta.exit / meta.reason in {max_hold_exit,
         stop_loss, take_profit, exit_signal} → INTENT_EXIT.
      4. meta.reduce_only / meta.is_reducing → INTENT_REDUCTION.
      5. meta.hedge / meta.is_hedge / strategy ending in `_hedge`
         → INTENT_HEDGE.
      6. ctx-supplied flip detection (``ctx['is_flip']`` truthy or
         ``ctx['intent_class']`` already set by the caller) → respect it.
      7. Default INTENT_ENTRY.

    The classifier is intentionally generous (any signal of "this is a
    protective trade" wins) so that ML enforcement cannot accidentally
    block an exit.
    """
    side = str(getattr(intent, "side", "") or "").strip().upper()
    action = str(getattr(intent, "action", "") or "").strip().upper()
    if side in {"EXIT", "CLOSE"} or action in {"EXIT", "CLOSE"}:
        return INTENT_EXIT

    tags = set(_intent_tags(intent))
    if "liquidation" in tags:
        return INTENT_LIQUIDATION
    if tags & {"exit", "close", "closing"}:
        return INTENT_EXIT
    if tags & {"reduce", "reducing", "reduction"}:
        return INTENT_REDUCTION
    if tags & {"hedge", "hedging"}:
        return INTENT_HEDGE

    meta = _intent_meta(intent)
    reason = str(meta.get("reason") or "").strip().lower()
    if (meta.get("is_exit") is True
            or meta.get("exit") is True
            or meta.get("is_closing") is True
            or reason in {"max_hold_exit", "stop_loss", "take_profit",
                          "exit_signal", "exit"}):
        return INTENT_EXIT
    if meta.get("liquidation") is True or reason == "liquidation":
        return INTENT_LIQUIDATION
    if meta.get("reduce_only") is True or meta.get("is_reducing") is True:
        return INTENT_REDUCTION
    if meta.get("hedge") is True or meta.get("is_hedge") is True:
        return INTENT_HEDGE

    strategy = str(getattr(intent, "strategy", "") or "").strip().lower()
    if strategy.endswith("_hedge") or strategy == "hedge":
        return INTENT_HEDGE

    # Caller-supplied overrides take precedence over default ENTRY.
    if isinstance(ctx, Mapping):
        explicit = str(ctx.get("intent_class") or "").strip().lower()
        if explicit in {INTENT_EXIT, INTENT_REDUCTION,
                        INTENT_HEDGE, INTENT_LIQUIDATION, INTENT_ENTRY}:
            return explicit
        if ctx.get("is_flip"):
            return INTENT_EXIT  # a flip closes the existing leg.

    return INTENT_ENTRY


def is_protective_intent(intent_class: str) -> bool:
    return intent_class != INTENT_ENTRY


def _canary_strategies() -> Optional[set]:
    """Comma-separated CHAD_ML_VETO_CANARY_STRATEGIES enables the
    allowlist. ``None`` means "no allowlist configured" — and in that
    case enforcement is denied for every strategy until an operator
    flips at least one strategy on. Empty string is treated as no
    allowlist (all-strategies *blocked* from enforcement).
    """
    raw = os.environ.get("CHAD_ML_VETO_CANARY_STRATEGIES", "")
    if not raw.strip():
        return None
    return {s.strip().lower() for s in raw.split(",") if s.strip()}


def enforcement_enabled() -> bool:
    """``True`` only when CHAD_ML_VETO_ENABLED is set to a truthy
    value. The default — empty / unset — must remain OFF so a fresh
    deploy or test run never blocks live trades."""
    return os.environ.get("CHAD_ML_VETO_ENABLED", "").strip().lower() in (
        "1", "true", "yes", "on"
    )


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features(intent: Any, ctx: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    """Extract feature dict from intent + context. Returns ``None``
    when extraction fails — the caller must treat that as fail-open
    (no enforcement)."""
    try:
        strategy = str(getattr(intent, "strategy", "") or "").lower()
        side = str(getattr(intent, "side", "") or "").upper()

        regime_data = (ctx.get("regime") if isinstance(ctx, Mapping) else {}) or {}
        regime = str(regime_data.get("regime", "unknown")).lower()

        prices = (ctx.get("prices") if isinstance(ctx, Mapping) else {}) or {}
        vix = float(prices.get("VIX", prices.get("^VIX", 20.0)) or 20.0)

        scr = (ctx.get("scr") if isinstance(ctx, Mapping) else {}) or {}
        sizing = float(scr.get("sizing_factor", 0.25) or 0.25)

        now = datetime.now(timezone.utc)

        health = (ctx.get("strategy_health") if isinstance(ctx, Mapping) else {}) or {}
        strat_health = (health.get(strategy) if isinstance(health, Mapping) else {}) or {}
        win_rate = float(
            strat_health.get(
                "win_rate", strat_health.get("health_score", 0.5)
            ) or 0.5
        )

        portfolio = (ctx.get("portfolio") if isinstance(ctx, Mapping) else {}) or {}
        equity = float(
            portfolio.get(
                "total_equity", portfolio.get("net_liquidation", 200000)
            )
            or 200000
        )

        return {
            "strategy_encoded": float(_STRATEGY_MAP.get(strategy, 15)),
            "regime_encoded": float(_REGIME_MAP.get(regime, 4)),
            "vix_level": vix,
            "hour_of_day": float(now.hour),
            "day_of_week": float(now.weekday()),
            "scr_sizing_factor": sizing,
            "is_buy": 1.0 if "BUY" in side else 0.0,
            "equity_normalized": equity / 200000.0,
            "recent_win_rate": win_rate,
            "regime_confidence": float(
                regime_data.get("confidence", 0.5) or 0.5
            ),
        }
    except Exception as e:  # noqa: BLE001 — fail-open
        logger.debug("ml_veto: feature_extraction_failed err=%s", e)
        return None


def _features_to_vector(features: Mapping[str, float]) -> List[float]:
    return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]


# ---------------------------------------------------------------------------
# Public scoring API
# ---------------------------------------------------------------------------


def score_intent(intent: Any, ctx: Optional[Mapping[str, Any]] = None) -> ShadowDecision:
    """Score an intent and return a structured shadow decision.

    The decision is the auditable contract Batch 9 expects: every
    field needed to reproduce the prediction (model_version,
    manifest_hash, threshold, features, prediction) is present, the
    enforcement classification is explicit (would_veto, final_action,
    intent_class), and a stable ``reason`` is emitted so log grep
    queries don't have to rely on free-form strings.

    ``score_intent`` is **fail-open**: any error path
    (model unavailable, manifest stale, feature extraction failure,
    prediction crash) collapses to ``final_action="shadow_only"`` /
    ``would_veto=False`` and returns rather than raising.
    """
    ctx = ctx if isinstance(ctx, Mapping) else {}
    manifest = _load_manifest()
    manifest_valid, manifest_reason = _manifest_is_valid(manifest)
    manifest_stale = bool(
        manifest_valid and isinstance(manifest, Mapping) and _manifest_is_stale(manifest)
    )
    model_version = _model_version(manifest)
    manifest_hash = _manifest_hash_short(manifest) if isinstance(manifest, Mapping) else "sha256:unknown"
    threshold = VETO_THRESHOLD
    enforcement = enforcement_enabled()

    intent_class = classify_intent(intent, ctx)
    canary = _canary_strategies()
    strategy = str(getattr(intent, "strategy", "") or "").strip().lower()
    canary_match = bool(canary and strategy in canary)

    # Extract features once. We carry raw zeros into the decision when
    # extraction fails so the shadow log still has structure.
    features = extract_features(intent, ctx) or {n: 0.0 for n in FEATURE_NAMES}

    # Reasons surfaced via this set are upstream-error signals: when
    # set, they take precedence over the "below_threshold" default so
    # operators can grep ``reason=model_unavailable`` directly.
    _ERROR_REASONS = {REASON_NO_MODEL, REASON_PREDICT_FAIL, REASON_FEATURE_FAIL}

    def _decide(prediction: float, reason: str, would_veto: bool) -> ShadowDecision:
        # Build the final action with the protective-intent and canary
        # guards applied. Even when enforcement is enabled, exits /
        # reductions / hedges / liquidations always pass.
        if not enforcement:
            final = ACTION_SHADOW_ONLY
            final_reason = reason if reason else REASON_DISABLED
        elif is_protective_intent(intent_class):
            # Protective intents must pass irrespective of error state —
            # they're protective regardless of why the model couldn't run.
            final = ACTION_PASS
            final_reason = REASON_PROTECTIVE
        elif reason in _ERROR_REASONS:
            # Model couldn't run / score — fail open with the explicit
            # error signal preserved.
            final = ACTION_PASS
            final_reason = reason
        elif canary is None or not canary_match:
            final = ACTION_PASS
            final_reason = REASON_NOT_IN_CANARY
        elif not manifest_valid:
            final = ACTION_PASS
            final_reason = manifest_reason or REASON_NO_MANIFEST
        elif manifest_stale:
            final = ACTION_PASS
            final_reason = REASON_MANIFEST_STALE
        elif would_veto:
            final = ACTION_VETO
            final_reason = REASON_VETO_FIRED
        else:
            final = ACTION_PASS
            final_reason = REASON_BELOW_THRESHOLD

        return ShadowDecision(
            model_version=model_version,
            manifest_path=str(MANIFEST_PATH),
            manifest_hash=manifest_hash,
            threshold=float(threshold),
            features=dict(features),
            prediction=float(prediction),
            would_veto=bool(would_veto),
            final_action=final,
            reason=final_reason,
            intent_class=intent_class,
            enforcement_enabled=enforcement,
            canary_match=canary_match,
            manifest_stale=manifest_stale,
            manifest_valid=manifest_valid,
        )

    # Fast-path: no model → fail-open shadow_only / pass.
    model = _load_model()
    if model is None:
        return _decide(0.0, REASON_NO_MODEL, would_veto=False)

    try:
        import numpy as np  # local import keeps top-level optional
        import xgboost as xgb  # type: ignore[import]
        vec = _features_to_vector(features)
        dmatrix = xgb.DMatrix(np.array([vec]), feature_names=FEATURE_NAMES)
        loss_prob = float(model.predict(dmatrix)[0])
    except Exception as e:  # noqa: BLE001 — fail-open
        logger.debug("ml_veto: prediction_failed err=%s", e)
        return _decide(0.0, REASON_PREDICT_FAIL, would_veto=False)

    would_veto = bool(loss_prob > threshold)
    decision = _decide(loss_prob, REASON_BELOW_THRESHOLD, would_veto=would_veto)

    if decision.final_action == ACTION_VETO:
        logger.warning(
            "ML_VETO model_version=%s manifest_hash=%s symbol=%s strategy=%s "
            "loss_prob=%.3f threshold=%.2f",
            decision.model_version,
            decision.manifest_hash,
            getattr(intent, "symbol", "?"),
            getattr(intent, "strategy", "?"),
            decision.prediction,
            decision.threshold,
        )
    return decision


def should_veto(intent: Any, ctx: Optional[Mapping[str, Any]] = None) -> Tuple[bool, float]:
    """Backwards-compatible scalar API used by older callers.

    Returns ``(would_veto, loss_probability)``. Mirrors the historic
    contract — ``would_veto=False`` on every error — but is now
    implemented in terms of ``score_intent`` so it inherits all the
    guard rails (protective-intent, canary, manifest staleness).
    The bool reflects ``would_veto`` (the model's opinion), not the
    final action; callers that care about enforcement should use
    ``score_intent`` directly.
    """
    decision = score_intent(intent, ctx)
    return decision.would_veto, decision.prediction


__all__ = [
    "ShadowDecision",
    "score_intent",
    "should_veto",
    "classify_intent",
    "is_protective_intent",
    "extract_features",
    "enforcement_enabled",
    "reset_manifest_cache",
    "FEATURE_NAMES",
    "MODEL_PATH",
    "MANIFEST_PATH",
    "VETO_THRESHOLD",
    "INTENT_ENTRY",
    "INTENT_EXIT",
    "INTENT_REDUCTION",
    "INTENT_HEDGE",
    "INTENT_LIQUIDATION",
    "ACTION_SHADOW_ONLY",
    "ACTION_VETO",
    "ACTION_PASS",
    "REASON_DISABLED",
    "REASON_PROTECTIVE",
    "REASON_NOT_IN_CANARY",
    "REASON_NO_MODEL",
    "REASON_NO_MANIFEST",
    "REASON_MANIFEST_STALE",
    "REASON_MANIFEST_LOW_SAMPLES",
    "REASON_FEATURE_FAIL",
    "REASON_PREDICT_FAIL",
    "REASON_VETO_FIRED",
    "REASON_BELOW_THRESHOLD",
]
