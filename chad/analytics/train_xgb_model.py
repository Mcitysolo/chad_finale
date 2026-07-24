from __future__ import annotations

"""
chad/analytics/train_xgb_model.py

Train an XGBoost model from CHAD trade history (Phase-7 safe baseline).

Design Goals
------------
- Production-safe:
    - Never crashes the process on bad/missing data.
    - Exits with code 0 when there is nothing to train on.
    - Exits with code 1 only for genuine training failures.
- No side effects outside the CHAD tree:
    - Reads from data/trades/trade_history_*.ndjson
    - Writes to shared/models/xgb_model.json and model_performance.json
- Typed, deterministic, and testable.

You MUST:
- `pip install xgboost` inside the CHAD venv before running this in earnest.
- Adjust feature engineering if your trade_history schema differs.
"""

import hashlib
import json
import logging
import math
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import numpy as np

try:
    import xgboost as xgb  # type: ignore[import]
except Exception:  # noqa: BLE001
    xgb = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "trades"
MODELS_DIR = ROOT / "shared" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Tracked baseline paths — read by the predictor only as a fallback
# when no promoted runtime-current model exists. The trainer must NOT
# overwrite these under normal operation; they represent the reviewed
# baseline and are restored from git when a regressed retrain lands.
MODEL_PATH = MODELS_DIR / "xgb_veto_model.json"
MANIFEST_PATH = MODELS_DIR / "xgb_veto_manifest.json"
PERF_PATH = MODELS_DIR / "model_performance.json"

# Candidate / promoted runtime locations (untracked under runtime/).
# Weekly retrain lands a candidate in CANDIDATES_DIR/<UTC ts>/; the
# operator promotes it via scripts/promote_xgb_veto.py into
# RUNTIME_MODEL_DIR which the predictor then prefers over MODEL_PATH.
CANDIDATES_DIR = ROOT / "runtime" / "models" / "xgb_veto" / "candidates"
RUNTIME_MODEL_DIR = ROOT / "runtime" / "models" / "xgb_veto" / "current"
RUNTIME_MODEL_PATH = RUNTIME_MODEL_DIR / "xgb_veto_model.json"
RUNTIME_MANIFEST_PATH = RUNTIME_MODEL_DIR / "xgb_veto_manifest.json"

# Manifest schema bumped when trainer started excluding quarantined,
# pnl_untrusted, untrusted-fill_id, nonfinite, and unusable-legacy
# rows from training labels (Batch 9).
MANIFEST_SCHEMA_VERSION = "xgb_manifest.v2"

# Minimum sample count for a training run to be considered "promotable".
# Training never overwrites the model below this floor; the prior model
# stays in place. Configurable via env so canary/local runs can lower it.
MIN_TRAINING_SAMPLES = 100

# Schema versions whose label is considered usable. Older schemas with
# unusable PnL semantics are excluded by default; an operator can opt
# them in by listing them in CHAD_ML_TRAIN_LEGACY_SCHEMAS.
USABLE_TRADE_SCHEMAS: Tuple[str, ...] = (
    "closed_trade.v1",
    "closed_trade.v2",
)

# Feature encodings — must match chad/analytics/ml_veto_predictor.py
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

LOG = logging.getLogger("chad.analytics.xgb_train")


@dataclass(frozen=True)
class TrainingResult:
    ok: bool
    reason: str
    n_samples: int
    n_features: int
    metrics: Dict[str, float]
    excluded: Dict[str, int] = field(default_factory=dict)
    dataset_hash: str = ""
    model_version: str = ""
    manifest_path: Optional[Path] = None
    backup_path: Optional[Path] = None


# --------------------------------------------------------------------------
# W6B-7 (D3): loud-but-green refusal
# --------------------------------------------------------------------------
#
# chad-xgb-train.service has sat in systemd `failed` since 2026-07-19, which is
# EXS5's only red. It is not crashing — it is REFUSING, correctly: the Epoch-3
# book yields 72 trusted closed trades against a required 100, because 2153 rows
# are untrusted and 6 quarantined. Fail-closed behaviour working as designed.
#
# The operator ruling (D3) is that refusing to train on untrusted rows is the
# system honouring its own rules, so it should exit 0 — but LOUDLY. Lowering
# MIN_TRAINING_SAMPLES is explicitly NOT the fix: training a live-money veto
# model on a book we have already declared untrustworthy is the worst available
# outcome. Leaving it red is not the fix either, because a permanently-red
# sentinel is one operators learn to ignore.
#
# Note the module's own docstring at main() ALREADY declared this contract:
#   "Returns 0 on: no trade data, no usable samples, training completed
#    successfully. Returns 1 only on hard training/model errors."
# `return 0 if result.ok else 1` contradicted it — an insufficient-samples
# refusal is not a "hard training/model error". So this is the code being made
# to honour an intent it had already written down, not a policy change.

XGB_TRAIN_STATE_PATH = ROOT / "runtime" / "xgb_train_state.json"
XGB_TRAIN_STATE_SCHEMA = "xgb_train_state.v1"
XGB_TRAIN_STATE_TTL_SECONDS = 30 * 60 * 60  # daily timer + slack

# Refusals are data-quality outcomes: the job ran correctly and declined. Any
# other failure is a genuine error and must stay exit 1.
DECLINE_REASONS = (
    "Not enough samples for training",
    "No trade rows",
    "No usable samples",
)


def _is_decline(reason: str) -> bool:
    return any(reason.startswith(p) for p in DECLINE_REASONS)


def _write_train_state(
    *,
    outcome: str,
    reason: str,
    usable_rows: int,
    excluded: Optional[Dict[str, int]] = None,
    model_version: str = "",
    dataset_hash: str = "",
    exclusions_loaded: bool = True,
) -> None:
    """Publish xgb_train_state.v1 so a refusal is visible without journal
    archaeology.

    The countdown fields are the point: `usable_rows` / `required_rows` /
    `rows_short` make "how far are we from being able to train?" a number an
    operator can watch, rather than a fact buried in a log line on a unit that
    reads as broken. Best-effort — a state-write failure must never turn a
    successful training run into a failed one.
    """
    payload: Dict[str, Any] = {
        "schema_version": XGB_TRAIN_STATE_SCHEMA,
        "ts_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ttl_seconds": XGB_TRAIN_STATE_TTL_SECONDS,
        "outcome": outcome,                 # trained | declined | error
        "reason": reason,
        "usable_rows": int(usable_rows),
        "required_rows": int(MIN_TRAINING_SAMPLES),
        "rows_short": max(0, int(MIN_TRAINING_SAMPLES) - int(usable_rows)),
        "exclusions": dict(excluded or {}),
        # False means the run could not tell trusted rows from untrusted ones.
        # A `trained` outcome with exclusions_loaded=false would be a red flag.
        "exclusions_loaded": bool(exclusions_loaded),
        "model_version": model_version,
        "dataset_hash": dataset_hash,
        "threshold_policy": (
            "MIN_TRAINING_SAMPLES is a trust floor, not a tuning knob. Lowering it "
            "to clear this state would train a live-money veto model on rows the "
            "system has already declared untrusted."
        ),
    }
    try:
        XGB_TRAIN_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = XGB_TRAIN_STATE_PATH.with_suffix(f".json.tmp.{os.getpid()}")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, XGB_TRAIN_STATE_PATH)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("could not write %s: %s", XGB_TRAIN_STATE_PATH, exc)


def _setup_logging() -> None:
    if LOG.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _load_trade_rows(max_files: int = 120) -> List[Dict[str, Any]]:
    """
    Load recent trade history rows from NDJSON files.

    Parameters
    ----------
    max_files : int
        Maximum number of daily files to load (newest first).

    Returns
    -------
    List[Dict[str, Any]]
        Parsed trade rows across the selected files.
    """
    if not DATA_DIR.is_dir():
        LOG.warning("Trade data dir missing: %s", DATA_DIR)
        return []

    files = sorted(DATA_DIR.glob("trade_history_*.ndjson"))
    if not files:
        LOG.warning("No trade_history_*.ndjson files found under %s", DATA_DIR)
        return []

    rows: List[Dict[str, Any]] = []
    for path in reversed(files[-max_files:]):
        LOG.info("Loading trades from %s", path)
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:  # noqa: BLE001
                        LOG.exception("Failed to parse line in %s", path)
        except Exception:  # noqa: BLE001
            LOG.exception("Failed to open trade history file: %s", path)
    return rows


def _legacy_schemas_allowed() -> Set[str]:
    """Optional opt-in legacy schemas via CHAD_ML_TRAIN_LEGACY_SCHEMAS.

    Operators can list schema_version values (comma-separated) that
    should be re-admitted into training labels even though they are
    older than the current usable schema set. Empty by default.
    """
    import os
    raw = os.environ.get("CHAD_ML_TRAIN_LEGACY_SCHEMAS", "")
    return {s.strip() for s in raw.split(",") if s.strip()}


def _resolve_pnl(payload: Mapping[str, Any]) -> Optional[float]:
    """Prefer the canonical per-trade pnl_breakdown.net_pnl when the
    breakdown is present and well-formed; otherwise fall back to
    payload.net_pnl, then payload.pnl. Returns None when no usable
    PnL is available, when the value is non-numeric, or when the
    value is non-finite (NaN/Inf).
    """
    breakdown = payload.get("pnl_breakdown")
    if isinstance(breakdown, Mapping):
        net = breakdown.get("net_pnl")
        if net is not None:
            try:
                v = float(net)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(v):
                return None
            return v

    for key in ("net_pnl", "pnl"):
        v_raw = payload.get(key)
        if v_raw is None:
            continue
        try:
            v = float(v_raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(v):
            return None
        return v
    return None


def _build_dataset(
    rows: List[Dict[str, Any]],
    invalid_fill_ids: Optional[Set[str]] = None,
    invalid_trade_hashes: Optional[Set[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Build the 10-feature matrix used by the ML veto predictor.

    Label: 1 = loss (pnl <= 0), 0 = win (pnl > 0). We predict P(loss).

    Trade rows are filtered to exclude:
      * quarantined records (by record_hash)
      * untrusted records (pnl_untrusted flag, tag, or extra)
      * derived closed trades whose fill_ids intersect the untrusted
        fill set (so a quarantined opening or closing fill can never
        seed a trusted training label)
      * non-finite PnL (NaN/Inf) and unparseable PnL
      * legacy schema_version rows whose label semantics are not
        currently considered usable (override via
        CHAD_ML_TRAIN_LEGACY_SCHEMAS)

    Returns ``(X, y, excluded)`` so callers can record the exclusion
    counts alongside the model manifest.
    """
    from datetime import datetime

    bad_fills = invalid_fill_ids or set()
    bad_hashes = invalid_trade_hashes or set()
    legacy_allow = _legacy_schemas_allowed()
    usable_schemas = set(USABLE_TRADE_SCHEMAS) | legacy_allow

    X: List[List[float]] = []
    y: List[int] = []
    excluded: Dict[str, int] = {
        "quarantined": 0,
        "untrusted": 0,
        "untrusted_fill_id": 0,
        "nonfinite_pnl": 0,
        "missing_pnl": 0,
        "legacy_schema": 0,
        "malformed": 0,
    }

    for raw in rows:
        if not isinstance(raw, dict):
            excluded["malformed"] += 1
            continue
        record_hash = raw.get("record_hash")
        payload = raw.get("payload")
        if not isinstance(payload, Mapping):
            payload = raw

        # Quarantined record_hash → drop.
        if isinstance(record_hash, str) and record_hash in bad_hashes:
            excluded["quarantined"] += 1
            continue

        # pnl_untrusted on payload, top-level, tags, or extra → drop.
        if _is_untrusted_payload(payload) or raw.get("pnl_untrusted") is True:
            excluded["untrusted"] += 1
            continue

        # Any fill_id intersection with the untrusted fill set → drop.
        fid = payload.get("fill_id")
        fids = payload.get("fill_ids")
        if isinstance(fid, str) and fid in bad_fills:
            excluded["untrusted_fill_id"] += 1
            continue
        if isinstance(fids, list) and any(
            isinstance(f, str) and f in bad_fills for f in fids
        ):
            excluded["untrusted_fill_id"] += 1
            continue

        # Schema gate — closed_trade.v1+ only unless explicitly opted in.
        schema_version = str(payload.get("schema_version") or "").strip()
        if schema_version and schema_version not in usable_schemas:
            excluded["legacy_schema"] += 1
            continue

        # Resolve PnL, preferring pnl_breakdown.net_pnl when present.
        pnl_f = _resolve_pnl(payload)
        if pnl_f is None:
            # Distinguish "missing entirely" from "present but unparseable/nonfinite":
            if (payload.get("pnl_breakdown") is None
                    and payload.get("net_pnl") is None
                    and payload.get("pnl") is None):
                excluded["missing_pnl"] += 1
            else:
                excluded["nonfinite_pnl"] += 1
            continue

        strategy = str(payload.get("strategy", "") or "").lower()
        side = str(payload.get("side", "") or "").upper()
        regime = str(payload.get("regime", "") or "unknown").lower()
        if regime not in _REGIME_MAP:
            regime = "unknown"

        hour = 12.0
        weekday = 2.0
        ts = (payload.get("exit_time_utc")
              or payload.get("entry_time_utc")
              or payload.get("timestamp_utc"))
        if isinstance(ts, str) and ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                hour = float(dt.hour)
                weekday = float(dt.weekday())
            except Exception:
                pass

        # Defaults for fields not stored in trade history — match predictor.
        vix = 20.0
        sizing = 0.25
        equity_norm = 1.0
        win_rate = 0.5
        regime_conf = 0.5

        X.append([
            float(_STRATEGY_MAP.get(strategy, 15)),
            float(_REGIME_MAP.get(regime, 4)),
            vix,
            hour,
            weekday,
            sizing,
            1.0 if "BUY" in side else 0.0,
            equity_norm,
            win_rate,
            regime_conf,
        ])
        y.append(0 if pnl_f > 0.0 else 1)

    if not X:
        return (
            np.empty((0, 0), dtype=float),
            np.empty((0,), dtype=int),
            excluded,
        )

    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), excluded


def _is_untrusted_payload(payload: Mapping[str, Any]) -> bool:
    """Mirror chad/utils/quarantine._payload_is_untrusted without
    importing it (so this trainer keeps a tight dependency surface
    when invoked under restricted environments)."""
    if not isinstance(payload, Mapping):
        return False
    if payload.get("pnl_untrusted") is True:
        return True
    extra = payload.get("extra")
    if isinstance(extra, Mapping) and extra.get("pnl_untrusted") is True:
        return True
    tags = payload.get("tags")
    if isinstance(tags, list) and any(
        str(t).strip().lower() == "pnl_untrusted" for t in tags
    ):
        return True
    return False


def _hash_dataset(X: np.ndarray, y: np.ndarray) -> str:
    """SHA256 over the row-major bytes of (X, y). Empty datasets hash
    to a stable sentinel so callers can detect "nothing to train"
    without parsing manifest fields."""
    if X.size == 0 or y.size == 0:
        return "sha256:empty"
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(y, dtype=np.int32).tobytes())
    return f"sha256:{h.hexdigest()}"


def _hash_file(path: Path) -> Optional[str]:
    try:
        with path.open("rb") as f:
            h = hashlib.sha256()
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return f"sha256:{h.hexdigest()}"
    except OSError:
        return None


def _train_model(
    X: np.ndarray,
    y: np.ndarray,
    excluded: Optional[Dict[str, int]] = None,
) -> TrainingResult:
    """
    Train an XGBoost regressor on the given dataset.

    Returns a TrainingResult describing success/failure and metrics.
    """
    excluded = dict(excluded or {})
    dataset_hash = _hash_dataset(X, y)

    if xgb is None:
        return TrainingResult(
            ok=False,
            reason="xgboost not installed in venv",
            n_samples=0,
            n_features=0,
            metrics={},
            excluded=excluded,
            dataset_hash=dataset_hash,
        )

    n_samples, n_features = X.shape
    if n_samples < MIN_TRAINING_SAMPLES:
        return TrainingResult(
            ok=False,
            reason=(
                f"Not enough samples for training "
                f"({n_samples} < {MIN_TRAINING_SAMPLES})"
            ),
            n_samples=int(n_samples),
            n_features=int(n_features),
            metrics={},
            excluded=excluded,
            dataset_hash=dataset_hash,
        )

    idx_split = int(0.8 * n_samples)
    if idx_split <= 0 or idx_split >= n_samples:
        return TrainingResult(
            ok=False,
            reason="Invalid train/validation split",
            n_samples=int(n_samples),
            n_features=int(n_features),
            metrics={},
            excluded=excluded,
            dataset_hash=dataset_hash,
        )

    X_train, X_val = X[:idx_split], X[idx_split:]
    y_train, y_val = y[:idx_split], y[idx_split:]

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_NAMES)

    params: Dict[str, Any] = {
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "seed": 42,
    }

    LOG.info("Training XGBoost model on %d samples, %d features", n_samples, n_features)
    evals_result: Dict[str, Any] = {}
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result,
        verbose_eval=False,
    )

    y_pred_prob = bst.predict(dval)
    if len(y_val) == 0:
        accuracy = float("nan")
        logloss = float("nan")
        veto_rate_at_065 = 0.0
    else:
        y_pred_label = (y_pred_prob > 0.5).astype(int)
        accuracy = float(np.mean(y_pred_label == y_val))
        eps = 1e-7
        p = np.clip(y_pred_prob, eps, 1 - eps)
        logloss = float(-np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p)))
        veto_rate_at_065 = float(np.mean(y_pred_prob > 0.65))

    base_loss_rate = float(np.mean(y)) if y.size else 0.0
    metrics = {
        "accuracy": float(accuracy),
        "logloss": float(logloss),
        "n_train": float(idx_split),
        "n_val": float(len(y_val)),
        "base_loss_rate": float(base_loss_rate),
        "val_veto_rate_at_0.65": float(veto_rate_at_065),
    }
    print(f"[xgb_train] accuracy={accuracy:.4f} logloss={logloss:.4f} "
          f"n_train={idx_split} n_val={len(y_val)} "
          f"base_loss_rate={base_loss_rate:.3f} "
          f"veto_rate@0.65={veto_rate_at_065:.3f}")

    # Candidate paths. Production trainer writes to
    # CANDIDATES_DIR/<UTC ts>/ so the tracked baseline at MODEL_PATH /
    # MANIFEST_PATH is never overwritten by the weekly timer. When the
    # module-level MODEL_PATH/MANIFEST_PATH have been monkey-patched
    # (test fixtures), the trainer honors those paths instead — the
    # adapter preserves the existing test surface that drives a real
    # train_model run against a tmp directory.
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    default_model = MODELS_DIR / "xgb_veto_model.json"
    default_manifest = MODELS_DIR / "xgb_veto_manifest.json"
    if MODEL_PATH != default_model or MANIFEST_PATH != default_manifest:
        candidate_dir = MODEL_PATH.parent
        target_model_path = MODEL_PATH
        target_manifest_path = MANIFEST_PATH
    else:
        candidate_dir = CANDIDATES_DIR / ts
        target_model_path = candidate_dir / "xgb_veto_model.json"
        target_manifest_path = candidate_dir / "xgb_veto_manifest.json"

    try:
        candidate_dir.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        LOG.exception("Failed to create candidate dir %s", candidate_dir)
        return TrainingResult(
            ok=False,
            reason="candidate_dir_create_failed",
            n_samples=int(n_samples),
            n_features=int(n_features),
            metrics=metrics,
            excluded=excluded,
            dataset_hash=dataset_hash,
        )

    # prior_model_backup points at the model the candidate would
    # supersede if promoted: runtime-current when present, else the
    # tracked baseline. The trainer no longer copies live model bytes
    # — promotion does that.
    if RUNTIME_MODEL_PATH.exists():
        backup_path: Optional[Path] = RUNTIME_MODEL_PATH
    elif MODEL_PATH.exists():
        backup_path = MODEL_PATH
    else:
        backup_path = None

    LOG.info("Saving XGBoost candidate model to %s", target_model_path)
    try:
        # Stage with a ".json" extension so xgboost picks the JSON
        # serializer (UBJSON is its silent default for unrecognized
        # extensions). os.replace() makes the publish atomic.
        tmp_model = target_model_path.with_name(
            f".write.{target_model_path.name}"
        )
        bst.save_model(str(tmp_model))
        import os as _os
        _os.replace(str(tmp_model), str(target_model_path))
    except Exception:  # noqa: BLE001
        LOG.exception("Failed to save XGBoost model to %s", target_model_path)
        return TrainingResult(
            ok=False,
            reason="model_save_failed",
            n_samples=int(n_samples),
            n_features=int(n_features),
            metrics=metrics,
            excluded=excluded,
            dataset_hash=dataset_hash,
            backup_path=backup_path,
        )

    print(f"[train_xgb_model] candidate written to {candidate_dir}")
    LOG.info("[train_xgb_model] candidate written to %s", candidate_dir)

    model_hash = _hash_file(target_model_path) or ""
    trained_at = datetime.now(timezone.utc).isoformat()
    # Stable model_version — encodes the candidate UTC timestamp so the
    # promoted model retains the identity of the training run that
    # produced it.
    model_version = f"xgb_veto_{ts}"

    # SS08 + Batch 9: manifest is the source-of-truth for the predictor's
    # safety gates. It records the dataset fingerprint, the per-reason
    # exclusion counts, validation metrics, and the rollback target so
    # operators can restore by copying the backup over the runtime
    # current model. The manifest lands next to the candidate model.
    manifest_written = False
    try:
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "model_path": str(target_model_path),
            "model_sha256": model_hash,
            "model_version": model_version,
            "trained_at_utc": trained_at,
            "training_samples": int(n_samples),
            "feature_names": list(FEATURE_NAMES),
            "metrics": {k: float(v) for k, v in metrics.items()},
            "validation_accuracy": float(metrics.get("accuracy", 0.0)),
            "validation_logloss": float(metrics.get("logloss", 0.0)),
            "dataset_hash": dataset_hash,
            "excluded": {k: int(v) for k, v in excluded.items()},
            "min_training_samples": int(MIN_TRAINING_SAMPLES),
            "prior_model_backup": str(backup_path) if backup_path else None,
            "candidate_id": ts,
            "candidate_dir": str(candidate_dir),
            "label_definition": "1=loss(net_pnl<=0), 0=win(net_pnl>0)",
            "pnl_source_priority": [
                "payload.pnl_breakdown.net_pnl",
                "payload.net_pnl",
                "payload.pnl",
            ],
            "training_filters": {
                "exclude_quarantined": True,
                "exclude_pnl_untrusted": True,
                "exclude_untrusted_fill_ids": True,
                "exclude_nonfinite_pnl": True,
                "usable_schemas": list(USABLE_TRADE_SCHEMAS)
                + sorted(_legacy_schemas_allowed()),
            },
        }
        import os as _os
        tmp_manifest = target_manifest_path.with_suffix(
            target_manifest_path.suffix + ".tmp"
        )
        tmp_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        _os.replace(str(tmp_manifest), str(target_manifest_path))
        manifest_written = True
        LOG.info("Manifest written to %s", target_manifest_path)
        print(f"Manifest written to {target_manifest_path}")
    except Exception:  # noqa: BLE001
        LOG.exception("Failed to write XGBoost manifest — continuing")

    LOG.info("Writing performance metrics to %s", PERF_PATH)
    try:
        PERF_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        LOG.exception("Failed to write performance metrics to %s", PERF_PATH)
        return TrainingResult(
            ok=True,
            reason="metrics_write_failed",
            n_samples=int(n_samples),
            n_features=int(n_features),
            metrics=metrics,
            excluded=excluded,
            dataset_hash=dataset_hash,
            model_version=model_version,
            manifest_path=target_manifest_path if manifest_written else None,
            backup_path=backup_path,
        )

    return TrainingResult(
        ok=True,
        reason="training_completed",
        n_samples=int(n_samples),
        n_features=int(n_features),
        metrics=metrics,
        excluded=excluded,
        dataset_hash=dataset_hash,
        model_version=model_version,
        manifest_path=target_manifest_path if manifest_written else None,
        backup_path=backup_path,
    )


def _load_exclusion_sets() -> Tuple[Set[str], Set[str], bool]:
    """Load the canonical (invalid_fill_ids, invalid_trade_hashes) sets
    used by every CHAD publisher, plus whether the load actually SUCCEEDED.

    W6B-7: the third element is new, and it matters. This function has always
    degraded to empty sets on failure and logged a WARNING — its own docstring
    admitted "the manifest will then record zero exclusions for all reasons".
    But a training run with empty exclusion sets is not a slightly-degraded
    run: it is a run that CANNOT TELL untrusted rows from trusted ones, and
    would happily train a live-money veto model on the whole contaminated
    book (2153 untrusted rows at time of writing) while reporting zero
    exclusions.

    That silently inverts the very principle D3 rests on — "refusing to train
    on untrusted rows is the system honouring its own rules". So the caller now
    treats a failed load as a DECLINE rather than proceeding, and the outcome
    is stamped into xgb_train_state.v1 as `exclusions_loaded`.

    Degrading to empty sets is kept (rather than raising) so the caller decides
    the policy in one place.
    """
    try:
        from chad.utils.quarantine import get_exclusion_sets
    except Exception:  # noqa: BLE001
        LOG.warning(
            "quarantine helper unavailable — cannot distinguish untrusted rows"
        )
        return set(), set(), False
    try:
        fill_ids, trade_hashes = get_exclusion_sets(
            runtime_dir=ROOT / "runtime",
            fills_dir=ROOT / "data" / "fills",
        )
        return fill_ids, trade_hashes, True
    except Exception as exc:  # noqa: BLE001
        LOG.warning("get_exclusion_sets failed err=%s — cannot filter the book", exc)
        return set(), set(), False


def main(argv: List[str] | None = None) -> int:
    """
    Entry point for the XGBoost training job.

    Returns 0 on:
    - no trade data,
    - no usable samples,
    - insufficient TRUSTED samples (a decline, not a failure — W6B-7/D3),
    - training completed successfully.

    Returns 1 only on hard training/model errors (xgboost missing, etc.).

    Every outcome publishes runtime/xgb_train_state.json (xgb_train_state.v1)
    carrying outcome/reason/usable_rows/required_rows/rows_short, so a decline
    is visible and its countdown to trainability is watchable.
    """
    _setup_logging()
    LOG.info("Starting XGB training job (trade history)")

    rows = _load_trade_rows()
    if not rows:
        LOG.warning("No trade rows loaded; DECLINED (not an error)")
        _write_train_state(
            outcome="declined", reason="No trade rows available", usable_rows=0,
        )
        return 0

    invalid_fill_ids, invalid_trade_hashes, exclusions_loaded = _load_exclusion_sets()
    if not exclusions_loaded:
        # W6B-7: fail closed. Without the exclusion sets we cannot tell trusted
        # rows from untrusted ones, so ANY training here would be training on
        # the unfiltered book — the exact outcome the sample threshold exists
        # to prevent. Decline (exit 0, loudly) rather than proceed.
        LOG.warning(
            "XGB_TRAIN_DECLINED reason='exclusion sets unavailable' — cannot "
            "distinguish untrusted rows, so training would use the unfiltered "
            "book. Exiting 0. See runtime/xgb_train_state.json."
        )
        _write_train_state(
            outcome="declined",
            reason="Exclusion sets unavailable — cannot filter untrusted rows",
            usable_rows=0,
            exclusions_loaded=False,
        )
        return 0

    if invalid_fill_ids or invalid_trade_hashes:
        LOG.info(
            "Training will exclude %d untrusted fill_ids and %d quarantined trade hashes",
            len(invalid_fill_ids),
            len(invalid_trade_hashes),
        )

    X, y, excluded = _build_dataset(rows, invalid_fill_ids, invalid_trade_hashes)
    LOG.info("Training set: %d usable rows, exclusions=%s", len(y), excluded)
    if X.size == 0 or y.size == 0:
        LOG.warning("No usable samples after feature building; DECLINED (not an error)")
        _write_train_state(
            outcome="declined",
            reason="No usable samples after feature building",
            usable_rows=0,
            excluded=excluded,
        )
        return 0

    result = _train_model(X, y, excluded)
    LOG.info(
        "Training result: ok=%s reason=%s metrics=%s excluded=%s "
        "dataset_hash=%s model_version=%s",
        result.ok,
        result.reason,
        result.metrics,
        result.excluded,
        result.dataset_hash,
        result.model_version,
    )

    # W6B-7 (D3): declined (data quality) vs error (genuine failure).
    if result.ok:
        _write_train_state(
            outcome="trained", reason=result.reason or "ok",
            usable_rows=result.n_samples, excluded=result.excluded,
            model_version=result.model_version, dataset_hash=result.dataset_hash,
        )
        return 0

    if _is_decline(result.reason):
        short = max(0, MIN_TRAINING_SAMPLES - int(result.n_samples))
        LOG.warning(
            "XGB_TRAIN_DECLINED reason=%r usable_rows=%d required_rows=%d "
            "rows_short=%d exclusions=%s — exiting 0. The refusal is the system "
            "honouring its own trust rules; MIN_TRAINING_SAMPLES is a trust "
            "floor, not a tuning knob. See runtime/xgb_train_state.json.",
            result.reason, int(result.n_samples), MIN_TRAINING_SAMPLES,
            short, result.excluded,
        )
        _write_train_state(
            outcome="declined", reason=result.reason,
            usable_rows=result.n_samples, excluded=result.excluded,
            dataset_hash=result.dataset_hash,
        )
        return 0

    LOG.error("XGB_TRAIN_ERROR reason=%r — exiting 1", result.reason)
    _write_train_state(
        outcome="error", reason=result.reason,
        usable_rows=result.n_samples, excluded=result.excluded,
        dataset_hash=result.dataset_hash,
    )
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
