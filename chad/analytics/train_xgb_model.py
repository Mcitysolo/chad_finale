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

import json
import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import xgboost as xgb  # type: ignore[import]
except Exception:  # noqa: BLE001
    xgb = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "trades"
MODELS_DIR = ROOT / "shared" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "xgb_veto_model.json"
PERF_PATH = MODELS_DIR / "model_performance.json"

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


def _build_dataset(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the 10-feature matrix used by the ML veto predictor.
    Label: 1 = loss (pnl <= 0), 0 = win (pnl > 0). We predict P(loss).

    Trade history rows may be flat or wrapped in a {"payload": ...} envelope.
    Most historical fields are absent; we substitute neutral defaults that
    match the runtime predictor's defaults so train and inference align.
    """
    from datetime import datetime

    X: List[List[float]] = []
    y: List[int] = []

    for raw in rows:
        r = raw.get("payload", raw) if isinstance(raw, dict) else raw
        if not isinstance(r, dict):
            continue
        pnl = r.get("pnl")
        if pnl is None:
            continue
        try:
            pnl_f = float(pnl)
        except Exception:
            continue

        strategy = str(r.get("strategy", "") or "").lower()
        side = str(r.get("side", "") or "").upper()
        regime = str(r.get("regime", "") or "unknown").lower()
        if regime not in _REGIME_MAP:
            regime = "unknown"

        # Time features from exit_time_utc when available
        hour = 12.0
        weekday = 2.0
        ts = r.get("exit_time_utc") or r.get("entry_time_utc") or r.get("timestamp_utc")
        if isinstance(ts, str) and ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                hour = float(dt.hour)
                weekday = float(dt.weekday())
            except Exception:
                pass

        # Defaults for fields not stored in trade history — match predictor
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
        return np.empty((0, 0), dtype=float), np.empty((0,), dtype=int)

    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def _train_model(X: np.ndarray, y: np.ndarray) -> TrainingResult:
    """
    Train an XGBoost regressor on the given dataset.

    Returns a TrainingResult describing success/failure and metrics.
    """
    if xgb is None:
        return TrainingResult(
            ok=False,
            reason="xgboost not installed in venv",
            n_samples=0,
            n_features=0,
            metrics={},
        )

    n_samples, n_features = X.shape
    if n_samples < 100:
        return TrainingResult(
            ok=False,
            reason=f"Not enough samples for training ({n_samples} < 100)",
            n_samples=int(n_samples),
            n_features=int(n_features),
            metrics={},
        )

    idx_split = int(0.8 * n_samples)
    if idx_split <= 0 or idx_split >= n_samples:
        return TrainingResult(
            ok=False,
            reason="Invalid train/validation split",
            n_samples=int(n_samples),
            n_features=int(n_features),
            metrics={},
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

    # SS08: snapshot the prior model with a UTC timestamp before overwriting
    # so each retrain is recoverable. Filename collisions are avoided because
    # the timestamp resolves to the second.
    backup_path: Optional[Path] = None
    if MODEL_PATH.exists():
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = MODEL_PATH.with_name(f"xgb_veto_model_{ts}.json")
        try:
            shutil.copy2(MODEL_PATH, backup_path)
            LOG.info("Backed up prior model to %s", backup_path)
            print(f"Backed up prior model to {backup_path}")
        except Exception:  # noqa: BLE001
            LOG.exception("Failed to back up prior model — continuing")
            backup_path = None

    LOG.info("Saving XGBoost model to %s", MODEL_PATH)
    try:
        bst.save_model(str(MODEL_PATH))
    except Exception:  # noqa: BLE001
        LOG.exception("Failed to save XGBoost model to %s", MODEL_PATH)
        return TrainingResult(
            ok=False,
            reason="model_save_failed",
            n_samples=int(n_samples),
            n_features=int(n_features),
            metrics=metrics,
        )

    # SS08: write a manifest beside the model recording training metadata,
    # the backup path of the prior model (rollback target), and the feature
    # set the model was trained on. Operators can restore by copying the
    # backup over MODEL_PATH.
    try:
        manifest = {
            "model_path": str(MODEL_PATH),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_samples": int(n_samples),
            "validation_accuracy": float(metrics.get("accuracy", 0.0)),
            "validation_logloss": float(metrics.get("logloss", 0.0)),
            "feature_names": list(FEATURE_NAMES),
            "prior_model_backup": str(backup_path) if backup_path else None,
            "schema_version": "xgb_manifest.v1",
        }
        manifest_path = MODEL_PATH.parent / "xgb_veto_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        LOG.info("Manifest written to %s", manifest_path)
        print(f"Manifest written to {manifest_path}")
    except Exception:  # noqa: BLE001
        LOG.exception("Failed to write XGBoost manifest — continuing")

    LOG.info("Writing performance metrics to %s", PERF_PATH)
    try:
        PERF_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        LOG.exception("Failed to write performance metrics to %s", PERF_PATH)
        # Model is still usable; treat as soft failure
        return TrainingResult(
            ok=True,
            reason="metrics_write_failed",
            n_samples=int(n_samples),
            n_features=int(n_features),
            metrics=metrics,
        )

    return TrainingResult(
        ok=True,
        reason="training_completed",
        n_samples=int(n_samples),
        n_features=int(n_features),
        metrics=metrics,
    )


def main(argv: List[str] | None = None) -> int:
    """
    Entry point for the XGBoost training job.

    Returns 0 on:
    - no trade data,
    - no usable samples,
    - training completed successfully.

    Returns 1 only on hard training/model errors (xgboost missing, etc.).
    """
    _setup_logging()
    LOG.info("Starting XGB training job (trade history)")

    rows = _load_trade_rows()
    if not rows:
        LOG.warning("No trade rows loaded; aborting XGB training")
        return 0

    X, y = _build_dataset(rows)
    if X.size == 0 or y.size == 0:
        LOG.warning("No usable samples after feature building; aborting")
        return 0

    result = _train_model(X, y)
    LOG.info(
        "Training result: ok=%s reason=%s metrics=%s",
        result.ok,
        result.reason,
        result.metrics,
    )

    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
