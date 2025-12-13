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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import xgboost as xgb  # type: ignore[import]
except Exception:  # noqa: BLE001
    xgb = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "trades"
MODELS_DIR = ROOT / "shared" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "xgb_model.json"
PERF_PATH = MODELS_DIR / "model_performance.json"

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


def _load_trade_rows(max_files: int = 30) -> List[Dict[str, Any]]:
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
    Build a simple feature matrix from raw trade rows.

    Expected keys (adjust to your schema as needed):
    - 'pnl'                     : realized PnL for the trade (float)
    - 'holding_period_minutes'  : optional, time in minutes
    - 'side'                    : "BUY" or "SELL"
    - 'volatility'              : optional float

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector (PnL).
    """
    X: List[List[float]] = []
    y: List[float] = []

    for r in rows:
        pnl = r.get("pnl")
        if pnl is None:
            continue

        try:
            hp = float(r.get("holding_period_minutes", 0.0) or 0.0)
        except Exception:
            hp = 0.0

        side_raw = str(r.get("side", "")).upper()
        is_long = 1.0 if side_raw == "BUY" else 0.0

        try:
            vol = float(r.get("volatility", 0.0) or 0.0)
        except Exception:
            vol = 0.0

        X.append([hp, is_long, vol])
        y.append(float(pnl))

    if not X:
        return np.empty((0, 0), dtype=float), np.empty((0,), dtype=float)

    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


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

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params: Dict[str, Any] = {
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
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

    y_pred = bst.predict(dval)
    if len(y_val) == 0:
        rmse = float("nan")
        mae = float("nan")
    else:
        rmse = float(np.sqrt(np.mean((y_pred - y_val) ** 2)))
        mae = float(np.mean(np.abs(y_pred - y_val)))

    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "n_train": float(idx_split),
        "n_val": float(len(y_val)),
    }

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
