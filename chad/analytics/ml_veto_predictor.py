"""
CHAD ML Veto Predictor
Loads trained XGBoost model and scores trade intents before submission.
Returns (should_veto, confidence) — False/0.0 on any failure.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("chad.ml_veto")

REPO_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = REPO_ROOT / "shared" / "models" / "xgb_veto_model.json"
VETO_THRESHOLD = float(os.environ.get("CHAD_ML_VETO_THRESHOLD", "0.65"))

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

_model = None
_model_load_attempted = False


def _load_model():
    global _model, _model_load_attempted
    if _model_load_attempted:
        return _model
    _model_load_attempted = True
    try:
        import xgboost as xgb
        if not MODEL_PATH.exists():
            logger.info("ml_veto: model not found at %s — veto disabled",
                        MODEL_PATH)
            return None
        bst = xgb.Booster()
        bst.load_model(str(MODEL_PATH))
        logger.info("ml_veto: model loaded from %s", MODEL_PATH)
        _model = bst
        return _model
    except Exception as e:
        logger.warning("ml_veto: model load failed err=%s — veto disabled", e)
        return None


def extract_features(intent: Any, ctx: Dict) -> Optional[list]:
    """Extract feature vector from intent + context."""
    try:
        strategy = str(getattr(intent, "strategy", "") or "").lower()
        side = str(getattr(intent, "side", "") or "").upper()

        regime_data = ctx.get("regime", {}) or {}
        regime = str(regime_data.get("regime", "unknown")).lower()

        prices = ctx.get("prices", {}) or {}
        vix = float(prices.get("VIX", prices.get("^VIX", 20.0)) or 20.0)

        scr = ctx.get("scr", {}) or {}
        sizing = float(scr.get("sizing_factor", 0.25) or 0.25)

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        health = ctx.get("strategy_health", {}) or {}
        strat_health = health.get(strategy, {}) or {}
        win_rate = float(strat_health.get("win_rate",
                        strat_health.get("health_score", 0.5)) or 0.5)

        portfolio = ctx.get("portfolio", {}) or {}
        equity = float(portfolio.get("total_equity",
                       portfolio.get("net_liquidation", 200000)) or 200000)

        return [
            float(_STRATEGY_MAP.get(strategy, 15)),
            float(_REGIME_MAP.get(regime, 4)),
            vix,
            float(now.hour),
            float(now.weekday()),
            sizing,
            1.0 if "BUY" in side else 0.0,
            equity / 200000.0,
            win_rate,
            float(regime_data.get("confidence", 0.5) or 0.5),
        ]
    except Exception as e:
        logger.debug("ml_veto: feature_extraction_failed err=%s", e)
        return None


def should_veto(intent: Any, ctx: Dict) -> Tuple[bool, float]:
    """
    Returns (veto: bool, loss_probability: float).
    Returns (False, 0.0) on any failure — never blocks due to model error.
    """
    try:
        model = _load_model()
        if model is None:
            return False, 0.0

        features = extract_features(intent, ctx)
        if features is None:
            return False, 0.0

        import xgboost as xgb
        import numpy as np
        dmatrix = xgb.DMatrix(np.array([features]), feature_names=FEATURE_NAMES)
        loss_prob = float(model.predict(dmatrix)[0])

        veto = loss_prob > VETO_THRESHOLD
        if veto:
            logger.info(
                "ml_veto: VETOED symbol=%s strategy=%s "
                "loss_prob=%.3f threshold=%.2f",
                getattr(intent, "symbol", "?"),
                getattr(intent, "strategy", "?"),
                loss_prob, VETO_THRESHOLD,
            )
        return veto, loss_prob
    except Exception as e:
        logger.debug("ml_veto: prediction_failed err=%s — not vetoing", e)
        return False, 0.0
