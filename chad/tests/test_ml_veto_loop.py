"""ML veto loop safety tests (Batch 9).

Pin the seven invariants Batch 9 must hold even before enforcement
is enabled:

  1. Training drops quarantined and untrusted trades from labels.
  2. Training manifest carries every audit-required field.
  3. Shadow log decision schema includes model_version, manifest_hash,
     features, prediction, threshold, would_veto, final_action,
     reason, and intent_class.
  4. Enforcement is OFF by default (CHAD_ML_VETO_ENABLED unset).
  5. Predictor never enforces against exit / reduction / hedge /
     liquidation intents — even with enforcement on and a positive
     model prediction.
  6. Predictor fails open when manifest is missing or stale —
     enforcement collapses to ``pass`` instead of vetoing.
  7. Canary strategy allowlist gates enforcement: enforcement on +
     canary unset / mismatched → no veto.

These tests do NOT enable live ML veto enforcement. They merely
verify that the safety gates behave correctly when toggled per-test.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping
from unittest import mock

import pytest

from chad.analytics import ml_veto_predictor as predictor
from chad.analytics import train_xgb_model as trainer


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _trade_row(
    *,
    strategy: str = "alpha",
    side: str = "BUY",
    pnl: float = 100.0,
    schema_version: str = "closed_trade.v1",
    pnl_breakdown: Mapping[str, float] | None = None,
    pnl_untrusted: bool = False,
    quarantined_record_hash: str | None = None,
    fill_ids: list[str] | None = None,
    extra: Mapping[str, Any] | None = None,
    record_hash: str | None = None,
) -> Dict[str, Any]:
    """Build a single trade-history row matching closed_trade.v1
    shape so trainer dataset filters can be exercised."""
    payload: Dict[str, Any] = {
        "schema_version": schema_version,
        "strategy": strategy,
        "side": side,
        "symbol": "SPY",
        "pnl": float(pnl),
        "net_pnl": float(pnl),
        "fill_ids": fill_ids or ["fillA", "fillB"],
        "exit_time_utc": "2026-05-01T15:00:00+00:00",
    }
    if pnl_breakdown is not None:
        payload["pnl_breakdown"] = dict(pnl_breakdown)
    if pnl_untrusted:
        payload["pnl_untrusted"] = True
    if extra is not None:
        payload["extra"] = dict(extra)
    return {
        "record_hash": record_hash or quarantined_record_hash or "rh_clean_001",
        "payload": payload,
    }


def _intent(
    *,
    strategy: str = "alpha",
    symbol: str = "SPY",
    side: str = "BUY",
    tags: list[str] | None = None,
    meta: Dict[str, Any] | None = None,
    action: str | None = None,
) -> SimpleNamespace:
    ns_kwargs: Dict[str, Any] = {
        "strategy": strategy,
        "symbol": symbol,
        "side": side,
        "tags": tags or [],
        "meta": meta or {},
    }
    if action is not None:
        ns_kwargs["action"] = action
    return SimpleNamespace(**ns_kwargs)


def _write_manifest(
    tmp_path: Path,
    *,
    samples: int = 500,
    accuracy: float = 0.6,
    trained_days_ago: float = 1.0,
    model_version: str = "xgb_veto_test_v1",
) -> Path:
    """Write a Batch-9 manifest to ``tmp_path`` with the operator-
    visible fields filled in. Returns the manifest path."""
    trained_at = (
        datetime.now(timezone.utc) - timedelta(days=trained_days_ago)
    ).isoformat()
    manifest = {
        "schema_version": trainer.MANIFEST_SCHEMA_VERSION,
        "model_path": str(tmp_path / "xgb_veto_model.json"),
        "model_sha256": "sha256:" + ("a" * 64),
        "model_version": model_version,
        "trained_at_utc": trained_at,
        "training_samples": samples,
        "feature_names": list(trainer.FEATURE_NAMES),
        "metrics": {
            "accuracy": accuracy,
            "logloss": 0.65,
            "n_train": samples * 0.8,
            "n_val": samples * 0.2,
            "base_loss_rate": 0.5,
        },
        "validation_accuracy": accuracy,
        "validation_logloss": 0.65,
        "dataset_hash": "sha256:" + ("b" * 64),
        "excluded": {"quarantined": 1, "untrusted": 2, "untrusted_fill_id": 3},
        "min_training_samples": trainer.MIN_TRAINING_SAMPLES,
        "prior_model_backup": None,
        "label_definition": "1=loss(net_pnl<=0), 0=win(net_pnl>0)",
        "pnl_source_priority": ["payload.pnl_breakdown.net_pnl"],
        "training_filters": {
            "exclude_quarantined": True,
            "exclude_pnl_untrusted": True,
            "exclude_untrusted_fill_ids": True,
            "exclude_nonfinite_pnl": True,
            "usable_schemas": list(trainer.USABLE_TRADE_SCHEMAS),
        },
    }
    path = tmp_path / "xgb_veto_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


@pytest.fixture
def isolated_predictor(tmp_path, monkeypatch):
    """Point the predictor at a fresh tmp_path manifest/model and
    reset its caches. Yields ``tmp_path``. Disables enforcement by
    default — individual tests opt in via monkeypatch.setenv."""
    monkeypatch.setattr(predictor, "MODEL_PATH", tmp_path / "xgb_veto_model.json")
    monkeypatch.setattr(predictor, "MANIFEST_PATH", tmp_path / "xgb_veto_manifest.json")
    monkeypatch.delenv("CHAD_ML_VETO_ENABLED", raising=False)
    monkeypatch.delenv("CHAD_ML_VETO_CANARY_STRATEGIES", raising=False)
    predictor.reset_manifest_cache()
    yield tmp_path
    predictor.reset_manifest_cache()


# ---------------------------------------------------------------------------
# 1. Training excludes untrusted/quarantined
# ---------------------------------------------------------------------------


def test_ml_training_excludes_untrusted_and_quarantined_trades():
    """Quarantined / pnl_untrusted / untrusted-fill_id / nonfinite /
    legacy-schema rows must not contribute training labels."""
    rows = [
        # 5 clean rows.
        _trade_row(record_hash="rh_clean_1", pnl=100, side="BUY",
                   fill_ids=["clean_a", "clean_b"]),
        _trade_row(record_hash="rh_clean_2", pnl=-50, side="SELL",
                   fill_ids=["clean_c", "clean_d"]),
        _trade_row(record_hash="rh_clean_3", pnl=25, side="BUY",
                   fill_ids=["clean_e", "clean_f"]),
        _trade_row(record_hash="rh_clean_4", pnl=-30, side="SELL",
                   fill_ids=["clean_g", "clean_h"]),
        _trade_row(record_hash="rh_clean_5", pnl=10, side="BUY",
                   fill_ids=["clean_i", "clean_j"]),
        # quarantined by record_hash.
        _trade_row(record_hash="rh_bad_quar", pnl=999),
        # explicit pnl_untrusted on payload.
        _trade_row(record_hash="rh_bad_untrusted_payload",
                   pnl=10, pnl_untrusted=True),
        # extra.pnl_untrusted=True.
        _trade_row(record_hash="rh_bad_untrusted_extra",
                   pnl=10, extra={"pnl_untrusted": True}),
        # fill_id intersects untrusted set.
        _trade_row(record_hash="rh_bad_fill",
                   pnl=10, fill_ids=["clean_x", "tainted_fill"]),
        # NaN PnL.
        _trade_row(record_hash="rh_bad_nan", pnl=float("nan")),
        # Inf PnL.
        _trade_row(record_hash="rh_bad_inf", pnl=float("inf")),
        # Legacy schema not in USABLE_TRADE_SCHEMAS.
        _trade_row(record_hash="rh_bad_legacy",
                   pnl=10, schema_version="legacy_v0"),
    ]
    bad_fills = {"tainted_fill"}
    bad_hashes = {"rh_bad_quar"}

    X, y, excluded = trainer._build_dataset(rows, bad_fills, bad_hashes)

    # Five clean rows survive.
    assert X.shape == (5, len(trainer.FEATURE_NAMES))
    assert len(y) == 5

    # Each exclusion bucket recorded exactly once.
    assert excluded["quarantined"] == 1
    assert excluded["untrusted"] == 2  # payload + extra
    assert excluded["untrusted_fill_id"] == 1
    assert excluded["nonfinite_pnl"] == 2  # NaN + Inf
    assert excluded["legacy_schema"] == 1

    # Labels: pnl>0 → 0 (win), pnl<=0 → 1 (loss).
    # Order of clean rows is preserved.
    assert list(y) == [0, 1, 0, 1, 0]


def test_ml_training_prefers_pnl_breakdown_net_pnl():
    """When pnl_breakdown.net_pnl is present and finite it wins over
    the legacy ``pnl`` field. Ensures Batch-9 per-trade breakdowns
    are consumed as the canonical source."""
    # pnl=100 (would be a win) but breakdown.net_pnl=-100 (loss).
    row = _trade_row(
        record_hash="rh_priority",
        pnl=100,
        pnl_breakdown={"net_pnl": -100, "gross_pnl": -90, "fees": -10},
    )
    X, y, _ = trainer._build_dataset([row], set(), set())
    assert len(y) == 1
    assert int(y[0]) == 1  # loss


# ---------------------------------------------------------------------------
# 2. Manifest required fields
# ---------------------------------------------------------------------------


def test_ml_training_manifest_contains_required_fields(tmp_path, monkeypatch):
    """Manifest must carry every field a downstream auditor needs to
    reproduce the model: dataset hash, sample count, feature list,
    metrics, training timestamp, model path, backup path, exclusion
    counts, training filters, schema version."""
    fake_model_path = tmp_path / "xgb_veto_model.json"
    fake_manifest_path = tmp_path / "xgb_veto_manifest.json"
    fake_perf_path = tmp_path / "model_performance.json"

    monkeypatch.setattr(trainer, "MODEL_PATH", fake_model_path)
    monkeypatch.setattr(trainer, "MANIFEST_PATH", fake_manifest_path)
    monkeypatch.setattr(trainer, "PERF_PATH", fake_perf_path)

    pytest.importorskip("xgboost")

    # Build a small but valid dataset (>= MIN_TRAINING_SAMPLES).
    rows = []
    for i in range(150):
        rows.append(
            _trade_row(
                record_hash=f"rh_{i}",
                pnl=10 if i % 2 == 0 else -10,
                strategy="alpha",
                side="BUY" if i % 3 == 0 else "SELL",
                fill_ids=[f"f_{i}_a", f"f_{i}_b"],
            )
        )
    X, y, excluded = trainer._build_dataset(rows, set(), set())
    result = trainer._train_model(X, y, excluded)

    assert result.ok, f"train failed: {result.reason}"
    assert fake_manifest_path.exists()
    manifest = json.loads(fake_manifest_path.read_text())

    required_fields = [
        "schema_version",
        "model_path",
        "model_sha256",
        "model_version",
        "trained_at_utc",
        "training_samples",
        "feature_names",
        "metrics",
        "dataset_hash",
        "excluded",
        "training_filters",
        "min_training_samples",
        "prior_model_backup",
    ]
    missing = [f for f in required_fields if f not in manifest]
    assert not missing, f"manifest missing fields: {missing}"

    # Field semantics.
    assert manifest["schema_version"] == trainer.MANIFEST_SCHEMA_VERSION
    assert manifest["training_samples"] == 150
    assert manifest["feature_names"] == list(trainer.FEATURE_NAMES)
    assert manifest["dataset_hash"].startswith("sha256:")
    assert manifest["model_sha256"].startswith("sha256:")
    assert "accuracy" in manifest["metrics"]
    assert manifest["training_filters"]["exclude_quarantined"] is True
    # min_training_samples must echo the trainer constant so an
    # operator can spot a manifest produced with a relaxed gate.
    assert manifest["min_training_samples"] == trainer.MIN_TRAINING_SAMPLES


# ---------------------------------------------------------------------------
# 3. Shadow log schema
# ---------------------------------------------------------------------------


def test_ml_veto_shadow_log_contains_model_version_features_prediction_threshold(
    isolated_predictor, monkeypatch
):
    """ShadowDecision must surface model_version, manifest_hash,
    features used, prediction, threshold, would_veto, final_action,
    reason, and intent_class — the contract Batch 9 audit pipelines
    consume."""
    _write_manifest(isolated_predictor)
    predictor.reset_manifest_cache()

    decision = predictor.score_intent(_intent(), {"regime": {"regime": "ranging"}})
    d = decision.to_dict()

    for field_name in (
        "model_version",
        "manifest_hash",
        "manifest_path",
        "threshold",
        "features",
        "prediction",
        "would_veto",
        "final_action",
        "reason",
        "intent_class",
        "enforcement_enabled",
        "manifest_valid",
        "manifest_stale",
    ):
        assert field_name in d, f"missing {field_name}"

    # Features must be the documented 10 named features.
    assert set(d["features"].keys()) == set(predictor.FEATURE_NAMES)
    assert isinstance(d["prediction"], float)
    assert d["threshold"] == predictor.VETO_THRESHOLD
    assert d["model_version"] == "xgb_veto_test_v1"
    assert d["intent_class"] == predictor.INTENT_ENTRY


# ---------------------------------------------------------------------------
# 4. Enforcement disabled by default
# ---------------------------------------------------------------------------


def test_ml_veto_enforcement_defaults_disabled(isolated_predictor, monkeypatch):
    """With CHAD_ML_VETO_ENABLED unset, every entry intent must
    resolve to final_action=shadow_only regardless of model output."""
    _write_manifest(isolated_predictor)
    monkeypatch.delenv("CHAD_ML_VETO_ENABLED", raising=False)
    predictor.reset_manifest_cache()

    # Force a "would_veto" by patching the model layer to emit a high
    # loss probability — even then the final action stays shadow_only.
    with mock.patch.object(predictor, "_load_model") as load_model, \
         mock.patch("xgboost.DMatrix"), \
         mock.patch.object(predictor, "_features_to_vector",
                           return_value=[0.0] * len(predictor.FEATURE_NAMES)):
        fake_booster = mock.Mock()
        fake_booster.predict.return_value = [0.99]
        load_model.return_value = fake_booster
        decision = predictor.score_intent(_intent())

    assert decision.would_veto is True
    assert decision.final_action == predictor.ACTION_SHADOW_ONLY
    assert decision.enforcement_enabled is False


def test_ml_veto_enforcement_enabled_flag_truthy_values(monkeypatch):
    """``enforcement_enabled`` must accept the canonical truthy
    values and treat everything else as off — including the empty
    string and unset variable."""
    for v in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("CHAD_ML_VETO_ENABLED", v)
        assert predictor.enforcement_enabled() is True
    for v in ("", "0", "false", "off", "no"):
        monkeypatch.setenv("CHAD_ML_VETO_ENABLED", v)
        assert predictor.enforcement_enabled() is False
    monkeypatch.delenv("CHAD_ML_VETO_ENABLED", raising=False)
    assert predictor.enforcement_enabled() is False


# ---------------------------------------------------------------------------
# 5. Never blocks exit / reduction / hedge / liquidation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "intent_kwargs,expected_class",
    [
        ({"side": "EXIT"}, predictor.INTENT_EXIT),
        ({"side": "CLOSE"}, predictor.INTENT_EXIT),
        ({"tags": ["exit"]}, predictor.INTENT_EXIT),
        ({"tags": ["close"]}, predictor.INTENT_EXIT),
        ({"tags": ["liquidation"]}, predictor.INTENT_LIQUIDATION),
        ({"tags": ["reduce"]}, predictor.INTENT_REDUCTION),
        ({"tags": ["hedge"]}, predictor.INTENT_HEDGE),
        ({"meta": {"reason": "max_hold_exit"}}, predictor.INTENT_EXIT),
        ({"meta": {"reason": "stop_loss"}}, predictor.INTENT_EXIT),
        ({"meta": {"reduce_only": True}}, predictor.INTENT_REDUCTION),
        ({"meta": {"is_hedge": True}}, predictor.INTENT_HEDGE),
        ({"meta": {"is_exit": True}}, predictor.INTENT_EXIT),
    ],
)
def test_ml_veto_never_blocks_exit_reduction_hedge_liquidation(
    isolated_predictor, monkeypatch, intent_kwargs, expected_class
):
    """Even with enforcement on, a permissive canary, a strong model
    veto signal, and a fresh manifest, a protective intent must
    resolve to final_action=pass — never veto."""
    _write_manifest(isolated_predictor)
    monkeypatch.setenv("CHAD_ML_VETO_ENABLED", "1")
    monkeypatch.setenv("CHAD_ML_VETO_CANARY_STRATEGIES", "alpha")
    predictor.reset_manifest_cache()

    intent = _intent(strategy="alpha", **intent_kwargs)
    assert predictor.classify_intent(intent) == expected_class

    with mock.patch.object(predictor, "_load_model") as load_model, \
         mock.patch("xgboost.DMatrix"):
        fake_booster = mock.Mock()
        fake_booster.predict.return_value = [0.95]  # well above 0.65
        load_model.return_value = fake_booster
        decision = predictor.score_intent(intent)

    assert decision.would_veto is True, "model should still produce its opinion"
    assert decision.final_action == predictor.ACTION_PASS, decision
    assert decision.reason == predictor.REASON_PROTECTIVE
    assert decision.intent_class == expected_class


def test_ml_veto_flip_intent_via_ctx_marked_protective(
    isolated_predictor, monkeypatch
):
    """When the live loop pre-detects a flip and passes
    ``ctx['is_flip']``, the predictor must classify it as a
    protective exit even when no tags are set."""
    _write_manifest(isolated_predictor)
    monkeypatch.setenv("CHAD_ML_VETO_ENABLED", "1")
    monkeypatch.setenv("CHAD_ML_VETO_CANARY_STRATEGIES", "alpha")
    predictor.reset_manifest_cache()

    with mock.patch.object(predictor, "_load_model") as load_model, \
         mock.patch("xgboost.DMatrix"):
        fake_booster = mock.Mock()
        fake_booster.predict.return_value = [0.95]
        load_model.return_value = fake_booster
        decision = predictor.score_intent(_intent(), {"is_flip": True})

    assert decision.intent_class == predictor.INTENT_EXIT
    assert decision.final_action == predictor.ACTION_PASS


# ---------------------------------------------------------------------------
# 6. Fails open on missing / stale / invalid manifest
# ---------------------------------------------------------------------------


def test_ml_veto_fails_open_when_manifest_missing_or_stale(
    isolated_predictor, monkeypatch
):
    """No manifest, stale manifest, or under-sampled manifest →
    final_action = pass with a stable reason. Enforcement never
    activates against a model the predictor cannot defend."""
    monkeypatch.setenv("CHAD_ML_VETO_ENABLED", "1")
    monkeypatch.setenv("CHAD_ML_VETO_CANARY_STRATEGIES", "alpha")

    # Case 1: no manifest at all.
    predictor.reset_manifest_cache()
    with mock.patch.object(predictor, "_load_model") as load_model, \
         mock.patch("xgboost.DMatrix"):
        fake_booster = mock.Mock()
        fake_booster.predict.return_value = [0.95]
        load_model.return_value = fake_booster
        decision = predictor.score_intent(_intent(strategy="alpha"))
    assert decision.manifest_valid is False
    assert decision.final_action == predictor.ACTION_PASS
    assert decision.reason == predictor.REASON_NO_MANIFEST

    # Case 2: manifest present but stale (trained 9999 days ago).
    _write_manifest(isolated_predictor, trained_days_ago=9999)
    predictor.reset_manifest_cache()
    with mock.patch.object(predictor, "_load_model") as load_model, \
         mock.patch("xgboost.DMatrix"):
        fake_booster = mock.Mock()
        fake_booster.predict.return_value = [0.95]
        load_model.return_value = fake_booster
        decision = predictor.score_intent(_intent(strategy="alpha"))
    assert decision.manifest_stale is True
    assert decision.final_action == predictor.ACTION_PASS
    assert decision.reason == predictor.REASON_MANIFEST_STALE

    # Case 3: manifest valid shape but training_samples below floor.
    _write_manifest(isolated_predictor, samples=10, trained_days_ago=1)
    predictor.reset_manifest_cache()
    with mock.patch.object(predictor, "_load_model") as load_model, \
         mock.patch("xgboost.DMatrix"):
        fake_booster = mock.Mock()
        fake_booster.predict.return_value = [0.95]
        load_model.return_value = fake_booster
        decision = predictor.score_intent(_intent(strategy="alpha"))
    assert decision.manifest_valid is False
    assert decision.final_action == predictor.ACTION_PASS
    assert decision.reason == predictor.REASON_MANIFEST_LOW_SAMPLES


def test_ml_veto_fails_open_when_model_load_raises(isolated_predictor, monkeypatch):
    """If the booster fails to load, score_intent must return a
    fail-open decision with reason=model_unavailable rather than
    raise — trade flow cannot be stranded by a model error."""
    _write_manifest(isolated_predictor)
    monkeypatch.setenv("CHAD_ML_VETO_ENABLED", "1")
    monkeypatch.setenv("CHAD_ML_VETO_CANARY_STRATEGIES", "alpha")
    predictor.reset_manifest_cache()

    with mock.patch.object(predictor, "_load_model", return_value=None):
        decision = predictor.score_intent(_intent(strategy="alpha"))
    assert decision.would_veto is False
    assert decision.final_action == predictor.ACTION_PASS
    assert decision.reason == predictor.REASON_NO_MODEL


# ---------------------------------------------------------------------------
# 7. Canary strategy allowlist
# ---------------------------------------------------------------------------


def test_ml_veto_canary_strategy_allowlist(isolated_predictor, monkeypatch):
    """With enforcement on and a fresh manifest, only strategies
    listed in CHAD_ML_VETO_CANARY_STRATEGIES are eligible for veto.
    Strategies outside the allowlist resolve to pass even when the
    model would veto."""
    _write_manifest(isolated_predictor)
    monkeypatch.setenv("CHAD_ML_VETO_ENABLED", "1")
    predictor.reset_manifest_cache()

    # No canary configured at all → enforcement denied for every
    # strategy. This is the safety-first default Batch 9 specifies.
    monkeypatch.delenv("CHAD_ML_VETO_CANARY_STRATEGIES", raising=False)
    predictor.reset_manifest_cache()
    with mock.patch.object(predictor, "_load_model") as load_model, \
         mock.patch("xgboost.DMatrix"):
        fake_booster = mock.Mock()
        fake_booster.predict.return_value = [0.95]
        load_model.return_value = fake_booster
        decision = predictor.score_intent(_intent(strategy="alpha"))
    assert decision.would_veto is True
    assert decision.final_action == predictor.ACTION_PASS
    assert decision.reason == predictor.REASON_NOT_IN_CANARY

    # Canary configured but strategy not in allowlist → pass.
    monkeypatch.setenv("CHAD_ML_VETO_CANARY_STRATEGIES", "delta,gamma")
    predictor.reset_manifest_cache()
    with mock.patch.object(predictor, "_load_model") as load_model, \
         mock.patch("xgboost.DMatrix"):
        fake_booster = mock.Mock()
        fake_booster.predict.return_value = [0.95]
        load_model.return_value = fake_booster
        decision = predictor.score_intent(_intent(strategy="alpha"))
    assert decision.canary_match is False
    assert decision.final_action == predictor.ACTION_PASS
    assert decision.reason == predictor.REASON_NOT_IN_CANARY

    # Strategy IS in allowlist → veto fires.
    monkeypatch.setenv("CHAD_ML_VETO_CANARY_STRATEGIES", "alpha,delta")
    predictor.reset_manifest_cache()
    with mock.patch.object(predictor, "_load_model") as load_model, \
         mock.patch("xgboost.DMatrix"):
        fake_booster = mock.Mock()
        fake_booster.predict.return_value = [0.95]
        load_model.return_value = fake_booster
        decision = predictor.score_intent(_intent(strategy="alpha"))
    assert decision.canary_match is True
    assert decision.final_action == predictor.ACTION_VETO
    assert decision.reason == predictor.REASON_VETO_FIRED
