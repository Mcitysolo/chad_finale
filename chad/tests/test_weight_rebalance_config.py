#!/usr/bin/env python3
"""
Tests for the weight_rebalance_config action kind and the governed
config/strategy_weights.json read path in StrategyAllocation.

Covers:
- validate_weight_payload: schema enforcement, edge cases, rejection modes
- apply_weight_rebalance_config: atomic write, document structure, idempotency
- StrategyAllocation.from_env_or_default: tier precedence (env > config > hardcoded)
- StrategyAllocation._load_from_config_file: corrupt/missing/bad-schema handling
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest

from chad.ops.action_applier import (
    DISPATCH,
    STRATEGY_WEIGHTS_PATH,
    STRATEGY_WEIGHTS_SCHEMA_VERSION,
    apply_weight_rebalance_config,
    utc_now,
    validate_weight_payload,
    write_json_atomic,
)
from chad.risk.dynamic_risk_allocator import (
    DEFAULT_STRATEGY_WEIGHTS,
    StrategyAllocation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_repo(tmp_path: Path) -> Path:
    """Create a minimal repo layout with config/ directory."""
    (tmp_path / "config").mkdir()
    return tmp_path


@pytest.fixture()
def valid_weights() -> Dict[str, float]:
    return {
        "alpha": 0.27,
        "beta": 0.22,
        "gamma": 0.13,
        "omega": 0.10,
        "delta": 0.05,
        "alpha_futures": 0.15,
        "crypto": 0.08,
    }


@pytest.fixture()
def valid_payload(valid_weights: Dict[str, float]) -> Dict[str, Any]:
    return {
        "weights": valid_weights,
        "reason": "Phase 1 rebalance",
    }


@pytest.fixture()
def valid_action(valid_payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "action_id": "test_action_001",
        "kind": "weight_rebalance_config",
        "payload": valid_payload,
        "status": "approved",
        "ts_utc": utc_now(),
        "expires_ts_utc": "2099-12-31T23:59:59Z",
    }


def _write_config(repo: Path, weights: Dict[str, float], **overrides: Any) -> Path:
    """Helper: write a strategy_weights.json into repo/config/."""
    doc: Dict[str, Any] = {
        "schema_version": STRATEGY_WEIGHTS_SCHEMA_VERSION,
        "weights": weights,
        "applied_action_id": None,
        "updated_ts_utc": utc_now(),
    }
    doc.update(overrides)
    p = repo / "config" / "strategy_weights.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return p


# ===================================================================
# validate_weight_payload
# ===================================================================


class TestValidateWeightPayload:
    """Validation must be fail-closed: reject anything ambiguous."""

    def test_valid_payload(self, valid_payload: Dict[str, Any]) -> None:
        weights = validate_weight_payload(valid_payload)
        assert isinstance(weights, dict)
        assert len(weights) == 7
        assert abs(sum(weights.values()) - 1.0) < 0.02

    def test_payload_not_dict(self) -> None:
        with pytest.raises(RuntimeError, match="payload_not_dict"):
            validate_weight_payload("not a dict")  # type: ignore[arg-type]

    def test_weights_missing(self) -> None:
        with pytest.raises(RuntimeError, match="weights_missing_or_empty"):
            validate_weight_payload({"reason": "no weights key"})

    def test_weights_empty_dict(self) -> None:
        with pytest.raises(RuntimeError, match="weights_missing_or_empty"):
            validate_weight_payload({"weights": {}})

    def test_negative_weight(self) -> None:
        with pytest.raises(RuntimeError, match="negative_weight"):
            validate_weight_payload({"weights": {"alpha": -0.1, "beta": 1.1}})

    def test_weight_exceeds_one(self) -> None:
        with pytest.raises(RuntimeError, match="weight_exceeds_1"):
            validate_weight_payload({"weights": {"alpha": 1.5}})

    def test_sum_too_high(self) -> None:
        with pytest.raises(RuntimeError, match="sum_out_of_range"):
            validate_weight_payload({"weights": {"alpha": 0.6, "beta": 0.6}})

    def test_sum_too_low(self) -> None:
        with pytest.raises(RuntimeError, match="sum_out_of_range"):
            validate_weight_payload({"weights": {"alpha": 0.1, "beta": 0.1}})

    def test_invalid_key_format(self) -> None:
        with pytest.raises(RuntimeError, match="invalid_key_format"):
            validate_weight_payload({"weights": {"123bad": 0.5, "alpha": 0.5}})

    def test_key_with_special_chars(self) -> None:
        with pytest.raises(RuntimeError, match="invalid_key_format"):
            validate_weight_payload({"weights": {"alpha-beta": 0.5, "gamma": 0.5}})

    def test_non_numeric_value(self) -> None:
        with pytest.raises(RuntimeError, match="value_not_number"):
            validate_weight_payload({"weights": {"alpha": "high"}})

    def test_duplicate_keys_via_case(self) -> None:
        # JSON dicts can't have true dupes, but mixed case normalizes to same key
        with pytest.raises(RuntimeError, match="duplicate_key"):
            validate_weight_payload({"weights": {"Alpha": 0.5, "alpha": 0.5}})

    def test_exactly_one_point_zero(self) -> None:
        weights = validate_weight_payload({"weights": {"alpha": 1.0}})
        assert weights == {"alpha": 1.0}

    def test_within_tolerance(self) -> None:
        # 0.99 + 0.005 = 0.995 — within 0.015 tolerance
        weights = validate_weight_payload(
            {"weights": {"alpha": 0.50, "beta": 0.495}}
        )
        assert abs(sum(weights.values()) - 0.995) < 1e-9


# ===================================================================
# apply_weight_rebalance_config
# ===================================================================


class TestApplyWeightRebalanceConfig:

    def test_applies_valid_action(
        self, valid_action: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Full happy path: write config, verify document structure."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "config" / "strategy_weights.json"
            monkeypatch.setattr(
                "chad.ops.action_applier.STRATEGY_WEIGHTS_PATH", config_path
            )

            result = apply_weight_rebalance_config(valid_action)
            assert result == str(config_path)
            assert config_path.is_file()

            doc = json.loads(config_path.read_text(encoding="utf-8"))
            assert doc["schema_version"] == STRATEGY_WEIGHTS_SCHEMA_VERSION
            assert doc["applied_action_id"] == "test_action_001"
            assert isinstance(doc["weights"], dict)
            assert len(doc["weights"]) == 7
            # Weights should be sorted by key
            keys = list(doc["weights"].keys())
            assert keys == sorted(keys)
            assert doc["reason"] == "Phase 1 rebalance"

    def test_preserves_previous_weights(
        self, valid_action: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """New config doc should contain the previous weights for audit trail."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "config" / "strategy_weights.json"
            config_path.parent.mkdir(parents=True)

            # Write an initial config
            initial = {
                "schema_version": STRATEGY_WEIGHTS_SCHEMA_VERSION,
                "weights": {"alpha": 0.5, "beta": 0.5},
                "applied_action_id": None,
                "updated_ts_utc": utc_now(),
            }
            config_path.write_text(json.dumps(initial) + "\n", encoding="utf-8")

            monkeypatch.setattr(
                "chad.ops.action_applier.STRATEGY_WEIGHTS_PATH", config_path
            )

            apply_weight_rebalance_config(valid_action)

            doc = json.loads(config_path.read_text(encoding="utf-8"))
            assert doc["previous_weights"] == {"alpha": 0.5, "beta": 0.5}

    def test_missing_action_id(self) -> None:
        with pytest.raises(RuntimeError, match="missing_action_id"):
            apply_weight_rebalance_config({"action_id": "", "payload": {}})

    def test_bad_payload_rejected(self) -> None:
        with pytest.raises(RuntimeError, match="weight_config"):
            apply_weight_rebalance_config(
                {"action_id": "x", "payload": {"weights": {"alpha": 5.0}}}
            )


# ===================================================================
# DISPATCH table
# ===================================================================


class TestDispatchTable:

    def test_weight_rebalance_config_registered(self) -> None:
        assert "weight_rebalance_config" in DISPATCH
        assert DISPATCH["weight_rebalance_config"] is apply_weight_rebalance_config

    def test_rebalance_execute_still_registered(self) -> None:
        from chad.ops.action_applier import apply_rebalance

        assert "rebalance_execute" in DISPATCH
        assert DISPATCH["rebalance_execute"] is apply_rebalance

    def test_unknown_kind_not_in_dispatch(self) -> None:
        assert DISPATCH.get("unknown_kind") is None


# ===================================================================
# StrategyAllocation — config file read path
# ===================================================================


class TestStrategyAllocationConfigFile:

    def test_loads_from_config_file(self, tmp_repo: Path, valid_weights: Dict[str, float]) -> None:
        _write_config(tmp_repo, valid_weights)
        alloc = StrategyAllocation.from_env_or_default(repo_root=tmp_repo)
        assert alloc.weights == valid_weights
        assert "config_file" in alloc.source

    def test_falls_back_to_hardcoded_when_no_config(self, tmp_repo: Path) -> None:
        # No config file written — should fall back
        alloc = StrategyAllocation.from_env_or_default(repo_root=tmp_repo)
        assert alloc.weights == DEFAULT_STRATEGY_WEIGHTS
        assert alloc.source == "hardcoded_default"

    def test_falls_back_on_corrupt_json(self, tmp_repo: Path) -> None:
        p = tmp_repo / "config" / "strategy_weights.json"
        p.write_text("not valid json {{{", encoding="utf-8")
        alloc = StrategyAllocation.from_env_or_default(repo_root=tmp_repo)
        assert alloc.weights == DEFAULT_STRATEGY_WEIGHTS
        assert alloc.source == "hardcoded_default"

    def test_falls_back_on_wrong_schema_version(self, tmp_repo: Path) -> None:
        _write_config(
            tmp_repo,
            {"alpha": 1.0},
            schema_version="strategy_weights.v999",
        )
        alloc = StrategyAllocation.from_env_or_default(repo_root=tmp_repo)
        assert alloc.weights == DEFAULT_STRATEGY_WEIGHTS

    def test_falls_back_on_empty_weights(self, tmp_repo: Path) -> None:
        _write_config(tmp_repo, {})
        alloc = StrategyAllocation.from_env_or_default(repo_root=tmp_repo)
        assert alloc.weights == DEFAULT_STRATEGY_WEIGHTS

    def test_falls_back_on_negative_weight(self, tmp_repo: Path) -> None:
        _write_config(tmp_repo, {"alpha": -0.5, "beta": 1.5})
        alloc = StrategyAllocation.from_env_or_default(repo_root=tmp_repo)
        assert alloc.weights == DEFAULT_STRATEGY_WEIGHTS

    def test_falls_back_on_zero_sum(self, tmp_repo: Path) -> None:
        _write_config(tmp_repo, {"alpha": 0.0, "beta": 0.0})
        alloc = StrategyAllocation.from_env_or_default(repo_root=tmp_repo)
        assert alloc.weights == DEFAULT_STRATEGY_WEIGHTS

    def test_env_takes_precedence_over_config(
        self,
        tmp_repo: Path,
        valid_weights: Dict[str, float],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_config(tmp_repo, valid_weights)
        monkeypatch.setenv("CHAD_STRATEGY_WEIGHTS", "alpha=0.6,beta=0.4")
        alloc = StrategyAllocation.from_env_or_default(repo_root=tmp_repo)
        assert alloc.weights == {"alpha": 0.6, "beta": 0.4}
        assert alloc.source == "env:CHAD_STRATEGY_WEIGHTS"

    def test_env_cleared_uses_config(
        self,
        tmp_repo: Path,
        valid_weights: Dict[str, float],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_config(tmp_repo, valid_weights)
        monkeypatch.delenv("CHAD_STRATEGY_WEIGHTS", raising=False)
        alloc = StrategyAllocation.from_env_or_default(repo_root=tmp_repo)
        assert alloc.weights == valid_weights
        assert "config_file" in alloc.source

    def test_normalized_still_works(self, tmp_repo: Path) -> None:
        _write_config(tmp_repo, {"alpha": 0.6, "beta": 0.3, "gamma": 0.1})
        alloc = StrategyAllocation.from_env_or_default(repo_root=tmp_repo)
        norm = alloc.normalized()
        assert abs(sum(norm.values()) - 1.0) < 1e-9

    def test_non_dict_config_file(self, tmp_repo: Path) -> None:
        p = tmp_repo / "config" / "strategy_weights.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        alloc = StrategyAllocation.from_env_or_default(repo_root=tmp_repo)
        assert alloc.weights == DEFAULT_STRATEGY_WEIGHTS

    def test_missing_schema_version_field(self, tmp_repo: Path) -> None:
        p = tmp_repo / "config" / "strategy_weights.json"
        doc = {"weights": {"alpha": 1.0}}
        p.write_text(json.dumps(doc), encoding="utf-8")
        alloc = StrategyAllocation.from_env_or_default(repo_root=tmp_repo)
        assert alloc.weights == DEFAULT_STRATEGY_WEIGHTS

    def test_auto_infers_repo_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """from_env_or_default() with no args should infer repo root from __file__."""
        monkeypatch.delenv("CHAD_STRATEGY_WEIGHTS", raising=False)
        alloc = StrategyAllocation.from_env_or_default()
        # Should load from the real config/strategy_weights.json we just created
        assert isinstance(alloc.weights, dict)
        assert len(alloc.weights) > 0
        assert sum(alloc.weights.values()) > 0


# ===================================================================
# Integration: end-to-end action apply -> allocator reads new weights
# ===================================================================


class TestEndToEnd:

    def test_apply_then_read(
        self,
        valid_action: Dict[str, Any],
        valid_weights: Dict[str, float],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Apply a weight config action, then verify the allocator reads it."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            config_path = repo / "config" / "strategy_weights.json"
            config_path.parent.mkdir(parents=True)

            monkeypatch.setattr(
                "chad.ops.action_applier.STRATEGY_WEIGHTS_PATH", config_path
            )
            monkeypatch.delenv("CHAD_STRATEGY_WEIGHTS", raising=False)

            # Apply the action
            apply_weight_rebalance_config(valid_action)

            # Read via allocator
            alloc = StrategyAllocation.from_env_or_default(repo_root=repo)
            assert alloc.weights == valid_weights
            assert "config_file" in alloc.source
            assert abs(sum(alloc.weights.values()) - 1.0) < 0.02
