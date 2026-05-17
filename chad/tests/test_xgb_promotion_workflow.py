"""chad/tests/test_xgb_promotion_workflow.py

Phase 3 promotion workflow tests for the XGB veto model
(docs/XGB_VETO_MODEL_ARTIFACT_HYGIENE_PLAN_2026-05-17.md).

Covers the three surfaces introduced by the artifact hygiene plan:
  - predictor runtime-current fallback (ml_veto_predictor)
  - trainer candidates-only write path (train_xgb_model)
  - promotion CLI (scripts/promote_xgb_veto.py): list, status,
    auto-gate, operator-approve override, atomic write, manifest
    audit trail.

All tests are hermetic — they only write under ``tmp_path`` via
monkeypatched module attrs. Nothing touches real ``shared/models/``
or ``runtime/models/`` paths.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pytest

from chad.analytics import ml_veto_predictor as predictor
from chad.analytics import train_xgb_model as trainer


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMOTE_SCRIPT = REPO_ROOT / "scripts" / "promote_xgb_veto.py"


# ---------------------------------------------------------------------------
# Promotion script loader (scripts/ is not on sys.path by default)
# ---------------------------------------------------------------------------


def _load_promote_module(monkeypatch, tmp_path: Path):
    """Load scripts/promote_xgb_veto.py as a fresh module and rewire
    its CANDIDATES_DIR / RUNTIME_* / BASELINE_* constants to tmp_path
    so the test is hermetic. Returns the module."""
    spec = importlib.util.spec_from_file_location(
        "_promote_xgb_veto_test_mod", PROMOTE_SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    candidates = tmp_path / "candidates"
    current = tmp_path / "current"
    baseline = tmp_path / "baseline"
    candidates.mkdir(parents=True, exist_ok=True)
    baseline.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(mod, "CANDIDATES_DIR", candidates)
    monkeypatch.setattr(mod, "RUNTIME_MODEL_DIR", current)
    monkeypatch.setattr(mod, "RUNTIME_MODEL_PATH", current / "xgb_veto_model.json")
    monkeypatch.setattr(mod, "RUNTIME_MANIFEST_PATH", current / "xgb_veto_manifest.json")
    monkeypatch.setattr(mod, "BASELINE_MODEL_PATH", baseline / "xgb_veto_model.json")
    monkeypatch.setattr(mod, "BASELINE_MANIFEST_PATH", baseline / "xgb_veto_manifest.json")
    return mod


def _write_candidate(
    candidates_dir: Path,
    candidate_id: str,
    accuracy: float,
    logloss: float,
    model_body: bytes = b'{"booster": "fake"}',
) -> Path:
    """Write a minimal candidate dir (model + manifest) and return its path."""
    cdir = candidates_dir / candidate_id
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "xgb_veto_model.json").write_bytes(model_body)
    manifest: Dict[str, Any] = {
        "schema_version": "xgb_manifest.v2",
        "model_path": str(cdir / "xgb_veto_model.json"),
        "model_sha256": "sha256:fakedigest",
        "model_version": f"xgb_veto_{candidate_id}",
        "trained_at_utc": "2026-05-17T02:00:00+00:00",
        "training_samples": 150,
        "feature_names": list(predictor.FEATURE_NAMES),
        "metrics": {"accuracy": float(accuracy), "logloss": float(logloss)},
        "candidate_id": candidate_id,
    }
    (cdir / "xgb_veto_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return cdir


def _write_baseline(baseline_dir: Path, accuracy: float, logloss: float) -> None:
    (baseline_dir / "xgb_veto_model.json").write_bytes(b'{"booster": "baseline"}')
    manifest: Dict[str, Any] = {
        "schema_version": "xgb_manifest.v2",
        "model_path": str(baseline_dir / "xgb_veto_model.json"),
        "model_sha256": "sha256:baseline",
        "model_version": "xgb_veto_baseline_20260510_020007",
        "trained_at_utc": "2026-05-10T02:00:00+00:00",
        "training_samples": 872,
        "feature_names": list(predictor.FEATURE_NAMES),
        "metrics": {"accuracy": float(accuracy), "logloss": float(logloss)},
    }
    (baseline_dir / "xgb_veto_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Predictor: runtime-current fallback
# ---------------------------------------------------------------------------


def _write_predictor_manifest(path: Path, version: str) -> None:
    payload: Dict[str, Any] = {
        "schema_version": "xgb_manifest.v2",
        "model_path": str(path.parent / "xgb_veto_model.json"),
        "model_version": version,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_samples": 500,
        "feature_names": list(predictor.FEATURE_NAMES),
        "metrics": {"accuracy": 0.75, "logloss": 0.55},
        "model_sha256": "sha256:abc",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_predictor_prefers_runtime_model_when_present(tmp_path, monkeypatch):
    """When runtime-current manifest exists, the predictor loads it
    in preference to the tracked baseline."""
    baseline_dir = tmp_path / "baseline"
    runtime_dir = tmp_path / "runtime"
    baseline_dir.mkdir()
    runtime_dir.mkdir()

    monkeypatch.setattr(predictor, "MANIFEST_PATH",
                        baseline_dir / "xgb_veto_manifest.json")
    monkeypatch.setattr(predictor, "MODEL_PATH",
                        baseline_dir / "xgb_veto_model.json")
    monkeypatch.setattr(predictor, "RUNTIME_MANIFEST_PATH",
                        runtime_dir / "xgb_veto_manifest.json")
    monkeypatch.setattr(predictor, "RUNTIME_MODEL_PATH",
                        runtime_dir / "xgb_veto_model.json")

    _write_predictor_manifest(predictor.MANIFEST_PATH, "xgb_veto_baseline")
    _write_predictor_manifest(predictor.RUNTIME_MANIFEST_PATH, "xgb_veto_runtime")

    predictor.reset_manifest_cache()
    manifest = predictor._load_manifest()
    assert manifest is not None
    assert manifest["model_version"] == "xgb_veto_runtime"


def test_predictor_falls_back_to_baseline_when_runtime_absent(tmp_path, monkeypatch):
    """When runtime-current is absent, the predictor falls back to
    the tracked baseline. This is the default state immediately
    after Phase 1 lands."""
    baseline_dir = tmp_path / "baseline"
    runtime_dir = tmp_path / "runtime"
    baseline_dir.mkdir()
    runtime_dir.mkdir()  # exists but empty

    monkeypatch.setattr(predictor, "MANIFEST_PATH",
                        baseline_dir / "xgb_veto_manifest.json")
    monkeypatch.setattr(predictor, "MODEL_PATH",
                        baseline_dir / "xgb_veto_model.json")
    monkeypatch.setattr(predictor, "RUNTIME_MANIFEST_PATH",
                        runtime_dir / "xgb_veto_manifest.json")
    monkeypatch.setattr(predictor, "RUNTIME_MODEL_PATH",
                        runtime_dir / "xgb_veto_model.json")

    _write_predictor_manifest(predictor.MANIFEST_PATH, "xgb_veto_baseline")

    predictor.reset_manifest_cache()
    manifest = predictor._load_manifest()
    assert manifest is not None
    assert manifest["model_version"] == "xgb_veto_baseline"


def test_predictor_fails_open_when_both_paths_missing(tmp_path, monkeypatch):
    """When neither runtime-current nor baseline manifest exists, the
    predictor must return None (fail-open) rather than raise."""
    monkeypatch.setattr(predictor, "MANIFEST_PATH",
                        tmp_path / "missing" / "xgb_veto_manifest.json")
    monkeypatch.setattr(predictor, "MODEL_PATH",
                        tmp_path / "missing" / "xgb_veto_model.json")
    monkeypatch.setattr(predictor, "RUNTIME_MANIFEST_PATH",
                        tmp_path / "missing_runtime" / "xgb_veto_manifest.json")
    monkeypatch.setattr(predictor, "RUNTIME_MODEL_PATH",
                        tmp_path / "missing_runtime" / "xgb_veto_model.json")

    predictor.reset_manifest_cache()
    assert predictor._load_manifest() is None
    # Predictor must keep working — _load_model returns None, score_intent
    # collapses to shadow_only fail-open.
    assert predictor._load_model() is None


# ---------------------------------------------------------------------------
# Trainer: candidates-only write surface
# ---------------------------------------------------------------------------


def _build_trainer_dataset(n: int = 150):
    """Build a tiny dataset that passes MIN_TRAINING_SAMPLES."""
    import numpy as np
    rows = []
    for i in range(n):
        rows.append({
            "record_hash": f"rh_{i}",
            "payload": {
                "strategy": "alpha",
                "side": "BUY" if i % 3 == 0 else "SELL",
                "regime": "ranging",
                "net_pnl": 10.0 if i % 2 == 0 else -10.0,
                "exit_time_utc": "2026-05-01T12:00:00+00:00",
                "schema_version": "closed_trade.v2",
                "fill_ids": [f"f_{i}_a", f"f_{i}_b"],
            },
        })
    X, y, excluded = trainer._build_dataset(rows, set(), set())
    return X, y, excluded


def test_trainer_writes_candidate_into_candidates_dir(tmp_path, monkeypatch):
    """With MODEL_PATH/MANIFEST_PATH at their production defaults,
    trainer must write into CANDIDATES_DIR/<ts>/, not shared/models."""
    pytest.importorskip("xgboost")

    candidates = tmp_path / "candidates"
    monkeypatch.setattr(trainer, "CANDIDATES_DIR", candidates)
    monkeypatch.setattr(trainer, "PERF_PATH", tmp_path / "model_performance.json")
    # MODEL_PATH / MANIFEST_PATH stay at production defaults so the
    # trainer's candidate-path selector picks the CANDIDATES_DIR branch.

    X, y, excluded = _build_trainer_dataset(150)
    result = trainer._train_model(X, y, excluded)
    assert result.ok, f"train failed: {result.reason}"

    # The candidate dir must be under CANDIDATES_DIR; the model and
    # manifest must exist inside it.
    assert result.manifest_path is not None
    assert candidates in result.manifest_path.parents
    assert (result.manifest_path.parent / "xgb_veto_model.json").is_file()
    assert result.manifest_path.is_file()


def test_trainer_candidate_dir_named_with_utc_timestamp(tmp_path, monkeypatch):
    """The candidate dir name must be an UTC timestamp of the form
    YYYYMMDD_HHMMSS — used as ``candidate_id`` for promotion."""
    pytest.importorskip("xgboost")

    candidates = tmp_path / "candidates"
    monkeypatch.setattr(trainer, "CANDIDATES_DIR", candidates)
    monkeypatch.setattr(trainer, "PERF_PATH", tmp_path / "model_performance.json")

    X, y, excluded = _build_trainer_dataset(150)
    result = trainer._train_model(X, y, excluded)
    assert result.ok

    cand_dir_name = result.manifest_path.parent.name
    # Validate "YYYYMMDD_HHMMSS" shape strictly.
    parsed = datetime.strptime(cand_dir_name, "%Y%m%d_%H%M%S")
    assert parsed.year >= 2026
    # The manifest must record the same candidate_id and a matching
    # model_version of the form xgb_veto_<ts>.
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["candidate_id"] == cand_dir_name
    assert manifest["model_version"] == f"xgb_veto_{cand_dir_name}"


def test_trainer_does_not_overwrite_shared_models(tmp_path, monkeypatch):
    """Trainer must not write to shared/models/xgb_veto_model.json
    under any normal code path — the tracked baseline is owned by
    the operator-controlled promotion process, not the timer."""
    pytest.importorskip("xgboost")

    candidates = tmp_path / "candidates"
    fake_shared = tmp_path / "fake_shared_models"
    fake_shared.mkdir()
    fake_baseline_model = fake_shared / "xgb_veto_model.json"
    fake_baseline_manifest = fake_shared / "xgb_veto_manifest.json"
    fake_baseline_model.write_bytes(b'{"baseline": true}')
    fake_baseline_manifest.write_text(json.dumps({"baseline": True}))
    baseline_model_mtime_before = fake_baseline_model.stat().st_mtime_ns
    baseline_manifest_mtime_before = fake_baseline_manifest.stat().st_mtime_ns

    # Make MODELS_DIR / MODEL_PATH / MANIFEST_PATH look like real
    # shared/models so the default-detection branch lands on CANDIDATES_DIR.
    monkeypatch.setattr(trainer, "MODELS_DIR", fake_shared)
    monkeypatch.setattr(trainer, "MODEL_PATH", fake_baseline_model)
    monkeypatch.setattr(trainer, "MANIFEST_PATH", fake_baseline_manifest)
    monkeypatch.setattr(trainer, "CANDIDATES_DIR", candidates)
    monkeypatch.setattr(trainer, "PERF_PATH", tmp_path / "model_performance.json")

    X, y, excluded = _build_trainer_dataset(150)
    result = trainer._train_model(X, y, excluded)
    assert result.ok

    # Tracked baseline files must be untouched.
    assert fake_baseline_model.read_bytes() == b'{"baseline": true}'
    assert fake_baseline_model.stat().st_mtime_ns == baseline_model_mtime_before
    assert fake_baseline_manifest.stat().st_mtime_ns == baseline_manifest_mtime_before
    # And the candidate must still have landed under CANDIDATES_DIR.
    assert candidates in result.manifest_path.parents


# ---------------------------------------------------------------------------
# Promotion CLI: --list / --status
# ---------------------------------------------------------------------------


def test_promote_list_shows_available_candidates(tmp_path, monkeypatch, capsys):
    mod = _load_promote_module(monkeypatch, tmp_path)
    _write_candidate(mod.CANDIDATES_DIR, "20260517_020001", 0.71, 0.59)
    _write_candidate(mod.CANDIDATES_DIR, "20260510_020007", 0.75, 0.53)

    rc = mod.main(["--list"])
    out = capsys.readouterr().out
    assert rc == 0
    # Both candidate ids appear, newest first.
    assert "20260517_020001" in out
    assert "20260510_020007" in out
    assert out.index("20260517_020001") < out.index("20260510_020007")


def test_promote_status_reports_active_model(tmp_path, monkeypatch, capsys):
    mod = _load_promote_module(monkeypatch, tmp_path)
    _write_baseline(mod.BASELINE_MODEL_PATH.parent, accuracy=0.75, logloss=0.53)

    rc = mod.main(["--status"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "active model" in out
    assert "source=baseline" in out
    assert "xgb_veto_baseline_20260510_020007" in out


# ---------------------------------------------------------------------------
# Promotion CLI: gate behavior
# ---------------------------------------------------------------------------


def test_promote_gate_passes_when_metrics_improve(tmp_path, monkeypatch, capsys):
    mod = _load_promote_module(monkeypatch, tmp_path)
    _write_baseline(mod.BASELINE_MODEL_PATH.parent, accuracy=0.70, logloss=0.60)
    _write_candidate(mod.CANDIDATES_DIR, "20260517_020001", 0.75, 0.55)

    rc = mod.main(["--candidate", "20260517_020001"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "promoted candidate 20260517_020001 -> current" in out
    assert mod.RUNTIME_MANIFEST_PATH.is_file()
    promoted = json.loads(mod.RUNTIME_MANIFEST_PATH.read_text())
    assert promoted["promoted_by"] == "auto_gate"


def test_promote_gate_blocks_when_accuracy_regresses(tmp_path, monkeypatch, capsys):
    mod = _load_promote_module(monkeypatch, tmp_path)
    _write_baseline(mod.BASELINE_MODEL_PATH.parent, accuracy=0.75, logloss=0.55)
    _write_candidate(mod.CANDIDATES_DIR, "20260517_020001", 0.71, 0.54)

    rc = mod.main(["--candidate", "20260517_020001"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "promotion blocked" in err
    # Nothing must have been written to runtime/current.
    assert not mod.RUNTIME_MANIFEST_PATH.exists()
    assert not mod.RUNTIME_MODEL_PATH.exists()


def test_promote_gate_blocks_when_logloss_worsens(tmp_path, monkeypatch, capsys):
    mod = _load_promote_module(monkeypatch, tmp_path)
    _write_baseline(mod.BASELINE_MODEL_PATH.parent, accuracy=0.75, logloss=0.53)
    # Higher accuracy AND higher (worse) logloss — gate must still block.
    _write_candidate(mod.CANDIDATES_DIR, "20260517_020001", 0.76, 0.59)

    rc = mod.main(["--candidate", "20260517_020001"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "promotion blocked" in err
    assert not mod.RUNTIME_MANIFEST_PATH.exists()


def test_promote_operator_approve_overrides_gate(tmp_path, monkeypatch, capsys):
    mod = _load_promote_module(monkeypatch, tmp_path)
    _write_baseline(mod.BASELINE_MODEL_PATH.parent, accuracy=0.75, logloss=0.53)
    _write_candidate(mod.CANDIDATES_DIR, "20260517_020001", 0.71, 0.59)

    rc = mod.main([
        "--candidate", "20260517_020001",
        "--operator-approve", "scheduled regression accepted for shadow soak",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "promoted candidate 20260517_020001 -> current" in out
    promoted = json.loads(mod.RUNTIME_MANIFEST_PATH.read_text())
    assert promoted["promoted_by"].startswith("operator_approve:")
    assert "scheduled regression" in promoted["promoted_by"]
    assert promoted["operator_approve_reason"] == \
        "scheduled regression accepted for shadow soak"


# ---------------------------------------------------------------------------
# Promotion CLI: manifest audit fields + atomic write
# ---------------------------------------------------------------------------


def test_promoted_manifest_contains_promotion_metadata(tmp_path, monkeypatch):
    mod = _load_promote_module(monkeypatch, tmp_path)
    _write_baseline(mod.BASELINE_MODEL_PATH.parent, accuracy=0.70, logloss=0.60)
    _write_candidate(mod.CANDIDATES_DIR, "20260517_020001", 0.75, 0.55)

    rc = mod.main(["--candidate", "20260517_020001"])
    assert rc == 0

    promoted = json.loads(mod.RUNTIME_MANIFEST_PATH.read_text())
    required = (
        "promoted_at_utc",
        "promoted_from_candidate",
        "promoted_by",
        "metrics_delta_accuracy",
        "metrics_delta_logloss",
    )
    missing = [k for k in required if k not in promoted]
    assert not missing, f"promoted manifest missing: {missing}"

    assert promoted["promoted_from_candidate"] == "20260517_020001"
    assert promoted["promoted_by"] == "auto_gate"
    assert abs(promoted["metrics_delta_accuracy"] - 0.05) < 1e-9
    assert abs(promoted["metrics_delta_logloss"] - (-0.05)) < 1e-9
    # ISO-8601 UTC timestamp.
    datetime.fromisoformat(promoted["promoted_at_utc"].replace("Z", "+00:00"))


def test_promoted_model_written_atomically(tmp_path, monkeypatch):
    """After a successful promote, only the final files exist —
    the ``.tmp`` staging artifacts must have been os.replace()'d
    into their final names, leaving no half-written state."""
    mod = _load_promote_module(monkeypatch, tmp_path)
    _write_baseline(mod.BASELINE_MODEL_PATH.parent, accuracy=0.70, logloss=0.60)
    cand_body = b'{"booster": "candidate-bytes-marker"}'
    _write_candidate(
        mod.CANDIDATES_DIR, "20260517_020001",
        accuracy=0.75, logloss=0.55, model_body=cand_body,
    )

    rc = mod.main(["--candidate", "20260517_020001"])
    assert rc == 0

    # Final files exist with the candidate's bytes.
    assert mod.RUNTIME_MODEL_PATH.is_file()
    assert mod.RUNTIME_MANIFEST_PATH.is_file()
    assert mod.RUNTIME_MODEL_PATH.read_bytes() == cand_body

    # No leftover staging files in the current dir.
    leftovers = sorted(p.name for p in mod.RUNTIME_MODEL_DIR.iterdir()
                       if p.name.endswith(".tmp"))
    assert leftovers == []
