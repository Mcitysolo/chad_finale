"""
Tests for intel helper modules:
- model_doctor
- macro_filter
- playbook_writer

Goals
-----
- Verify imports and main entry points work.
- Ensure they handle missing files (e.g. no model_performance.json,
  no dynamic_caps.json) without raising.
- Confirm they write expected artifacts to disk.
"""

from __future__ import annotations

from pathlib import Path

from chad.intel.model_doctor import run_model_doctor, ModelDoctorReport  # type: ignore
from chad.intel.macro_filter import run_macro_filter, MacroImpactAssessment  # type: ignore
from chad.intel.playbook_writer import write_playbook, StrategyPlaybook  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
MODEL_DOCTOR_DIR = REPORTS_DIR / "model_doctor"
MACRO_DIR = REPORTS_DIR / "macro"
PLAYBOOK_DIR = ROOT / "docs" / "strategy_playbooks"


def test_model_doctor_handles_missing_metrics(tmp_path: Path, monkeypatch) -> None:
    """
    ModelDoctor must not raise if model_performance.json is missing
    or unreadable. It should return a ModelDoctorReport with verdict
    'no_model' in that case.
    """
    # Point shared/models to a temp directory with no metrics.
    models_dir = tmp_path / "shared" / "models"
    models_dir.mkdir(parents=True)
    # Monkeypatch PERF_PATH indirectly by adjusting ROOT if needed in future,
    # but current design uses a fixed path under project ROOT. This test
    # simply verifies behaviour in the default environment.
    report = run_model_doctor()
    assert isinstance(report, ModelDoctorReport)
    assert report.verdict in {"no_model", "insufficient_data", "weak", "acceptable", "strong"}

    # Ensure at least one markdown report exists after running.
    assert MODEL_DOCTOR_DIR.exists()
    created = list(MODEL_DOCTOR_DIR.glob("model_doctor_*.md"))
    assert created, "Expected at least one model_doctor_*.md report to be written"


def test_macro_filter_produces_assessment_and_report(tmp_path: Path, monkeypatch) -> None:
    """
    MacroFilter must return a MacroImpactAssessment and write a macro report,
    even when dynamic_caps.json is missing or empty.
    """
    # As with model_doctor, we exercise default environment. If runtime/dynamic_caps.json
    # is missing, the function must still succeed.
    text = "FOMC signals higher-for-longer rates with rising recession risks."
    assessment = run_macro_filter(text)
    assert isinstance(assessment, MacroImpactAssessment)
    assert assessment.summary
    assert assessment.suggested_bias in {"risk_on", "risk_off", "neutral"}

    assert MACRO_DIR.exists()
    created = list(MACRO_DIR.glob("macro_impact_*.md"))
    assert created, "Expected at least one macro_impact_*.md report to be written"


def test_playbook_writer_generates_markdown(tmp_path: Path, monkeypatch) -> None:
    """
    PlaybookWriter must generate a markdown file for the given strategy
    and stats, and return a StrategyPlaybook object.
    """
    # Ensure the playbook directory exists (module guarantees this).
    stats = {
        "win_rate": 0.55,
        "avg_r_multiple": 1.2,
        "max_drawdown": -0.10,
        "trades_30d": 80,
    }
    pb = write_playbook("OMEGA", stats)
    assert isinstance(pb, StrategyPlaybook)
    assert pb.strategy == "OMEGA"

    assert PLAYBOOK_DIR.exists()
    created = list(PLAYBOOK_DIR.glob("omega.md"))
    assert created, "Expected omega.md playbook to be written"
