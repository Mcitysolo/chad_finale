from __future__ import annotations

"""
chad/intel/model_doctor.py

ModelDoctor – analyze ML model performance and produce structured reports.

Purpose
-------
- Read model performance metrics written by train_xgb_model.py.
- Classify health of the model using clear, deterministic thresholds.
- Emit a structured ModelDoctorReport and a Markdown report for operators.

Design Principles
-----------------
- Pure and side-effect minimal:
    - Reads from shared/models/model_performance.json (if present).
    - Writes Markdown to reports/model_doctor/.
- Safe:
    - Never raises on missing/invalid metrics; uses "no_model" verdict instead.
- Extensible:
    - Thresholds are explicit and easy to tune.
    - Can be extended to include multiple models or additional metrics.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "shared" / "models"
REPORTS_DIR = ROOT / "reports" / "model_doctor"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PERF_PATH = MODELS_DIR / "model_performance.json"


@dataclass(frozen=True)
class ModelDoctorReport:
    """
    Structured summary of model performance.

    Attributes
    ----------
    generated_at_iso:
        ISO-8601 timestamp in UTC when this report was generated.
    model_path:
        Directory where models and metrics are stored.
    performance:
        Raw performance metrics loaded from model_performance.json.
    verdict:
        One of:
            - "no_model"         : metrics file missing/unreadable
            - "insufficient_data": metrics present but unusable (NaNs, etc.)
            - "strong"           : metrics meet strict thresholds
            - "acceptable"       : metrics usable but not outstanding
            - "weak"             : metrics fall below minimum quality bar
    notes:
        Human-readable notes for operators.
    """

    generated_at_iso: str
    model_path: str
    performance: Dict[str, Any]
    verdict: str
    notes: str


def _load_performance() -> Optional[Dict[str, Any]]:
    """
    Attempt to load performance metrics from disk.

    Returns
    -------
    dict or None
        Parsed metrics dict on success, None otherwise.
    """
    if not PERF_PATH.is_file():
        return None
    try:
        return json.loads(PERF_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _build_verdict(perf: Dict[str, Any]) -> str:
    """
    Compute a simple verdict based on RMSE/MAE thresholds.

    Thresholds are intentionally conservative and can be tuned later without
    changing the calling code.
    """
    try:
        rmse = float(perf.get("rmse", float("nan")))
        mae = float(perf.get("mae", float("nan")))
    except Exception:
        return "insufficient_data"

    # NaN check
    if rmse != rmse or mae != mae:
        return "insufficient_data"

    # Thresholds: adjust as you learn more about your model's range.
    if rmse < 50.0 and mae < 25.0:
        return "strong"
    if rmse < 100.0 and mae < 50.0:
        return "acceptable"
    return "weak"


def run_model_doctor() -> ModelDoctorReport:
    """
    Run the Model Doctor diagnostic and emit a structured report.

    Returns
    -------
    ModelDoctorReport
        The structured report for further processing or display.
    """
    now = datetime.now(timezone.utc).isoformat()
    perf = _load_performance()

    if perf is None:
        report = ModelDoctorReport(
            generated_at_iso=now,
            model_path=str(MODELS_DIR),
            performance={},
            verdict="no_model",
            notes="No model_performance.json found. Run train_xgb_model.py first.",
        )
    else:
        verdict = _build_verdict(perf)
        notes = (
            "Model performance metrics loaded successfully. Verdict is based on "
            "simple RMSE/MAE thresholds and should be tuned as you gain more "
            "experience with live performance."
        )
        report = ModelDoctorReport(
            generated_at_iso=now,
            model_path=str(MODELS_DIR),
            performance=perf,
            verdict=verdict,
            notes=notes,
        )

    # Persist Markdown report for operators.
    slug = now.replace(":", "").replace("-", "").replace("+", "").replace(".", "")
    out_path = REPORTS_DIR / f"model_doctor_{slug}.md"
    lines = [
        "# CHAD Model Doctor Report",
        "",
        f"Generated at: `{report.generated_at_iso}`",
        "",
        f"Model directory: `{report.model_path}`",
        "",
        "## Verdict",
        "",
        f"- Verdict: **{report.verdict}**",
        "",
        "## Metrics",
        "",
    ]
    for key, value in report.performance.items():
        lines.append(f"- **{key}**: {value}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            report.notes,
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")

    return report


if __name__ == "__main__":  # pragma: no cover
    # Allow quick CLI inspection:
    r = run_model_doctor()
    print(json.dumps(asdict(r), indent=2))
