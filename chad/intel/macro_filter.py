from __future__ import annotations

"""
chad/intel/macro_filter.py

MacroFilter – GPT-ready macro/news impact assessment (Phase-7 heuristic baseline).

Purpose
-------
- Accept a raw macro/news text snippet.
- Optionally read current dynamic caps (runtime/dynamic_caps.json).
- Produce a structured MacroImpactAssessment:
    - summary
    - suggested bias (risk_on / risk_off / neutral)
    - per-strategy impact notes
    - operator-facing Markdown report under reports/macro/

Design Principles
-----------------
- Deterministic, side-effect minimal, and safe:
    - Never raises on missing dynamic_caps.json.
    - Works even with empty or noisy text.
- GPT-ready:
    - You can later replace the simple heuristics with calls to gpt_client
      while keeping the same public interface.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / "runtime"
REPORTS_DIR = ROOT / "reports" / "macro"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DYNAMIC_CAPS_PATH = RUNTIME_DIR / "dynamic_caps.json"


@dataclass(frozen=True)
class MacroImpactAssessment:
    """
    Structured macro impact assessment.

    Attributes
    ----------
    generated_at_iso:
        ISO-8601 timestamp (UTC) of when this assessment was generated.
    summary:
        Cleaned macro/news summary (currently just the input text).
    suggested_bias:
        One of "risk_on", "risk_off", or "neutral".
    affected_strategies:
        Mapping from strategy name to a short note (e.g. "monitor", "tilt_down").
    notes:
        Additional commentary for operators.
    """

    generated_at_iso: str
    summary: str
    suggested_bias: str
    affected_strategies: Dict[str, str]
    notes: str


def _load_dynamic_caps() -> Optional[Dict[str, Any]]:
    """
    Load dynamic caps from runtime/dynamic_caps.json if present.

    Returns
    -------
    dict or None
        Parsed JSON on success, None otherwise.
    """
    if not DYNAMIC_CAPS_PATH.is_file():
        return None
    try:
        return json.loads(DYNAMIC_CAPS_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _infer_bias(raw_text: str) -> str:
    """
    Infer a simple risk bias from macro text using keyword heuristics.

    This is intentionally simple and deterministic. You can later replace this
    with GPT-driven classification without changing callers.
    """
    text = raw_text.lower()

    risk_off_keywords = [
        "recession",
        "slowdown",
        "hard landing",
        "stagflation",
        "war",
        "conflict",
        "default",
        "credit stress",
        "crisis",
        "sell-off",
        "volatility spike",
    ]
    risk_on_keywords = [
        "soft landing",
        "disinflation",
        "easing",
        "rate cuts",
        "growth surprise",
        "record earnings",
        "rally",
        "risk-on",
    ]

    if any(k in text for k in risk_off_keywords):
        return "risk_off"
    if any(k in text for k in risk_on_keywords):
        return "risk_on"
    return "neutral"


def _strategy_notes_from_caps(caps: Optional[Dict[str, Any]], bias: str) -> Dict[str, str]:
    """
    Construct a simple per-strategy note mapping based on dynamic caps and bias.

    Current behaviour:
    - If caps is None: returns an empty dict.
    - Otherwise:
        - For risk_off bias: mark all strategies as "monitor / tilt_down".
        - For risk_on bias: mark all as "monitor / tilt_up".
        - For neutral: mark all as "monitor".
    """
    if caps is None:
        return {}

    strategy_caps = caps.get("strategy_caps", {}) or {}
    notes: Dict[str, str] = {}

    for strat_name in strategy_caps.keys():
        if bias == "risk_off":
            notes[strat_name] = "monitor / tilt_down"
        elif bias == "risk_on":
            notes[strat_name] = "monitor / tilt_up"
        else:
            notes[strat_name] = "monitor"

    return notes


def run_macro_filter(raw_text: str) -> MacroImpactAssessment:
    """
    Run the macro filter on a raw news/macro snippet.

    Parameters
    ----------
    raw_text : str
        Arbitrary macro/news text. May be short or long; no strict format.

    Returns
    -------
    MacroImpactAssessment
        Structured summary of the macro context and its suggested bias.
    """
    now = datetime.now(timezone.utc).isoformat()
    cleaned = raw_text.strip()

    caps = _load_dynamic_caps()
    bias = _infer_bias(cleaned)
    affected = _strategy_notes_from_caps(caps, bias)

    if caps is None:
        base_notes = (
            "Dynamic caps file not found; assessment does not incorporate current "
            "capital allocation. Once runtime/dynamic_caps.json is present, this "
            "module will automatically include per-strategy caps."
        )
    else:
        base_notes = (
            "Assessment is heuristic and based on simple keyword rules. Consider "
            "augmenting this with GPT-based reasoning via chad.intel.gpt_client."
        )

    assessment = MacroImpactAssessment(
        generated_at_iso=now,
        summary=cleaned,
        suggested_bias=bias,
        affected_strategies=affected,
        notes=base_notes,
    )

    # Persist Markdown report for operators.
    slug = now.replace(":", "").replace("-", "").replace("+", "").replace(".", "")
    out_path = REPORTS_DIR / f"macro_impact_{slug}.md"
    lines = [
        "# Macro Impact Assessment",
        "",
        f"Generated at: `{assessment.generated_at_iso}`",
        "",
        "## Summary",
        "",
        assessment.summary or "(no summary provided)",
        "",
        "## Suggested Bias",
        "",
        f"- **Bias**: {assessment.suggested_bias}",
        "",
        "## Affected Strategies",
        "",
    ]
    if assessment.affected_strategies:
        for strat, note in assessment.affected_strategies.items():
            lines.append(f"- **{strat}**: {note}")
    else:
        lines.append("- (no strategy caps available)")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            assessment.notes,
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")

    return assessment


if __name__ == "__main__":  # pragma: no cover
    sample = "FOMC signals higher-for-longer rates with rising recession risks."
    a = run_macro_filter(sample)
    print(json.dumps(asdict(a), indent=2))  # type: ignore[name-defined]
