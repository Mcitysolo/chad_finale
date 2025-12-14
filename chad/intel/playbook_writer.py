from __future__ import annotations

"""
chad/intel/playbook_writer.py

PlaybookWriter – generate per-strategy playbooks as Markdown.

Purpose
-------
- Convert raw strategy statistics into human-readable playbooks.
- Provide a stable, file-based interface for operators and future GPT layers.
- Keep logic deterministic, side-effect minimal, and testable.

Design
------
- Pure function `write_playbook(strategy, stats)`:
    - Writes docs/strategy_playbooks/{strategy}.md
    - Returns a StrategyPlaybook dataclass instance.
- Safe for repeated calls:
    - Overwrites the same file path deterministically.
- GPT-ready:
    - Edge summary / notes are deterministic placeholders that can be replaced
      by gpt_client-driven content without changing the function signature.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]
PLAYBOOK_DIR = ROOT / "docs" / "strategy_playbooks"
PLAYBOOK_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class StrategyPlaybook:
    """
    Structured description of a strategy playbook.

    Attributes
    ----------
    strategy:
        Strategy name (e.g. "ALPHA", "BETA", "ALPHA_CRYPTO").
    description:
        High-level purpose of the strategy.
    edge_summary:
        Summary of the edge this strategy is designed to exploit.
    guardrails:
        Description of key constraints, caps, and safety mechanisms.
    notes:
        Additional free-form commentary.
    """

    strategy: str
    description: str
    edge_summary: str
    guardrails: str
    notes: str


def write_playbook(strategy: str, stats: Dict[str, Any]) -> StrategyPlaybook:
    """
    Write or update the Markdown playbook for a given strategy.

    Parameters
    ----------
    strategy : str
        Strategy name. Typically StrategyName.name or similar.
    stats : dict
        Raw statistics or metadata about the strategy's recent behaviour
        (e.g. win rate, Sharpe-like metrics, drawdown, trade counts).

    Returns
    -------
    StrategyPlaybook
        The structured playbook representation.
    """
    strategy_upper = strategy.upper()
    description = f"Playbook for {strategy_upper} based on recent performance and configuration."

    # Deterministic placeholders, ready for GPT enhancements later.
    edge_summary = (
        "Edge summary is currently heuristic and should be refined over time. "
        "You can replace this section with GPT-generated insights that explain "
        "how this strategy finds and exploits its edge in different regimes."
    )
    guardrails = (
        "Respect global caps, SCR state, LiveGate decisions, and per-strategy "
        "notional limits. This strategy must never violate portfolio-level "
        "risk constraints or daily loss thresholds."
    )
    notes = (
        "Use this playbook as a living document. As CHAD learns and you refine "
        "risk and execution, update this file to reflect best practices and "
        "lessons learned from real performance."
    )

    pb = StrategyPlaybook(
        strategy=strategy_upper,
        description=description,
        edge_summary=edge_summary,
        guardrails=guardrails,
        notes=notes,
    )

    path = PLAYBOOK_DIR / f"{strategy_upper.lower()}.md"

    lines = [
        f"# {strategy_upper} Strategy Playbook",
        "",
        "## Overview",
        "",
        pb.description,
        "",
        "## Recent Stats (raw)",
        "",
    ]

    # Attach raw stats in a stable, human-readable format.
    for key, value in sorted(stats.items(), key=lambda kv: kv[0]):
        lines.append(f"- **{key}**: {value}")

    lines.extend(
        [
            "",
            "## Edge Summary",
            "",
            pb.edge_summary,
            "",
            "## Guardrails",
            "",
            pb.guardrails,
            "",
            "## Notes",
            "",
            pb.notes,
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")

    return pb


if __name__ == "__main__":  # pragma: no cover
    # Simple smoke-test example (no external dependencies).
    sample_stats = {
        "win_rate": 0.62,
        "avg_r_multiple": 1.4,
        "max_drawdown": -0.08,
        "trades_30d": 124,
    }
    pb = write_playbook("ALPHA", sample_stats)
    print(f"Wrote playbook for {pb.strategy} to {PLAYBOOK_DIR}")
