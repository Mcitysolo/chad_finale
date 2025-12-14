from __future__ import annotations

"""
CHAD Phase 10 – Lessons Engine (Post-Trade Teacher)

This module analyzes *recent losing trades* and uses GPT (via GPTClient)
to generate structured LessonsLearned objects plus a Markdown report.

STRICT GUARANTEES:
- READ-ONLY: consumes historical trade logs only.
- NEVER touches brokers, SCR, risk caps, or execution config.
- Advisory-only: writes reports into reports/lessons/.
- All GPT output is validated against LessonsLearned schema.

Typical usage (from a cron job, systemd timer, or manual CLI):

    from chad.intel.lessons_engine import run_lessons_job
    lessons = run_lessons_job(days_back=7, max_trades=100)

Outputs:
- JSON:    reports/lessons/LESSONS_{date}_{range}.json
- Markdown:reports/lessons/LESSONS_{date}_{range}.md
"""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from chad.intel.gpt_client import GPTClient
from chad.intel.schemas import LessonsLearned, LessonItem

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #

TRADES_DIR = Path("/home/ubuntu/CHAD FINALE/data/trades")
REPORTS_DIR = Path("/home/ubuntu/CHAD FINALE/reports/lessons")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


SYSTEM_PROMPT = """
You are CHAD's post-trade teacher.

Your role:
- Read a set of losing trades (already filtered for you).
- Identify recurring PATTERNS and MISTAKES.
- Produce structured, concise lessons for the human operator.

CRITICAL RULES:
- NEVER recommend placing trades.
- NEVER change risk limits or configuration.
- DO NOT give financial advice.
- ONLY produce JSON that matches the provided schema.

Output schema (LessonsLearned):

{
  "date_range":    "string describing window, e.g. '2025-12-01 to 2025-12-07'",
  "top_patterns":  ["short, punchy pattern summaries"],
  "lessons": [
    {
      "issue": "description of a problem or pitfall",
      "example": "short example traded scenario (no sensitive details)",
      "recommendation": "practical rule-of-thumb to avoid it next time"
    }
  ],
  "summary_markdown": "Optional Markdown write-up (<= 1200 words)."
}

Don't use disclaimers or meta comments. Focus on clear, direct, practical
insights and rules-of-thumb.
"""

USER_PROMPT_TEMPLATE = """
Date range: {date_range}

Recent losing trades sample (pre-filtered):
{trades_json}

Your task:
- Extract the BIG recurring patterns.
- Turn them into a small set of lessons (3–10).
- Produce JSON that matches LessonsLearned exactly.
"""


@dataclass(frozen=True)
class LosingTradeSample:
    """
    Lightweight representation of a losing trade for GPT consumption.
    We avoid dumping every field to control token usage.
    """

    symbol: str
    strategy: str
    pnl: float
    opened_at: Optional[str]
    closed_at: Optional[str]
    meta: Dict[str, Any]


# --------------------------------------------------------------------------- #
# Core helpers: trade collection & filtering
# --------------------------------------------------------------------------- #

def _parse_trade_line(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _is_losing_trade(tr: Dict[str, Any]) -> bool:
    pnl = tr.get("pnl")
    try:
        pnl_val = float(pnl)
    except (TypeError, ValueError):
        return False
    return pnl_val < 0.0


def _extract_sample(tr: Dict[str, Any]) -> LosingTradeSample:
    symbol = str(tr.get("symbol", "UNKNOWN"))
    strategy = str(tr.get("strategy", "UNKNOWN"))
    pnl_val = float(tr.get("pnl", 0.0))

    opened_at = tr.get("opened_at") or tr.get("entry_at") or None
    closed_at = tr.get("closed_at") or tr.get("exit_at") or None

    meta: Dict[str, Any] = {}
    for key in ("reason", "tags", "timeframe", "context", "notes"):
        if key in tr:
            meta[key] = tr[key]

    return LosingTradeSample(
        symbol=symbol,
        strategy=strategy,
        pnl=pnl_val,
        opened_at=opened_at,
        closed_at=closed_at,
        meta=meta,
    )


def _collect_recent_losing_trades(
    *,
    days_back: int,
    max_trades: int,
) -> tuple[List[LosingTradeSample], str]:
    """
    Collect recent losing trades from NDJSON files in TRADES_DIR.

    The function:
    - Looks at trade_history_*.ndjson files.
    - Filters to trades whose 'pnl' < 0.
    - Restricts to trades closed within [now - days_back, now].
    - Returns up to max_trades samples.

    Returns:
        (samples, date_range_str)
    """
    now = datetime.now(timezone.utc)
    start_dt = now - timedelta(days=days_back)
    date_range_str = f"{start_dt.date().isoformat()} to {now.date().isoformat()}"

    if not TRADES_DIR.is_dir():
        return [], date_range_str

    # Collect all trade_history_* files, newest first.
    files = sorted(
        [p for p in TRADES_DIR.glob("trade_history_*.ndjson") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    samples: List[LosingTradeSample] = []

    for path in files:
        if len(samples) >= max_trades:
            break

        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if len(samples) >= max_trades:
                        break

                    tr = _parse_trade_line(line)
                    if not tr:
                        continue
                    if not _is_losing_trade(tr):
                        continue

                    # If a closed_at/exit_at timestamp exists, we can filter by time.
                    closed_raw = tr.get("closed_at") or tr.get("exit_at")
                    if isinstance(closed_raw, str):
                        try:
                            closed_dt = datetime.fromisoformat(
                                closed_raw.replace("Z", "+00:00")
                            )
                            if closed_dt < start_dt:
                                # too old
                                continue
                        except ValueError:
                            # ignore bad timestamps; still include trade
                            pass

                    samples.append(_extract_sample(tr))
        except OSError:
            # Skip unreadable files silently; robustness over perfection.
            continue

    return samples, date_range_str


# --------------------------------------------------------------------------- #
# GPT + LessonsLearned construction
# --------------------------------------------------------------------------- #

def run_lessons_job(
    *,
    days_back: int = 7,
    max_trades: int = 100,
    client: Optional[GPTClient] = None,
    save: bool = True,
) -> LessonsLearned:
    """
    Run the Lessons Engine job:

    - Collects recent losing trades.
    - Calls GPT (via GPTClient) with structured prompts.
    - Validates JSON response against LessonsLearned schema.
    - Optionally saves JSON + Markdown to reports/lessons/.

    Returns:
        LessonsLearned instance.

    Raises:
        GPTClientError, ValueError on schema failures.
    """
    if client is None:
        client = GPTClient()

    trades, date_range = _collect_recent_losing_trades(
        days_back=days_back,
        max_trades=max_trades,
    )

    # If there are no losing trades, we still produce a lesson object that says so.
    if not trades:
        lessons = LessonsLearned(
            date_range=date_range,
            top_patterns=["No losing trades in the selected window."],
            lessons=[
                LessonItem(
                    issue="No recent losing trades",
                    example="All trades in this window had pnl >= 0.",
                    recommendation=(
                        "Keep logging trades and re-run the lessons job after "
                        "you have a more representative set of winners and losers."
                    ),
                )
            ],
            summary_markdown=(
                f"# CHAD Lessons ({date_range})\n\n"
                "No losing trades were detected in the specified period. "
                "Re-run this report later when the sample includes some losers "
                "to extract meaningful lessons."
            ),
        )
        if save:
            _save_lessons(lessons)
        return lessons

    # Prepare a lean JSON sample for GPT (avoid dumping everything).
    sample_for_gpt: List[Dict[str, Any]] = []
    for s in trades:
        sample_for_gpt.append(
            {
                "symbol": s.symbol,
                "strategy": s.strategy,
                "pnl": s.pnl,
                "opened_at": s.opened_at,
                "closed_at": s.closed_at,
                "meta": s.meta,
            }
        )

    user_prompt = USER_PROMPT_TEMPLATE.format(
        date_range=date_range,
        trades_json=json.dumps(sample_for_gpt, indent=2),
    )

    raw_json = client.chat_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.2,
        max_output_tokens=1200,
        extra_context={"role": "lessons_engine"},
    )

    try:
        lessons = LessonsLearned(**raw_json)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"GPT output did not match LessonsLearned schema: {exc}") from exc

    if save:
        _save_lessons(lessons)

    return lessons


def _save_lessons(lessons: LessonsLearned) -> None:
    """
    Save LessonsLearned as both JSON and Markdown under reports/lessons/.
    """
    utc_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    slug = lessons.date_range.replace(" ", "_").replace(":", "-")
    slug = "".join(ch for ch in slug if ch.isalnum() or ch in ("_", "-"))

    json_path = REPORTS_DIR / f"LESSONS_{utc_date}_{slug}.json"
    md_path = REPORTS_DIR / f"LESSONS_{utc_date}_{slug}.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(lessons.dict(), f, indent=2)

    # If GPT did not provide a summary_markdown, synthesize a minimal one.
    md_content = lessons.summary_markdown or _render_lessons_markdown(lessons)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"[LESSONS_ENGINE] Saved JSON -> {json_path}")
    print(f"[LESSONS_ENGINE] Saved Markdown -> {md_path}")


def _render_lessons_markdown(lessons: LessonsLearned) -> str:
    """
    Fallback Markdown renderer if GPT doesn't supply summary_markdown.
    """
    lines: List[str] = []
    lines.append(f"# CHAD Lessons ({lessons.date_range})")
    lines.append("")
    if lessons.top_patterns:
        lines.append("## Top Patterns")
        for p in lessons.top_patterns:
            lines.append(f"- {p}")
        lines.append("")

    if lessons.lessons:
        lines.append("## Detailed Lessons")
        for idx, item in enumerate(lessons.lessons, start=1):
            lines.append(f"### Lesson {idx}: {item.issue}")
            if item.example:
                lines.append("")
                lines.append(f"**Example:** {item.example}")
            if item.recommendation:
                lines.append("")
                lines.append(f"**Recommendation:** {item.recommendation}")
            lines.append("")

    return "\n".join(lines)
