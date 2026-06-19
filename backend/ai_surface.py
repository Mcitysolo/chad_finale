from __future__ import annotations

import functools
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from chad.intel.gpt_client import GPTClient, GPTClientError
from chad.intel.research_engine import run_research_scenario_from_request
from chad.intel.risk_explainer import build_risk_explanation, render_risk_summary_text

# lessons engine exists per repo audit
from chad.intel.lessons_engine import run_lessons_job  # type: ignore

# ------------------------------------------------------------
# Paths (SSOT-aligned artifact locations)
# ------------------------------------------------------------
REPO_DIR = Path(os.environ.get("CHAD_REPO_DIR", "/home/ubuntu/chad_finale")).resolve()
REPORTS_DIR = Path(os.environ.get("CHAD_REPORTS_DIR", str(REPO_DIR / "reports"))).resolve()
LOG_GPT_DIR = Path(os.environ.get("CHAD_GPT_LOG_DIR", str(REPO_DIR / "logs" / "gpt"))).resolve()

REPORTS_OPS_DIR = REPORTS_DIR / "ops"
REPORTS_PORTFOLIO_MEMOS_DIR = REPORTS_DIR / "portfolio_memos"
REPORTS_LESSONS_DIR = REPORTS_DIR / "lessons"

for d in (REPORTS_OPS_DIR, REPORTS_PORTFOLIO_MEMOS_DIR, REPORTS_LESSONS_DIR, LOG_GPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def utc_now_compact() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        pass

def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")

# ------------------------------------------------------------
# Schemas
# ------------------------------------------------------------
class DeskNoteResponse(BaseModel):
    schema_version: str = "desk_note.v1"
    ts_utc: str
    output_path: str
    summary_markdown: str

class PortfolioMemoResponse(BaseModel):
    schema_version: str = "portfolio_memo.v1"
    ts_utc: str
    output_path: str
    summary_markdown: str

class WhyBlockedResponse(BaseModel):
    schema_version: str = "why_blocked_aiwrap.v1"
    ts_utc: str
    risk_explanation: Dict[str, Any] = Field(default_factory=dict)
    summary_text: str

class WeeklyLessonsResponse(BaseModel):
    schema_version: str = "weekly_lessons.v1"
    ts_utc: str
    output_path_json: Optional[str] = None
    output_path_md: Optional[str] = None
    summary_markdown: str

class ResearchRequest(BaseModel):
    symbol: str
    scenario_timeframe: str = "1m"
    question: str

class ResearchResponse(BaseModel):
    schema_version: str = "research_scenario.v1"
    ts_utc: str
    output_path: Optional[str] = None
    payload: Dict[str, Any]

# ------------------------------------------------------------
# Router
# ------------------------------------------------------------
router = APIRouter(prefix="/ai", tags=["ai"])

@functools.lru_cache(maxsize=1)
def _client() -> GPTClient:
    # Loads /etc/chad/openai.env internally (best-effort) and enforces rate limits.
    # L-05 fix: return ONE process-lifetime GPTClient so a single requests.Session
    # (gpt_client.py:238) is reused across requests, instead of leaking a fresh,
    # never-closed keep-alive Session per call (open sockets on the long-lived
    # backend). lru_cache does NOT cache exceptions, so a failed construction
    # (e.g. missing OPENAI_API_KEY) is retried on the next call.
    return GPTClient()

@router.get("", include_in_schema=False)
def ai_root() -> Dict[str, Any]:
    return {
        "ok": True,
        "ts_utc": utc_now_iso(),
        "message": "CHAD AI surface is mounted. See /ai/desk_note, /ai/portfolio_memo, /ai/why_blocked, /ai/weekly_lessons, /ai/research.",
    }

@router.get("/why_blocked", response_model=WhyBlockedResponse)
def ai_why_blocked() -> WhyBlockedResponse:
    exp = build_risk_explanation()
    text = render_risk_summary_text(exp)
    return WhyBlockedResponse(
        ts_utc=utc_now_iso(),
        risk_explanation=exp.model_dump() if hasattr(exp, "model_dump") else exp.dict(),  # pydantic v2/v1 safety
        summary_text=text,
    )

@router.get("/desk_note", response_model=DeskNoteResponse)
def ai_desk_note() -> DeskNoteResponse:
    """
    Phase 10: Daily Desk Note (advisory).
    Uses GPT if configured; otherwise writes a deterministic fallback note.
    Writes: reports/ops/DESK_NOTE_<ts>.md and logs/gpt/desk_note_<ts>.json (metadata only).
    """
    ts = utc_now_iso()
    ts_compact = utc_now_compact()

    # Build safe context (no secrets)
    try:
        exp = build_risk_explanation()
        ctx = render_risk_summary_text(exp)
    except Exception:
        ctx = "Risk summary unavailable."

    md = ""
    meta: Dict[str, Any] = {"ts_utc": ts, "ok": False, "model": None, "error": None}

    try:
        c = _client()
        meta["model"] = getattr(c, "_config", None).model_name if hasattr(c, "_config") else None
        system_prompt = (
            "You are CHAD's Daily Desk Note writer. "
            "Output short operator-grade markdown. "
            "No disclaimers, no meta-talk, no secrets."
        )
        user_prompt = f"Write today's desk note using this risk summary:\n\n{ctx}\n\nFormat as Markdown."
        out = c.chat_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            max_output_tokens=800,
            extra_context={"role": "desk_note"},
        )
        # chat_json expects JSON; if it returned dict with markdown field use it; else fallback
        if isinstance(out, dict) and isinstance(out.get("markdown"), str):
            md = out["markdown"]
        else:
            md = f"# CHAD Desk Note\n\n{ctx}\n\n(Non-GPT fallback: model output did not include markdown field.)"
        meta["ok"] = True
    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        md = f"# CHAD Desk Note\n\n{ctx}\n\n(Non-GPT fallback: OpenAI not available or failed.)"

    out_path = REPORTS_OPS_DIR / f"DESK_NOTE_{ts_compact}.md"
    log_path = LOG_GPT_DIR / f"desk_note_{ts_compact}.json"

    atomic_write_text(out_path, md + ("\n" if not md.endswith("\n") else ""))
    atomic_write_json(log_path, meta)

    return DeskNoteResponse(ts_utc=ts, output_path=str(out_path), summary_markdown=md)

@router.get("/portfolio_memo", response_model=PortfolioMemoResponse)
def ai_portfolio_memo(profile: str = Query(default="BALANCED")) -> PortfolioMemoResponse:
    """
    Phase 10: Portfolio Memo (advisory).
    Uses GPT if configured; otherwise writes a deterministic fallback memo based on latest Phase 9 memo.
    Writes: reports/portfolio_memos/PORTFOLIO_MEMO_<profile>_<ts>.md
    """
    ts = utc_now_iso()
    ts_compact = utc_now_compact()
    prof = (profile or "BALANCED").strip().upper()

    # Use existing Phase 9 deterministic memo as context if present
    existing = sorted(REPORTS_PORTFOLIO_MEMOS_DIR.glob(f"MEMO_{prof}_*.md"))
    ctx = ""
    if existing:
        try:
            ctx = existing[-1].read_text(encoding="utf-8")[:4000]
        except Exception:
            ctx = ""

    md = ""
    try:
        c = _client()
        system_prompt = (
            "You are CHAD's Portfolio Memo writer. "
            "Output short operator-grade markdown. "
            "No disclaimers, no meta-talk."
        )
        user_prompt = (
            f"Write a portfolio memo for profile {prof}.\n\n"
            f"Context (may be empty):\n{ctx}\n\n"
            "Format as Markdown with: Summary, Income Sleeve Notes, Rebalance Notes."
        )
        out = c.chat_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            max_output_tokens=900,
            extra_context={"role": "portfolio_memo", "profile": prof},
        )
        if isinstance(out, dict) and isinstance(out.get("markdown"), str):
            md = out["markdown"]
        else:
            md = f"# CHAD Portfolio Memo — {prof}\n\n{ctx}\n\n(Non-GPT fallback: model output did not include markdown field.)"
    except Exception:
        md = f"# CHAD Portfolio Memo — {prof}\n\n{ctx}\n\n(Non-GPT fallback: OpenAI not available or failed.)"

    out_path = REPORTS_PORTFOLIO_MEMOS_DIR / f"PORTFOLIO_MEMO_{prof}_{ts_compact}.md"
    atomic_write_text(out_path, md + ("\n" if not md.endswith("\n") else ""))

    return PortfolioMemoResponse(ts_utc=ts, output_path=str(out_path), summary_markdown=md)

@router.get("/weekly_lessons", response_model=WeeklyLessonsResponse)
def ai_weekly_lessons(days_back: int = Query(default=7, ge=1, le=30), max_trades: int = Query(default=120, ge=10, le=1000)) -> WeeklyLessonsResponse:
    """
    Phase 10: Weekly lessons learned from recent trades.
    Uses lessons_engine (GPT-backed if available) and writes reports/lessons.
    """
    ts = utc_now_iso()
    try:
        lessons = run_lessons_job(days_back=days_back, max_trades=max_trades)
        # lessons engine writes files itself; we return best-effort pointers if present
        md = getattr(lessons, "summary_markdown", None) or ""
        if not md:
            md = f"# CHAD Lessons\n\n(lessons_engine returned without markdown)\n"
        # Attempt to find newest artifacts
        latest_json = sorted(REPORTS_LESSONS_DIR.glob("LESSONS_*.json"))
        latest_md = sorted(REPORTS_LESSONS_DIR.glob("LESSONS_*.md"))
        return WeeklyLessonsResponse(
            ts_utc=ts,
            output_path_json=str(latest_json[-1]) if latest_json else None,
            output_path_md=str(latest_md[-1]) if latest_md else None,
            summary_markdown=md,
        )
    except Exception as exc:
        md = f"# CHAD Lessons\n\nFailed to generate lessons: {type(exc).__name__}\n"
        return WeeklyLessonsResponse(ts_utc=ts, summary_markdown=md)

@router.post("/research", response_model=ResearchResponse)
def ai_research(req: ResearchRequest) -> ResearchResponse:
    """
    Phase 10: Structured research scenario (advisory-only).
    Delegates to research_engine which validates schema and writes reports/research/.
    """
    ts = utc_now_iso()
    try:
        # Convert to the schema expected by research_engine wrapper
        from chad.intel.schemas import ResearchRequestInput  # local import to keep surface lightweight
        r = ResearchRequestInput(symbol=req.symbol, scenario_timeframe=req.scenario_timeframe, question=req.question)
        scenario = run_research_scenario_from_request(r, client=_client(), save=True)
        payload = scenario.model_dump() if hasattr(scenario, "model_dump") else scenario.dict()

        # Find most recent saved file for this symbol
        rdir = REPORTS_DIR / "research"
        out_path = None
        try:
            candidates = sorted(rdir.glob(f"RESEARCH_{req.symbol.strip().upper()}_*.json"))
            if candidates:
                out_path = str(candidates[-1])
        except Exception:
            out_path = None

        return ResearchResponse(ts_utc=ts, output_path=out_path, payload=payload)
    except GPTClientError as exc:
        raise HTTPException(status_code=503, detail=f"gpt_unavailable: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"research_failed: {type(exc).__name__}") from exc
