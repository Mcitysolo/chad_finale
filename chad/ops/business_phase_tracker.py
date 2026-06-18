#!/usr/bin/env python3
"""
BusinessPhaseTracker — explicit BUILD/GROW/PAY phase publisher.

Reads withdrawal_authorization (which already classifies the phase)
and enriches it with plain-English context, growth metrics, and the
"next milestone" so the operator and the Telegram/dashboard surfaces
have a single, coherent business view.

INPUTS:
  - runtime/withdrawal_authorization.json (authoritative phase)
  - runtime/portfolio_snapshot.json (current equity)
  - runtime/scr_state.json
  - runtime/equity_history.ndjson (for growth metrics)
  - runtime/tier_state.json (optional, for context)

OUTPUT:
  - runtime/business_phase.json
    {
      "schema_version": "business_phase.v1",
      "phase": "BUILD" | "GROW" | "PAY",
      "phase_description": str,
      "current_equity_usd": float,
      "seed_capital_usd": float,
      "growth_pct_from_seed": float,
      "days_in_phase": int,
      "next_phase_requirement": str,
      "compound_metrics": {
          "total_return_pct": float,
          "annualized_return_pct": float,
          "days_active": int,
          "high_water_mark_usd": float
      },
      "ts_utc": ISO-8601
    }

PHASE 12C per SSOT v8.2 roadmap.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

LOG = logging.getLogger("chad.ops.business_phase_tracker")

REPO_ROOT = Path("/home/ubuntu/chad_finale")
RUNTIME_DIR = REPO_ROOT / "runtime"

WITHDRAWAL_PATH = RUNTIME_DIR / "withdrawal_authorization.json"
SNAPSHOT_PATH = RUNTIME_DIR / "portfolio_snapshot.json"
SCR_PATH = RUNTIME_DIR / "scr_state.json"
HISTORY_PATH = RUNTIME_DIR / "equity_history.ndjson"
TIER_STATE_PATH = RUNTIME_DIR / "tier_state.json"
OUT_PATH = RUNTIME_DIR / "business_phase.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOG.warning("read_failed path=%s err=%s", path, exc)
        return {}


def _read_history() -> List[Dict[str, Any]]:
    if not HISTORY_PATH.is_file():
        return []
    out: List[Dict[str, Any]] = []
    try:
        for line in HISTORY_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    except Exception as exc:
        LOG.warning("history_read_failed: %s", exc)
    return out


def _parse_iso(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def _phase_description(
    phase: str,
    seed: float,
    build_threshold: float,
    authorized_salary: float,
    scr_state: str,
) -> str:
    if phase == "BUILD":
        return (
            f"Building the engine. Equity must grow above ${build_threshold:,.0f} "
            f"before paying anything out."
        )
    if phase == "GROW":
        return (
            "Engine is built. Now growing the account before salary starts. "
            "SCR must reach CONFIDENT first."
        )
    if phase == "PAY":
        return (
            f"Engine running well. Salary authorized at ${authorized_salary:,.0f}/month "
            f"from surplus above high water mark."
        )
    return f"Unknown phase: {phase}"


def _next_phase_requirement(
    phase: str,
    current_equity: float,
    build_threshold: float,
    scr_state: str,
    hwm: float,
    history_days: int,
    min_history_days: int,
) -> str:
    if phase == "BUILD":
        gap = max(0.0, build_threshold - current_equity)
        return (
            f"Reach ${build_threshold:,.0f} equity to enter GROW phase "
            f"(need +${gap:,.0f} more)."
        )
    if phase == "GROW":
        items = []
        if scr_state.upper() != "CONFIDENT":
            items.append(f"SCR must promote from {scr_state} to CONFIDENT")
        if history_days < min_history_days:
            items.append(
                f"Need {min_history_days - history_days}+ more days of equity history"
            )
        if current_equity < hwm:
            items.append(f"Recover above high water mark ${hwm:,.0f}")
        if not items:
            return "Hold equity above HWM until SCR clears, then PAY phase unlocks."
        return "To enter PAY phase: " + "; ".join(items) + "."
    if phase == "PAY":
        return (
            "Maintain equity above high water mark with no >5% drawdown "
            "to keep salary authorized."
        )
    return ""


def _days_in_phase(history: List[Dict[str, Any]], current_phase: str) -> int:
    """Return actual calendar days since phase was entered.

    Reads phase_entered_utc from withdrawal_authorization.json if present;
    falls back to len(history) for compatibility.
    """
    try:
        if WITHDRAWAL_PATH.is_file():
            wa = json.loads(WITHDRAWAL_PATH.read_text(encoding="utf-8"))
            entered = wa.get("phase_entered_utc", "")
            if entered:
                dt = datetime.fromisoformat(
                    str(entered).replace("Z", "+00:00")
                )
                delta = datetime.now(timezone.utc) - dt
                return max(0, delta.days)
    except Exception:
        pass
    # Fallback: row count (prior behavior)
    if not history:
        return 0
    return len(history)


def _compound_metrics(
    history: List[Dict[str, Any]],
    seed: float,
    current_equity: float,
    hwm: float,
) -> Dict[str, Any]:
    """Return total/annualized return + days_active + hwm."""
    total_return_pct = 0.0
    if seed > 0:
        total_return_pct = (current_equity - seed) / seed * 100.0

    # days_active = span between first record and now (or 0)
    days_active = 0
    if history:
        first_ts = _parse_iso(history[0].get("ts_utc", ""))
        if first_ts is not None:
            delta = _utc_now() - first_ts
            days_active = max(1, delta.days)

    annualized = 0.0
    # Annualized CAGR is meaningless with very little history; require
    # at least 14 days before publishing it.
    if days_active >= 14 and seed > 0 and current_equity > 0:
        try:
            ratio = current_equity / seed
            if ratio > 0:
                annualized = (ratio ** (365.0 / days_active) - 1.0) * 100.0
                # Sanity clamp to keep absurd extrapolations off the dashboard.
                annualized = max(-99.0, min(annualized, 1000.0))
        except (ZeroDivisionError, ValueError, OverflowError):
            annualized = 0.0

    return {
        "total_return_pct": round(total_return_pct, 2),
        "annualized_return_pct": round(annualized, 2),
        "days_active": days_active,
        "high_water_mark_usd": round(hwm, 2),
    }


def build_payload() -> Optional[Dict[str, Any]]:
    withdrawal = _read_json(WITHDRAWAL_PATH)
    snap = _read_json(SNAPSHOT_PATH)
    scr = _read_json(SCR_PATH)
    history = _read_history()

    # The business view is denominated in USD (seed, salary, and HWM are all
    # USD). It is driven EXCLUSIVELY by the authoritative USD equity
    # (portfolio_snapshot_publisher: total_equity_usd_authoritative + usd_ok),
    # mirroring chad/risk/tier_manager.py. FAIL-CLOSED: if usd_ok is false — or
    # the field is absent / not numeric (FX unavailable, stale snapshot) — HOLD
    # the last published business_phase.json (return None; main() does NOT
    # republish). We must NEVER fall back to the CAD component sum or a null USD.
    usd_ok = bool(snap.get("usd_ok", False))
    total_usd = snap.get("total_equity_usd_authoritative")
    if not (usd_ok and isinstance(total_usd, (int, float))):
        LOG.warning(
            "BUSINESS_PHASE_HELD_NO_USD_RATE usd_ok=%s total_usd=%s "
            "(no authoritative USD equity; holding prior business_phase, CAD never used)",
            usd_ok, total_usd,
        )
        return None
    current_equity = float(total_usd)

    seed = float(withdrawal.get("seed_capital_usd", 50000.0))
    hwm = float(withdrawal.get("high_water_mark_usd", current_equity))
    phase = str(withdrawal.get("phase", "BUILD")).upper()
    authorized_salary = float(withdrawal.get("authorized_withdrawal_usd", 0.0))
    scr_state = str(scr.get("state", withdrawal.get("scr_state", "UNKNOWN")))

    # Build threshold: seed * 1.20 (mirrors withdrawal_manager default).
    build_threshold = seed * 1.20

    # min_history_days: we don't read the policy file directly; use 14 (default).
    min_history_days = 14

    growth_pct = (current_equity - seed) / seed * 100.0 if seed > 0 else 0.0

    description = _phase_description(
        phase, seed, build_threshold, authorized_salary, scr_state
    )
    next_req = _next_phase_requirement(
        phase, current_equity, build_threshold, scr_state, hwm,
        len(history), min_history_days,
    )
    days_in_phase = _days_in_phase(history, phase)
    metrics = _compound_metrics(history, seed, current_equity, hwm)

    return {
        "schema_version": "business_phase.v1",
        "phase": phase,
        "phase_description": description,
        "current_equity_usd": round(current_equity, 2),
        "current_equity_currency": "USD",
        "current_equity_currency_ok": True,
        "seed_capital_usd": round(seed, 2),
        "growth_pct_from_seed": round(growth_pct, 2),
        "days_in_phase": days_in_phase,
        "next_phase_requirement": next_req,
        "compound_metrics": metrics,
        "ts_utc": _utc_now_iso(),
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    payload = build_payload()
    if payload is None:
        # FAIL-CLOSED / HOLD-PRIOR (mirror tier_manager): no authoritative USD
        # equity available — leave the last published business_phase.json intact.
        LOG.warning("business_phase_held — no authoritative USD equity; prior file kept")
        return 0

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(OUT_PATH)

    LOG.info(
        "business_phase_published phase=%s equity=$%.2f growth=%.2f%% next=%s",
        payload["phase"], payload["current_equity_usd"],
        payload["growth_pct_from_seed"], payload["next_phase_requirement"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
