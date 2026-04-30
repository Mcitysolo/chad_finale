#!/usr/bin/env python3
"""
WithdrawalManager — the "CHAD as a business" salary engine.

Computes how much profit (if any) is safe to withdraw as operator
salary while protecting the seed capital and growth trajectory.

CORE PRINCIPLE — DO NOT EAT THE SEED.
The trading account is the engine. Pay yourself only from sustained
surplus above a rolling high-water mark, only when SCR is confident,
and never during drawdowns. The withdrawal authorization is advisory
only — CHAD never moves money. The operator decides.

INPUTS:
  - runtime/equity_history.ndjson  (daily snapshots)
  - runtime/scr_state.json         (must be CONFIDENT for withdrawal)
  - runtime/portfolio_snapshot.json (current equity)
  - config/withdrawal_policy.json  (operator-tuned rules)

OUTPUT:
  - runtime/withdrawal_authorization.json
    {
      "phase": "BUILD" | "GROW" | "PAY",
      "current_equity_usd": float,
      "high_water_mark_usd": float,
      "drawdown_from_hwm_pct": float,
      "spendable_surplus_usd": float,
      "authorized_withdrawal_usd": float,
      "reason": "human-readable explanation",
      "ts_utc": ISO-8601
    }

PHASES:
  BUILD — equity below seed * 1.20 → withdraw zero, focus on building
  GROW  — equity above seed * 1.20 but SCR not CONFIDENT → reinvest
  PAY   — SCR CONFIDENT, above HWM, no recent drawdown → salary OK

WITHDRAWAL FORMULA (PAY phase only):
  surplus = current_equity - high_water_mark
  authorized = min(surplus * payout_rate, max_monthly_salary)
  + drawdown veto: if last 30d max_drawdown > 5%, withdraw zero
  + SCR veto: if state != CONFIDENT, withdraw zero
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

LOG = logging.getLogger("chad.risk.withdrawal_manager")

REPO_ROOT = Path("/home/ubuntu/chad_finale")
RUNTIME_DIR = REPO_ROOT / "runtime"
CONFIG_DIR = REPO_ROOT / "config"

EQUITY_HISTORY_PATH = RUNTIME_DIR / "equity_history.ndjson"
SCR_PATH = RUNTIME_DIR / "scr_state.json"
SNAPSHOT_PATH = RUNTIME_DIR / "portfolio_snapshot.json"
POLICY_PATH = CONFIG_DIR / "withdrawal_policy.json"
OUT_PATH = RUNTIME_DIR / "withdrawal_authorization.json"


# Default policy if config/withdrawal_policy.json is absent
DEFAULT_POLICY: Dict[str, Any] = {
    "schema_version": "withdrawal_policy.v1",
    "seed_capital_usd": 50000.0,
    "build_phase_threshold_multiplier": 1.20,
    "payout_rate_above_hwm": 0.30,
    "max_monthly_salary_usd": 2000.0,
    "drawdown_veto_pct": 5.0,
    "drawdown_lookback_days": 30,
    "require_scr_confident": True,
    "minimum_history_days": 14,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOG.warning("read_failed path=%s err=%s", path, exc)
        return {}


def _read_equity_history() -> List[Dict[str, Any]]:
    if not EQUITY_HISTORY_PATH.is_file():
        return []
    out: List[Dict[str, Any]] = []
    try:
        for line in EQUITY_HISTORY_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    except Exception as exc:
        LOG.warning("history_read_failed: %s", exc)
    return out


def _load_policy() -> Dict[str, Any]:
    on_disk = _read_json(POLICY_PATH)
    if not on_disk:
        return dict(DEFAULT_POLICY)
    merged = dict(DEFAULT_POLICY)
    merged.update(on_disk)
    return merged


@dataclass
class WithdrawalAuthorization:
    phase: str
    current_equity_usd: float
    seed_capital_usd: float
    high_water_mark_usd: float
    drawdown_from_hwm_pct: float
    spendable_surplus_usd: float
    authorized_withdrawal_usd: float
    scr_state: str
    history_days: int
    reason: str
    ts_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_authorization(
    current_equity: float,
    history: List[Dict[str, Any]],
    scr_state: str,
    policy: Dict[str, Any],
) -> WithdrawalAuthorization:
    """Pure function: given inputs, return the authorization decision."""
    seed = float(policy["seed_capital_usd"])
    build_mult = float(policy["build_phase_threshold_multiplier"])
    payout_rate = float(policy["payout_rate_above_hwm"])
    max_salary = float(policy["max_monthly_salary_usd"])
    dd_veto = float(policy["drawdown_veto_pct"])
    dd_lookback = int(policy["drawdown_lookback_days"])
    require_confident = bool(policy["require_scr_confident"])
    min_history = int(policy["minimum_history_days"])

    # Compute high water mark from history
    if history:
        equities = [float(r.get("total_equity_usd", 0.0)) for r in history]
        hwm = max(equities)
    else:
        hwm = current_equity

    # Compute drawdown over lookback window
    cutoff = datetime.now(timezone.utc) - timedelta(days=dd_lookback)
    recent = []
    for r in history:
        try:
            ts = datetime.fromisoformat(r["ts_utc"].replace("Z", "+00:00"))
            if ts > cutoff:
                recent.append(float(r["total_equity_usd"]))
        except Exception:
            continue
    recent.append(current_equity)
    if recent:
        recent_peak = max(recent)
        drawdown_pct = (recent_peak - current_equity) / recent_peak * 100 if recent_peak > 0 else 0.0
    else:
        drawdown_pct = 0.0

    drawdown_from_hwm = (hwm - current_equity) / hwm * 100 if hwm > 0 else 0.0

    # PHASE DETERMINATION
    build_threshold = seed * build_mult
    if current_equity < build_threshold:
        phase = "BUILD"
    elif require_confident and scr_state.upper() != "CONFIDENT":
        phase = "GROW"
    else:
        phase = "PAY"

    # WITHDRAWAL LOGIC
    authorized = 0.0
    reasons = []

    if phase == "BUILD":
        reasons.append(
            f"BUILD phase: equity ${current_equity:,.0f} below "
            f"build threshold ${build_threshold:,.0f} (seed ${seed:,.0f} × {build_mult})"
        )
    elif phase == "GROW":
        reasons.append(
            f"GROW phase: equity above build threshold but SCR is "
            f"{scr_state} (need CONFIDENT). Reinvesting profits."
        )
    else:  # PAY phase
        # Insufficient history?
        if len(history) < min_history:
            phase = "GROW"
            reasons.append(
                f"GROW phase override: only {len(history)} days of equity history "
                f"(need {min_history}+). Building track record before paying."
            )
        # Drawdown veto?
        elif drawdown_pct > dd_veto:
            phase = "GROW"
            reasons.append(
                f"GROW phase override: {dd_lookback}d drawdown {drawdown_pct:.1f}% "
                f"exceeds veto threshold {dd_veto}%. No salary during drawdown."
            )
        # Below HWM?
        elif current_equity < hwm:
            phase = "GROW"
            reasons.append(
                f"GROW phase override: equity ${current_equity:,.0f} below "
                f"high water mark ${hwm:,.0f}. Recover before paying."
            )
        else:
            surplus = current_equity - hwm
            authorized = min(surplus * payout_rate, max_salary)
            reasons.append(
                f"PAY phase: equity ${current_equity:,.0f} at high water mark "
                f"${hwm:,.0f}. Surplus ${surplus:,.0f} × {payout_rate*100:.0f}% "
                f"payout rate = ${authorized:,.0f} authorized (capped at "
                f"${max_salary:,.0f}/month)."
            )
            if authorized > 0 and phase == "PAY":
                try:
                    from chad.utils.telegram_notify import notify
                    notify(
                        f"💰 SALARY AUTHORIZED — ${authorized:.2f}/month\n"
                        f"Equity: ${current_equity:,.2f} | HWM: ${hwm:,.2f}",
                        severity="info",
                        dedupe_key="salary_authorized",
                    )
                except Exception:
                    pass

    return WithdrawalAuthorization(
        phase=phase,
        current_equity_usd=current_equity,
        seed_capital_usd=seed,
        high_water_mark_usd=hwm,
        drawdown_from_hwm_pct=drawdown_from_hwm,
        spendable_surplus_usd=max(0.0, current_equity - hwm),
        authorized_withdrawal_usd=authorized,
        scr_state=scr_state,
        history_days=len(history),
        reason=" | ".join(reasons),
        ts_utc=_utc_now_iso(),
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    snap = _read_json(SNAPSHOT_PATH)
    if not snap:
        LOG.error("portfolio_snapshot_missing — withdrawal manager cannot run")
        return 1
    current_equity = (
        float(snap.get("ibkr_equity", 0.0))
        + float(snap.get("kraken_equity", 0.0))
        + float(snap.get("coinbase_equity", 0.0))
    )

    history = _read_equity_history()

    scr_data = _read_json(SCR_PATH)
    scr_state = str(scr_data.get("state", "UNKNOWN")).upper()

    policy = _load_policy()

    auth = compute_authorization(current_equity, history, scr_state, policy)

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(auth.to_dict(), indent=2), encoding="utf-8")
    tmp.replace(OUT_PATH)

    LOG.info(
        "withdrawal_authorization phase=%s authorized=$%.2f hwm=$%.2f current=$%.2f reason=%s",
        auth.phase, auth.authorized_withdrawal_usd, auth.high_water_mark_usd,
        auth.current_equity_usd, auth.reason,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
