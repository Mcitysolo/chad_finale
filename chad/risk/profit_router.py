"""
chad/risk/profit_router.py

Profit Router — 50 / 30 / 20 allocation of realized PnL.

On every profitable close, CHAD routes the realized PnL three ways:

    50% → trading_capital   (stays in the account, compounds trading)
    30% → beta_allocation   (earmarked for Beta — the long-term holder)
    20% → amplifier_allocation
                            (boost for the best-performing strategy)

Advisory only
-------------
CHAD currently runs on a single IBKR paper account, so we don't
physically transfer capital between accounts. Instead, we record the
routing decisions in runtime/profit_routing.json as an accounting
ledger. Beta deployment will later read ``get_beta_accumulated()`` to
know how much earmarked capital is available.

The router never raises — all I/O is wrapped and falls back to in-
memory accounting if the file is unreadable, so a bad routing write
cannot take down the hot path that calls it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROUTING_PATH = REPO_ROOT / "runtime" / "profit_routing.json"

SCHEMA_VERSION = "profit_routing.v1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class RoutingDecision:
    realized_pnl: float
    source_strategy: str
    trading_capital: float
    beta_allocation: float
    amplifier_allocation: float
    routing_timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "realized_pnl": float(self.realized_pnl),
            "source_strategy": str(self.source_strategy),
            "trading_capital": float(self.trading_capital),
            "beta_allocation": float(self.beta_allocation),
            "amplifier_allocation": float(self.amplifier_allocation),
            "routing_timestamp": self.routing_timestamp,
        }


class ProfitRouter:
    """
    Fixed 50/30/20 split. Tunables live on the class so tests can
    exercise the no-routing path without mutating global state.
    """

    REINVEST_PCT = 0.50
    BETA_PCT = 0.30
    AMPLIFIER_PCT = 0.20

    def __init__(
        self,
        routing_path: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.routing_path = Path(routing_path) if routing_path is not None else DEFAULT_ROUTING_PATH
        self.log = logger or LOG

    # ---- Public API -----------------------------------------------------

    def route_profit(
        self,
        realized_pnl: float,
        closing_strategy: str,
        account_equity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute and persist a routing decision for a single realized PnL event.

        Returns a dict describing the routing. If ``realized_pnl <= 0`` the
        return is ``{"no_routing": True, "reason": "not_profitable"}`` and
        nothing is written — we only route winnings.
        """
        try:
            pnl = float(realized_pnl)
        except (TypeError, ValueError):
            return {"no_routing": True, "reason": "invalid_pnl"}

        if pnl <= 0 or pnl != pnl:  # NaN guard
            return {"no_routing": True, "reason": "not_profitable"}

        decision = RoutingDecision(
            realized_pnl=pnl,
            source_strategy=str(closing_strategy or "unknown"),
            trading_capital=pnl * self.REINVEST_PCT,
            beta_allocation=pnl * self.BETA_PCT,
            amplifier_allocation=pnl * self.AMPLIFIER_PCT,
            routing_timestamp=_utc_now_iso(),
        )

        self._append_decision(decision, account_equity=account_equity)
        return decision.to_dict()

    def get_beta_accumulated(self) -> float:
        """Total beta_allocation earmarked across all historical routings."""
        state = self._read_state()
        total = 0.0
        for entry in state.get("decisions", []):
            try:
                total += float(entry.get("beta_allocation") or 0.0)
            except (TypeError, ValueError):
                continue
        return total

    def get_amplifier_accumulated(self) -> float:
        state = self._read_state()
        total = 0.0
        for entry in state.get("decisions", []):
            try:
                total += float(entry.get("amplifier_allocation") or 0.0)
            except (TypeError, ValueError):
                continue
        return total

    def get_totals(self) -> Dict[str, float]:
        state = self._read_state()
        totals = {"trading_capital": 0.0, "beta_allocation": 0.0, "amplifier_allocation": 0.0}
        for entry in state.get("decisions", []):
            for k in totals:
                try:
                    totals[k] += float(entry.get(k) or 0.0)
                except (TypeError, ValueError):
                    continue
        return totals

    def get_beta_remaining(self) -> float:
        """Net beta budget available: accumulated minus already consumed.
        Floored at 0.0 — consumed can never exceed allocated.
        """
        try:
            data = self._read_state()
            allocated = float(
                data.get("totals", {}).get("beta_allocation", 0.0) or 0.0
            )
            consumed = float(data.get("consumed_beta_usd", 0.0) or 0.0)
            return max(0.0, allocated - consumed)
        except Exception:
            return 0.0

    def get_amplifier_remaining(self) -> float:
        """Net amplifier budget available: accumulated minus consumed.
        Floored at 0.0.
        """
        try:
            data = self._read_state()
            allocated = float(
                data.get("totals", {}).get("amplifier_allocation", 0.0) or 0.0
            )
            consumed = float(data.get("consumed_amplifier_usd", 0.0) or 0.0)
            return max(0.0, allocated - consumed)
        except Exception:
            return 0.0

    def mark_beta_consumed(self, amount_usd: float) -> bool:
        """Record that beta_allocation budget was consumed.
        Uses read-modify-write with atomic tmp+replace.
        Returns True on success, False on failure.
        """
        try:
            data = self._read_state()
            prior = float(data.get("consumed_beta_usd", 0.0) or 0.0)
            data["consumed_beta_usd"] = round(prior + float(amount_usd), 10)
            self._write_state(data)
            return True
        except Exception:
            return False

    def mark_amplifier_consumed(self, amount_usd: float) -> bool:
        """Record that amplifier_allocation budget was consumed.
        Uses read-modify-write with atomic tmp+replace.
        Returns True on success, False on failure.
        """
        try:
            data = self._read_state()
            prior = float(data.get("consumed_amplifier_usd", 0.0) or 0.0)
            data["consumed_amplifier_usd"] = round(
                prior + float(amount_usd), 10)
            self._write_state(data)
            return True
        except Exception:
            return False

    # ---- Persistence ----------------------------------------------------

    def _read_state(self) -> Dict[str, Any]:
        empty = {
            "schema_version": SCHEMA_VERSION,
            "decisions": [],
            "consumed_beta_usd": 0.0,
            "consumed_amplifier_usd": 0.0,
        }
        if not self.routing_path.is_file():
            return dict(empty)
        try:
            data = json.loads(self.routing_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.log.warning("profit_router: read failed path=%s err=%s", self.routing_path, exc)
            return dict(empty)
        if not isinstance(data, dict):
            return dict(empty)
        data.setdefault("decisions", [])
        if not isinstance(data["decisions"], list):
            data["decisions"] = []
        data.setdefault("consumed_beta_usd", 0.0)
        data.setdefault("consumed_amplifier_usd", 0.0)
        return data

    def _write_state(self, state: Dict[str, Any]) -> None:
        """Atomic tmp+replace write. Raises OSError on failure so the
        caller can log/handle (mirrors the inline pattern used by
        _append_decision)."""
        self.routing_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.routing_path.with_suffix(self.routing_path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(self.routing_path)

    def _append_decision(
        self,
        decision: RoutingDecision,
        account_equity: Optional[float] = None,
    ) -> None:
        state = self._read_state()
        entry = decision.to_dict()
        if account_equity is not None:
            try:
                entry["account_equity_at_decision"] = float(account_equity)
            except (TypeError, ValueError):
                pass
        decisions: List[Dict[str, Any]] = state.get("decisions") or []
        decisions.append(entry)

        totals = {"trading_capital": 0.0, "beta_allocation": 0.0, "amplifier_allocation": 0.0}
        for d in decisions:
            for k in totals:
                try:
                    totals[k] += float(d.get(k) or 0.0)
                except (TypeError, ValueError):
                    continue

        state.update({
            "schema_version": SCHEMA_VERSION,
            "updated_ts_utc": _utc_now_iso(),
            "decisions": decisions,
            "totals": totals,
        })
        # consumed_* fields are loaded by _read_state and must survive the
        # update() above untouched. setdefault is a belt-and-braces guard.
        state.setdefault("consumed_beta_usd", 0.0)
        state.setdefault("consumed_amplifier_usd", 0.0)

        try:
            self._write_state(state)
        except OSError as exc:
            self.log.warning(
                "profit_router: write failed path=%s err=%s",
                self.routing_path, exc,
            )


__all__ = ["ProfitRouter", "RoutingDecision", "SCHEMA_VERSION", "DEFAULT_ROUTING_PATH"]
