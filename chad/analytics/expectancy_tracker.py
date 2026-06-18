"""Per-strategy expectancy tracker.

Reads data/trades/trade_history_*.ndjson, skips untrusted/pre-rebuild
trades, and writes runtime/expectancy_state.json with per-strategy
win rate, avg win/loss, expectancy, and a plain-English status.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

from chad.analytics.futures_classifier import is_futures_row

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRADES_GLOB = os.path.join(REPO_ROOT, "data", "trades", "trade_history_*.ndjson")
PNL_STATE = os.path.join(REPO_ROOT, "runtime", "pnl_state.json")
DYNAMIC_CAPS = os.path.join(REPO_ROOT, "runtime", "dynamic_caps.json")
OUTPUT_PATH = os.path.join(REPO_ROOT, "runtime", "expectancy_state.json")


def _status(total: int, win_rate: float, expectancy: float) -> str:
    if total < 10:
        return "new"
    if win_rate >= 0.55 and expectancy > 0:
        return "performing"
    if win_rate >= 0.45 and expectancy > 0:
        return "watch"
    return "underperforming"


def _iter_trades():
    # Quarantine awareness: skip records explicitly listed in
    # runtime/quarantine_manifest_*.json plus any fill_id flagged
    # pnl_untrusted in data/fills/FILLS_*.ndjson, so polluted trades
    # — including derived closed trades referencing untrusted fills —
    # cannot re-enter expectancy_state -> winner_scaling/strategy_health.
    try:
        from chad.utils.quarantine import get_exclusion_sets
        invalid_fill_ids, invalid_trade_hashes = get_exclusion_sets(
            runtime_dir=os.path.join(REPO_ROOT, "runtime"),
            fills_dir=os.path.join(REPO_ROOT, "data", "fills"),
        )
    except Exception:
        invalid_fill_ids, invalid_trade_hashes = set(), set()

    for path in sorted(glob.glob(TRADES_GLOB)):
        if ".scr_reset_bak" in path or path.endswith(".bak"):
            continue
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    rh = rec.get("record_hash") if isinstance(rec, dict) else None
                    if isinstance(rh, str) and rh in invalid_trade_hashes:
                        continue
                    payload = rec.get("payload", rec)
                    if not isinstance(payload, dict):
                        continue
                    fid = payload.get("fill_id")
                    if isinstance(fid, str) and fid in invalid_fill_ids:
                        continue
                    fids = payload.get("fill_ids")
                    if isinstance(fids, list) and any(
                        isinstance(f, str) and f in invalid_fill_ids for f in fids
                    ):
                        continue
                    if payload.get("pnl_untrusted") is True:
                        continue
                    if payload.get("historical_pre_rebuild") is True:
                        continue
                    tags = payload.get("tags") or []
                    if isinstance(tags, list) and (
                        "pnl_untrusted" in tags or "historical_pre_rebuild" in tags
                    ):
                        continue
                    # item 5b: drop futures rows so expectancy_state.json agrees
                    # with the SCR effective sample (same classifier). Bug-B
                    # futures contamination must not feed winner_scaling /
                    # strategy_health via per-strategy expectancy.
                    _extra = payload.get("extra") or {}
                    _sec_type = _extra.get("secType") if isinstance(_extra, dict) else None
                    if is_futures_row(payload.get("symbol"), _sec_type):
                        continue
                    yield payload
        except FileNotFoundError:
            continue


def _load_json(path: str) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def compute() -> dict[str, Any]:
    per: dict[str, dict[str, Any]] = {}
    total_clean = 0
    total_pnl = 0.0

    for p in _iter_trades():
        strat = p.get("strategy") or "unknown"
        pnl = p.get("pnl")
        try:
            pnl = float(pnl)
        except (TypeError, ValueError):
            continue
        if strat == "unknown":
            continue
        s = per.setdefault(
            strat,
            {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_pnl_sum": 0.0,
                "loss_pnl_sum": 0.0,
                "total_pnl": 0.0,
                "best_trade": None,
                "worst_trade": None,
            },
        )
        s["total_trades"] += 1
        s["total_pnl"] += pnl
        if pnl > 0:
            s["wins"] += 1
            s["win_pnl_sum"] += pnl
        elif pnl < 0:
            s["losses"] += 1
            s["loss_pnl_sum"] += pnl
        if s["best_trade"] is None or pnl > s["best_trade"]:
            s["best_trade"] = pnl
        if s["worst_trade"] is None or pnl < s["worst_trade"]:
            s["worst_trade"] = pnl
        total_clean += 1
        total_pnl += pnl

    caps = _load_json(DYNAMIC_CAPS)
    weights = caps.get("normalized_weights") or {}

    strategies: dict[str, Any] = {}
    for strat, s in per.items():
        total = s["total_trades"]
        wins = s["wins"]
        losses = s["losses"]
        decided = wins + losses
        win_rate = (wins / decided) if decided else 0.0
        avg_win = (s["win_pnl_sum"] / wins) if wins else 0.0
        avg_loss = (s["loss_pnl_sum"] / losses) if losses else 0.0
        expectancy = (win_rate * avg_win) + ((1.0 - win_rate) * avg_loss)
        alloc = weights.get(strat)
        strategies[strat] = {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 3),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "expectancy": round(expectancy, 2),
            "total_pnl": round(s["total_pnl"], 2),
            "best_trade": round(s["best_trade"] or 0.0, 2),
            "worst_trade": round(s["worst_trade"] or 0.0, 2),
            "current_allocation_pct": round(alloc * 100.0, 1) if isinstance(alloc, (int, float)) else None,
            "status": _status(total, win_rate, expectancy),
        }

    ranked = [
        (name, st["expectancy"], st["total_trades"])
        for name, st in strategies.items()
        if st["total_trades"] >= 5
    ]
    top = max(ranked, key=lambda x: x[1])[0] if ranked else None
    worst = min(ranked, key=lambda x: x[1])[0] if ranked else None

    pnl_state = _load_json(PNL_STATE)
    account_equity = pnl_state.get("account_equity")
    realized = pnl_state.get("realized_pnl", round(total_pnl, 2))

    return {
        "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "account_equity": account_equity,
        "realized_pnl_total": realized,
        "strategies": strategies,
        "top_performer": top,
        "worst_performer": worst,
        "total_clean_trades": total_clean,
    }


def write_state(state: dict[str, Any], path: str = OUTPUT_PATH) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def main() -> int:
    state = compute()
    write_state(state)
    json.dump(state, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
