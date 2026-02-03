from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# =========================
# Time helpers
# =========================

def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# =========================
# Safe parsing helpers
# =========================

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        f = float(x)
        if f != f or f in (float("inf"), float("-inf")):
            return default
        return f
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_str(x: Any, default: str = "") -> str:
    try:
        return str(x)
    except Exception:
        return default


def _read_json(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    try:
        if not path.is_file():
            return None, f"missing:{path}"
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"parse_error:{path.name}:{type(exc).__name__}:{exc}"


def _walk_dicts(x: Any) -> Iterable[dict]:
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from _walk_dicts(v)
    elif isinstance(x, list):
        for v in x:
            yield from _walk_dicts(v)


# =========================
# Paths (SSOT)
# =========================

@dataclass(frozen=True)
class EnginePaths:
    repo_dir: Path
    runtime_dir: Path
    config_dir: Path

    @staticmethod
    def from_env() -> "EnginePaths":
        repo_dir = Path(os.environ.get("CHAD_REPO_DIR", "/home/ubuntu/chad_finale")).resolve()
        runtime_dir = Path(os.environ.get("CHAD_RUNTIME_DIR", str(repo_dir / "runtime"))).resolve()
        config_dir = Path(os.environ.get("CHAD_CONFIG_DIR", str(repo_dir / "config"))).resolve()
        return EnginePaths(repo_dir=repo_dir, runtime_dir=runtime_dir, config_dir=config_dir)


# =========================
# Domain models
# =========================

@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float
    currency: str
    exchange: str
    sec_type: str
    market_price: float
    market_value: float
    avg_cost: float

    def notional_proxy(self) -> float:
        """
        Deterministic, read-only notional proxy.

        Priority:
          1) market_value (if present)
          2) abs(qty) * market_price (if price present)
          3) abs(qty) * avg_cost (if avg_cost present)
          4) abs(qty) (fallback)
        """
        mv = float(self.market_value or 0.0)
        if mv > 0.0:
            return mv

        q = abs(float(self.qty or 0.0))
        if q <= 0.0:
            return 0.0

        mp = float(self.market_price or 0.0)
        if mp > 0.0:
            return q * mp

        avg = float(self.avg_cost or 0.0)
        if avg > 0.0:
            return q * avg

        return q


@dataclass(frozen=True)
class Target:
    symbol: str
    weight: float


# =========================
# Portfolio engine (read-only)
# =========================

class PortfolioEngine:
    """
    Read-only portfolio engine (SSOT compliance):
      - Active: derived from runtime/positions_snapshot.json
      - Targets: derived from config/portfolio_profiles.json, deterministic fallback
      - Rebalance latest: computed on-demand (no writes, no broker calls)
    """

    def __init__(self, paths: Optional[EnginePaths] = None) -> None:
        self.paths = paths or EnginePaths.from_env()

    # ---------- Active portfolio ----------

    def load_active_positions(self) -> Dict[str, Any]:
        p = self.paths.runtime_dir / "positions_snapshot.json"
        obj, err = _read_json(p)
        if err:
            return {"ok": False, "error": err, "ts_utc": utc_now_iso(), "positions": []}

        positions: List[Position] = []

        # Heuristic: find dicts that look like positions
        for d in _walk_dicts(obj):
            sym = _safe_str(d.get("symbol") or d.get("ticker") or "").strip().upper()
            if not sym:
                continue

            qty = _safe_float(
                d.get("qty") if "qty" in d else d.get("position") if "position" in d else d.get("quantity"),
                0.0,
            )
            if qty == 0.0:
                continue

            currency = _safe_str(d.get("currency") or "USD").strip().upper() or "USD"
            exchange = _safe_str(d.get("exchange") or d.get("primaryExchange") or "SMART").strip().upper() or "SMART"
            sec_type = _safe_str(d.get("sec_type") or d.get("secType") or d.get("asset_class") or "STK").strip().upper() or "STK"

            mp = _safe_float(d.get("marketPrice") if "marketPrice" in d else d.get("market_price"), 0.0)
            mv = _safe_float(d.get("marketValue") if "marketValue" in d else d.get("market_value"), 0.0)
            avg = _safe_float(
                d.get("avg_cost")
                if "avg_cost" in d
                else d.get("averageCost")
                if "averageCost" in d
                else d.get("avgCost"),
                0.0,
            )

            positions.append(
                Position(
                    symbol=sym,
                    qty=float(qty),
                    currency=currency,
                    exchange=exchange,
                    sec_type=sec_type,
                    market_price=float(mp),
                    market_value=float(mv),
                    avg_cost=float(avg),
                )
            )

        # Deduplicate by symbol (sum qty; keep best notional/price hints)
        by: Dict[str, Position] = {}
        for pos in positions:
            if pos.symbol not in by:
                by[pos.symbol] = pos
            else:
                prior = by[pos.symbol]
                qty2 = prior.qty + pos.qty
                mp2 = prior.market_price if prior.market_price > 0 else pos.market_price
                mv2 = prior.market_value + pos.market_value
                avg2 = prior.avg_cost if prior.avg_cost > 0 else pos.avg_cost
                by[pos.symbol] = Position(
                    symbol=pos.symbol,
                    qty=qty2,
                    currency=prior.currency or pos.currency,
                    exchange=prior.exchange or pos.exchange,
                    sec_type=prior.sec_type or pos.sec_type,
                    market_price=mp2,
                    market_value=mv2,
                    avg_cost=avg2,
                )

        out_positions: List[dict] = []
        total_notional = 0.0
        for sym in sorted(by.keys()):
            pos = by[sym]
            n = pos.notional_proxy()
            total_notional += n
            out_positions.append(
                {
                    "symbol": pos.symbol,
                    "qty": pos.qty,
                    "currency": pos.currency,
                    "exchange": pos.exchange,
                    "sec_type": pos.sec_type,
                    "market_price": pos.market_price,
                    "market_value": pos.market_value,
                    "avg_cost": pos.avg_cost,
                    "notional_proxy": n,
                }
            )

        return {
            "ok": True,
            "ts_utc": utc_now_iso(),
            "source_path": str(p),
            "positions": out_positions,
            "total_notional_proxy": total_notional,
        }

    # ---------- Targets ----------

    def _load_profiles(self) -> Tuple[Optional[dict], Optional[str]]:
        p = self.paths.config_dir / "portfolio_profiles.json"
        obj, err = _read_json(p)
        if err:
            return None, err
        if not isinstance(obj, dict):
            return None, "profiles_invalid_shape"
        return obj, None

    def _fallback_tickers(self) -> List[str]:
        # 1) config/universe.json
        u = self.paths.config_dir / "universe.json"
        obj, err = _read_json(u)
        tickers: List[str] = []
        if not err and isinstance(obj, dict):
            for key in ("tickers", "symbols", "universe"):
                v = obj.get(key)
                if isinstance(v, list):
                    tickers = [str(x).strip().upper() for x in v if str(x).strip()]
                    break

        # 2) runtime/full_execution_cycle_last.json summary.tick_symbols
        if not tickers:
            p = self.paths.runtime_dir / "full_execution_cycle_last.json"
            plan, perr = _read_json(p)
            if not perr and isinstance(plan, dict):
                summ = plan.get("summary") or {}
                if isinstance(summ, dict) and isinstance(summ.get("tick_symbols"), list):
                    tickers = [str(x).strip().upper() for x in summ.get("tick_symbols") if str(x).strip()]

        # Dedup + stable order
        out: List[str] = []
        seen = set()
        for t in tickers:
            if not t or t in seen:
                continue
            seen.add(t)
            out.append(t)
        return out

    def get_targets(self, profile: str) -> Dict[str, Any]:
        profile_key = _safe_str(profile).strip().upper()
        cfg, err = self._load_profiles()
        if err:
            return {"ok": False, "error": err, "ts_utc": utc_now_iso(), "profile": profile_key, "targets": []}

        profiles = cfg.get("profiles") if isinstance(cfg.get("profiles"), dict) else {}
        prof = profiles.get(profile_key) if isinstance(profiles, dict) else None
        if not isinstance(prof, dict):
            return {"ok": False, "error": f"unknown_profile:{profile_key}", "ts_utc": utc_now_iso(), "profile": profile_key, "targets": []}

        max_symbols = _safe_int(prof.get("max_symbols"), 25)
        targets_raw = prof.get("targets")

        targets: List[Target] = []

        # explicit targets
        if isinstance(targets_raw, list) and targets_raw:
            for t in targets_raw:
                if not isinstance(t, dict):
                    continue
                sym = _safe_str(t.get("symbol")).strip().upper()
                w = _safe_float(t.get("weight"), 0.0)
                if sym and w > 0:
                    targets.append(Target(symbol=sym, weight=w))

        # fallback targets (equal weights)
        if not targets:
            tickers = self._fallback_tickers()[: max(0, max_symbols)]
            if tickers:
                w = 1.0 / float(len(tickers))
                targets = [Target(symbol=s, weight=w) for s in tickers]

        # normalize weights
        ssum = sum(t.weight for t in targets)
        norm: List[dict] = []
        if ssum > 0:
            for t in targets:
                norm.append({"symbol": t.symbol, "weight": float(t.weight / ssum)})
        else:
            norm = []

        return {
            "ok": True,
            "ts_utc": utc_now_iso(),
            "schema_version": "portfolio_targets.v1",
            "profile": profile_key,
            "max_symbols": max_symbols,
            "targets": norm,
            "source": {
                "profiles_path": str(self.paths.config_dir / "portfolio_profiles.json"),
                "fallback_universe_path": str(self.paths.config_dir / "universe.json"),
                "fallback_plan_path": str(self.paths.runtime_dir / "full_execution_cycle_last.json"),
            },
        }

    # ---------- Rebalance (read-only, computed on demand) ----------

    def get_rebalance_latest(self, profile: str) -> Dict[str, Any]:
        active = self.load_active_positions()
        targets = self.get_targets(profile)

        if not active.get("ok"):
            return {"ok": False, "error": f"active_positions_error:{active.get(error)}", "ts_utc": utc_now_iso()}
        if not targets.get("ok"):
            return {"ok": False, "error": f"targets_error:{targets.get(error)}", "ts_utc": utc_now_iso()}

        pos_list = active.get("positions") or []
        total = _safe_float(active.get("total_notional_proxy"), 0.0)

        current_w: Dict[str, float] = {}
        for p in pos_list:
            if not isinstance(p, dict):
                continue
            sym = _safe_str(p.get("symbol")).strip().upper()
            n = _safe_float(p.get("notional_proxy"), 0.0)
            if sym and n > 0 and total > 0:
                current_w[sym] = current_w.get(sym, 0.0) + (n / total)

        target_w: Dict[str, float] = {}
        for t in targets.get("targets") or []:
            if not isinstance(t, dict):
                continue
            sym = _safe_str(t.get("symbol")).strip().upper()
            w = _safe_float(t.get("weight"), 0.0)
            if sym and w >= 0:
                target_w[sym] = target_w.get(sym, 0.0) + w

        # union symbols
        all_syms = sorted(set(current_w.keys()) | set(target_w.keys()))

        diffs: List[dict] = []
        for sym in all_syms:
            cw = float(current_w.get(sym, 0.0))
            tw = float(target_w.get(sym, 0.0))
            diffs.append(
                {
                    "symbol": sym,
                    "current_weight": cw,
                    "target_weight": tw,
                    "delta_weight": float(tw - cw),
                }
            )

        return {
            "ok": True,
            "ts_utc": utc_now_iso(),
            "schema_version": "rebalance_latest.v1",
            "profile": _safe_str(profile).strip().upper(),
            "active": {
                "source_path": active.get("source_path"),
                "total_notional_proxy": float(active.get("total_notional_proxy") or 0.0),
                "positions_count": int(len(pos_list)),
            },
            "diffs": diffs,
            "notes": [
                "Read-only on-demand rebalance preview.",
                "Uses notional_proxy derived from positions_snapshot.json (marketValue preferred, else qty*marketPrice, else qty*avg_cost, else qty).",
                "No broker calls; no orders; no state writes.",
            ],
        }
