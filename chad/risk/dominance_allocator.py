from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

ROOT = Path("/home/ubuntu/chad_finale")
DATA_TRADES = ROOT / "data" / "trades"
RUNTIME = ROOT / "runtime"
OUT_PATH = RUNTIME / "dominance_allocator.json"

PLACEHOLDERS = {"paper_exec", "unknown", "manual", ""}

DEFAULT_BASE_WEIGHTS: Dict[str, float] = {
    "alpha": 0.10,
    "beta_trend": 0.30,
    "gamma": 0.25,
    "delta": 0.20,
    "omega": 0.10,
    "alpha_crypto": 0.03,
    "alpha_forex": 0.02,
}

MIN_WEIGHT = 0.02
MAX_WEIGHT = 0.60

WEIGHT_RETURN = 0.45
WEIGHT_WINRATE = 0.20
WEIGHT_PROFIT_FACTOR = 0.15
WEIGHT_ACTIVITY = 0.10
WEIGHT_DRAWDOWN = -0.10

LOOKBACK_DAYS = 14
MIN_TRADES_FOR_CONFIDENCE = 3


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_z(ts: Optional[datetime] = None) -> str:
    value = ts or utc_now()
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        s = value.strip()
        return s if s else default
    s = str(value).strip()
    return s if s else default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def normalize_strategy(value: Any) -> str:
    return safe_str(value, "").strip().lower()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def daterange_strings(days: int) -> List[str]:
    now = utc_now()
    return [
        (now - timedelta(days=i)).strftime("%Y%m%d")
        for i in range(days)
    ]


@dataclass(slots=True)
class TradeRow:
    strategy: str
    pnl: float
    ts_utc: str
    symbol: str
    side: str
    notional: float


@dataclass(slots=True)
class StrategyStats:
    strategy: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    total_pnl: float = 0.0
    avg_notional: float = 0.0
    max_drawdown_like: float = 0.0
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        win_rate = self.wins / self.trades if self.trades else 0.0
        profit_factor = (
            self.gross_profit / abs(self.gross_loss)
            if self.gross_loss < 0.0
            else (999.0 if self.gross_profit > 0 else 0.0)
        )
        return {
            "strategy": self.strategy,
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(win_rate, 6),
            "gross_profit": round(self.gross_profit, 6),
            "gross_loss": round(self.gross_loss, 6),
            "profit_factor": round(profit_factor, 6),
            "total_pnl": round(self.total_pnl, 6),
            "avg_notional": round(self.avg_notional, 6),
            "max_drawdown_like": round(self.max_drawdown_like, 6),
            "score": round(self.score, 6),
        }


class TradeLedgerReader:
    def __init__(self, trades_dir: Path = DATA_TRADES, lookback_days: int = LOOKBACK_DAYS) -> None:
        self.trades_dir = trades_dir
        self.lookback_days = lookback_days

    def read(self) -> List[TradeRow]:
        rows: List[TradeRow] = []
        for ymd in daterange_strings(self.lookback_days):
            path = self.trades_dir / f"trade_history_{ymd}.ndjson"
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                payload = obj.get("payload", obj)
                strategy = normalize_strategy(payload.get("strategy"))
                if strategy in PLACEHOLDERS:
                    continue
                rows.append(
                    TradeRow(
                        strategy=strategy,
                        pnl=safe_float(payload.get("pnl"), 0.0),
                        ts_utc=safe_str(obj.get("timestamp_utc") or payload.get("exit_time_utc"), ""),
                        symbol=safe_str(payload.get("symbol"), ""),
                        side=safe_str(payload.get("side"), ""),
                        notional=safe_float(payload.get("notional"), 0.0),
                    )
                )
        return rows


class DominanceScorer:
    def __init__(self, base_weights: Optional[Dict[str, float]] = None) -> None:
        self.base_weights = dict(base_weights or DEFAULT_BASE_WEIGHTS)

    def compute_stats(self, rows: Iterable[TradeRow]) -> Dict[str, StrategyStats]:
        grouped: Dict[str, List[TradeRow]] = {}
        for row in rows:
            grouped.setdefault(row.strategy, []).append(row)

        stats: Dict[str, StrategyStats] = {}
        for strategy, items in grouped.items():
            st = StrategyStats(strategy=strategy)
            running = 0.0
            peak = 0.0
            notional_sum = 0.0

            for row in items:
                pnl = row.pnl
                st.trades += 1
                st.total_pnl += pnl
                notional_sum += row.notional

                if pnl > 0:
                    st.wins += 1
                    st.gross_profit += pnl
                elif pnl < 0:
                    st.losses += 1
                    st.gross_loss += pnl

                running += pnl
                peak = max(peak, running)
                dd = running - peak
                st.max_drawdown_like = min(st.max_drawdown_like, dd)

            st.avg_notional = notional_sum / st.trades if st.trades else 0.0
            stats[strategy] = st

        for strategy in self.base_weights:
            stats.setdefault(strategy, StrategyStats(strategy=strategy))

        for st in stats.values():
            st.score = self._score(st)

        return stats

    def _score(self, st: StrategyStats) -> float:
        if st.trades == 0:
            return 0.0

        win_rate = st.wins / st.trades if st.trades else 0.0
        profit_factor = (
            st.gross_profit / abs(st.gross_loss)
            if st.gross_loss < 0.0
            else (3.0 if st.gross_profit > 0 else 0.0)
        )

        normalized_return = math.tanh(st.total_pnl / 250.0)
        normalized_win = max(0.0, min(1.0, win_rate))
        normalized_pf = max(0.0, min(3.0, profit_factor)) / 3.0
        normalized_activity = min(1.0, st.trades / 10.0)
        normalized_dd = min(1.0, abs(st.max_drawdown_like) / 150.0)

        confidence = min(1.0, st.trades / max(1, MIN_TRADES_FOR_CONFIDENCE))

        raw = (
            WEIGHT_RETURN * normalized_return
            + WEIGHT_WINRATE * normalized_win
            + WEIGHT_PROFIT_FACTOR * normalized_pf
            + WEIGHT_ACTIVITY * normalized_activity
            + WEIGHT_DRAWDOWN * normalized_dd
        )

        return max(0.0, raw * confidence)

    def allocate(self, stats: Mapping[str, StrategyStats]) -> Dict[str, float]:
        adjusted: Dict[str, float] = {}

        for strategy, base in self.base_weights.items():
            score = stats[strategy].score if strategy in stats else 0.0
            # blend base + dominance multiplier
            adjusted[strategy] = base * (1.0 + score * 2.5)

        total = sum(adjusted.values())
        if total <= 0:
            return dict(self.base_weights)

        normalized = {k: v / total for k, v in adjusted.items()}

        # clamp
        clamped = {
            k: min(MAX_WEIGHT, max(MIN_WEIGHT, v))
            for k, v in normalized.items()
        }

        # renormalize after clamp
        total2 = sum(clamped.values())
        final = {k: v / total2 for k, v in clamped.items()}
        return final


def main() -> int:
    reader = TradeLedgerReader()
    rows = reader.read()

    scorer = DominanceScorer()
    stats = scorer.compute_stats(rows)
    weights = scorer.allocate(stats)

    payload = {
        "ts_utc": iso_z(),
        "lookback_days": LOOKBACK_DAYS,
        "valid_trade_rows": len(rows),
        "base_weights": DEFAULT_BASE_WEIGHTS,
        "dominance_weights": weights,
        "strategy_stats": {
            k: v.to_dict()
            for k, v in sorted(stats.items(), key=lambda kv: (-kv[1].score, kv[0]))
        },
        "ranking": [
            k for k, _ in sorted(stats.items(), key=lambda kv: (-kv[1].score, kv[0]))
        ],
    }

    atomic_write_json(OUT_PATH, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
