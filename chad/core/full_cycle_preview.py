#!/usr/bin/env python3
"""
chad/core/full_cycle_preview.py

Production-grade, read-only preview + observability harness for CHAD.

Purpose
-------
This module is the canonical operator-facing "what would CHAD do right now?"
tool, without sending broker orders.

It is intentionally:
- read-only
- deterministic
- defensive
- compatible with partial runtime state
- useful both for current live brains and newly added engines like alpha_futures

Key capabilities
----------------
1) Strategy registry inspection
2) Latest runtime artifact inspection:
   - runtime/full_execution_cycle_last.json
   - runtime/dynamic_caps.json
3) Execution plan summary
4) IBKR preview summary
5) Optional deterministic futures smoke test using synthetic bars
   to validate the alpha_futures engine without requiring a live futures feed

Notes
-----
- This script does NOT place broker orders.
- This script does NOT mutate runtime artifacts unless explicitly asked via --write-json.
- This script is safe to run repeatedly.

Recommended usage
-----------------
cd /home/ubuntu/chad_finale
source venv/bin/activate
python3 -m chad.core.full_cycle_preview
python3 -m chad.core.full_cycle_preview --with-synthetic-futures
python3 -m chad.core.full_cycle_preview --json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from chad.strategies import iter_strategy_registrations
from chad.types import (
    AssetClass,
    LegendConsensus,
    MarketContext,
    MarketTick,
    PortfolioSnapshot,
    Position,
    StrategyConfig,
)


# ============================================================================
# Constants / paths
# ============================================================================

REPO_ROOT = Path("/home/ubuntu/chad_finale")
RUNTIME_DIR = REPO_ROOT / "runtime"
FULL_CYCLE_ARTIFACT = RUNTIME_DIR / "full_execution_cycle_last.json"
DYNAMIC_CAPS_ARTIFACT = RUNTIME_DIR / "dynamic_caps.json"
DEFAULT_OUTPUT_JSON = RUNTIME_DIR / "full_cycle_preview_latest.json"

LOGGER = logging.getLogger("chad.full_cycle_preview")


# ============================================================================
# Generic helpers
# ============================================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_z(ts: Optional[datetime] = None) -> str:
    value = ts or utc_now()
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        out = value.strip()
        return out if out else default
    out = str(value).strip()
    return out if out else default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def read_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        LOGGER.exception("failed_read_json path=%s", path)
        return default


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def json_ready(value: Any) -> Any:
    """
    Best-effort conversion into JSON-safe shapes.
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    try:
        return asdict(value)  # dataclass
    except Exception:
        pass
    if hasattr(value, "__dict__"):
        return {str(k): json_ready(v) for k, v in vars(value).items()}
    return repr(value)


# ============================================================================
# Registry inspection
# ============================================================================

@dataclass(slots=True)
class StrategyRegistrationRow:
    name: str
    enabled: bool
    target_universe_count: int
    target_universe_sample: List[str]
    max_gross_exposure: Optional[float]
    notes: str


class StrategyRegistryInspector:
    def inspect(self) -> List[StrategyRegistrationRow]:
        rows: List[StrategyRegistrationRow] = []
        for reg in iter_strategy_registrations():
            config = reg.build_config()
            universe = list(config.target_universe or [])
            rows.append(
                StrategyRegistrationRow(
                    name=safe_str(getattr(config.name, "value", config.name)),
                    enabled=bool(config.enabled),
                    target_universe_count=len(universe),
                    target_universe_sample=universe[:8],
                    max_gross_exposure=config.max_gross_exposure,
                    notes=safe_str(config.notes),
                )
            )
        return rows


# ============================================================================
# Artifact inspection
# ============================================================================

@dataclass(slots=True)
class OrderRow:
    strategy: str
    symbol: str
    side: str
    size: float
    price: float
    notional: float
    asset_class: str
    contributors: List[str]


@dataclass(slots=True)
class IntentRow:
    strategy: str
    symbol: str
    side: str
    sec_type: str
    exchange: str
    currency: str
    order_type: str
    quantity: float
    notional_estimate: float


@dataclass(slots=True)
class FullCycleSummary:
    artifact_exists: bool
    top_keys: List[str]
    counts: Dict[str, int]
    orders_count: int
    total_notional: float
    orders: List[OrderRow]
    intents: List[IntentRow]
    futures_orders_count: int


@dataclass(slots=True)
class DynamicCapsSummary:
    artifact_exists: bool
    ts_utc: str
    total_equity: float
    portfolio_risk_cap: float
    normalized_weights: Dict[str, float]
    strategy_caps: Dict[str, float]


class ArtifactRepository:
    def __init__(self, *, full_cycle_path: Path, dynamic_caps_path: Path) -> None:
        self.full_cycle_path = full_cycle_path
        self.dynamic_caps_path = dynamic_caps_path

    def load_full_cycle_summary(self) -> FullCycleSummary:
        raw = read_json(self.full_cycle_path, {})
        if not isinstance(raw, Mapping) or not raw:
            return FullCycleSummary(
                artifact_exists=False,
                top_keys=[],
                counts={},
                orders_count=0,
                total_notional=0.0,
                orders=[],
                intents=[],
                futures_orders_count=0,
            )

        counts_raw = raw.get("counts") if isinstance(raw.get("counts"), Mapping) else {}
        counts = {str(k): safe_int(v) for k, v in counts_raw.items()}

        orders_raw = raw.get("orders")
        if not isinstance(orders_raw, list):
            orders_raw = []

        intents_raw = raw.get("ibkr_intents")
        if not isinstance(intents_raw, list):
            intents_raw = []

        orders: List[OrderRow] = []
        futures_orders_count = 0
        total_notional = 0.0

        for obj in orders_raw:
            if not isinstance(obj, Mapping):
                continue
            asset_class = safe_str(obj.get("asset_class"), "").lower()
            if asset_class == "futures":
                futures_orders_count += 1

            notional = safe_float(obj.get("notional"), 0.0)
            total_notional += notional

            contributors_raw = obj.get("contributors") if isinstance(obj.get("contributors"), list) else []
            contributors = [safe_str(v) for v in contributors_raw if safe_str(v)]

            orders.append(
                OrderRow(
                    strategy=safe_str(obj.get("primary_strategy") or obj.get("strategy")),
                    symbol=safe_str(obj.get("symbol")),
                    side=safe_str(obj.get("side")),
                    size=safe_float(obj.get("size")),
                    price=safe_float(obj.get("price")),
                    notional=notional,
                    asset_class=asset_class,
                    contributors=contributors,
                )
            )

        intents: List[IntentRow] = []
        for obj in intents_raw:
            if not isinstance(obj, Mapping):
                continue
            intents.append(
                IntentRow(
                    strategy=safe_str(obj.get("strategy")),
                    symbol=safe_str(obj.get("symbol")),
                    side=safe_str(obj.get("side")),
                    sec_type=safe_str(obj.get("sec_type")),
                    exchange=safe_str(obj.get("exchange")),
                    currency=safe_str(obj.get("currency")),
                    order_type=safe_str(obj.get("order_type")),
                    quantity=safe_float(obj.get("quantity")),
                    notional_estimate=safe_float(obj.get("notional_estimate")),
                )
            )

        return FullCycleSummary(
            artifact_exists=True,
            top_keys=sorted(str(k) for k in raw.keys()),
            counts=counts,
            orders_count=len(orders),
            total_notional=total_notional,
            orders=orders,
            intents=intents,
            futures_orders_count=futures_orders_count,
        )

    def load_dynamic_caps_summary(self) -> DynamicCapsSummary:
        raw = read_json(self.dynamic_caps_path, {})
        if not isinstance(raw, Mapping) or not raw:
            return DynamicCapsSummary(
                artifact_exists=False,
                ts_utc="",
                total_equity=0.0,
                portfolio_risk_cap=0.0,
                normalized_weights={},
                strategy_caps={},
            )

        nw = raw.get("normalized_weights") if isinstance(raw.get("normalized_weights"), Mapping) else {}
        caps = raw.get("strategy_caps") if isinstance(raw.get("strategy_caps"), Mapping) else {}

        return DynamicCapsSummary(
            artifact_exists=True,
            ts_utc=safe_str(raw.get("ts_utc")),
            total_equity=safe_float(raw.get("total_equity")),
            portfolio_risk_cap=safe_float(raw.get("portfolio_risk_cap")),
            normalized_weights={str(k): safe_float(v) for k, v in nw.items()},
            strategy_caps={str(k): safe_float(v) for k, v in caps.items()},
        )


# ============================================================================
# Futures smoke test
# ============================================================================

@dataclass(slots=True)
class SmokeTestSignalRow:
    strategy: str
    symbol: str
    side: str
    size: float
    confidence: float
    asset_class: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FuturesSmokeTestResult:
    executed: bool
    mode: str
    ok: bool
    signal_count: int
    signals: List[SmokeTestSignalRow]
    error: str = ""


class FuturesSmokeTester:
    """
    Deterministic alpha_futures validator.

    This does not need live futures data. It builds a synthetic MarketContext
    with stable, trending futures bars to validate:
    - enum/type wiring
    - registry wiring
    - handler execution
    - signal construction
    """

    def __init__(self) -> None:
        # late imports so normal preview still works even if alpha_futures is absent
        from chad.strategies.alpha_futures import alpha_futures_handler
        self.alpha_futures_handler = alpha_futures_handler

    def run(self, *, synthetic: bool) -> FuturesSmokeTestResult:
        if not synthetic:
            return FuturesSmokeTestResult(
                executed=False,
                mode="disabled",
                ok=True,
                signal_count=0,
                signals=[],
            )

        try:
            ctx = self._build_synthetic_context()
            raw_signals = self.alpha_futures_handler(ctx) or []

            out: List[SmokeTestSignalRow] = []
            for sig in raw_signals:
                meta = getattr(sig, "meta", {}) if isinstance(getattr(sig, "meta", {}), Mapping) else {}
                out.append(
                    SmokeTestSignalRow(
                        strategy=safe_str(getattr(getattr(sig, "strategy", None), "value", getattr(sig, "strategy", ""))),
                        symbol=safe_str(getattr(sig, "symbol", "")),
                        side=safe_str(getattr(getattr(sig, "side", None), "value", getattr(sig, "side", ""))),
                        size=safe_float(getattr(sig, "size", 0.0)),
                        confidence=safe_float(getattr(sig, "confidence", 0.0)),
                        asset_class=safe_str(getattr(getattr(sig, "asset_class", None), "value", getattr(sig, "asset_class", ""))),
                        meta={str(k): json_ready(v) for k, v in dict(meta).items()},
                    )
                )

            return FuturesSmokeTestResult(
                executed=True,
                mode="synthetic",
                ok=True,
                signal_count=len(out),
                signals=out,
            )
        except Exception as exc:
            return FuturesSmokeTestResult(
                executed=True,
                mode="synthetic",
                ok=False,
                signal_count=0,
                signals=[],
                error=f"{type(exc).__name__}: {exc}",
            )

    def _build_synthetic_context(self) -> MarketContext:
        now = utc_now()

        def mk_tick(symbol: str, price: float) -> MarketTick:
            return MarketTick(
                symbol=symbol,
                price=price,
                size=1.0,
                exchange=None,
                timestamp=now,
                source="synthetic_preview",
            )

        ticks = {
            "MES": mk_tick("MES", 5200.0),
            "MNQ": mk_tick("MNQ", 18350.0),
            "MCL": mk_tick("MCL", 77.0),
            "MGC": mk_tick("MGC", 2190.0),
        }

        bars = {
            "MES": self._trend_bars(start=5050.0, step=3.2, pullback_every=9, low_wobble=2.5, high_wobble=3.5),
            "MNQ": self._trend_bars(start=17800.0, step=7.0, pullback_every=7, low_wobble=5.0, high_wobble=7.5),
            "MCL": self._trend_bars(start=72.0, step=0.14, pullback_every=6, low_wobble=0.18, high_wobble=0.21),
            "MGC": self._trend_bars(start=2140.0, step=1.9, pullback_every=8, low_wobble=1.7, high_wobble=2.1),
        }

        spread_bps = {k: 2.0 for k in ticks}
        liquidity_usd = {
            "MES": 25_000_000.0,
            "MNQ": 18_000_000.0,
            "MCL": 8_500_000.0,
            "MGC": 7_500_000.0,
        }

        portfolio = PortfolioSnapshot(
            timestamp=now,
            cash=250_000.0,
            positions={},
            extra={"equity": 250_000.0, "net_liq": 250_000.0},
        )

        return MarketContext(
            now=now,
            ticks=ticks,
            legend=LegendConsensus(as_of=now, weights={}),
            portfolio=portfolio,
            bars=bars,
            spread_bps=spread_bps,
            liquidity_usd=liquidity_usd,
            volume_usd=None,
            dollar_volume=None,
            volatility=None,
        )

    def _trend_bars(
        self,
        *,
        start: float,
        step: float,
        pullback_every: int,
        low_wobble: float,
        high_wobble: float,
        length: int = 80,
    ) -> List[Dict[str, Any]]:
        """
        Deterministic trend series.
        No randomness: stable across runs.
        """
        bars: List[Dict[str, Any]] = []
        px = start
        ts0 = utc_now() - timedelta(minutes=length)

        for i in range(length):
            drift = step
            if i % pullback_every == 0 and i > 0:
                drift *= -0.35

            open_px = px
            close_px = max(0.01, px + drift)
            high_px = max(open_px, close_px) + high_wobble
            low_px = min(open_px, close_px) - low_wobble
            volume = 1000.0 + (i * 7.0)

            bars.append(
                {
                    "open": round(open_px, 6),
                    "high": round(high_px, 6),
                    "low": round(low_px, 6),
                    "close": round(close_px, 6),
                    "volume": round(volume, 6),
                    "ts_utc": (ts0 + timedelta(minutes=i)).isoformat().replace("+00:00", "Z"),
                }
            )
            px = close_px

        return bars


# ============================================================================
# Rendering / output
# ============================================================================

class PreviewRenderer:
    def __init__(self, *, use_json: bool) -> None:
        self.use_json = use_json

    def render(self, payload: Mapping[str, Any]) -> str:
        if self.use_json:
            return json.dumps(json_ready(payload), indent=2, sort_keys=True)

        lines: List[str] = []
        runtime = payload.get("runtime", {})
        registry = payload.get("strategy_registry", [])
        full_cycle = payload.get("full_cycle", {})
        caps = payload.get("dynamic_caps", {})
        futures = payload.get("futures_smoke_test", {})

        lines.append("=== CHAD Full Cycle Preview (Read-Only) ===")
        lines.append(f"now: {runtime.get('now_utc', '')}")
        lines.append(f"python: {runtime.get('python_version', '')}")
        lines.append(f"platform: {runtime.get('platform', '')}")
        lines.append("")

        lines.append("--- Strategy Registry ---")
        lines.append(f"registered_strategies: {len(registry)}")
        for row in registry:
            name = row.get("name", "")
            enabled = row.get("enabled", False)
            tuc = row.get("target_universe_count", 0)
            mge = row.get("max_gross_exposure", None)
            lines.append(
                f"  strategy: {name} enabled={enabled} "
                f"target_universe_count={tuc} max_gross_exposure={mge}"
            )
        lines.append("")

        lines.append("--- Latest Runtime Artifact ---")
        lines.append(f"artifact_exists: {full_cycle.get('artifact_exists', False)}")
        lines.append(f"top_keys: {full_cycle.get('top_keys', [])}")
        lines.append(f"counts: {full_cycle.get('counts', {})}")
        lines.append("")

        lines.append("--- ExecutionPlan ---")
        lines.append(f"orders_count:   {full_cycle.get('orders_count', 0)}")
        lines.append(f"total_notional: {safe_float(full_cycle.get('total_notional', 0.0)):.2f}")
        lines.append(f"futures_orders: {full_cycle.get('futures_orders_count', 0)}")
        for o in full_cycle.get("orders", []):
            lines.append(
                "  order: "
                f"strategy={o.get('strategy')} "
                f"symbol={o.get('symbol')} "
                f"side={o.get('side')} "
                f"size={o.get('size')} "
                f"price={o.get('price'):.4f} "
                f"notional={o.get('notional'):.2f} "
                f"asset_class={o.get('asset_class')} "
                f"contributors={o.get('contributors')}"
            )
        lines.append("")

        lines.append("--- IBKR StrategyTradeIntents (preview only) ---")
        intents = full_cycle.get("intents", [])
        lines.append(f"intents_count: {len(intents)}")
        for i in intents:
            lines.append(
                "  intent: "
                f"strategy={i.get('strategy')} "
                f"symbol={i.get('symbol')} "
                f"side={i.get('side')} "
                f"sec_type={i.get('sec_type')} "
                f"exchange={i.get('exchange')} "
                f"currency={i.get('currency')} "
                f"order_type={i.get('order_type')} "
                f"quantity={i.get('quantity')} "
                f"notional_estimate={i.get('notional_estimate'):.2f}"
            )
        lines.append("")

        lines.append("--- Dynamic Caps ---")
        lines.append(f"artifact_exists: {caps.get('artifact_exists', False)}")
        lines.append(f"ts_utc: {caps.get('ts_utc', '')}")
        lines.append(f"total_equity: {safe_float(caps.get('total_equity', 0.0)):.2f}")
        lines.append(f"portfolio_risk_cap: {safe_float(caps.get('portfolio_risk_cap', 0.0)):.2f}")
        lines.append(f"normalized_weights: {caps.get('normalized_weights', {})}")
        lines.append("")

        lines.append("--- Alpha Futures Smoke Test ---")
        lines.append(f"executed: {futures.get('executed', False)}")
        lines.append(f"mode: {futures.get('mode', 'disabled')}")
        lines.append(f"ok: {futures.get('ok', False)}")
        lines.append(f"signal_count: {futures.get('signal_count', 0)}")
        if futures.get("error"):
            lines.append(f"error: {futures.get('error')}")
        for sig in futures.get("signals", []):
            lines.append(
                "  futures_signal: "
                f"strategy={sig.get('strategy')} "
                f"symbol={sig.get('symbol')} "
                f"side={sig.get('side')} "
                f"size={sig.get('size')} "
                f"confidence={sig.get('confidence')} "
                f"asset_class={sig.get('asset_class')}"
            )

        lines.append("")
        lines.append("[full_cycle_preview] NOTE: No broker calls were made. This is a logical preview only.")
        return "\n".join(lines)


# ============================================================================
# App
# ============================================================================

@dataclass(slots=True)
class PreviewAppConfig:
    full_cycle_path: Path = FULL_CYCLE_ARTIFACT
    dynamic_caps_path: Path = DYNAMIC_CAPS_ARTIFACT
    write_json_path: Optional[Path] = None
    use_json: bool = False
    with_synthetic_futures: bool = False


class PreviewApp:
    def __init__(
        self,
        *,
        cfg: PreviewAppConfig,
        artifact_repo: ArtifactRepository,
        registry_inspector: StrategyRegistryInspector,
        renderer: PreviewRenderer,
    ) -> None:
        self.cfg = cfg
        self.artifact_repo = artifact_repo
        self.registry_inspector = registry_inspector
        self.renderer = renderer

    def run(self) -> int:
        registry_rows = [json_ready(r) for r in self.registry_inspector.inspect()]
        full_cycle = json_ready(self.artifact_repo.load_full_cycle_summary())
        dynamic_caps = json_ready(self.artifact_repo.load_dynamic_caps_summary())

        futures_smoke_test = self._run_futures_smoke_test()

        payload: Dict[str, Any] = {
            "runtime": {
                "now_utc": iso_z(),
                "repo_root": str(REPO_ROOT),
                "python_executable": sys.executable,
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
            },
            "strategy_registry": registry_rows,
            "full_cycle": full_cycle,
            "dynamic_caps": dynamic_caps,
            "futures_smoke_test": futures_smoke_test,
        }

        text = self.renderer.render(payload)
        print(text)

        if self.cfg.write_json_path is not None:
            atomic_write_json(self.cfg.write_json_path, json_ready(payload))

        return 0

    def _run_futures_smoke_test(self) -> Dict[str, Any]:
        try:
            tester = FuturesSmokeTester()
            result = tester.run(synthetic=self.cfg.with_synthetic_futures)
            return json_ready(result)
        except Exception as exc:
            return {
                "executed": self.cfg.with_synthetic_futures,
                "mode": "synthetic" if self.cfg.with_synthetic_futures else "disabled",
                "ok": False,
                "signal_count": 0,
                "signals": [],
                "error": f"{type(exc).__name__}: {exc}",
            }


# ============================================================================
# CLI
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CHAD read-only full-cycle preview / observability harness")
    p.add_argument("--artifact-path", default=str(FULL_CYCLE_ARTIFACT))
    p.add_argument("--dynamic-caps-path", default=str(DYNAMIC_CAPS_ARTIFACT))
    p.add_argument("--write-json", default="", help="Optional output path for preview JSON artifact")
    p.add_argument("--json", action="store_true", help="Render JSON instead of human-readable text")
    p.add_argument(
        "--with-synthetic-futures",
        action="store_true",
        help="Run deterministic alpha_futures smoke test using synthetic bars",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = build_arg_parser().parse_args(argv)

    cfg = PreviewAppConfig(
        full_cycle_path=Path(args.artifact_path),
        dynamic_caps_path=Path(args.dynamic_caps_path),
        write_json_path=Path(args.write_json) if safe_str(args.write_json) else None,
        use_json=bool(args.json),
        with_synthetic_futures=bool(args.with_synthetic_futures),
    )

    app = PreviewApp(
        cfg=cfg,
        artifact_repo=ArtifactRepository(
            full_cycle_path=cfg.full_cycle_path,
            dynamic_caps_path=cfg.dynamic_caps_path,
        ),
        registry_inspector=StrategyRegistryInspector(),
        renderer=PreviewRenderer(use_json=cfg.use_json),
    )
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
