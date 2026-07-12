"""
Kraken margin/BP shadow-gate glue (U2 / CRYPTO-2).

Mirrors the IBKR G3C wiring (margin_shadow_gate.build_default_shadow_gate +
order_view_from_intent) for the crypto lane:

  * `kraken_order_view_from_intent` maps a Kraken StrategyTradeIntent
    (pair/side/volume/price/notional_estimate — NO asset_class) onto the dict
    `chad.risk.margin_block.decide` reads, with asset_class="crypto" set
    explicitly so it routes to the already-built `_decide_crypto` lane
    (available_cad capacity).
  * `build_default_kraken_shadow_gate` constructs a MarginShadowGate with a
    Kraken snapshot source (runtime/kraken_balances.json -> KrakenBuyingPower
    Snapshot). `build_default_shadow_gate` is IBKR-only, so this is a parallel
    builder. Same config (config/margin_block.json, mode=shadow); no new
    thresholds. Longer TTL default (600s) because the balance producer refreshes
    ~300s (the 30s default would spuriously STALE->fail-closed).

Shadow contract is identical to IBKR: evaluate everything, block nothing, emit
MARGIN_SHADOW markers + evidence, fail-OPEN with MARGIN_GATE_ERROR.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from chad.execution.margin_shadow_gate import (
    MARKER_ERROR,
    MarginShadowGate,
    default_evidence_dir,
)
from chad.risk.margin_block import (
    MarginBlockConfigError,
    load_frozen_config,
)

LOGGER = logging.getLogger("chad.execution.kraken_margin_gate")

_REPO_ROOT = Path("/home/ubuntu/chad_finale")
_KRAKEN_LANE_TTL_SECONDS = 600.0  # producer refreshes ~300s; 30s default => spurious STALE

# Kraken REST altname -> CHAD canonical (for a readable symbol in the order_view).
_PAIR_TO_CANONICAL: Dict[str, str] = {
    "XBTUSD": "BTC-USD", "ETHUSD": "ETH-USD", "SOLUSD": "SOL-USD",
    "XBTCAD": "BTC-CAD", "ETHCAD": "ETH-CAD",
}


def _finite_pos(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")) or f <= 0.0:
        return None
    return f


def kraken_order_view_from_intent(intent: Any, *, order_id: str) -> Dict[str, Any]:
    """Map a Kraken StrategyTradeIntent onto the mapping margin_block.decide reads.

    asset_class is hardcoded "crypto" (the Kraken lane is crypto-only) so decide()
    routes to _decide_crypto. currency="USD" because notional_estimate is USD;
    _order_notional_cad converts to the CAD balance basis. whatif_init_margin is
    None (shadow issues no whatIf call).
    """
    pair = str(getattr(intent, "pair", "") or "").strip().upper()
    symbol = _PAIR_TO_CANONICAL.get(pair, pair)
    return {
        "order_id": order_id,
        "symbol": symbol,
        "side": str(getattr(intent, "side", "") or "").strip().upper(),
        "asset_class": "crypto",
        "qty": _finite_pos(getattr(intent, "volume", None)),
        "currency": "USD",
        "notional": _finite_pos(getattr(intent, "notional_estimate", None)),
        "price": _finite_pos(getattr(intent, "price", None)),
        "multiplier": 1.0,
        "whatif_init_margin": None,
        "strategy": str(getattr(intent, "strategy", "") or "").strip() or "alpha_crypto",
    }


def make_kraken_snapshot_source(
    *,
    balances_path: Path,
    ttl_seconds: float = _KRAKEN_LANE_TTL_SECONDS,
    prices_usd: Optional[Mapping[str, float]] = None,
    now_fn: Optional[Callable[[], float]] = None,
) -> Callable[[Dict[str, Any]], Tuple[Any, Any]]:
    """Read-only snapshot source: runtime/kraken_balances.json -> KrakenBuyingPowerSnapshot.

    Fail-closed: a missing/stale/unreadable balance file yields an unusable
    snapshot (the gate then honestly surfaces the gap in shadow, blocking nothing).
    """
    from chad.risk.kraken_bp_provider import KrakenBuyingPowerProvider

    provider = KrakenBuyingPowerProvider(ttl_seconds=float(ttl_seconds))
    now = now_fn or time.time

    def _source(_order_view: Dict[str, Any]) -> Tuple[Any, Any]:
        snap = provider.load_from_file(
            Path(balances_path), now_epoch=float(now()), prices_usd=prices_usd
        )
        positions: List[Dict[str, Any]] = []  # Kraken position context not published yet
        return snap, positions

    return _source


def build_default_kraken_shadow_gate(
    *,
    repo_root: Path = _REPO_ROOT,
    config_path: Optional[Path] = None,
    balances_path: Optional[Path] = None,
    evidence_dir: Optional[Path] = None,
    ttl_seconds: float = _KRAKEN_LANE_TTL_SECONDS,
    prices_usd: Optional[Mapping[str, float]] = None,
    now_fn: Optional[Callable[[], float]] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[MarginShadowGate]:
    """Build the Kraken shadow gate. FAIL-OPEN: returns None on any config problem
    (order flow byte-identical). Reuses config/margin_block.json (mode=shadow)."""
    log = logger or LOGGER
    cfg_path = config_path or (repo_root / "config" / "margin_block.json")
    try:
        config = load_frozen_config(cfg_path)
    except (MarginBlockConfigError, OSError, ValueError) as exc:
        log.error(
            f"{MARKER_ERROR} kraken config_load_failed path={cfg_path} "
            f"detail={type(exc).__name__}: {exc} (fail-open: Kraken margin gate NOT wired)"
        )
        return None
    bal_path = balances_path or (repo_root / "runtime" / "kraken_balances.json")
    ev = evidence_dir if evidence_dir is not None else default_evidence_dir(repo_root)
    return MarginShadowGate(
        config,
        snapshot_source=make_kraken_snapshot_source(
            balances_path=bal_path, ttl_seconds=ttl_seconds,
            prices_usd=prices_usd, now_fn=now_fn,
        ),
        open_orders_source=None,
        evidence_path=ev,
        logger=log,
    )
