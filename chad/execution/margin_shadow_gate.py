"""chad/execution/margin_shadow_gate.py — Phase C shadow wiring of the margin/BP gate.

The built-but-inert margin BLOCK gate (Phases A+B: buying_power_provider, kraken_bp_provider,
pending_exposure_ledger, margin_block.decide) is wired here into the IBKR order-submission
chokepoint in **SHADOW mode** (design SSOT ``docs/CHAD_MARGIN_BLOCK_DESIGN_v2.2.md`` Part 7
step 2 / Part 8 Phase C). Shadow contract, summarized from the design:

  * evaluates **every** order using **cached data only** — no synchronous broker calls in
    the submit path (P3);
  * runs the decision pipeline (the ledger *reserve* is exercised each call; the
    *reconcile*-from-broker path runs only when an ``open_orders_source`` is wired — in the
    default production wiring it is not yet, so the ledger is flagged stale rather than
    reconciled), **only the ALLOW/BLOCK outcome suppressed** — so a would-block surfaces in
    the logs, never as a real block. Cross-call reserve→*release* lifecycle (the design's
    "leaked-reservation surfaces in shadow" goal) needs the persistent maintained ledger and
    is the G3b companion (see the per-evaluation-ledger note on ``_evaluate_inner``);
  * logs a grep-able marker per evaluation and appends the verdict to a dedicated evidence
    ndjson (NOT ``runtime/``) so the enforce-flip (G3b) can be judged on a corpus;
  * **blocks nothing** while ``mode == "shadow"`` and is **provably side-effect-free on the
    submission outcome** (design/task constraint 4);
  * **fails OPEN in shadow, loudly**: if the gate raises or its inputs are stale/missing, it
    logs ``MARGIN_GATE_ERROR`` and proceeds — shadow must never be able to stop trading
    (task constraint 1). In ENFORCE the same error path fails **closed** (see
    :meth:`MarginShadowGate.should_block`).

The enforce path (``mode`` ∈ {enforce_paper, enforce_live} → real BLOCK before the
idempotency claim) is implemented and tested behind the config read, but flipping
``config/margin_block.json`` ``mode`` is a separate future authorization (G3b) — this module
never mutates that file.

Data reality (verified 2026-07-10): the account ``ExcessLiquidity`` / ``FullInitMarginReq``
fields the gate needs are NOT yet published to any read-only runtime snapshot
(``runtime/positions_truth.json`` carries ``NetLiquidation`` only). So the DEFAULT production
snapshot source returns a fail-closed sentinel and shadow will honestly emit
``STALE_OR_MISSING_MARGIN_DATA`` for real orders — exactly what shadow exists to surface. The
snapshot source is injectable so tests (and, later, a runtime account-summary publisher)
supply complete snapshots that exercise the true ALLOW/BLOCK and enforce paths.

Isolation: imports only stdlib + the pure ``chad.risk`` gate modules + the canonical FX
constant. It imports NOTHING from ``chad.execution.ibkr_adapter`` (the adapter imports THIS),
so there is no cycle, and it issues NO broker I/O.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from chad.constants.fx import USDCAD_CONVERSION_CONSTANT
from chad.risk.buying_power_provider import BuyingPowerSnapshot
from chad.risk.margin_block import (
    AllowOrBlock,
    MarginBlockConfig,
    MarginBlockConfigError,
    decide,
    load_frozen_config,
)
from chad.risk.pending_exposure_ledger import PendingExposureLedger

__all__ = [
    "ShadowVerdict",
    "MarginShadowGate",
    "order_view_from_intent",
    "build_default_shadow_gate",
    "MARKER_SHADOW",
    "MARKER_ERROR",
]

LOGGER = logging.getLogger("chad.execution.margin_shadow_gate")

MARKER_SHADOW = "MARGIN_SHADOW"
MARKER_ERROR = "MARGIN_GATE_ERROR"

# Evidence schema version + default location (design/task constraint 3: dedicated ndjson
# under data/, NEVER runtime/).
EVIDENCE_SCHEMA_VERSION = "margin_shadow.v1"
_DEFAULT_EVIDENCE_SUBDIR = ("data", "margin_shadow")

# SnapshotSource(order_view) -> (account_snapshot, positions). OpenOrdersSource() -> specs.
SnapshotSource = Callable[[Dict[str, Any]], Tuple[Any, Any]]
OpenOrdersSource = Callable[[], Any]


# --------------------------------------------------------------------------- #
# The observability verdict carried back to the chokepoint.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ShadowVerdict:
    """One shadow evaluation's outcome (never blocks in shadow; observability + evidence).

    ``evaluated`` is True iff the pure gate ran (False → the wiring itself errored and this is
    a fail-open sentinel). ``would_block`` is the gate's TRUE decision (a BLOCK) regardless of
    whether we act on it. ``staleness`` flags that an input was stale/missing/incomplete (so
    enforce-readiness can be judged later). ``reserved`` mirrors the gate's ledger reservation.
    """

    evaluated: bool
    verdict: str                 # "ALLOW" | "BLOCK" | "ERROR"
    reason: str
    detail: str
    mode: str
    symbol: str
    side: str
    strategy: str
    asset_class: str
    order_id: str
    est_exposure_cad: Optional[float]
    headroom_cad: Optional[float]
    staleness: bool
    staleness_detail: str
    would_block: bool
    reserved: bool
    ts_utc: str
    netliq_context_cad: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "ts_utc": self.ts_utc,
            "evaluated": self.evaluated,
            "verdict": self.verdict,
            "reason": self.reason,
            "detail": self.detail,
            "mode": self.mode,
            "symbol": self.symbol,
            "side": self.side,
            "strategy": self.strategy,
            "asset_class": self.asset_class,
            "order_id": self.order_id,
            "est_exposure_cad": self.est_exposure_cad,
            "headroom_cad": self.headroom_cad,
            "staleness": self.staleness,
            "staleness_detail": self.staleness_detail,
            "would_block": self.would_block,
            "reserved": self.reserved,
            "netliq_context_cad": self.netliq_context_cad,
            **({"extra": self.extra} if self.extra else {}),
        }

    def marker_line(self) -> str:
        """The single grep-able MARGIN_SHADOW marker line (task constraint 2)."""
        return (
            f"{MARKER_SHADOW} verdict={self.verdict} reason={self.reason} "
            f"symbol={self.symbol or '-'} strategy={self.strategy or '-'} "
            f"est_exposure={_fmt(self.est_exposure_cad)} headroom={_fmt(self.headroom_cad)} "
            f"staleness={str(self.staleness).lower()} mode={self.mode}"
        )


def _fmt(x: Optional[float]) -> str:
    return "null" if x is None else f"{x:.2f}"


def _iso_utc(epoch: float) -> str:
    """UTC ISO-8601 from an epoch (pure function of the injected epoch; no wall-clock).

    The evidence date-partition is derived from ``ShadowVerdict.ts_utc`` (itself this value),
    so the whole evidence artifact is deterministic in the injected ``now_epoch``."""
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(epoch)))
    except (ValueError, OverflowError, OSError):
        return "1970-01-01T00:00:00Z"


# --------------------------------------------------------------------------- #
# Order-view adapter: NormalizedIntent -> the gate's order duck-type.
# --------------------------------------------------------------------------- #
def _finite_pos(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if (math.isfinite(f) and f > 0.0) else None


def order_view_from_intent(intent: Any, *, order_id: str) -> Dict[str, Any]:
    """Map a NormalizedIntent (or any object exposing the same attributes) onto the mapping
    :func:`chad.risk.margin_block.decide` reads (order_id, symbol, side, asset_class, qty,
    currency, notional, price, multiplier, whatif_init_margin).

    ``asset_class`` is normalized ``etf`` → ``equity`` (ETFs are RegT-margined like equities,
    and the gate's rate table keys only equity/crypto/futures). ``whatif_init_margin`` is
    ``None`` in shadow: the cached-data-only path issues no whatIf broker call, so the
    independent estimate governs (design §2.1)."""
    asset_class = str(getattr(intent, "asset_class", "") or "").strip().lower()
    if asset_class == "etf":
        asset_class = "equity"
    meta = getattr(intent, "meta", None)
    multiplier = None
    if isinstance(meta, dict):
        multiplier = meta.get("multiplier")
    return {
        "order_id": order_id,
        "symbol": str(getattr(intent, "symbol", "") or "").strip().upper(),
        "side": str(getattr(intent, "side", "") or "").strip().upper(),
        "asset_class": asset_class,
        "qty": _finite_pos(getattr(intent, "quantity", None)),
        "currency": str(getattr(intent, "currency", "") or "").strip().upper() or "USD",
        "notional": _finite_pos(getattr(intent, "notional_estimate", None)),
        "price": _finite_pos(getattr(intent, "limit_price", None)),
        "multiplier": multiplier if multiplier is not None else 1.0,
        "whatif_init_margin": None,
        "strategy": str(getattr(intent, "strategy", "") or "").strip() or "unknown",
    }


def _est_exposure_cad(order_view: Dict[str, Any], config: MarginBlockConfig) -> Optional[float]:
    """Best-effort CAD exposure estimate for the marker/evidence (observability only).

    Mirrors ``chad.risk.margin_block._to_cad`` using the canonical FX constant
    (``chad.constants.fx.USDCAD_CONVERSION_CONSTANT``) × the config's conservative bias — the
    same overstate-USD rule. Not a decision input (the gate computes its own)."""
    notional = order_view.get("notional")
    if notional is None:
        price = order_view.get("price")
        qty = order_view.get("qty")
        mult = order_view.get("multiplier") or 1.0
        if price is None or qty is None:
            return None
        notional = float(price) * float(qty) * float(mult)
    ccy = str(order_view.get("currency") or "USD").upper()
    if ccy == "CAD":
        return float(notional)
    if ccy == "USD":
        return float(notional) * USDCAD_CONVERSION_CONSTANT * config.fx_conservative_bias_mult
    return None


def _headroom_cad(account_snapshot: Any) -> Optional[float]:
    """ExcessLiquidity as a coarse headroom proxy for the marker (None if snapshot unusable)."""
    if account_snapshot is None or not bool(getattr(account_snapshot, "usable", False)):
        return None
    ex = getattr(account_snapshot, "excess_liquidity", None)
    if ex is None or isinstance(ex, bool):
        return None
    try:
        f = float(ex)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# --------------------------------------------------------------------------- #
# The shadow gate.
# --------------------------------------------------------------------------- #
class MarginShadowGate:
    """Evaluates each intent through the pure gate, logs + records, and (only in enforce)
    tells the caller to block. In shadow it is provably side-effect-free on submission."""

    def __init__(
        self,
        config: MarginBlockConfig,
        *,
        snapshot_source: SnapshotSource,
        open_orders_source: Optional[OpenOrdersSource] = None,
        evidence_path: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
        ledger_ttl_seconds: Optional[float] = None,
    ) -> None:
        self._config = config
        self._snapshot_source = snapshot_source
        self._open_orders_source = open_orders_source
        self._evidence_path = evidence_path
        self._log = logger or LOGGER
        self._ledger_ttl_seconds = (
            float(ledger_ttl_seconds) if ledger_ttl_seconds is not None else None
        )

    @property
    def config(self) -> MarginBlockConfig:
        return self._config

    @property
    def mode(self) -> str:
        return self._config.mode

    # -- the evaluation (never raises into the order path) -------------------- #
    def evaluate(self, order_view: Dict[str, Any], *, now_epoch: float) -> ShadowVerdict:
        """Evaluate one order. NEVER raises: any internal failure returns a fail-open ERROR
        verdict (marker MARGIN_GATE_ERROR). Marker + evidence emission are best-effort and
        cannot fail the evaluation. Returns a :class:`ShadowVerdict` for the chokepoint."""
        try:
            verdict = self._evaluate_inner(order_view, now_epoch)
        except Exception as exc:  # noqa: BLE001 - fail-open in shadow (task constraint 1)
            verdict = self._error_verdict(order_view, exc, now_epoch)
        self._safe_marker(verdict)
        self._safe_write_evidence(verdict)
        return verdict

    def _evaluate_inner(self, order_view: Dict[str, Any], now_epoch: float) -> ShadowVerdict:
        account_snapshot, positions = self._snapshot_source(order_view)

        # Per-evaluation ledger, rebuilt from broker open-orders truth (cached, read-only).
        # A fresh-per-call ledger exercises reserve + reconcile-from-truth without any
        # cross-call persistence, so it is provably side-effect-free (constraint 4) and cannot
        # accumulate phantom reservations at the pre-claim chokepoint (which has no broker
        # orderId yet). Persistent reserve→release-on-fill lifecycle is the G3b companion.
        ledger = self._new_ledger()
        rebuilt = self._rebuild_ledger(ledger, now_epoch)

        result: AllowOrBlock = decide(
            order_view, account_snapshot, positions, ledger, self._config, now=now_epoch
        )

        usable = bool(getattr(account_snapshot, "usable", False))
        staleness = (not usable) or (not rebuilt)
        staleness_bits: List[str] = []
        if not usable:
            staleness_bits.append(
                f"account_snapshot={getattr(account_snapshot, 'reason', 'missing')}"
            )
        if not rebuilt:
            staleness_bits.append("ledger_open_orders_unavailable")

        verdict_str = "ALLOW" if result.allowed else "BLOCK"
        return ShadowVerdict(
            evaluated=True,
            verdict=verdict_str,
            reason=result.reason.value,
            detail=result.detail,
            mode=self._config.mode,
            symbol=str(order_view.get("symbol") or ""),
            side=str(order_view.get("side") or ""),
            strategy=str(order_view.get("strategy") or "unknown"),
            asset_class=str(order_view.get("asset_class") or ""),
            order_id=str(order_view.get("order_id") or ""),
            est_exposure_cad=_est_exposure_cad(order_view, self._config),
            headroom_cad=_headroom_cad(account_snapshot),
            staleness=staleness,
            staleness_detail=";".join(staleness_bits),
            would_block=result.blocked,
            reserved=result.reserved,
            ts_utc=_iso_utc(now_epoch),
            netliq_context_cad=_netliq(account_snapshot),
        )

    def _error_verdict(self, order_view: Dict[str, Any], exc: Exception, now_epoch: float) -> ShadowVerdict:
        return ShadowVerdict(
            evaluated=False,
            verdict="ERROR",
            reason=MARKER_ERROR,
            detail=f"{type(exc).__name__}: {exc}",
            mode=self._config.mode,
            symbol=str(order_view.get("symbol") or ""),
            side=str(order_view.get("side") or ""),
            strategy=str(order_view.get("strategy") or "unknown"),
            asset_class=str(order_view.get("asset_class") or ""),
            order_id=str(order_view.get("order_id") or ""),
            est_exposure_cad=None,
            headroom_cad=None,
            staleness=True,
            staleness_detail="gate_internal_error",
            would_block=False,   # shadow proceeds; enforce fails closed via should_block
            reserved=False,
            ts_utc=_iso_utc(now_epoch),
        )

    def _new_ledger(self) -> PendingExposureLedger:
        if self._ledger_ttl_seconds is not None:
            return PendingExposureLedger(ttl_seconds=self._ledger_ttl_seconds)
        return PendingExposureLedger()

    def _rebuild_ledger(self, ledger: PendingExposureLedger, now_epoch: float) -> bool:
        """Rebuild the ledger from a cached, read-only broker open-orders snapshot. Returns
        True iff a snapshot was available (else the ledger stays empty, flagged staleness).
        Never issues broker I/O; never raises (fail-open — treated as unavailable)."""
        if self._open_orders_source is None:
            return False
        try:
            open_orders = self._open_orders_source()
        except Exception:  # noqa: BLE001 - a source failure is 'unavailable', never fatal
            return False
        if open_orders is None:
            return False
        try:
            ledger.rebuild_from_broker(open_orders, now=now_epoch)
            return True
        except Exception:  # noqa: BLE001
            return False

    # -- enforcement decision (shadow: always False) -------------------------- #
    def should_block(self, verdict: ShadowVerdict) -> bool:
        """Whether the caller must actually block this order. In shadow ALWAYS False
        (blocks nothing). In enforce: block on a real BLOCK verdict, and fail **closed** on
        an internal wiring error (evaluated == False) — the enforce mirror of the shadow
        fail-open (task constraint 1 / design Part 7)."""
        if not self._config.is_enforce:
            return False
        if not verdict.evaluated:
            return True  # enforce fails closed on a wiring error
        return verdict.would_block

    # -- observability (best-effort; never raises) ---------------------------- #
    def _safe_marker(self, verdict: ShadowVerdict) -> None:
        try:
            if verdict.evaluated:
                self._log.info(
                    verdict.marker_line(),
                    extra={"margin_shadow": verdict.to_dict()},
                )
            else:
                self._log.error(
                    f"{MARKER_ERROR} symbol={verdict.symbol or '-'} "
                    f"strategy={verdict.strategy or '-'} detail={verdict.detail} "
                    f"mode={verdict.mode} (fail-open: proceeding)",
                    extra={"margin_shadow": verdict.to_dict()},
                )
        except Exception:  # noqa: BLE001 - a logging failure must not affect the order path
            pass

    def _safe_write_evidence(self, verdict: ShadowVerdict) -> None:
        path = self._resolve_evidence_path(verdict)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(verdict.to_dict(), sort_keys=True, ensure_ascii=True) + "\n")
        except Exception:  # noqa: BLE001 - evidence is best-effort; never fail the order path
            pass

    def _resolve_evidence_path(self, verdict: ShadowVerdict) -> Optional[Path]:
        if self._evidence_path is None:
            return None
        base = self._evidence_path
        # A directory (or extensionless path) → date-partition; a concrete file → use as-is.
        if base.suffix == ".ndjson":
            return base
        day = verdict.ts_utc[:10].replace("-", "") if verdict.ts_utc else "19700101"
        return base / f"margin_shadow_{day}.ndjson"


def _netliq(account_snapshot: Any) -> Optional[float]:
    if account_snapshot is None:
        return None
    v = getattr(account_snapshot, "net_liquidation", None)
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# --------------------------------------------------------------------------- #
# Default production wiring — fail-open config load + read-only runtime source.
# --------------------------------------------------------------------------- #
def _default_ibkr_snapshot_source(
    runtime_dir: Path, config: MarginBlockConfig
) -> SnapshotSource:
    """Read-only source over existing runtime snapshots (no broker I/O).

    Reality (2026-07-10): ``ExcessLiquidity`` / ``FullInitMarginReq`` are NOT published to any
    runtime file, so this returns a fail-closed BuyingPowerSnapshot with a precise reason — the
    gate then honestly emits STALE_OR_MISSING in shadow. ``NetLiquidation`` (present in
    ``positions_truth.json``) and positions are surfaced for context/future-readiness. Replace
    this source once a runtime account-summary publisher lands (the G3b prerequisite)."""

    def _source(_order_view: Dict[str, Any]) -> Tuple[Any, Any]:
        netliq: Optional[float] = None
        positions: List[Dict[str, Any]] = []
        try:
            truth = json.loads((runtime_dir / "positions_truth.json").read_text(encoding="utf-8"))
            cash = (truth.get("cash") or {}).get("CAD") or {}
            nl = cash.get("NetLiquidation")
            netliq = float(nl) if isinstance(nl, (int, float)) and not isinstance(nl, bool) else None
            for p in truth.get("positions") or []:
                sym = str(p.get("symbol") or "").strip().upper()
                if sym:
                    positions.append({"symbol": sym, "qty": p.get("position")})
        except Exception:  # noqa: BLE001 - unreadable runtime → fully fail-closed sentinel
            pass
        snap = BuyingPowerSnapshot.fail_closed(
            reason="MARGIN_FIELDS_NOT_PUBLISHED",
            detail=(
                "ExcessLiquidity/FullInitMarginReq absent from runtime (positions_truth.json "
                f"carries NetLiquidation={netliq!r} only); shadow surfaces the data gap"
            ),
            ttl_seconds=config.account_data_ttl_seconds,
        )
        return snap, positions

    return _source


def build_default_shadow_gate(
    *,
    repo_root: Path,
    config_path: Optional[Path] = None,
    runtime_dir: Optional[Path] = None,
    evidence_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional["MarginShadowGate"]:
    """Build the production shadow gate from ``config/margin_block.json``. FAIL-OPEN: on any
    config problem it logs ``MARGIN_GATE_ERROR`` and returns ``None`` (no gate wired → order
    flow byte-identical) — a broken margin config must NEVER stop trading in shadow. When the
    config is enforce-mode, the same could block; that flip is the deliberate G3b step."""
    log = logger or LOGGER
    cfg_path = config_path or (repo_root / "config" / "margin_block.json")
    try:
        config = load_frozen_config(cfg_path)
    except (MarginBlockConfigError, OSError, ValueError) as exc:
        log.error(
            f"{MARKER_ERROR} config_load_failed path={cfg_path} detail={type(exc).__name__}: {exc} "
            "(fail-open: margin gate NOT wired, order flow unchanged)"
        )
        return None
    rt = runtime_dir or (repo_root / "runtime")
    ev = evidence_dir or repo_root.joinpath(*_DEFAULT_EVIDENCE_SUBDIR)
    return MarginShadowGate(
        config,
        snapshot_source=_default_ibkr_snapshot_source(rt, config),
        open_orders_source=None,  # no read-only open-orders snapshot yet → ledger flagged stale
        evidence_path=ev,
        logger=log,
    )
