"""chad/validation/verdict.py — Phase 5 verdict mapping (SSOT Part 4).

Maps the already-computed harness metrics (Phase-1 scoring + Phase-3 Deflated Sharpe
/ ruin, plus the OOS access count and feature-parity status) onto exactly one verdict
per head and one for the portfolio (SSOT
``docs/CHAD_EDGE_VALIDATION_HARNESS_DESIGN_v1.1.md`` Part 4). It is the "hard to pass,
unable to cheat itself" gate — and, critically, it is **pure**:

  * It computes NO metrics itself (no scoring, no DSR, no ruin — those are Phases 1/3;
    re-deriving them here would violate the SSOT single-spine / DRY constraint).
  * It performs **NO I/O of any kind** — it opens no file, imports no filesystem or OS
    writer, and in particular NEVER writes ``ready_for_live`` or any ``runtime/`` state.
    A ``PASS`` is only ever the string ``"PASS (candidate)"`` — evidence for a human
    decision. The machine never flips the live gate itself (SSOT §1.2 / Part 0). Its
    entire surface is ``(metrics in) → (VerdictResult out)``.

Verdict types (SSOT §4.1) and the fixed precedence they are tested in:
  1. ``NOT_REPLAYABLE`` — the head could not be replayed on historical data (feature
     parity ≠ REPLAYABLE, so it was never scored, SSOT §V1). A data-availability fact
     that precedes every numeric judgement.
  2. ``CONTAMINATED`` — the sealed OOS box was opened more than once
     (``oos_access_count > 1``, SSOT §3.1). Any numeric verdict on a twice-opened box
     is untrustworthy, so this overrides the numeric checks below.
  3. ``INSUFFICIENT_DATA`` — below a pre-registered minimum (§4.3: N_min=30 OOS trades,
     W_min=6 walk-forward windows, R_min=3 OOS regimes), or the Phase-0 data-quality
     audit FAILed for an involved symbol. The honest default (Part 0).
  4. ``FAIL`` — enough data, edge does not survive: deflated Sharpe not > 0 at
     confidence, OR cost-adjusted CAGR ≤ 0, OR worst-quantile ruin above bound, OR
     regime-fragile without scoped sizing (§4.1).
  5. ``PASS (candidate)`` — survives all of the above. Still only evidence.

Portfolio logic (SSOT §4.2, S5): :func:`decide_portfolio_verdict` requires BOTH the
portfolio track's own bar AND a stated fraction of capital allocated to individually
surviving heads — a portfolio cannot pass on allocator luck carrying failing heads.

Isolation (SSOT §1.2 / §2): standard-library only — :mod:`math`, :mod:`dataclasses`,
:mod:`enum`, :mod:`typing`. No numpy, no broker, no ``runtime/`` reader, no filesystem
access, no live-loop dependency.

Sentinel convention: a ``None`` metric where a positive result is required (e.g. an
undefined DSR or ruin once minimums are met) fails that check — the edge/safety could
not be *confirmed*, and the strict standard treats unconfirmed as not-passed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, Optional

__all__ = [
    "N_MIN",
    "W_MIN",
    "R_MIN",
    "Verdict",
    "VerdictThresholds",
    "DEFAULT_THRESHOLDS",
    "CheckOutcome",
    "HeadMetrics",
    "PortfolioMetrics",
    "VerdictResult",
    "decide_verdict",
    "decide_portfolio_verdict",
    "PASS_LABEL",
]

# --------------------------------------------------------------------------- #
# Pre-registered minimums (SSOT §4.3 — FIXED at Phase 3, before any sealed run).
# Restated here as the authority the verdict enforces; changing them is a config
# change that invalidates the seal + increments the deflation trial count (§3.2).
# --------------------------------------------------------------------------- #
N_MIN: Final[int] = 30  # >= 30 OOS trades per head being judged
W_MIN: Final[int] = 6   # >= 6 purged/embargoed walk-forward windows
R_MIN: Final[int] = 3   # >= 3 distinct regimes represented in OOS

# A PASS is ALWAYS a candidate — evidence for a human, never an auto-unlock.
PASS_LABEL: Final[str] = "PASS (candidate)"


class Verdict(Enum):
    """The five verdict types (SSOT §4.1). ``PASS`` is always labeled a *candidate*."""

    PASS = "PASS"
    FAIL = "FAIL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NOT_REPLAYABLE = "NOT_REPLAYABLE"
    CONTAMINATED = "CONTAMINATED"


@dataclass(frozen=True)
class VerdictThresholds:
    """The frozen pass/fail thresholds (SSOT §4.3 "starting threshold values").

    Defaults are the SSOT-committed starting values. A real sealed run hash-freezes a
    chosen instance via :mod:`chad.validation.config_freeze` and echoes it in the
    report; changing any of these after a FAIL is a penalised config change (§3.2).
    """

    n_min: int = N_MIN
    w_min: int = W_MIN
    r_min: int = R_MIN
    # Deflated Sharpe must exceed the deflation benchmark with >= this probability
    # ("deflated Sharpe > 0 at 95%"): DSR ∈ [0,1] is that very probability, §3.3.
    dsr_confidence: float = 0.95
    # Cost-adjusted CAGR must be strictly greater than this (§4.1: CAGR > 0).
    min_cagr: float = 0.0
    # Worst-quantile ruin probability must be strictly below this (§3.6 / §4.3: < 1%).
    ruin_bound: float = 0.01
    # Edge must exist in at least this many regimes, else regime-scoped sizing (§4.1).
    min_regimes_with_edge: int = 2
    # Portfolio PASS: at least this fraction of capital in individually-surviving
    # heads (§4.2, S5). A portfolio cannot pass on allocator luck.
    min_surviving_capital_fraction: float = 0.5

    def __post_init__(self) -> None:
        for name in ("n_min", "w_min", "r_min", "min_regimes_with_edge"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be an int >= 1, got {v!r}")
        if not (0.0 < float(self.dsr_confidence) <= 1.0):
            raise ValueError(f"dsr_confidence must be in (0, 1], got {self.dsr_confidence}")
        if isinstance(self.min_cagr, bool) or not isinstance(self.min_cagr, (int, float)):
            raise ValueError(f"min_cagr must be a real number, got {self.min_cagr!r}")
        if not (0.0 < float(self.ruin_bound) <= 1.0):
            raise ValueError(f"ruin_bound must be in (0, 1], got {self.ruin_bound}")
        if not (0.0 <= float(self.min_surviving_capital_fraction) <= 1.0):
            raise ValueError(
                f"min_surviving_capital_fraction must be in [0, 1], got "
                f"{self.min_surviving_capital_fraction}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_min": self.n_min,
            "w_min": self.w_min,
            "r_min": self.r_min,
            "dsr_confidence": self.dsr_confidence,
            "min_cagr": self.min_cagr,
            "ruin_bound": self.ruin_bound,
            "min_regimes_with_edge": self.min_regimes_with_edge,
            "min_surviving_capital_fraction": self.min_surviving_capital_fraction,
        }


DEFAULT_THRESHOLDS: Final[VerdictThresholds] = VerdictThresholds()


@dataclass(frozen=True)
class CheckOutcome:
    """One named pass/fail check with the value it tested and a human reason."""

    name: str
    passed: bool
    value: Optional[float]
    threshold: Optional[float]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class HeadMetrics:
    """Everything the verdict needs about one head (all pre-computed upstream).

    ``replayable`` is ``True`` only when the head passed feature parity AND was actually
    backtested (a REPLAYABLE head that produced tracks). ``deflated_sharpe_worst`` and
    ``worst_quantile_ruin`` are the WORST-quantile values from the Phase-3 seed sweep
    (SSOT §3.6: thresholds apply to the worst quantile, never the mean). ``oos_source``
    records whether the OOS metrics came from the sealed box (``"sealed_oos"``) or the
    dev decoy (``"decoy"``); ``final_run`` records whether this was a sealed run.
    """

    head: str
    parity_status: str
    replayable: bool
    data_quality_status: str          # "CLEAN" | "WARN" | "FAIL" (worst involved symbol)
    oos_access_count: int
    n_oos_trades: int
    n_walk_forward_windows: int
    n_regimes_in_oos: int
    deflated_sharpe_worst: Optional[float]
    cost_adj_cagr: Optional[float]
    worst_quantile_ruin: Optional[float]
    regimes_with_edge: int
    regime_scoped_sizing: bool
    final_run: bool
    oos_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "head": self.head,
            "parity_status": self.parity_status,
            "replayable": self.replayable,
            "data_quality_status": self.data_quality_status,
            "oos_access_count": self.oos_access_count,
            "n_oos_trades": self.n_oos_trades,
            "n_walk_forward_windows": self.n_walk_forward_windows,
            "n_regimes_in_oos": self.n_regimes_in_oos,
            "deflated_sharpe_worst": self.deflated_sharpe_worst,
            "cost_adj_cagr": self.cost_adj_cagr,
            "worst_quantile_ruin": self.worst_quantile_ruin,
            "regimes_with_edge": self.regimes_with_edge,
            "regime_scoped_sizing": self.regime_scoped_sizing,
            "final_run": self.final_run,
            "oos_source": self.oos_source,
        }


@dataclass(frozen=True)
class VerdictResult:
    """One head's (or the portfolio's) verdict + the full check trail (report-embedded).

    ``label`` is the display label (``"PASS (candidate)"`` for a pass, else the verdict
    value). ``is_candidate`` is ``True`` only for a PASS — a reminder that even a pass is
    just evidence. ``final_run`` mirrors the metric so a decoy-run pass is never mistaken
    for a sealed-run pass.
    """

    head: str
    verdict: Verdict
    label: str
    is_candidate: bool
    reasons: tuple[str, ...]
    checks: tuple[CheckOutcome, ...]
    minimums: dict[str, Any]
    final_run: bool
    oos_source: str
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "head": self.head,
            "verdict": self.verdict.value,
            "label": self.label,
            "is_candidate": self.is_candidate,
            "reasons": list(self.reasons),
            "checks": [c.to_dict() for c in self.checks],
            "minimums": self.minimums,
            "final_run": self.final_run,
            "oos_source": self.oos_source,
            "warnings": list(self.warnings),
        }


def _finite(x: Optional[float]) -> bool:
    """True iff ``x`` is a real, finite number (a ``None``/NaN metric is not usable)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _minimums_block(metrics: HeadMetrics, thresholds: VerdictThresholds) -> dict[str, Any]:
    """The §4.3 minimums block echoed into every verdict for auditability."""
    return {
        "n_oos_trades": {"value": metrics.n_oos_trades, "min": thresholds.n_min},
        "walk_forward_windows": {"value": metrics.n_walk_forward_windows, "min": thresholds.w_min},
        "regimes_in_oos": {"value": metrics.n_regimes_in_oos, "min": thresholds.r_min},
        "data_quality_status": metrics.data_quality_status,
    }


def decide_verdict(
    metrics: HeadMetrics, thresholds: VerdictThresholds = DEFAULT_THRESHOLDS
) -> VerdictResult:
    """Map one head's metrics to a verdict (SSOT Part 4). Pure — no metrics, no I/O.

    Applies the fixed precedence documented in the module header:
    NOT_REPLAYABLE → CONTAMINATED → INSUFFICIENT_DATA → FAIL → PASS (candidate).
    A ``PASS`` is always labeled ``"PASS (candidate)"`` and never anything a machine
    could read as an auto-unlock. This function writes nothing anywhere.
    """
    if not isinstance(metrics, HeadMetrics):
        raise ValueError(f"metrics must be a HeadMetrics, got {type(metrics).__name__}")
    if not isinstance(thresholds, VerdictThresholds):
        raise ValueError(f"thresholds must be a VerdictThresholds, got {type(thresholds).__name__}")

    minimums = _minimums_block(metrics, thresholds)
    warnings: list[str] = []
    if not metrics.final_run:
        warnings.append(
            f"non-sealed run: OOS metrics came from '{metrics.oos_source}', not the "
            "sealed out-of-sample partition — any verdict here is a dry-run, not evidence"
        )

    # --- 1. NOT_REPLAYABLE — never scored on historical data (SSOT §V1). ------ #
    if not metrics.replayable:
        return VerdictResult(
            head=metrics.head,
            verdict=Verdict.NOT_REPLAYABLE,
            label=Verdict.NOT_REPLAYABLE.value,
            is_candidate=False,
            reasons=(
                f"feature parity is {metrics.parity_status!r} (not REPLAYABLE); the head "
                "depends on inputs unavailable historically, so it is reported honestly "
                "and NOT scored (SSOT §V1 / §4.1)",
            ),
            checks=(),
            minimums=minimums,
            final_run=metrics.final_run,
            oos_source=metrics.oos_source,
            warnings=tuple(warnings),
        )

    # --- 2. CONTAMINATED — the sealed box was opened more than once (§3.1). ---- #
    if metrics.oos_access_count > 1:
        return VerdictResult(
            head=metrics.head,
            verdict=Verdict.CONTAMINATED,
            label=Verdict.CONTAMINATED.value,
            is_candidate=False,
            reasons=(
                f"the sealed OOS partition was opened {metrics.oos_access_count} times "
                "(> 1); the out-of-sample result is contaminated and cannot be trusted "
                "(SSOT §3.1)",
            ),
            checks=(),
            minimums=minimums,
            final_run=metrics.final_run,
            oos_source=metrics.oos_source,
            warnings=tuple(warnings),
        )

    # --- 3. INSUFFICIENT_DATA — below a pre-registered minimum (§4.3). -------- #
    insufficient: list[str] = []
    if metrics.n_oos_trades < thresholds.n_min:
        insufficient.append(
            f"only {metrics.n_oos_trades} OOS trades (< N_min={thresholds.n_min})"
        )
    if metrics.n_walk_forward_windows < thresholds.w_min:
        insufficient.append(
            f"only {metrics.n_walk_forward_windows} walk-forward windows (< W_min={thresholds.w_min})"
        )
    if metrics.n_regimes_in_oos < thresholds.r_min:
        insufficient.append(
            f"only {metrics.n_regimes_in_oos} distinct OOS regimes (< R_min={thresholds.r_min})"
        )
    if metrics.data_quality_status == "FAIL":
        insufficient.append("Phase-0 data-quality audit is FAIL for an involved symbol")
    if insufficient:
        return VerdictResult(
            head=metrics.head,
            verdict=Verdict.INSUFFICIENT_DATA,
            label=Verdict.INSUFFICIENT_DATA.value,
            is_candidate=False,
            reasons=tuple(
                ["below pre-registered minimums (SSOT §4.3) — honest default, not a fabricated pass:"]
                + insufficient
            ),
            checks=(),
            minimums=minimums,
            final_run=metrics.final_run,
            oos_source=metrics.oos_source,
            warnings=tuple(warnings),
        )

    # --- 4. FAIL vs 5. PASS — minimums met; apply the strict thresholds. ------ #
    checks = _survival_checks(metrics, thresholds)
    failed = [c for c in checks if not c.passed]
    if failed:
        return VerdictResult(
            head=metrics.head,
            verdict=Verdict.FAIL,
            label=Verdict.FAIL.value,
            is_candidate=False,
            reasons=tuple(
                ["sufficient data, but the edge does not survive the strict standard (SSOT §4.1):"]
                + [c.reason for c in failed]
            ),
            checks=checks,
            minimums=minimums,
            final_run=metrics.final_run,
            oos_source=metrics.oos_source,
            warnings=tuple(warnings),
        )

    return VerdictResult(
        head=metrics.head,
        verdict=Verdict.PASS,
        label=PASS_LABEL,
        is_candidate=True,
        reasons=(
            "survives OOS + costs + punitive deflation + regime slices + worst-quantile "
            "ruin — PASS (candidate). Still only evidence for a human decision; the "
            "machine never flips ready_for_live (SSOT §4.1 / Part 0).",
        ),
        checks=checks,
        minimums=minimums,
        final_run=metrics.final_run,
        oos_source=metrics.oos_source,
        warnings=tuple(warnings),
    )


def _survival_checks(
    metrics: HeadMetrics, thresholds: VerdictThresholds
) -> tuple[CheckOutcome, ...]:
    """The four §4.1 survival checks (minimums already satisfied by the caller)."""
    # (a) Deflated Sharpe > 0 at confidence: DSR (a probability) >= dsr_confidence.
    dsr = metrics.deflated_sharpe_worst
    dsr_ok = _finite(dsr) and float(dsr) >= thresholds.dsr_confidence  # type: ignore[arg-type]
    dsr_check = CheckOutcome(
        name="deflated_sharpe",
        passed=dsr_ok,
        value=(float(dsr) if _finite(dsr) else None),
        threshold=thresholds.dsr_confidence,
        reason=(
            f"worst-quantile deflated Sharpe {dsr!r} "
            + ("passes" if dsr_ok else "does not reach")
            + f" the >= {thresholds.dsr_confidence} confidence bar (deflated Sharpe > 0 at "
            f"{thresholds.dsr_confidence:.0%})"
        ),
    )

    # (b) Cost-adjusted CAGR > min_cagr.
    cagr = metrics.cost_adj_cagr
    cagr_ok = _finite(cagr) and float(cagr) > thresholds.min_cagr  # type: ignore[arg-type]
    cagr_check = CheckOutcome(
        name="cost_adjusted_cagr",
        passed=cagr_ok,
        value=(float(cagr) if _finite(cagr) else None),
        threshold=thresholds.min_cagr,
        reason=(
            f"cost-adjusted CAGR {cagr!r} "
            + ("is > " if cagr_ok else "is not > ")
            + f"{thresholds.min_cagr}"
        ),
    )

    # (c) Worst-quantile ruin < ruin_bound.
    ruin = metrics.worst_quantile_ruin
    ruin_ok = _finite(ruin) and float(ruin) < thresholds.ruin_bound  # type: ignore[arg-type]
    ruin_check = CheckOutcome(
        name="worst_quantile_ruin",
        passed=ruin_ok,
        value=(float(ruin) if _finite(ruin) else None),
        threshold=thresholds.ruin_bound,
        reason=(
            f"worst-quantile ruin {ruin!r} "
            + ("is < " if ruin_ok else "is not < ")
            + f"the {thresholds.ruin_bound} bound (ergodicity over expected value, SSOT §3.6)"
        ),
    )

    # (d) Edge in >= min_regimes_with_edge regimes, else regime-scoped sizing.
    regime_ok = (
        metrics.regimes_with_edge >= thresholds.min_regimes_with_edge
        or metrics.regime_scoped_sizing
    )
    regime_check = CheckOutcome(
        name="regime_breadth",
        passed=regime_ok,
        value=float(metrics.regimes_with_edge),
        threshold=float(thresholds.min_regimes_with_edge),
        reason=(
            f"edge present in {metrics.regimes_with_edge} regime(s); "
            + (
                "meets the breadth bar"
                if regime_ok
                else f"< {thresholds.min_regimes_with_edge} and sizing is not regime-scoped "
                "→ regime-fragile"
            )
        ),
    )
    return (dsr_check, cagr_check, ruin_check, regime_check)


# --------------------------------------------------------------------------- #
# Portfolio verdict (SSOT §4.2, S5).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PortfolioMetrics:
    """Portfolio-track metrics + the per-head survival needed for the §4.2 pass logic.

    ``portfolio`` is the portfolio track judged exactly like a head (the 50/30/20
    allocator is itself replayed inside the portfolio track, S5). ``surviving_heads`` /
    ``total_heads`` count individually PASSing heads. ``capital_fraction_in_surviving_heads``
    is the fraction of replayed capital the allocator put into those surviving heads.
    """

    portfolio: HeadMetrics
    surviving_heads: int
    total_heads: int
    capital_fraction_in_surviving_heads: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "portfolio": self.portfolio.to_dict(),
            "surviving_heads": self.surviving_heads,
            "total_heads": self.total_heads,
            "capital_fraction_in_surviving_heads": self.capital_fraction_in_surviving_heads,
        }


def decide_portfolio_verdict(
    pm: PortfolioMetrics, thresholds: VerdictThresholds = DEFAULT_THRESHOLDS
) -> VerdictResult:
    """Verdict for the portfolio track (SSOT §4.2, S5). Pure — no metrics, no I/O.

    The portfolio track is first judged exactly like a head via :func:`decide_verdict`
    (so NOT_REPLAYABLE / CONTAMINATED / INSUFFICIENT_DATA / FAIL propagate unchanged).
    A portfolio bar that would otherwise PASS is downgraded to FAIL unless at least
    ``thresholds.min_surviving_capital_fraction`` of capital sits in individually
    surviving heads — a portfolio must not pass on allocator luck carrying failing
    heads.
    """
    if not isinstance(pm, PortfolioMetrics):
        raise ValueError(f"pm must be a PortfolioMetrics, got {type(pm).__name__}")
    base = decide_verdict(pm.portfolio, thresholds)
    if base.verdict is not Verdict.PASS:
        # Re-label the head as "portfolio" but keep the propagated verdict + trail.
        return VerdictResult(
            head="portfolio",
            verdict=base.verdict,
            label=base.label,
            is_candidate=base.is_candidate,
            reasons=base.reasons,
            checks=base.checks,
            minimums=base.minimums,
            final_run=base.final_run,
            oos_source=base.oos_source,
            warnings=base.warnings,
        )

    frac = float(pm.capital_fraction_in_surviving_heads)
    survivor_ok = frac >= thresholds.min_surviving_capital_fraction
    survivor_check = CheckOutcome(
        name="surviving_capital_fraction",
        passed=survivor_ok,
        value=frac,
        threshold=thresholds.min_surviving_capital_fraction,
        reason=(
            f"{frac:.0%} of capital in individually-surviving heads "
            f"({pm.surviving_heads}/{pm.total_heads} heads); "
            + (
                "meets the §4.2 floor"
                if survivor_ok
                else f"< {thresholds.min_surviving_capital_fraction:.0%} floor — a portfolio "
                "cannot pass on allocator luck carrying failing heads"
            )
        ),
    )
    checks = base.checks + (survivor_check,)
    if survivor_ok:
        return VerdictResult(
            head="portfolio",
            verdict=Verdict.PASS,
            label=PASS_LABEL,
            is_candidate=True,
            reasons=base.reasons
            + (
                f"portfolio bar met AND {frac:.0%} of capital in individually-surviving "
                "heads (SSOT §4.2, S5).",
            ),
            checks=checks,
            minimums=base.minimums,
            final_run=base.final_run,
            oos_source=base.oos_source,
            warnings=base.warnings,
        )
    return VerdictResult(
        head="portfolio",
        verdict=Verdict.FAIL,
        label=Verdict.FAIL.value,
        is_candidate=False,
        reasons=(
            "portfolio deflated-Sharpe bar met, but the §4.2 capital-in-survivors floor "
            "is not: " + survivor_check.reason,
        ),
        checks=checks,
        minimums=base.minimums,
        final_run=base.final_run,
        oos_source=base.oos_source,
        warnings=base.warnings,
    )
