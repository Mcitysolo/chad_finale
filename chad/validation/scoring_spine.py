"""chad/validation/scoring_spine.py — Phase 1 shared scoring spine.

The SINGLE module through which every performance number in the edge-validation
harness is computed. Stage 1 (synthetic backtest trades) and Stage 2 (real
post-Epoch-3 effective trades) both feed their results through *this* module and
no other — there is no second copy of this scoring arithmetic *within the
harness* (DRY is a first-class SSOT constraint,
``docs/CHAD_EDGE_VALIDATION_HARNESS_DESIGN_v1.1.md`` §1.3). Other ``sharpe_like``
proxies do exist elsewhere in CHAD (``chad.risk`` / ``chad.ops`` /
``chad.analytics``), but the §1.2 isolation boundary forbids the harness from
importing them, so this spine is deliberately the harness's own canonical,
returns-based definition rather than a re-use of those.

Isolation (SSOT §1.2 / §2): pure, offline, deterministic. Imports only the
standard library — no numpy, no broker, no ``runtime/`` reader, no live-loop
dependency. numpy is present in the environment but is deliberately avoided so
the arithmetic is transparent, hand-re-derivable, and byte-reproducible; there
is no numeric reason here that would justify its non-determinism risk.

Scope (Phase 1): return/risk metrics from a returns (or equity) series, plus
per-trade PnL statistics. The Deflated Sharpe Ratio is **NOT** here — it belongs
to Phase 3 (``significance.py``, SSOT §3.3 / Part 6).

--------------------------------------------------------------------------------
Conventions (documented once, enforced everywhere)
--------------------------------------------------------------------------------
Returns are **simple (arithmetic) per-period returns**: ``r_t = P_t/P_{t-1} - 1``.
An equity curve is rebuilt as ``E_0 = 1, E_t = E_{t-1} * (1 + r_t)`` — a positive
rescaling of any real equity path, so drawdown *percentages and indices* are
identical whether you pass returns or an equity series.

``periods_per_year`` (default 252 = daily) annualizes. ``risk_free_rate`` is an
**annual** rate; the per-period risk-free used for excess returns is
``risk_free_rate / periods_per_year`` (default 0). Volatility, Sharpe, and
Sortino use the **sample** standard deviation (``ddof = 1``, i.e. divide by
``n - 1``) — the institutional convention — so they require ``n >= 2``.

Sentinel convention — degenerate DATA never raises; it yields a well-defined
sentinel so a report can print ``null`` instead of crashing:
  * ``None``  = *undefined / insufficient data* (empty series; <2 points where a
    sample stdev is required; a zero denominator such as zero volatility, zero
    downside deviation, or no losing trades for profit factor; a CAGR whose
    annualized factor overflows float64 — real but not finitely representable).
  * ``0.0``   = a genuinely-computed zero (e.g. an all-flat series has 0.0 total
    return and 0.0 max drawdown — distinct from ``None`` "no data").
Invalid *configuration* (non-positive ``periods_per_year`` or an explicit
non-positive ``years``) is a caller bug, not degenerate data, so it raises
``ValueError`` (fail-fast). Empty / single-element / all-zero / all-negative data
never raises.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final, Optional, Sequence, Union

__all__ = [
    "DEFAULT_PERIODS_PER_YEAR",
    "ScoreResult",
    "equity_to_returns",
    "score_equity",
    "score_returns",
    "score_trades",
]

# A numeric input value. ``bool`` is a subclass of ``int`` but is rejected as a
# price/return by :func:`_as_float` so ``True`` cannot masquerade as ``1.0``.
Number = Union[int, float]

DEFAULT_PERIODS_PER_YEAR: Final[int] = 252


# --------------------------------------------------------------------------- #
# Result model — flat, serialisable, embedded verbatim by every later report.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScoreResult:
    """All scoring outputs in one JSON-serialisable record.

    A single flat container is intentional: both entry points (:func:`score_returns`
    and :func:`score_trades`) return this same type, filling only their portion and
    leaving the rest ``None``. ``kind`` records which entry point produced it so a
    consumer never mistakes an unset field for a computed zero.

    Fields are plain ``str``/``int``/``float``/``None`` — :meth:`to_dict` is
    directly ``json.dumps``-able with no custom encoder.
    """

    kind: str                                   # "returns" | "trades"
    label: Optional[str] = None                 # optional caller-supplied name

    # --- return/risk metrics (populated by score_returns / score_equity) --- #
    n_periods: Optional[int] = None
    periods_per_year: Optional[float] = None
    risk_free_rate: Optional[float] = None
    years: Optional[float] = None               # horizon used for CAGR
    total_return: Optional[float] = None
    cagr: Optional[float] = None
    volatility: Optional[float] = None          # annualized sample stdev of returns
    sharpe: Optional[float] = None              # annualized
    sortino: Optional[float] = None             # annualized
    max_drawdown: Optional[float] = None        # positive magnitude (0.20 == 20% down)
    max_drawdown_peak_index: Optional[int] = None
    max_drawdown_trough_index: Optional[int] = None

    # --- per-trade statistics (populated by score_trades) ------------------ #
    n_trades: Optional[int] = None
    win_rate: Optional[float] = None            # wins / n_trades (breakevens in denom)
    avg_win: Optional[float] = None             # mean of pnl > 0
    avg_loss: Optional[float] = None            # mean of pnl < 0 (a negative number)
    profit_factor: Optional[float] = None       # gross_profit / |gross_loss|
    total_pnl: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict of every field (JSON-serialisable, key order stable)."""
        return {
            "kind": self.kind,
            "label": self.label,
            "n_periods": self.n_periods,
            "periods_per_year": self.periods_per_year,
            "risk_free_rate": self.risk_free_rate,
            "years": self.years,
            "total_return": self.total_return,
            "cagr": self.cagr,
            "volatility": self.volatility,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_peak_index": self.max_drawdown_peak_index,
            "max_drawdown_trough_index": self.max_drawdown_trough_index,
            "n_trades": self.n_trades,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "total_pnl": self.total_pnl,
        }


# --------------------------------------------------------------------------- #
# Small pure numeric helpers — hand-rolled (not statistics.stdev) so the exact
# arithmetic is transparent to a reviewer's independent re-derivation and the
# summation order is fixed and deterministic.
# --------------------------------------------------------------------------- #
def _as_float(x: Number) -> float:
    """Coerce a numeric input to ``float``; reject ``bool`` and non-numbers.

    ``bool`` is a subclass of ``int`` — accepting it would let ``True`` be scored
    as a ``1.0`` return, so it is rejected explicitly.
    """
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError(f"expected a real number, got {type(x).__name__}: {x!r}")
    return float(x)


def _mean(xs: Sequence[float]) -> float:
    """Arithmetic mean. Caller guarantees ``len(xs) >= 1``."""
    return math.fsum(xs) / len(xs)


def _sample_stdev(xs: Sequence[float], mean: float) -> float:
    """Sample standard deviation (ddof=1). Caller guarantees ``len(xs) >= 2``.

    ``sqrt( sum((x - mean)**2) / (n - 1) )``. ``math.fsum`` bounds floating-point
    accumulation error without sacrificing determinism (fixed input order).
    """
    n = len(xs)
    variance = math.fsum((x - mean) ** 2 for x in xs) / (n - 1)
    return math.sqrt(variance)


def _equity_curve(returns: Sequence[float]) -> list[float]:
    """Rebuild a normalized equity curve ``[1.0, ...]`` from simple returns.

    Length is ``len(returns) + 1``; index 0 is the pre-first-return start point.
    """
    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1.0 + r))
    return equity


def _max_drawdown(equity: Sequence[float]) -> tuple[float, int, int]:
    """Largest peak-to-trough decline on ``equity`` as a positive magnitude.

    Returns ``(magnitude, peak_index, trough_index)`` where ``magnitude`` is
    ``-min_t(E_t / running_peak_t - 1)`` (>= 0), and the indices are into the
    passed ``equity`` sequence. A monotonically non-decreasing curve → ``0.0``
    magnitude with both indices ``0``. Scale-invariant: rescaling ``equity`` by a
    positive constant leaves magnitude and indices unchanged. When several
    equal-height peaks precede the worst trough, ``peak_index`` is the *earliest*
    such peak (a deterministic tie-break); magnitude and ``trough_index`` are
    unaffected by the tie-break.
    """
    peak_val = equity[0]
    peak_idx = 0
    worst = 0.0
    worst_peak_idx = 0
    worst_trough_idx = 0
    for i, e in enumerate(equity):
        if e > peak_val:
            peak_val = e
            peak_idx = i
        drawdown = e / peak_val - 1.0  # <= 0
        if drawdown < worst:
            worst = drawdown
            worst_trough_idx = i
            worst_peak_idx = peak_idx
    # Normalize a no-decline curve to a clean +0.0 magnitude (avoid -0.0).
    magnitude = -worst if worst < 0.0 else 0.0
    return (magnitude, worst_peak_idx, worst_trough_idx)


# --------------------------------------------------------------------------- #
# Public helpers for the "equity series in" input shape (SSOT §1.3).
# --------------------------------------------------------------------------- #
def equity_to_returns(equity: Sequence[Number]) -> list[float]:
    """Convert an equity curve to simple per-period returns.

    ``r_t = E_t / E_{t-1} - 1`` for ``t = 1..len(equity)-1``. Returns ``[]`` for a
    series with fewer than two points (no period to measure). Raises ``ValueError``
    if any prior equity value is non-positive (a return is undefined across a
    zero/negative base — a data-integrity problem, not a degenerate-but-valid one).
    """
    vals = [_as_float(e) for e in equity]
    out: list[float] = []
    for i in range(1, len(vals)):
        prev = vals[i - 1]
        if prev <= 0.0:
            raise ValueError(f"equity[{i - 1}] = {prev} is non-positive; return undefined")
        out.append(vals[i] / prev - 1.0)
    return out


def score_equity(
    equity: Sequence[Number],
    *,
    periods_per_year: Number = DEFAULT_PERIODS_PER_YEAR,
    risk_free_rate: Number = 0.0,
    years: Optional[Number] = None,
    label: Optional[str] = None,
) -> ScoreResult:
    """Score an equity curve by converting it to returns then delegating.

    Thin, DRY wrapper over :func:`score_returns` — no scoring arithmetic lives
    here. Drawdown indices/magnitude are identical to scoring the equivalent
    returns because the rebuilt curve is a positive rescaling of ``equity``.
    """
    return score_returns(
        equity_to_returns(equity),
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
        years=years,
        label=label,
    )


# --------------------------------------------------------------------------- #
# THE spine: returns series → return/risk metrics.
# --------------------------------------------------------------------------- #
def score_returns(
    returns: Sequence[Number],
    *,
    periods_per_year: Number = DEFAULT_PERIODS_PER_YEAR,
    risk_free_rate: Number = 0.0,
    years: Optional[Number] = None,
    label: Optional[str] = None,
) -> ScoreResult:
    """Score a sequence of simple per-period returns.

    Parameters
    ----------
    returns:
        Simple per-period returns (``0.01`` == +1%). May be empty.
    periods_per_year:
        Annualization factor (252 daily default). Must be > 0 (else ``ValueError``).
    risk_free_rate:
        **Annual** risk-free rate; per-period excess uses ``rf / periods_per_year``.
    years:
        Explicit horizon for CAGR. If ``None``, derived as ``n / periods_per_year``.
        Must be > 0 when supplied (else ``ValueError``).
    label:
        Optional name carried into the result (e.g. a head name).

    Metric definitions
    ------------------
    * total_return = ``prod(1 + r_t) - 1``.
    * cagr         = ``(1 + total_return) ** (1 / years) - 1``; ``-1.0`` if the
      final equity is exactly 0; ``None`` if final equity < 0 (real root undefined)
      or if the annualized factor overflows float64 (a huge gain over a tiny
      horizon — real but not finitely representable, so it collapses to ``None``
      rather than raising).
    * volatility   = ``sample_stdev(returns) * sqrt(periods_per_year)`` (n >= 2).
    * sharpe       = ``mean(excess)/sample_stdev(excess) * sqrt(periods_per_year)``,
      ``excess_t = r_t - rf/periods_per_year`` (n >= 2; ``None`` if stdev == 0).
    * sortino      = ``mean(excess)/downside_dev * sqrt(periods_per_year)`` where
      ``downside_dev = sqrt( sum(min(0, excess_t)**2) / n )`` — the standard target
      downside deviation dividing by the **full** count n, not the downside count
      (n >= 2; ``None`` if there is no downside, i.e. downside_dev == 0).
    * max_drawdown = positive magnitude of the largest peak-to-trough decline on
      the rebuilt equity curve, with peak/trough indices into that curve.

    Degenerate data returns sentinels (never raises): ``[]`` → all metrics ``None``,
    ``n_periods = 0``. See the module docstring for the full sentinel convention.
    """
    ppy = _as_float(periods_per_year)
    if ppy <= 0.0:
        raise ValueError(f"periods_per_year must be > 0, got {ppy}")
    rf_annual = _as_float(risk_free_rate)
    if years is not None:
        years_f: Optional[float] = _as_float(years)  # never None: _as_float raises or returns float
        if years_f <= 0.0:
            raise ValueError(f"years must be > 0 when supplied, got {years_f}")
    else:
        years_f = None

    rets = [_as_float(r) for r in returns]
    n = len(rets)

    # Empty series → nothing computable; honest all-None result.
    if n == 0:
        return ScoreResult(
            kind="returns",
            label=label,
            n_periods=0,
            periods_per_year=ppy,
            risk_free_rate=rf_annual,
        )

    sqrt_ppy = math.sqrt(ppy)
    rf_per_period = rf_annual / ppy

    # --- total return & CAGR (n >= 1) --- #
    equity = _equity_curve(rets)
    final_equity = equity[-1]
    total_return = final_equity - 1.0

    horizon = years_f if years_f is not None else n / ppy
    if final_equity > 0.0:
        # A large gain over a tiny horizon (e.g. one +100000% bar annualized with
        # horizon = 1/252) can push the annualized factor past float64's range.
        # That value is real but not finitely representable, so it collapses to the
        # None sentinel rather than raising OverflowError — honouring the module's
        # "degenerate data never raises" contract.
        try:
            cagr: Optional[float] = final_equity ** (1.0 / horizon) - 1.0
        except OverflowError:
            cagr = None
        else:
            if not math.isfinite(cagr):
                cagr = None
    elif final_equity == 0.0:
        cagr = -1.0  # total wipeout annualizes to -100%
    else:
        cagr = None  # negative final equity → real root undefined

    # --- max drawdown (n >= 1) --- #
    dd_mag, dd_peak, dd_trough = _max_drawdown(equity)

    # --- volatility / Sharpe / Sortino (require n >= 2 for sample stdev) --- #
    volatility: Optional[float] = None
    sharpe: Optional[float] = None
    sortino: Optional[float] = None
    if n >= 2:
        mean_ret = _mean(rets)
        vol_periodic = _sample_stdev(rets, mean_ret)
        volatility = vol_periodic * sqrt_ppy

        excess = [r - rf_per_period for r in rets]
        mean_excess = _mean(excess)
        # stdev of excess == stdev of returns (subtracting a constant), but compute
        # on `excess` explicitly so the Sharpe formula reads as written.
        std_excess = _sample_stdev(excess, mean_excess)
        if std_excess > 0.0:
            sharpe = (mean_excess / std_excess) * sqrt_ppy

        downside_sq_sum = math.fsum(min(0.0, e) ** 2 for e in excess)
        downside_dev = math.sqrt(downside_sq_sum / n)
        if downside_dev > 0.0:
            sortino = (mean_excess / downside_dev) * sqrt_ppy

    return ScoreResult(
        kind="returns",
        label=label,
        n_periods=n,
        periods_per_year=ppy,
        risk_free_rate=rf_annual,
        years=horizon,
        total_return=total_return,
        cagr=cagr,
        volatility=volatility,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=dd_mag,
        max_drawdown_peak_index=dd_peak,
        max_drawdown_trough_index=dd_trough,
    )


# --------------------------------------------------------------------------- #
# The spine: per-trade PnL → trade statistics (Stage-1 and Stage-2 share this).
# --------------------------------------------------------------------------- #
def score_trades(
    pnls: Sequence[Number],
    *,
    label: Optional[str] = None,
) -> ScoreResult:
    """Score a sequence of per-trade PnL values (currency units).

    A trade with ``pnl > 0`` is a win, ``pnl < 0`` a loss, ``pnl == 0`` a
    break-even (counted in ``n_trades`` and thus in the ``win_rate`` denominator,
    but is neither a win nor a loss).

    Fields
    ------
    * n_trades      = ``len(pnls)``.
    * total_pnl     = ``sum(pnls)`` (``0.0`` for an empty book — a defined sum).
    * win_rate      = ``wins / n_trades``; ``None`` if there are no trades.
    * avg_win       = mean of winning PnLs; ``None`` if none.
    * avg_loss      = mean of losing PnLs (a negative number); ``None`` if none.
    * profit_factor = ``gross_profit / |gross_loss|``; ``None`` when there are no
      losing trades (division by zero → undefined/infinite); ``0.0`` when there are
      losses but no wins.

    Never raises on degenerate data (``[]`` → ``n_trades = 0``, ``total_pnl = 0.0``,
    every ratio ``None``).
    """
    vals = [_as_float(p) for p in pnls]
    n = len(vals)

    if n == 0:
        return ScoreResult(kind="trades", label=label, n_trades=0, total_pnl=0.0)

    wins = [p for p in vals if p > 0.0]
    losses = [p for p in vals if p < 0.0]
    gross_profit = math.fsum(wins)
    gross_loss = math.fsum(losses)  # <= 0

    win_rate = len(wins) / n
    avg_win: Optional[float] = (gross_profit / len(wins)) if wins else None
    avg_loss: Optional[float] = (gross_loss / len(losses)) if losses else None
    if gross_loss < 0.0:
        profit_factor: Optional[float] = gross_profit / -gross_loss
    else:
        profit_factor = None  # no losing trades → undefined (would be +inf)

    return ScoreResult(
        kind="trades",
        label=label,
        n_trades=n,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        total_pnl=math.fsum(vals),
    )
