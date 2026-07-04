"""chad/validation/regime_labeler.py — Phase 2 INDEPENDENT regime labeler.

The harness's own, dead-simple, fully-auditable market-regime ruler (SSOT
``docs/CHAD_EDGE_VALIDATION_HARNESS_DESIGN_v1.1.md`` §3.4 / V4). It slices results
by market mood so an edge that exists in only one regime is flagged
*regime-fragile* rather than passed.

CRITICAL — independence (the whole point of V4): this module DOES NOT and MUST NOT
import or call ``chad.analytics.regime_classifier`` (or any other CHAD regime
component). That classifier is the component we just proved emitted a false
``trending_bear``; letting the defendant pick the judge is exactly the circularity
V4 forbids. CHAD's own label may be carried alongside for *side-by-side reporting
only* (the optional ``chad_labels`` passthrough), never as the labeling authority.
The isolation test (SSOT §2) enforces this transitively — any such import would pull
``chad.analytics.*`` into the harness closure and fail the allowlist check — and an
AST import-scan test asserts no ``import`` statement in this file references it (the
name legitimately appears in this docstring, so the scan inspects import nodes only,
not raw text).

Definition (trailing benchmark return sign × realized-vol tercile):
  For a benchmark price series (e.g. SPY closes) and a trailing ``lookback`` window:
    * trailing_return[t] = price[t] / price[t - lookback] - 1   (causal; uses only past)
    * realized_vol[t]    = sample stdev of the ``lookback`` simple returns ending at t
    * trend  = FLAT if |trailing_return| < flat_threshold, else BULL (>0) / BEAR (<0)
    * vol    = "vol" if realized_vol is in the top tercile of the series, else "calm"
  giving one of {bull_calm, bull_vol, bear_calm, bear_vol, flat}. FLAT carries no
  vol suffix (a directionless market is flat regardless of its vol).

On the vol tercile being a full-sample statistic: the tercile cut is computed over
the whole provided series. This is legitimate here and is NOT lookahead in the
sense that matters, because the labeler is a **post-hoc descriptive slicer** of
already-computed results — its labels are never fed back to any strategy as a live
signal (that would be the lookahead the backtest engine, Phase 4 §3.8, structurally
forbids). It answers "relative to this whole history, was this a high-vol day?" —
a cross-sectional attribution question. The per-bar trend and vol computations are
themselves causal (trailing window only); only the tercile threshold is global, and
that is by design and documented.

Isolation (SSOT §1.2 / §2): pure, offline, deterministic, standard-library only —
no numpy, no broker, no ``runtime/`` reader, no live-loop / analytics dependency.

Sentinel / raise convention (mirrors the rest of the harness): degenerate-but-valid
data never raises. Bars without enough trailing history, or whose trailing window
contains a non-positive price, are labeled :attr:`Regime.UNKNOWN` (the honest "can't
label" sentinel, analogous to ``INSUFFICIENT_DATA``); a series with zero vol
dispersion classifies every bar as ``calm`` (no meaningful tercile) with a recorded
warning. Only invalid *configuration* raises ``ValueError`` (``lookback < 2``,
``flat_threshold < 0``, ``vol_high_quantile`` outside ``(0, 1)``, a ``chad_labels``
length that does not match the price series).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Sequence, Union

__all__ = [
    "Regime",
    "RegimeConfig",
    "DEFAULT_REGIME_CONFIG",
    "RegimeLabel",
    "RegimeSeries",
    "label_series",
]

Number = Union[int, float]


class Regime(Enum):
    """The independent regime taxonomy (SSOT §3.4). ``UNKNOWN`` is the sentinel
    for a bar that cannot be labeled (insufficient trailing history / bad price)."""

    BULL_CALM = "bull_calm"
    BULL_VOL = "bull_vol"
    BEAR_CALM = "bear_calm"
    BEAR_VOL = "bear_vol"
    FLAT = "flat"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RegimeConfig:
    """Deterministic, conservative parameters for the independent regime ruler.

    ``lookback`` is the trailing window (in bars) for both the trend return and the
    realized-vol estimate; it must be ``>= 2`` (a sample stdev needs two returns).
    ``flat_threshold`` is the absolute trailing-return band (as a fraction over the
    lookback) below which the market is called FLAT. ``vol_high_quantile`` is the
    tercile cut for "vol" vs "calm" — ``2/3`` means the top third of realized-vol
    bars are "vol". No parameter depends on wall-clock time.
    """

    lookback: int = 20
    flat_threshold: float = 0.02
    vol_high_quantile: float = 2.0 / 3.0

    def __post_init__(self) -> None:
        if isinstance(self.lookback, bool) or not isinstance(self.lookback, int):
            raise ValueError(f"lookback must be an int, got {self.lookback!r}")
        if self.lookback < 2:
            raise ValueError(f"lookback must be >= 2, got {self.lookback}")
        if isinstance(self.flat_threshold, bool) or not isinstance(
            self.flat_threshold, (int, float)
        ):
            raise ValueError(f"flat_threshold must be a real number, got {self.flat_threshold!r}")
        if self.flat_threshold < 0.0:
            raise ValueError(f"flat_threshold must be >= 0, got {self.flat_threshold}")
        if isinstance(self.vol_high_quantile, bool) or not isinstance(
            self.vol_high_quantile, (int, float)
        ):
            raise ValueError(
                f"vol_high_quantile must be a real number, got {self.vol_high_quantile!r}"
            )
        if not (0.0 < float(self.vol_high_quantile) < 1.0):
            raise ValueError(
                f"vol_high_quantile must be in (0, 1), got {self.vol_high_quantile}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lookback": self.lookback,
            "flat_threshold": self.flat_threshold,
            "vol_high_quantile": self.vol_high_quantile,
        }


DEFAULT_REGIME_CONFIG: RegimeConfig = RegimeConfig()


@dataclass(frozen=True)
class RegimeLabel:
    """The label (and its raw inputs) for a single bar index.

    ``trailing_return`` and ``realized_vol`` are ``None`` for an ``UNKNOWN`` bar.
    ``vol_high`` is the boolean tercile membership used to pick calm/vol (``None`` for
    ``UNKNOWN``, and also ``None`` when the series had no vol dispersion). ``chad_label``
    is the optional side-by-side passthrough — NEVER an input to the labeling.
    """

    index: int
    regime: Regime
    trailing_return: Optional[float]
    realized_vol: Optional[float]
    vol_high: Optional[bool]
    chad_label: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "regime": self.regime.value,
            "trailing_return": self.trailing_return,
            "realized_vol": self.realized_vol,
            "vol_high": self.vol_high,
            "chad_label": self.chad_label,
        }


@dataclass(frozen=True)
class RegimeSeries:
    """The full labeling of a price series plus provenance for a report.

    ``vol_threshold`` is the realized-vol tercile cut actually used (``None`` when the
    series had no vol dispersion, so every labelable bar was classified ``calm``).
    ``counts`` maps each :class:`Regime` value to its occurrence count.
    """

    labels: tuple[RegimeLabel, ...]
    vol_threshold: Optional[float]
    n_labeled: int
    counts: dict[str, int]
    config_echo: dict[str, Any]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": [label.to_dict() for label in self.labels],
            "vol_threshold": self.vol_threshold,
            "n_labeled": self.n_labeled,
            "counts": self.counts,
            "config_echo": self.config_echo,
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------- #
# Small pure numeric helpers — hand-rolled for transparent re-derivation and
# deterministic summation order (same discipline as scoring_spine).
# --------------------------------------------------------------------------- #
def _sample_stdev(returns: Sequence[float]) -> float:
    """Sample standard deviation (ddof=1) of ``returns``. Caller guarantees len>=2."""
    n = len(returns)
    mean = math.fsum(returns) / n
    variance = math.fsum((r - mean) ** 2 for r in returns) / (n - 1)
    return math.sqrt(variance)


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile (type 7) of an ascending ``sorted_vals``.

    ``q`` in ``[0, 1]``. Caller guarantees a non-empty, ascending sequence. Matches
    the common (numpy-default) definition so the tercile cut is reproducible.
    """
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _as_float(x: Number) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"price series must contain real numbers, got {type(x).__name__}: {x!r}")
    return float(x)


def label_series(
    prices: Sequence[Number],
    *,
    config: RegimeConfig = DEFAULT_REGIME_CONFIG,
    chad_labels: Optional[Sequence[Optional[str]]] = None,
) -> RegimeSeries:
    """Label each bar of a benchmark ``prices`` series with an independent regime.

    Pure function of the price series (SSOT §3.4). For each bar ``t >= lookback`` it
    computes a causal trailing return and realized vol; the trend gives BULL / BEAR /
    FLAT and the top realized-vol tercile gives the calm/vol suffix. Bars with
    ``t < lookback`` — or whose trailing window contains a non-positive price — are
    :attr:`Regime.UNKNOWN`. Deterministic; never raises on data (only on invalid
    config, see the module docstring).

    ``chad_labels``, if given, must be the same length as ``prices``; each entry is
    attached to the corresponding :class:`RegimeLabel` as ``chad_label`` for
    side-by-side reporting ONLY — it never influences the independent label.
    """
    if not isinstance(config, RegimeConfig):
        raise ValueError(f"config must be a RegimeConfig, got {type(config).__name__}")

    price_list = [_as_float(p) for p in prices]
    n = len(price_list)

    if chad_labels is not None and len(chad_labels) != n:
        raise ValueError(
            f"chad_labels length ({len(chad_labels)}) must match prices length ({n})"
        )

    def _chad_at(i: int) -> Optional[str]:
        if chad_labels is None:
            return None
        raw = chad_labels[i]
        return None if raw is None else str(raw)

    lookback = config.lookback
    warnings: list[str] = []

    # --- empty / too-short series → all UNKNOWN (documented sentinel) -------- #
    if n == 0:
        warnings.append("empty price series; nothing to label")
        return RegimeSeries(
            labels=(),
            vol_threshold=None,
            n_labeled=0,
            counts={r.value: 0 for r in Regime},
            config_echo=config.to_dict(),
            warnings=tuple(warnings),
        )

    # --- Pass 1: causal trailing return + realized vol per labelable bar ---- #
    trailing_returns: list[Optional[float]] = [None] * n
    realized_vols: list[Optional[float]] = [None] * n
    for t in range(lookback, n):
        window = price_list[t - lookback : t + 1]  # lookback+1 prices → lookback returns
        if any(p <= 0.0 for p in window):
            # Non-positive price makes returns/vol undefined → leave this bar UNKNOWN.
            continue
        rets = [window[k + 1] / window[k] - 1.0 for k in range(lookback)]
        trailing_returns[t] = window[-1] / window[0] - 1.0
        realized_vols[t] = _sample_stdev(rets)

    labelable_vols = [v for v in realized_vols if v is not None]
    if n <= lookback or not labelable_vols:
        warnings.append(
            f"fewer than lookback+1={lookback + 1} usable prices; "
            "no bar has enough trailing history to label"
        )

    # --- vol tercile threshold over the whole series (post-hoc slicer) ------ #
    vol_threshold: Optional[float] = None
    if labelable_vols:
        vmin = min(labelable_vols)
        vmax = max(labelable_vols)
        if vmax > vmin:
            vol_threshold = _quantile(sorted(labelable_vols), config.vol_high_quantile)
        else:
            # No dispersion (e.g. constant returns) → the tercile is meaningless;
            # classify every labelable bar as calm rather than fabricating "vol".
            warnings.append(
                "realized-vol dispersion is zero; all labelable bars classified 'calm'"
            )

    # --- Pass 2: assign regimes -------------------------------------------- #
    labels: list[RegimeLabel] = []
    counts: dict[str, int] = {r.value: 0 for r in Regime}
    n_labeled = 0
    for t in range(n):
        tr = trailing_returns[t]
        rv = realized_vols[t]
        if tr is None or rv is None:
            regime = Regime.UNKNOWN
            vol_high: Optional[bool] = None
        else:
            n_labeled += 1
            vol_high = vol_threshold is not None and rv >= vol_threshold
            # An exactly-zero trailing return is directionless → FLAT regardless of
            # flat_threshold (so a valid flat_threshold=0.0 does not asymmetrically
            # fold a 0.0 return into BEAR via the tr>0.0 test below).
            if tr == 0.0 or abs(tr) < config.flat_threshold:
                regime = Regime.FLAT
            elif tr > 0.0:
                regime = Regime.BULL_VOL if vol_high else Regime.BULL_CALM
            else:
                regime = Regime.BEAR_VOL if vol_high else Regime.BEAR_CALM
        counts[regime.value] += 1
        labels.append(
            RegimeLabel(
                index=t,
                regime=regime,
                trailing_return=tr,
                realized_vol=rv,
                vol_high=vol_high,
                chad_label=_chad_at(t),
            )
        )

    return RegimeSeries(
        labels=tuple(labels),
        vol_threshold=vol_threshold,
        n_labeled=n_labeled,
        counts=counts,
        config_echo=config.to_dict(),
        warnings=tuple(warnings),
    )
