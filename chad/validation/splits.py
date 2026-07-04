"""chad/validation/splits.py — Phase 2 partition + purged, embargoed walk-forward.

The train/validation/OOS partition math and the purged + embargoed walk-forward
window generator for the edge-validation harness (SSOT
``docs/CHAD_EDGE_VALIDATION_HARNESS_DESIGN_v1.1.md`` §3.7). Its one job is to
carve bar-index ranges such that **no label leaks from the future into the past**:
a bar whose label (a multi-bar holding period) reaches into a test range is purged
from the adjacent train range, and a configurable embargo adds a further gap.

This module is deliberately NOT the OOS lockbox (that hash-seal / ``--final-run``
enforcement is Phase 5, SSOT §3.1). It only computes deterministic index ranges;
it never reads or writes any file, never touches ``runtime/``, and imports only
the standard library (SSOT §1.2 / §2 isolation).

--------------------------------------------------------------------------------
The leakage model (documented once, enforced by construction)
--------------------------------------------------------------------------------
A **label horizon** ``h`` (per head; a holding period in bars) means the label of
a bar at index ``i`` is realised over the forward window ``[i, i+h]`` — it depends
on bars up to ``i + h``. If ``i + h >= test_start`` that label was computed using
bars inside the test set: leakage. So any train bar with ``i + h >= test_start`` is
**purged**. An **embargo** ``e`` removes a further ``e`` bars immediately before the
test set, guarding against residual serial correlation across the boundary.

Concretely, a test range that begins at ``test_start`` keeps train bars only up to
(exclusive) ``test_start - h - e``. Equivalently the effective train end is pulled
back by exactly ``h + e`` bars from the naive boundary. This makes the guarantee
``max(train_index) + h < test_start`` hold for every window — asserted in tests.

The purge is applied on the TRAIN side (the set being fit); the test/OOS range is
kept intact. Leakage direction is past→future *via labels*, so it is the training
bars whose forward-looking labels must be removed, not the test bars.

Sentinel / raise convention (mirrors ``scoring_spine``/``cost_model`` and SSOT
honesty): degenerate-but-valid sizing (too few bars to form any window, an empty
partition slice) never raises — it yields an empty window list / empty index tuple
and records a ``warning``. Only invalid *configuration* raises ``ValueError``
(non-positive sizes, negative horizon/embargo, out-of-range fractions, unknown
mode, non-integer counts). ``label_horizon`` is a **required** keyword everywhere
(SSOT §3.7: "purge/embargo requires a declared per-head label horizon") — pass
``0`` explicitly only for a single-bar label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "Partition",
    "WalkForwardWindow",
    "partition",
    "generate_walk_forward",
]

WalkForwardMode = Literal["rolling", "expanding"]


# --------------------------------------------------------------------------- #
# Validation helpers — invalid CONFIG fails fast (never silently coerced).
# --------------------------------------------------------------------------- #
def _require_int(name: str, value: Any, *, minimum: int) -> int:
    """Require a plain ``int`` (reject bool) ``>= minimum``, else ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_fraction(name: str, value: Any) -> float:
    """Require a real number strictly inside ``(0, 1)``, else ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    v = float(value)
    if not (0.0 < v < 1.0):
        raise ValueError(f"{name} must be in the open interval (0, 1), got {v}")
    return v


# --------------------------------------------------------------------------- #
# Partition — chronological train / validation / OOS with boundary purge+embargo.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Partition:
    """A chronological three-way split of ``n`` bar indices with boundary purging.

    ``*_range`` are the raw half-open contiguous slices ``[start, end)`` in time
    order (train before val before oos). ``*_indices`` are the usable indices after
    purging + embargo: ``train_indices`` drops its tail whose labels reach into val,
    ``val_indices`` drops its tail whose labels reach into oos, and ``oos_indices``
    is the full oos slice (nothing follows it to leak into). All tuples are sorted
    ascending and mutually disjoint.
    """

    n: int
    train_range: tuple[int, int]
    val_range: tuple[int, int]
    oos_range: tuple[int, int]
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    oos_indices: tuple[int, ...]
    label_horizon: int
    embargo: int
    train_frac: float
    val_frac: float
    oos_frac: float
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict echo (JSON-serialisable; index tuples become lists)."""
        return {
            "n": self.n,
            "train_range": list(self.train_range),
            "val_range": list(self.val_range),
            "oos_range": list(self.oos_range),
            "train_indices": list(self.train_indices),
            "val_indices": list(self.val_indices),
            "oos_indices": list(self.oos_indices),
            "label_horizon": self.label_horizon,
            "embargo": self.embargo,
            "train_frac": self.train_frac,
            "val_frac": self.val_frac,
            "oos_frac": self.oos_frac,
            "warnings": list(self.warnings),
        }


def partition(
    n: int,
    *,
    label_horizon: int,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    oos_frac: float = 0.2,
    embargo: int = 0,
) -> Partition:
    """Split ``n`` bars chronologically into train/val/OOS with boundary purging.

    Boundaries are ``t = floor(n*train_frac)`` and ``v = t + floor(n*val_frac)``;
    ranges are ``train=[0,t)``, ``val=[t,v)``, ``oos=[v,n)`` (deterministic). Then
    a gap of ``label_horizon + embargo`` bars is purged from the *tail* of train
    (its labels would reach into val) and from the tail of val (into oos), so no
    train/val label window overlaps the next slice.

    Fractions must be positive and sum to 1.0 (within 1e-9). Degenerate sizing
    (an empty slice, or a purge that empties train/val) never raises — it records a
    ``warning`` and returns empty index tuples. Only invalid config raises
    ``ValueError``: non-positive/non-int ``n``, out-of-range or non-summing
    fractions, negative ``label_horizon``/``embargo``.
    """
    _require_int("n", n, minimum=1)
    h = _require_int("label_horizon", label_horizon, minimum=0)
    e = _require_int("embargo", embargo, minimum=0)
    tf = _require_fraction("train_frac", train_frac)
    vf = _require_fraction("val_frac", val_frac)
    of = _require_fraction("oos_frac", oos_frac)
    if abs((tf + vf + of) - 1.0) > 1e-9:
        raise ValueError(
            f"train_frac + val_frac + oos_frac must sum to 1.0, got {tf + vf + of}"
        )

    t = int(n * tf)
    v = t + int(n * vf)
    # Clamp defensively so rounding can never push v past n or below t.
    v = min(max(v, t), n)
    train_range = (0, t)
    val_range = (t, v)
    oos_range = (v, n)

    gap = h + e
    warnings: list[str] = []

    # Train keeps indices strictly below (t - gap): their label window [i, i+h] ends
    # before val_start=t, and the embargo trims a further `e` bars.
    train_cut = max(0, t - gap)
    train_indices = tuple(range(0, train_cut))
    # Val keeps indices strictly below (v - gap) within [t, v).
    val_cut = max(t, v - gap)
    val_indices = tuple(range(t, val_cut))
    oos_indices = tuple(range(v, n))

    if t == 0:
        warnings.append("train slice is empty (n too small for train_frac)")
    elif not train_indices:
        warnings.append("train slice fully purged by label_horizon+embargo gap")
    if v == t:
        warnings.append("validation slice is empty (n too small for val_frac)")
    elif not val_indices:
        warnings.append("validation slice fully purged by label_horizon+embargo gap")
    if v == n:
        warnings.append("oos slice is empty (n too small for oos_frac)")

    return Partition(
        n=n,
        train_range=train_range,
        val_range=val_range,
        oos_range=oos_range,
        train_indices=train_indices,
        val_indices=val_indices,
        oos_indices=oos_indices,
        label_horizon=h,
        embargo=e,
        train_frac=tf,
        val_frac=vf,
        oos_frac=of,
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------- #
# Purged + embargoed walk-forward windows.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WalkForwardWindow:
    """One walk-forward window: a train range strictly before a test range.

    ``train_start``/``train_end`` and ``test_start``/``test_end`` are half-open. The
    gap ``test_start - train_end == label_horizon + embargo`` is the purge+embargo
    region (bars belonging to neither set), so ``max(train_indices) + label_horizon
    < test_start`` by construction — no label leaks into the test set.
    ``purged_count`` is the number of bars removed from the naive (gap-free) train.
    """

    index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    label_horizon: int
    embargo: int
    purged_count: int
    mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "train_indices": list(self.train_indices),
            "test_indices": list(self.test_indices),
            "label_horizon": self.label_horizon,
            "embargo": self.embargo,
            "purged_count": self.purged_count,
            "mode": self.mode,
        }


def generate_walk_forward(
    n: int,
    *,
    train_size: int,
    test_size: int,
    label_horizon: int,
    embargo: int = 0,
    step: int | None = None,
    mode: WalkForwardMode = "rolling",
) -> list[WalkForwardWindow]:
    """Generate the deterministic sequence of purged+embargoed walk-forward windows.

    Each window places a ``test_size`` test range after a ``train_size`` train range
    (``expanding`` mode anchors train at index 0 and grows it instead). A gap of
    ``label_horizon + embargo`` bars separates train from test, so every train bar's
    label window ends before ``test_start`` (no leakage). Windows advance by ``step``
    (default ``test_size`` → non-overlapping test blocks) until the test range would
    exceed ``n``.

    Returns ``[]`` (documented sentinel, no raise) when the series is too short to
    fit even one window (``train_size + label_horizon + embargo + test_size > n``).
    Raises ``ValueError`` only on invalid config: non-positive ``train_size`` /
    ``test_size`` / ``step``, non-int / negative ``n`` / ``label_horizon`` /
    ``embargo``, or an unknown ``mode``.
    """
    _require_int("n", n, minimum=1)
    _require_int("train_size", train_size, minimum=1)
    _require_int("test_size", test_size, minimum=1)
    h = _require_int("label_horizon", label_horizon, minimum=0)
    e = _require_int("embargo", embargo, minimum=0)
    if step is None:
        step = test_size
    else:
        _require_int("step", step, minimum=1)
    if mode not in ("rolling", "expanding"):
        raise ValueError(f"mode must be 'rolling' or 'expanding', got {mode!r}")

    gap = h + e
    windows: list[WalkForwardWindow] = []
    # The first test block sits after a full train block plus the gap.
    first_test_start = train_size + gap
    k = 0
    while True:
        test_start = first_test_start + k * step
        test_end = test_start + test_size
        if test_end > n:
            break
        # Effective (post-purge) train end is pulled back by the gap from test_start.
        train_end = test_start - gap
        if mode == "expanding":
            train_start = 0
        else:  # rolling
            train_start = train_end - train_size
        # train_start >= 0 is guaranteed: train_end >= train_size (first window) and
        # grows with k, so rolling train_start = train_end - train_size >= 0.
        train_indices = tuple(range(train_start, train_end))
        test_indices = tuple(range(test_start, test_end))
        windows.append(
            WalkForwardWindow(
                index=k,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_indices=train_indices,
                test_indices=test_indices,
                label_horizon=h,
                embargo=e,
                purged_count=gap,
                mode=mode,
            )
        )
        k += 1

    return windows
