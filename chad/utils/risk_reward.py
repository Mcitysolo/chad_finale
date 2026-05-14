from __future__ import annotations

from typing import Optional

MIN_RR_RATIO: float = 1.5
"""Minimum acceptable reward-to-risk ratio for new entries."""


def compute_rr_ratio(
    target_pts: float,
    stop_pts: float,
) -> Optional[float]:
    """Return reward/risk ratio, or None if inputs are degenerate.

    Parameters
    ----------
    target_pts:
        Expected profit distance in price points. Must be positive.
    stop_pts:
        Stop-loss distance in price points. Must be positive.

    Returns
    -------
    float | None
        target_pts / stop_pts, or None when target_pts <= 0 or stop_pts <= 0.
    """
    if stop_pts <= 0.0 or target_pts <= 0.0:
        return None
    return float(target_pts) / float(stop_pts)


def passes_rr_gate(
    target_pts: float,
    stop_pts: float,
    *,
    min_ratio: float = MIN_RR_RATIO,
) -> bool:
    """Return True when the trade meets the minimum R:R threshold.

    This gate is fail-open on degenerate inputs. A missing or invalid target
    estimate must not block an otherwise valid signal.
    """
    ratio = compute_rr_ratio(target_pts, stop_pts)
    if ratio is None:
        return True
    return ratio >= float(min_ratio)


__all__ = [
    "MIN_RR_RATIO",
    "compute_rr_ratio",
    "passes_rr_gate",
]
