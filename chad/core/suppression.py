#!/usr/bin/env python3
"""
chad/core/suppression.py

Canonical suppression reason codes for CHAD (SSOT v6.4).

These codes are the standard vocabulary for why a signal, strategy, or
intent was rejected at any point in the pipeline.  Each code maps to a
specific rejection site:

    NO_SIGNAL                   - strategy_router: strategy had zero signals
    AFFORDABILITY_FAILED        - policy / execution: notional exceeds position sizing
    DUPLICATE_SIGNAL            - signal_guard: identical fingerprint already active
    SAME_SIDE_POSITION_OPEN     - position_guard: same strategy+symbol+side already open
    BROKER_TRUTH_CONFLICT       - live_loop broker rebuild: guard entry closed by broker truth
    COOLDOWN_ACTIVE             - signal_guard: fingerprint within 10-min cooldown window
    LOWER_PRIORITY_THAN_SELECTED - strategy_router: lost weight/preference competition
    POLICY_BLOCKED              - policy engine: violated a risk policy rule
    HARD_CAP_BLOCKED            - daily_throttle: global/symbol/strategy notional cap exceeded

Callers adopt these codes incrementally.  The enum is the single source
of truth for suppression vocabulary — ad-hoc reason strings should be
migrated to these codes over time.
"""

from __future__ import annotations

from enum import Enum


class SuppressionReason(str, Enum):
    NO_SIGNAL = "no_signal"
    AFFORDABILITY_FAILED = "affordability_failed"
    DUPLICATE_SIGNAL = "duplicate_signal"
    SAME_SIDE_POSITION_OPEN = "same_side_position_open"
    BROKER_TRUTH_CONFLICT = "broker_truth_conflict"
    COOLDOWN_ACTIVE = "cooldown_active"
    LOWER_PRIORITY_THAN_SELECTED = "lower_priority_than_selected"
    POLICY_BLOCKED = "policy_blocked"
    HARD_CAP_BLOCKED = "hard_cap_blocked"
