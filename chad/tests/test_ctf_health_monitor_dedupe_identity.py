"""CTF-T2: health-monitor dedupe must key on alert IDENTITY, not message text.

Regression guard for the duplicate-delivery defect: the dedupe key used to be
``f"health_{rule_id}_{title[:30]}"``, which embedded fluctuating numeric values
from the finding title (e.g. "SCR gap 214 raw vs 67 effective"). Every cycle the
value changed, the key changed, and the same finding was re-delivered — observed
on disk as R13 SCR-gap producing 7 distinct dedupe files, sent ~every 5 minutes.

These tests pin the invariant: findings that differ ONLY in fluctuating numeric
values collapse to the SAME dedupe key (so they dedupe), while findings about
DIFFERENT entities keep DISTINCT keys (so real alerts are not swallowed).
"""
from chad.ops.health_monitor import _alert_dedupe_key
from chad.ops.health_monitor_rules import Finding


def _f(rule_id: str, title: str) -> Finding:
    return Finding(
        rule_id=rule_id, severity="WARNING", title=title,
        description="", remedy_type="NOTIFY_ONLY", remedy_action="notify",
    )


def test_fluctuating_value_yields_stable_key():
    # R13 SCR-gap: the gap value churns cycle to cycle — this was the bug.
    k1 = _alert_dedupe_key(_f("R13", "SCR gap 200 raw vs 67 effective"))
    k2 = _alert_dedupe_key(_f("R13", "SCR gap 214 raw vs 67 effective"))
    k3 = _alert_dedupe_key(_f("R13", "SCR gap 217 raw vs 67 effective"))
    assert k1 == k2 == k3


def test_feed_stale_age_does_not_change_key():
    # R02 feed staleness embeds a live age in seconds.
    a = _alert_dedupe_key(_f("R02", "Feed STALE (notify): var_state.json (12345s old, TTL=3600s)"))
    b = _alert_dedupe_key(_f("R02", "Feed STALE (notify): var_state.json (99999s old, TTL=3600s)"))
    assert a == b


def test_disk_pct_does_not_change_key():
    a = _alert_dedupe_key(_f("R07", "Disk 87.3% full — WARNING"))
    b = _alert_dedupe_key(_f("R07", "Disk 91.8% full — WARNING"))
    assert a == b


def test_different_entities_keep_distinct_keys():
    # Same rule, different feed/strategy/service -> must NOT collide.
    assert _alert_dedupe_key(_f("R02", "Feed STALE (notify): var_state.json (1s old)")) != \
           _alert_dedupe_key(_f("R02", "Feed STALE (notify): regime_state.json (1s old)"))
    # The old [:30] truncation collapsed alpha_crypto and alpha_intraday; the
    # identity key (digits stripped, wider slice) keeps them distinct.
    assert _alert_dedupe_key(_f("R09", "Edge decay halt active: alpha_crypto")) != \
           _alert_dedupe_key(_f("R09", "Edge decay halt active: alpha_intraday"))
    assert _alert_dedupe_key(_f("R09", "Edge decay halt active: gamma")) != \
           _alert_dedupe_key(_f("R09", "Edge decay halt active: omega"))


def test_different_rules_keep_distinct_keys():
    assert _alert_dedupe_key(_f("R09", "Edge decay halt active: gamma")) != \
           _alert_dedupe_key(_f("R20", "Edge decay halt active: gamma"))


def test_digits_inside_identifier_survive():
    # Futures-style symbols carry digits that ARE identity (M6E, M2K) — a digit
    # preceded by a letter is preserved, so these stay distinct.
    assert _alert_dedupe_key(_f("R05", "Reconciliation RED: M6E")) != \
           _alert_dedupe_key(_f("R05", "Reconciliation RED: M2K"))


def test_key_is_prefixed_and_bounded():
    k = _alert_dedupe_key(_f("R20", "IBKR Gateway version is stale"))
    assert k.startswith("health_R20_")
    assert len(k) <= len("health_R20_") + 48
