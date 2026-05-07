"""
Tests for chad/execution/ibkr_client_ids.py (GAP-010).

Guards the canonical IBKR client id registry: every named id must exist,
no two names may share a value, the four target Python files must reference
the registry rather than re-introduce the literal ids, and the registered
values must match the historical values that systemd units / env config
have always used.
"""

from __future__ import annotations

from pathlib import Path

from chad.execution import ibkr_client_ids


REPO_ROOT = Path(__file__).resolve().parents[2]


# Names that callers (and the surrounding hardening checklist) rely on.
EXPECTED_REGISTRY_NAMES = {
    "LIVE_LOOP",
    "PRICE_PROVIDER",
    "HISTORICAL_PROVIDER",
    "PRICE_CACHE_REFRESH",
    "BAR_PROVIDER",
    "PORTFOLIO_COLLECTOR",
    "RECONCILER",
    "PAPER_FILL_HARVESTER",
    "BROKER_EVENTS_COLLECTOR",
}


# Values frozen by GAP-010: behavior must not change. If any of these flip,
# something has either renamed a service or accidentally re-pointed a
# connection.
EXPECTED_VALUES = {
    "LIVE_LOOP": 99,
    "PRICE_PROVIDER": 9030,
    "HISTORICAL_PROVIDER": 9034,
    "PRICE_CACHE_REFRESH": 9035,
    "BAR_PROVIDER": 9021,
    "PORTFOLIO_COLLECTOR": 9003,
    "PAPER_FILL_HARVESTER": 79,
    "PORTFOLIO_SNAPSHOT_PUBLISHER": 84,
    "RECONCILER": 83,
    "BROKER_EVENTS_COLLECTOR": 118,
    "HEALTHCHECK": 9001,
    "PAPER_SHADOW_RUNNER": 9013,
    "ADVISORY_ENGINE": 9036,
    "DASHBOARD_API": 80,
    "OPTIONS_CHAIN": 88,
    "NIGHTLY_BARS_REFRESH": 9053,
}


def test_ibkr_client_ids_have_no_collisions() -> None:
    # assert_no_collisions raises ValueError on duplicates; calling it
    # exercises the same logic the module runs at import time.
    ibkr_client_ids.assert_no_collisions()

    cmap = ibkr_client_ids.client_id_map()
    values = list(cmap.values())
    assert len(values) == len(set(values)), (
        f"duplicate client ids in registry: {sorted(values)}"
    )

    all_ids = ibkr_client_ids.all_client_ids()
    assert sorted(set(values)) == all_ids
    # No one is allowed to grab the IB Gateway "0" wildcard via the registry.
    assert all(cid > 0 for cid in all_ids), "client ids must be positive"


def test_ibkr_client_id_registry_contains_expected_names() -> None:
    cmap = ibkr_client_ids.client_id_map()
    missing = EXPECTED_REGISTRY_NAMES - set(cmap.keys())
    assert not missing, f"registry missing expected names: {sorted(missing)}"

    for name in EXPECTED_REGISTRY_NAMES:
        assert hasattr(ibkr_client_ids, name), (
            f"registry must export module-level constant {name}"
        )
        assert getattr(ibkr_client_ids, name) == cmap[name], (
            f"{name} module attr disagrees with client_id_map()"
        )


def test_ibkr_client_id_values_preserved() -> None:
    """Pin the historical values — GAP-010 must not change behavior."""
    cmap = ibkr_client_ids.client_id_map()
    for name, expected in EXPECTED_VALUES.items():
        assert name in cmap, f"registry missing constant {name}"
        assert cmap[name] == expected, (
            f"{name} changed: registry={cmap[name]} expected={expected}"
        )


def test_ibkr_python_files_do_not_hardcode_replaced_client_ids() -> None:
    """The four target files must reference the registry, not bare ints."""
    # (relative path, forbidden literal substring)
    forbidden = [
        ("chad/core/live_loop.py", "clientId=99"),
        ("chad/market_data/ibkr_price_provider.py", "clientId=9030"),
        ("chad/market_data/ibkr_historical_provider.py", "clientId=9034"),
        ("chad/market_data/price_cache_refresh.py", "clientId=9035"),
    ]
    for rel, literal in forbidden:
        path = REPO_ROOT / rel
        assert path.is_file(), f"target file not found: {rel}"
        text = path.read_text(encoding="utf-8")
        assert literal not in text, (
            f"{rel} still hardcodes '{literal}' — must use "
            f"chad.execution.ibkr_client_ids registry constant"
        )

    # And each target file must import from the registry.
    for rel, _ in forbidden:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "chad.execution.ibkr_client_ids" in text, (
            f"{rel} must import from chad.execution.ibkr_client_ids"
        )
