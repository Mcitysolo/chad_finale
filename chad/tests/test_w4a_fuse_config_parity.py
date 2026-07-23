"""W4A-2 — fuse-box config maps: registry parity + sector coverage (hard leg).

W3B-9 perimeter idiom: config drift must never brick the engine (the runtime
bucket builder warns), but CI fails loud. Pins:
- families union ⊆ DECLARED and ⊇ ACTIVE (no undeclared name may enter a
  fuse bucket; every active strategy must belong to a family);
- families are disjoint (a strategy in two families would double-trip);
- setup-fuse strategies are declared and family-covered;
- every config/universe.json symbol (equities + futures roots) and every
  crypto-lane pair has a sector row (a symbol added to the universe without
  a sector lands in the never-trippable sector:unmapped bucket at runtime —
  this test turns that silent drain into a CI failure);
- sector map has no duplicate symbol across sectors.
"""

from __future__ import annotations

import json
from pathlib import Path

from chad.risk.fuse_box import FuseBoxConfig
from chad.strategy_registry import active_strategy_values
from chad.types import StrategyName

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

ACTIVE = set(active_strategy_values())
DECLARED = {s.value for s in StrategyName}

CFG = FuseBoxConfig.load()
SECTORS = json.loads(
    (REPO_ROOT / "config" / "symbol_sectors.json").read_text()
)["sectors"]


def _family_union() -> set:
    out: set = set()
    for members in CFG.families.values():
        out |= set(members)
    return out


def test_families_bounded_by_registry():
    union = _family_union()
    assert union <= DECLARED, f"undeclared names: {sorted(union - DECLARED)}"
    assert ACTIVE <= union, f"ACTIVE names missing a family: {sorted(ACTIVE - union)}"


def test_families_disjoint():
    seen: dict = {}
    for fam, members in CFG.families.items():
        for m in members:
            assert m not in seen, f"{m} in both {seen[m]} and {fam}"
            seen[m] = fam


def test_setup_fuse_strategies_declared_and_family_covered():
    setups = set(CFG.setup_fuse_strategies)
    assert setups, "setup_fuses.enabled_strategies must not be empty (P13)"
    assert setups <= DECLARED, sorted(setups - DECLARED)
    assert setups <= _family_union(), sorted(setups - _family_union())


def _sector_symbols() -> set:
    out: set = set()
    for symbols in SECTORS.values():
        out |= set(symbols)
    return out


def test_sector_map_covers_universe():
    uni = json.loads((REPO_ROOT / "config" / "universe.json").read_text())
    equities = set(uni.get("symbols") or [])
    futures = {f["symbol"] for f in uni.get("futures") or []}
    mapped = _sector_symbols()
    missing = (equities | futures) - mapped
    assert not missing, (
        f"universe symbols without a sector row (would drain into the "
        f"never-trippable sector:unmapped bucket): {sorted(missing)}"
    )


def test_sector_map_covers_crypto_lane():
    mapped = _sector_symbols()
    assert {"BTC-USD", "ETH-USD", "SOL-USD"} <= mapped


def test_sector_map_no_duplicate_symbols():
    seen: dict = {}
    for sector, symbols in SECTORS.items():
        for s in symbols:
            assert s not in seen, f"{s} in both {seen[s]} and {sector}"
            seen[s] = sector


def test_sector_names_reserved_unmapped():
    """'unmapped' is the runtime bucket for symbols missing from this map —
    a config row by that name would collide with the never-trippable
    accounting bucket."""
    assert "unmapped" not in SECTORS
