from __future__ import annotations

import json
from pathlib import Path

from chad.portfolio.portfolio_engine import EnginePaths, PortfolioEngine


def test_portfolio_engine_targets_and_rebalance_smoke(tmp_path: Path) -> None:
    # Minimal smoke test: engine can run with env-based paths when files exist.
    # We do NOT fabricate full snapshots here; this test checks method availability + return keys.

    # This repo’s engine reads from paths.config_dir/runtime_dir; we just validate function presence.
    paths = EnginePaths(
        repo_dir=Path("/home/ubuntu/chad_finale"),
        runtime_dir=Path("/home/ubuntu/CHAD FINALE/runtime"),
        config_dir=Path("/home/ubuntu/CHAD FINALE/config"),
    )
    eng = PortfolioEngine(paths=paths)

    t = eng.get_targets("BALANCED")
    assert isinstance(t, dict)
    assert "ok" in t
    assert "targets" in t

    r = eng.get_rebalance_latest("BALANCED")
    assert isinstance(r, dict)
    assert "ok" in r
    assert "diffs" in r
