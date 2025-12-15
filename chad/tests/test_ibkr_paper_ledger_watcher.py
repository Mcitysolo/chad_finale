from pathlib import Path
import json

from chad.portfolio.ibkr_paper_ledger_watcher import load_config, LedgerConfig, run_once


def test_load_config_missing_file_defaults_disabled(tmp_path: Path):
    cfg = load_config(tmp_path / "missing.json")
    assert isinstance(cfg, LedgerConfig)
    assert cfg.enabled is False


def test_run_once_disabled_writes_preview_only(tmp_path: Path, monkeypatch):
    # Force config disabled and redirect report dir + state path
    cfg = LedgerConfig(
        enabled=False,
        state_path=tmp_path / "state.json",
        reports_dir=tmp_path / "reports",
    )
    rep = run_once(cfg)
    assert rep["enabled"] is False
    assert "preview_path" in rep
    preview = Path(rep["preview_path"])
    assert preview.is_file()
    j = json.loads(preview.read_text())
    assert j["enabled"] is False
    # No state file should be required when disabled
    assert not (tmp_path / "state.json").exists()
