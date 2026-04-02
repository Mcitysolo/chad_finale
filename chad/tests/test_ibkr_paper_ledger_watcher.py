from pathlib import Path
import json

from chad.portfolio.ibkr_paper_ledger_watcher import (
    LedgerConfig,
    OpenStateStore,
    PaperLedgerWatcher,
    PlanAttributionResolver,
    StrategyAttributionService,
)


class _StubGateway:
    """Minimal BrokerGateway stub — never called when config is disabled."""
    def connect(self):
        raise AssertionError("connect should not be called when disabled")

    def disconnect(self):
        pass

    def current_positions(self):
        return []

    def recent_fills(self):
        return []


def test_load_config_missing_file_defaults_disabled(tmp_path: Path):
    cfg = LedgerConfig.load(tmp_path / "missing.json")
    assert isinstance(cfg, LedgerConfig)
    assert cfg.enabled is False


def test_run_once_disabled_writes_report(tmp_path: Path):
    cfg = LedgerConfig(
        enabled=False,
        state_path=tmp_path / "state.json",
        reports_dir=tmp_path / "reports",
        plan_artifact_path=tmp_path / "plan.json",
    )
    state_store = OpenStateStore(cfg.state_path)
    resolver = PlanAttributionResolver(cfg.plan_artifact_path)
    attribution = StrategyAttributionService(cfg, resolver)
    gateway = _StubGateway()

    watcher = PaperLedgerWatcher(cfg, gateway, state_store, attribution)
    report = watcher.run_once()

    assert report.config_enabled is False
    assert report.ok is True
    assert any("disabled" in w for w in report.warnings)

    # Report file should be written to reports_dir
    report_files = list((tmp_path / "reports").glob("*.json"))
    assert len(report_files) == 1
    j = json.loads(report_files[0].read_text())
    assert j["config_enabled"] is False

    # No state file should be created when disabled
    assert not (tmp_path / "state.json").exists()
