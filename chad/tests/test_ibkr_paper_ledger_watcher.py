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


def test_load_config_reads_ibkr_block(tmp_path: Path):
    """LedgerConfig.load must pick up host/port/client_id from the ibkr block."""
    cfg_path = tmp_path / "ibkr_paper_ledger.json"
    cfg_path.write_text(json.dumps({
        "enabled": True,
        "ibkr": {
            "host": "127.0.0.1",
            "port": 4002,
            "client_id": 9040,
        },
    }))
    cfg = LedgerConfig.load(cfg_path)
    assert cfg.ibkr_host == "127.0.0.1"
    assert cfg.ibkr_port == 4002
    assert cfg.ibkr_client_id == 9040


def test_load_config_does_not_resolve_to_zero_when_ibkr_block_present(tmp_path: Path):
    """When the ibkr block is present with a non-zero client_id, the loaded
    config must not silently fall back to clientId=0 (the IB Gateway wildcard,
    which is unsafe for the watcher service)."""
    cfg_path = tmp_path / "ibkr_paper_ledger.json"
    cfg_path.write_text(json.dumps({
        "enabled": True,
        "ibkr": {
            "host": "127.0.0.1",
            "port": 4002,
            "client_id": 9040,
        },
    }))
    cfg = LedgerConfig.load(cfg_path)
    assert cfg.ibkr_client_id != 0
    assert cfg.ibkr_client_id == 9040


def test_load_config_uses_registry_constant_for_ledger_watcher():
    """The runtime config in this repo must point at the registered
    LEDGER_WATCHER client id, so a future rename of the constant fails the
    test and forces the config to be updated together."""
    from chad.execution.ibkr_client_ids import LEDGER_WATCHER
    repo_cfg = Path(__file__).resolve().parents[2] / "runtime" / "ibkr_paper_ledger.json"
    if not repo_cfg.is_file():
        # The runtime file is gitignored; skip cleanly when not present.
        return
    raw = json.loads(repo_cfg.read_text())
    ibkr = raw.get("ibkr") or {}
    assert ibkr.get("client_id") == LEDGER_WATCHER, (
        f"runtime/ibkr_paper_ledger.json client_id={ibkr.get('client_id')} "
        f"must equal ibkr_client_ids.LEDGER_WATCHER={LEDGER_WATCHER}"
    )
    assert ibkr.get("client_id") != 0, "client_id=0 is unsafe for the watcher"


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
