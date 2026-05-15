"""Phase B Item 6 — synthetic options Greeks metadata tests."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from chad.market_data.options_greeks_publisher import (
    GREEKS_FILE_NAME,
    GREEKS_SCHEMA_VERSION,
    build_greeks_payload,
)
from chad.utils.options_greeks_gate import (
    DEFAULT_CALL_DELTA,
    DEFAULT_PUT_DELTA,
    GREEKS_FILE_TTL,
    GreeksResult,
    get_option_greeks,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ALPHA_OPTIONS_SRC = REPO_ROOT / "chad" / "strategies" / "alpha_options.py"
CHAIN_REFRESH_SRC = REPO_ROOT / "chad" / "market_data" / "options_chain_refresh.py"
DEPLOY_DIR = REPO_ROOT / "deploy"


# ---------------------------------------------------------------------------
# Helpers — synthetic runtime dir
# ---------------------------------------------------------------------------


def _write_chain_cache(
    runtime_dir: Path,
    *,
    spot: float = 500.0,
    days_out: int = 30,
    schema_v2: bool = True,
) -> str:
    """Write a minimal v2 chain cache. Returns the expiry string used."""
    today = datetime.now(timezone.utc).date()
    expiry_date = today + timedelta(days=days_out)
    expiry_str = expiry_date.strftime("%Y%m%d")
    chain = {
        "symbol": "SPY",
        "exchange": "SMART",
        "expirations": [expiry_str],
        "strikes": [
            spot - 10.0, spot - 5.0, spot - 2.0,
            spot,
            spot + 2.0, spot + 5.0, spot + 10.0,
        ],
        "ts_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "ttl_seconds": 3600,
    }
    if schema_v2:
        chain["spot_price"] = float(spot)
    doc = {
        "ts_utc": chain["ts_utc"],
        "chains": {"SPY": chain},
    }
    if schema_v2:
        doc["schema_version"] = "options_chain_cache.v2"
    (runtime_dir / "options_chains_cache.json").write_text(
        json.dumps(doc, indent=2), encoding="utf-8"
    )
    return expiry_str


# ---------------------------------------------------------------------------
# 1. options_chain_refresh v2 payload preserves shape (static + helper).
# ---------------------------------------------------------------------------


def test_options_chain_refresh_v2_constants_present() -> None:
    src = CHAIN_REFRESH_SRC.read_text(encoding="utf-8")
    assert "options_chain_cache.v2" in src
    # spot_price must be persisted into the per-symbol payload
    assert '"spot_price"' in src
    # Old keys must remain (backward compatibility)
    for key in ('"symbol"', '"exchange"', '"expirations"', '"strikes"',
                '"ts_utc"', '"ttl_seconds"'):
        assert key in src, f"missing legacy key in refresh source: {key}"


# ---------------------------------------------------------------------------
# 2. options_greeks_publisher missing chain cache returns valid empty/error.
# ---------------------------------------------------------------------------


def test_publisher_missing_chain_returns_valid_error_schema(tmp_path: Path) -> None:
    payload = build_greeks_payload(runtime_dir=tmp_path)
    assert payload["schema_version"] == GREEKS_SCHEMA_VERSION
    assert payload["status"] in ("error", "partial")
    assert isinstance(payload["symbols"], dict) and not payload["symbols"]
    assert payload["summary"]["symbols_processed"] == 0
    assert payload["source"]["provider"] == "synthetic_black_scholes_vix"


# ---------------------------------------------------------------------------
# 3. Synthetic chain produces SPY symbol with expirations + strikes.
# ---------------------------------------------------------------------------


def test_publisher_synthetic_chain_produces_spy(tmp_path: Path) -> None:
    expiry = _write_chain_cache(tmp_path, spot=500.0, days_out=30)
    payload = build_greeks_payload(runtime_dir=tmp_path)
    assert payload["status"] in ("ok", "partial")
    spy = payload["symbols"].get("SPY")
    assert spy is not None
    assert spy["data_available"] is True
    assert expiry in spy["expirations"]
    strikes = spy["expirations"][expiry]["strikes"]
    assert len(strikes) >= 3


# ---------------------------------------------------------------------------
# 4. Nearest ATM strike is flagged near_atm=True; others False.
# ---------------------------------------------------------------------------


def test_publisher_marks_atm_strike(tmp_path: Path) -> None:
    expiry = _write_chain_cache(tmp_path, spot=500.0, days_out=30)
    payload = build_greeks_payload(runtime_dir=tmp_path)
    strikes = payload["symbols"]["SPY"]["expirations"][expiry]["strikes"]
    atm_flags = [v.get("near_atm") for v in strikes.values()]
    assert atm_flags.count(True) == 1
    assert any(v is False for v in atm_flags)


# ---------------------------------------------------------------------------
# 5. Call delta is positive and in [0.01, 0.99].
# ---------------------------------------------------------------------------


def test_publisher_call_delta_bounds(tmp_path: Path) -> None:
    expiry = _write_chain_cache(tmp_path, spot=500.0, days_out=30)
    payload = build_greeks_payload(runtime_dir=tmp_path)
    strikes = payload["symbols"]["SPY"]["expirations"][expiry]["strikes"]
    for v in strikes.values():
        cd = v.get("call_delta")
        assert cd is not None
        assert 0.01 <= cd <= 0.99, f"call_delta out of bounds: {cd}"


# ---------------------------------------------------------------------------
# 6. Put delta is negative and in [-0.99, -0.01].
# ---------------------------------------------------------------------------


def test_publisher_put_delta_bounds(tmp_path: Path) -> None:
    expiry = _write_chain_cache(tmp_path, spot=500.0, days_out=30)
    payload = build_greeks_payload(runtime_dir=tmp_path)
    strikes = payload["symbols"]["SPY"]["expirations"][expiry]["strikes"]
    for v in strikes.values():
        pd = v.get("put_delta")
        assert pd is not None
        assert -0.99 <= pd <= -0.01, f"put_delta out of bounds: {pd}"


# ---------------------------------------------------------------------------
# 7. get_option_greeks missing file returns default call delta +0.5.
# ---------------------------------------------------------------------------


def test_gate_missing_file_call_default(tmp_path: Path) -> None:
    result = get_option_greeks(
        "SPY", "20260619", 500.0, "C", runtime_dir=tmp_path
    )
    assert isinstance(result, GreeksResult)
    assert result.delta == DEFAULT_CALL_DELTA
    assert result.source == "default"


# ---------------------------------------------------------------------------
# 8. get_option_greeks missing file returns default put delta -0.5.
# ---------------------------------------------------------------------------


def test_gate_missing_file_put_default(tmp_path: Path) -> None:
    result = get_option_greeks(
        "SPY", "20260619", 500.0, "P", runtime_dir=tmp_path
    )
    assert result.delta == DEFAULT_PUT_DELTA
    assert result.source == "default"


# ---------------------------------------------------------------------------
# 9. Exact strike lookup works.
# ---------------------------------------------------------------------------


def test_gate_exact_strike_lookup(tmp_path: Path) -> None:
    expiry = _write_chain_cache(tmp_path, spot=500.0, days_out=30)
    # Publish greeks file
    from chad.market_data.options_greeks_publisher import _atomic_write
    payload = build_greeks_payload(runtime_dir=tmp_path)
    _atomic_write(tmp_path / GREEKS_FILE_NAME, payload)

    # ATM strike (spot=500) is among the strike set written
    result = get_option_greeks(
        "SPY", expiry, 500.0, "C", runtime_dir=tmp_path
    )
    assert result.source != "default"
    assert 0.01 <= result.delta <= 0.99
    assert result.near_atm is True


# ---------------------------------------------------------------------------
# 10. Nearest-strike fallback works.
# ---------------------------------------------------------------------------


def test_gate_nearest_strike_fallback(tmp_path: Path) -> None:
    expiry = _write_chain_cache(tmp_path, spot=500.0, days_out=30)
    from chad.market_data.options_greeks_publisher import _atomic_write
    payload = build_greeks_payload(runtime_dir=tmp_path)
    _atomic_write(tmp_path / GREEKS_FILE_NAME, payload)

    # 501.5 is not a stored strike (we wrote 490/495/498/500/502/505/510),
    # so the gate must fall through to the nearest neighbor.
    result = get_option_greeks(
        "SPY", expiry, 501.5, "C", runtime_dir=tmp_path
    )
    assert result.source != "default"
    assert 0.01 <= result.delta <= 0.99


# ---------------------------------------------------------------------------
# 11. alpha_options source contains new metadata keys.
# ---------------------------------------------------------------------------


def test_alpha_options_meta_keys_present_in_source() -> None:
    src = ALPHA_OPTIONS_SRC.read_text(encoding="utf-8")
    required = [
        '"long_delta"',
        '"short_delta"',
        '"net_delta_estimate"',
        '"long_theo_price"',
        '"short_theo_price"',
        '"long_delta_source"',
        '"short_delta_source"',
        "from chad.utils.options_greeks_gate import",
    ]
    for needle in required:
        assert needle in src, f"alpha_options missing: {needle}"


# ---------------------------------------------------------------------------
# 12. alpha_options sizing block unchanged (static assertions).
# ---------------------------------------------------------------------------


def test_alpha_options_sizing_block_unchanged() -> None:
    src = ALPHA_OPTIONS_SRC.read_text(encoding="utf-8")
    # The three sizing-critical fragments must still be present verbatim.
    assert "risk_budget = equity * tuning.max_risk_per_trade_pct" in src
    assert "contracts = int(risk_budget / spread.max_loss_per_contract)" in src
    assert "max_loss_per_contract" in src
    assert "net_debit_estimate" in src
    # The TradeSignal still passes size=float(contracts) — the metadata
    # patch must not have rerouted size off the contracts count.
    assert "size=float(contracts)" in src


# ---------------------------------------------------------------------------
# 13. Deploy service/timer files exist with expected directives.
# ---------------------------------------------------------------------------


def test_deploy_service_and_timer_present() -> None:
    svc = DEPLOY_DIR / "chad-options-greeks-refresh.service"
    tmr = DEPLOY_DIR / "chad-options-greeks-refresh.timer"
    assert svc.is_file(), f"missing {svc}"
    assert tmr.is_file(), f"missing {tmr}"
    svc_text = svc.read_text(encoding="utf-8")
    tmr_text = tmr.read_text(encoding="utf-8")
    assert "ExecStart=" in svc_text
    assert "chad.market_data.options_greeks_publisher" in svc_text
    assert "OnUnitActiveSec=86400" in tmr_text


# ---------------------------------------------------------------------------
# 14. CLI --dry-run produces a valid schema document.
# ---------------------------------------------------------------------------


def test_publisher_cli_dry_run_valid_schema(tmp_path: Path) -> None:
    _write_chain_cache(tmp_path, spot=500.0, days_out=30)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "chad.market_data.options_greeks_publisher",
            "--runtime-dir",
            str(tmp_path),
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    assert result.returncode == 0, (
        f"dry-run failed rc={result.returncode}\nstdout={result.stdout}\n"
        f"stderr={result.stderr}"
    )
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == GREEKS_SCHEMA_VERSION
    assert payload["status"] in ("ok", "partial")
    assert "SPY" in payload["symbols"]


# ---------------------------------------------------------------------------
# Sanity: file TTL constant within reasonable bound.
# ---------------------------------------------------------------------------


def test_gate_file_ttl_default_constant() -> None:
    assert GREEKS_FILE_TTL >= 3600
