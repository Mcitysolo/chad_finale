#!/usr/bin/env python3
"""
BOX-034A Inc 3 Step 0b — portfolio_snapshot_publisher FX/CAD tests.

Covers:
- _validate_usdcad / _get_live_usdcad_rate band logic (no fallback leaks):
  in-band -> live mid; unavailable -> None; out-of-band (inverted ~0.73 /
  garbage) -> None.
- main(): Kraken USD->CAD conversion uses the live rate in the correct
  direction (184.58 * ~1.37 -> ~252.85) and tags currency_ok=true.
- main(): FX unavailable -> fail-closed (prior kraken_equity preserved,
  currency_ok=false, KRAKEN_FX_UNAVAILABLE logged, ibkr_equity_usd_display None).

No real IBKR connection: all IO is patched. Safe under CHAD_SKIP_IB_CONNECT=1.
"""
import json
import logging

import chad.ops.portfolio_snapshot_publisher as pub


# --------------------------------------------------------------------------
# Rate validation / live-rate helper
# --------------------------------------------------------------------------
def test_validate_usdcad_in_band():
    assert pub._validate_usdcad(1.37) == 1.37
    assert pub._validate_usdcad(pub.USDCAD_BAND_LOW) == pub.USDCAD_BAND_LOW
    assert pub._validate_usdcad(pub.USDCAD_BAND_HIGH) == pub.USDCAD_BAND_HIGH


def test_validate_usdcad_rejects_out_of_band_and_garbage():
    # inverted USD-per-CAD (~0.73) must be rejected, NOT silently used
    assert pub._validate_usdcad(0.73) is None
    assert pub._validate_usdcad(1.19) is None
    assert pub._validate_usdcad(1.51) is None
    assert pub._validate_usdcad(0.0) is None
    assert pub._validate_usdcad(-1.37) is None
    assert pub._validate_usdcad(99999.0) is None
    assert pub._validate_usdcad(None) is None
    assert pub._validate_usdcad(float("nan")) is None
    assert pub._validate_usdcad("not-a-number") is None


def test_get_live_usdcad_rate_via_injected_fetcher():
    # in-band live mid passes through
    assert pub._get_live_usdcad_rate(fetcher=lambda: 1.37) == 1.37
    # quote unavailable -> None (no fallback constant)
    assert pub._get_live_usdcad_rate(fetcher=lambda: None) is None
    # inverted quote rejected by band
    assert pub._get_live_usdcad_rate(fetcher=lambda: 0.73) is None
    # garbage rejected by band
    assert pub._get_live_usdcad_rate(fetcher=lambda: 9999.0) is None


# --------------------------------------------------------------------------
# main() write behaviour
# --------------------------------------------------------------------------
def _patch_main(monkeypatch, tmp_path, *, usdcad, ibkr_usd, kraken_usd, prior=None):
    out = tmp_path / "portfolio_snapshot.json"
    if prior is not None:
        out.write_text(json.dumps(prior), encoding="utf-8")
    monkeypatch.setattr(pub, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(pub, "OUT_PATH", out)
    monkeypatch.setattr(pub, "_get_live_usdcad_rate", lambda *a, **k: usdcad)
    monkeypatch.setattr(pub, "_ibkr_equity_usd", lambda _u: ibkr_usd)
    monkeypatch.setattr(pub, "_read_kraken_usd_equity", lambda: kraken_usd)
    rc = pub.main()
    data = json.loads(out.read_text(encoding="utf-8"))
    return rc, data


def test_kraken_converted_to_cad_correct_direction(monkeypatch, tmp_path):
    # live rate from the real Kraken file: 252.8538 CAD / 184.583274 USD ~= 1.3698
    rc, data = _patch_main(
        monkeypatch, tmp_path,
        usdcad=1.3698, ibkr_usd=200000.0, kraken_usd=184.583274,
    )
    assert rc == 0
    assert data["kraken_equity_currency"] == "CAD"
    assert data["kraken_equity_currency_ok"] is True
    # 184.583274 * 1.3698 ~= 252.86 — correct direction (NOT the ~135 inversion)
    assert 252.0 < data["kraken_equity"] < 253.7
    assert data["kraken_equity"] > 200.0  # proves direction is USD->CAD, not CAD->USD
    assert data["ibkr_equity_usd_display"] == 200000.0


def test_kraken_fail_closed_when_fx_unavailable(monkeypatch, tmp_path, caplog):
    prior = {
        "ibkr_equity": 295687.28,
        "ibkr_equity_currency": "CAD",
        "ibkr_equity_currency_ok": True,
        "kraken_equity": 999.99,  # prior value to be preserved
    }
    with caplog.at_level(logging.ERROR, logger=pub.LOG.name):
        rc, data = _patch_main(
            monkeypatch, tmp_path,
            usdcad=None, ibkr_usd=None, kraken_usd=184.583274, prior=prior,
        )
    assert rc == 0
    # prior kraken_equity preserved (NOT overwritten, NOT mis-tagged as CAD)
    assert data["kraken_equity"] == 999.99
    assert data["kraken_equity_currency_ok"] is False
    # never fabricate a USD display value from a missing rate
    assert data["ibkr_equity_usd_display"] is None
    # canonical collector value preserved via read-through
    assert data["ibkr_equity"] == 295687.28
    # loud, greppable warning emitted
    assert any("KRAKEN_FX_UNAVAILABLE" in r.getMessage() for r in caplog.records)


def test_ibkr_usd_display_none_does_not_block_kraken_cad(monkeypatch, tmp_path):
    # IBKR equity unavailable but FX is live: kraken still converts, display None
    rc, data = _patch_main(
        monkeypatch, tmp_path,
        usdcad=1.3698, ibkr_usd=None, kraken_usd=184.583274,
    )
    assert rc == 0
    assert data["ibkr_equity_usd_display"] is None
    assert data["kraken_equity_currency_ok"] is True
    assert 252.0 < data["kraken_equity"] < 253.7
