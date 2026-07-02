#!/usr/bin/env python3
"""
BOX-034A Inc 3 Step 0b — portfolio_snapshot_publisher FX/CAD tests.

Covers:
- _validate_usdcad / _get_live_usdcad_rate band logic (no fallback leaks):
  in-band -> live mid; unavailable -> None; out-of-band (inverted ~0.73 /
  garbage) -> None.
- main(): Kraken leg is NATIVE CAD (C-ii) — no USD round-trip, no usdcad
  multiply; currency_ok=true whenever a native CAD value is readable.
- main(): Kraken CAD unreadable -> fail-closed (prior kraken_equity preserved,
  currency_ok=false, KRAKEN_CAD_UNAVAILABLE logged). FX unavailable now only
  affects ibkr_equity_usd_display (None), NOT the native Kraken CAD leg.

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
def _patch_main(monkeypatch, tmp_path, *, usdcad, ibkr_usd, kraken_cad, prior=None):
    out = tmp_path / "portfolio_snapshot.json"
    if prior is not None:
        out.write_text(json.dumps(prior), encoding="utf-8")
    monkeypatch.setattr(pub, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(pub, "OUT_PATH", out)
    monkeypatch.setattr(pub, "_get_live_usdcad_rate", lambda *a, **k: usdcad)
    monkeypatch.setattr(pub, "_ibkr_equity_usd", lambda _u: ibkr_usd)
    # C-ii: the Kraken leg is now NATIVE CAD (no USD round-trip). Inject the
    # native CAD value directly; kraken_cad=None triggers the fail-closed branch.
    monkeypatch.setattr(pub, "_read_kraken_cad_equity", lambda *a, **k: kraken_cad)
    rc = pub.main()
    data = json.loads(out.read_text(encoding="utf-8"))
    return rc, data


def test_kraken_leg_is_native_cad_passthrough(monkeypatch, tmp_path):
    # C-ii: the Kraken account is natively CAD. The leg passes the native CAD
    # value through verbatim — no usdcad multiply, no USD round-trip.
    rc, data = _patch_main(
        monkeypatch, tmp_path,
        usdcad=1.3698, ibkr_usd=200000.0, kraken_cad=252.85,
    )
    assert rc == 0
    assert data["kraken_equity_currency"] == "CAD"
    assert data["kraken_equity_currency_ok"] is True
    assert data["kraken_equity"] == 252.85  # native CAD verbatim, no conversion
    assert data["ibkr_equity_usd_display"] == 200000.0


def test_kraken_fail_closed_when_cad_unreadable(monkeypatch, tmp_path, caplog):
    # C-ii: the Kraken leg fails closed ONLY when the native CAD value is
    # unreadable (no cad_equivalent, no reconstructable balance) — NOT because
    # the USDCAD feed is dark. Prior kraken_equity is preserved, never mis-tagged.
    prior = {
        "ibkr_equity": 295687.28,
        "ibkr_equity_currency": "CAD",
        "ibkr_equity_currency_ok": True,
        "kraken_equity": 999.99,  # prior value to be preserved
    }
    with caplog.at_level(logging.ERROR, logger=pub.LOG.name):
        rc, data = _patch_main(
            monkeypatch, tmp_path,
            usdcad=None, ibkr_usd=None, kraken_cad=None, prior=prior,
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
    assert any("KRAKEN_CAD_UNAVAILABLE" in r.getMessage() for r in caplog.records)


def test_ibkr_usd_display_none_does_not_block_kraken_cad(monkeypatch, tmp_path):
    # IBKR equity unavailable: the native-CAD Kraken leg is unaffected; only the
    # ibkr_equity_usd_display goes None.
    rc, data = _patch_main(
        monkeypatch, tmp_path,
        usdcad=1.3698, ibkr_usd=None, kraken_cad=252.85,
    )
    assert rc == 0
    assert data["ibkr_equity_usd_display"] is None
    assert data["kraken_equity_currency_ok"] is True
    assert data["kraken_equity"] == 252.85


# --------------------------------------------------------------------------
# Authoritative USD equity (additive; fail-closed, never CAD fallback)
# --------------------------------------------------------------------------
def test_total_usd_authoritative_valid_rate(monkeypatch, tmp_path):
    # Prior carries a CAD ibkr_equity (preserved via read-through) so we can
    # prove the authoritative USD total is NOT the CAD sum the tier_manager /
    # equity_history publishers currently compute.
    prior = {
        "ibkr_equity": 190000.0,  # CAD (canonical collector value, preserved)
        "ibkr_equity_currency": "CAD",
        "ibkr_equity_currency_ok": True,
        "kraken_equity": 12345.0,  # overwritten this cycle
    }
    rc, data = _patch_main(
        monkeypatch, tmp_path,
        usdcad=1.4, ibkr_usd=136000.0, kraken_cad=280.0, prior=prior,
    )
    assert rc == 0
    assert data["usd_ok"] is True
    assert data["usdcad_rate_used"] == 1.4
    # kraken native CAD this cycle = 280.0; converted back = 280.0 / 1.4 = 200.0 USD.
    # coinbase is 0.0 -> total == ibkr_usd + kraken_usd exactly (coinbase handled).
    assert data["coinbase_equity"] == 0.0
    assert abs(data["total_equity_usd_authoritative"] - (136000.0 + 200.0)) < 1e-6
    assert abs(data["total_equity_usd_authoritative"] - 136200.0) < 1e-6
    # MUST NOT equal the CAD sum (ibkr_equity + kraken_equity + coinbase, all CAD)
    cad_sum = data["ibkr_equity"] + data["kraken_equity"] + data["coinbase_equity"]
    assert abs(cad_sum - 190280.0) < 1e-6  # sanity on the CAD sum itself
    assert data["total_equity_usd_authoritative"] != cad_sum
    assert data["total_equity_usd_authoritative"] < cad_sum  # USD < CAD at >1.0 rate


def test_total_usd_authoritative_fail_closed_when_rate_none(monkeypatch, tmp_path):
    # FX unavailable (e.g. weekend): authoritative USD total fails closed to
    # None — it must NEVER fall back to the CAD figure.
    prior = {
        "ibkr_equity": 190000.0,  # CAD
        "ibkr_equity_currency": "CAD",
        "ibkr_equity_currency_ok": True,
        "kraken_equity": 280.0,
    }
    rc, data = _patch_main(
        monkeypatch, tmp_path,
        usdcad=None, ibkr_usd=None, kraken_cad=280.0, prior=prior,
    )
    assert rc == 0
    assert data["total_equity_usd_authoritative"] is None
    assert data["usd_ok"] is False
    assert data["usdcad_rate_used"] is None
    # explicitly NOT the preserved CAD figure
    assert data["total_equity_usd_authoritative"] != data["ibkr_equity"]


def test_total_usd_authoritative_fail_closed_when_ibkr_component_none(monkeypatch, tmp_path):
    # Rate is live (so kraken converts) but the IBKR USD component is missing:
    # "every component converts" is violated -> total None, usd_ok False, yet
    # usdcad_rate_used still records the rate that WAS applied to kraken.
    rc, data = _patch_main(
        monkeypatch, tmp_path,
        usdcad=1.4, ibkr_usd=None, kraken_cad=280.0,
    )
    assert rc == 0
    assert data["total_equity_usd_authoritative"] is None
    assert data["usd_ok"] is False
    assert data["usdcad_rate_used"] == 1.4
    assert data["kraken_equity_currency_ok"] is True  # kraken leg still native CAD
