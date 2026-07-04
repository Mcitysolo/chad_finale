"""Tests for chad/validation/report_writer.py — Phase 5 signed report (SSOT Part 4 / §3.8).

Fixture-only. They pin: the artifact embeds every required section; it is deterministic
(same inputs → byte-identical json + md); the content signature verifies and detects
tampering; and writing produces exactly two files under the caller's out_dir (never a
runtime/ path).
"""

from __future__ import annotations

import json

import pytest

from chad.validation.report_writer import (
    SCHEMA_VERSION,
    UNIVERSE_PROVENANCE_NOTE,
    build_report,
    render_markdown,
    report_basename,
    sign_report,
    verify_signature,
    write_report,
)

_GEN = "2026-07-04T00:00:00Z"


def _sections() -> dict:
    return dict(
        generated_at=_GEN,
        stage="historical",
        final_run=False,
        code_commit="deadbeef",
        data_quality={
            "worst_status": "WARN",
            "symbols": [{"symbol": "M6E", "status": "WARN", "bar_count": 75,
                         "first_date": "2026-03-17", "last_date": "2026-07-02",
                         "quote_currency": "USD"}],
            "involved_symbols": ["M6E"],
        },
        parity_map=[{"head": "alpha_forex", "status": "REPLAYABLE"}],
        parity_table="HEAD  STATUS\nalpha_forex  REPLAYABLE",
        heads=[{
            "head": "alpha_forex",
            "symbol": "M6E",
            "metrics": {"parity_status": "REPLAYABLE", "n_oos_trades": 2,
                        "n_walk_forward_windows": 3, "n_regimes_in_oos": 0,
                        "deflated_sharpe_worst": None, "cost_adj_cagr": -0.04,
                        "worst_quantile_ruin": 0.0},
            "verdict": {"verdict": "INSUFFICIENT_DATA", "label": "INSUFFICIENT_DATA",
                        "reasons": ["below minimums"]},
            "backtest_summary": {"backtested": True},
        }],
        portfolio={
            "verdict": {"verdict": "INSUFFICIENT_DATA", "label": "INSUFFICIENT_DATA", "reasons": []},
            "capital_fraction_in_surviving_heads": 0.0,
            "surviving_heads": 0, "total_heads": 1,
        },
        frozen_config={
            "frozen": {"config_hash": "cfg123", "frozen_at": _GEN, "thresholds": {}, "cost_config": {}},
            "trial_count": 0, "last_verdict": None, "superseded_hashes": [],
        },
        oos={"access_count": 0, "source": "decoy", "sealed": True,
             "seal": {"oos_hash": "abc", "n_oos": 2}, "log_integrity_ok": True, "contaminated": False},
        thresholds={"n_min": 30, "w_min": 6, "r_min": 3},
        verdict_summary={"counts": {"INSUFFICIENT_DATA": 1}, "portfolio_verdict": "INSUFFICIENT_DATA"},
    )


def _built() -> dict:
    return build_report(**_sections())


# --------------------------------------------------------------------------- #
# Structure / embedding.
# --------------------------------------------------------------------------- #
def test_report_embeds_all_sections() -> None:
    r = _built()
    assert r["schema_version"] == SCHEMA_VERSION
    assert r["generated_at"] == _GEN
    assert r["final_run"] is False
    for key in (
        "provenance", "thresholds", "config_frozen", "oos", "data_quality",
        "parity_map", "parity_table", "heads", "portfolio", "verdict_summary",
    ):
        assert key in r, f"missing report section {key!r}"
    assert r["provenance"]["universe_bias"] == UNIVERSE_PROVENANCE_NOTE  # SSOT §V3 flag
    assert r["oos"]["access_count"] == 0
    assert r["config_frozen"]["frozen"]["config_hash"] == "cfg123"


def test_build_rejects_bad_args() -> None:
    s = _sections()
    s["final_run"] = "no"  # not a bool
    with pytest.raises(ValueError):
        build_report(**s)


# --------------------------------------------------------------------------- #
# Signature.
# --------------------------------------------------------------------------- #
def test_sign_and_verify() -> None:
    signed = sign_report(_built())
    assert "signature" in signed
    assert signed["signature"]["algo"] == "sha256"
    assert verify_signature(signed) is True


def test_signature_detects_tampering() -> None:
    signed = sign_report(_built())
    assert verify_signature(signed) is True
    signed["heads"][0]["verdict"]["verdict"] = "PASS"  # forge a pass
    assert verify_signature(signed) is False


def test_unsigned_report_does_not_verify() -> None:
    assert verify_signature(_built()) is False


def test_sign_is_idempotent() -> None:
    once = sign_report(_built())
    twice = sign_report(once)  # re-signing strips the old signature first
    assert once["signature"]["content_sha256"] == twice["signature"]["content_sha256"]


# --------------------------------------------------------------------------- #
# Determinism.
# --------------------------------------------------------------------------- #
def test_signed_report_is_deterministic() -> None:
    a = sign_report(_built())
    b = sign_report(build_report(**_sections()))
    assert a == b
    assert a["signature"]["content_sha256"] == b["signature"]["content_sha256"]


def test_written_files_are_byte_identical_across_runs(tmp_path) -> None:
    signed = sign_report(_built())
    j1, m1 = write_report(signed, tmp_path / "a")
    j2, m2 = write_report(signed, tmp_path / "b")
    assert j1.read_bytes() == j2.read_bytes()
    assert m1.read_bytes() == m2.read_bytes()


# --------------------------------------------------------------------------- #
# Write behaviour.
# --------------------------------------------------------------------------- #
def test_write_report_creates_json_and_md_and_verifies(tmp_path) -> None:
    signed = sign_report(_built())
    json_path, md_path = write_report(signed, tmp_path / "out")
    assert json_path.name == report_basename(_GEN) + ".json"
    assert md_path.name == report_basename(_GEN) + ".md"
    # Round-trip: the persisted json still verifies.
    reloaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert verify_signature(reloaded) is True
    # Exactly two artifact files were written, both under out_dir.
    files = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert files == sorted([json_path.name, md_path.name])


def test_report_basename_sanitizes() -> None:
    b = report_basename("2026-07-04T00:00:00Z")
    assert ":" not in b
    assert b.startswith("edge_report_")
    with pytest.raises(ValueError):
        report_basename("")


# --------------------------------------------------------------------------- #
# Markdown.
# --------------------------------------------------------------------------- #
def test_markdown_is_deterministic_and_has_sections() -> None:
    signed = sign_report(_built())
    md1 = render_markdown(signed)
    md2 = render_markdown(signed)
    assert md1 == md2
    for heading in (
        "# CHAD Edge-Validation Report",
        "## Verdict summary",
        "## OOS lockbox",
        "## Per-head verdicts",
        "## Data quality",
        "## Feature-parity map",
        "## Frozen config",
        "## Signature",
    ):
        assert heading in md1, f"markdown missing section {heading!r}"
    # The dry-run caveat is surfaced for a non-final run.
    assert "dry run" in md1


def test_markdown_flags_contamination() -> None:
    s = _sections()
    s["oos"] = dict(s["oos"])
    s["oos"]["access_count"] = 2
    s["oos"]["contaminated"] = True
    md = render_markdown(sign_report(build_report(**s)))
    assert "CONTAMINATED" in md
