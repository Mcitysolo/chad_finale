"""End-to-end test for the Stage-2 trade-log flow (Phase 6, SSOT §1.3 / Part 6).

Proves the seam works whole: synthetic ndjson fixtures → the fail-closed adapter →
per-strategy net returns via the IDENTICAL cost path → the SAME Stage-1 scoring spine +
verdict engine → a signed, self-verifying report. The Stage-1 engine must CONSUME the
adapter output WITHOUT error (the Phase-6 acceptance bar); with < 30 trades and no
walk-forward windows the honest verdict is INSUFFICIENT_DATA, never a fabricated pass.

Everything here is a temp-dir fixture — no real ledger, network, broker, or runtime state.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chad.validation.cli import main, run_stage2_trade_log
from chad.validation.report_writer import verify_signature
from chad.validation.verdict import Verdict


def _fill(strategy="alpha_a", pnl=12.0, price=190.0, qty=10.0, day="03",
          seq=1, hashv="h", untrusted=False, validate_only=False, status=None,
          symbol="AAPL", asset_class="equity", broker="paper_exec"):
    extra = {}
    if untrusted:
        extra["pnl_untrusted"] = True
        extra["pnl_untrusted_reason"] = "scr_reset_clean_slate"
    if validate_only:
        extra["validate_only"] = True
        extra["pnl_untrusted"] = True
    payload = {
        "schema_version": "closed_trade.v1",
        "broker": broker, "symbol": symbol, "asset_class": asset_class,
        "side": "BUY", "strategy": strategy, "quantity": qty, "fill_price": price,
        "notional": qty * price, "pnl": pnl, "is_live": False, "regime": "risk_on",
        "entry_time_utc": f"2026-07-{day}T10:00:00+00:00",
        "exit_time_utc": f"2026-07-{day}T11:00:00+00:00", "tags": [], "extra": extra,
    }
    if status is not None:
        payload["status"] = status
    return {"payload": payload, "record_hash": hashv, "sequence_id": seq,
            "timestamp_utc": f"2026-07-{day}T11:00:00+00:00"}


def _write_ledger(tmp_path: Path, records, day="20260703") -> Path:
    src = tmp_path / "trades"
    src.mkdir(exist_ok=True)
    (src / f"trade_history_{day}.ndjson").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )
    return src


# --------------------------------------------------------------------------- #
# 1. Happy path — admitted real fills are scored; report is signed + verifies.
# --------------------------------------------------------------------------- #
def test_stage2_scores_admitted_fills_and_signs(tmp_path):
    records = [
        _fill(strategy="alpha_a", pnl=15.0, seq=1, hashv="a"),
        _fill(strategy="alpha_a", pnl=-8.0, seq=2, hashv="b"),
        _fill(strategy="alpha_b", pnl=3.0, seq=3, hashv="c", symbol="MSFT"),
        _fill(strategy="alpha_a", pnl=6.0, seq=4, hashv="d", untrusted=True),   # excluded
        _fill(strategy="alpha_b", pnl=0.0, seq=5, hashv="e", validate_only=True),  # excluded
    ]
    src = _write_ledger(tmp_path, records)

    signed = run_stage2_trade_log(
        repo_root=tmp_path, out_dir=tmp_path / "out", trades_dir=src,
        since="2026-07-01", until="2026-07-31",
        generated_at="2026-07-10T00:00:00Z", code_commit="deadbeef",
    )

    # Report is well-formed + tamper-evident.
    assert signed["stage"] == "stage2_trade_log"
    assert verify_signature(signed) is True

    # Two strategy heads scored (alpha_a: 2 trusted, alpha_b: 1 trusted); untrusted excluded.
    heads = {h["head"]: h for h in signed["heads"]}
    assert set(heads) == {"alpha_a", "alpha_b"}
    assert heads["alpha_a"]["metrics"]["n_oos_trades"] == 2
    assert heads["alpha_b"]["metrics"]["n_oos_trades"] == 1

    # Every head has a verdict; thin data → INSUFFICIENT_DATA (never a fabricated pass).
    for h in signed["heads"]:
        assert h["verdict"]["verdict"] == Verdict.INSUFFICIENT_DATA.value
        assert h["metrics"]["oos_source"] == "live_trade_log"
        assert h["metrics"]["n_walk_forward_windows"] == 0

    # Adapter manifest embedded; the trust gate excluded exactly the 2 untrusted rows.
    manifest = signed["data_quality"]["stage2_adapter_manifest"]
    assert manifest["admitted"] == 3
    assert manifest["excluded_by_reason"].get("pnl_untrusted") == 1
    assert manifest["excluded_by_reason"].get("validate_only") == 1

    # Portfolio verdict present.
    assert signed["verdict_summary"]["portfolio_verdict"] == Verdict.INSUFFICIENT_DATA.value


def test_stage2_net_return_reflects_cost_haircut(tmp_path):
    """A head's net PnL must be strictly below its gross PnL — the S4 haircut was charged."""
    records = [_fill(strategy="alpha_a", pnl=50.0, seq=i, hashv=f"h{i}") for i in range(3)]
    src = _write_ledger(tmp_path, records)
    signed = run_stage2_trade_log(
        repo_root=tmp_path, out_dir=tmp_path / "out", trades_dir=src,
        generated_at="2026-07-10T00:00:00Z", code_commit="x",
    )
    summ = signed["heads"][0]["trade_summary"]
    assert summ["n_trades"] == 3
    assert summ["gross_pnl"] == pytest.approx(150.0)
    assert summ["total_cost"] > 0.0
    assert summ["net_pnl"] < summ["gross_pnl"]


# --------------------------------------------------------------------------- #
# 2. Untrusted-only input — zero admitted, still a valid signed report.
# --------------------------------------------------------------------------- #
def test_stage2_all_untrusted_yields_zero_heads_signed(tmp_path):
    records = [_fill(validate_only=True, seq=i, hashv=f"h{i}") for i in range(5)]
    src = _write_ledger(tmp_path, records)
    signed = run_stage2_trade_log(
        repo_root=tmp_path, out_dir=tmp_path / "out", trades_dir=src,
        generated_at="2026-07-10T00:00:00Z", code_commit="x",
    )
    assert signed["heads"] == []
    assert verify_signature(signed) is True
    assert signed["verdict_summary"]["portfolio_verdict"] == Verdict.INSUFFICIENT_DATA.value
    assert signed["data_quality"]["stage2_adapter_manifest"]["admitted"] == 0


def test_stage2_empty_ledger_dir(tmp_path):
    signed = run_stage2_trade_log(
        repo_root=tmp_path, out_dir=tmp_path / "out", trades_dir=tmp_path / "empty",
        generated_at="2026-07-10T00:00:00Z", code_commit="x",
    )
    assert verify_signature(signed) is True
    assert signed["heads"] == []


# --------------------------------------------------------------------------- #
# 3. No OOS lockbox / no runtime writes for Stage 2.
# --------------------------------------------------------------------------- #
def test_stage2_never_seals_oos_or_touches_runtime(tmp_path):
    records = [_fill(seq=i, hashv=f"h{i}") for i in range(3)]
    src = _write_ledger(tmp_path, records)
    out = tmp_path / "out"
    signed = run_stage2_trade_log(
        repo_root=tmp_path, out_dir=out, trades_dir=src,
        generated_at="2026-07-10T00:00:00Z", code_commit="x",
    )
    assert signed["oos"]["source"] == "live_trade_log"
    assert signed["oos"]["sealed"] is False
    assert signed["oos"]["contaminated"] is False
    # No lockbox / freeze artifacts created (Phase-0..5 machinery untouched).
    assert not (out / "lockbox").exists()
    assert not (out / "freeze").exists()
    # Adapter artifacts ARE under out/stage2.
    assert (out / "stage2").is_dir()
    assert any((out / "stage2").glob("stage2_manifest_*.json"))


# --------------------------------------------------------------------------- #
# 4. CLI end-to-end (main) — returns 0, writes the report artifact.
# --------------------------------------------------------------------------- #
def test_cli_stage2_end_to_end(tmp_path, capsys):
    records = [_fill(seq=i, hashv=f"h{i}") for i in range(4)]
    src = _write_ledger(tmp_path, records)
    out = tmp_path / "out"
    rc = main([
        "--stage", "stage2", "--since", "2026-07-01", "--until", "2026-07-31",
        "--trades-dir", str(src), "--out-dir", str(out), "--repo-root", str(tmp_path),
        "--now", "2026-07-10T00:00:00Z", "--code-commit", "abc123",
    ])
    assert rc == 0
    stdout = capsys.readouterr().out
    assert "stage2_trade_log" in stdout
    assert "PORTFOLIO verdict: INSUFFICIENT_DATA" in stdout
    # Report artifact written at the top level; adapter artifacts under out/stage2.
    assert any(out.glob("edge_report_*.json"))
    assert any((out / "stage2").glob("stage2_trades_*.ndjson"))


def test_cli_stage2_rejects_bad_since(tmp_path):
    assert main(["--stage", "stage2", "--since", "bad", "--repo-root", str(tmp_path),
                 "--trades-dir", str(tmp_path)]) == 2


def test_stage2_provenance_makes_no_false_seal_or_replay_claim(tmp_path):
    """The signed artifact must NOT carry the Stage-1 hash-seal / replay-reconstruction
    provenance (a live trade log has neither) — else it would contradict its own oos/parity
    sections. Verify the Stage-2 overrides are honest."""
    records = [_fill(seq=i, hashv=f"h{i}") for i in range(3)]
    src = _write_ledger(tmp_path, records)
    signed = run_stage2_trade_log(
        repo_root=tmp_path, out_dir=tmp_path / "out", trades_dir=src,
        generated_at="2026-07-10T00:00:00Z", code_commit="x",
    )
    prov = signed["provenance"]
    assert "hash-seal" not in prov["oos_discipline"].lower()
    assert "CONTAMINATED" not in prov["oos_discipline"]
    assert prov["oos_discipline"].startswith("N/A for Stage 2")
    assert "replayed" not in prov["replay_reconstruction"].lower() or \
        prov["replay_reconstruction"].startswith("N/A for Stage 2")
    assert prov["replay_reconstruction"].startswith("N/A for Stage 2")
    # And the universe-bias note (a genuine, stage-agnostic caveat) is retained.
    assert "universe_bias" in prov


def test_cli_historical_still_works(tmp_path):
    """Regression: adding stage2 must not break the default historical stage dispatch."""
    rc = main(["--stage", "historical", "--repo-root", str(tmp_path),
               "--out-dir", str(tmp_path / "out"), "--bars-dir", str(tmp_path / "nobars"),
               "--now", "2026-07-10T00:00:00Z", "--code-commit", "x"])
    # No bars dir → historical run still completes (heads NOT_REPLAYABLE / no symbols); rc 0.
    assert rc == 0
