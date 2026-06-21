"""
Currency audit — Bucket A: relabel report-only ``*_usd`` fields that hold CAD
values to ``*_cad`` and make the dashboard display CAD honestly.

This is a pure label/display correctness change over values that are ALREADY
CAD. It asserts:

  * the relabeled fields carry IDENTICAL CAD values (no value moved);
  * the drawdown / VaR percentage math is identical pre/post the relabel;
  * the OLD ``*_usd`` equity keys are GONE from the two modules' dataclasses
    and emitted state dicts (the VaR/exposure ``*_usd`` fields are a separate
    Bucket-B question and are deliberately left untouched);
  * the dashboard surfaces the CAD currency tag faithfully — and, because
    ``account_equity_currency_ok`` is currently false, labels it "CAD*" /
    "CAD (unverified)" rather than asserting a clean "CAD";
  * Total Paper PnL and today's realized PnL labels are NOT relabeled.

ZERO behavioral change: ``enforcement_active`` stays False throughout; no gate,
sizing, or decision path is touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chad.risk import drawdown_guard, portfolio_var  # noqa: E402


def _write_ndjson(path: Path, rows) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_json(path: Path, obj) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# A. drawdown_guard — current_equity_usd -> current_equity_cad, hwm_usd -> hwm_cad
# ---------------------------------------------------------------------------


def test_drawdown_relabel_carries_identical_cad_values_and_pct(tmp_path: Path) -> None:
    rt = tmp_path / "runtime"
    rt.mkdir()
    _write_ndjson(
        rt / "equity_history.ndjson",
        [
            {"date_utc": "2026-04-01", "total_equity_cad": 200000.0},  # HWM
            {"date_utc": "2026-04-02", "total_equity_cad": 170000.0},
        ],
    )
    _write_json(rt / "portfolio_snapshot.json", {"ibkr_equity": 170000.0})

    report = drawdown_guard.compute_drawdown(
        equity_history_path=rt / "equity_history.ndjson",
        portfolio_snapshot_path=rt / "portfolio_snapshot.json",
    )

    # Values land on the CAD-named fields, unchanged.
    assert report.hwm_cad == pytest.approx(200000.0)
    assert report.current_equity_cad == pytest.approx(170000.0)
    # Percentage math is identical to the pre-relabel formula.
    expected_pct = (170000.0 - 200000.0) / 200000.0 * 100.0
    assert report.drawdown_pct == pytest.approx(expected_pct)
    assert report.enforcement_active is False

    # The dataclass no longer carries the *_usd equity attribute names.
    assert not hasattr(report, "current_equity_usd")
    assert not hasattr(report, "hwm_usd")
    assert hasattr(report, "current_equity_cad")
    assert hasattr(report, "hwm_cad")


def test_drawdown_state_dict_has_cad_keys_and_no_usd_equity_keys(tmp_path: Path) -> None:
    rt = tmp_path / "runtime"
    rt.mkdir()
    _write_ndjson(
        rt / "equity_history.ndjson",
        [
            {"date_utc": "2026-04-01", "total_equity_cad": 100000.0},
            {"date_utc": "2026-04-02", "total_equity_cad": 95000.0},
        ],
    )
    _write_json(rt / "portfolio_snapshot.json", {"ibkr_equity": 95000.0})

    report = drawdown_guard.compute_drawdown(
        equity_history_path=rt / "equity_history.ndjson",
        portfolio_snapshot_path=rt / "portfolio_snapshot.json",
    )
    state = drawdown_guard.report_to_state_dict(report, ts_utc="2026-06-21T00:00:00Z")

    assert state["current_equity_cad"] == 95000.0
    assert state["hwm_cad"] == 100000.0
    # The renamed equity keys must be gone.
    assert "current_equity_usd" not in state
    assert "hwm_usd" not in state


# ---------------------------------------------------------------------------
# B. portfolio_var — portfolio_equity_usd -> portfolio_equity_cad
# ---------------------------------------------------------------------------


def test_var_relabel_carries_identical_cad_equity(tmp_path: Path) -> None:
    rt = tmp_path / "runtime"
    rt.mkdir()
    _write_json(rt / "positions_truth.json", {"positions": []})
    _write_json(rt / "portfolio_snapshot.json", {"ibkr_equity": 123456.0})

    report = portfolio_var.compute_portfolio_var(
        positions_truth_path=rt / "positions_truth.json",
        portfolio_snapshot_path=rt / "portfolio_snapshot.json",
        bars_dir=tmp_path / "bars",
    )

    assert report.portfolio_equity_cad == pytest.approx(123456.0)
    assert not hasattr(report, "portfolio_equity_usd")
    assert hasattr(report, "portfolio_equity_cad")

    state = portfolio_var.report_to_state_dict(report, ts_utc="2026-06-21T00:00:00Z")
    assert state["portfolio_equity_cad"] == 123456.0
    assert "portfolio_equity_usd" not in state
    # Bucket-B fields (USD-denominated exposures/VaR) are intentionally retained.
    assert "var_95_1day_usd" in state
    assert "var_99_1day_usd" in state


# ---------------------------------------------------------------------------
# C. Source-level guard — no *_usd equity REPORT field name remains in the two
#    modules. (The VaR/exposure *_usd fields are out of scope by design.)
# ---------------------------------------------------------------------------


def test_no_usd_equity_report_field_remains_in_modules() -> None:
    dd_src = Path(drawdown_guard.__file__).read_text(encoding="utf-8")
    var_src = Path(portfolio_var.__file__).read_text(encoding="utf-8")

    # The renamed equity field names must not appear as live identifiers/keys.
    # (They survive only inside the explanatory comments noting the rename.)
    def _noncomment_lines(src: str):
        return [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]

    dd_code = "\n".join(_noncomment_lines(dd_src))
    var_code = "\n".join(_noncomment_lines(var_src))

    assert "current_equity_usd" not in dd_code
    assert "hwm_usd" not in dd_code
    assert "portfolio_equity_usd" not in var_code


# ---------------------------------------------------------------------------
# D. Dashboard — CAD honesty on the account value; PnL labels untouched
# ---------------------------------------------------------------------------


@pytest.fixture()
def api(monkeypatch):
    import importlib

    monkeypatch.setenv("CHAD_DASHBOARD_PASSWORD", "test_value_12345")
    sys.modules.pop("chad.dashboard.api", None)
    mod = importlib.import_module("chad.dashboard.api")
    return mod


def _portfolio_with(api, tmp_path, monkeypatch, pnl: dict) -> dict:
    monkeypatch.setattr(api, "RUNTIME", tmp_path)
    _write_json(tmp_path / "pnl_state.json", pnl)
    _write_json(
        tmp_path / "scr_state.json",
        {"sizing_factor": 1.0, "stats": {"total_pnl": 27494.0, "effective_trades": 216, "win_rate": 0.731}},
    )
    return api.StateBuilder()._portfolio()


def test_dashboard_account_value_labels_cad_unverified(api, tmp_path, monkeypatch) -> None:
    # Mirrors live pnl_state.json: CAD equity, currency tag present but unverified.
    p = _portfolio_with(
        api,
        tmp_path,
        monkeypatch,
        {
            "account_equity": 161606.79,
            "account_equity_currency": "CAD",
            "account_equity_currency_ok": False,
            "realized_pnl": 12.34,
        },
    )

    # Faithful: CAD is surfaced, asterisked because *_ok is false.
    assert p["account_value_label"].endswith(" CAD*")
    assert p["account_value_label"].startswith("$161,607")
    assert p["account_value_currency"] == "CAD"
    assert p["account_value_currency_ok"] is False
    assert p["account_value_currency_label"] == "CAD*"
    assert p["account_value_currency_note"] == "CAD (unverified)"

    # Total Paper PnL and today's realized PnL are paper-ledger units — NOT relabeled.
    assert "CAD" not in p["total_paper_pnl_label"]
    assert "CAD" not in p["today_realized_pnl_label"]


def test_dashboard_account_value_verified_currency_no_asterisk(api, tmp_path, monkeypatch) -> None:
    p = _portfolio_with(
        api,
        tmp_path,
        monkeypatch,
        {
            "account_equity": 100000.0,
            "account_equity_currency": "CAD",
            "account_equity_currency_ok": True,
            "realized_pnl": None,
        },
    )
    assert p["account_value_label"] == "$100,000 CAD"
    assert p["account_value_currency_ok"] is True
    assert p["account_value_currency_label"] == "CAD"
    assert p["account_value_currency_note"] == "CAD"


def test_dashboard_account_value_no_currency_tag_is_unchanged(api, tmp_path, monkeypatch) -> None:
    # No currency tag in pnl_state -> byte-identical to the pre-change label.
    p = _portfolio_with(
        api,
        tmp_path,
        monkeypatch,
        {"account_equity": 100000.0, "realized_pnl": 12.34},
    )
    assert p["account_value_label"] == "$100,000"
    assert p["account_value_currency"] is None
    assert p["account_value_currency_ok"] is None
    assert p["account_value_currency_label"] is None
    assert p["account_value_currency_note"] is None


def test_fmt_money_currency_is_opt_in(api) -> None:
    # Default (no currency) is byte-identical to legacy behavior.
    assert api._fmt_money(161606.79) == "$161,607"
    assert api._fmt_money(-50.0) == "-$50"
    assert api._fmt_money_signed(9880.5) == "+$9,880"
    assert api._fmt_money_signed(-12.0) == "-$12"
    assert api._fmt_money_signed(0) == "$0"
    # Opt-in suffix.
    assert api._fmt_money(161606.79, currency="CAD*") == "$161,607 CAD*"
    assert api._fmt_money_signed(9880.5, currency="CAD") == "+$9,880 CAD"
