"""P0A-A2 — OnFailure drop-ins wire failure-prone oneshots to the alert handler.

Validates the repo-tracked systemd drop-ins (design artifacts; installed by the
operator via ops/systemd/INSTALL_onfailure.md — never by the agent). Guards the
`%N`-not-`%n` correctness: the template chad-service-alert@.service re-appends
`.service` (`--failed-unit %i.service`), so `%n` would double the suffix.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SYSTEMD = REPO / "ops" / "systemd"

EXPECTED_UNITS = {
    "chad-brain-returns",
    "chad-clean-soak-evaluator",
    "chad-ibkr-collector",
    "chad-ibkr-daily-bars-refresh",
    "chad-intel-cache",
    "chad-kraken-collector",
    "chad-kraken-pnl-watcher",
    "chad-lifecycle-replay-engine",
    "chad-portfolio-merge",
}


def _dropin_files():
    return sorted(SYSTEMD.glob("*.service.d/10-onfailure.conf"))


def test_all_expected_units_have_a_dropin():
    covered = {p.parent.name.removesuffix(".service.d") for p in _dropin_files()}
    missing = EXPECTED_UNITS - covered
    assert not missing, f"missing OnFailure drop-ins for: {sorted(missing)}"


@pytest.mark.parametrize("conf", _dropin_files(), ids=lambda p: p.parent.name)
def test_dropin_wires_alert_handler_with_capital_N(conf):
    text = conf.read_text(encoding="utf-8")
    assert "[Unit]" in text
    assert "OnFailure=chad-service-alert@%N.service" in text, (
        f"{conf} must wire the alert handler with %N"
    )


@pytest.mark.parametrize("conf", _dropin_files(), ids=lambda p: p.parent.name)
def test_dropin_never_uses_lowercase_n(conf):
    """%n would keep the .service suffix and the template would double it."""
    text = conf.read_text(encoding="utf-8")
    # The literal specifier `%n.service` (lowercase) must never appear.
    assert "@%n.service" not in text, f"{conf} uses %n — would double .service"


def test_ibkr_collector_dropin_coexists_with_timeout_guards():
    """The added drop-in must not clobber the existing timeout-guard drop-in."""
    d = SYSTEMD / "chad-ibkr-collector.service.d"
    assert (d / "10-onfailure.conf").is_file()
    assert (d / "10-timeout-guards.conf").is_file()


def test_install_doc_present_and_complete():
    doc = SYSTEMD / "INSTALL_onfailure.md"
    assert doc.is_file(), "INSTALL_onfailure.md must exist"
    text = doc.read_text(encoding="utf-8")
    assert "systemctl daemon-reload" in text
    # census R44: latched handler must be cleared once.
    assert "reset-failed chad-service-alert@chad-ibkr-bar-provider.service" in text
    # must not silently omit the %N rationale
    assert "%N" in text and "%n" in text
