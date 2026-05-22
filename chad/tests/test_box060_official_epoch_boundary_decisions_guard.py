"""Official Matrix Box 060 — Epoch-boundary operator decisions policy guard.

Acceptance criterion (Evidence-Locked Completion Matrix v0.1):
    "deferred services like full-cycle refresh / crypto risk-off are enabled
     or formally deferred."

This guard does NOT enable or disable any service. The audit at
`runtime/completion_matrix_evidence/BOX-060_OFFICIAL_epoch_boundary_operator_decisions_resolved.md`
inventories the systemd units that are currently disabled / inactive,
and `ops/pending_actions/BOX-060_epoch_boundary_operator_decisions.md`
records the deferral decision for each. These tests prove:

  1. The policy doc exists at the canonical path.
  2. The policy enumerates every deferred unit identified by the
     audit (so a unit cannot silently disappear from the decision
     register).
  3. The policy does not assert that any service has been "enabled"
     or that live trading is now authorized — Box-060 records
     decisions; it does not perform them.

If a deferred service is later enabled with operator approval, the
test list `DEFERRED_UNITS` must be updated in the SAME commit that
lands the enable action and the operator-approval evidence.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_DOC = REPO_ROOT / "ops" / "pending_actions" / "BOX-060_epoch_boundary_operator_decisions.md"

# Eight chad timers that are currently disabled+inactive (per Box-060
# audit). The policy doc must mention each by name. The list is exact;
# unit names should be checked verbatim.
DEFERRED_UNITS = (
    "chad-full-cycle-refresh.timer",
    "chad-crypto-risk-off.timer",
    "chad-crypto-risk-notify.timer",
    "chad-ibkr-cash-collector.timer",
    "chad-price-cache-refresh.timer",
    "chad-stop-refresh.timer",
    "chad-symbol-bench.timer",
    "chad-warmup.timer",
)

# Two units explicitly called out by the Box-060 acceptance criterion.
EXPLICIT_BOX060_DECISIONS = (
    "full-cycle refresh",
    "crypto risk-off",
)

# Forbidden assertion phrases. The policy describes decisions; it must
# not assert services have been enabled or that live trading is
# authorized. We verb-anchor to avoid false positives on descriptive
# usage (e.g. "this timer would be enabled if X").
FORBIDDEN_PHRASES = (
    "live trading authorized",
    "live trading is authorized",
    "ready_for_live=true",
    "ready_for_live = true",
    "ready_for_live is true",
    "service has been enabled",
    "timer has been enabled",
)


def test_box060_policy_doc_exists() -> None:
    assert POLICY_DOC.exists(), (
        f"Box-060 policy missing at {POLICY_DOC}. The Box-060 closure "
        f"requires the policy doc to remain on disk."
    )


def test_box060_policy_enumerates_every_deferred_unit() -> None:
    text = POLICY_DOC.read_text(encoding="utf-8", errors="ignore")
    missing = [u for u in DEFERRED_UNITS if u not in text]
    assert not missing, (
        f"Box-060 policy missing deferred-unit names {missing!r}. Every "
        f"unit identified by the Box-060 audit must appear by exact name "
        f"in the decision register so it cannot silently disappear."
    )


def test_box060_policy_records_explicit_box060_decisions() -> None:
    text_lower = POLICY_DOC.read_text(encoding="utf-8", errors="ignore").lower()
    missing = [p for p in EXPLICIT_BOX060_DECISIONS if p.lower() not in text_lower]
    assert not missing, (
        f"Box-060 policy missing required headline decisions {missing!r}. "
        f"The acceptance criterion names full-cycle refresh and crypto "
        f"risk-off explicitly; both must appear in the policy."
    )


def test_box060_policy_does_not_authorize_or_claim_enabled() -> None:
    """Verb-anchored check (mirrors Box-059) so the policy can legitimately
    describe what a *future* enable / authorization will look like without
    triggering this guard. Only fails when the forbidden phrase appears
    after an assertion verb."""
    text = POLICY_DOC.read_text(encoding="utf-8", errors="ignore").lower()
    offenders: list[str] = []
    for phrase in FORBIDDEN_PHRASES:
        p = phrase.lower()
        for verb in (" is ", " has been ", " becomes ", " was ", " set to "):
            if (verb + p) in text:
                offenders.append(f"{verb.strip()} {phrase}")
    assert not offenders, (
        f"Box-060 policy contains forbidden assertions {offenders!r}. The "
        f"policy DEFERS services; it does not enable them and does not "
        f"authorize live trading."
    )
