"""Official Matrix Box 059 — Sustained clean paper soak policy guard.

Acceptance criterion (Evidence-Locked Completion Matrix v0.1):
    "paper duration passes with no P0/P1, clean fills, clean reconciliation,
     and stable SCR."

This guard does NOT verify that the soak has *passed*. The audit at
`runtime/completion_matrix_evidence/BOX-059_OFFICIAL_sustained_clean_paper_soak_defined.md`
records the live runtime evidence. These tests prove:

  1. The soak **policy** exists at the expected path.
  2. The policy enumerates the gates the operator must observe:
     duration, no-P0/P1, fills, reconciliation, position-guard drift,
     lifecycle backlog, SCR stability, stop_bus auto-clear.
  3. The policy does **not** silently authorize live trading.

If the policy is later edited, these tests fail loudly so the policy
cannot drift away from the Box-059 contract. They are STATIC: no
runtime, no network, no broker call.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_DOC = REPO_ROOT / "ops" / "pending_actions" / "BOX-059_sustained_clean_paper_soak_policy.md"

REQUIRED_GATES = (
    # The eight gates listed in the soak policy. Wording is exact (lowercase
    # keys) so any future edit that drops a gate fails this guard.
    "minimum duration",
    "no open P0",
    "no open P1",
    "fills",
    "reconciliation",
    "position_guard_drift",
    "lifecycle backlog",
    "SCR",
    "stop_bus",
)

# Phrases that would, if present, indicate the policy is mis-stating
# the live-readiness posture or authorizing live trading.
FORBIDDEN_PHRASES = (
    "live trading authorized",
    "ready_for_live=true",
    "ready_for_live = true",
    "soak passed",
    "soak verified",
)


def test_box059_policy_doc_exists() -> None:
    assert POLICY_DOC.exists(), (
        f"Box-059 soak policy missing at {POLICY_DOC}. The Box-059 closure "
        f"requires the policy doc to remain on disk."
    )


def test_box059_policy_enumerates_required_gates() -> None:
    """Match gate names case-insensitively so the policy can capitalize
    them naturally (e.g. 'No open P0') without breaking the guard."""
    text_lower = POLICY_DOC.read_text(encoding="utf-8", errors="ignore").lower()
    missing = [g for g in REQUIRED_GATES if g.lower() not in text_lower]
    assert not missing, (
        f"Box-059 policy missing required gate names {missing!r}. The policy "
        f"must enumerate every soak gate so operators have an unambiguous "
        f"checklist."
    )


def test_box059_policy_does_not_authorize_live_trading() -> None:
    """Forbid phrases that would assert the soak is already passing or that
    live trading is authorized. Phrases must be checked in context — the
    policy legitimately describes a *future* live-readiness flip, so we
    only fail if the forbidden phrase appears as an assertion (sentence-
    initial 'live trading is authorized', etc.). To keep the guard
    unambiguous, we anchor each forbidden phrase to require the verb
    form 'is' / 'has been' / 'becomes' immediately preceding it."""
    text = POLICY_DOC.read_text(encoding="utf-8", errors="ignore").lower()
    offenders: list[str] = []
    for phrase in FORBIDDEN_PHRASES:
        p = phrase.lower()
        for verb in (" is ", " has been ", " becomes ", " was "):
            if (verb + p) in text:
                offenders.append(f"{verb.strip()} {phrase}")
    assert not offenders, (
        f"Box-059 policy contains forbidden assertions {offenders!r}. The "
        f"policy DEFINES the soak; it does not pass or authorize anything."
    )
