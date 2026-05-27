# OPTIONS-CHAIN OnFailure directive — chad-options-chain-refresh.service

# HIGH_ID: OPTIONS-CHAIN-1 (companion / systemd portion)
# Status: PROPOSED
# Severity: HIGH (companion to OPTIONS-CHAIN-1 main PA at ops/pending_actions/R3_silent_unit_state_and_market_data_failures_2026-05-27.md)
# Effort: S

# Source audit
- `reports/forensic_audits/CHAD_FORENSIC_FULL_SYSTEM_AUDIT_20260527T142951Z.md`
  - §1 Executive Summary item #3 (silent and partial publisher failures)
  - §14.3 (chad-options-chain-refresh root cause)
  - §19 closeout item 3 (OPTIONS-CHAIN-1)

# Problem statement
`chad-options-chain-refresh.service` correctly degrades when the IBKR
`usfarm` / `ushmds` / `secdefil` farms are unresponsive (PR-04 graceful
degradation: empty cache with `error` field + `runtime/options_chain_refresh_failure.json`).
Code-side hardening landed in this batch:
- New health-monitor rule `R17b` reads the failure artefact and emits a
  CRITICAL finding (loud alert path) when the artefact is fresh.
- Strategies `alpha_options` and `omega_momentum_options` now call the
  shared `chain_usability()` validator and fail-closed (abstain / fall
  back to synthetic pricing) when the failure artefact is fresh OR the
  cache is stale.

What is still missing: a systemd-level `OnFailure=` directive on
`chad-options-chain-refresh.service` so that an *exit-code* failure
(not just a written-empty-cache failure) also routes a Telegram alert
via the standard chad-alert escalation unit. This PA proposes that
operator-applied edit.

# Evidence
- Audit §14.3: service is in `failed (exit-code)` state since 12:33:07Z;
  the failed unit state means the timer's next-fire will not retry until
  the next scheduled tick.
- `systemctl show chad-options-chain-refresh.service --property=OnFailure`
  is currently empty (per audit §5.5 OnFailure handling table).
- The escalation unit pattern is already in use elsewhere in CHAD; the
  audit recommends mirroring that pattern.

# Required closeout (operator-domain — not applied here)
1. Identify the standard escalation unit used by other chad-* services
   (e.g. `chad-alert@.service` or equivalent).
2. Add to `/etc/systemd/system/chad-options-chain-refresh.service`:
   ```
   [Unit]
   OnFailure=chad-alert@%n.service
   ```
3. `systemctl daemon-reload` (operator-domain).
4. Validate with a deliberate failure injection (operator-authorised
   test path) that the Telegram alert fires.

# Acceptance criteria
- `systemctl show chad-options-chain-refresh.service --property=OnFailure`
  returns the configured escalation unit.
- A deliberately-failed run produces a Telegram alert via the standard
  path within < 5 min.
- The new health-monitor rule `R17b` continues to fire as a redundant
  signal — defense-in-depth.

# Required tests
- (No new tests — this is an operator-applied unit edit; the code-side
  R17b rule is already covered by Phase 3 tests.)

# Operator approvals required
- **Approval 1 (systemd unit edit):** explicit GO for the unit-file edit
  per CLAUDE.md governance rule #6.
- **Approval 2 (daemon-reload):** explicit GO for `systemctl daemon-reload`
  after the edit.
- **Approval 3 (failure injection):** if a deliberate-failure validation
  is desired, operator authorises the failure-injection test path.

# Session 1 impact
- **Does this item block Session 1?** NO.
- The code-side fail-closed and loud-alert paths are in place.
- The systemd-level alert is defense-in-depth; absence does not affect
  paper-mode safety because (a) strategies abstain when the chain is
  unavailable and (b) R17b already surfaces the failure to the operator.

# No-live confirmation
This Pending Action does not authorize live trading.
ready_for_live must remain false.
allow_ibkr_live must remain false.
allow_ibkr_paper must remain true.
No broker orders may be placed or cancelled under this Pending Action.
