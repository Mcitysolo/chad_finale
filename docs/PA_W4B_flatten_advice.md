# PA-W4B — Pending Actions: Flatten-All (LC6) + Exit-Advice (J16)

Filed 2026-07-23 at W4B Phase-2 closure (branch `goal/wave4-flatten`,
worktree `chad_w4b`). Everything below is **operator-gated**; nothing here
was applied by the build wave. Governance: Pending Actions only — no direct
config mutation, no flag flips, no restarts.

## PA-W4B-1 — First flatten DRILL run (operator terminal, Channel 1)

The drill is the default no-`--execute` invocation of the flatten CLI
(`scripts/` — order-guard blocklisted; the operator runs it manually from
the terminal). It performs real read-only probes and pushes every IBKR
close through the adapter's `dry_run` short-circuit; it cancels nothing,
closes nothing, sends nothing (narration is formatted-not-sent).

Review artifact: `reports/ratification/PROOF_FLATTEN_DRILL_<YYYYMMDD>.json`
(`flatten_drill_proof.v1`).

Acceptance:
- `overall = DRILL_COMPLETE` (any `DRILL_GAPS` entry must be dispositioned);
- `probe.visibility_split` shows `all_open_orders >= own_scoped_count`
  (the cross-client 0/0 false-negative class is the thing being proven);
- `cancel.collateral_non_chad` enumerates every non-CHAD order that a real
  run WOULD kill, by name (D3 rider);
- `resolution.untouched` names every operator symbol that stays standing
  (D1 rider);
- clamp check: every target `quantity` ≤ the broker qty in `probe.positions`.

## PA-W4B-2 — CHAD_EXIT_ADVICE record→consume flip (SEPARATE operator GO)

The advice rule ships would-only: `record` mode emits `ADVICE_WOULD_CLOSE`
evidence and can never submit. The flip to `consume` is its own operator GO
(D6 rider), judged on the pre-registered criteria via
`scripts/exit_advice_report.py` (`exit_advice_review.v1`):

- corpus window: ≥ 5 trading days of record-mode evidence;
- every `ADVICE_WOULD_CLOSE` row reviewed by the operator;
- ALL GO criteria PASS: `recorder_flag_wall_intact`,
  `zero_advice_closes_on_excluded`, `all_clamps_within_broker`,
  `would_fire_corpus_nonempty`, `nothing_consumed_pre_flip`.

Expected first live case: `gamma|PSQ`. `gamma|MSFT` must appear ONLY as
`excluded=true` records — a single consumable MSFT row is a NO-GO.

Note: `CHAD_EXIT_ADVICE` is currently **unset (off)** on the live units.
Even record-mode activation needs a systemd drop-in + gated restart —
a separate operator action, not part of this wave.

## PA-W4B-3 — SSOT-v9.9 backlog amendment: flatten-drill cadence (D8)

Honest note (P7): **no SSOT text mandates a quarterly flatten drill.** What
exists: the chaos-drill re-drill carried with no cadence (SSOT v9.8:446)
and the 30-day STOP drill (`docs/PHASE9_PROMOTION_CHECKLIST.md:24`).
Proposal for the next SSOT cut (v9.9): a quarterly flatten-drill cadence,
aligned with the 30-day STOP drill, with the ratification proof artifact
above as the completion record. Until ratified, the drill cadence is
operator discretion — the code deliberately does NOT invent a cadence check.

## Standing facts for the reviewer

- Flatten gates are D4-minimal BY DECISION: paper/dry_run fail-closed +
  live_readiness cross-check + typed tokens. NOT SCR/reconciliation-gated —
  emergencies are precisely when those are broken.
- `--scope broker-all` (touches operator-excluded positions) requires the
  second token; the GAP-001 `apply_close_intents` chokepoint stays intact
  for every other caller in the system.
- Post-flatten hold (D5) = `operator_intent EXIT_ONLY` TTL 24h — never
  stop_bus/STOP (both freeze the close machinery, P12).
