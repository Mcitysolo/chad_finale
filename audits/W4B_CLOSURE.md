# W4B Phase-2 CLOSURE — Flatten-All (LC6) + Exit-Routing Unification (J16)

2026-07-23, branch `goal/wave4-flatten` (worktree `chad_w4b`, base
`main@6979b99`). Built under D1–D8 GO + riders (report names operator
symbols left open; proof enumerates non-CHAD orders killed; consume flip =
separate operator GO; close-stamp rider done in W4B-2). **Stopped at the
push/merge decision** — no push, no merge, no flag flips, no live restarts.

## Commits

| # | Commit | Contents |
|---|---|---|
| W4B-0 | 05da61b | baseline capture — 15-fail set @ 6979b99 |
| W4B-1 | c7f8036 | exit-advice recorder, 3-site seam, heartbeat (J16) |
| W4B-2 | 88890af | close-provenance stamps at `_close_intent_to_ibkr` (cross-lane rider) |
| W4B-3 | a81dcdd | advice aggregation + overlay `strategy_advice` rule (would-only in record) |
| W4B-4 | d7ff63a | advice review tool (`exit_advice_review.v1`) + e2e receipt regressions |
| W4B-5 | 85fc043 | flatten core — gates/tokens/probe/scope/clamps + FLATTEN_ALL=7716 registry |
| W4B-6 | 0b061f1 | flatten act phases — cancel/close/confirm both lanes + orchestration |
| W4B-7 | (this) | drill polish, order-guard blocklist, PA doc, closure record |

## Verification discipline

Every commit: `py_compile` + full-suite set-diff vs `audits/W4B_BASELINE.md`
(failures ⊆ the named baseline-15 every step; final: 15 failed / 4100+
passed) + `CHAD_SKIP_IB_CONNECT=1 full_cycle_preview` smoke at the wiring
commits. Neighbor overlay suites re-verified green before resuming the
half-built units (106 + 18 passed).

## Riders honored (where)

- **Report names operator symbols left open** — `resolve_targets.untouched`
  (incl. `broker_flat_ledger_only` split-brain naming) + Phase-3
  `untouched_named` restatement with `EXCLUDED_UNTOUCHED` verdicts.
- **Proof enumerates non-CHAD orders killed** — `cancel.collateral_non_chad`
  by order/clientId, in drill AND execute payloads.
- **Consume flip = separate operator GO** — record-mode is would-only by
  construction; the flip criteria are pre-registered in
  `docs/PA_W4B_flatten_advice.md` (PA-W4B-2) against the
  `exit_advice_review.v1` GO block.
- **Close-stamp rider** — landed W4B-2; e2e test proves the stamps ride
  every advice-fired close to the adapter boundary.

## Deliberate design facts (for the future reader)

- Drill and execute share ONE code path; drill differs only in adapter
  `dry_run`, cancel no-op, kraken build-never-dispatch, narration
  formatted-not-sent, and verdict `DRILL_COMPLETE|DRILL_GAPS`.
- Chad legs submit through `apply_close_intents` (chokepoint + evidence +
  positive-confirmation guard mutation) behind a `RecordingAdapter` proxy so
  the per-order SLA still gets the `SubmittedOrder`s. Broker-all operator
  legs bypass the chokepoint BY DESIGN under the double token — same adapter.
- The order-guard hook now blocklists the flatten CLI (worktree copy;
  becomes active at merge). Agent shells cannot even name it; the test file
  deliberately avoids the blocked token in its filename so suites keep
  running post-merge.
- Kraken lane truth is the FIFO lot book; pair mapping uses the engine's own
  inverse table (`XBTUSD`←`BTC-USD`) because slash/identity forms do not
  round-trip and would OPEN a new position instead of reducing.

## Activation reality (nothing armed by this wave)

`CHAD_EXIT_ADVICE` unset (off) on live units — even record mode needs a
drop-in + gated restart. The flatten CLI is operator-terminal-only and
drill-by-default. All operator steps are in `docs/PA_W4B_flatten_advice.md`.
