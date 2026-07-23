# PA-INC-0723 — Pending Actions: drill-exhaust incident disposition & remediation

Companion to `audits/INCIDENT_20260723_DRILL_EXHAUST_FALSE_FLAT.md`.
Everything here is **operator-gated** (GO required per item). Code fixes land
as W4B-8 commits — repo-side only; they activate at the next gated
chad-live-loop restart. No runtime file below is touched by the build wave.

## PA-INC-1 — Quarantine the incident's ledger rows (GO required)

Prepared manifest (exact IDs, generated from the ledgers):
`docs/incident_0723_quarantine_manifest.json` — 7 dry_run fill_ids +
8 fake closed-trade record_hashes (net fake pnl ≈ −77.81).

Operator applies by copying it to **both** consumer locations:

```
cp docs/incident_0723_quarantine_manifest.json runtime/quarantine_manifest_incident0723.json
cp docs/incident_0723_quarantine_manifest.json data/fills/quarantine_incident0723.json
```

- `runtime/quarantine_manifest_*.json` → SCR (`trade_stats_engine`) +
  `expectancy_tracker` exclusion sets (`chad/utils/quarantine.py:get_quarantine_sets`).
- `data/fills/quarantine_*.json` → trade_closer sidecar
  (`_extract_fill` quarantine check) so a future full-rescan can never
  re-ingest the dry_run rows.

Effect: the 8 fake round-trips leave SCR/expectancy at their next compute;
FILLS/trade_history files stay unmodified (hash-chain intact).

## PA-INC-2 — Book repair: restore guard/queue truth (GO required, report-first)

Current corruption (after quarantine, still present in *state* files):
- `trade_closer_state.json` queues: phantom `("g","PSQ")` SELL 5 lot (citing
  quarantined dry_run fill `0be6fb53…`); gamma lots for IWM 200 / MA 10 /
  SVXY 156 / UNH 228 / V 181+9 / PSQ 5 were consumed by the fake round-trips.
- `position_guard.json`: `gamma|IWM`, `gamma|MA`, `gamma|PSQ` closed;
  SVXY/UNH/V re-seeded only at the small re-entry adds (7/6/5).

**Option A (recommended) — broker-truth re-seed.** Run the guarded
DRIFT-RECON ledger↔broker reconciliation (dry-run first, review, then
operator-typed `--execute`): broker_sync anchors re-seed to the live probe
(IWM 200, MA 10, PSQ 10, SVXY 163, UNH 234, V 195), the phantom `g|PSQ`
queue lot is cleared (its fill is quarantined), and the ISSUE-56 v2 partial-
attribution machinery re-claims strategy ownership over subsequent cycles.
No hand-edits to the guard.

**Option B — overlay-managed unwind of the +23 adds.** Reduce-only closes of
PSQ 5 / SVXY 7 / UNH 6 / V 5 via the exit overlay, then Option A for the
remaining book. More churn; only preferable if the operator wants the adds
gone rather than absorbed.

**+23-share disposition decision (operator):** keep-and-absorb (Option A
alone) vs unwind (Option B). The adds are same-direction extensions of
held longs, ≈ $4.8k notional.

Sequencing: PA-INC-1 → deploy W4B-8 fixes → gated restart → PA-INC-2.
(Restart first, or the running loop's stale in-memory closer state can
rewrite what the repair just fixed.)

## PA-INC-3 — Make the EXIT_ONLY brake real (D4)

- **D4a (landed in W4B-8, activates on next timer fire after deploy):**
  `cmd_refresh` must preserve an unexpired, explicitly-`set` EXIT_ONLY /
  DENY_ALL instead of unconditionally rewriting ALLOW_LIVE every 10 min.
- **D4b (design item, separate GO):** the paper entry path never consults
  operator_intent — no component in live_loop/orchestrator/ibkr_adapter
  reads it. Proposal: LiveGate-style entry check in the paper submit path
  (entries refused under EXIT_ONLY/DENY_ALL; closes/overlay exempt).
  Until D4b lands, the only real paper brake is stopping chad-live-loop.

## PA-INC-4 — Activation & verification sequence (operator GO)

1. Review + merge W4B-8 commits (already pushed or push-pending per repo policy).
2. Apply PA-INC-1 (quarantine copies).
3. `sudo systemctl restart chad-live-loop` (explicit instruction required).
4. Verify within 2 cycles:
   - no new `dry_run`/`market_closed` rows appended to `data/fills/FILLS_*.ndjson`
     (writer-side guard active);
   - `position_guard_drift.json` counts stop growing;
   - SCR rollup drops the 8 quarantined trades (effective_trades −8).
5. Run PA-INC-2 Option A dry-run; review; typed `--execute` on GO.
6. Re-run the flatten DRILL (now exhaust-safe) and confirm:
   `overall=DRILL_COMPLETE`, **zero rows added to FILLS**, PSQ correctly
   attributed (`gamma|PSQ` targeted, no `not_chad_attributed` remainder).

## W4B-8 code fixes (repo-side, mapped to defects)

| Commit | Defect | Change |
|--------|--------|--------|
| W4B-8a | D1 | remove `"dry_run"` from `trade_closer._TRUSTED_FILL_STATUSES` + regression replay test (drill rows → ingest → rebuild → guard must NOT flat) |
| W4B-8b | D2 | `apply_close_intents` refuses to write evidence for unconfirmed adapter statuses (mirror of hot-path SKIP_EVIDENCE guard) |
| W4B-8c | D3 | tier-1 close-price cascade requires confirmed statuses |
| W4B-8d | D4a | `cmd_refresh` preserves unexpired explicit holds |
| W4B-8e | — | flatten CLI bare-terminal-proof (sys.path self-locate, mode inference from drop-ins, subprocess test to DRILL_COMPLETE) + PA-W4B-1 canonical command |
| W4B-8f | — | exhaust-hygiene exclusion tests pinning every consumer-census site |

Deferred (named, not silent): D5 ML veto enforcement (separate GO),
D6 titlecase `Filled` canon mapping (PA-EP8v2 scope), D7 harvester
double-write (EP backlog).
