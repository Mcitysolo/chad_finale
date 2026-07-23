# W4A Phase-2 GO Record — decisions, riders, rebase re-verification

**Recorded 2026-07-23.** The prior session committed PLAN_W4A (Phase 1, `173df9b`)
and ended before receiving its GO. This session received the GO and resumed:
D1–D8 **all as recommended** in PLAN_W4A §10, with three riders and two
INCIDENT-0723 inheritances, recorded here verbatim before any code.

## 1. Rebase

Branch base was `6979b99`; main advanced 17 commits (W4B flatten/advice
`af1afcd..8c04de1` + INCIDENT-0723 closeout `8e672f4..5013893`). Rebased clean:
plan commit `173df9b` → `975d5a9` on `main@5013893`. All plan line-anchors
re-verified on the rebased tree (§3).

## 2. The GO — decisions and riders

| Decision | Outcome |
|---|---|
| D1 | AS RECOMMENDED — both grains (family + setup_family), COEXIST with PerStrategyLossGuard. |
| D2 | AS RECOMMENDED — (a) stamp live regime at `trade_closer.to_payload`, additive, **forward-only**. **RIDER:** unstamped rows (all history + any stamp gap) count in an `unknown` regime bucket that may trip the **GLOBAL** leg of a fuse but never a regime-scoped leg. A regime-scoped bucket leg only ever counts rows whose stamp matches exactly; `regime=None`/absent ⇒ `unknown`. |
| D3 | AS RECOMMENDED — subsume `symbol_performance_blocker`; blocker untouched, no-op while its state file stays empty. |
| D4 | AS RECOMMENDED — (A) per-intent quantity multiplier post-SCR-CAUTIOUS (`live_loop.py` ~:2729) + Kraken twin in `execution_pipeline` (`chad/execution/execution_pipeline.py:1481` — plan cited the right lines, path corrected here). |
| D5 | AS RECOMMENDED — wire −15% emergency as fuse-gate entry-block, explicitly NOT stop_bus. **RIDER:** a **named regression test** proves a tripped LC5 emergency halt still permits (i) exit-overlay closes and (ii) flatten-all — "exits always free" is a test, not a placement argument. Delivered as `chad/tests/test_w4a_lc5_emergency_exits_free.py` (see §5). |
| D6 | AS RECOMMENDED — `drawdown_state.v2` bump + EXS7 pin/EXS1 row updated in the same commit. |
| D7 | AS RECOMMENDED — new `config/feed_policies.json`; exactly the two P18 wired sites. |
| D8 | DEFERRAL ACCEPTED — **and since absorbed**: the close-provenance stamp PLAN_W4A proposed for a later lane was built by **W4B-2 (`88890af`)** at `chad/core/position_reconciler.py:399-413` (`meta["action"]="CLOSE"`, `meta["close_origin"]="apply_close_intents"`, + reason/position_key/open_side/close_side; idempotency fingerprint proven unchanged). W4A cites and composes with it (§4) — nothing is rebuilt. The adapter-level fuse *backstop* remains deferred (unnecessary for correctness; stage-3 placement + structural bypass, per plan). |

## 3. Premise re-verification on the rebased tree

Verified directly this session (P-refs are PLAN_W4A §1):

| Ref | Status after rebase |
|---|---|
| P1 anchors | loss-guard block now `live_loop.py:2011-2047` (was 2009-2048) — same site. |
| P8/P9 stage-3 | **W4B added zero gates to stage-3 or `ibkr_adapter._submit_intent`** (`git diff af1afcd..5013893` touches neither file). Cluster order unchanged: SCR PAUSED :2665-2698 (flip-only bypass :2676) → CAUTIOUS scale :2700-2729 → same-side-open :2740-2754 → symbol blocker :2756-2767 (flip ∪ EXIT/CLOSE bypass :2759-2761) → cooldown :2769 → ML veto :2782+ → futures cap/gate :2871-2948 → submit. Fuse-gate slot (beside symbol blocker) and LC5 slot (post-:2729) both stand. |
| P9 close-stamp | **Partially superseded, favourably**: "close stamp dropped in `_close_intent_to_ibkr`" is no longer true — W4B-2 stamps provenance there (§2 D8). Overlay/reconciler closes still bypass stage-3 structurally (`apply_close_intents` :440 → adapter direct), so the fuse invariant's structural leg is unchanged. |
| P11 template | Margin-gate hook anchors moved: `ibkr_adapter.py:2544` (`_evaluate_margin_gate`), ctor :2016/:2024; Kraken slot `kraken_executor.py:322,371-376`. Pattern unchanged. |
| P13 | `to_payload` now `trade_closer.py:251`; setup_family forwarding :315. Unchanged. |
| P14 | Still true — no regime field in `trade_closer.py` (grep clean). D2(a) remains necessary. |
| P16 | Unchanged; evaluator stays in-loop, state published per cycle. |
| P18 | SCR gate still reads raw + fail-open (`SCR_GATE_READ_FAILED (fail-open)` :2697). DQ site 1 target confirmed. Note the PAUSED bypass is **flip-only** (:2676) — narrower than the symbol-blocker predicate; the DQ fail-closed verb will use the *wide* predicate (flip ∪ EXIT/CLOSE ∪ protective) per plan §8.1 so it can never be stricter against exits than today's gate. |

## 4. New seams since the plan — how the fuses compose (GO-required citations)

**4.1 Exit-advice seam (J16, W4B-1/-3/-4).** The D4-filter + advice recorder
(`chad/core/exit_advice.py:260-310`) runs at signal stage — sites
`live_execution_router.py:80,100`, `ibkr_execution_runner.py:434,447`,
`full_execution_cycle.py:227,241` — **upstream of stage-3**, so a fuse block can
never starve the recorder: dropped-urge evidence is written before any fuse
sees an intent. In `consume` mode the advice-fired close submits via
`position_exit_overlay.py:816-826 → :1093-1100 → apply_close_intents`
(`position_reconciler.py:440-494`) — the reconciler direct path that bypasses
stage-3. **Composition rule (binding for every W4A commit): no fuse code on the
`apply_close_intents → submit_strategy_trade_intents` path.** That path is the
sole equity/ETF close authority AND the flatten-all path; fuse enforcement
lives only at stage-3 (entries) + the Kraken mirror (reduce_only/`crypto_exit|`
exempt).

**4.2 Close-provenance stamps (W4B-2).** `position_reconciler.py:399-413`
stamps `meta.action="CLOSE"` / `meta.close_origin="apply_close_intents"`.
Composition: (i) the fuse gate's bypass predicate additionally honours
`meta.action=="CLOSE"` as a belt over the structural bypass — a stamped close
that somehow reached stage-3 would still pass; (ii) the deferred adapter
backstop, if a later wave builds it, keys on this stamp (its stated purpose,
per the W4B-2 commit message).

**4.3 Flatten-all (LC6, W4B-5/-6/-7).** Runs in its **own process**
(clientId `FLATTEN_ALL=7716`, `ibkr_client_ids.py:175`), never imports
live_loop; closes go `run_flatten → submit_closes → apply_close_intents`
(CHAD legs) or `_close_intent_to_ibkr` direct (operator legs, broker-all
scope) → `IbkrAdapter._submit_intent`; Kraken legs via TrustedFillEngine with
`flatten|` idempotency prefix and `reduce_only` reclamp. No stage-3 traversal
⇒ no fuse can touch it structurally; the D5 rider proves it by test anyway.
The W4B-7 order-guard blocklist (`.claude/hooks/chad-order-guard.sh`) is an
agent-shell channel-separation hook, not a runtime gate — no fuse interaction
(observed live this session: it blocked this agent's grep that named the
flatten script).

**4.4 Operator holds (W4B-8d).** `operator_intent_refresher.py` now preserves
unexpired EXIT_ONLY/DENY_ALL holds. Fuses neither read nor write
operator_intent — no interaction; noted so nobody wires a fuse into that
mechanism later without a decision.

## 5. INCIDENT-0723 inheritances (binding on the build)

**(a) Drill/rehearsal rows must be excluded from fuse counting BY
CONSTRUCTION.** The incident proved `status=dry_run` FILLS rows minted 8 fake
closed trades (`audits/INCIDENT_20260723_DRILL_EXHAUST_FALSE_FLAT.md` §3.2) —
which, being gamma losses dated in the current session era, would be visible
to a naive fuse counter TODAY (the 0723 quarantine manifest is prepared at
`docs/incident_0723_quarantine_manifest.json` but NOT yet operator-applied to
`runtime/`). Exclusion sites the incident fixes created, which the fuse box
cites and joins:
- `trade_closer.py:352` `_TRUSTED_FILL_STATUSES={"filled","paper_fill"}` (W4B-8a, root fix; ingest gate :443-446);
- `position_reconciler.py:77-83` `_EVIDENCE_SKIP_FILL_STATUSES` (W4B-8b, canonical never-money skip-set; writer gate :509);
- `position_reconciler.py:88` `_PRICE_ELIGIBLE_FILL_STATUSES` (W4B-8c, price cascade gate :308-311);
- `chad/validation/trade_log_adapter.py:95,394-417` harness trusted set + `trust_exclusion`;
- `chad/portfolio/ibkr_paper_ledger_watcher.py:295,315-316`; `chad/core/position_guard.py:33,298`;
- pinned cross-site by `chad/tests/test_w4b8_exhaust_hygiene_sites.py:109-120` (no trusted set may intersect the skip-set).

The fuse box becomes **consumer #7 of that census**: its trusted-status set
joins the 8f disjointness invariant, and — because `closed_trade.v1` carries
no status field and legacy fakes carry no in-band marker — the counting engine
independently **verifies fill provenance**: for the session window it maps
window-dated `FILLS_*.ndjson` `fill_id→status`, and any closed trade citing a
window-local fill whose status is not genuine (∉ {"paper_fill","fill","filled"}
lowercased) is excluded and tallied as `excluded_unverified_provenance`. This
catches the 8 existing fakes (dry_run exit legs) AND the reverse shape
(dry_run entry leg, paper_fill exit — the 13:31:30 PSQ row) without trusting
upstream ingest fixes, plus quarantine sets (`chad/utils/quarantine.py:268`,
matched on record_hash / fill_id / any of `payload.fill_ids`, the
`trade_stats_engine.py:395-421` idiom) and the pnl_untrusted /
scoring_excluded / validate_only mirrors. **Fuse-side test:** a replayed
drill (the `test_incident_0723_dry_run_exclusion.py` row shapes) must produce
zero fuse counter movement — a drill must never trip a fuse.

**(b) Fuse state joins the sentinel TTL table.** `runtime/fuse_box_state.json`
gets a `config/exterminator.json` feeds row (`ttl_verified: true`, publisher
citation) in the same commit that first publishes it heartbeat-inclusive-of-OFF
— a silent fuse is the XOV lesson wearing a new hat. (Plan §7.4 already
required this; the GO elevates it to inheritance status: it lands at W4A-1
with the state publisher, not at W4A-9.)

## 6. Build-plan deltas vs PLAN_W4A §9 (GO-driven, no scope growth)

- W4A-1 additionally: provenance predicate #7 (§5a), fuse trusted set added to
  the 8f census invariant, exterminator feeds row + drill-never-trips test
  (moved up from W4A-9 per §5b).
- W4A-2 regime stamp carries the D2 rider semantics (unknown bucket, global
  leg only).
- W4A-7 additionally: `chad/tests/test_w4a_lc5_emergency_exits_free.py` — the
  D5 rider named tests: `test_lc5_emergency_halt_permits_overlay_closes`,
  `test_lc5_emergency_halt_permits_flatten_all`.
- W4A-9 keeps: EXS7 pin, `scripts/clear_fuse.py`, docs, closure record.

Everything else per PLAN_W4A §§2-9 unchanged. STOP remains at the push/merge
decision.
