# Bug B — Pending Action: Fix the futures runaway (fill-feedback corruption + position-unaware entry)

**Status:** PENDING — spec only; NO code until operator GO. Risk-relevant (touches trade_closer seeder, paper-fill harvester, live_loop entry gate).
**Date:** 2026-06-03
**Related:** Bug B root-cause forensic 2026-06-03; gate drop-in 91-disable-futures-exec.conf (2026-05-30, triple-redundant, ARMED + actively firing); harvester trade-history bridge commit 90cf45e; GAP-028 (broker_sync strategy-scoped, PERMISSIVE).

## 1. Root cause (two composed defects)
**Defect B (ROOT — fill-feedback corruption).** chad/portfolio/ibkr_paper_fill_harvester.py (5-min timer; bridge added 90cf45e) harvests each broker fill into FILLS_*.ndjson AND writes a closed_trade.v1 record into trade_history_*.ndjson for the SAME single fill (pnl=0.0, entry_time==exit_time, entry_price==exit_price, fill_ids=[<one fill_id>], tag ibkr_harvest, ~lines 428-447). Every harvested OPEN is mis-recorded as an instantly-closed zero-PnL round trip. TradeCloser.seed_processed_from_trade_history() (trade_closer.py:627) then marks those fill_ids already-consumed -> the open is never enqueued into the FIFO -> the position queue is empty despite the broker holding the position -> _rebuild_guard_from_paper_ledger mirrors the empty queue -> guard auto-closes the entry within ~1 cycle -> is_same_side_open() sees no open -> re-entry allowed -> +2/cycle -> broker accumulates (M6E 334 BUYs / 0 SELLs -> +217). Proof-by-contrast: gamma_futures|M2K only partially poisoned (16/88) -> 72 clean fills enqueued -> queue persists -> same-side blocked (M2K absent from SKIP logs).
**Defect A (AMPLIFIER — position-unaware entry).** omega_macro + gamma_futures emit directional entries every cycle with no position-awareness (no broker read, no cumulative cap, no cooldown); max_contracts is per-ORDER only. alpha_futures is position-aware but reads the same poisoned ledger. All three rely on the single trade_closer-FIFO-backed same-side dedupe, which Defect B corrupts.

## 2. The fix (sequenced; env gate stays ARMED throughout)
**Fix B — stop the phantom-close poisoning (ROOT; ship first).** Per upstream-abstain-over-downstream-repair, source fix primary + consumer defense:
- (Source, primary) Harvester: first understand commit 90cf45e's intent. Stop writing closed_trade.v1 round-trip records for individual OPEN fills — restrict to genuine matched closes, or remove if redundant with TradeCloser's own FIFO emission. The harvester owns FILLS_*.ndjson; closed_trade.v1 emission is TradeCloser's responsibility.
- (Consumer, defense-in-depth) seed_processed_from_trade_history(): ignore harvester phantoms. VALIDATE the discriminator before relying on it — prefer the specific `ibkr_harvest` tag over a structural `len(fill_ids) < 2` heuristic; confirm no legitimate close carries the chosen signature.
- Note: historical poisoned fill_ids stay processed (NO retro-rebuild); current broker truth carried by broker_sync; legacy 217 resolved by disposition (§4), not retro-rebuild.

**Fix A — cumulative broker-truth position cap (the real guardrail; ship second).** At the shared live_loop entry chokepoint (next to is_same_side_open, ~line 1934): before submitting a FUT open, read net broker position for the symbol (positions_truth / broker_sync|<sym>, any-strategy attribution) and refuse adds beyond spec.max_contracts CUMULATIVE. Converts per-order max into the per-position bound the docstrings already claim. Shared — one place fixes omega_macro, gamma_futures, alpha_futures.

**Optional belt (after A+B):** per-(strategy,symbol) entry cooldown in live_loop, reusing the omega.py/alpha_intraday.py pattern.

## 3. Sequencing & gate policy
1. The env gate (91-disable-futures-exec.conf) stays ARMED throughout — load-bearing until Fix A is live-verified.
2. Ship Fix B -> verify: a NEW futures OPEN fill enqueues into the FIFO and surfaces as guard-open (persists, not auto-closed within a cycle); same-side dedupe blocks re-entry.
3. Ship Fix A -> verify: cumulative cap binds in logs (a FUT open beyond cumulative max refused with a distinct marker).
4. ONLY THEN is the gate removable — via a SEPARATE gated PA (observe-first).
5. Disposition (§4) is a SEPARATE gated action, orthogonal to A/B, safe with the gate armed.

## 4. Disposition of the legacy book (separate gated PA)
Existing M6E +217 / M2K -25 / MCL +50 (~$4.5M CAD net, 17.6x gross, +$39.9K CAD uPnL) is a runaway artifact, not a thesis. Recommended: flatten (gate permits exits; banks +$39.9K, kills leverage, cleans Epoch 3 statistics; re-accumulation blocked by the armed gate). Staged unwind (M6E first, ~79% of gross) is the lower-impact alternative. Hold not recommended (contaminates equity/SCR/risk-cap stats). Requires its own PA + GO (places exit orders).

## 5. Acceptance criteria
- After Fix B: a new futures open fill appears in the trade_closer FIFO and the position_guard strategy|symbol entry is OPEN and persists; phantom closed_trade.v1 no longer emitted (source) and/or no longer consumed by the seeder (consumer).
- After Fix A: a FUT open exceeding cumulative spec.max_contracts is refused with a distinct log marker; cumulative position never exceeds the cap.
- Gate removable only after BOTH verified live.

## 6. Out of scope
- Retro-rebuilding historical poisoned queues (poisoned ids stay processed; legacy position handled by §4 flatten).
- Changing broker_sync strategy-scoping (GAP-028 Option B PERMISSIVE).
- Removing the env gate (separate gated PA after Fix A verified).

## 7. Risks & guardrails
- Risk-relevant trading-state code (trade_closer seeder, harvester, live_loop entry gate). Each increment read-confirmed before edit; full suite + targeted tests; no live restart without a per-increment gate (rule #7).
- Fix B discriminator validated (no legit close skipped) before relying on it.
- Gate MUST remain armed until Fix A is live-verified.
- live_loop changes require a gated live_loop restart, with verification the runaway is now self-blocked (not merely gate-blocked).

## 8. Implementation gate
No code until operator GO on this spec. Then: Fix B (read-confirm harvester+seeder -> implement -> tests -> verify queue rebuild on new fills) -> Fix A (read-confirm entry gate -> implement -> tests -> verify cap binds) -> separate gated PAs for disposition (flatten) and gate removal.
