# Pending Action — Bug B disposition: futures book + env-gate decision
Date: 2026-06-04  •  Author: TEAM CHAD (issued) / SOLO (decides)  •  Status: PENDING operator decision  •  Priority: MEDIUM (no operational/runaway risk; live market exposure persists)

## 1. Context
Bug B core is closed (Fix B + Fix A live-verified; env gate armed AND cap proven behind it = belt-and-suspenders). Two INDEPENDENT disposition decisions remain, both the operator's call: (A) flatten the runaway futures book; (B) remove the env-gate band-aid so the cap is the sole futures guard. This PA records the decision facts and engineering-safe orderings. It executes nothing. When a path is chosen, gate removal in particular gets its own Channel 1 execution PA.

## 2. Current book (broker truth, positions_truth GREEN @ 2026-06-04T01:21:36Z; no broker calls made)
| Symbol | conId | Net | Avg entry | Last | uPnL USD | uPnL CAD |
|---|---|---|---|---|---|---|
| M6E | 838628915 | +217 | 1.16413 | 1.1611 | -8,209 | -11,405 |
| M2K | 770561189 | -25 | 2890.75 | 2901.8 | -1,381 | -1,919 |
| MCL | 661016559 | +50 | 88.38 | 95.13 | +33,733 | +46,870 |
| Total | — | — | — | — | +24,143 | +33,546 |
- FX 1.38942 USD/CAD (from portfolio_snapshot pair). Multipliers: M6E 12,500 EUR, M2K $5, MCL 100 bbl.
- Gross notional ~$3.99M USD (~$5.54M CAD) -> ~17.9x gross leverage vs ~309,449 CAD equity; net ~$3.26M USD -> ~14.6x. ~79% of gross is the M6E long.
- Static: YES. Last futures fill 2026-06-01T17:19:39Z (pre-fix); zero futures fills since; net pinned via CAP_BLOCK every cycle. MES/MGC are replay-only artifacts, flat at broker.
- LIVE RISK, NOT FROZEN PnL: mark-to-market moves with prices. Total is +$33.5K CAD now (M6E has flipped to a small loss as EUR slid; MCL carries the book). The +PnL can erode; this remains a live ~17.9x position.

## 3. Open-order caveat — RESOLVED 2026-06-04 (re-probe before actual flatten)
M6E order 5137 was carried REAL_BROKER_ORDER_ACTIVE on 2026-05-27. Verified 2026-06-04 via read-only IBKR probe (ib_async readonly=True, clientId 9777 confirmed free, reqAllOpenOrdersAsync across ALL clientIds): the account has ZERO open/working orders, and order 5137 is NOT PRESENT (filled/cancelled/expired since 05-27 — consistent with the last M6E fills landing 2026-06-01 via broker_order_id 7713, after which the book froze at +217). The earlier "5137" broker_events hits were coincidental sha256 substrings, not an order id. No working order of any kind exists to collide with a flatten.
NOTE: point-in-time read — re-run the same read-only probe immediately before any actual flatten execution as good practice.

## 4. Gate removal — mechanics + effect
- The three flags exist ONLY in /etc/systemd/system/chad-live-loop.service.d/91-disable-futures-exec.conf. Removal = delete/disable that drop-in -> daemon-reload -> gated chad-live-loop restart (Channel 1, rules #6/#7, explicit GO).
- CAP-COMPLETENESS CONFIRMED — complete replacement, no hole: all three flags feed _futures_execution_disabled (live_loop.py:200) consumed at one site (2277); the cap (2227) sits at the same chokepoint with the byte-identical predicate (sec_type==FUT and not exit and not flip), evaluated BEFORE the gate. Journal shows only CAP_BLOCK, zero DISABLED_SKIP. Everything the gate blocks, the cap blocks or blocks harder (over-cap -> BLOCK; unknown symbol -> cap 0 -> BLOCK; stale/not-GREEN truth -> UNVERIFIED fail-closed).
- On removal the intended deltas: under-cap opens resume (ZN/ZB/MYM/MES/MNQ/MGC across omega_macro / gamma_futures / alpha_futures / alpha_intraday_micro) and |net|-reducing opens pass (risk-reducing). Over-cap legs stay blocked: M6E (217>>3), M2K (-25>>5), MCL (50>>2) keep getting CAP_BLOCK.

## 5. Flatten — mechanics (informational; not a recommendation to trade)
- Closes flow WITH the gate armed: cap and gate both exempt exits/flips; the reconciler close path (apply_close_intents, live_loop.py:1700) runs outside both guards. A flatten needs no gate change -> (A) and (B) are independent.
- No automated path will flatten this book: all futures guard entries are open:false -> reconciler emits nothing; no strategy holds open attribution -> no exit intents; chad-micro-eod-flatten is scoped to alpha_intraday_micro MES/MNQ only; 0 drifts.
- Practical flatten = operator action: manual close at IBKR, or a one-off operator invocation of the apply_close_intents pattern (writes proper paper-exec evidence; M6E/M2K/MCL are NOT in the 9-equity exclusion list). Fills harvest back as broker_sync; Fix B already prevents phantom-close poisoning of the seeder. Resolve order 5137 first.

## 6. Sequencing outcomes
- (a) keep gate + leave book: status quo; double-blocked opens; book stays on at ~17.9x gross (live market risk).
- (b) flatten -> then remove gate: cleanest; cap-only futures resumes from a flat book; cap's first real open-test starts at net 0; opens never interact with the runaway book.
- (c) remove gate -> then flatten: DOMINATED; same end state as (b) but a backup-free window at peak exposure. Avoid.
- (d) keep gate + flatten only: most conservative; removes book risk, defers the gate decision indefinitely.

PREREQUISITE on gate removal (added 2026-06-04): options (b) and (c) — anything that REMOVES the env gate / re-enables futures execution — now require the Bug A fix (L1 event-loop/fd leak) to land FIRST. The env gate is the leak's current mask; re-enabling futures re-arms the high-frequency caller behind the 2026-05-30 fd-exhaustion freeze. The Fix A cap blocks over-cap submits but does not address the leak. Options (a) and (d) (gate stays armed) are unaffected. See PA L1_bug_a_event_loop_leak_2026-06-04.

## 7. Recommendation (ordering only — NOT advising the flatten trade itself)
The two decisions are independent (flatten needs no gate change; gate removal needs no flatten — the cap holds either way). If both are ever done, the engineering-safe orderings are (b), or (d) then later (b). (c) is strictly dominated. Whichever path: re-verify/cancel order 5137 broker-side first; gate removal is a separate Channel 1 Pending Action (drop-in change + gated restart, rules #6/#7).

## 8. Status log
- 2026-06-04: authored from read-only disposition audit (positions_truth GREEN @ 01:21:36Z, no broker calls). PENDING operator decision on (A) flatten and (B) gate removal. No action taken.
- 2026-06-04: §3 open-order caveat RESOLVED via read-only reqAllOpenOrdersAsync probe (clientId 9777, readonly=True) — zero open orders account-wide, order 5137 NOT PRESENT. Overall decision (A flatten / B gate removal) remains PENDING operator decision.
- 2026-06-04: L1/Bug A read-first found the env gate doubles as the mask for a dormant event-loop/fd leak (root cause: per-call thread+loop mint in ibkr_adapter._call_with_timeout). Gate-removal options (b/c) re-classified as GATED on the Bug A fix landing first; flatten (A) and gate-keep options unaffected.
