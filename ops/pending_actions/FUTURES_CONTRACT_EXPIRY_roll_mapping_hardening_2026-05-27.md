# FUTURES-ROLL — Contract-Expiry / Roll-Forward Registry Has Not Advanced Past 2026-05-27 Expiries

# Status: PROPOSED

# Source audit
- `reports/forensic_audits/CHAD_FORENSIC_FULL_SYSTEM_AUDIT_20260527T142951Z.md`
  - §14.1 (chad-ibkr-bar-provider 588 errors/24h sample)
  - §14.2 (chad-live-loop Error 201 physical-delivery rejections)
  - §17 R7-A (root-cause grouping)
  - §19 closeout item 4 (FUTURES-ROLL-1)

# Severity
**HIGH** (Effort M) per audit §18.

# Problem statement
The futures-contract registry / front-month resolver is **still polling contracts that expire today (2026-05-27)** even though the strategy/risk layer has effectively stopped routing them. Result:

- `chad-ibkr-bar-provider.service` emits **~588 ERROR/Exception lines in 24 h** — overwhelmingly `Error 162: HMDS query returned no data: SILK6@COMEX Trades` (silver futures, conId 788288505, `lastTradeDateOrContractMonth=20260527`).
- `chad-live-loop.service` emits `Error 201: Order rejected — IBKR near-expiration / physical-delivery rules. Place order during contract delivery window. Please declare intent in the Futures Physical Delivery Tool.` for conId 712565978 (multiple distinct reqIds, all today).
- Position reconciliation surfaces show `MCL` and `MES` as continuing positions with their own roll-forward stories; today's MGC drift (see R1 Pending Action) had a parallel cause class (broker_truth_rebuild).

The system is **safe today** because the offending contracts are not in the live-routed regime universe, but the noise masks real errors in audit greps and the rejection on a near-expiry future has actually been attempted by the live-loop.

# Evidence
1. **chad-ibkr-bar-provider sample** (5 of 588 lines):
   ```
   ib_async.wrapper Error 162, reqId 12811: Historical Market Data Service error message:HMDS query returned no data: SILK6@COMEX Trades, contract: Contract(secType='FUT', conId=788288505, symbol='SI', lastTradeDateOrContractMonth='20260527', multiplier='1000', exchange='COMEX', currency='USD', localSymbol='SILK6', tradingClass='SIL')
   ```
   Pattern: every ~30 s, same conId, same `localSymbol=SILK6`, same `lastTradeDateOrContractMonth=20260527`.

2. **chad-live-loop sample** (4 of 6+ lines today):
   ```
   Error 201, reqId 5238: Order rejected - reason:This order does not comply with our order handling rules for derivatives subject to IBKR near-expiration and physical delivery risk policies.
   Please refer to our website for further details.
   You are placing an order during the contracts delivery window.
   Before submitting this order, please declare your intent to receive in the Futures Physical Delivery Tool.
   ```
   conId 712565978 (separate from SILK6). Multiple reqIds today (5230, 5232, 5236, 5238).

3. **Today's routed futures universe (from live-loop log):** alpha_futures and gamma_futures on MES, MGC, MCL, M2K, M6E (per `INTENT` and `ALWAYS-ACTIVE ROUTE` entries). **SI is not in any routed strategy today.**

4. **Position snapshot at audit time:** MES, MCL, M2K, M6E present as 19-symbol universe. SI absent.

5. **Existing related work:**
   - `git log` shows `cdab294 PR-M2K-MYM: map micro futures in IBKR bar provider` (recent).
   - `ecb370c PR-M2K-MYM: mark bar-provider mapping runtime verified` (verified for M2K/MYM only).
   - `d3fc7a1 GAP-037 (Box-016/030): futures contract construction in close-intent + MCL post-2026-05-18 schedule advance + MYM registry` — MCL was rolled; SI was not.
   - `chad/market_data/futures_contract_resolver` is referenced in `position_guard.meta.contract_month_source` (per today's MGC entry: `contract_month=202606`).
6. **chad-futures-roll-refresh.timer** exists and is active per `systemctl list-timers chad-*`. It runs daily, but has evidently **not advanced SI past today's expiry**.

# Affected files / services / artifacts
- `chad-ibkr-bar-provider.service` (noisy)
- `chad-live-loop.service` (Error 201 attempts on near-expiry contract)
- `chad-futures-roll-refresh.service` + `.timer` (daily refresh — apparently incomplete for SI)
- `chad/market_data/futures_contract_resolver.py` (or equivalent — referenced by position_guard.meta)
- `chad/market_data/ibkr_bars_provider.py` (or similar — bar provider universe)
- `chad/market_data/ibkr_daily_bars_refresh.py` (related)
- `runtime/futures_roll_state.json` (`futures_roll_state.v1` — declared in v9.3 §12.3, present in runtime)
- `runtime/ibkr_bars_cache.json` (no schema_version, but actively refreshed every ~10 s)
- Position guard meta for futures positions (`position_guard.json::alpha_futures|MGC::meta::contract_month_source=chad.market_data.futures_contract_resolver`)

# Root-cause hypothesis
1. **Universe vs registry divergence**: the bar provider polls a static or semi-static list of futures (`SI`, `MES`, `MCL`, etc.). The `futures_contract_resolver` advances the front-month for the strategies that route a given symbol, but the bar provider's polling list is not gated by routing — it polls SI even though no strategy is currently using SI.
2. **No "expiry-aware skip"**: when `lastTradeDateOrContractMonth` is today or in the past for a polled contract, the bar provider should skip the poll (or use the next contract month) rather than request data from IBKR for a contract that no longer exists in the HMDS.
3. **Live-loop Error 201**: a strategy emitted an intent on a near-expiry contract (likely MCL or SI's next-month equivalent), and the executor sent the order before consulting the IBKR physical-delivery rule. The 201 is a soft fail (order rejected, no fill) but indicates the executor's pre-flight contract-eligibility check is missing or stale.

# Why this matters
1. **Journal noise** — 588 errors/24h on a single expired contract masks real errors in operator triage. The audit's "every check finds a new surprise" pattern partly traces to this signal-to-noise gap.
2. **Soft fails on real orders** — Error 201 on a near-expiry contract means the strategy got close to a real broker rejection at order-submit time. On a live day this would still be a soft fail but would also pollute the broker-order audit trail with rejected attempts.
3. **Roll-forward governance** — the `chad-futures-roll-refresh.timer` is supposed to manage this; its evident incompleteness for SI (and possibly other instruments) is a process gap.
4. **R1 ledger reconciliation** — futures positions opened on near-expiry contracts (MGC today) generate ledger entries that the snapshot publisher then has to reconcile with broker truth; the dual-authority gap (R1) is amplified by every roll event.

# Current safety posture
- All live gates held: `ready_for_live=false`, `allow_ibkr_live=false`, `allow_ibkr_paper=true`.
- Error 162 is a data-fetch error, not a fill/PnL event. No financial impact.
- Error 201 is a broker-rejected order, not a filled order. No financial impact (paper mode).
- The position guard's `contract_month_resolved_at_utc` for MGC today shows the resolver IS firing for actively-routed symbols (MGC resolved to 202606); the gap is for symbols polled by the bar provider but not routed.
- v8.9 SS01 NEVER_AUTO_RESTART invariant on `chad-live-loop` and `chad-orchestrator` is preserved.

# Scope for future remediation
1. **Bar-provider expiry gate**: in the bar provider universe iterator, skip any contract whose `lastTradeDateOrContractMonth` ≤ today. Optionally substitute the next-month contract via the same `futures_contract_resolver`.
2. **Roll-refresh completeness**: extend `chad-futures-roll-refresh.service` to cover SI (and audit for any other instrument whose roll-forward is missing). Add a test that asserts every futures symbol in the bar-provider universe has a non-expired `lastTradeDateOrContractMonth` after the refresh runs.
3. **Live-loop pre-flight contract eligibility**: before submitting any futures order, check `lastTradeDateOrContractMonth` and the IBKR physical-delivery window. If within the rejection window, abstain at the executor with an `INTENT_ABSTAIN: contract_in_physical_delivery_window` log line rather than letting IBKR reject.
4. **Roll-state observability**: ensure `runtime/futures_roll_state.json` reflects the canonical front-month per symbol and ties to `futures_contract_resolver` output; surface a Telegram alert when a known symbol's resolver returns an expired contract.
5. **Add `schema_version`** to `runtime/ibkr_bars_cache.json` (currently absent — overlaps with SCHEMA-VERSION-1 / MED).

# Explicitly out of scope
- Live-mode enablement.
- Cancellation of any live broker order (none placed under this Pending Action).
- Modification of `chad/strategies/*.py` to add per-strategy expiry-awareness — this is at the bar-provider and executor levels.
- Migration off ib_async (separate Pending Action).
- Modification of `runtime/epoch_state.json`.

# Required tests
- New: test that asserts the bar-provider universe iterator skips any contract whose `lastTradeDateOrContractMonth` is in the past or today (mocked clock).
- New: test that asserts every symbol in the bar-provider universe has a non-expired roll-state entry after `chad-futures-roll-refresh.service` runs.
- New: pre-flight contract-eligibility check in the executor for futures intents; mock the IBKR physical-delivery window and assert abstain.
- Existing must continue to pass: `chad/tests/test_*futures*.py`, `chad/tests/test_pr_m2k_mym*.py`, `chad/tests/test_*bar_provider*.py`, plus full baseline.

# Required runtime verification
After remediation lands:
- `chad-ibkr-bar-provider` ERROR/Exception count for SI conId 788288505 drops to 0 within 24 h of the PR landing.
- `chad-live-loop` Error 201 physical-delivery rejections drop to 0 within 24 h on a normal trading day.
- `runtime/futures_roll_state.json` reflects the current month for every routed and polled futures symbol.
- No new "near-expiration" rejections in `data/broker_events/BROKER_EVENTS_IBKR_*.ndjson` after the executor pre-flight gate lands.

# Operator approvals required
- **Approval 1**: skip-expired logic in bar provider (low risk — additive).
- **Approval 2**: completeness audit of `chad-futures-roll-refresh.service` and which symbols to add.
- **Approval 3**: executor pre-flight contract-eligibility check (medium risk — could mask a legitimate intent if mis-configured).

# Definition of done
1. Bar-provider expiry-gate merged; SI noise drops to 0 in 24 h.
2. `chad-futures-roll-refresh.service` covers all routed and polled futures symbols.
3. Executor pre-flight contract-eligibility check merged; abstain log present when applicable.
4. Tests merged and passing.
5. `runtime/ibkr_bars_cache.json` carries `schema_version`.
6. Updated SSOT or forward-erratum doc reflects the futures-roll governance.
7. CLAUDE.md Pre-Live Operator Tasks updated (if any item relates to futures-roll readiness — currently none explicit).

# No-live confirmation
This Pending Action does not authorize live trading.
ready_for_live must remain false.
allow_ibkr_live must remain false.
allow_ibkr_paper must remain true.
No broker orders may be placed or cancelled under this Pending Action.

# Session 1 impact
- **Does this item block Session 1 evaluation (window opens 2026-05-28T00:00:00Z)?** NO.
- The 588 errors/24h are noise, not data corruption. The Error 201 rejections are soft fails and do not register as paper_fills.
- The remediation cannot land before Session 1 opens (~9 h from this Pending Action creation); it is an Effort-M workstream.
- **Mitigation for Session 1**: monitor `chad-ibkr-bar-provider` and `chad-live-loop` journals during the Session 1 window. If new Error 201 rejections occur on symbols other than SI/MCL (e.g. a different physical-delivery futures), escalate immediately. Session 1's pass criteria do not include "zero journal errors" — they include "no new untruth" and "no live enablement", both of which are unaffected by today's expired-contract noise.
- **Latest acceptable closure date**: before the next futures contract roll cycle (front-month roll is monthly for many futures; SI specifically rolled today; the next likely high-noise roll is in ~30 days). Targeting closure within the Epoch 3 5-session window is reasonable.
