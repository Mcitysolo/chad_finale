# R3 — Silent Unit State and Market-Data Refresh Failure (chad-options-chain-refresh)

# Status: PROPOSED

# Source audit
- `reports/forensic_audits/CHAD_FORENSIC_FULL_SYSTEM_AUDIT_20260527T142951Z.md`
  - §5.3 (Tier 4 failed-unit detail)
  - §14.3 (Tier 13 journal sample)
  - §17 R3-A (root-cause grouping)
  - §19 closeout item 3 (OPTIONS-CHAIN-1)

# Severity
**HIGH** (Effort S) per audit §18.

# Problem statement
`chad-options-chain-refresh.service` failed with exit code 1 at 2026-05-27T12:33:07Z and is in `failed (Result: exit-code)` state. The failure is **honestly recorded** on disk (PR-04 hardening: empty cache with `error` field + `runtime/options_chain_refresh_failure.json` failure artifact), but the systemd unit state remains `failed` until the next scheduled timer fire — and **no `OnFailure=` directive routes a Telegram or operator-facing alert**, so the only signal that the publisher is broken is an operator manually running `systemctl --failed` or noticing `options_chains_cache.json` is empty.

This is "loud but not routed": the data trail is correct, but no human is paged.

# Evidence
1. **Failure detail (`systemctl status chad-options-chain-refresh.service`):**
   ```
   × chad-options-chain-refresh.service - CHAD Pre-Market Options Chain Cache Refresh
        Loaded: loaded (/etc/systemd/system/chad-options-chain-refresh.service; disabled; preset: enabled)
   TriggeredBy: ● chad-options-chain-refresh.timer
        Active: failed (Result: exit-code) since Wed 2026-05-27 12:33:07 UTC; 2h+ ago
       Process: 1692763 ExecStart=… python3 -m chad.market_data.options_chain_refresh (code=exited, status=1/FAILURE)
   ```
2. **Journal root-cause sequence (12:32:30Z → 12:33:07Z):**
   - `Error 200, reqId 6: No security definition has been found for the request, contract: Stock(symbol='SPY', exchange='SMART', currency='USD')`
   - `reqMktData_failed symbol=SPY err=Contract Stock(symbol='SPY', …) can't be hashed because no 'conId' value exists. Qualify contract to populate 'conId'.`
   - `Error 2103: Market data farm connection is broken: usfarm`
   - `Error 2105: HMDS data farm connection is broken: ushmds`
   - `Error 2108: Market data farm connection is inactive but should be available upon demand: usfuture`
   - `Error 2157: Sec-def data farm connection is broken: secdefil`
   - `Error 10168: Requested market data is not subscribed. Delayed market data is not enabled.`
   - After 3 attempts × 30s timeout: `SPY: FAILED after attempts=3 last_error='timeout_after_30.0s'`
3. **PR-04 graceful degradation artifacts (still present at audit time):**
   - `runtime/options_chains_cache.json` — present, `ts_utc=2026-05-27T12:33:07Z`, `chains_count=0`, `error` field populated.
   - `runtime/options_chain_refresh_failure.json` — present, `blocked_reason=ibkr_contract_details_unresponsive`, `last_successful_ts=2026-05-26T12:30:31Z`.
4. **24 h ERROR count for the unit** (per audit Tier 13): 24 errors, all from this single failed run.
5. **No Telegram alert was emitted** — `chad-telegram-bot` journal shows 0 errors in last 24 h for this event; no `OnFailure=` directive routes a notification.
6. **Same-cause family observed in `chad-live-loop`:** `Error 1100: Connectivity between IBKR and Trader Workstation has been lost` (broker-side outages correlate with the 12:34Z and 13:07Z stop_bus flutter events).

# Affected files / services / artifacts
- `chad-options-chain-refresh.service` (FAILED; `disabled; preset: enabled` — triggered by timer only)
- `chad-options-chain-refresh.timer`
- `/etc/systemd/system/chad-options-chain-refresh.service`
- `chad/market_data/options_chain_refresh.py`
- `runtime/options_chains_cache.json` (`options_chain_cache.v2`, currently empty + error field)
- `runtime/options_chain_refresh_failure.json` (failure artifact, PR-04)
- `chad/ops/health_monitor.py` (consumer of failure artifact; behaviour during failure UNKNOWN — REQUIRES AUDIT)
- `chad-telegram-bot.service` (current OnFailure= alert sink; bypassed today)
- Strategies that depend on options chains: `alpha_options`, `omega_momentum_options` (currently regime-gated off in `trending_bull`, see `chad/strategies/__init__.py` and runtime evidence — REGIME_GATE_DROP for `gamma/AAPL/AVGO/GLD`, etc.)
- Telegram dedupe artifact: `runtime/telegram_dedupe_health_R17_Optionschainrefreshfailed.json` (exists, 43 bytes, mtime ~14:27Z) — indicates the health-monitor rule R17 fires but the alert may be deduped away on subsequent runs.

# Root-cause hypothesis
The IBKR market-data farms (`usfarm`, `ushmds`, `secdefil`, `usfuture`) experienced a coincident outage between 12:32Z and 12:33Z. Without farm connectivity, `reqContractDetails` cannot qualify the SPY contract (cannot populate `conId`), and without `conId` the option chain cannot be requested. The 3-attempt retry × 30 s timeout exhausts at 12:33:07Z and the publisher exits with status=1. The IBKR data farms recovered later in the day (other services like the bar provider and live-loop continued to operate, albeit with the broker_latency flutter at 12:34Z and 13:07Z), but **the timer has not re-fired** because the unit is "pre-market" (one-shot per day at a specific scheduled time, NOT a periodic refresh).

This is a **scheduling design** issue compounded by the alert-routing gap.

# Why this matters
1. **Detection**: Without `OnFailure=` routing, a critical pre-market publisher can stay failed until the operator notices manually. On a live day this would be unacceptable.
2. **Strategy gating**: The strategies that consume the chains (`alpha_options`, `omega_momentum_options`) are currently regime-gated off, so the operational impact today is minimal — but a regime flip to chop/range during the window with an empty chains cache would silently skip every option intent rather than emitting a loud "no chain available" abstain log.
3. **Repeat probability**: IBKR data-farm flutter is recurring (see also IBKR_RELIABILITY Pending Action). A failure-routing fix is leverage against the next recurrence.
4. **R17 health-monitor rule fires** (per `runtime/telegram_dedupe_health_R17_Optionschainrefreshfailed.json` presence) but the dedupe layer may suppress repeated alerts on the same artifact — operator visibility is **UNKNOWN — REQUIRES AUDIT** of dedupe TTL and reset conditions.

# Current safety posture
- All live gates held: `ready_for_live=false`, `allow_ibkr_live=false`, `allow_ibkr_paper=true`.
- Options-routed strategies (`alpha_options`, `omega_momentum_options`) are regime-gated off in `trending_bull`; no option intents are being emitted into an empty chains cache.
- Defense-in-depth (PR-04) means the chains cache is **explicitly empty with an error field**, not silently stale; consumers that respect the contract abstain rather than route into a corrupt chain.
- The failure artifact is on disk and time-stamped.

# Scope for future remediation
1. **Add `OnFailure=` directive** to `chad-options-chain-refresh.service` that triggers a Telegram alert via the existing alerting stack (most likely a service `chad-options-chain-refresh-alert.service` invoked by `OnFailure=`).
2. **Re-evaluate scheduling**: either
   a. Keep one-shot pre-market and add a single retry timer N minutes later if `runtime/options_chain_refresh_failure.json` is present and `chains_count=0`; OR
   b. Convert to a periodic refresh (every hour during US market session) with idempotent writes and short-TTL re-fetch on stale entries.
3. **Audit Telegram-dedupe TTL** for `R17_Optionschainrefreshfailed` to ensure repeated alerts are not suppressed indefinitely; require a daily reset.
4. **Add a regression test** that simulates an IBKR farm outage (mocked) and asserts the publisher writes the failure artifact AND emits the alert path.
5. **Consider widening the alert scope**: same `OnFailure=` pattern for other one-shot pre-market timers (futures roll, daily bars refresh, etc.).

# Explicitly out of scope
- Live-mode enablement.
- Restarting the failed service (operator action, separate Pending Action).
- Modifying `chad/market_data/options_chain_refresh.py` retry logic in this Pending Action.
- Modifying the IBKR client / farm reconnection logic (handled by IBKR_RELIABILITY Pending Action).
- Modifying any other systemd unit file.

# Required tests
- New: mock IBKR farm-broken scenario; assert publisher exits with status=1, writes both `runtime/options_chains_cache.json` (with `error` field) and `runtime/options_chain_refresh_failure.json`, and that the alert path is invoked.
- New: test that `OnFailure=` directive exists on the service unit (a `pytest` that reads `/etc/systemd/system/chad-options-chain-refresh.service` and asserts the directive — operator action to add the directive itself).
- Existing must continue to pass: `chad/tests/test_options_chain_refresh.py`, `chad/tests/test_health_monitor_*`, plus the full baseline.

# Required runtime verification
After remediation lands:
- On the next failed run (real or operator-induced via test), an alert appears in Telegram within 1 cycle.
- `runtime/options_chains_cache.json` continues to carry the `error` field on failure.
- `runtime/options_chain_refresh_failure.json` continues to be written on failure.
- After IBKR farm recovery, the next scheduled or retry fire repopulates `chains_count > 0` and clears `error`.
- `R17_Optionschainrefreshfailed` health rule clears on recovery.

# Operator approvals required
- **Approval 1**: addition of `OnFailure=` directive to the systemd unit (counts as systemd file modification — per `CLAUDE.md` §6 "Never modify systemd service files without explicit instruction").
- **Approval 2**: scheduling change (pre-market one-shot vs periodic).
- **Approval 3**: Telegram alert content/format.

# Definition of done
1. `OnFailure=` directive deployed to `chad-options-chain-refresh.service` (and any sibling pre-market timers identified as needing the same).
2. Telegram alert path verified end-to-end via a controlled fault-injection test.
3. Telegram-dedupe TTL audit completed and documented.
4. New regression test merged and passing.
5. Operator runbook updated to include "check `runtime/options_chain_refresh_failure.json` if R17 alert fires."
6. SSOT or forward-erratum doc updated.

# No-live confirmation
This Pending Action does not authorize live trading.
ready_for_live must remain false.
allow_ibkr_live must remain false.
allow_ibkr_paper must remain true.
No broker orders may be placed or cancelled under this Pending Action.

# Session 1 impact
- **Does this item block Session 1 evaluation (window opens 2026-05-28T00:00:00Z)?** NO.
- The options-chain failure is from 12:33Z on Day-0 (already DIRTY by other criteria); the failed unit state is on Day-0, not Session 1.
- Options-routed strategies are regime-gated off in `trending_bull`, so an empty chains cache does not produce silent bad fills during Session 1 unless the regime flips and the cache is still empty.
- **Mitigation if the unit is still in `failed` state when Session 1 opens**: the timer's next scheduled fire-time should re-attempt; if it succeeds, the unit state self-clears. If the next fire still fails, the operator should investigate (out of scope of this Pending Action, but the alert routing would have caught it).
- **Latest acceptable closure date**: before the next regime transition that would route options strategies (operator-domain estimate), OR before the next planned live-readiness window — whichever is earlier.
