# CTF-T3 Forensic — edge_decay halt on `alpha_crypto`

**Date:** 2026-07-14 · **Mode:** read-only (no code/state changed) · **Author:** CHAD/Claude

## Question
Why is edge_decay halting `alpha_crypto`, given all `alpha_crypto` fills are
`pnl_untrusted` / `validate_only` from July 2–4? Is the halt correct-but-stale,
or is the monitor scoring data it should exclude?

## Evidence (verified 2026-07-14)
1. **Halt record** — `runtime/strategy_allocations.json`:
   ```json
   "alpha_crypto": {"halted": true, "halt_reason": "consecutive_negative_10",
                    "halted_at": "2026-07-02T02:57:40.335191+00:00",
                    "consecutive_negative": 10, "cleared_at": null}
   ```
2. **All alpha_crypto trade data is non-scoreable.** 2,148 `alpha_crypto`
   records across `data/trades/trade_history_2026070{2,3,4}.ndjson`
   (660 / 1,082 / 406). Every single one:
   - `pnl == 0.0` (never `< 0`),
   - `side == BUY` only (no realized closes → no realized P&L),
   - tagged `pnl_untrusted` + `validate_only`,
   - `extra.pnl_untrusted == true`, `extra.pnl_untrusted_reason ==
     "kraken_paper_validate_only_no_realized_fill"`.
   No alpha_crypto activity after 2026-07-04 (the CRYPTO-TRUST trusted-fills
   build is committed but not deployed).
3. **The monitor already excludes 100% of them.** `_iter_trade_payloads`
   (chad/risk/edge_decay_monitor.py) drops any record whose `tags` contains
   `pnl_untrusted`. Simulated on live data, `collect_recent_trades_by_strategy()`
   returns a trusted ledger of `{gamma, manual}` only — **`alpha_crypto` is
   absent** (0 trusted trades).
4. **The exclusion predates the halt.** The tags-based `pnl_untrusted` filter
   was added 2026-04-22 (commit `1cf9f44`) — 71 days before the 2026-07-02 halt.
   So even at halt time the filter was active, and every 07-02 record is zero-pnl
   + tagged untrusted. **`consecutive_negative_10` cannot be reproduced from the
   present ledger** (0.0 is not `< 0`, and all rows are excluded anyway).
5. **The halt is sticky and un-recomputable.** `EdgeDecayMonitor.check_strategy`
   only ever calls `set_strategy_halted` (adds a halt); it never calls
   `clear_strategy_halt`. `check_all` iterates only strategies *present in the
   trusted ledger* — since `alpha_crypto` has 0 trusted trades it is never
   re-evaluated. Only manual `scripts/clear_edge_decay.py` removes the flag.
6. **The halt has live effect.** `chad/core/live_loop.py` (~L1433–1438) refreshes
   its halted set from the persisted `strategy_allocations.json`, so the stale
   flag actively gates `alpha_crypto` activation.

## Finding
The halt is **stale / orphaned**, not a current mis-scoring.
- The monitor is **correctly excluding** the validate_only data *now* — it is
  NOT scoring data it should exclude.
- But the halt persists on evidence that is **absent from the current ledger**.
  Because the untrusted-exclusion filter has been active since 2026-04-22 and
  every alpha_crypto record is zero-pnl + tagged untrusted, the recorded
  `consecutive_negative_10` is **unreproducible** from present data. Whatever set
  it (a transient pre-tag/pre-zero state, an epoch/migration re-stamp, or a
  non-monitor writer) left an artifact the monitor has no path to clear.
- Net: `alpha_crypto` is a strategy that has produced **zero scoreable trades
  ever** (all fills are validate_only placeholders carrying no realizable-loss
  signal), yet is held down by a decay halt no current evidence supports.

### Secondary observation (latent, low-severity)
`_iter_trade_payloads` checks top-level `payload.get("pnl_untrusted")` (L181),
but in these records the flag lives **only** in `payload.extra.pnl_untrusted`
and in `tags`. The sole thing excluding them today is the `tags` check (L185–189).
A future validate_only record that carried `extra.pnl_untrusted=true` but lacked
the `"pnl_untrusted"` tag would leak into scoring — a single point of failure.
(Impact is bounded: validate_only pnl is 0.0, which cannot create a negative
streak, so a leak could not itself halt a strategy.)

## Proposed disposition (no action taken — read-only task)
1. **Clear the stale halt** via `scripts/clear_edge_decay.py alpha_crypto`
   (operator Pending Action). Rationale: no current trusted evidence supports a
   decay halt; the monitor cannot auto-clear it.
2. **Sequence the clear with the CRYPTO-TRUST deploy.** The crypto lane currently
   emits only validate_only fills (trusted-fills build committed 7e9d982/5f7aa3a,
   NOT deployed). Clearing now merely lets alpha_crypto resume emitting more
   validate_only placeholders — cosmetic, not real trading. Clear the halt at the
   same time trusted crypto fills go live, so the strategy restarts clean and
   immediately scoreable.
3. **Optional hardening PA (separate, default-safe — do not bundle):** in
   `_iter_trade_payloads`, also exclude `payload.extra.pnl_untrusted is True` and
   `payload.validate_only is True`, closing the single-point-of-failure at L181.
   Low urgency; makes the untrusted exclusion robust to a missing tag.

**Governance note:** items 1–3 are Pending Actions. Per CHAD rule 3 (no direct
config/state mutation) none were executed; this document is the finding of record.
