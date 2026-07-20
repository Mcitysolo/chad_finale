# PFF1 — UNH churn case (for the context builder)

**Date:** 2026-07-20 · **Symbol:** UNH · **Status:** report-only (no fix here; the
double-book fix is PFF1-Q1, this file is the Wave-2 churn case). Runtime was
read-only for this analysis.

## What happened (the loop)

The position-aware equity **exit overlay went ACTIVE** today (env override
`CHAD_POSITION_EXIT_OVERLAY=active` in
`chad-live-loop.service.d/97-exit-overlay-active.conf`; heartbeat
`runtime/exit_overlay_heartbeat.json` → `mode: active`). At **13:50:55Z** the
overlay's **ATR trailing stop** fired for UNH — the sole non-`no_condition_met`
record in `data/exit_overlay/exit_overlay_20260720.ndjson` today
(`reason: atr_trailing_stop`, `atr_stop = 423.70`; the 13:50:00 quote marked UNH
at `423.0`, i.e. below the trail).

The overlay issued a **reduce-only SELL of the full 273-share gamma long** (the
Epoch-3 seed lot adopted 2026-07-15). Within seconds-to-minutes **gamma re-bought
228 shares** (5 @ 13:51, 223 @ 13:55) because gamma's own signal was still long.
Net path: **273 long → 0 → 228 long**. The overlay closed the position; the
strategy immediately re-opened it. That is the churn: an exit that the strategy
undoes on the next tick.

After the re-buy the trail re-anchored ~2.5 ATR below the new peak
(`atr_stop` dropped to ~386 by 13:54Z), so *this* lot will not re-fire until UNH
falls ~8–9%. The loop is structural, not one-off: any time the overlay's trigger
is met on a position a strategy still wants, the strategy buys it back.

## Churn cost today (the 228-share round trip)

Two views — the broker-truth fills (harvester, account DUR119533) are the real
executions; the executor SIM aggregate mis-marked the sell:

| leg | qty | avg price (broker truth) | avg price (executor SIM) |
|-----|-----|--------------------------|--------------------------|
| SELL (overlay close) | 273 | 424.88 | **423.00** (stale ref, quote_age 55s, `ref_only_no_nbbo`) |
| BUY (gamma re-open)  | 228 | 424.97 | 425.50 |

- **Round-tripped quantity:** 228 shares (~$97k notional).
- **Price churn (broker truth):** (424.97 − 424.88) × 228 = **+$21.91** adverse.
- **Modeled fees (IBKR fixed, $0.005/sh, $1 min):** sell 273 = $1.365, buy 5 =
  $1.00, buy 223 = $1.115 → **≈ $3.48**.
- **Real churn cost ≈ $25** (≈ $22 price + $3.48 fees) for the round trip.
- **Executor-SIM view says ≈ $570** only because the SIM close printed at a stale
  `423.00` ref (55s old, no NBBO) while the broker filled at `424.88` — a
  **$1.88/sh ($513) SIM-vs-broker mispricing** that is its own evidence-quality
  defect, not realized churn.

**The dollar slippage is small. The real damage is the booking.** Because the
same order is written from two fill sources (executor SIM aggregate + harvester
broker slices) and both fed FIFO (see PFF1-Q1), this one 273-share sell booked
**~506 shares of closes**: 1 untrusted seed close (+625.17) plus **6 fabricated
"trusted" closes (−145.79 over 233 shares)** that never economically happened —
each churn loop mints ~6 fake effective trades, poisons win_rate / sharpe_like,
and leaves a **guard split-brain** (gamma UNH = 0 vs broker = 228,
`broker_untracked_position`). Until PFF1-Q1 deploys, *every* churn loop
manufactures scoring pollution far larger than its $25 slippage.

## How often does the loop re-fire? (at current ATR)

Inputs (from the overlay + `data/bars/1d/UNH.json`):

- Wilder **ATR₁₄ ≈ 14.30** (overlay value; ≈ 3.4% of the 426.09 close).
- **Trail distance = `atr_trail_mult` × ATR = 2.5 × 14.30 = 35.75** (≈ **8.4%**).
- Hard stop = 8% of the entry anchor (`hard_stop_price = 388.32`).
- Max-hold = 20 calendar days (equity) — negligible on a weekly horizon.
- UNH is volatile now: last-5 daily ranges 6.03 / 10.37 / 9.74 / **41.25** / 12.37.

The trail fires when price gives back **2.5 ATR from its favourable extreme**.
Modelling daily moves as ~1 ATR with no drift, the expected time to a 2.5-ATR
peak-to-trough drawdown is on the order of **2.5² ≈ 6 trading days** (√t /
reflection heuristic). After each fire the strategy re-buys and the trail
re-anchors 2.5 ATR away, so a *fresh* 2.5-ATR excursion is needed each time.

**Estimate: ≈ 0.8–1 churn loop per week per contested symbol** at UNH's current
ATR — spiking on high-range days (a single 41-pt / ~9.7% day like this week can
trip the trail same-session). The 8% hard stop (~2.7 ATR) is the same order of
magnitude and does not materially raise the rate. This is **per symbol**: the
active overlay contests every equity/ETF position across ~16 strategies, so the
**book-level rate is a small multiple** of the per-symbol figure — call it a
handful of loops/week while the overlay and the strategies disagree.

At ~$25 realized cost/loop that is only ~$20–100/week of slippage; the binding
cost is the **scoring corruption per loop** (≈6 fake effective trades + guard
drift), which is why this belongs in front of the context builder as a
**structural overlay-vs-strategy conflict**, not a slippage line item.

## The real question for Wave 2

The overlay closes what the strategies want to keep. Options to stop the *churn*
(distinct from the *double-book*, already fixed in PFF1-Q1):

1. **Suppress re-entry into a symbol the overlay just force-closed** (a cooldown
   / "overlay owns this symbol until it re-arms" lock), so gamma cannot instantly
   undo the exit.
2. **Give the overlay authority to keep the position flat** rather than reduce-
   only-then-let-the-strategy-rebuy.
3. **Reconcile the overlay's exit thesis with the strategy's entry thesis** so
   they do not fight every trail trigger.

Not decided here — Wave-2 owns the disposition. This file is the evidence packet.
