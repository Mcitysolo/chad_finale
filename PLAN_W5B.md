# PLAN_W5B — Wave 5 Lane B: The Portfolio Allocator (R1, SHADOW)

**Phase 1 — PLAN ONLY.** No code in this commit. Worktree `chad_w5b`, branch
`goal/wave5-allocator`, base `main@e86eaaf` (the W4A fuse box merge; same base
as Lane A `chad_w5a`; verified). Live tree untouched. Phase 2 begins only after
operator decisions D1–D8.

Method: three parallel read-only audits over the worktree (exposure-vector
computability, limits sourcing, book + evidence conventions) before any design.
Several of the task's own framing hints were half-right — §1 records the verified
ground truth this plan is built on. In particular: the illustrative
"long MNQ + long NVDA + long semis" basket is not what the book holds *today*;
the real ~11-symbol book is a subtler version of the same disease (§1-P2), and
the plan is grounded in that real book, not the hypothetical.

**The named gap this closes.** Nothing in CHAD computes *marginal portfolio
risk*. Per-order caps exist and enforce (`policy.py` `max_total_exposure`
$500k / `max_symbol_exposure` $150k; `sizing_config` 5% position); a
*margin*-based aggregate gross exists in shadow (`margin_block` 2.0×NetLiq). But
no layer normalizes **every intent + every open position into one exposure
vector and sums the correlated tickets**. Concretely, the live book is long SPY
+ AAPL + MSFT + V + MA + UNH + BAC + LLY (eight high-beta long-S&P tickets)
partially offset by PSQ (short-Nasdaq) and SVXY (short-VIX) — one levered
long-beta bet spread across ten tickets, with two inverse ETFs nobody nets
against it. R1 observes that. In SHADOW it **blocks nothing** — it logs what it
*would* approve / resize / reject and why.

---

## 0. Scope and non-negotiables

- **One item, SHADOW-only.** Stage 2 explicitly wants R1 in shadow: observe
  every intent, log the would-verdict, block nothing. The flag is tri-state
  `off | shadow` (NO `enforce`) — there is **no `should_block`-True path in the
  code at all**. R1-enforce is a named future wave, out of W5B scope. Shadow is
  byte-identical to today (evidence + state only).
- **Prime invariant: the allocator never evaluates a close.** It must not see
  overlay closes, crypto-overlay closes, reconciler closes, strategy
  exits/flips/protectives, or **flatten** intents. Enforced structurally (§5) —
  not by convention. Extra salient after INC-0723 (a flatten drill already
  caused a real incident); the allocator must be provably blind to flatten.
- One change at a time; set-diff test methodology (§9); commits prefixed `W5B`.
- No runtime mutation, no systemd changes, no config mutation outside the
  worktree.
- **Lane-A territory is off-limits.** No edits to `chad/analytics/
  implementation_shortfall.py`, `chad/analytics/excursion_recorder.py`,
  `chad/validation/*` (frozen), the `to_payload` TCA stamp, or the overlay
  `_save_anchors` hook. W5B and W5A both add to `config/exterminator.json`
  (W5A: `clock_health`/EXS10; W5B: a `feeds{}` row + an EXS7 pin) — **disjoint
  keys**, merge-order coordination flagged (D8).
- **The allocator reads; it never re-prices.** It consumes `positions_truth.
  json`, `price_cache.json`, `pnl_state.json`, `data/bars/1d/*`,
  `config/symbol_sectors.json` — all read-only. It computes no P&L, mutates no
  book, changes no `intent.quantity`.

## 1. Audited ground truth (premise verification)

| # | Premise / claim | Verdict | Evidence |
|---|---|---|---|
| P1 | "normalize every intent + every open position to a vector" — the intent already carries the fields | **HALF-TRUE.** The intent (`StrategyTradeIntent`) carries `symbol/sec_type/exchange/currency/side/quantity/notional_estimate/limit_price/strategy/expected_price/meta` — but **NO `asset_class`, `sector`, `multiplier`, or `venue` field**. asset_class lives one layer up on `PlannedOrder` then collapses to `sec_type`; venue is the `(sec_type,exchange,currency)` triple; sector/mult are not on the intent at all — all must be *derived* by the allocator | intent `chad/execution/ibkr_executor.py:25-80`; PlannedOrder.asset_class `chad/execution/execution_pipeline.py:465`; chokepoint loop `chad/core/live_loop.py:2704` |
| P2 | The "one levered bet across many tickets" is a hypothetical (MNQ/NVDA/semis) | **The real book is worse-hidden.** 11 symbols open, ALL equity/ETF, all `gamma` (+ `broker_sync`/`epoch3_adopted` reconciliation mirrors): AAPL, BAC, LLY, MA, MSFT, UNH, V (megacap long) + SPY (long index) + PSQ (short-Nasdaq) + SVXY (short-VIX). Eight long-beta tickets + two inverse ETFs = a concentrated long-S&P bet no layer aggregates. No futures/options/crypto open today | `runtime/position_guard.json`; `runtime/positions_truth.json` (11× `secType:STK`); registry `chad/strategy_registry.py:104-144` |
| P3 | delta-dollars is computable | **YES for equity/ETF; PARTIAL for futures.** delta$ = signed(qty)·price·mult. Price = `price_cache.json` (35 symbols incl futures roots, TTL 300s, bar-close fallback). Equity mult=1 (implicit). Futures point-values exist but split across **3 unreconciled tables** covering **9 micro roots** (MES 5, MNQ 2, MCL 100, MGC 10, MYM 0.5, M2K 5, ZN 1000, ZB 1000, M6E 12500) — **no full-size ES/NQ/CL/GC**, no M6A/M6B | price `chad/market_data/price_cache_refresh.py:28`; specs `chad/strategies/alpha_futures.py:94-103` + `chad/strategies/omega_macro.py:76-101` + dup `chad/risk/futures_position_sizer.py:10-15`; math `alpha_futures.py:305` |
| P4 | beta-weighted equity is computable | **NO — fully DEFERRED (R3).** There is **no statistical beta-to-SPY anywhere**. Every "beta" hit is a strategy *name*; the only numeric beta is an unused `FMPCompanyProfile.beta` fetch field, never stored, never joined | `chad/strategy_registry.py:107-108`; unused field `chad/market_data/fmp_client.py:67,319` |
| P5 | options Greeks are usable | **NO — DEFERRED (R3).** Options strategies are active but **real contracts are not built** — OPTIONS proxy-route through the underlying (`options_chain_routing_not_wired`); zero OPT in the book. B5 = the synthetic Black-Scholes greeks feed (`options_greeks.v1`, `provider_status:approximated`, `ttl_seconds:90000` ⇒ daily-stale, `gamma`/`theta` **null**). Options exposure today = the equity-proxy delta$, already captured | proxy `chad/execution/execution_pipeline.py:702-724`; B5 `docs/CHAD_GAPS_TO_CLOSE.md:28`, `runtime/options_greeks.json` |
| P6 | sector map must be built | **NO — already exists (W4A).** `config/symbol_sectors.json` (`symbol_sectors.v1`, 13 buckets over the universe) shipped for the LC3 sector fuse. Reuse it; invert sector→[symbols] to symbol→sector; unmapped ⇒ `unmapped` (count-only, never binds — the W4A LC3 idiom) | `config/symbol_sectors.json`; W4A LC3 `PLAN_W4A.md §4` |
| P7 | asset_class + venue are computable | **YES (derived), with two caveats.** `AssetClass{equity,etf,crypto,forex,cash,futures,options}`; intent side maps class→sec_type; venue split crypto→Kraken else→IBKR. Caveat A: the position classifier `position_exit_overlay._asset_class` has **no crypto/options branch and omits MYM from `_FUTURES_SYMBOLS`** (MYM misclassifies as equity). Caveat B: only two venues wired | enum `chad/types.py:69-76`; split `chad/execution/execution_pipeline.py:1392-1412`; classifier bug `chad/risk/position_exit_overlay.py:144,354-359` |
| P8 | open positions carry a price/mark | **NO inline mark.** `position_guard.json` = qty/side only. `positions_truth.json` carries `avgCost` + `position` + `secType` (schema `positions_truth.v1`, ttl 90, GREEN) — the only entry-cost source. A live *mark* requires a `symbol→price_cache.json` join (price_cache is ~55s-stale worst case, per the flag-gated `portfolio_marks` note) | `runtime/positions_truth.json`; `chad/risk/portfolio_marks.py:1-33` |
| P9 | placement — the stage-3 per-intent chokepoint | VERIFIED (the fuse box proved it): stage-3 gate cluster `live_loop.py:2665+`, per-intent loop at `:2704`, symbol-blocker / fuse-gate hook ~`:2759`. **Overlay/reconciler/flatten closes never traverse stage-3** (`apply_close_intents → adapter` direct) — an entry-scoped observer is closes-blind *by construction* | `PLAN_W4A.md §2, P8/P9`; loop `chad/core/live_loop.py:2704`; fuse-gate wiring `chad/risk/fuse_box.py` |
| P10 | shadow-gate + evidence conventions to copy | VERIFIED, exact template = the margin shadow gate: `should_block()→False` in shadow, fail-open `ERROR` verdict, `margin_shadow.v1` dated ndjson under `data/margin_shadow/`, `ShadowVerdict.to_dict` row shape, pytest `evidence_dir`-required leak guard | `chad/execution/margin_shadow_gate.py:83-155,392-401,544-554` |
| P11 | heartbeat + sentinel + coach conventions | VERIFIED: heartbeat doctrine (`fuse_box.publish_state` every cycle incl OFF, `<stem>.vN`, ts_utc+ttl_seconds, TTL 180); sentinel `feeds{}` row shape (`{path,format,ts_field,ttl_seconds,ttl_verified,ttl_source,warn_after_seconds,fail_after_seconds}`, `ttl_verified:false ⇒ operator_verify`); coach `_tpl_edge_decay` streak template + value-free `dedupe_identity(event,id)` + `runtime/dedupe/` (fails open, digits stripped — CTF-T2) | `chad/risk/fuse_box.py:75,957-984,1119-1132`; `config/exterminator.json` feeds; `chad/utils/coach_voice.py:552-572`; `chad/utils/telegram_notify.py:169-196` |
| P12 | correlation data is too sparse/stale | **FALSE — cheap and fresh.** `data/bars/1d/<SYM>.json`: all 11 book symbols carry ~251 daily bars (a full year, `2025-07-23`→`2026-07-22`), refreshed nightly through prior close. `portfolio_var.py` already loads these, computes per-symbol daily log-returns + population σ, and stops one step short of a covariance matrix (`independent_asset_assumption`). An 11×11 Pearson matrix is negligible compute | bars `data/bars/1d/`, producer `chad/market_data/nightly_bars_refresh.py`; `chad/risk/portfolio_var.py:50,58-59,262-381` |
| P13 | equity basis for a gross cap | **CAD-only.** The single `currency_ok=true` equity is CAD ≈ **$990,919** (`pnl_state.json::account_equity`) / $990,666 (`portfolio_snapshot::ibkr_equity`). The USD authoritative equity is **null** (no FX rate captured). Live SCR is **WARMUP** (sizing_factor 0.10, 71 effective trades) — not the CONFIDENT snapshot in CLAUDE.md | `runtime/pnl_state.json`; `runtime/portfolio_snapshot.json:3,6`; `runtime/scr_state.json` |
| P14 | existing caps to cite (no invented numbers) | VERIFIED: `policy.py:665` gross **$500k** + `:666` per-symbol **$150k** (BOTH enforced today); `margin_block.json:11` gross **2.0×NetLiq** (shadow); `sizing_config.json:16` **5%** position, `:14` sector **$5,000** (small-acct calibration); Kraken wallet **$184.58 USD** (`kraken_balances.json`) vs $690.51 dynamic cap. **No net-exposure notional cap and no IBKR-specific venue cap exist anywhere** | `chad/policy.py:665-666`; `config/margin_block.json:11`; `config/sizing_config.json:14,16`; `runtime/kraken_balances.json`; `runtime/dynamic_caps.json:262` |

## 2. Architecture — one observer, one exposure spine

New module family under `chad/risk/` (nothing new on a timer; the observer
attaches to the existing per-intent chokepoint):

```
chad/risk/portfolio_allocator.py        # core: ExposureVector, book+intent normalization,
                                         #       provisional-book accumulation, limit evaluation, Verdict
chad/risk/allocator_shadow_gate.py       # per-intent observer (margin-shadow-gate pattern); would-only, never blocks
chad/risk/allocator_correlation.py       # read-only rolling-correlation DIAGNOSTIC over data/bars/1d (advisory)
config/portfolio_limits.json             # firm-level caps, every number cited (§4)
runtime/portfolio_allocator_state.json   # allocator_state.v1 — heartbeat every cycle incl OFF (§6)
data/allocator_shadow/allocator_shadow_YYYYMMDD.ndjson  # allocator_shadow.v1 per-intent verdict rows
```

**Reuse, don't rebuild:** `config/symbol_sectors.json` (W4A, P6),
`data/bars/1d/*` + the return/σ approach from `portfolio_var.py` (P12, but the
allocator gets its *own* helper — it must NOT mutate portfolio_var's
hash-pinned `var_state.v1` contract), `price_cache.json` (marks),
`positions_truth.json` (book + avgCost), the futures point-value tables (read,
not rewritten — D2).

**Observer placement (P9):** constructed once per cycle in `live_loop.run_once`
with an open-book snapshot (`positions_truth.json`); called **per intent inside
the existing loop** at `live_loop.py:2704`, beside the fuse gate (~`:2759`),
sharing its bypass predicate. Each call adds the intent's exposure to a running
**provisional book** (= open book + intents already seen this cycle) and returns
the would-verdict for that intent's *marginal* contribution. Because shadow
blocks nothing, every intent is added to the provisional book regardless of
verdict — so the corpus honestly shows "the 3rd correlated ticket *would* have
breached gross," exactly as enforcement would have seen it. Failure-soft: an
evaluator exception returns a fail-open `ERROR` verdict and never kills the cycle
(margin-gate precedent).

**Modes** (tri-state, garbage → off; **no enforce path exists**):

| Flag | Values | Default | Gates |
|---|---|---|---|
| `CHAD_ALLOCATOR` | off \| shadow | off | the whole observer (evaluation + evidence + state) |

Shadow = full evaluation, `allocator_shadow.v1` evidence + `allocator_state.v1`
heartbeat, zero behavior change (no `should_block`-True path built at all). OFF ⇒
the heartbeat still writes (all-off must be distinguishable from dead) but no
per-intent rows and no book snapshot beyond the heartbeat summary.

## 3. The exposure vector (what is honestly computable TODAY)

For every intent (marginal) and every open position (base), the allocator emits:

```
ExposureVector = {
  symbol, strategy, side,
  delta_usd,                 # signed(qty)·price·multiplier   — the load-bearing dimension
  beta_weighted_usd,         # delta_usd·beta                  — beta DEFERRED (see below)
  beta, beta_source,         # provenance for the above
  sector,                    # config/symbol_sectors.json (inverted); "unmapped" ⇒ count-only
  asset_class,               # equity|etf|futures|crypto|... from sec_type/secType directly
  venue,                     # IBKR | KRAKEN (derived: crypto⇒Kraken, else IBKR)
  price, price_source, price_staleness   # price_cache mark | intent limit/expected | bar fallback
}
```

**Honest computability (P3–P8):**

| Dimension | Today | Basis | Gap / disposition |
|---|---|---|---|
| **delta_usd** | ✅ equity/ETF · ⚠️ futures | qty·price·mult; equity mult=1; price from `price_cache` (intent leg uses `limit_price`→`expected_price`→cache) | Futures mult from a consolidated allocator-local reference over the 9 known roots (D2); an **unmapped futures root ⇒ delta_usd=null + loud**, never silently 0. The live book is 11× STK ⇒ fully computable now |
| **beta_weighted_usd** | ❌ deferred | no beta source anywhere (P4) | v1 ships the dimension with `beta=1.0`, `beta_source:"default_1.0"` (so `beta_weighted_usd ≡ delta_usd` and the vector *shape* is stable for R3). Optional static config beta map = operator_verify. True beta = R3 (D1) |
| **sector** | ✅ | invert `config/symbol_sectors.json` (P6) | unmapped ⇒ `unmapped` bucket, count-only, never binds (loud in evidence) |
| **asset_class** | ✅ | classify from `sec_type`/`secType` directly | The allocator does NOT inherit `position_exit_overlay._asset_class` (P7 MYM bug); it reads sec_type and flags MYM/crypto/options gaps as upstream defects, not its own |
| **venue** | ✅ (derived) | crypto⇒Kraken else⇒IBKR (P7) | no explicit venue enum; two venues wired |

**Deferred, explicitly (not in W5B):** beta-weighting (R3), real options-contract
greeks (R3), full-size futures multipliers + the 3-table unification, the MYM
classifier fix, a USD equity basis. All named as follow-ups (§8, §10-D).

## 4. Limits schema — `config/portfolio_limits.json` (every number sourced)

`portfolio_limits.v1`. Every starting number cites an existing artifact; where
no source exists, the field is `operator_verify` and **report-only** (computed,
never would-rejected) until ratified. No invented limits.

```jsonc
{
  "schema_version": "portfolio_limits.v1",
  "equity_basis": {                                   // P13
    "value_cad": 990919.08,
    "source": "runtime/pnl_state.json::account_equity (CAD, currency_ok=true)",
    "usd_note": "USD authoritative equity is null (no FX rate). CAD is the only currency_ok figure.",
    "refresh": "read live each cycle; never hardcode the balance"
  },
  "firm": {
    "gross_exposure": {
      "cap_notional_usd": 500000,
      "source": "chad/policy.py:665 max_total_exposure (ENFORCED today, per-order grain)",
      "cap_mult_netliq": 2.0,
      "mult_source": "config/margin_block.json:11 aggregate gross (SHADOW, margin-based)",
      "currency": "operator_verify",              // D7: $500k label vs CAD-priced book
      "binds": true
    },
    "net_exposure": {
      "cap_notional_usd": null,
      "source": "operator_verify — NO net-exposure notional cap exists anywhere in the repo",
      "proposal_basis": "e.g. 1.0×equity; requires ratification",
      "binds": false                              // D3: computed + reported, never would-rejected until set
    },
    "per_symbol_concentration": {
      "cap_notional_usd": 150000,
      "source": "chad/policy.py:666 max_symbol_exposure (ENFORCED today)",
      "cap_pct_equity": 0.05,
      "pct_source": "config/sizing_config.json:16 max_position_pct",
      "binds": true
    },
    "per_sector": {
      "cap_notional_usd": 5000,
      "source": "config/sizing_config.json:14 max_sector_exposure",
      "operator_verify": true,
      "note": "$5k was calibrated for a small account; mismatched to a ~$990k book.",
      "binds": false                              // D4: report-only until book-rescaled
    },
    "per_venue": {
      "kraken": {
        "cap_usd": 184.58,
        "source": "runtime/kraken_balances.json usd_equivalent (the $185 wallet)",
        "dynamic_cap_usd": 690.51,
        "dynamic_source": "runtime/dynamic_caps.json:262 alpha_crypto (exceeds the wallet; wallet is the true binding constraint)",
        "binds": true
      },
      "ibkr": {
        "cap_usd": null,
        "source": "operator_verify — no IBKR-specific venue cap; IBKR falls under firm gross $500k",
        "binds": false
      }
    }
  },
  "scr_sizing_reference": {                          // reference only — the allocator OBSERVES, it does not re-apply SCR
    "WARMUP": 0.10, "CAUTIOUS": 0.25, "CONFIDENT": 1.00, "PAUSED": 0.00,
    "source": "chad/analytics/shadow_confidence_router.py:79-82",
    "live_state": "WARMUP (runtime/scr_state.json)"
  },
  "coach": { "reject_streak_n": 3, "reject_streak_n_verified": false }  // operator_verify (P11 ttl idiom)
}
```

**Binding vs report-only (the honesty split):** gross, per-symbol, per-venue-Kraken
would-reject from hard-sourced, already-enforced numbers. Net-exposure,
per-sector, and per-venue-IBKR are **computed and reported but never
would-rejected** until an operator ratifies a source — a would-reject on an
admittedly-mismatched or nonexistent number would poison the shadow corpus.

## 5. Placement + the "exits always free" invariant

**Where:** the observer is called per intent at `live_loop.py:2704`, beside the
fuse gate (~`:2759`), sharing its bypass predicate:

```
bypass = is_flip_signal(intent)
       OR intent.side in {EXIT, CLOSE}
       OR is_protective_signal(intent)          # reduce/hedge/stop_loss/liquidation tags
```

A bypassed intent is **never evaluated and never produces a row** — so the
corpus is entries-only and cannot be misread as gating an exit. Overlay,
crypto-overlay, reconciler, and **flatten** closes never reach `:2704` at all
(P9: `apply_close_intents → adapter` direct) — the invariant holds *structurally*
first, and the predicate covers strategy-emitted exits on top. In shadow the
`should_block` result is unconditionally False anyway, so exits are triply free.

**Named invariant test** (mirrors W4A's `test_w4a_lc5_emergency_exits_free`):
`test_w5b_allocator_exits_always_free` — feeds the observer a batch of overlay
closes, a flatten-all batch (FLATTEN_ALL sentinel), flips, and protective
reduces, and asserts **zero verdict rows written** and `intent.quantity`
byte-identical for every one. Plus a `test_w5b_allocator_byte_identical_when_off`
(flag off ⇒ no rows, no state beyond the heartbeat) and a
`test_w5b_shadow_never_blocks` (shadow ⇒ `should_block()` False for every input,
including a would-reject).

## 6. Evidence, heartbeat, sentinel, coach (all four surfaces, P10/P11)

**(a) Per-intent verdict rows** — `data/allocator_shadow/allocator_shadow_YYYYMMDD.ndjson`,
`allocator_shadow.v1`, one JSON object per line (`sort_keys`, date-partitioned
from the verdict's own `ts_utc`), mirroring `ShadowVerdict.to_dict`:

```
schema_version, ts_utc, evaluated, mode,
verdict            : "WOULD_APPROVE" | "WOULD_RESIZE" | "WOULD_REJECT" | "ERROR",
which_limit        : "gross" | "net" | "per_symbol" | "per_sector" | "venue" | null,
breach_by_usd, breach_by_pct,                       // how much over — numbers here, never in a title
would_resize_to_qty,                                // for WOULD_RESIZE: the qty that would fit
symbol, side, strategy, asset_class, venue, sector, order_id,
intent_delta_usd, intent_beta_weighted_usd, beta_source,
book_gross_usd, book_net_usd, book_sector_usd,      // provisional book AFTER this intent
headroom_usd,
price_source, price_staleness, equity_staleness,
correlation_note,                                   // advisory (§7), never binding
extra
```

`should_block()→False` always (shadow contract, P10). `evaluate()` never raises;
internal error ⇒ fail-open `ERROR` verdict + `ALLOCATOR_SHADOW_ERROR` marker.
Grep-able marker per row: `ALLOCATOR_SHADOW verdict=… which_limit=… would=… mode=…`.
Pytest `evidence_dir`-required leak guard (the `build_default_shadow_gate`
pattern, `margin_shadow_gate.py:544-554`).

**(b) Heartbeat state** — `runtime/portfolio_allocator_state.json`,
`allocator_state.v1`, written **every cycle including OFF** (fuse-box doctrine,
P11), `ts_utc` + `ttl_seconds:180` via `write_runtime_state_json`:

```
{ schema_version:"allocator_state.v1", ts_utc, ttl_seconds:180, mode,
  session_date, epoch_started_at_utc,
  book: { symbols:11, gross_usd, net_usd, by_sector:{...}, by_venue:{...},
          equity_basis_cad, equity_staleness },
  cycle: { intents_evaluated, would_approve, would_resize, would_reject,
           by_limit:{gross:…, per_symbol:…, venue:…} },
  correlation: { computed:true, max_abs_offdiag:…, cluster_note:… },   // §7 advisory
  staleness: { price:"fresh", equity:"fresh" } }
```

**(c) Sentinel visibility** — one `feeds{}` row in `config/exterminator.json`
for `portfolio_allocator_state` (`ttl_seconds:180`, `ttl_verified:true`,
`ttl_source:"artifact:ttl_seconds; publisher pins allocator_state.v1 at
chad/risk/portfolio_allocator.py:<LINE>"`, warn/fail 180/360) + an EXS7
`schema_contracts.enforced` pin. **Disjoint from W5A's `clock_health`/EXS10
edits** — coordination in D8.

**(d) Coach NOTIFY — would-reject STREAKS only, dedupe-stable** (P11): NOT
per-intent (that is exactly the R13 flood CTF-T2 fixed). Emit
`format_alert("allocator_reject_streak", facts)` only when the **same limit
dimension** would-rejects `reject_streak_n` (default 3, `operator_verify`)
consecutive intents in a cycle/session — mirroring `_tpl_edge_decay`'s
`consecutive_losses`. Dedupe key = `allocator_reject_<dimension>` (digits
stripped, value-free — `dedupe_identity`); the count and $ live only in the
evidence + the PRO line, never the headline. Fails open on a corrupt dedupe file.

## 7. The correlation honesty question — recommendation

**Recommendation: static sector buckets BIND; a bounded rolling-correlation
DIAGNOSTIC observes (advisory-only). Buckets gate, correlation informs.**

The data is cheap and fresh (P12): all 11 book symbols carry ~251 daily bars
refreshed nightly, and `portfolio_var.py` already computes the returns + σ. An
11×11 Pearson matrix each cycle is negligible. So the cost argument is *not*
"correlation is expensive" — it is "correlation as a *binding limit* is
premature and unsourced." Concretely:

- **Why buckets bind, not correlation.** (1) R2's EWMA correlation is explicitly
  Stage 3 — making correlation a limit now pre-empts and would later collide
  with R2's design. (2) **No correlation threshold is sourced anywhere** — any
  binding "|ρ|>x ⇒ resize" number would be invented (violates the §4 rule).
  (3) The 11-name book is polluted with `broker_sync`/`epoch3_adopted`
  reconciliation namespaces (P2); a static sector map is deterministic and
  already ratified (`symbol_sectors.json`, W4A), a live matrix on a tiny noisy
  book is not.
- **Why compute the diagnostic anyway.** It is nearly free and it *de-risks R2*:
  the shadow corpus will show whether the static buckets are **missing real
  correlation clusters** — e.g. does PSQ/SVXY actually offset the megacap longs,
  or are AAPL/MSFT/V/MA one bucket the sector map splits? R1's `correlation_note`
  (advisory) + the `allocator_state.correlation` summary give R2 a ready-made
  input when it lands. The allocator's own `allocator_correlation.py` reads
  `data/bars/1d` directly (does NOT touch portfolio_var's `var_state` contract).
- **Cost of the rejected alternative** (correlation binding now): ratify
  unsourced thresholds, entangle with `portfolio_var.py`, and redo it all when
  R2's EWMA arrives — net negative. **Static-buckets-only** (no diagnostic) is
  the cheap fallback if the operator wants zero new compute, but it forfeits the
  free R2 de-risking. Recommend buckets + advisory diagnostic (D5).

## 8. Interaction analysis (REQUIRED)

### 8.1 The prime invariant — the allocator is closes-blind
| Path | Why the allocator can never touch it |
|---|---|
| Equity/crypto exit-overlay closes | Never traverse `:2704` (P9: `apply_close_intents → adapter` direct). Structural. |
| Reconciler closes, **flatten** | Same direct path; never reach the observer. INC-0723-salient: flatten is provably unseen. |
| Strategy exits/flips/protectives | Bypass predicate (flip / EXIT / CLOSE / protective tags) ⇒ **no evaluation, no row** (§5). |
| Every input, shadow mode | `should_block()` unconditionally False (P10) — nothing blocks regardless. |

### 8.2 Composition with the existing (per-order) caps — R1 is the missing *aggregate* layer
| Existing mechanism | Grain / basis | R1 relationship | Conflict? |
|---|---|---|---|
| `policy.py` gross $500k / per-symbol $150k | per-order / per-symbol notional, ENFORCED | R1 sums the SAME dollars across correlated tickets at **portfolio grain** — the layer policy.py cannot see. Cites policy's numbers as its starting caps | No — R1 blocks nothing; different grain |
| `margin_block` gross 2.0×NetLiq | aggregate, **margin/NetLiq** basis, shadow | R1 gross is **delta-dollar** based (different lens); both shadow. R1 cites 2.0× as the mult reference | No — complementary lenses |
| SCR sizing (WARMUP 0.10…) | global state → qty multiplier | R1 **observes**, never re-applies; SCR reference in config for context only | No |
| `sizing_config` 5% / sector $5k | composite per-symbol/sector | R1 cites 5% (binds) and $5k (report-only, rescale D4) | No |
| Per-strategy loss limits, edge-decay, fuse box (LC2/3/5) | outcome/streak based, entry gates | Orthogonal — those read *closed-trade outcomes*; R1 reads *open exposure + pending intents*. Distinct dedupe keys | No |
| `portfolio_var.py` (`var_state.v1`) | independent-σ VaR | R1 reuses the *approach* (returns/σ) in its OWN helper; must NOT mutate the hash-pinned `var_state` contract | No — separate module |

**Feedback-loop check:** R1 consumes the book snapshot + this cycle's intents and
writes evidence/state; it feeds no counter that feeds itself, and in shadow it
changes no intent — no loop, no oscillation.

### 8.3 What W5B does NOT do (named, so nobody blames the allocator)
Beta-weighting (R4-P4 → R3); real options greeks (P5 → R3); full-size futures
multipliers + the 3-table unification (P3); the MYM classifier fix (P7); a USD
equity basis (P13); **any enforce path** (Stage-2 = shadow). All follow-ups.

## 9. Phase-2 commit plan (each: build → set-diff green → commit, prefix `W5B`)

| # | Commit | Contents |
|---|---|---|
| W5B-0 | baseline capture | full-suite run in worktree; record failing-test set + count as `audits/W5B_BASELINE.md` (W4A/W5A methodology) |
| W5B-1 | exposure core | `portfolio_allocator.py` `ExposureVector` + book/intent normalization (delta_usd, sector-inversion, asset_class-from-sec_type, venue) + allocator-local futures multiplier reference (D2); stdlib, unit tests over synthetic book + the real 11-symbol book |
| W5B-2 | limits + evaluation | `config/portfolio_limits.json` + loader + limit engine (WOULD_APPROVE/RESIZE/REJECT, binding-vs-report-only split); tests incl. "3rd correlated ticket would-rejects gross" |
| W5B-3 | shadow gate + placement | `allocator_shadow_gate.py` + stage-3 observer hook + provisional-book accumulation + bypass predicate; byte-identical-when-off + `exits_always_free` + shadow-never-blocks named tests |
| W5B-4 | evidence + heartbeat | `allocator_shadow.v1` rows + `allocator_state.v1` heartbeat (incl OFF) + book/venue/sector summary; leak-guard + heartbeat tests |
| W5B-5 | correlation diagnostic | `allocator_correlation.py` (read-only rolling ρ over `data/bars/1d`, advisory `correlation_note` + state summary); tests; does not touch `portfolio_var.py` |
| W5B-6 | coach + sentinel | would-reject-streak `format_alert` (dedupe-stable) + `exterminator.json` feeds row + EXS7 pin (coord D8) + check-count pin; dedupe + streak tests |
| W5B-7 | closure | docs + PLAN_W5B closure record (decisions → commits). No Lane-A file touched |

Set-diff: after every commit, `python3 -m pytest chad/tests/ -q` in the worktree;
failure set must be ⊆ W5B-0 baseline (named-test diff, not count). `py_compile`
per changed file; `CHAD_SKIP_IB_CONNECT=1 full_cycle_preview` smoke at W5B-3/-4
(the wiring commits). No pushes, no merges — same STOP as W4A/W4B/W5A.

## 10. Decision points (operator input required before Phase 2)

- **D1 — Beta-weighting disposition.** (a) ship the `beta_weighted_usd` dimension
  with `beta=1.0`+`beta_source:"default_1.0"` (vector shape stable for R3,
  honest that it's unweighted today); (b) add a static config beta map now
  (operator_verify each value); (c) omit the dimension until R3. *Recommended:
  (a)* — no beta source exists (P4); a fake map is worse than an honest 1.0.
- **D2 — Futures multiplier source.** Allocator-local reference table sourced
  from `alpha_futures.DEFAULT_SPECS` + `omega_macro` (M6E) over the 9 known
  roots; an unmapped root ⇒ `delta_usd=null` + loud, never silent 0. Do NOT
  rewrite the 3 existing tables (unification is a named follow-up). *Recommended:
  yes* (the live book is all-equity, so this only bites future futures intents).
- **D3 — Net-exposure cap.** No source exists (P14). Compute net, **report-only**
  (never would-reject) until an operator ratifies a basis, vs propose 1.0×equity
  now. *Recommended: report-only + named proposal* (no invented binding number).
- **D4 — Per-sector cap.** The $5k source is small-account calibrated (P14).
  **Report-only** (advisory flag) until book-rescaled, vs rescale now
  (operator_verify). *Recommended: report-only until ratified.*
- **D5 — Correlation (the honesty question).** Static buckets bind + rolling-ρ
  advisory diagnostic (recommended, §7), vs buckets-only (cheapest, forfeits R2
  de-risking), vs correlation-binding-now (rejected — unsourced thresholds,
  R2 collision). *Recommended: buckets bind, diagnostic observes.*
- **D6 — Kraken mirror + cross-venue gross.** IBKR-lane observer only (the book
  has no crypto today) + a thin would-only Kraken mirror in the Kraken exec path
  for structural parity (W4A Kraken-mirror doctrine). Firm gross/net use the
  open-book snapshot (both venues) as the base; same-cycle cross-lane intents are
  an approximation captured next cycle (documented). *Recommended: IBKR observer
  + thin Kraken mirror; approximation noted* (alt: IBKR-only, Kraken deferred).
- **D7 — Currency of the exposure vector / gross basis.** The `policy.py` $500k
  cap is USD-labeled; the book is CAD-priced and equity is CAD-only (USD null,
  P13). Compute the vector in each instrument's native currency and compare gross
  against… which currency? *Recommended: compute + report in CAD (the only
  `currency_ok` basis), tag `gross.currency:"operator_verify"`, and would-reject
  on gross only once the operator confirms the $500k cap's currency* — until
  then gross is report-only too. (This keeps §4's no-invented-number rule honest
  about units, not just magnitudes.)
- **D8 — `exterminator.json` coordination + shadow-only confirmation.** W5B adds
  a `feeds{}` row + EXS7 pin; W5A adds `clock_health`/EXS10 — disjoint keys, but
  same file across two worktrees. Land W5B's sentinel bits in W5B-6 and sequence
  the merges (whichever merges second rebases the file). And confirm the
  headline scope: **W5B is `off|shadow` only — no enforce path is built**;
  R1-enforce is a separate future wave. *Recommended: sequence merges; confirm
  shadow-only.*

— END PLAN (Phase 1). STOP here; Phase 2 requires D1–D8.
