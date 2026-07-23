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

---
---

# PHASE 2 — GO RECORD, REBASE ONTO W5A, AND PREMISE RE-VERIFICATION

**Status: Phase 2 OPEN.** The Phase-1 plan was committed at `bca56e2` and the
session ended before the GO arrived. This section records (§11) the merge of
Lane A onto this branch, (§12) every Phase-1 premise that the merge or the
passage of time invalidated, (§13) the operator's GO and its amendments, and
(§14) the revised commit sequence actually built. Phase-1 text above is left
**unedited** — corrections live here, so the diff between what was planned and
what was true stays legible.

## 11. Base move: `main@e86eaaf` → `main@b5d0ee2` (Lane A merged in)

Phase 1 was written against `main@e86eaaf` (the W4A fuse-box merge). Lane A
(W5A, the measurement layer) has since merged to main at `b5d0ee2`. This branch
was moved forward by **merge, not rebase** (`7cfca0c`), deliberately: `bca56e2`
is cited by name in the closure records of the prior session, and rebasing would
have rewritten that hash. The merge was clean — **zero conflicts** — because the
two lanes are genuinely disjoint (§11.1).

### 11.1 What W5A actually touched, and what it means for W5B

| W5A surface | Overlap with W5B | Consequence |
|---|---|---|
| `chad/analytics/implementation_shortfall.py`, `chad/analytics/excursion_recorder.py` (new) | none | Lane-A territory, untouched (§0) |
| `chad/execution/trade_closer.py` (+51, the `to_payload` TCA/MAE-MFE stamp) | none | close-side; the allocator is entry-side |
| `chad/risk/position_exit_overlay.py` (+50), `chad/risk/crypto_exit_overlay.py` (+37) | none | overlay watermark hooks; the allocator never evaluates overlay closes (§5) |
| `chad/ops/exterminator_sentinel.py` (+185, EXS10 + embedded-schema validation) | **shared file, W5B-5** | additive; W5B adds a `feeds{}` row + an `enforced` pin |
| `config/exterminator.json` (+40: `schema_contracts.embedded`, `clock_health`) | **shared file, W5B-5** | **D8 is now MOOT** — see below |
| `chad/core/live_loop.py`, `chad/policy.py`, `config/sizing_config.json` | **not touched by W5A** | P9 and P14 survive intact |

**D8 dissolves.** The Phase-1 concern was two worktrees editing
`config/exterminator.json` concurrently and racing at merge. Lane A has already
merged; this branch now *contains* its edits. There is no race left to sequence —
W5B simply adds its keys on top of W5A's. The keys remain disjoint
(W5A: `schema_contracts.embedded`, `clock_health`; W5B: a `feeds{}` row +
`schema_contracts.enforced`). D8's second half (confirm shadow-only) is
answered by the GO: confirmed, `off|shadow`, no enforce path.

### 11.2 Should the allocator's evidence rows reference the W5A stamps? — **NO (values), YES (one soft handle)**

The operator asked this explicitly. Ruling, with the reasoning that produced it:

**The allocator must NOT carry `implementation_shortfall` or `mae_mfe` values.**
Three independent reasons, any one sufficient:

1. **Grain mismatch.** Both W5A blocks are *per-closed-lap realized* measures
   stamped at `trade_closer.to_payload` onto a `closed_trade.v1` row. An
   allocator row is a *pre-trade would-verdict on an intent*, emitted at stage-3.
   At emission there is no lap, no fill, and no realized cost — in shadow the
   intent may never even be submitted.
2. **The join key does not exist yet at emission.** `CONTRACT_W5A_harness_handoff.md
   §4` names `record_hash` — the closed-trade hash-chain identity — as the lap
   join key. It is minted at close. Nothing at stage-3 can reference it.
3. **It would be fake precision of exactly the kind this plan refuses
   elsewhere.** An allocator row carrying an `is_usd` field would read as though
   the shadow verdict were cost-aware. It is not, and no field should imply it.

**What the allocator SHOULD carry is the join *handle*, not the values** — so a
future lane can walk would-verdict → fill → closed lap → TCA without W5B having
claimed anything. But this surfaced a real defect, so the handle is weaker than
expected:

> **FINDING W5B-F1 (new, verified).** `StrategyTradeIntent`
> (`chad/execution/ibkr_executor.py:26-80`) is frozen and defines **neither
> `idempotency_key` nor `trace_id`**. PA-EP3 threaded the canonical join key into
> paper evidence via `execution_id=(getattr(intent,"idempotency_key","") or
> getattr(intent,"trace_id","") or "")` at `live_loop.py:3288-3292` — on the
> **IBKR lane both getattrs miss, so `execution_id` is always the empty string**.
> PA-EP3's join spine is live on the Kraken lane and **empty on the IBKR lane**.
> This is not W5B's to fix (it is an intent-schema change on a frozen dataclass
> in the execution hot path), but it bounds what any future would-verdict →
> realized-cost join can achieve, and it is filed as a follow-up.

So the allocator stamps a **soft correlation tuple** — `(ts_utc, symbol, side,
strategy, cycle_seq)` — and names it soft in the schema. It does not fabricate a
hard key that the execution layer does not mint.

One further honest note, recorded so a later reader does not assume identity:
the allocator's price ladder is `limit_price → expected_price → price_cache`,
while W5A's `decision_price_source` ladder is `submit_quote.ref_price →
expected_price`. **They agree on the `expected_price` fallback and differ on the
primary leg** — the allocator sits *upstream* of submit, so no `submit_quote`
exists when it evaluates. A future comparison of assumed-vs-realized decision
price must account for that, not treat the two as the same number.

## 12. Premise re-verification — five Phase-1 claims corrected

Re-audited against the post-merge tree and live runtime on 2026-07-23. Most of
§1 survives; these five do not, and two of them change the design.

| # | Phase-1 claim | Status now | Correction |
|---|---|---|---|
| **C1** | P10: copy the shadow-gate template from `chad/execution/margin_shadow_gate.py` (also cited at `fuse_gate.py:8`) | **FILE DOES NOT EXIST** | `margin_shadow_gate.py` is in **no ref of this repo** (`git log --all` empty; only `chad/risk/margin_block.py` + `config/margin_block.json` are in-tree). It is unpushed work in the `chad_finale` worktree ("3 commits push denied"). **Substitute template: `chad/risk/fuse_gate.py` + `chad/risk/fuse_box.py`** — W4A, in-tree, tested, and literally the neighbour at the call site. Strictly better: the allocator can *import* `fuse_gate.is_exit_like` instead of hand-rolling the §5 bypass predicate, so gate and allocator agree by construction on what a close is. `margin_block.json`'s 2.0×NetLiq citation (P14) is unaffected — that file is in-tree |
| **C2** | P9: per-intent loop at `live_loop.py:2704`; fuse hook ~`:2759` | **line numbers wrong** | Verified post-merge: `for intent in intents:` is at **`:2723`**; `FuseGate()` constructed at `:2700-2706`; `_fuse_gate.should_block(intent)` called at **`:2893-2895`**. W5A did not touch `live_loop.py` — the Phase-1 numbers were simply off. The *structure* (P9) is exactly as claimed; only the anchors move |
| **C3** | P2: 11-symbol book = AAPL, BAC, LLY, MA, MSFT, UNH, V, SPY, PSQ, SVXY (10 named for an 11-count) | **book moved; the count was one short** | Live `positions_truth.json` today: AAPL, BAC, LLY, MA, MSFT, UNH, V, SPY, **IWM**, PSQ, SVXY. IWM (a *second* long index ETF) was missing from the Phase-1 list. The disease is **worse** than described, not better — see §12.1 |
| **C4** | P13: equity basis = `pnl_state.json::account_equity`, "CAD, `currency_ok=true`" | **wrong artifact** | `pnl_state.json` has **no `currency`/`currency_ok` key at all** (the dashboard reads `account_equity_currency_ok`, which is likewise absent). The keys the plan attributed to it live on `portfolio_snapshot.json`: `ibkr_equity=993000.30`, `ibkr_equity_currency=CAD`, `ibkr_equity_currency_ok=true`. **Use `portfolio_snapshot.json`** — it is also what `tier_manager.py:587` reads, fail-closed, so the allocator inherits a ratified precedent instead of inventing one |
| **C5** | §7: "**No correlation threshold is sourced anywhere** in the repo" | **FALSE** | `config/sizing_config.json::correlation_monitor.threshold = **0.65**`, with `chad/risk/correlation_monitor.py` **WIRED** into `chad/execution/execution_pipeline.py:264` as a live size multiplier (the R6 correlation reducer). The config note even records the measured book statistic: *"a book averaging 0.654 pairwise correlation"*. §7's second argument was factually wrong — see §13.3 |

### 12.1 The book today — the finding is sharper than Phase 1 claimed

Computed from live `positions_truth.json` × `price_cache.json`, sectors from
`config/symbol_sectors.json`:

| Symbol | Qty | Price | delta_usd | Sector |
|---|---:|---:|---:|---|
| LLY | 182 | 1184.00 | **$215,488** | healthcare |
| SPY | 247 | 739.42 | **$182,637** | index_etf |
| UNH | 240 | 422.70 | $101,448 | healthcare |
| V | 195 | 350.76 | $68,398 | financials |
| IWM | 200 | 291.97 | $58,394 | index_etf |
| BAC | 213 | 61.28 | $13,053 | financials |
| MSFT | 34 | 381.65 | $12,976 | mega_tech |
| SVXY | 163 | 56.60 | $9,226 | vol_etp |
| MA | 10 | 530.06 | $5,301 | financials |
| AAPL | 12 | 321.01 | $3,852 | mega_tech |
| PSQ | 10 | 26.52 | $265 | inverse_etf |

**GROSS = $671,037 · NET = $671,037 · every single position is LONG.**

Three things follow, and all three strengthen the case for the allocator:

- **Gross ≡ net.** Phase 1 described "eight long-beta tickets partially offset by
  two inverse ETFs." There is no offset. PSQ — the *only* genuinely short-beta
  ticket — is **$265**, i.e. 0.04% of gross: rounding error. And the "offsetting"
  SVXY is a **long short-vol** position, which is *risk-on*: it adds to the
  long-beta bet rather than hedging it. The book is a one-way levered long.
- **Two positions already exceed the ENFORCED per-symbol cap.** `policy.py:666`
  `max_symbol_exposure = $150,000` is enforced *today* — yet LLY sits at $215,488
  and SPY at $182,637. Not a violation: the cap is **per-order**, and a position
  accumulated over several orders never meets it. This is the missing-aggregate-layer
  thesis demonstrated on live data, and it means the shadow corpus will produce
  real would-rejects from a **hard-sourced, already-ratified** number on day one.
- **Sector concentration is real:** healthcare $316,936 (47.2% of gross),
  index_etf $241,031 (35.9%), financials $86,751 (12.9%), mega_tech $16,828,
  vol_etp $9,226, inverse_etf $265.

### 12.2 The currency problem is worse than D7 assumed — and it forces the derivation's units

`portfolio_snapshot.json` reports `usd_ok=false`, `usdcad_rate_used=null`,
`total_equity_usd_authoritative=null`. The book is **USD-priced**
(`positions_truth` rows are all `currency:USD`; `price_cache` is USD); the only
`currency_ok` equity is **CAD**. **No FX rate exists anywhere to bridge them.**

Therefore `gross/equity = 671,037/993,000 = 0.676×` — the ratio Phase 1's D7
would have used — is a **mixed-unit number and is meaningless**. At a plausible
USDCAD it would be nearer 0.93×. A ceiling expressed as a multiple of equity
would silently inherit that error and be wrong by ~37%.

**Consequence for the GO's OPERATOR_VERIFY 1/2:** the derived shadow thresholds
are denominated in **USD notional**, derived from USD-denominated sources only
(`policy.py`'s USD caps and the USD-priced book). The CAD equity is recorded in
the config as context and explicitly **not** used as a divisor. This keeps §4's
no-invented-number rule honest about *units*, not just magnitudes.

## 13. The GO — operator ruling, 2026-07-23

All Phase-1 recommendations **accepted**: D1(a) `beta=1.0`/`beta_source:"default_1.0"`;
D2 allocator-local futures reference, unmapped root ⇒ `null` + loud; D3
net-exposure report-only; D5 static buckets bind; D6 IBKR observer + thin Kraken
mirror; D7 as amended by §12.2; D8 moot per §11.1 + shadow-only **confirmed**.
Plus four amendments:

### 13.1 OPERATOR_VERIFY 1 — the gross-exposure ceiling

> *Accept the derivation from current book shape + tier headroom as the SHADOW
> threshold; stamp config `basis:"shadow_derivation_2026-07"` — it generates
> would-reject evidence, it is NOT a ratified limit; the enforce-era number gets
> its own PA.*

**Derivation (auditable, USD, per §12.2):**

- Only USD-denominated firm-level exposure number in the repo:
  `policy.py:665 max_total_exposure = $500,000` (enforced, per-order grain).
- Live book gross = **$671,037** — already 1.34× that number, which is precisely
  why a per-order cap cannot express a portfolio ceiling.
- **Tier headroom:** live tier is `SCALE` (`tier_state.json`, band
  CAD 226,560 → 14,160,000, demotion floor CAD 203,904). SCALE's `risk_profile`
  is **all-null** — the tier imposes *no* competing notional ceiling, so the
  derivation is unconstrained from above and must supply its own discipline.
- **Chosen: `gross_cap_notional_usd = 750,000` = 1.5 × the enforced per-order
  gross.** It sits **above** the current book (headroom $78,963, so the corpus is
  not saturated on day one and every row is a genuine *marginal* verdict) and
  **below** the only existing aggregate reference (`margin_block` 2.0×NetLiq).
  A ceiling *below* the live book would would-reject every intent forever and
  the corpus would carry no information.

Stamped `basis:"shadow_derivation_2026-07"`, `ratified:false`,
`enforce_era_requires_pa:true`.

### 13.2 OPERATOR_VERIFY 2 — the per-sector cap

Same treatment, same stamp. The Phase-1 source (`sizing_config` `$5,000`) is
small-account calibration and is **not** used: against this book every one of the
six live sectors breaches it, which is saturation, not evidence.

**Derivation: `per_sector_cap_notional_usd = 375,000 = 0.5 × the gross ceiling`** —
"no single sector may exceed half the firm gross ceiling." Against the live book
that puts healthcare ($316,936) at **84.5% of its sector cap with $58,064 of
headroom** — inside the limit today, so the *next* healthcare ticket is what
trips it. That is exactly the marginal concentration signal R1 exists to produce.
Stamped identically; the `$5,000` figure is retained in the config as
`superseded_source` with the reason, so nothing is quietly dropped.

**D4 changes as a result:** Phase 1 recommended per-sector be report-only. Under
the GO it **binds in shadow** against the derived number. "Binds in shadow" means
only that it generates `WOULD_REJECT` evidence — there is still no enforce path
anywhere in W5B.

### 13.3 CORRELATION — rolling ρ deferred to R2; the plan's own reasoning was partly wrong

> *Static sector buckets ratified; rolling correlation deferred to R2 with real
> sample depth (your own fake-precision argument governs).*

**Effect: the Phase-1 `W5B-5` commit (`allocator_correlation.py`) is CUT.** No
rolling-correlation module is built in this wave.

The ruling is right, but §7's stated reasons were not, and the record should say
so. §7 argued "no correlation threshold is sourced anywhere" — **false** (C5):
`sizing_config.correlation_monitor.threshold=0.65` is sourced *and wired* into
live sizing at `execution_pipeline.py:264`. So the real argument for deferral is
the **opposite** of the one Phase 1 gave: a correlation reducer already exists at
per-order grain, and a second, independently-thresholded correlation layer at
portfolio grain would **double-count the same effect** through two unreconciled
thresholds. That is a stronger reason to defer than "unsourced" ever was, and it
hands R2 a concrete first task: reconcile with `correlation_monitor`, don't
duplicate it.

The `allocator_state` heartbeat therefore carries a **declarative** correlation
field — `{"mode":"static_sector_buckets","rolling_deferred_to":"R2",
"existing_per_order_reducer":"chad/risk/correlation_monitor.py"}` — naming the
regime and the deferral. **No ρ is computed and no numeric correlation appears
anywhere in W5B**, so there is no fake precision to mistake for a measurement.

### 13.4 BYPASS FINDING — a named standing finding

> *The flip-executor/reconciler stage-3 bypass is a named standing finding — in
> the plan, and as a sentinel-visible note in the allocator's first report. It
> bounds what shadow evidence can claim and heads the agenda of any future
> enforce-flip PA.*

**FINDING W5B-SF1 (standing).** The same structural property that makes the
allocator **safe** — overlay, crypto-overlay, reconciler and flatten closes go
`apply_close_intents → adapter` direct and never traverse stage-3 (P9) — also
makes it **partially blind**. Those paths can change the book without the
allocator ever seeing the intent that changed it. Precisely:

- The allocator's provisional book is *open book at cycle start + entries it saw
  this cycle*. Any position change arriving through a bypassing path lands in the
  next cycle's snapshot, not this one.
- A **flip** is the sharp case: its closing leg bypasses, and it is the executor
  and reconciler — not stage-3 — that move the book. A flip can therefore move
  gross materially between two allocator evaluations, and the second evaluation
  will be computed against a book that is one cycle stale.
- INC-0723 is the live precedent for how much a bypassing path can move real
  quantities without stage-3 participating.

**What this bounds.** Shadow evidence supports claims of the form *"this entry
would have breached the ceiling given the book as of cycle start."* It does
**not** support *"gross never exceeded X"* — the allocator cannot see every
mutation, only entries. Any future enforce-flip PA must open on this item: an
enforcing allocator with a one-cycle-stale book could reject a legitimate entry
or admit a breaching one, so enforce requires either book re-read at evaluation
or explicit participation from the bypassing paths.

**Surfacing.** Carried as `standing_findings[]` in every
`allocator_state.v1` heartbeat (so the sentinel sees it on the very first report
and on every subsequent one, not just once), and in the closure record.

### 13.5 The exits-always-free invariant

Unchanged and reaffirmed: the allocator never evaluates overlay closes,
advice-fired closes, or flatten. Enforced structurally (placement) + by predicate
(`fuse_gate.is_exit_like`, now **imported rather than re-implemented**, per C1) +
by the absence of any enforce path. Pinned by
`test_w5b_allocator_exits_always_free`.

## 14. Revised Phase-2 commit sequence

Phase-1 §9's sequence, with the correlation commit cut (§13.3) and the rest
renumbered to stay contiguous:

| # | Commit | Contents | Δ vs Phase 1 |
|---|---|---|---|
| W5B-0 | baseline + plan amendment | full-suite baseline → `audits/W5B_BASELINE.md`; this §11-§14 | expanded (premise corrections) |
| W5B-1 | exposure core | `portfolio_allocator.py`: `ExposureVector`, book/intent normalization, futures reference (D2) | as planned |
| W5B-2 | limits + evaluation | `config/portfolio_limits.json` (§13.1/§13.2 derivations) + loader + limit engine | thresholds now derived, not report-only |
| W5B-3 | shadow gate + placement | `allocator_shadow_gate.py` + stage-3 hook at `:2723`/`:2893` + provisional book + `is_exit_like` bypass | anchors corrected (C2), predicate imported (C1) |
| W5B-4 | evidence + heartbeat | `allocator_shadow.v1` rows + `allocator_state.v1` incl. OFF + `standing_findings` | + §13.4 surfacing |
| W5B-5 | coach + sentinel | reject-streak `format_alert` + `exterminator.json` feeds row + EXS7 pin | was W5B-6; correlation commit cut |
| W5B-6 | closure | closure record + follow-ups (W5B-F1, W5B-SF1) | was W5B-7 |

Set-diff methodology, `py_compile`, and the `full_cycle_preview` smoke at W5B-3/-4
are unchanged from §9. **Same STOP as W4A/W4B/W5A: no push, no merge.**
