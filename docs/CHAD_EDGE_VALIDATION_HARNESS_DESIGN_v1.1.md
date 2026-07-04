# CHAD Edge-Validation Harness — Design Document v1.1

**Status:** DESIGN / FOR REVIEW — no code written yet. Supersedes v1.
**Gate it controls:** `ready_for_live = false` (the single machine that could ever justify unlocking it)
**Repo:** `chad_finale` @ `769227d` | **Date:** 2026-07-04 | **Author:** TEAM CHAD
**Operator decisions locked:** (1) Maximally strict proof standard. (2) Historical backtest first; live-trade-log validation plugs into the same scoring spine later.

---

## CHANGELOG v1 → v1.1 (what the review changed and why)

The v1 philosophy stands; several of its strongest guarantees were *promises* that needed to become *mechanisms*. v1.1 closes:

- **F1 — OOS lockbox.** Out-of-sample discipline is now mechanically enforced (hash-seal + `--final-run` flag + immutable access log + contamination flag), not honor-system.
- **F2 — Config freeze + pre-registered minimums.** Thresholds and costs are hash-frozen before the sealed OOS run; post-FAIL config changes count as new trials. Minimum sample sizes are pre-registered *in this doc* (§4.3) so "sufficient data" is not negotiated after seeing results.
- **V1 — Per-head feature-parity audit.** Phase 4 begins by classifying every input each head reads live as reconstructable / approximable / unavailable; category-(c) heads get honest `NOT_REPLAYABLE` status instead of a silent degraded replay.
- **V2 — Pessimistic intrabar rule.** When a daily bar can't disambiguate stop-vs-target, assume the *worse* outcome; log every ambiguous fill.
- **V3 — Universe provenance flag.** The 52-symbol selection is documented and flagged as a known upward (survivorship/selection) bias in every verdict.
- **V4 — Independent regime labeler.** The harness slices regimes with its OWN dead-simple, tested definition — NOT CHAD's classifier (the component we just proved buggy). CHAD's label is reported alongside, never as authority.
- **S1 — Punitive trial count.** Deflation uses a deliberately punitive N (documented + justified), because only surviving heads are visible and every abandoned head/sweep was also a trial.
- **S2 — Block bootstrap.** Ruin is computed with a stationary/block bootstrap that preserves loss-clustering; IID reported alongside; trust the worse number.
- **S3 — Seed sweep + worst-quantile verdict.** The sealed run sweeps a fixed seed set; thresholds apply to the worst quantile, not the mean.
- **S4 — Costs on Stage-2 fills.** Real paper fills get the same spread/slippage haircut as synthetic Stage-1 trades.
- **S5 — Allocator-inclusive portfolio pass logic.** The 50/30/20 allocator is replayed as part of the portfolio track; portfolio PASS requires the portfolio bar PLUS a stated fraction of capital in individually-surviving heads.
- **NEW Phase 0 — Bar-corpus forensic audit.** Data quality (adjustment, gaps, stale prints, FX provenance) is validated before anything is backtested. Garbage bars → confident garbage verdict, so this comes first.
- **Plus:** per-head label horizon for purged walk-forward; transitive-import closure test; DSR verified against published worked examples.

---

## PART 0 — PLAIN ENGLISH (read this first)

### What this machine is
The **proof machine**. One job: answer, with evidence, *does CHAD actually have an edge — or did it get lucky?* Nothing risks a real dollar until it says **yes** on strict terms. It is the wall between paper and live money.

### Why it must be the strictest thing in CHAD
Your rule is **no ruin paths — one bankroll**. The most dangerous thing we could build is a *weak* proof machine, because a number that *looks* like proof but isn't is what blows up a single bankroll. So this machine is built to be **hard to pass, and unable to cheat itself**.

### The five ways a system fools itself — and how v1.1 physically blocks each
1. **Grading your own exam (peeking at the test data).** v1 *promised* not to peek. v1.1 **locks the test data in a sealed box** — the machine literally refuses to open it except on one final, logged run, and stamps the report "contaminated" if it was opened more than once. A promise became a lock.
2. **Pretending trading is free.** Every trade — synthetic *and* real paper trade — is charged realistic fees, spread, and slippage. Under-charging is the classic backtest lie; we over-charge on purpose.
3. **Getting lucky in one kind of market.** Tested across market moods over time. And crucially, the "what mood is the market in?" ruler is a **simple, independent one built inside the machine** — NOT CHAD's own mood-detector (the one we just caught lying about "bear market"). You don't let the defendant pick the judge.
4. **Trying 17 things and cheering the lucky one.** The machine mathematically discounts the best results for how many strategies were tried — and counts *generously* (including abandoned experiments), because underestimating this is the #1 way rigor-looking machines hand out false confidence.
5. **Moving the goalposts after a bad result.** The pass/fail thresholds and cost settings are **frozen and fingerprinted before the test data is opened.** Change them after a FAIL and the machine counts it as another attempt and flags the result. No quiet "nudge the settings and re-run."

### The honesty defaults
- If there isn't enough data to judge strictly, the verdict is **`INSUFFICIENT_DATA`** — never a fabricated pass. With only 52 symbols of daily bars, this will *often* be the honest answer at first. That is the machine working.
- A pass is labeled **`PASS (candidate)`** — evidence for *your* decision. The machine **never** flips `ready_for_live` itself.

---

## PART 1 — SCOPE & BOUNDARIES

### 1.1 What it IS
A standalone, offline, deterministic evaluation engine that replays strategy logic (or a trade log) over historical data, charges realistic costs, enforces sealed train/validation/OOS partitions and purged walk-forward, computes cost-adjusted + multiple-testing-corrected + ruin metrics, and emits a signed, reproducible verdict artifact.

### 1.2 What it is NOT (hard boundaries)
- Never places orders, connects to any broker, or reads live market data. Fully offline.
- Never writes `ready_for_live`, mutates `runtime/` trading state, or influences the live loop.
- Never runs inside / is imported by the live trading path (enforced by transitive-import test, §6 Phase 1).
- Never auto-unlocks any gate. Output is evidence for a human decision.
- Never fabricates a pass on thin/absent data → `INSUFFICIENT_DATA`.

### 1.3 Two inputs, one spine
- **Stage 1 — Historical backtest (BUILD NOW):** replay head logic over historical bars → synthetic trades → scored.
- **Stage 2 — Live-trade-log validation (LATER):** ingest real post-Epoch-3 effective trades through the *identical* scoring spine; **same cost haircut applies (S4)**. Only the input adapter differs. One `ScoringSpine`, no duplicate scoring logic.

---

## PART 2 — ARCHITECTURE

```
chad/validation/                    # isolated package; imported by NOTHING in live trading
  __init__.py
  bar_audit.py            # Phase 0: forensic data-quality audit of the bar corpus
  data_provider.py        # loads audited bars; strict no-lookahead windowing; per-head label horizon
  regime_labeler.py       # INDEPENDENT regime definition (NOT chad's classifier) + tests
  cost_model.py           # commissions + spread + slippage (+ optional impact); applies to BOTH stages
  feature_parity.py       # Phase 4 gate: per-head input reconstructability audit → REPLAYABLE / NOT_REPLAYABLE
  backtest_engine.py      # Stage-1 replay → synthetic trades; pessimistic intrabar resolution
  trade_log_adapter.py    # Stage-2 real-trade ingest (tested stub now) → same spine, same costs
  scoring_spine.py        # THE shared spine: Sharpe/Sortino/DD/CAGR
  significance.py         # deflated Sharpe (punitive N), block bootstrap ruin, seed sweep
  splits.py               # train/val/OOS + purged+embargoed walk-forward
  oos_lockbox.py          # hash-seal OOS, --final-run gate, immutable access log, contamination flag
  config_freeze.py        # hash-freeze thresholds+costs before sealed run; post-FAIL change = new trial
  verdict.py              # thresholds → PASS(candidate) / FAIL / INSUFFICIENT_DATA / NOT_REPLAYABLE
  report_writer.py        # edge_report_<ts>.json (signed) + human-readable .md
  cli.py                  # `python -m chad.validation.cli --stage historical [--final-run]`
tests/validation/         # unit + property + known-answer (incl. published DSR worked examples)
docs/EDGE_HARNESS_DESIGN_v1.1.md   # this doc, committed as build SSOT
```

**Isolation guarantee (strengthened):** a test asserts the *full transitive import closure* of `chad.validation` excludes `live_loop`, broker adapters, and any `runtime/` reader — catching forward-imports through strategy modules, not just direct reverse-imports.

---

## PART 3 — THE STRICT STANDARD (concrete mechanisms)

### 3.1 OOS lockbox (F1 — the core enforcement)
- At split time, the OOS partition is **hash-sealed**; its hash is recorded.
- The engine **refuses to score OOS** unless invoked with explicit `--final-run`.
- Every OOS access appends to an **immutable run-log** (timestamp + config hash + code commit).
- The report records **OOS access count**; count > 1 ⇒ verdict auto-flagged **CONTAMINATED**.
- All development/debugging runs against **train/validation + a synthetic decoy OOS set** only. The real OOS is never touched during the 6 build phases.

### 3.2 Config freeze (F2)
- Thresholds and all cost parameters are **committed + hashed before** the sealed OOS run (`config_freeze.py`).
- Any config change after a FAIL **invalidates the seal** and **increments the trial count** used in deflation (§3.3). Moving goalposts is mathematically penalized, not silently allowed.

### 3.3 Multiple-testing correction — punitive N (S1)
- Report the **Deflated Sharpe Ratio** (Bailey & López de Prado), per-head and portfolio.
- **N (effective trials) is deliberately punitive**: not the ~17 survivors, but a documented multiple (starting 5–10×) accounting for abandoned heads, parameter sweeps, and the classifier iterations in CHAD's history. The chosen N and its justification are printed in every report and revisitable — but never *below* survivor count.
- DSR implementation is **verified against published worked examples**, not only internal fixtures (an off-by-one here defeats the whole machine invisibly).

### 3.4 Independent regime labeler (V4 — no circularity)
- Regime slices use a **dead-simple, auditable definition computed inside `chad/validation/`**: e.g. trailing SPY return sign × realized-vol tercile → {bull_calm, bull_vol, bear_calm, bear_vol, flat}. Fully unit-tested.
- **CHAD's own regime classifier is NOT the slicing authority** (it's the component we just proved emitted false `trending_bear`). It is reported *alongside* for comparison only.
- An edge that exists in only one regime is flagged **regime-fragile** — not a pass unless sizing is formally restricted to that regime.

### 3.5 Cost model (S4) + pessimistic execution (V2)
- Charge commission + half-spread (entry & exit) + volatility/volume-linked slippage on **every** trade — synthetic (Stage 1) **and** real paper fills (Stage 2), since IBKR paper fills are optimistic (mid-ish, no queue/partials).
- **Pessimistic intrabar rule:** when a daily bar's range contains both stop and target, assume **the stop hit first**. Every ambiguous fill is logged and counted.
- All cost params conservative by default and printed in the report.

### 3.6 Ergodicity / ruin — block bootstrap + seed sweep (S2, S3)
- Compute **time-average growth**, path-dependent drawdown distribution, and **ruin probability** — the last via a **stationary/block bootstrap** (block length tied to observed autocorrelation) that preserves loss-clustering. IID bootstrap reported alongside; **trust the worse number.**
- The sealed run **sweeps a fixed seed set (e.g. 25)**; ruin/DSR are reported as distributions, and **verdict thresholds apply to the worst quantile**, not the mean.
- A great Sharpe with a non-trivial ruin path **FAILS**, regardless of Sharpe. Ergodicity over expected value, enforced.

### 3.7 Splits & label horizon
- Purged + embargoed walk-forward. **Purge/embargo requires a declared per-head label horizon** (holding period) — logged per head — because multi-day holds create overlapping labels that must be purged to prevent leakage.

### 3.8 Determinism
- Same inputs → byte-identical report (within the seed-sweep contract). Anti-lookahead enforced structurally and tested with poisoned future data that must NOT alter past decisions.

---

## PART 4 — THE VERDICT

### 4.1 Verdict types
- **`INSUFFICIENT_DATA`** — below pre-registered minimums (§4.3). Honest default.
- **`NOT_REPLAYABLE`** — a head depends on inputs unavailable historically (V1); it is not scored in Stage 1 rather than silently degraded.
- **`FAIL`** — sufficient data, edge does not survive: deflated Sharpe not > 0 at confidence, OR cost-adjusted CAGR ≤ 0, OR worst-quantile ruin above bound, OR regime-fragile without scoped sizing.
- **`PASS (candidate)`** — survives OOS + costs + punitive deflation + regime slices + worst-quantile ruin. **Still just evidence for a human decision.**

### 4.2 Portfolio-vs-head logic (S5)
- The **50/30/20 sleeve allocator is replayed as part of the portfolio track** (it is itself a strategy with degrees of freedom, so it is inside the backtest, not assumed).
- Portfolio `PASS (candidate)` requires: the portfolio deflated-Sharpe bar **AND** at least a **stated fraction of capital allocated to individually-surviving heads** (a portfolio cannot pass on the back of 14/17 failing heads carried by allocator luck).

### 4.3 Pre-registered minimums (F2 — set NOW, not after results)
A verdict stricter than `INSUFFICIENT_DATA` requires at least:
- **≥ 30 OOS trades** per head being judged (**N_min = 30**),
- **≥ 6 walk-forward windows** (**W_min = 6**),
- **≥ 3 distinct regimes** represented in OOS (**R_min = 3**),
- data-quality audit (Phase 0) **PASS** for every symbol involved.

**Committed values + justification (pre-registered at Phase 3, 2026-07-04, before any sealed run — these are now FIXED; changing them post-FAIL is a config change that invalidates the seal and increments the deflation trial count per §3.2):**
- **N_min = 30 OOS trades/head.** Below ≈30 trades the per-head Sharpe and — critically — its skewness and kurtosis (which the Deflated Sharpe Ratio consumes, §3.3) are dominated by sampling noise, so the DSR denominator is unreliable. 30 is the conventional central-limit floor and is consistent with the borderline sample sizes (n ≈ 24–29) of the published PSR/DSR worked examples this phase encodes as known-answer anchors. With 52 daily-bar symbols and limited history this will frequently NOT be met → honest `INSUFFICIENT_DATA` (the intended default, Part 0).
- **W_min = 6 walk-forward windows.** ≥6 non-overlapping purged/embargoed OOS folds give ≥5 degrees of freedom for a cross-window consistency check — the floor below which "the edge is stable through time" cannot be distinguished from a lucky single stretch. Achievable from a few years of daily bars under the §3.7 splitter; fewer windows is not enough time-diversity to judge stability strictly.
- **R_min = 3 distinct regimes in OOS.** The independent labeler (§3.4) defines 5 regimes {bull_calm, bull_vol, bear_calm, bear_vol, flat}. The pass condition (§4.2) already requires edge in **≥2** regimes, so the *representation* minimum must exceed the pass bar: requiring ≥3 distinct regimes present ensures the ≥2-regime test is assessed against a non-degenerate regime sample rather than fabricated from a market that only exhibited one or two moods over the OOS span.

Starting threshold values (frozen before sealed run, all logged): deflated Sharpe > 0 at 95%, cost-adjusted CAGR > 0, worst-quantile ruin < 1%, edge in ≥ 2 regimes or regime-scoped sizing, IS→OOS degradation within a sane band.

---

## PART 5 — DATA REALITY (honest note)

- **52 daily-bar symbols** on disk: enough to build and exercise the harness; **not** enough for high-confidence verdicts. Expect frequent honest `INSUFFICIENT_DATA`. Expanding history/intraday is a later data task, out of scope.
- **Universe provenance (V3):** the 52 symbols are (to be documented) CHAD's tradable universe — selected *with* knowledge of history → survivorship/selection bias. Every verdict flags this as a known **upward** bias; results are conditional on the universe.
- **Effective trade corpus:** currently empty of trusted trades (Stage 2 waits on the soak). Adapter built + tested as a stub now.
- **Bar quality (Phase 0):** unverified until audited — split/dividend adjustment, gaps, stale prints, and **FX provenance** (CAD/USD-mixed symbols checked against the canonical `USDCAD_CONVERSION_CONSTANT` discipline). Garbage bars are audited out *before* backtesting.

---

## PART 6 — BUILD PLAN (phases for `/goal`; each `/local-review`-gated, paused before commit)

- **Phase 0 — Bar-corpus forensic audit.** `bar_audit.py`: validate the 52-symbol corpus (adjustment, gaps, stale prints, FX provenance). Every later report embeds a data-quality section. *Nothing is backtested on unaudited bars.*
- **Phase 1 — Skeleton + scoring spine + isolation test.** `chad/validation/` package, `scoring_spine.py` with known-answer tests, and the transitive-import-closure test (nothing live imports the package; the package imports no live-loop/broker/runtime deps).
- **Phase 2 — Cost model + splits + independent regime labeler.** `cost_model.py`, `splits.py` (purged/embargoed walk-forward + per-head label horizon), `regime_labeler.py` (independent, tested). Anti-lookahead poison tests.
- **Phase 3 — Significance + ruin + pre-registered minimums.** `significance.py` (DSR w/ punitive N verified against published examples; block-bootstrap ruin; seed sweep), and commit the exact N_min/W_min/R_min into this doc.
- **Phase 4 — Feature-parity audit + backtest engine.** `feature_parity.py` FIRST (classify each head REPLAYABLE / NOT_REPLAYABLE), then `backtest_engine.py` + `data_provider.py` (pessimistic intrabar). Determinism test.
- **Phase 5 — Lockbox + config-freeze + verdict + report + CLI.** `oos_lockbox.py`, `config_freeze.py`, `verdict.py`, `report_writer.py`, `cli.py`. First honest end-to-end verdict on real data — run against train/val + decoy OOS only (the real OOS stays sealed until an operator-authorized `--final-run`).
- **Phase 6 — Trade-log adapter (Stage 2 seam).** `trade_log_adapter.py` + fixtures, same spine, same cost haircut, ready for the soak's real trades.

**Schedule note:** six institutional-grade phases through a paste-and-return loop is *not* a one-sitting job. No phase is rushed past its `/local-review` gate to hit a timeline. "This weekend" starts Phase 0-1; the rest proceeds at the pace correctness allows.

---

## PART 7 — WHAT THIS DOES FOR CHAD

When complete, CHAD has a **real, strict, self-honest gate to live money** — one that is *physically unable* to peek at its own test data, that charges every trade realistically, that judges market regimes with an independent ruler rather than its own (just-fixed) one, that discounts for every strategy ever tried, and that fails anything with a ruin path no matter how good its Sharpe. Until it renders a `PASS (candidate)` on genuine sealed out-of-sample data, `ready_for_live` stays locked — now earned by evidence, not asserted.

It is the most important thing left to build in CHAD, it is built entirely offline, and every one of its strongest guarantees is now a mechanism rather than a promise.

---

*End of design v1.1. No code before this is committed as the build SSOT. Phase 0 begins on operator GO.*
