# PENDING ACTION — Crypto exploration mode (paper-epoch-only), UC1/U2-B

- **Filed:** 2026-07-17
- **Type:** Pending Action — config change **PREPARED, NOT APPLIED**
- **Status:** PROPOSED. Awaiting typed operator GO. Per CLAUDE.md §3, risk caps / strategy
  config are never mutated directly.
- **Author:** UC1 (ULTRA-CLOSE)
- **Companion:** `docs/ULTRA_CLOSE_AUDIT_2026-07-17.md`; exit path shipped in `15d1c04`
  (UC1-2, shadow).

---

## 0. What is being proposed (and what is NOT)

**Proposed:** add `alpha_crypto` to the regime rosters that currently exclude it, behind a new
paper-only flag `CHAD_CRYPTO_EXPLORATION` (**default OFF**), and tag every fill with the regime
that was live when it was produced, so the edge harness can later slice crypto edge **by
regime**.

**NOT proposed here:** flipping the crypto exit overlay to ACTIVE (separate PA); deploying
anything; wiring the crypto overlay into `live_loop`; any live-trading change whatsoever.

**Ordering is deliberate and non-negotiable: the EXIT PATH LANDS FIRST.** Exploration without a
deterministic exit is how CHAD acquired the inventory the exit audit was written about. The exit
overlay (`15d1c04`) exists before exploration is even proposed, and exploration must not be
enabled until that overlay is at minimum evaluating the crypto book in shadow.

---

## 1. Rationale — exploration is how CHAD learns *when* crypto works

The current position is an **assumption**, not a finding.

`config/regime_activation_matrix.json` (`schema_version=regime_activation_matrix.v1`,
`{"regimes": {<regime>: [<strategy>, ...]}}`):

| Regime | `alpha_crypto`? |
|---|---|
| `trending_bull` | ✅ |
| `trending_bear` | ✅ |
| `ranging` | ❌ |
| `volatile` | ✅ |
| `unknown` | ✅ |
| `adverse` | ❌ (empty roster — silences everything) |

**The live regime is `ranging`** (`runtime/regime_state.json`: `regime="ranging"`,
`previous_regime="ranging"`, `confidence=0.618`) — the **one non-adverse regime that excludes
`alpha_crypto`**. This, and not a halt, is why crypto is silent: `runtime/last_route_decision.json`
shows `alpha_crypto: "no_signal"`, `available_strategies.alpha_crypto: 0`.

*(Correcting the record: the `alpha_crypto` edge_decay halt was **CLEARED** by the operator on
2026-07-15T14:59:10Z — `runtime/strategy_allocations.json`: `halted=false`,
`clear_reason="stale halt from validate_only era, per CTF-T3 forensic"`. `is_strategy_halted('alpha_crypto')`
returns False. The halt is not the blocker.)*

**Why the `ranging` exclusion is an assumption:** git archaeology shows no evidence file, no
backtest, and no PA justifying crypto's removal from the `ranging` roster. Meanwhile the entire
crypto sample is unusable as evidence — all 2,148 legacy `alpha_crypto` fills are
`validate_only` with `pnl=0.0`, which is exactly why the CRYPTO-TRUST engine (`7e9d982`,
deployed) was built. **CHAD has never had a single scoreable crypto round-trip.** So the claim
"crypto doesn't work when ranging" is not a finding CHAD made — it is a guess CHAD inherited,
and the current config guarantees it can never be tested, because `ranging` has been the live
regime continuously.

**Four reasons this is the right time:**

1. **Paper is cheap tuition.** Worst case is bounded by the wallet (§3), and the lesson is
   permanent.
2. **The harness needs cross-regime samples.** Edge Harness Phase 2 ships a `regime_labeler`
   and Phase 3 ships DSR significance with pre-registered minimums (`N_min=30`, `W_min=6`,
   `R_min=3`). **`R_min=3` cannot be met by a strategy that only ever trades in 3 regimes it
   was pre-assigned.** Without cross-regime samples the harness can never say anything about
   crypto at all — not "it works", not "it doesn't".
3. **The evidence is now trustworthy for the first time.** The trusted-fill engine marks against
   live ticks, charges the real taker fee, realizes FIFO PnL, and stamps
   `provenance=SIMULATED_AGAINST_LIVE_TICKS` / `trust_state=TRUSTED`, deliberately omitting
   `validate_only`/`pnl_untrusted` so Stage-2 admits. Verified: `trust_exclusion(record) is
   None` (test in `15d1c04`). A crypto fill produced today is *scoreable*; one produced before
   07-12 was not.
4. **The exit exists now.** Exploration that cannot exit is accumulation.

**The falsifiable claim being tested:** *"`alpha_crypto` has no edge in `ranging`."* Today that
is unfalsifiable by construction. This PA makes it testable, on paper, for ≤$185.

---

## 2. The change (PREPARED — apply ONLY on typed GO)

### 2a. `config/regime_activation_matrix.json` — add `alpha_crypto` to the two rosters that lack it

```diff
   "ranging": [
     "beta", "delta_pairs", "gamma_reversion", "gamma", "omega_macro"
+    , "alpha_crypto"
   ],
```

`adverse` stays **empty** — an empty roster is a deliberate global silence, and exploration must
never override a risk-off state. **Only `ranging` changes.**

### 2b. Gate: `CHAD_CRYPTO_EXPLORATION` (default OFF)

The roster edit alone must not arm anything. `filter_intents_by_regime`
(`chad/portfolio/regime_activation.py:95`, called at `chad/core/live_loop.py:2495-2500` for both
lanes) consults the matrix; the flag gates whether the **exploration-added** entry is honored:

- Flag **unset/0** (default): exploration roster entries are dropped exactly as today. The
  matrix edit is **inert**. This is the rollback state.
- Flag **1** + `CHAD_EXECUTION_MODE=paper`: exploration entries are honored, and every resulting
  intent carries `exploration=true`.
- Flag **1** + mode ≠ paper: **fail-closed, refuse, and log loudly.** See §5.

### 2c. Per-fill regime tagging (the point of the exercise)

**The field already exists and is universally unpopulated.** `PaperExecEvidence`
(`chad/execution/paper_exec_evidence_writer.py:442`) declares `"regime"` in `__slots__` (`:470`)
and defaults it at `:518`:

```python
self.regime = _safe_str(kwargs.pop("regime", "paper"), "paper")
```

Grepping `regime=` across `chad/execution/` and `chad/core/` for evidence/trade-result writers
returns **zero callers** — **no fill row anywhere carries a real regime; every row says
`"paper"`.** The trusted engine is no exception: `_build_evidence_kwargs` (`:485`) omits
`regime` entirely, and `_build_trade_history_kwargs` (`:543`) **hardcodes `regime="paper"`**.

**Proposed (option (a) — preferred):** thread `regime` onto the intent at `live_loop.py:2497`,
where `read_regime_state()` has *already* been called, and have `process_intent` forward
`getattr(intent, "regime", ...)` into both `_build_evidence_kwargs` and
`_build_trade_history_kwargs`. This keeps the engine pure and matches how `markers` already
flow. *(Rejected option (b): read regime inside the `_build_*` kwargs — simpler, but adds
hidden I/O to a pure-ish core.)*

Without this, exploration produces fills that **cannot be sliced by regime** — which is the
entire justification for exploration. **2c is not optional; it is the deliverable.** 2a/2b
without 2c is just "trade more crypto".

---

## 3. Worst case is bounded — $185, and the existing gates already do the work

Per U2-C, the sizing path is **verified, not rebuilt**:

- **Wallet:** ~$185. Kraken paper leg is native CAD (`kraken_cad_equity_fix`).
- **Min-size gate:** `chad/execution/kraken_min_size.py:60` `decide_min_size(...) ->
  MinSizeDecision` — PASS/BUMP/SKIP, markers `CRYPTO_MIN_SIZE_BUMP` / `CRYPTO_BELOW_MIN_SKIP`,
  wired at the `execution_pipeline.py` sizing chokepoint. A sub-minimum order is SKIPPED, not
  rounded up into a position the wallet cannot carry.
- **SCR sizing:** `sizing_factor=0.1` at WARMUP (`runtime/scr_state.json` @ 13:34:12Z) — every
  exploration order is sized at **one tenth** while SCR is in WARMUP.
- **Margin shadow gate:** the Kraken BP/margin chokepoint observes (never blocks) and writes
  evidence to `data/margin_shadow/`.
- **Exit:** the crypto overlay (`15d1c04`) bounds holding period at 4 days and loss at 12% of a
  **real** cost basis — once it is wired and armed.

**Worst realistic case:** the wallet. ~$185 of paper capital, on a leg whose fills are
simulated against live ticks. There is no live crypto money at risk in any branch of this PA.

**⚠️ Precondition — the book is polluted right now.** `kraken_trusted_lots` holds **5 stale
rows: 12.5 SOL long @ ~$76.62 ≈ $958 notional** — against a $185 wallet. Four are byte-identical
at `2026-07-12T22:04:18Z`, minted between commits `7e9d982` (21:51) and `5f7aa3a` (22:11): a
dev/manual exercise during the build, not a real position and not a test leak (every test
injects `db_path=tmp_path`). Live SOL is 73.61, so the book is ~$36 underwater and **5 days
old** — it would trip the crypto overlay's 4-day max-hold on the first evaluation.

**These 5 rows must be purged (or the store repointed) before exploration OR the crypto overlay
is armed**, or CHAD's first crypto "exit" will close a position that never economically existed.
Purging is a runtime mutation and is therefore **its own Pending Action** — filed here as a
blocker, not performed. *(Related hardening, also not performed: `TrustedFillEngine.__init__:384`
falls back to the real `runtime/exec_state_paper.sqlite3` with **no `_under_active_pytest()`
guard**, unlike `build_default_overlay:945-971`. Any future test that forgets `book=` writes to
production.)*

---

## 4. Rollback

**`CHAD_CRYPTO_EXPLORATION=0` (or unset).** That is the whole procedure — the flag is the
rollback, and the default is already the rolled-back state.

- Roster entries added under exploration are **inert** whenever the flag is off, so the config
  edit needs no revert to make the system safe.
- Full revert if desired: `git revert` the matrix commit.
- Nothing about this PA is irreversible: no live orders, no broker state, no schema migration.
- Positions opened during exploration are **not** rolled back by the flag — they are closed by
  the crypto exit overlay (or, absent it, by an operator flatten). *This is the reason the exit
  path lands first.*

---

## 5. Pre-registered criterion for ever carrying exploration into live

# **NONE.**

**Exploration is explicitly a paper-epoch-only device. It dies at live. There is no criterion,
no threshold, and no sample size that promotes it.**

This is not a placeholder for a number to be filled in later. It is the design:

- Exploration deliberately trades a roster CHAD believes to be **wrong**, to find out whether the
  belief is true. That is a **measurement instrument**, not a strategy. The value is the
  *evidence*, never the *P&L*.
- What exploration *may* produce is a **finding** — e.g. "`alpha_crypto` shows DSR-significant
  edge in `ranging` over N≥30 trades, W≥6 windows, R≥3 regimes". That finding is then argued in
  **its own PA**, on its own evidence, against the live-promotion checklist. **The roster would
  be changed permanently and exploration would still be OFF.** Exploration is the road to the
  conclusion; it is never the vehicle that carries it into live.
- **Fail-closed enforcement, not just documentation.** The flag must refuse to arm outside
  paper: if `CHAD_CRYPTO_EXPLORATION=1` while `CHAD_EXECUTION_MODE != paper`, the gate **refuses,
  logs `CRYPTO_EXPLORATION_REFUSED_NON_PAPER` loudly, and drops the exploration entries** — it
  does not silently downgrade. LiveGate must treat an armed exploration flag at a live-posture
  transition as a **blocking** condition.
- **The paper epoch bounds it.** Exploration is scoped to Epoch-3. It does not survive an epoch
  reset by default.

**If a future reader is looking for the criterion that promotes exploration to live: there isn't
one, and that is the point. Add a finding, not a flag.**

---

## 6. Verification before GO

1. Purge/repoint the 5 stale SOL lots (§3) — **own PA, must land first**.
2. Crypto exit overlay wired into the hot path and evaluating the crypto book **in shadow**,
   with ≥1 session of evidence.
3. `py_compile` + full `pytest` (vs known-5) + `CHAD_SKIP_IB_CONNECT=1 full_cycle_preview` clean.
4. Confirm `CHAD_CRYPTO_EXPLORATION` unset ⇒ routing decisions are **byte-identical** to today
   (the matrix edit is provably inert).
5. Confirm a fill produced with the flag ON carries a real `regime` (not `"paper"`) and is
   Stage-2 admissible.
6. Re-read `runtime/regime_state.json` at GO time — if the regime has left `ranging`, the
   exploration edit changes nothing and the GO should be re-scoped.

---

## 7. Files cited

- `config/regime_activation_matrix.json`; `chad/portfolio/regime_activation.py:95`;
  `chad/core/live_loop.py:2439,2495-2500`
- `runtime/regime_state.json`; `runtime/last_route_decision.json`;
  `runtime/strategy_allocations.json`; `runtime/scr_state.json`
- `chad/execution/paper_exec_evidence_writer.py:442,470,518`
- `chad/core/kraken_trusted_fill_engine.py:240-268,370-384,408,485,527,543`
- `chad/execution/kraken_min_size.py:60`; `chad/core/kraken_execution.py:42-46,504-532`
- `chad/validation/trade_log_adapter.py:261-294`
- `chad/risk/crypto_exit_overlay.py`; `config/crypto_exit_overlay.json` (UC1-2, `15d1c04`)
