# INCIDENT-0723 — Drill dry_run exhaust → false-flat guard → live re-entries

**Filed:** 2026-07-23 ~13:50Z (during incident; entries still retrying, held by adapter idempotency)
**Severity:** P1 (real paper-account position changes minted from a corrupted book)
**Class:** XOV-2345's disease through a new door — *fabricated-flat* instead of
*unreadable-flat*. Same house rule violated both times: **a rebuild treated
unverified rows as position truth.** XOV-2345: a dead shared `ib` cache read
byte-identical-to-flat. Today: rehearsal (`status=dry_run`) rows in the money
ledger read as real closes.

---

## 1. Causal chain (every step verified, citations inline)

1. **Drill runs wrote rehearsal rows into the money ledger.** The first live
   flatten drill (five aborted attempts 04:01–04:10Z, then the successful
   10:26Z run, `reports/ratification/PROOF_FLATTEN_DRILL_20260723.json`
   `overall=DRILL_COMPLETE`) submits its closes through
   `apply_close_intents` with a `dry_run=True` adapter
   (`scripts/flatten_all.py:933`, `:759`). `apply_close_intents`
   (`chad/core/position_reconciler.py:505-540`) writes fill evidence for
   **whatever status the adapter returns** — it has no equivalent of the hot
   path's `SKIP_EVIDENCE_UNCONFIRMED_ORDER_STATUS` guard. Result: 15
   `status=dry_run` rows in `data/fills/FILLS_20260723.ndjson`
   (seq 35–67 PSQ ×5 attempts; seq 375–379 IWM/MA/SVXY/UNH/V from the
   10:26 run).
2. **The trade closer ingested them as real fills.**
   `chad/execution/trade_closer.py:347`:
   `_TRUSTED_FILL_STATUSES = frozenset({"filled", "paper_fill", "dry_run"})`
   — `dry_run` is **explicitly blessed** in the FIFO ingest allowlist
   (`_extract_fill`, `:434-440`). The dry_run SELLs netted real open lots and
   minted **8 fake closed round-trips** into
   `data/trades/trade_history_20260723.ndjson` (§3.2), plus a **phantom short
   lot** `("g","PSQ") SELL 5` in `runtime/trade_closer_state.json` queues
   (citing dry_run fill_id `0be6fb53…`).
3. **The guard rebuild mirrored the corrupted queues.**
   `_rebuild_guard_from_paper_ledger` (`chad/core/live_loop.py:755-860`)
   faithfully mirrors trade_closer queues (GAP-028 Option B, by design):
   at 04:02:03 it minted `g|PSQ` open SELL 5 (guard `opened_at` matches); at
   10:26:54 — one rebuild cycle after the 10:26:04 drill — it closed
   `gamma|IWM`, `gamma|MA` (`closed_by=paper_ledger_rebuild`, `:852`) and the
   SVXY/UNH/V legs. **The entire CHAD-attributed book read flat while the
   broker held every share.** Drift v3 (`runtime/position_guard_drift.json`
   13:25:57Z) flagged it correctly: 5× `broker_untracked_position` + PSQ
   `qty_mismatch` (guard −5 vs broker +5).
4. **ctx sight showed empty → re-entries.** The W2B ContextBuilder position
   loader reads the guard (`CHAD_CTX_POSITIONS=on` on the live unit), so
   strategies saw `positions={}`. The D4 guardrail governs **sells only**
   (overlay = sole equity/ETF sell authority) — nothing gates buys. At the
   13:30Z open:
   - 13:30:37 — the reconciler **bought 5 PSQ for real** (`paper_fill`,
     fill `d9f5f894…` @26.61) to "close" the phantom `g|PSQ` short
     (its overnight attempts left 465 `market_closed` pseudo-rows, first at
     04:02:05 — 41s after drill attempt #1; contamination propagated in ONE
     live-loop cycle).
   - 13:31:06–09 — gamma **re-entered** SVXY +7, UNH +6, V +5 (`Filled`).
   - PSQ BUY 8 `Submitted` 13:31:04, **unfilled** (broker PSQ=10 at 13:43:37
     snapshot); retries `duplicate_blocked` — adapter idempotency is the only
     thing holding the line. GOOGL BUY 5 repeatedly `Cancelled`, no position.
   - ML veto saw it (`loss_prob=0.696 > 0.65`) but is **shadow-only** — did
     not block.

### Why the drill classified PSQ `[not_chad_attributed]` (the W4B-8 trigger)

The 10:26 drill was **right about the book it was shown**: by then
`gamma|PSQ` was (falsely) closed and `g|PSQ` was an open SELL opposing the
broker long, so `resolve_targets` correctly found no closable CHAD leg and
named the remainder `not_chad_attributed`. **The books were wrong, not the
drill.** The drill's own earlier exhaust corrupted the book it later read.

---

## 2. Consumer census — who reads FILLS rows, and their status treatment

| # | Consumer | Site | dry_run treatment | Verdict |
|---|----------|------|-------------------|---------|
| 1 | trade_closer FIFO ingest | `chad/execution/trade_closer.py:347,434-440` | **BLESSED** | **ROOT CAUSE — fix** |
| 2 | apply_close_intents evidence writer | `chad/core/position_reconciler.py:505-540` | writes any status (dry_run, market_closed, …) | **FIX (writer-side)** |
| 3 | Reconciler tier-1 close-price cascade | `chad/core/position_reconciler.py:285-296` | only rejects `_REJECTED_FILL_STATUSES` → dry_run/market_closed rows can price closes | **FIX** |
| 4 | Guard rebuild | `chad/core/live_loop.py:755-860` | mirrors queues (no direct read) | safe once #1 fixed (GAP-028 by design) |
| 5 | Guard close confirmation | `chad/core/position_guard.py:298-337` (`_CONFIRMED_FILL_STATUSES={"filled","paper_fill"}`) | rejected | SAFE ✓ (but see Q1) |
| 6 | Stage-2 harness adapter | `chad/validation/trade_log_adapter.py:95,413-415` (`{"paper_fill","fill"}`) | excluded | SAFE ✓ |
| 7 | Paper ledger watcher | `chad/portfolio/ibkr_paper_ledger_watcher.py:316` (`{"paper_fill","filled"}`) | excluded | SAFE ✓ |
| 8 | Dashboard API | `chad/dashboard/api.py:962` (`paper_fill` only) | excluded | SAFE ✓ |
| 9 | Options position monitor | `chad/portfolio/options_position_monitor.py:57` | excluded | SAFE ✓ |
| 10 | SCR (`trade_stats_engine`) + expectancy (`expectancy_tracker`) | read `trade_history_*` **derived by #1**; filter only quarantine/pnl_untrusted | **contaminated transitively** (§3.2) | fixed via #1 + quarantine manifest (PA) |
| 11 | profit_lock | `chad/risk/profit_lock.py:530` (excludes preview/paper_only/what_if + pnl_untrusted) | dry_run rows carry no realized pnl; closed-trade inputs come via #10 | covered by #1 + PA quarantine |

SCR/harness/expectancy exclusion is therefore proven **at the chokepoints**:
the harness has its own belt (row 6); SCR/expectancy are downstream of the
trade closer, whose ingest fix (row 1) is their protection; the fake rows
already minted need the quarantine manifest (PA).

---

## 3. Damage quantification

### 3.1 Real position changes (broker snapshot 13:43:37Z)

| Symbol | Fill (t, qty, px) | Notional | Broker now | Guard strategy-net now | Untracked delta |
|--------|-------------------|----------|------------|------------------------|-----------------|
| PSQ  | 13:30:37 BUY 5 @26.61 (phantom-short buyback) | $133.05 | 10 | 0 | **10** |
| SVXY | 13:31:06 BUY 7 @56.26 (re-entry) | $393.82 | 163 | 7 | **156** |
| UNH  | 13:31:07 BUY 6 @428.51 (re-entry) | $2,571.06 | 234 | 6 | **228** |
| V    | 13:31:09 BUY 5 @349.58 (re-entry) | $1,747.90 | 195 | 5 | **190** |
| IWM  | — (no re-entry yet) | — | 200 | 0 | **200** |
| MA   | — (no re-entry yet) | — | 10 | 0 | **10** |

**Unintended adds: +23 shares ≈ $4,845.83 notional.** All adds to existing
longs — no flips, no shorts created (clamps + idempotency held; PSQ BUY 8
still unfilled/duplicate_blocked). Excluded-symbol drift rows (AAPL/BAC/LLY/
MSFT/SPY `mixed_ownership_info`) predate this incident.

### 3.2 Fake realized PnL banked into trade_history (feeds SCR + expectancy)

| t (UTC) | strategy | sym | qty | fake pnl | fill_ids (entry→exit) |
|---------|----------|-----|-----|----------|------------------------|
| 04:01:34 | gamma | PSQ | 5 | +0.10 | 7c9c252a → **19638191 (dry_run)** |
| 10:26:30 | gamma | IWM | 200 | +100.00 | RECON_ADOPT_IWM → **f2558419 (dry_run)** |
| 10:26:30 | gamma | MA | 10 | −107.60 | 59e82e43 → **6161fc20 (dry_run)** |
| 10:26:30 | gamma | SVXY | 156 | −4.68 | RECON_ADOPT_SVXY → **7be4b358 (dry_run)** |
| 10:26:30 | gamma | UNH | 228 | +923.40 | PFF1_REATTR_UNH → **c9d9ee67 (dry_run)** |
| 10:26:30 | gamma | V | 181 | −964.73 | RECON_ADOPT_V → **92ff0e87 (dry_run)** |
| 10:26:30 | gamma | V | 9 | −24.30 | 7c7cc6b8 → **92ff0e87 (dry_run)** |
| 13:31:30 | g | PSQ | 5 | −0.00 | **0be6fb53 (dry_run)** → d9f5f894 |

**Net fake PnL ≈ −77.81** (individual distortions up to ±964.73). These rows
are NOT quarantined and currently count in SCR effective_trades/win_rate and
expectancy_state. **The ULTRA-CLOSE audit's warning — "seed-lot closes are
SCOREABLE and bank FAKE PnL into SCR/Stage-2" — happened today, verbatim.**

### 3.3 Ledger contamination (FILLS_20260723, 540 rows total)

`market_closed` 465 · `rejected` 46 · `dry_run` 15 · `paper_fill` 10 ·
`Filled` 3 · `PendingCancel` 1 → **530/540 rows (98%) are non-fill exhaust.**
The 465 `market_closed` rows are apply_close_intents spamming the phantom
`g|PSQ` BUY-back every live-loop cycle 04:02:05→13:29 (writer-side defect
D2 — systemic, not drill-specific).

---

## 4. Defect register

- **D1 (root)** — `trade_closer.py:347` blesses `dry_run` as a trusted money
  status. Fix: remove it; rehearsal rows must never enter FIFO netting.
- **D2** — `apply_close_intents` writes evidence for unconfirmed adapter
  statuses (dry_run/market_closed/…) into the money ledger; the hot-path
  executor already refuses these (`SKIP_EVIDENCE_UNCONFIRMED_ORDER_STATUS`).
  Fix: same refusal at this writer.
- **D3** — Reconciler tier-1 close-price cascade
  (`position_reconciler.py:285-296`) accepts dry_run/market_closed rows as
  "broker-confirmed" prices. Fix: require confirmed statuses.
- **D4** — **The EXIT_ONLY brake is decorative in paper.** (a)
  `chad-operator-intent-refresh.timer` fires `cmd_refresh`
  (`operator_intent_refresher.py:345-368`) every 10 min and unconditionally
  rewrites ALLOW_LIVE — the granted hold at 13:43:03Z was stomped at
  13:43:11Z. (b) No paper-lane component consults operator_intent at all
  (zero references in live_loop/orchestrator/ibkr_adapter). The flatten
  script's D5 post-flatten hold inherits both defects.
- **D5** — ML veto (loss_prob 0.696 > 0.65) is shadow-only; watched the
  re-entries happen.
- **D6** — Status canon misses titlecase `Filled`
  (`STATUS_CANON_UNMAPPED status='Filled'` ×2 per fill in live logs);
  raw `Filled` rows land in the ledger (3 today). Cosmetic for the closer
  (it lowercases) but breaks the PA-EP8 "single canonical status" invariant.
- **D7** — Harvester double-write: each real fill lands twice
  (executor `Filled` + harvester `paper_fill`, e.g. seq 534-536 vs 537-540;
  same mechanism gave PSQ two BUY 5 rows on 07-21 — which is why the FIFO
  had a second PSQ lot for the drill exhaust to net). Known EP-class issue,
  now with a demonstrated amplification path.

## 5. Open questions

- **Q1** — `gamma|PSQ` closed 04:10:36 stamped `closed_by=position_reconciler`
  + `closed_fill_id=19638191 (dry_run)`. The only writer of that stamp is
  `apply_close_intents` (`position_reconciler.py:602`), which is gated by
  `is_fill_confirmed` — and that rejects `dry_run` as currently written.
  Either the 5th drill invocation differed (operator was debugging;
  pre-retention, no journal) or an evidence-status mutation slipped through.
  Remediation forecloses every candidate path (D1+D2+D3), so this does not
  block; noted for honesty.
- **Q2** — Why the FIFO/guard carried a `("g","PSQ")` BUY leg *before* 04:01
  (the drill closed TWO PSQ legs on attempt #1: strategies `g` and `gamma`).
  `"g"` smells like `"gamma"` string-truncation; the one grep-hit
  (`live_loop.py:737`) is benign (tuple). Origin predates journal retention.

## 6. Immediate state / brake status

- EXIT_ONLY applied 13:43:03Z per operator GO → **stomped 13:43:11Z** by the
  auto-refresh timer (D4a); even if preserved, no paper-lane consumer (D4b).
  **Honest answer: the granted brake cannot be delivered by operator_intent
  as wired.** Bleed rate is slow (adds are 5-8 shares; PSQ-8 held by
  idempotency; GOOGL cancelling). The hard brake remains operator-side:
  `sudo systemctl stop chad-live-loop` (Channel 1) — stops entries AND the
  phantom-close machinery.
- SVXY/UNH/V guard entries have "healed" onto the small re-adds (open at
  7/6/5); the original 156/228/190 remain broker-untracked. IWM/MA/PSQ have
  zero open strategy legs vs broker 200/10/10.

## 7. Remediation & disposition

See `docs/PA_INCIDENT_0723_disposition.md` — all book repairs and
reconciliation runs are **operator-gated**; code fixes land as W4B-8 commits
(repo-side, activate at gated restart).
