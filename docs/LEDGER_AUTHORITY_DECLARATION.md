# Ledger Authority Declaration

**Status:** normative. **Landed:** 2026-07-24 (W6B-12, closing P2-11).
**Scope:** documentation only — this file declares an authority split that already
exists in code. No behaviour changed to produce it.

---

## The problem this closes

CHAD keeps **two** durable records of paper trading, written by different components on
different cadences, keyed differently, and consumed by disjoint sets of modules. Neither
is labelled as authoritative for anything in particular. P2-11 named the resulting hazard:
an operator or a future change can reach for whichever ledger is nearer to hand and get a
defensible-looking answer to a question that ledger cannot actually answer.

This file states which ledger answers which question, and what to do when they disagree.

---

## The two ledgers

| | **Position-authority ledger** | **Canonical closed-trade ledger** |
|---|---|---|
| Path | `runtime/ibkr_paper_ledger_state.json` | `data/trades/trade_history_YYYYMMDD.ndjson` |
| Shape | single JSON object, **hash-keyed** dict | append-only NDJSON, one row per closed trade |
| Grain | current open lots / position identity | completed round-trips |
| Written by | `chad/portfolio/ibkr_paper_ledger_watcher.py` | `chad/portfolio/ibkr_paper_trade_result_logger.py` |
| Answers | *"what do we hold right now, and under what lot identity?"* | *"what did we earn, and which strategy earned it?"* |
| Primary consumer | `chad/validators/position_authority.py:250` | 20+ modules (see below) |
| Integrity | rebuildable from broker truth | **hash-chained**; rows are immutable once written |

### Who reads the position-authority ledger

A small, position-shaped set — all of them asking "what is held":
`chad/validators/position_authority.py`, `chad/core/position_truth_engine.py`,
`chad/ops/lifecycle_truth_publisher.py`, `chad/ops/lifecycle_replay_drift_audit.py`,
`chad/ops/lifecycle_replay_coverage.py`, `ops/reconcile_positions.py`.

### Who reads the closed-trade ledger

A much larger, money-shaped set — all of them asking "what did it earn":
`tier_risk_enforcer`, `savage_allocator`, `dominance_allocator`, `edge_decay_monitor`,
`per_strategy_loss_guard`, `symbol_performance_blocker`, `fuse_box`, `dashboard/api`,
`metrics_server`, `daily_performance_report`, `weekly_investor_report`,
`clean_soak_evaluator`, `exterminator_sentinel`, and others.

**That asymmetry is the point.** Every risk gate and every performance number in the
system is downstream of the closed-trade ledger. The position-authority ledger feeds
reconciliation and nothing that sizes a trade.

---

## The declaration

### 1. Position identity → `ibkr_paper_ledger_state.json`

Questions about *what is currently held* — symbol, quantity, lot identity, cost basis of
an open lot — are answered by the position-authority ledger, cross-checked against broker
truth by the reconciliation publisher.

**It is not a P&L source.** It carries cost bases for open lots, and those cost bases are
exactly where fabricated/seed-lot values live (see the `UNATTRIBUTED_EPOCH3_ACCUMULATION`
rows EXS3 currently reports). Deriving realized P&L from it launders an untrusted number
into a trusted-looking one.

### 2. Realized P&L and strategy attribution → `trade_history_YYYYMMDD.ndjson`

Questions about *what was earned* — realized P&L, per-strategy attribution, win rate,
expectancy, edge decay, tier enforcement — are answered by the closed-trade ledger, and
**only** after the standard exclusion chain has been applied.

The exclusion order is not optional and is not re-implementable per consumer. It lives at
`chad/analytics/trade_stats_engine.py:658`:

```
warmup -> manual -> validate_only -> untrusted
```

A row carrying `pnl_untrusted` or `scoring_excluded` is **present in the file and excluded
from scoring**. Reading the ledger without the exclusion chain does not give you a
slightly-optimistic number; it gives you a number that includes fabricated cost bases.

### 3. They are not interchangeable, in either direction

- The position ledger cannot answer a P&L question — it has no notion of a completed
  round-trip.
- The closed-trade ledger cannot answer a position question — a symbol absent from it may
  be held and simply not yet closed, and a symbol present in it may have been re-entered
  since.

### 4. When they disagree

Disagreement between the two is **not** resolvable by preferring one ledger, because they
are not measuring the same thing. The resolution order is:

1. **Broker truth wins on positions.** Neither ledger outranks the broker. The drift
   detector (`chad/core/position_guard.py::detect_guard_vs_broker_truth_drift`) and
   `runtime/position_guard_drift.json` exist for exactly this, and disagreement is an
   operator event, not a computation to arbitrate.
2. **The closed-trade ledger wins on realized P&L**, because it is hash-chained and
   append-only: a row, once written, is immutable and independently verifiable. The
   position ledger is rebuilt from broker state and carries no such chain.
3. **A position/P&L contradiction is a reconciliation finding, never an averaging
   exercise.** Do not split the difference, and do not pick the one that looks healthier.

### 5. What this declaration does not do

It does not make either ledger *correct*. Both currently carry known contamination —
Bug-B futures rows, Epoch-3 seed lots with fabricated cost bases, the four `pnl_untrusted`
leaks EXS3 reports. This file governs **which ledger is asked which question**; the
trustworthiness of the rows inside is the exclusion chain's job, and it is separately
audited.

---

## For future changes

- A new consumer of realized P&L reads the **closed-trade ledger through the exclusion
  chain**. It does not read the position ledger, and it does not re-derive its own
  exclusion logic.
- A new consumer of position state reads the **position-authority ledger** and reconciles
  against broker truth. It does not infer holdings from closed trades.
- Any change that makes one ledger derive from the other must be raised as a Pending
  Action first. Collapsing the split would make the hash-chain guarantee unenforceable
  and is not a refactor.

---

## Evidence

| Claim | Source |
|---|---|
| Position ledger is hash-keyed, position-shaped | `chad/validators/position_authority.py:7,250` |
| Position ledger writer | `chad/portfolio/ibkr_paper_ledger_watcher.py:79` |
| Closed-trade ledger writer | `chad/portfolio/ibkr_paper_trade_result_logger.py` |
| Closed-trade ledger is hash-chained | W5A-6 amended-R2 record, `docs/CONTRACT_W5A_harness_handoff.md` |
| Exclusion order | `chad/analytics/trade_stats_engine.py:658` |
| Untrusted rows are present-but-excluded | `runtime/reports/EXTERMINATOR_SENTINEL_LATEST.json` → EXS3 evidence |
| Broker-truth drift path | `chad/core/position_guard.py`; `runtime/position_guard_drift.json` |
| Consumer split (6 vs 20+) | repo-wide search, 2026-07-24 |
