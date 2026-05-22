# BOX-047 (Official Matrix) — Dual ledger authority policy

- **Box number (Official Matrix):** 047
- **Box title (Official Matrix):** NEW-GAP-052 dual ledger confusion resolved — `ibkr_paper_ledger.json` vs `ibkr_paper_ledger_state.json` authority is declared
- **Stage:** Stage 3 — Engineering, tests, SSOT, and hidden-gap closure
- **Cut timestamp (UTC):** 2026-05-20T19:16:57Z
- **HEAD at cut:** `bbe7525` (short) — "GAP-039 (Phase-58/59): relocate stop-bus evaluate before early-return"
- **Branch:** `main`

> **Numbering disambiguation:** Official Matrix Box 047 ("NEW-GAP-052
> dual ledger confusion resolved") is distinct from the Supplemental
> Annex Box 047 ("TEST_BASELINE_FULL_GREEN_OR_CLASSIFIED") closed
> earlier in this run.

---

## 0. Scope and safety statement

- **CHAD remains PAPER.** `CHAD_EXECUTION_MODE=paper`.
- **live trading not authorized.** This policy does not flip
  `ready_for_live` and does not authorize live trading.
- **No runtime mutation.** No ledger file is modified by this policy
  declaration.

---

## 1. The two files (factual inventory, read-only)

### 1.1 `runtime/ibkr_paper_ledger.json` — **CONFIG**

```
-rw-r--r-- 375 bytes  mtime 2026-05-08 15:34 UTC  (world-readable)
```

Top-level keys (verified by `python3 -c "json.load(open(...))"`):

```json
{
  "enabled":                 true,
  "mode":                    "paper",
  "positions_snapshot_path": "/home/ubuntu/chad_finale/runtime/positions_snapshot.json",
  "preview_dir":             "/home/ubuntu/chad_finale/reports/ledger",
  "state_path":              "/home/ubuntu/chad_finale/runtime/ibkr_paper_ledger_state.json",
  "ttl_seconds":             300,
  "ibkr":                    { ... }
}
```

**This is the watcher service's configuration file.** It contains
**no position data**. The `state_path` field tells the watcher where
to read/write the actual state.

### 1.2 `runtime/ibkr_paper_ledger_state.json` — **STATE**

```
-rw------- 11370 bytes  mtime 2026-05-20 12:39 UTC  (operator-only — mode 0600)
```

Top-level schema: dict keyed by 64-char SHA256 record hash; each
value is a dict with `{account_id, attribution_source, avg_cost,
conId, currency, opened_at_utc, plan_now_iso, plan_path, qty,
secType, source_strategies, strategy, symbol, tags}`.

At audit time: 22 hash-keyed records.

**This is CHAD's internal "belief file" of currently-open paper
positions.** It is rewritten atomically every watcher cycle by
`OpenStateStore.save()`.

### 1.3 Append-only audit / event ledgers (separate authority)

| File pattern                                                  | Role                                                              |
| ------------------------------------------------------------- | ----------------------------------------------------------------- |
| `data/fills/FILLS_YYYYMMDD.ndjson`                            | append-only **fill events** (one row per executed fill)            |
| `data/broker_events/BROKER_EVENTS_IBKR_YYYYMMDD.ndjson`        | append-only **broker callback stream** (heartbeats + lifecycle)    |
| `data/trades/trade_history_YYYYMMDD.ndjson`                   | append-only **closed-trade history** (PnL rows)                    |

One file per UTC day; never overwritten; primary audit-history source.

### 1.4 Per-order lifecycle state machine (separate authority — different file)

| File                                                          | Role                                                              |
| ------------------------------------------------------------- | ----------------------------------------------------------------- |
| `runtime/ibkr_adapter_state.sqlite3` (table `ibkr_exec_state`) | PendingSubmit / Submitted / Filled / etc lifecycle state machine; retention via `ops/sqlite_retention.py` (Box 049 Patchset O). |

Out of scope for this box but called out so no operator confuses it
with the two `*_ledger*.json` files.

---

## 2. Producers / consumers (grep-verified)

### 2.1 `runtime/ibkr_paper_ledger.json` (CONFIG)

| Code path                                                                                                | Role     |
| -------------------------------------------------------------------------------------------------------- | -------- |
| `config/runtime_seed/ibkr_paper_ledger.json`                                                              | seed source for initial deployment (operator copies into runtime/)  |
| `chad/portfolio/ibkr_paper_ledger_watcher.py:76` (`CONFIG_PATH_DEFAULT`)                                  | reader (loads at service start; resolves `state_path` etc.)         |
| `chad/portfolio/ibkr_paper_ledger_watcher.py` (LedgerConfig dataclass)                                    | parser (`enabled`, `mode`, `state_path`, `ttl_seconds`, …)         |
| `chad/execution/ibkr_client_ids.py:144`                                                                   | comment-reference ("LedgerConfig.load() reads in runtime/ibkr_paper_ledger.json") |
| `audit_chad.sh`                                                                                          | audit script (presence check)                                       |
| `deploy/drop-ins/chad-ibkr-paper-ledger-watcher.service.d_override.conf`                                  | systemd unit reference                                              |
| `chad/tests/test_ibkr_paper_ledger_watcher.py`                                                            | test fixtures                                                       |

**No code path mutates this file at runtime** — it is operator-managed
and seeded from `config/runtime_seed/`.

### 2.2 `runtime/ibkr_paper_ledger_state.json` (STATE)

**Writers:**

| Code path                                                                                                | Notes                                                                                          |
| -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `chad/portfolio/ibkr_paper_ledger_watcher.py:656` (`OpenStateStore.save()` → `atomic_write_json`)        | **PRIMARY WRITER.** Atomic write on every watcher cycle.                                       |
| `ops/reconcile_repair_ibkr_ledger_state.py`                                                              | **OPERATOR-ONLY SURGICAL REPAIR** (Phase 12). Reads broker snapshot, repairs `state.open[*].qty` to match. Fail-closed; emits immutable report under `reports/reconciliation/`. |
| `ops/bin/chad_paper_trade_executor.py`                                                                    | secondary writer in the paper-trade executor lane                                              |

**Readers:**

| Code path                                                                                                | Role                                                                                                                                  |
| -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `chad/ops/lifecycle_truth_publisher.py:478` (`_normalize_ledger_open_records`)                            | normalises both flat-dict and `{"open": {...}}` shapes; computes `ledger_state_positions_count` and publishes into `runtime/trade_lifecycle_state.json`. |
| `chad/ops/lifecycle_replay_drift_audit.py`                                                               | drift audit between replay and ledger state.                                                                                          |
| `chad/ops/lifecycle_replay_coverage.py`                                                                   | replay-coverage check.                                                                                                                |
| `ops/reconcile_positions.py` (`chad_state` = `"ibkr_paper_ledger_state.json"`)                            | feeds `runtime/reconciliation_state.json` (CHAD state vs broker truth).                                                                |
| `chad/tests/test_positions_truth_classifier.py`                                                          | tests pin both flat and wrapped schemas (lines 56, 218, 229, 279, 284, 297, 302, 314, 319).                                            |

**No other code mutates this file.**

### 2.3 Append-only NDJSON files (`data/fills`, `data/broker_events`, `data/trades`)

Produced by the orchestrator / IBKR adapter on every event. Consumed
by `lifecycle_truth_publisher`, daily ops reports, audit tooling.
Never overwritten — date-stamped one file per UTC day.

---

## 3. Canonical authority policy (declarable)

| Concern                                                                              | Canonical file                                                                                  | NOT canonical for this concern                                                                                                       |
| ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Paper-watcher service configuration (`enabled`, `mode`, `state_path`, `ttl_seconds`, …) | **`runtime/ibkr_paper_ledger.json`** (config)                                                  | Not `…_state.json` (no config fields).                                                                                                |
| Current paper-account open-position belief (qty / avg_cost / strategy / opened_at_utc / hash-keyed records) | **`runtime/ibkr_paper_ledger_state.json`** (state)                                              | Not `…ledger.json` (no position fields). Not the NDJSON files (those are append-only history, not derived current-state).            |
| Append-only fill events                                                              | **`data/fills/FILLS_YYYYMMDD.ndjson`**                                                            | Not `…_state.json` (overwritten atomically; cannot serve as audit trail).                                                              |
| Append-only broker event stream                                                       | **`data/broker_events/BROKER_EVENTS_IBKR_YYYYMMDD.ndjson`**                                      | Not `…_state.json`.                                                                                                                  |
| Append-only closed-trade history                                                       | **`data/trades/trade_history_YYYYMMDD.ndjson`**                                                   | Not `…_state.json`.                                                                                                                  |
| Per-order lifecycle state machine                                                     | **`runtime/ibkr_adapter_state.sqlite3` (table `ibkr_exec_state`)**                                | Not either `*_ledger*.json`. Separate retention via `ops/sqlite_retention.py`.                                                       |
| Broker-truth current positions                                                        | **`runtime/positions_snapshot.json`** (broker-side)                                              | Not `…_state.json` (CHAD belief, not broker truth). Cross-reference: Box 046 retraction.                                              |
| Reconciliation truth (CHAD belief vs broker)                                          | **`runtime/reconciliation_state.json`** (derived); if `status: RED` with `broker_source: "unavailable:"`, answer is **UNKNOWN / requires audit**. | Not either ledger file alone — reconciliation derives from both.                                                                       |

### 3.1 Hard rules

1. **NEVER** treat `runtime/ibkr_paper_ledger.json` (config) as a
   position-state source. It has no `qty` / `avg_cost` / `symbol` /
   `strategy` fields — it has paths, mode flags, and a TTL.
2. **NEVER** treat `runtime/ibkr_paper_ledger_state.json` as an
   append-only audit trail. It is overwritten atomically every
   watcher cycle. For audit history, use the `data/*.ndjson` files
   or the SQLite `ibkr_exec_state` table.
3. **NEVER** manually edit either file outside operator-approved
   tools. The only sanctioned non-watcher mutator of `…_state.json`
   is `ops/reconcile_repair_ibkr_ledger_state.py` (Phase 12 surgical
   repair, fail-closed, emits immutable report).
4. **Reconciliation reads** `runtime/positions_snapshot.json`
   (broker truth) vs `runtime/ibkr_paper_ledger_state.json` (CHAD
   belief) — never the config file.
5. **Lifecycle truth publisher reads** `…_state.json` for `open`
   records (flat or `{"open": {...}}` shape; both supported) — never
   the config file.
6. **When `reconciliation_state.status: RED` with `broker_source: "unavailable:"`,
   do NOT substitute either `…_state.json` or `positions_snapshot.json`
   as current truth in isolation.** The right answer is **UNKNOWN /
   requires audit** (cross-reference: Box 046 retraction).

### 3.2 Operator quick-reference

> **"Where do I look for ____?"**
>
> - **"Is the paper-ledger watcher enabled / what mode is it in?"** → `runtime/ibkr_paper_ledger.json` (CONFIG).
> - **"What positions does CHAD currently believe it has open in paper?"** → `runtime/ibkr_paper_ledger_state.json` (STATE).
> - **"What fills happened today?"** → `data/fills/FILLS_YYYYMMDD.ndjson`.
> - **"Did broker send us a fill / heartbeat / lifecycle event?"** → `data/broker_events/BROKER_EVENTS_IBKR_YYYYMMDD.ndjson`.
> - **"Which trades closed today and with what PnL?"** → `data/trades/trade_history_YYYYMMDD.ndjson`.
> - **"Where is the per-order state machine?"** → `runtime/ibkr_adapter_state.sqlite3` table `ibkr_exec_state`.
> - **"What does the broker say my positions are right now?"** → `runtime/positions_snapshot.json` (broker truth).
> - **"Are CHAD's beliefs reconciled with broker truth?"** → `runtime/reconciliation_state.json`. If status=RED with broker_source unavailable → UNKNOWN / requires audit.

---

## 4. Stale / conflicting claims scan

Method: grep / inspect every code/doc reference to `ibkr_paper_ledger`
or `ibkr_paper_ledger_state` for confusion or conflation.

| Reference                                                                                                | Status                                                                                                                                                                                              |
| -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `chad/portfolio/ibkr_paper_ledger_watcher.py` docstring (line 39: "expects a runtime config file, by default: /home/ubuntu/chad_finale/runtime/ibkr_paper_ledger.json") | **Correct** — labels the file as "config", not as state. Aligns with §3.                                                                                                                            |
| `chad/portfolio/ibkr_paper_ledger_watcher.py:76` (`CONFIG_PATH_DEFAULT`)                                  | **Correct** — variable name explicitly says "CONFIG".                                                                                                                                              |
| `chad/portfolio/ibkr_paper_ledger_watcher.py:79` (`DEFAULT_STATE_PATH`)                                    | **Correct** — variable name explicitly says "STATE".                                                                                                                                               |
| `chad/execution/ibkr_client_ids.py:144` ("LedgerConfig.load() reads in runtime/ibkr_paper_ledger.json")   | **Correct** — labels the file as the LedgerConfig source.                                                                                                                                          |
| `chad/ops/lifecycle_truth_publisher.py:26` ("runtime/ibkr_paper_ledger_state.json (optional hints)")     | **Correct** — labels as the state file, and as a hint source (not the canonical-truth source for everything).                                                                                       |
| `chad/ops/lifecycle_truth_publisher.py:111-126` (`_normalize_ledger_open_records`)                       | **Correct** — handles both flat-dict and `{"open": {...}}` shapes; explicitly tolerates either.                                                                                                    |
| `ops/reconcile_positions.py:10-19` (docstring "Reads ONLY: runtime/positions_snapshot.json, runtime/ibkr_paper_ledger_state.json") | **Correct** — explicitly names the two reconciliation inputs; never reads the config file.                                                                                                          |
| `ops/reconcile_repair_ibkr_ledger_state.py` header                                                       | **Correct** — explicitly describes its scope as "Ledger State Repair" against the broker snapshot; fail-closed posture documented.                                                                  |
| `ops/pending_actions/GAP-027_mes_paper_ledger_stale_position.md`                                          | **Correct** — refers to `runtime/ibkr_paper_ledger_state.json` (the state file). No confusion with the config file.                                                                                  |
| `config/runtime_seed/ibkr_paper_ledger.json`                                                              | **Correct** — pure config; no position fields.                                                                                                                                                      |
| `chad/tests/test_positions_truth_classifier.py` (write/read of `ibkr_paper_ledger_state.json`)            | **Correct** — tests pin the state schema (both flat and wrapped shapes), not the config schema.                                                                                                    |
| `chad/tests/test_ibkr_paper_ledger_watcher.py`                                                            | **Correct** — tests the watcher (which itself touches both files appropriately).                                                                                                                    |

**No conflicting or stale claim was found.** The two files are
already correctly distinguished in code (via variable names
`CONFIG_PATH_DEFAULT` vs `DEFAULT_STATE_PATH`) and in docstrings.
What was missing was an operator-facing policy document — this doc
fills that gap.

`conflicting_claims_remaining: false`.

---

## 5. Patches summary

| Patch class            | Action                                                                                                                  |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Production code        | **None** — code already correctly distinguishes config vs state (`CONFIG_PATH_DEFAULT` vs `DEFAULT_STATE_PATH` in the watcher). No code change required by this policy. |
| Live config            | **None** — no `runtime/*.json` mutated.                                                                                  |
| Documentation          | **Added** `ops/pending_actions/BOX-047_dual_ledger_authority_policy.md` (this file).                                     |
| Evidence               | **Added** `runtime/completion_matrix_evidence/BOX-047_OFFICIAL_dual_ledger_confusion_resolved.md` (paired with this doc). |
| Tests                  | **None added** — `chad/tests/test_ibkr_paper_ledger_watcher.py` and `chad/tests/test_positions_truth_classifier.py` already pin the schemas. Adding a static test that re-asserts §3 wording would not detect any regression that the existing tests do not already catch. |
| Frozen historical SSOT | **Unchanged** — forward-only.                                                                                            |
| Staged / committed     | **None** — no `git add` / `commit` / `push` / `rm`.                                                                       |

---

## 6. False-closure guardrails

This policy does NOT:

- claim CHAD is complete.
- claim CHAD is live-ready.
- authorize live trading.
- mutate any ledger file (config or state).
- assert current paper positions. (At audit time
  `reconciliation_state.status: RED` and `broker_source: "unavailable:"`
  — the only safe current-position answer is **UNKNOWN / requires
  audit** per the policy itself §3.1 rule 6 + Box 046 retraction.)

**live trading not authorized. CHAD remains PAPER.**
