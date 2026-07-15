# INSTALL — Exterminator Sentinel Stage 1 (chad-exterminator-sentinel)

**Status:** Repo-only design artifacts. Nothing here is installed, enabled,
reloaded or started automatically. Per CLAUDE.md governance rules #6/#7 the
agent never installs unit files or starts services — the operator runs the
commands below during a maintenance window, gated by
`ops/pending_actions/EXTERMINATOR_stage1_deploy.md`.

## What it is

`chad/ops/exterminator_sentinel.py` — a pure read-only scanner. It runs 8
checks, each returning `ok | warn | fail` with the evidence behind the verdict,
and writes:

| Path | Shape |
|---|---|
| `runtime/reports/EXTERMINATOR_SENTINEL_LATEST.json` | full report, `exterminator_sentinel.v1`, rewritten each run |
| `runtime/reports/EXTERMINATOR_SENTINEL_HISTORY.ndjson` | one compact verdict line appended per run |

Those two paths are the **only** writes the module is permitted to make. It
does not mutate runtime state, call brokers, restart services, stage or commit,
or auto-heal anything it finds. A coach-voiced Telegram summary is sent **only
when a check fails** (`NOTIFY_ONLY`), deduped per the CTF-T2 identity rule.

## Relationship to the old scanner

This supersedes `chad/ops/exterminator.py` (`exterminator_report.v1`), which
last ran **2026-05-05** against Epoch 2, has **zero production importers**, and
was never scheduled. That module is left in place — `test_pr09_position_truth_
contract.py` imports it — and is not wired to any timer. Four of its checks
were provably blind:

| Old check | Why it was blind |
|---|---|
| `EX006` placeholder fills | probed `runtime/paper_fills_today.ndjson`, a path that has never existed; and tested `fill_price == 100.0`, a value the evidence writer deliberately zeroes (`paper_exec_evidence_writer.py:1952`) |
| `EX009` reconciliation | read `reconciliation_state.status`, which in paper mode is a `broker_sync`-only self-comparison |
| `EX012` ML | grepped logs for `ML_SHADOW`; zero hits exist |
| `EX013` schema | checked key *presence* only, not pinned versions or required keys |

## The 8 checks

| ID | Name | Fails when |
|---|---|---|
| EXS1 | stale_feeds | a feed with a **verified** TTL breaches `fail_after_seconds`; also carries the R14 blind cross-check |
| EXS2 | placeholder_fills | a placeholder-priced fill was written **without** the untrusted/rejected demotion |
| EXS3 | untrusted_fills | `validate_only`/`pnl_untrusted` rows sit in a **scored** store |
| EXS4 | reconciliation_drift | the guard disagrees with the **independent** collector (see below) |
| EXS5 | failed_services | a `chad-*` unit is in the failed state |
| EXS6 | dirty_git | uncommitted tracked changes in production paths (warn) |
| EXS7 | schema_breaks | a runtime contract violates its pinned `schema_version` or lacks a required key |
| EXS8 | ml_anomalies | never fails — emits `warn no_baseline` honestly (see below) |

### EXS4 — the independent-leg rule (the lesson of XOV-2345)

Position truth has three legs and **only one is independent**:

| Leg | Source | Independent? |
|---|---|---|
| `reconciliation_state.json` | IBKR clientId=83 vs the guard's `broker_sync` rows — and in paper mode filters to `strategy == "broker_sync"` (`reconciliation_publisher.py:166`) | scoped-blind |
| `position_guard_drift.json` | the guard's `broker_sync` rows vs the guard's strategy rows — **both legs read from `position_guard.json`** | **no** — compares that file to itself |
| `positions_snapshot.json` | clientId=99, separate process, own `IB()`, own timer (`ibkr_portfolio_collector_v2.py:125`) | **yes** |

When a dead shared `ib` cache false-flatted the guard, the first two legs agreed
that all was well because they were reading the same corrupted source, and
reconciliation stayed **GREEN through a whole-book false-flat**. So EXS4
compares the guard against `positions_snapshot.json`, never against
`broker_sync` alone, and explicitly reports **leg disagreement** when the
same-source legs claim clean while the independent leg indicts.

Two correctness rules this check must keep:

- **Never sum the guard's legs.** The guard dual-books: `gamma|UNH 273` and
  `broker_sync|UNH 273` describe the *same* 273 shares. Summing invents a 2.0×
  phantom on every mixed symbol.
- **Mixed-ownership symbols are informational, not drift.** For operator-owned
  symbols the broker total blends operator shares with CHAD lots, so a
  strategy-vs-broker delta is not like-with-like. Inflating drift there
  re-introduces the false-RED class WKF U3 fixed (`position_guard.py:686`).

If `positions_snapshot.json` is stale or absent, EXS4 degrades to
`warn — Independent leg blind`. It never reports a false `ok`.

### EXS1 — TTL table and the operator_verify contract

TTLs live in `config/exterminator.json`. Every feed must either **cite** its TTL
source or flag `operator_verify`. **An unratified TTL can only warn, never
fail** — a sentinel must not fail a gate on a number nobody ratified. The
artifact's own `ttl_seconds` always wins over the table, so the config can never
silently drift from the publisher.

| Feed | TTL | Source |
|---|---|---|
| `var_state` | 3600 | artifact + `portfolio_var.py:388` |
| `scr_state` | 180 | artifact + `ops/scr_state_sync.py:82` |
| `positions_snapshot` | 300 | artifact + `ibkr_portfolio_collector_v2.py:310` |
| `kraken_ticks` | 30 | artifact (`kraken_prices.json`) |
| `bars` | 900 | **operator_verify** — no TTL declared anywhere; warn-only |
| `exit_overlay_heartbeat` | 900 | artifact + `position_exit_overlay.py:105` |
| `equity_history` | 86400 | **operator_verify** — daily cadence, no TTL declared; warn-only |

The **R14 blind check** (`health_monitor_rules.py:1134`) is integrated here: a
fresh heartbeat reporting `evaluated=0` is only *blind* if an independently
published, **provably fresh** truth artifact shows held positions. If that truth
is stale, absent or genuinely flat, the check stays silent — unproven input is
silence, not an alert.

### EXS8 — why it says `no_baseline`

There is **no production veto-rate baseline** anywhere in the repo or runtime.
Veto decisions emit `ML_SHADOW`/`ML_VETO` log lines only — no counter, no
artifact — so no live rate can be computed. The manifest's
`val_veto_rate_at_0.65` is a *training-time validation statistic*; comparing a
production rate against it would be a category error. EXS8 therefore reports
`warn no_baseline` honestly and additionally flags that the live manifest
(`xgb_veto_20260510_020007`) is older than its own 30-day staleness threshold.

## Pre-install verification (read-only, safe to run now)

```bash
cd /home/ubuntu/chad_finale && source venv/bin/activate

# 1. Unit tests (63) — includes the anti-auto-healing locks
python3 -m pytest chad/tests/test_exterminator_sentinel.py -q

# 2. Dry-run against live runtime WITHOUT writing to runtime/reports
python3 - <<'PY'
from pathlib import Path
from chad.ops.exterminator_sentinel import ExterminatorSentinel
s = ExterminatorSentinel(reports_dir=Path("/tmp/ext_dryrun"))
r = s.run()
print(r["overall_status"], r["counts"])
for c in r["checks"]:
    print(f"  {c['check_id']:6} {c['status']:5} {c['title']}")
PY
```

## Install (operator, gated — requires explicit GO)

```bash
# 0. The unit's ReadWritePaths must exist before first start.
mkdir -p /home/ubuntu/chad_finale/runtime/reports

# 1. Copy the unit files.
sudo cp /home/ubuntu/chad_finale/ops/systemd/chad-exterminator-sentinel.service \
        /etc/systemd/system/chad-exterminator-sentinel.service
sudo cp /home/ubuntu/chad_finale/ops/systemd/chad-exterminator-sentinel.timer \
        /etc/systemd/system/chad-exterminator-sentinel.timer
sudo systemctl daemon-reload

# 2. Run ONCE by hand and read the report before enabling the timer.
sudo systemctl start chad-exterminator-sentinel.service
journalctl -u chad-exterminator-sentinel -n 20 --no-pager
python3 -m json.tool /home/ubuntu/chad_finale/runtime/reports/EXTERMINATOR_SENTINEL_LATEST.json | head -40

# 3. Only if step 2 looks right, enable the 5-minute timer.
sudo systemctl enable --now chad-exterminator-sentinel.timer
systemctl list-timers chad-exterminator-sentinel --no-pager
```

## Expected first-run findings (as of 2026-07-15)

The scan is **expected to report `overall=fail` on first run** — these are
pre-existing conditions the sentinel is designed to surface, not regressions it
introduces. Do not treat a first-run `fail` as an install problem:

- **EXS1 fail** — `var_state.json` is ~69 days stale against its own 3600s TTL
  (a ~1663× breach) and has been published as fresh.
- **EXS3 fail** — one `pnl_untrusted` TLT row (INCIDENT-0713) sits in
  `data/trades/trade_history_20260713.ndjson`. SCR currently drops it via the
  *manual* bucket (`excluded_untrusted=0, excluded_manual=1`), so it is not
  being scored today — but the scored store is contaminated.
- **EXS5 fail** — `chad-ibkr-daily-bars-refresh.service` is in `failed`.
- **EXS6/EXS7/EXS8 warn** — uncommitted `config/tiers.json` +
  `config/withdrawal_policy.json`; 5 runtime contracts with no pinned schema;
  no ML veto baseline.

## Rollback

```bash
sudo systemctl disable --now chad-exterminator-sentinel.timer
sudo rm -f /etc/systemd/system/chad-exterminator-sentinel.{service,timer}
sudo systemctl daemon-reload
```

Nothing else needs undoing: the sentinel only ever wrote its own two report
artifacts, which are safe to delete.
