# INSTALL — VaR Publisher Timer (chad-var-publisher)

**Status:** Repo-only design artifacts. Nothing here is installed, enabled,
reloaded or started automatically. Per CLAUDE.md governance rules #6/#7 the
agent never installs unit files or starts services — the operator runs the
commands below during a maintenance window.

## Why this exists

`ops/var_publisher.py` already renders a schema-correct `var_state.v1` and
writes `runtime/var_state.json` atomically. It has **no scheduler** — no timer,
no importer, no caller. The artifact therefore froze at
`ts_utc=2026-05-07T13:39:30Z` (~72 days stale), which is exactly why the
Exterminator Sentinel's `EXS1` (`check_stale_feeds`) fails the `var_state` feed
row (`config/exterminator.json`: `ttl_verified=true`, `fail_after_seconds=7200`).

**The fix for EXS1 is a timer, not a timestamp change.** A publisher run at a
cadence below the 7200s EXS1 floor restamps `ts_utc` from a fresh recompute and
clears the row. Stamping `ts_utc` from *data-time* instead would make the
artifact **older** and trip EXS1 harder — do not do that.

## The input-freshness guard (W1A-1, already merged)

A naive timer introduces a new hazard: it would recompute every 30 minutes and
stamp `ts_utc=now` **even against stale bars/prices** — the "stale-as-fresh"
failure re-emerging one layer down, invisible to EXS1 and the A4 metrics guard
(both inspect only the artifact's own `ts_utc`). `ops/var_publisher.py` now gates
on **input** freshness:

- It checks the mtime of the positions file in use, `price_cache.json`, and the
  per-symbol daily bars actually consumed.
- If any exceeds `CHAD_VAR_INPUT_MAX_AGE_SECONDS` (default **172800s / 48h**, to
  tolerate weekend / market-closed gaps), an otherwise-`ok` report is downgraded
  to `status="stale_inputs"` and the additive `inputs_fresh=false` /
  `oldest_input` / `oldest_input_age_seconds` fields name the culprit.
- The A4 consumer (`metrics_server.py`: `var_ok = status=="ok" and not stale`)
  therefore correctly reports `chad_var_status_ok=0` on stale inputs.
- `--allow-stale-inputs` suppresses only the status downgrade for a manual
  diagnostic run; it never masks the recorded freshness fields.

`enforcement_active` stays `False`: VaR gates no trading. This is observability
hygiene only.

## Artifact + cadence contract

| Path | Shape |
|---|---|
| `runtime/var_state.json` | full report, `var_state.v1`, rewritten each run |

| Consumer | Freshness window |
|---|---|
| EXS1 `check_stale_feeds` | `fail_after_seconds=7200` (2h) — the **strictest** |
| artifact `ttl_seconds` | `3600` (1h) |
| `metrics_server` / R02 daily map | `86400` (24h) |

The **30-minute** timer cadence (`OnUnitActiveSec=1800`) is deliberately below
the 3600s TTL and the 7200s EXS1 floor. **Do not pick a cadence > 1h** or EXS1
stays red between runs.

## Activation gap (read this)

**This repo change does not activate anything.** Until an operator installs and
enables the timer below, `runtime/var_state.json` stays frozen and EXS1's
`var_state` row stays **red**. The repo work (publisher freshness guard + unit
files + this doc) is complete and tested; the runtime EXS1 clears only
post-deploy.

## Pre-install verification (read-only, safe to run now)

```bash
cd /home/ubuntu/chad_finale && source venv/bin/activate

# 1. Unit tests for the freshness guard (tmp_path only — never touches runtime).
python3 -m pytest chad/tests/test_w1a_var_publisher_freshness.py -q

# 2. Confirm the -m entrypoint resolves WITHOUT writing to real runtime.
python3 -m ops.var_publisher --help

# 3. Dry-run against a scratch runtime dir (writes only under /tmp).
CHAD_RUNTIME_DIR=/tmp/var_dryrun python3 -m ops.var_publisher | python3 -m json.tool
```

## Install (operator, gated — requires explicit GO)

```bash
# 0. runtime/ (the ReadWritePath) already exists in production; nothing to mkdir.

# 1. Copy the unit files.
sudo cp /home/ubuntu/chad_finale/ops/systemd/chad-var-publisher.service \
        /etc/systemd/system/chad-var-publisher.service
sudo cp /home/ubuntu/chad_finale/ops/systemd/chad-var-publisher.timer \
        /etc/systemd/system/chad-var-publisher.timer
sudo systemctl daemon-reload

# 2. Run ONCE by hand and read the artifact before enabling the timer.
sudo systemctl start chad-var-publisher.service
journalctl -u chad-var-publisher -n 20 --no-pager
python3 -m json.tool /home/ubuntu/chad_finale/runtime/var_state.json | head -40
# Expect status in {ok, stale_inputs, insufficient_data}; if stale_inputs,
# read oldest_input to see which market-data feed is behind.

# 3. Only if step 2 looks right, enable the 30-minute timer.
sudo systemctl enable --now chad-var-publisher.timer
systemctl list-timers chad-var-publisher --no-pager

# 4. Confirm EXS1 clears on the next sentinel run.
python3 -m json.tool /home/ubuntu/chad_finale/runtime/reports/EXTERMINATOR_SENTINEL_LATEST.json \
  | grep -A3 var_state
```

## Rollback

```bash
sudo systemctl disable --now chad-var-publisher.timer
sudo rm -f /etc/systemd/system/chad-var-publisher.{service,timer}
sudo systemctl daemon-reload
```

Nothing else needs undoing: the publisher only ever wrote `runtime/var_state.json`.
