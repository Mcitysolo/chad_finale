# CHAD Ops Runbook — Runtime Self-Heal + Proof Logging

Generated: 2026-02-26T19:36:09Z (UTC)  
Canonical root: `/home/ubuntu/chad_finale`

## What this runbook is
This runbook covers CHAD’s **runtime self-heal** workflow:
- How to rebuild `runtime/` safely
- How to verify CHAD is healthy afterwards
- Where proof logs are stored
- How to rollback using archived snapshots

CHAD provides:
- `chad-runtime-rebuild.service` (systemd oneshot)
- `/usr/local/bin/chad-rebuild` (operator wrapper)
- Proof logs: `runtime/proofs/RUNTIME_REBUILD_PROOF_*.json`

---

## Quick Start (30 seconds)

### 1) Rebuild runtime + generate proof
```bash
sudo chad-rebuild
```

### 2) Confirm API is up
```bash
curl -fsS http://127.0.0.1:9618/health | python3 -m json.tool | sed -n '1,80p'
curl -fsS http://127.0.0.1:9618/status | python3 -m json.tool | sed -n '1,170p'
```

### 3) Confirm latest proof exists
```bash
ls -1t runtime/proofs/RUNTIME_REBUILD_PROOF_*.json | head -n 1
```

---

## What “Healthy” Looks Like

### `/health`
- `healthy: true`
- `exec_mode: dry_run` (until you explicitly enable live)
- No connection refused / JSON parse errors

### `/status` (key checks)
- `runtime_files.*.exists` should be true for:
  - `feed_state`
  - `positions_snapshot`
  - `reconciliation_state`
  - `dynamic_caps`
  - `operator_intent`
  - `portfolio_snapshot`
  - `scr_state`
  - `tier_state`

- `live_gate.operator_mode` should **NOT** be `DENY_ALL` due to:
  - `operator_intent_json_decode_error`
  - `operator_intent_stale_or_missing`

Expected operator intent state after rebuild:
- `operator_mode: ALLOW`
- `operator_reason: runtime_rebuild_allow_entries_non_live`
- `ttl_seconds: 900`
- `ts_utc` should be recent

---

## Proof Logging (Audit Trail)

Every `sudo chad-rebuild` generates a proof JSON:

Path:
- `runtime/proofs/RUNTIME_REBUILD_PROOF_<UTCSTAMP>.json`

It includes:
- `ts_utc` (when proof was generated)
- `git_sha` (exact code version at rebuild time)
- Per-artifact:
  - exists / bytes / mtime / age_s

### View latest proof quickly
```bash
LATEST="$(ls -1t runtime/proofs/RUNTIME_REBUILD_PROOF_*.json | head -n1)"
python3 -m json.tool "$LATEST" | sed -n '1,140p'
```

---

## Rollback / Restore (If Needed)

If a rebuild causes unexpected behavior, rollback runtime using an archived snapshot:

1) List available runtime snapshots:
```bash
ls -1dt _archive/runtime_snapshot_* | head -n 5
```

2) Restore the most recent snapshot:
```bash
SNAP="$(ls -1dt _archive/runtime_snapshot_* | head -n1)"
rm -rf runtime
cp -a "$SNAP" runtime
```

3) Re-run:
```bash
sudo chad-rebuild
```

---

## Troubleshooting

### A) Rebuild service fails
```bash
systemctl --no-pager -l status chad-runtime-rebuild.service | sed -n '1,200p'
journalctl -u chad-runtime-rebuild.service --no-pager -n 200
```

### B) Operator intent issues (DENY_ALL)
Check operator intent file:
```bash
ls -lh runtime/operator_intent.json
python3 -m json.tool runtime/operator_intent.json | sed -n '1,80p'
```

If missing, run rebuild again:
```bash
sudo chad-rebuild
```

### C) Broker evidence missing / stale
Broker events live here:
- `data/broker_events/BROKER_EVENTS_IBKR_YYYYMMDD.ndjson`

Check latest:
```bash
ls -1t data/broker_events/BROKER_EVENTS_IBKR_*.ndjson | head -n 1
tail -n 3 "$(ls -1t data/broker_events/BROKER_EVENTS_IBKR_*.ndjson | head -n 1)"
```

---

## Canonical Commands Reference
- Rebuild + proof:
  - `sudo chad-rebuild`
- Service:
  - `sudo systemctl start chad-runtime-rebuild.service`
- API health/status:
  - `curl http://127.0.0.1:9618/health`
  - `curl http://127.0.0.1:9618/status`
- Proofs:
  - `runtime/proofs/RUNTIME_REBUILD_PROOF_*.json`

