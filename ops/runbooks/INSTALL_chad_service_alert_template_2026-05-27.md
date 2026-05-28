### Title
CHAD Service Failure Alert Template + OnFailure Patches — Channel 1 Install Runbook (2026-05-27)

### Status
PROPOSED — requires explicit operator authorization before execution

### Audit trail
- Sourced from:
  `reports/forensic_audits/CHAD_FORENSIC_FULL_SYSTEM_AUDIT_20260527T142951Z.md`
- Decision 3 of the institutional closeout plan
- Related Pending Actions:
  - `ops/pending_actions/OPTIONS_CHAIN_OnFailure_directive_2026-05-27.md`
  - `ops/pending_actions/PORT_BINDING_systemd_unit_edits_2026-05-27.md`
- Channel 2 artifacts (already landed):
  - `chad/ops/service_failure_alert.py`
  - `ops/systemd_templates/chad-service-alert@.service`
  - `ops/systemd_templates/patches/chad-options-chain-refresh.patch`
  - `ops/systemd_templates/patches/chad-backend.patch`
  - `config/port_binding_allowlist.json`
- Governance: this runbook performs systemd mutations and therefore
  requires explicit operator authorization per **CLAUDE.md rule #6**
  ("Never modify systemd service files without explicit instruction").
  All commands below are operator-executed (`sudo`); Claude must NOT
  run any of them.

### Pre-install verification (no mutation)
Commands the operator runs to confirm the templates and patches exist in the repo:

```bash
cd /home/ubuntu/chad_finale
ls -lh ops/systemd_templates/chad-service-alert@.service
ls -lh ops/systemd_templates/patches/chad-options-chain-refresh.patch
ls -lh ops/systemd_templates/patches/chad-backend.patch
cat ops/systemd_templates/chad-service-alert@.service
```

Expected: all three files exist; the template prints with the
`chad-service-alert@.service` [Unit] header.

### Step 1 — Backup current unit files

```bash
sudo cp /etc/systemd/system/chad-options-chain-refresh.service \
  /etc/systemd/system/chad-options-chain-refresh.service.bak.2026-05-27
sudo cp /etc/systemd/system/chad-backend.service \
  /etc/systemd/system/chad-backend.service.bak.2026-05-27
ls -lh /etc/systemd/system/chad-*.service.bak.2026-05-27
```

Expected: two backup files present, owned by root.

### Step 2 — Install the template alert unit

```bash
sudo cp ops/systemd_templates/chad-service-alert@.service \
  /etc/systemd/system/chad-service-alert@.service
sudo chmod 644 /etc/systemd/system/chad-service-alert@.service
ls -lh /etc/systemd/system/chad-service-alert@.service
```

Expected: template unit installed at `/etc/systemd/system/chad-service-alert@.service`, mode 644.

### Step 3 — Apply OnFailure= directive to chad-options-chain-refresh
Operator edits the unit file by hand following the patch comments in
`ops/systemd_templates/patches/chad-options-chain-refresh.patch`.
The single added line is `OnFailure=chad-service-alert@%n.service`
inside the `[Unit]` section.

Verify:

```bash
grep -n "OnFailure" /etc/systemd/system/chad-options-chain-refresh.service
# Expect: OnFailure=chad-service-alert@%n.service
```

### Step 4 — Apply OnFailure= directive + uvicorn host change to chad-backend
Operator edits the unit file by hand following the patch comments in
`ops/systemd_templates/patches/chad-backend.patch`. Two changes:
1. Add `OnFailure=chad-service-alert@%n.service` inside `[Unit]`.
2. Change the uvicorn `--host` argument from `0.0.0.0` to `127.0.0.1`
   in the `ExecStart=` line.

Verify:

```bash
grep -n "OnFailure" /etc/systemd/system/chad-backend.service
grep -n "uvicorn" /etc/systemd/system/chad-backend.service
# Expect: OnFailure=chad-service-alert@%n.service
# Expect: --host 127.0.0.1 (not --host 0.0.0.0)
```

### Step 5 — systemctl daemon-reload

```bash
sudo systemctl daemon-reload
```

Expected: silent success (no output).

### Step 6 — Validate without restart

```bash
systemctl cat chad-options-chain-refresh.service | grep OnFailure
systemctl cat chad-backend.service | grep -E "OnFailure|uvicorn"
systemctl cat chad-service-alert@.service | head -20
```

Expected: each grep prints the new line; `systemctl cat chad-service-alert@.service`
prints the template header + comment block.

### Step 7 — Test the alert path with a synthetic failure

```bash
/home/ubuntu/chad_finale/venv/bin/python3 -m chad.ops.service_failure_alert \
  --failed-unit chad-options-chain-refresh.service \
  --severity HIGH \
  --include-journal-tail 50 \
  --include-runtime-snapshot \
  --dry-run
```

Expected: structured JSON payload printed; **no** Telegram message sent
(because `--dry-run`). Payload must contain:
- `schema_version=service_failure_alert.v1`
- `journal_tail` (list of strings)
- `runtime_snapshot` (metadata-only dict, no contents)
- `active_unit_status`

### Step 8 — Confirm port 9618 binding (after chad-backend restart)
**This requires a chad-backend restart, which is OPERATOR-AUTHORIZED ONLY**
per CLAUDE.md rule #7. If the operator wishes to restart at this time:

```bash
sudo systemctl restart chad-backend.service
sleep 2
ss -tlnp | grep -E ":9618" | head -5
# Expect: 127.0.0.1:9618 (not 0.0.0.0:9618)
curl -s http://127.0.0.1:9618/health || echo "backend not yet reachable"
```

If the operator defers the restart, port 9618 remains on `0.0.0.0`
until the next natural service restart. The OnFailure directive added
in Step 4 is unaffected — it activates on the next failure regardless
of restart timing.

### Rollback
If any step fails:

```bash
sudo cp /etc/systemd/system/chad-options-chain-refresh.service.bak.2026-05-27 \
  /etc/systemd/system/chad-options-chain-refresh.service
sudo cp /etc/systemd/system/chad-backend.service.bak.2026-05-27 \
  /etc/systemd/system/chad-backend.service
sudo rm -f /etc/systemd/system/chad-service-alert@.service
sudo systemctl daemon-reload
```

Expected: original units restored, template removed, daemon reloaded.
No further action required for rollback.

### No-live confirmation
This runbook does **not** enable live trading.
`ready_for_live` must remain false.
`allow_ibkr_live` must remain false.
`allow_ibkr_paper` must remain true.
No broker orders may be placed or cancelled.

### Acceptance criteria
- `chad-service-alert@.service` installed at `/etc/systemd/system/`.
- `OnFailure=chad-service-alert@%n.service` present in both target units.
- `chad-backend` `ExecStart` uses `uvicorn --host 127.0.0.1 --port 9618`.
- Dry-run test of `service_failure_alert` produces structured JSON
  conforming to `service_failure_alert.v1`.
- `ss -tlnp` shows `127.0.0.1:9618` after the chad-backend restart
  (operator-authorized; may be deferred).
- Backup files `chad-*.service.bak.2026-05-27` present for rollback.
