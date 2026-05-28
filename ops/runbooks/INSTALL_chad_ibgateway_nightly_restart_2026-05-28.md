### Title
CHAD IB Gateway Nightly Restart — Channel 1 Install Runbook (2026-05-28)

### Status
PROPOSED — requires explicit operator authorization before execution.

### Audit trail
- Sourced from:
  `reports/forensic_audits/CHAD_FORENSIC_FULL_SYSTEM_AUDIT_20260527T142951Z.md`
- Addresses the recurring Gateway socket-leak wedge documented in:
  `ops/pending_actions/IBKR_RELIABILITY_socket_backpressure_and_gateway_churn_2026-05-27.md`
- Implements a partial scope of:
  `ops/pending_actions/IBKR_AUTO_RECOVERY_design_2026-05-27.md`
  (the scheduled-restart layer; auto-recovery on a *detected* wedge is a
  separate, still-pending phase).
- Prerequisite: Fix A symmetric hysteresis on `broker_latency`
  (commits `f3ab3d8` + `291bb84`, production-active since
  `2026-05-28T19:55:24Z`). This hysteresis is what makes the brief 60-90s
  restart window survivable without tripping `stop_bus`.
- Incident #3 of the wedge pattern occurred `2026-05-28T04:28-17:34 UTC`,
  causing the 13-hour halt that dirtied Paper Epoch 3 Session 1.
- Channel 2 artifacts (already landed in the repo):
  - `ops/systemd_templates/chad-ibgateway-nightly-restart.service`
  - `ops/systemd_templates/chad-ibgateway-nightly-restart.timer`
  - `scripts/post_gateway_restart_verify.py`
  - `chad/tests/test_post_gateway_restart_verify.py`
- Governance: this runbook performs systemd mutations and therefore requires
  explicit operator authorization per **CLAUDE.md rule #6** ("Never modify
  systemd service files without explicit instruction") and rule #7 ("Never
  restart live services without explicit instruction"). All commands below
  are operator-executed (`sudo`); Claude must NOT run any of them.

### Pre-install verification (no mutation)
```bash
cd /home/ubuntu/chad_finale
ls -lh ops/systemd_templates/chad-ibgateway-nightly-restart.service
ls -lh ops/systemd_templates/chad-ibgateway-nightly-restart.timer
ls -lh scripts/post_gateway_restart_verify.py
test -x scripts/post_gateway_restart_verify.py && echo "verifier executable OK"
```
Expected: all three files exist; the verifier is executable.

### Step 1 — Backup any prior version (idempotency safety)
```bash
test ! -f /etc/systemd/system/chad-ibgateway-nightly-restart.service || \
  sudo cp /etc/systemd/system/chad-ibgateway-nightly-restart.service \
    /etc/systemd/system/chad-ibgateway-nightly-restart.service.bak.$(date -u +%Y%m%dT%H%M%SZ)
test ! -f /etc/systemd/system/chad-ibgateway-nightly-restart.timer || \
  sudo cp /etc/systemd/system/chad-ibgateway-nightly-restart.timer \
    /etc/systemd/system/chad-ibgateway-nightly-restart.timer.bak.$(date -u +%Y%m%dT%H%M%SZ)
```
Expected: backups created only if prior versions already exist (first install
is a no-op here).

### Step 2 — Install unit files
```bash
sudo cp ops/systemd_templates/chad-ibgateway-nightly-restart.service /etc/systemd/system/
sudo cp ops/systemd_templates/chad-ibgateway-nightly-restart.timer /etc/systemd/system/
sudo chmod 644 /etc/systemd/system/chad-ibgateway-nightly-restart.service
sudo chmod 644 /etc/systemd/system/chad-ibgateway-nightly-restart.timer
```
Expected: both units installed at `/etc/systemd/system/`, mode 644, owned by root.

### Step 3 — daemon-reload
```bash
sudo systemctl daemon-reload
```
Expected: silent success (no output).

### Step 4 — Enable timer (but do NOT start the service manually)
```bash
sudo systemctl enable chad-ibgateway-nightly-restart.timer
sudo systemctl start chad-ibgateway-nightly-restart.timer
```
Expected: timer enabled and active. Note we start the *timer*, never the
service — the service is fired only by the timer (or by an explicit manual
trigger in Step 7).

### Step 5 — Verify timer is scheduled
```bash
systemctl list-timers chad-ibgateway-nightly-restart.timer --no-pager
systemctl status chad-ibgateway-nightly-restart.timer --no-pager
```
Expected: `NEXT` shows the next 03:15 UTC; `LEFT` shows time remaining.

### Step 6 — Dry-run verifier (no actual restart)
```bash
/home/ubuntu/chad_finale/venv/bin/python3 \
  scripts/post_gateway_restart_verify.py
```
Expected: exit 0 if the Gateway is currently healthy; an alert artifact is
written to `reports/gateway_restart_log/<ts>.json`. (This step does NOT
restart anything — it only reads the live Gateway state.)

### Step 7 — Optional: trigger one manual run to validate the full cycle
```bash
# OPERATOR DECIDES whether to run this immediately or wait for the first
# scheduled fire. Manual trigger forces a Gateway restart NOW.
#
# sudo systemctl start chad-ibgateway-nightly-restart.service
# journalctl -u chad-ibgateway-nightly-restart.service --no-pager | tail -30
```
Expected (if run): journal shows the restart, the 60s sleep, then the verifier
summary line and exit code.

### Verification commands after first scheduled fire
```bash
systemctl status chad-ibgateway-nightly-restart.service --no-pager
journalctl -u chad-ibgateway-nightly-restart.service --since "12 hours ago" --no-pager
ls -lh reports/gateway_restart_log/ | tail -10
cat reports/gateway_restart_log/$(ls -t reports/gateway_restart_log/ | head -1)
```
Expected: a clean service run, exit code 0, and a fresh artifact whose
`overall_ok` is `true`.

### Rollback
```bash
sudo systemctl stop chad-ibgateway-nightly-restart.timer
sudo systemctl disable chad-ibgateway-nightly-restart.timer
sudo rm /etc/systemd/system/chad-ibgateway-nightly-restart.service
sudo rm /etc/systemd/system/chad-ibgateway-nightly-restart.timer
sudo systemctl daemon-reload
```
Expected: timer stopped and disabled, both units removed, daemon reloaded.
No further action required for rollback.

### No-live confirmation
This runbook does **not** authorize live trading.
`ready_for_live` must remain false.
`allow_ibkr_live` must remain false.
`allow_ibkr_paper` must remain true.
No broker orders may be placed or cancelled.

### Acceptance criteria
- Timer enabled and shows `NEXT` firing at 03:15 UTC tomorrow.
- Verifier dry-run exits 0 against the currently-healthy Gateway.
- First scheduled fire produces a journal entry with exit code 0 and a
  `reports/gateway_restart_log/` artifact.
- Latency post-restart is below 2000ms within 120s.
- `stop_bus` does NOT trip during the restart window (proves Fix A hysteresis
  is doing its job).
