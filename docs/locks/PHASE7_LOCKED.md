# PHASE 7 LOCKED — Always-On Engine (DRY_RUN) + LiveGate Enforcement

Date (UTC): 2025-12-29

Phase 7 Guarantees:
- Always-on orchestrator cycles continuously under systemd.
- Backend API is live and reports Phase 7 DRY_RUN-only posture.
- LiveGate is the SSOT for execution permission and includes:
  - operator intent fields (operator_mode/operator_reason)
  - allow_ibkr_paper + allow_ibkr_live fields
  - STOP enforcement capability (DENY_ALL)
- Execution remains hard-locked to DRY_RUN (ibkr_dry_run=true).
- Paper shadow automation is OFF by default:
  - chad-paper-shadow-runner.timer is disabled
  - chad-paper-shadow-tick.timer remains masked
  - paper execution is manual only unless explicitly enabled later.

Authoritative proof commands:
- systemctl status chad-orchestrator.service chad-backend.service --no-pager -n 12
- curl -sS http://127.0.0.1:9618/health | python3 -m json.tool
- curl -sS http://127.0.0.1:9618/live-gate | python3 -m json.tool
- curl -sS http://127.0.0.1:9618/operator-intent | python3 -m json.tool
- systemctl status chad-paper-shadow-runner.timer chad-paper-shadow-tick.timer --no-pager -n 12
- journalctl -u chad-orchestrator.service --since "-10 min" --no-pager -o cat | tail -n 120
