> **STATUS UPDATE 2026-05-27:**
> Channel 2 design and patch artifacts landed in commit <will-be-filled-after-commit>.
> See:
> - `ops/systemd_templates/chad-service-alert@.service`
> - `ops/systemd_templates/patches/chad-backend.patch`
> - `config/port_binding_allowlist.json`
> - `ops/runbooks/INSTALL_chad_service_alert_template_2026-05-27.md`
>
> Awaiting Channel 1 operator-authorized install per CLAUDE.md rule #6.

# PORT-BINDING systemd unit edits — chad-backend.service (port 9618)

# HIGH_ID: PORT-BINDING-1 (systemd portion)
# Status: PROPOSED
# Severity: HIGH (companion to ops/pending_actions/PORT_BINDING_localhost_only_hardening_2026-05-27.md)
# Effort: S

# Source audit
- `reports/forensic_audits/CHAD_FORENSIC_FULL_SYSTEM_AUDIT_20260527T142951Z.md`
  - §9.1 Tier 8 bound ports
  - §19 closeout item 6 (PORT-BINDING-1)

# Problem statement
Code-side defaults for `chad-shadow-status` (port 9619) and `chad-metrics`
(port 9620) are now `127.0.0.1` by default with documented env-var overrides
(`CHAD_STATUS_HOST`, `CHAD_METRICS_HOST`). The third audited port — `9618`
served by `uvicorn backend.app:app` — receives its bind host via the systemd
ExecStart command-line argument `--host 0.0.0.0`, not from any Python config.

The fix requires editing `/etc/systemd/system/chad-backend.service` (or its
drop-in `chad-backend.service.d/override.conf`) to flip `--host 0.0.0.0` to
`--host 127.0.0.1`. Unit-file edits are operator-domain per CLAUDE.md
governance rule #6.

# Evidence
- `ps -p 1068618 -o cmd=` (audit-time):
  ```
  /home/ubuntu/chad_finale/venv/bin/python3 -m uvicorn backend.app:app \
      --host 0.0.0.0 --port 9618 --log-level info
  ```
- `ls -lh /etc/systemd/system/chad-backend.service`
  + drop-ins under `/etc/systemd/system/chad-backend.service.d/`.
- AWS Security Group default-deny (audit §9.2) confirms the port is not
  externally reachable.

# Required closeout (operator-domain — not applied here)
1. Inspect both the unit file and its drop-ins:
   ```
   systemctl cat chad-backend.service
   ```
2. Edit the appropriate file to change `--host 0.0.0.0` → `--host 127.0.0.1`.
   (Prefer editing a drop-in `chad-backend.service.d/override.conf` if it
   already exists, to keep the base unit unmodified.)
3. `systemctl daemon-reload`.
4. `systemctl restart chad-backend.service` (requires explicit GO per
   CLAUDE.md rule #7).
5. Verify with `ss -tlnp | grep :9618` shows `127.0.0.1:9618`.

# Acceptance criteria
- `ss -tlnp` reports `127.0.0.1:9618` (not `0.0.0.0:9618`) after the
  operator-approved restart.
- `python -m chad.validators.port_binding --check --live-check` exits 0.
- AWS SG default-deny remains the second-layer guard.
- All clients of the backend continue to function:
  - `chad/utils/telegram_bot.py` uses `http://127.0.0.1:9618` already.
  - `chad/core/orchestrator.py` uses `http://127.0.0.1:9618/live-gate`.
  - `chad/core/paper_shadow_runner.py` uses `http://127.0.0.1:9618/live-gate`.
  Both already target localhost, so the re-bind is transparent.

# Required tests
- (No new automated test for systemd edits; the code-side validator
  `chad/validators/port_binding.py` already covers the in-repo surface.)

# Operator approvals required
- **Approval 1 (unit edit):** explicit GO for the unit-file or drop-in edit.
- **Approval 2 (daemon-reload):** explicit GO for `systemctl daemon-reload`.
- **Approval 3 (service restart):** explicit GO for `systemctl restart
  chad-backend.service` per CLAUDE.md rule #7.

# Session 1 impact
- **Does this item block Session 1?** NO.
- The 0.0.0.0 binding is a defense-in-depth gap. AWS SG default-deny
  contains the exposure today.

# No-live confirmation
This Pending Action does not authorize live trading.
ready_for_live must remain false.
allow_ibkr_live must remain false.
allow_ibkr_paper must remain true.
No broker orders may be placed or cancelled under this Pending Action.
