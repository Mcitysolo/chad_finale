# PORT-BINDING — Localhost-only bind hardening for 9618/9619/9620

# HIGH_ID: PORT-BINDING-1
# Status: PROPOSED
# Severity: HIGH
# Effort: S

# Source audit
- `reports/forensic_audits/CHAD_FORENSIC_FULL_SYSTEM_AUDIT_20260527T142951Z.md`
  - §9.1 (Tier 8 bound ports)
  - §9.2 (AWS Security Group inspection)
  - §1 Executive Summary (additional HIGH-severity finding #1)
  - §18 Findings Priority Matrix — Effort S / Severity HIGH cell
  - §19 closeout item 6 (PORT-BINDING-1)

# Problem statement
Three CHAD internal HTTP/Prometheus endpoints currently bind to `0.0.0.0` rather than `127.0.0.1`:

```
0.0.0.0:9618 — chad-shadow-status.service (python pid 1068618, fd=6)
0.0.0.0:9619 — chad-shadow-status python (pid 2175218, fd=3)
0.0.0.0:9620 — chad-metrics (pid 2525497, fd=3)
```

These services exist to expose internal state to the operator and to Prometheus / Grafana on the same host. They have **no documented need** to listen on external interfaces. The remaining CHAD-internal listener `127.0.0.1:8765` (dashboard API) is bound localhost-only, which is the correct shape.

Per the audit §9.2, the AWS Security Groups attached to this instance do **not** explicitly open 9618 / 9619 / 9620. Default-deny SG behaviour means these ports are not externally reachable as of 2026-05-27. The risk is therefore a **defense-in-depth gap**, not an active exposure — but a single misconfigured SG change, instance migration, or VPC peering arrangement could open the ports without any code change visible to CHAD operators.

# Evidence
1. `ss -tlnp 2>/dev/null | grep -E ':961[89]|:9620'` from the audit:
   ```
   LISTEN 0  128  0.0.0.0:9618  0.0.0.0:*  users:(("python",pid=1068618,fd=6))
   LISTEN 0  128  0.0.0.0:9619  0.0.0.0:*  users:(("python",pid=2175218,fd=3))
   LISTEN 0  128  0.0.0.0:9620  0.0.0.0:*  users:(("python",pid=2525497,fd=3))
   ```
2. `aws ec2 describe-security-groups` (audit §9.2) confirms the 3 SGs attached to the instance (`sg-0fcc5d2109705572a`, `sg-0a300561f18f9c4e2`, `sg-0cc1289edcfcca335`) do not open 9618/9619/9620 to any CIDR.
3. The peer `127.0.0.1:8765` listener (dashboard API) demonstrates the correct binding shape is already in use elsewhere in CHAD.
4. `gaps_to_close` finding P2-5 (historical) referenced this exact gap; the v9.3 SSOT acknowledges it as defense-in-depth.

# Affected services / configs
- `chad-shadow-status.service` — owner of port 9618
- A second `chad-shadow-status` python process — owner of port 9619 (likely a child/extension; needs entry-point inventory)
- `chad-metrics.service` (or equivalent owner of pid 2525497) — owner of port 9620
- Python entry-points that construct the HTTP servers (e.g. `uvicorn.run(host=..., port=...)`, `prometheus_client.start_http_server(host=..., port=...)`)
- Any config file or env var that supplies the bind host

# Root-cause hypothesis
The HTTP server entry-points default to `0.0.0.0` because:
1. Prometheus client `start_http_server(port, addr='0.0.0.0')` defaults to all interfaces when `addr` is unspecified.
2. Uvicorn / FastAPI / aiohttp `run(host='0.0.0.0', port=…)` is the common copy-paste idiom.
3. No CHAD coding convention has been enforced to require `127.0.0.1` for internal listeners.

The fix is mechanical: change the default to `127.0.0.1` at the code/config layer, gate any non-localhost binding behind an explicit env var (`CHAD_BIND_HOST_<service>`), and add a validator that fails CI / pre-flight if any audited port defaults to non-localhost without an allowlisted reason.

# Required closeout
1. **Read-only inventory** of every owning entry-point for 9618, 9619, 9620 (file path, function, current bind host).
2. **Default change** in the owning code/config: `0.0.0.0 → 127.0.0.1`.
3. **Env-var override** in the same code path: respect `CHAD_BIND_HOST_<service>` if set, with a documented allowlist.
4. **Validator** at `chad/validators/port_binding.py` that scans repo config/code and fails when any audited port defaults to non-localhost without an exception.
5. **Pending Action** (separate) for any required `/etc/systemd/system/chad-*.service` edits if a unit file passes the bind host as an ExecStart argument — the unit-file edit is operator-domain, not in scope here.
6. **Telegram alert path** is not required; the AWS SG default-deny already provides external containment.

# Acceptance criteria
- `ss -tlnp` (after operator-approved service restart, NOT in this PA scope) shows `127.0.0.1:9618`, `127.0.0.1:9619`, `127.0.0.1:9620` only.
- `python -m chad.validators.port_binding --check` exits 0 against the merged repo state.
- AWS SG default-deny remains the second layer of defense; this PA does not modify SGs.
- No external impact: the only callers of these ports (Prometheus scraper, operator dashboards, ad-hoc curl) all live on the same host and already use `127.0.0.1` URLs.

# Required tests
- `chad/tests/test_port_binding_validator.py` — must cover:
  - Default config has 127.0.0.1 → validator exits 0.
  - 0.0.0.0 default without allowlist exception → validator exits non-zero.
  - 0.0.0.0 with explicit allowlist entry → validator passes with a warning.
  - Each of 9618 / 9619 / 9620 is covered.

# Operator approvals required
- **Approval 1 (code default change):** operator approves the `0.0.0.0 → 127.0.0.1` default in the owning entry-points.
- **Approval 2 (systemd unit edit, if needed):** if any unit file passes a bind host argument, operator authorises the unit edit and the subsequent service restart. Unit-file edits are operator-domain (per CLAUDE.md governance rule #6).
- **Approval 3 (service restart):** explicit operator GO before any `systemctl restart` of shadow-status / metrics services.

# Session 1 impact
- **Does this item block Session 1 evaluation (window opens 2026-05-28T00:00:00Z)?** NO.
- The 0.0.0.0 binding does not affect any paper-trading decision path or any SCR / live-gate field.
- The audit explicitly classifies impact as **contained** by AWS SG default-deny.
- Code-default and validator changes can land before Session 1 with zero service restart; the actual port re-bind requires service restart and therefore lands AFTER Session 1.

# No-live confirmation
This Pending Action does not authorize live trading.
ready_for_live must remain false.
allow_ibkr_live must remain false.
allow_ibkr_paper must remain true.
No broker orders may be placed or cancelled under this Pending Action.
