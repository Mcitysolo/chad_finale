### Title
IBKR Gateway Version Upgrade — Current 10.37 → Target 10.45+

### Status
PROPOSED — awaiting operator decision and Channel 3 execution

### Severity
MEDIUM — current version (1037 = 10.37) is documented stale per external
IBKR operator-community recommendation (≥10.45 / build 1045). No immediate
operational fault attributable to version alone, but the socket leak / wedge
pattern (incidents on Day-0, today's 13h halt, and today's afternoon spike)
MAY be reduced by upgrading. Cannot be confirmed without the upgrade.

### Source authority
- External IBKR operator research summary (provided 2026-05-28)
- Audit: `reports/forensic_audits/CHAD_FORENSIC_FULL_SYSTEM_AUDIT_20260527T142951Z.md`
- Tool output: `chad/tools/ibkr_gateway_version_check`
  (build=1037, target=1045, severity=stale, delta=8)
- IBKR Trader Workstation release notes (operator must verify the current
  version at install time — the 10.45 figure is the external report's claim
  as of May 2026; operator must confirm against the IBKR portal)

### Evidence (today's incidents)
- 2026-05-28T04:28-17:34 UTC: 13h Gateway hard-wedge, socket leak, 16+
  CLOSE-WAIT (incident #3 of this pattern)
- 2026-05-28T18:12 UTC: latency spike to 2962ms within 40min of restart
  (resolved by Fix A hysteresis going forward)
- Fix A + Fix B address the symptoms; the version upgrade addresses one
  hypothesized root cause

### Affected components
- `/home/ubuntu/Jts/ibgateway/1037/` (entire installation)
- `/etc/chad/ibc/config.ini` (tws-version arg passed to ibcstart.sh,
  currently "1037")
- `/etc/systemd/system/chad-ibgateway.service` (ExecStart hardcodes "1037")

### Upgrade procedure (Channel 3 — operator-led)
1. Operator logs into IBKR account portal
2. Download latest stable IB Gateway for Linux (.sh installer)
3. Verify checksum against IBKR's published value
4. Extract to `/home/ubuntu/Jts/ibgateway/<NEW_BUILD>/` (parallel install,
   does not overwrite 1037)
5. Update `/etc/chad/ibc/config.ini` if any config syntax changed (compare
   release notes)
6. Update `/etc/systemd/system/chad-ibgateway.service` ExecStart to use the
   new build number (e.g. "1045" instead of "1037")
7. `systemctl daemon-reload`
8. `systemctl restart chad-ibgateway.service`
9. Verify: run `chad/tools/ibkr_gateway_version_check` (severity should flip
   to "info")
10. Monitor latency / stop_bus / failed services for 24h
11. If healthy, the old 1037 install can remain on disk for rollback
12. If unhealthy, rollback: revert systemd ExecStart to "1037",
    daemon-reload, restart

### Rollback plan
Both versions remain on disk after step 4. To rollback:
- Revert ExecStart in the systemd unit to the old build number
- `daemon-reload` + restart
- Verify the version check shows the old version restored

### Required tests after upgrade
- `chad/tools/ibkr_gateway_version_check` reports severity="info"
- Latency holds <2000ms for 24h post-upgrade (Fix A counter stays 0)
- No new STOP_BUS_TRIGGERED entries from broker_latency
- All 6 dependent services restart cleanly without manual intervention
- `runtime/ibkr_status.json::ok=True` persistently

### Acceptance criteria
- New Gateway version installed and live
- Old 1037 install preserved on disk for 7 days for rollback safety
- 24-hour clean uptime post-upgrade
- `chad-ibgateway-nightly-restart.timer` (Fix B) continues to function
- No regression in any of the audit-fixed HIGH items

### Operator approvals required
- Operator GO to download the new version from the IBKR portal
- Operator GO to edit `/etc/systemd/system/chad-ibgateway.service`
- Operator GO to `systemctl restart chad-ibgateway.service`

### Risks
- Config syntax may have changed in the new IBC version — release notes must
  be read carefully
- New version may introduce regressions; rollback path documented
- Brief service interruption during the restart (Fix A hysteresis mitigates a
  stop_bus trip)
- If the upgrade fixes the wedge, Fix B becomes "belt and suspenders" — still
  valuable as defense in depth

### No-live confirmation
This Pending Action does not authorize live trading.
`ready_for_live` must remain false.
`allow_ibkr_live` must remain false.
`allow_ibkr_paper` must remain true.
No broker orders may be placed or cancelled under this Pending Action.

### Sequencing
This PA depends on no other open PA. It can execute before or after the Fix B
Channel 1 install. Order is operator discretion.
