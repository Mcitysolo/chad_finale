# PA — OnFailure paging + hardening for drawdown-publisher / ibkr-watchdog units

- Status: PROPOSED (W3B-3, 2026-07-22). Awaiting operator GO + manual apply.
- Class: systemd unit modification — governance #6/#7 territory, operator-only.
- Prereq reading: docs/DRAWDOWN_WATCHDOG_UNITS_PARITY.md (repo copies are
  byte-identical to installed; this PA is the recorded improvement delta).

## Problem

`chad-drawdown-publisher.service` and `chad-ibkr-watchdog.service` lack
`OnFailure=chad-service-alert@%N.service`. The VaR publisher gained it in
P0A-A2 ("publisher deaths page"); these two 120s-cadence publishers did not.
A crash-looping publisher today surfaces only through artifact staleness
(W3B-1 rows: warn at 300/360s, fail at 900/720s) — minutes of delay and no
crash context, versus an immediate OnFailure page with the unit name.

Secondary: chad-drawdown-publisher.service has no sandbox hardening, unlike
both the VaR publisher and the watchdog service.

## Proposed delta (apply to BOTH the installed units and the repo copies)

`chad-drawdown-publisher.service` — add under `[Unit]`:

    OnFailure=chad-service-alert@%N.service

and under `[Service]` (mirroring chad-var-publisher.service):

    NoNewPrivileges=true
    PrivateTmp=true
    ProtectSystem=full
    ReadWritePaths=/home/ubuntu/chad_finale/runtime

`chad-ibkr-watchdog.service` — add under `[Unit]`:

    OnFailure=chad-service-alert@%N.service

(Sandbox lines already present.)

## Apply sequence (operator, one unit pair at a time)

1. Edit the repo copy under `ops/systemd/` first; commit.
2. `sudo cp` to `/etc/systemd/system/`, `sudo systemctl daemon-reload`.
3. `sudo systemctl start <service>` once by hand; verify artifact updates and
   journal is clean; verify `cmp` parity repo-vs-installed.
4. Negative test (optional but recommended): temporarily break the ExecStart,
   start once, confirm the chad-service-alert@ page fires, restore, re-verify.

## Rollback

Remove the added lines from the installed unit, `daemon-reload`; revert the
repo-copy commit. Parity check per the parity doc.
