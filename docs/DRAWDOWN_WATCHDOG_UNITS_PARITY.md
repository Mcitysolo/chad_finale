# Drawdown-Publisher + IBKR-Watchdog Units — Repo Parity Record (W3B-3)

Status: REPO-TRACKING ONLY. Both unit pairs are ALREADY INSTALLED, enabled,
and firing on a 120s cadence at `/etc/systemd/system/`. Nothing in this
commit installs, modifies, or restarts anything (CLAUDE.md governance #6/#7).

## Why this exists

The 2026-07-22 W3B investigation found these four units existed ONLY in
`/etc/systemd/system` — zero repo copies. That is configuration drift by
construction: a rebuild from the repo would silently lose both publishers,
recreating the exact P1-3 (drawdown_state 10 days stale) / P1-4
(ibkr_watchdog_last 44 days stale) outages that were closed on 2026-06-17.
This mirrors the VaR pattern (`ops/systemd/chad-var-publisher.{service,timer}`
+ `docs/VAR_PUBLISHER_INSTALL.md`): the repo is the design record, the
operator installs.

## Files

| Repo file (ops/systemd/) | Installed at | Parity |
|---|---|---|
| chad-drawdown-publisher.service | /etc/systemd/system/ | byte-identical, verified 2026-07-22 (`cmp`) |
| chad-drawdown-publisher.timer | /etc/systemd/system/ | byte-identical, verified 2026-07-22 |
| chad-ibkr-watchdog.service | /etc/systemd/system/ | byte-identical, verified 2026-07-22 |
| chad-ibkr-watchdog.timer | /etc/systemd/system/ | byte-identical, verified 2026-07-22 |

The repo copies deliberately mirror the installed reality INCLUDING its
imperfections — see "Known deltas vs the VaR pattern" below. Improvements go
through a Pending Action, not through silently-divergent repo copies.

## Known deltas vs the VaR pattern (NOT applied — proposed in a PA)

1. **No `OnFailure=chad-service-alert@%N.service`** on either service. The VaR
   publisher has it (P0A-A2: publisher deaths page). Until an operator applies
   the PA below, a crashing drawdown/watchdog publisher only surfaces via the
   W3B-1 freshness rows (EXS1 + feed_watchdog), not via an immediate page.
   → `ops/pending_actions/PA_onfailure_drawdown_watchdog_2026-07-22.md`
2. The drawdown service lacks the VaR service's sandbox hardening
   (`ProtectSystem=full`, `PrivateTmp`, `NoNewPrivileges`); the watchdog
   service already has it. Same PA covers it.

## Freshness nets watching these artifacts (W3B-1)

- `config/exterminator.json` feeds: `drawdown_state` (warn 300 / fail 900),
  `ibkr_watchdog_last` (warn 360 / fail 720, ts_unix).
- `chad/ops/feed_watchdog.py` WATCHED_FEEDS: both, mtime-based, same warn TTLs.

## Operator: keeping parity

If you change the installed units, update the repo copies in the same change
(or vice versa via install). Parity check, read-only:

    for u in chad-drawdown-publisher.service chad-drawdown-publisher.timer \
             chad-ibkr-watchdog.service chad-ibkr-watchdog.timer; do
      cmp "ops/systemd/$u" "/etc/systemd/system/$u" && echo "OK $u"
    done

## Operator: fresh-host install (disaster recovery only — units are already live)

    sudo cp ops/systemd/chad-drawdown-publisher.{service,timer} /etc/systemd/system/
    sudo cp ops/systemd/chad-ibkr-watchdog.{service,timer} /etc/systemd/system/
    sudo systemctl daemon-reload
    # run each service ONCE by hand and inspect the artifact + journal:
    sudo systemctl start chad-drawdown-publisher.service
    sudo systemctl start chad-ibkr-watchdog.service
    cat runtime/drawdown_state.json; cat runtime/ibkr_watchdog_last.json
    # only then arm the timers:
    sudo systemctl enable --now chad-drawdown-publisher.timer chad-ibkr-watchdog.timer
    # confirm the W3B-1 sentinel rows clear on the next 5-min sentinel run.

Rollback: `sudo systemctl disable --now <timer>` then remove the copies and
`daemon-reload`. The repo copies stay — they are the record.
