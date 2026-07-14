# INSTALL — OnFailure alert wiring for failure-prone oneshots (P0A-A2)

**Status:** Channel-2 design artifacts. These drop-ins are written to the repo
only. Nothing here is installed or reloaded automatically — the operator runs
the commands below during a maintenance window. Per CLAUDE.md governance rules
#6/#7, installed unit files and services are never modified/restarted by the
agent.

## What this does

Each drop-in adds one line to a failure-prone oneshot:

```ini
[Unit]
OnFailure=chad-service-alert@%N.service
```

When the unit enters `failed`, systemd instantiates the existing template
`chad-service-alert@<unit>.service` (already installed at
`/etc/systemd/system/chad-service-alert@.service`), which captures a journal
tail, writes a `reports/service_failures/*.json` artifact, and sends a Telegram
alert. After P0A-A1, a dedupe-suppressed duplicate no longer latches the
handler into `failed` (that was the 2026-06-27 root cause), and the artifact
records `telegram_sent` / `telegram_delivery_status` / `delivery_error`.

### `%N` vs `%n` (important)

Use `%N` (capital). `%N` is the failed unit name **without** its `.service`
suffix; the template's `ExecStart` re-appends it (`--failed-unit %i.service`),
yielding the correct `chad-service-alert@<unit>.service`. `%n` keeps the
suffix and would produce `chad-...service.service`. The one already-wired unit
(`chad-options-chain-refresh`) resolves to
`chad-service-alert@chad-options-chain-refresh.service`, confirming `%N`.

## Units covered (9)

Derived from the repo `ops/systemd/` tracked oneshots (all previously unwired)
plus census R43. All are `Type=oneshot`, all had `OnFailure=[]` before this:

| unit | source | note |
|---|---|---|
| `chad-ibkr-daily-bars-refresh` | census R43 (BROKEN_SILENT) | ran failed exit-1 ~17h, no OnFailure; stale bars read as fresh. Fix the exit-1 root cause separately (contract resolution). |
| `chad-ibkr-collector` | ops/systemd/ | broker collector; adds a 2nd drop-in next to `10-timeout-guards.conf` |
| `chad-kraken-collector` | ops/systemd/ | crypto market-data collector |
| `chad-kraken-pnl-watcher` | ops/systemd/ | crypto PnL watcher |
| `chad-intel-cache` | ops/systemd/ | intel cache refresh |
| `chad-brain-returns` | ops/systemd/ | returns/brain batch |
| `chad-clean-soak-evaluator` | ops/systemd/ | soak evaluator |
| `chad-lifecycle-replay-engine` | ops/systemd/ | lifecycle replay batch |
| `chad-portfolio-merge` | ops/systemd/ | **timer currently masked → /dev/null**; drop-in matters only if run manually |

> The base unit for `chad-ibkr-daily-bars-refresh` is not tracked in
> `ops/systemd/`; only its drop-in dir is provided here. `mkdir -p` in the
> commands below creates the target drop-in dir regardless.
>
> This set is intentionally scoped to the two enumeration sources named in the
> task (`ops/systemd/` + census). ~90 other installed oneshots remain unwired;
> the same one-line pattern applies to any the operator wants to add.

## Install (operator, in a maintenance window)

Copy each drop-in to its `/etc/systemd/system/<unit>.service.d/` dir, then
reload. Copy/paste as-is:

```bash
cd /home/ubuntu/chad_finale
for u in chad-brain-returns chad-clean-soak-evaluator chad-ibkr-collector \
         chad-ibkr-daily-bars-refresh chad-intel-cache chad-kraken-collector \
         chad-kraken-pnl-watcher chad-lifecycle-replay-engine chad-portfolio-merge; do
  sudo mkdir -p "/etc/systemd/system/${u}.service.d"
  sudo cp "ops/systemd/${u}.service.d/10-onfailure.conf" \
          "/etc/systemd/system/${u}.service.d/10-onfailure.conf"
done
sudo systemctl daemon-reload
```

### Also clear the latched alert handler (census R44)

The alert handler instance for the bar-provider is latched `failed` (exit-4)
since 2026-06-27. The exit-4 code path is fixed by P0A-A1, but the latched
state must be cleared once so future failures can re-instantiate it:

```bash
sudo systemctl reset-failed chad-service-alert@chad-ibkr-bar-provider.service
```

## Verify (after install)

```bash
# Each unit should now report the OnFailure dependency:
for u in chad-ibkr-daily-bars-refresh chad-ibkr-collector chad-kraken-collector \
         chad-kraken-pnl-watcher chad-intel-cache chad-brain-returns \
         chad-clean-soak-evaluator chad-lifecycle-replay-engine chad-portfolio-merge; do
  printf '%-40s ' "$u"
  systemctl show "$u.service" -p OnFailure --value
done
# Expected per unit: chad-service-alert@<unit>.service

# The handler instance should no longer be failed:
systemctl is-failed chad-service-alert@chad-ibkr-bar-provider.service   # -> inactive

# End-to-end (optional) — force a failure and confirm an artifact appears:
#   sudo systemctl start chad-ibkr-daily-bars-refresh.service   # if it fails, an artifact + Telegram should follow
#   ls -t reports/service_failures/ | head
```

## Rollback

```bash
for u in chad-brain-returns chad-clean-soak-evaluator chad-ibkr-collector \
         chad-ibkr-daily-bars-refresh chad-intel-cache chad-kraken-collector \
         chad-kraken-pnl-watcher chad-lifecycle-replay-engine chad-portfolio-merge; do
  sudo rm -f "/etc/systemd/system/${u}.service.d/10-onfailure.conf"
done
sudo systemctl daemon-reload
```
