# CHAD Systemd Drop-in Archive

These files are copies of the active systemd drop-in overrides from
/etc/systemd/system/{service}.service.d/ on the production server.

## Deployment

To restore on a new server, for each file named
{service}.service.d_{conf-filename}:

1. Create directory: sudo mkdir -p /etc/systemd/system/{service}.service.d/
2. Copy file: sudo cp {file} /etc/systemd/system/{service}.service.d/{conf-filename}
3. After all files are copied: sudo systemctl daemon-reload

## Critical drop-ins

- chad-live-loop.service.d_10-always-active-routing.conf: CHAD_ALWAYS_ACTIVE_ROUTING=1
- chad-live-loop.service.d_20-execution-mode.conf: CHAD_EXECUTION_MODE=paper
- chad-orchestrator.service.d_40-allocator-v3.conf: Allocator v3 configuration
- chad-orchestrator.service.d_20-beta.env.conf: Beta sleeve configuration

## Last archived
2026-04-10
