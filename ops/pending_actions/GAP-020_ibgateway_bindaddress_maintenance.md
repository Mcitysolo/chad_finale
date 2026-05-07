# GAP-020 — IB Gateway localhost-only socket bind hardening

Status: Deferred maintenance
Risk level: Low
Created: 2026-05-07

## Current state

IB Gateway currently listens on wildcard port 4002, but external access is protected by:

- UFW allowing 4002 only on loopback
- UFW denying 4002 globally
- /home/ubuntu/Jts/jts.ini containing TrustedIPs=127.0.0.1
- /home/ubuntu/Jts/jts.ini containing ApiOnly=true

## Recommended maintenance fix

During the next planned IB Gateway maintenance window:

1. Back up /etc/chad/ibc/config.ini
2. Add BindAddress=127.0.0.1 to /etc/chad/ibc/config.ini
3. Confirm TrustedIPs=127.0.0.1
4. Restart only chad-ibgateway.service
5. Verify socket binding with: sudo ss -tulpn | grep 4002

Expected after restart:

127.0.0.1:4002

Not:

*:4002

## Do not apply during active paper-soak unless needed

This is defense-in-depth, not an emergency. Current protection is already layered through UFW + TrustedIPs + ApiOnly.
