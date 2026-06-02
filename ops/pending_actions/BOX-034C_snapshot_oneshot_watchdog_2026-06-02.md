---
# BOX-034C — Snapshot Oneshot Watchdog / Timeout (robustness follow-up)
Date: 2026-06-02 · Status: PENDING (no implementation until operator GO)
Risk class: robustness/observability

## 1. Problem (discovered 2026-06-02)
chad-portfolio-snapshot.service is a timer-driven oneshot with NO run timeout. On 2026-06-02 it hung in `activating` for ~13.5h — the process held an IBKR connection and received updatePortfolio streams but never completed its fetch-write-exit. Because the oneshot never finished, the timer could not re-fire (Trigger: n/a), so kraken_equity + ibkr_equity_usd_display were frozen/absent for ~13h. A manual `systemctl restart` cleared it; the fresh run completed cleanly in ~4s. The specific hang trigger is opaque (journal rotated past it).

## 2. Impact
- This time the casualty was the publisher (kraken + usd_display = de-minimis), because the collector independently owns the canonical ibkr_equity (BOX-034A Inc 1/2) and stayed fresh.
- LATENT RISK: any snapshot oneshot can hang identically. If chad-ibkr-collector hung this way, the canonical ibkr_equity (~99.9% of book) would freeze silently — a serious, currently-undetected failure mode.

## 3. Fix (proposed)
- Add a bounded run timeout to the snapshot oneshots (systemd TimeoutStartSec, e.g. 90-120s) so a hung run is SIGTERM'd and the timer re-fires on schedule. Apply to chad-portfolio-snapshot.service AND chad-ibkr-collector.service (and peer snapshot oneshots).
- Consider a code-level asyncio.wait_for around the IBKR connect/fetch so the process self-aborts rather than relying solely on systemd.
- Health-monitor staleness watchdog: alert when a snapshot file's mtime exceeds N x its expected period, so a frozen feed is surfaced, not silent.
- Likely in the Bug A async/event-loop hang family; a timeout is the robust mitigation regardless of the specific trigger.

## 4. Acceptance
- A deliberately-hung run is SIGTERM'd within the timeout and the timer re-fires on the next interval.
- Snapshot-staleness alert fires when mtime exceeds threshold.

## 5. Out of scope
The specific 13h-hang root cause (journal rotated; not reproducible on demand). Bug A's broader event-loop remediation.

## 6. Implementation gate
No systemd/code changes until operator GO.
---
