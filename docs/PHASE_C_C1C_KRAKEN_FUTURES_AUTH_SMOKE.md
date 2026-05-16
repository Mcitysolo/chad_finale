# Phase C Item 1C — Kraken Futures Authenticated Smoke-Test Scaffold

## Status
AUTHENTICATED SMOKE TEST SCAFFOLD ONLY.

This is a read-only diagnostic tool. It is not authorized for live trading,
is not wired into strategies, execution routing, `chad.core.live_loop`, or
any systemd unit, and **never places orders**.

## What it does
- Detects whether `KRAKEN_FUTURES_API_KEY` and `KRAKEN_FUTURES_API_SECRET`
  are present (env var or `/etc/chad/kraken.env`).
- Validates that `KrakenFuturesClient` can be constructed in dry-run mode
  with no credentials and reports the expected fail-closed state.
- Refuses speculative private network calls. A read-only private endpoint
  must be explicitly certified before `--live-readonly` will succeed.
- Emits machine-readable status lines: `KRAKEN_FUTURES_AUTH_SMOKE status=<status>`.
- Exits with distinct codes per state (0 ok, 1 unexpected, 2 missing creds,
  3 endpoint not certified).

## What it does not do
- No order placement.
- No `submit_order` calls.
- No strategy routing.
- No execution-pipeline integration.
- No systemd or service registration.
- No live trading unlock.
- No secret values are printed to stdout or stderr.

## CLI
```
python3 -m chad.tools.kraken_futures_auth_smoke              # default mode
python3 -m chad.tools.kraken_futures_auth_smoke --dry-run     # construct-only
python3 -m chad.tools.kraken_futures_auth_smoke --live-readonly
```

## Exit codes
| Code | Meaning |
|------|---------|
| 0    | `dry_run_ok` or (future) `live_readonly_ok` |
| 1    | Unexpected error |
| 2    | `missing_credentials` |
| 3    | `credentials_present_endpoint_not_certified` |

## Operator unlock steps
1. Create a Kraken Futures API key/secret pair in the Kraken Futures UI.
   Use the most restrictive permissions possible (no withdrawal, no trade
   until certified).
2. Add to `/etc/chad/kraken.env`:
   ```
   KRAKEN_FUTURES_API_KEY=...
   KRAKEN_FUTURES_API_SECRET=...
   ```
3. Run a key-name verification only (no values printed):
   ```
   sudo grep -E '^(KRAKEN_FUTURES_API_KEY|KRAKEN_FUTURES_API_SECRET)=' \
        /etc/chad/kraken.env | sed 's/=.*/=<REDACTED>/'
   ```
4. Run the dry-run smoke (no credentials needed):
   ```
   python3 -m chad.tools.kraken_futures_auth_smoke --dry-run
   ```
5. After a read-only Kraken Futures private endpoint has been certified in
   code (a follow-up Phase C item), only then run:
   ```
   python3 -m chad.tools.kraken_futures_auth_smoke --live-readonly
   ```
6. **Never paste API keys into chat or commit them anywhere.**

## Current expected behavior
- With no Futures keys configured:
  - Default mode  -> `status=missing_credentials`, exit 2.
  - `--dry-run`   -> `status=dry_run_ok`, exit 0.
  - `--live-readonly` -> `status=missing_credentials`, exit 2.
- With Futures keys configured but no certified endpoint:
  - Default mode  -> `status=credentials_present_endpoint_not_certified`, exit 3.
  - `--live-readonly` -> `status=credentials_present_endpoint_not_certified`, exit 3.

## Future work
- Certify a single Kraken Futures read-only private endpoint (e.g. account
  status / open positions).
- Replace the placeholder `perform_private_readonly_probe` with a call into
  that certified endpoint via `KrakenFuturesClient`.
- Run the live-readonly smoke under operator supervision before any further
  execution-pipeline integration.
