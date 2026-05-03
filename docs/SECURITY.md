# CHAD Security Policy

This document defines the canonical security expectations for CHAD —
specifically how secrets are stored and rotated, how the dashboard
authenticates operators, how live/paper promotion is gated, and how
broker connections are scoped at process boundaries.

It is the source of truth for SSOT items SS02 (secret handling) and
SS03 (dashboard auth requirements) and complements the OPS runbook.

---

## 1. Secret handling (SS02)

### 1.1 Storage

All operator secrets live under `/etc/chad/` as `EnvironmentFile=`
inputs to systemd units. The runtime never reads secrets from the
repository, runtime/, or any user shell history.

| File                       | Contains                                    |
| -------------------------- | ------------------------------------------- |
| `/etc/chad/telegram.env`   | Telegram bot token + admin chat IDs         |
| `/etc/chad/claude.env`     | Anthropic API key (Claude advisor)          |
| `/etc/chad/chad.env`       | Execution mode flags + non-credential env   |
| `/etc/chad/ibkr.env`       | IBKR account credentials / clientId base    |

### 1.2 Permissions

All `/etc/chad/*.env` files MUST be:

- Owner: `ubuntu:ubuntu`
- Mode: `600` (owner read/write only)
- Never world-readable, never group-readable

Verification one-liner:

```bash
ls -l /etc/chad/*.env
```

If any file shows other than `-rw-------` for `ubuntu:ubuntu`, fix
immediately with `chmod 600` and `chown ubuntu:ubuntu`.

### 1.3 Source control

`/etc/chad/` is outside the repository tree by design. The repo's
`.gitignore` excludes any `*.env` file at any depth so an accidental
copy into `chad_finale/` cannot be committed. Never paste a token into
a markdown file, a comment, a commit message, or a GitHub issue.

### 1.4 Rotation

| Secret                | Rotation cadence                        |
| --------------------- | --------------------------------------- |
| Telegram bot token    | Annually, or immediately on suspicion   |
| Anthropic API key     | Per Anthropic key-management guidance   |
| IBKR credentials      | Per IBKR / IB policy                    |

Rotation procedure:

1. Generate new credential at the provider's console.
2. Edit `/etc/chad/<file>.env` in place; preserve mode/owner.
3. `sudo systemctl restart` only the units that consume that file.
4. Confirm the new credential is live (Telegram echo, IBKR connect,
   Claude advisor heartbeat) before revoking the old one.
5. Revoke the old credential at the provider's console.

### 1.5 Incident response

If a secret is suspected leaked (committed to git, posted in a chat,
appears in a journalctl excerpt shared externally):

1. **Revoke immediately** at the provider dashboard — don't wait.
2. Issue a replacement credential.
3. Apply per the rotation procedure above.
4. Audit `journalctl --since "<leak time>"` for unauthorized usage
   between leak and revocation.
5. File a postmortem entry under `docs/` — what leaked, when, the
   blast radius, and the control change to prevent recurrence.

---

## 2. Dashboard authentication (SS03)

The CHAD operator dashboard is the only externally reachable web
surface. Its auth model:

### 2.1 Mechanism

- HTTP Basic over TLS for the initial credential challenge.
- On success, server issues a server-side session token: 32 bytes of
  `secrets.token_urlsafe()`-grade randomness, stored server-side and
  referenced by an opaque cookie.
- All subsequent requests authenticate via the cookie; the Basic
  challenge is not re-presented inside the session window.

### 2.2 Session lifetime

- TTL: 24 hours from issuance.
- Renewal: a fresh session is issued on the next interactive login;
  there is no sliding refresh.
- Logout: the server-side session token is deleted, invalidating the
  cookie everywhere — not just in the originating browser.

### 2.3 Cookie flags

- `HttpOnly` — JavaScript cannot read the cookie.
- `SameSite=Strict` — provides the primary CSRF defence.
- `Secure` — sent over TLS only.

### 2.4 CSRF posture

`SameSite=Strict` is sufficient because the dashboard performs no
cross-origin state-changing requests. State-changing endpoints
additionally require an explicit POST and reject GET.

### 2.5 Brute-force defence

- 5 failed attempts within a 5-minute window from a single source IP
  triggers a 5-minute lockout for that IP.
- Lockout state is in-memory; the dashboard service runs single-
  instance behind systemd, so a process restart resets counters by
  design.
- Successful logins reset the counter for that IP.

### 2.6 Audit trail

- Every login attempt — success or failure — is emitted to the
  systemd journal via `SyslogIdentifier=chad-dashboard`.
- Logs include IP, outcome, and timestamp; never the credential.
- Routine review: `journalctl -u chad-dashboard | grep AUTH`.

---

## 3. Live / paper promotion workflow

Mode promotion is the most security-sensitive operation in the system
and requires explicit operator action — no automation flips this flag.

### 3.1 Required state

- Operator physically present (terminal, not chat).
- LiveGate green (see `runtime/live_readiness.json`).
- Reconciliation green.
- No open paper positions the operator hasn't reviewed.

### 3.2 Procedure

1. **Stage** — set `CHAD_EXECUTION_MODE=ibkr_paper` in
   `/etc/chad/chad.env` and restart `chad-live-loop.service`.
2. **Verify** — confirm LiveGate accepts and the loop runs cleanly
   for at least one full execution cycle.
3. **Promote** — set `CHAD_EXECUTION_MODE=ibkr_live` and restart.
4. **Record** — commit a change to repo with the promotion timestamp
   and operator name in the commit body. The commit is the audit
   record; do not amend or force-push it.

### 3.3 Rollback

If anything looks wrong post-promotion:

1. `sudo $EDITOR /etc/chad/chad.env` — revert the mode line.
2. `sudo systemctl restart chad-live-loop.service`.
3. Confirm in the journal that the loop is back in paper mode.
4. File an incident note before resuming any work.

---

## 4. Import-time broker connections

### 4.1 Policy

CHAD modules MUST NOT open network connections at import time. All
IBKR / Coinbase / Kraken connect calls must live inside an explicit
bootstrap function (e.g. `connect()`, `start()`, `run()`), never at
module top level.

This protects:

- Test runs (which set `CHAD_SKIP_IB_CONNECT=1` to short-circuit any
  remaining import-time hooks).
- Tooling that imports CHAD modules to read schema or call helpers
  without intending to talk to a broker.
- Recovery scenarios where the broker is intentionally unavailable.

### 4.2 Current status

- `CHAD_SKIP_IB_CONNECT=1` is required for the test suite. Setting
  it disables any residual import-time connection paths.
- A future cleanup will move the last remaining `ib.connect()` calls
  into an explicit bootstrap function so the env-var workaround can
  be retired.

### 4.3 How to add a new broker integration

1. Place the SDK import at module top-level — that's fine.
2. Place the connect / authenticate call inside a function the
   process must call deliberately.
3. Add a CI check (or `pytest` import smoke test) that imports the
   module with all broker env vars unset and confirms no socket is
   opened.
