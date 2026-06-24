# PA — "Paper Account Notice" auto-dismiss watcher (2026-06-24)

**Status:** Pending Action — AUTHORED ONLY. Nothing installed, armed, reloaded,
or restarted by this task. Files staged in-repo; arming is a separate operator
step behind explicit GO.

**Trading posture unchanged:** PAPER. This PA introduces no order, broker,
config, or runtime mutation. The watcher is read-mostly (X11 queries + at most
one click on a single named dialog) and writes exactly one JSON artifact per run.

> ⚠️ **THIS PA TOUCHES A LIVE SERVICE PATH.** The drop-in, once installed, runs
> as an `ExecStartPost` of the **live** `chad-ibgateway.service` (paper Gateway,
> the broker-truth feed). Installing/arming changes what happens on every
> Gateway (re)start. Treat install + arm as a change to a live unit; do it only
> under GO, during a maintenance window, with the verify gates below.

---

## Problem

`chad-ibgateway.service` runs IBC 3.23.0 + IB Gateway 10.37 (build 1037) headless
on Xvfb `:99`. After a successful "Paper Log In", IBKR shows a modal titled
exactly **`Paper Account Notice`**. IBC 3.23.0's `AcceptNonBrokerageAccountWarning`
title-match does not recognise this renamed dialog, so it is never auto-clicked.
While the modal is up it blocks API port **4002** from opening — i.e. the broker
feed never comes up until the dialog is dismissed by hand.

Live confirmation (read-only probe, 2026-06-24, `DISPLAY=:99`):
- `xdotool search --name "Paper Account Notice"` → window id `12583125`
- `xdotool getwindowgeometry 12583125` →
  `Position: 638,453 (screen: 0)` / `Geometry: 644x175`
- single `OK` button, bottom-centre.

## Files authored (the entire change surface)

| File | Role |
|---|---|
| `scripts/dismiss_paper_account_notice.py` | Bounded, idempotent watcher/clicker (stdlib + xdotool). |
| `tests/test_dismiss_paper_account_notice.py` | Hermetic unit tests (geometry/parse/artifact). |
| `systemd/chad-ibgateway-drop-in/60-paper-notice-autodismiss.conf` | **Staged** drop-in (NOT installed). |
| `ops/pending_actions/PA_paper_notice_autodismiss_2026-06-24.md` | This document. |

Nothing else is modified. `scripts/post_gateway_restart_verify.py` is **NOT**
touched (it remains read-only health-only).

---

## Watcher behaviour (summary)

- Reads `DISPLAY` from env (default `:99`). Nothing about the on-screen position
  is hard-coded — geometry is read live each tick.
- Poll budget **120s**, interval **3s**. Each tick: `search --name
  "Paper Account Notice"`; if found → `getwindowgeometry` → parse
  `Position`/`Geometry` → click target = `(x + w//2, y + h - 22)` (bottom-centre,
  derived live; for the live window this is **(960, 606)**).
- **Two safety gates before any pointer/key action:**
  1. **ARM gate** — clicking happens only if `CHAD_NOTICE_DISMISS_ARMED=1`.
     Unset/!= "1" ⇒ DRY-RUN: geometry math + log only, **no click**.
  2. **Title-equality gate** — immediately before clicking, the window id is
     re-confirmed to still resolve to the exact title via `search` only
     (`_assert_target_window`). The only xdotool verbs the script ever calls are
     `search`, `getwindowgeometry`, `mousemove`, `click`, `windowactivate`,
     `key Return`. It is structurally impossible to act on any other window.
- Terminal outcomes (all **exit 0** — must never crash the Gateway start):
  `DISMISSED` (clicked, window gone), `NOT_SEEN` (no notice in 120s — benign
  normal-login path), `STUCK` (seen but still open after 3 attempts — non-fatal),
  `DRY_RUN` (seen while un-armed; self-test/observe-only).
- **Only write:** one file `reports/gateway_paper_notice_log/<UTC_BASIC_ISO>.json`
  (dir created if absent). No network, no broker/order/config/runtime mutation.

---

## Type= / TimeoutStartSec analysis (required)

Facts from `/etc/systemd/system/chad-ibgateway.service`:
- **`Type=simple`** (line 9). For `Type=simple`, systemd considers the main
  process started as soon as `ExecStart` (line 32) is forked, then runs
  `ExecStartPost` during the `start-post` phase. **So an `ExecStartPost` fires
  immediately after the Gateway is spawned — exactly while the login + dialog are
  happening. This is the correct hook point** (the watcher then polls for up to
  120s, covering the dialog's appearance). No `Type=` change or backgrounding is
  required for *timing*.
- **No explicit `TimeoutStartSec`** in the unit ⇒ it inherits
  `DefaultTimeoutStartUSec` = **90s** (`systemctl show -p DefaultTimeoutStartUSec`
  → `1min 30s`; `systemctl show chad-ibgateway.service -p TimeoutStartUSec` →
  `1min 30s`). The `start-post` phase is bounded by this start timeout.
- **`Restart=always`** (line 34).

**Interaction (the one real footgun):** a *synchronous* `ExecStartPost` that runs
the full 120s budget would exceed the 90s start timeout. The 120s path only
happens on `NOT_SEEN` (notice never appears); in the case this PA exists to fix
(notice present), the watcher dismisses and exits in seconds — far under any
timeout. But to make the `NOT_SEEN` worst case safe, the staged drop-in sets
**`TimeoutStartSec=180`** (a safe-direction change: it only lengthens how long
systemd waits before declaring start failure). Without it, a `NOT_SEEN` run could
trip a 90s start-timeout and, with `Restart=always`, loop.

**Decision: keep `Type=simple`, synchronous `ExecStartPost`, and raise
`TimeoutStartSec` to 180.** No backgrounding (`&`/`setsid`) — backgrounding from
`ExecStartPost` is killed with the unit's cgroup on stop/restart and loses the
exit-status/journal visibility a synchronous post gives us.

**Dependency-delay note (blast radius):** while `ExecStartPost` runs, the unit is
in `activating (start-post)`, so reverse-dependents that order `After=` it wait
until the watcher exits. Reverse deps today (`systemctl list-dependencies
--reverse chad-ibgateway.service`): `chad-ibkr-collector`, `chad-ibkr-health`,
`chad-positions-snapshot`, `chad-options-chain-refresh`, `chad-portfolio-snapshot`,
`chad-ibkr-daily-bars-refresh`, `chad-ibgateway-nightly-restart`. In the dismiss
case the added delay is seconds; in `NOT_SEEN` it is up to 120s. Acceptable, but
operator should be aware on restart.

---

## Consequence Block

- **What changes when INSTALLED (and the unit reloaded):** every
  `chad-ibgateway.service` (re)start gains an `ExecStartPost` that runs the
  watcher and raises `TimeoutStartSec` 90s→180s. Until armed, the watcher is
  observe-only (DRY-RUN) — it logs geometry and writes an artifact but does **not**
  click. Once armed (`CHAD_NOTICE_DISMISS_ARMED=1`), it clicks the `OK` button of
  the `Paper Account Notice` dialog (and only that dialog) to dismiss it.
- **Blast radius:**
  - *Scope of action:* a single click + Return on one window whose title is
    asserted (twice) to equal `Paper Account Notice`, at coordinates derived live
    from that window. No other window can be targeted by construction.
  - *Service:* live `chad-ibgateway.service` start sequence. A bug that hangs is
    bounded by the 120s budget + 10s per-subprocess timeout, and capped by
    `TimeoutStartSec=180`. The script returns 0 on every path, so it cannot mark
    the unit failed.
  - *Filesystem:* one JSON per run under `reports/gateway_paper_notice_log/`.
  - *Reverse-dependents:* delayed by the watcher's runtime during start (see
    dependency-delay note).
- **What does NOT change:** no broker/order/config/runtime/network mutation; no
  posture change (stays PAPER); `post_gateway_restart_verify.py` untouched; the
  existing `50-onfailure-alert.conf` drop-in untouched.
- **Reversibility:** fully reversible. The drop-in is additive; removing it
  restores the prior unit byte-for-byte. The watcher leaves no state except
  append-only artifact files (safe to delete).
- **Rollback:**
  ```bash
  sudo rm /etc/systemd/system/chad-ibgateway.service.d/60-paper-notice-autodismiss.conf
  sudo systemctl daemon-reload
  # (optional) confirm the override is gone:
  systemctl cat chad-ibgateway.service | grep -c 60-paper-notice-autodismiss   # expect 0
  ```
  No Gateway restart is required to roll back; the override simply stops applying
  to the next start. To also revert in-flight, restart the Gateway under GO.

---

## Install procedure (STAGED — behind explicit GO)

Pre-install gates (all must pass):
```bash
cd /home/ubuntu/chad_finale && source venv/bin/activate
python3 -m py_compile scripts/dismiss_paper_account_notice.py
python3 -m pytest tests/test_dismiss_paper_account_notice.py -q          # expect green
# Prove the watcher is harmless un-armed against the live dialog (no click):
CHAD_NOTICE_DISMISS_ARMED= DISPLAY=:99 \
  venv/bin/python3 scripts/dismiss_paper_account_notice.py               # expect status=DRY_RUN, click_target [960,606]
```

Install (root):
```bash
sudo install -m 0644 -o root -g root \
  /home/ubuntu/chad_finale/systemd/chad-ibgateway-drop-in/60-paper-notice-autodismiss.conf \
  /etc/systemd/system/chad-ibgateway.service.d/60-paper-notice-autodismiss.conf
sudo systemctl daemon-reload
systemctl cat chad-ibgateway.service            # confirm the ExecStartPost + TimeoutStartSec=180 now appear
```

Arm (SECOND, deliberate step — only after observing ≥1 clean DRY-RUN install):
```bash
# Uncomment the Environment line in the installed drop-in, then:
sudo sed -i 's/^#Environment=CHAD_NOTICE_DISMISS_ARMED=1/Environment=CHAD_NOTICE_DISMISS_ARMED=1/' \
  /etc/systemd/system/chad-ibgateway.service.d/60-paper-notice-autodismiss.conf
sudo systemctl daemon-reload
```

Activate / verify (requires a Gateway restart — **explicit GO only**, per
governance rule 7):
```bash
sudo systemctl restart chad-ibgateway.service
# Post-verify gates:
journalctl -u chad-ibgateway.service -b --no-pager | grep dismiss_paper_account_notice
ls -t /home/ubuntu/chad_finale/reports/gateway_paper_notice_log/ | head -1   # newest artifact
#  -> status should be DISMISSED (or NOT_SEEN on a login with no dialog)
# Confirm API port opened (the whole point):
ss -ltnp | grep 4002 || echo "4002 not yet listening — investigate"
python3 scripts/post_gateway_restart_verify.py    # existing read-only health check, must pass
```

---

## Explicit-path git staging (NO `git add -A`)

```bash
cd /home/ubuntu/chad_finale
git add scripts/dismiss_paper_account_notice.py
git add tests/test_dismiss_paper_account_notice.py
git add systemd/chad-ibgateway-drop-in/60-paper-notice-autodismiss.conf
git add ops/pending_actions/PA_paper_notice_autodismiss_2026-06-24.md
git status --porcelain   # confirm ONLY the four paths above are staged
```

Suggested commit (author-only; commit when the operator approves):
```
feat(ops): staged auto-dismiss watcher for IBC 'Paper Account Notice' modal (inert)
```

---

## Notes / residual

- The bare `ExecStartPost` line specified for the drop-in is preserved verbatim;
  `TimeoutStartSec=180` is added alongside it (active in the staged file but inert
  until installed) because the 90s default is shorter than the 120s watcher budget
  — see the Type= analysis. If the operator prefers not to change the start
  timeout, the alternative is to lower the watcher's `NOT_SEEN` budget below 90s;
  that is a code change and is intentionally NOT done here.
- The watcher avoids `xdotool getwindowname` deliberately — it is outside the
  permitted verb set; title equality is asserted using `search` only.
- Sibling artifact-log convention matched: filenames use basic ISO
  (`YYYYMMDDTHHMMSSZ.json`, as in `reports/gateway_restart_log/`); the `ts_utc`
  field inside the JSON is canonical extended ISO.
