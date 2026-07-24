# PENDING ACTION — W6B-13: hold cancel-entry-orders

**Status:** BUILT, DARK. Default OFF. Requires operator GO **per invocation**.
**Type:** new operator tool + one-line order-guard extension.
**Risk:** **HIGH** — this cancels working orders at the broker.
**Prepared:** 2026-07-24 (W6B-13, INCIDENT-0723 item 4)

---

## Scope of this PA

This PA covers **enabling the flag for one specific hold application**. It is not a
standing authorisation. The design, tests, and dry-run path land dark under
`CHAD_HOLD_CANCEL_ENTRIES` (unset → the step cannot execute).

## The gap

Both INCIDENT-0723 D4 defects are fixed: a hold **persists**
(`operator_intent_refresher._unexpired_hold`) and is **consulted**
(`live_gate._load_operator_intent`, failing closed to `DENY_ALL`). Neither reaches orders
**already working at the broker**. `live_gate`/`operator_intent` are admission control on
*new* intents; an order submitted seconds before the hold keeps living at IBKR and can
still fill.

The incident record's own language — *"the hard brake remains operator-side: `systemctl
stop chad-live-loop`"* — is an admission that no in-band brake reaches working orders.

## The two traps, and how each is handled

`chad/core/paper_position_closer.py:359-367` already cancels orders. Reusing it as-is
would be actively dangerous:

**Trap 1 — clientId-scoped enumeration.** It uses `ib.openOrders()`, which returns nothing
for other clients' orders **without erroring**. A hold-cancel built on it would report
*"cancelled 0, all clear"* while orders under another clientId keep working. This is the
single most likely way to produce a confident, wrong success, so it is the primary
correctness requirement here, not a footnote.

→ `enumerate_all_open_orders()` uses `reqAllOpenOrders*` and **aborts** if unavailable. It
never falls back. `test_aborts_when_only_the_clientid_scoped_call_exists` pins it, and
every report stamps `client_ids_seen` as positive evidence the sweep really was
cross-client.

**Trap 2 — indiscriminate cancellation.** Its loop cancels *every* open order. In a flatten
that is intended; applied to a hold it strips protective and exit orders and leaves
positions naked — strictly worse than the problem being solved.

→ Entry-only, fail-closed. An order is cancelled **only** if positively classified ENTRY.

## Classification has no CHAD tag to lean on

Worth stating plainly because it shaped the whole design: **CHAD does not set `orderRef` on
broker orders**, so an order enumerated from the broker carries no `intent_class`.
Entry-vs-exit cannot be looked up — it is inferred from broker-visible attributes plus the
position book. Every inference failure resolves to *leave it alone*.

Protective checks run **before** any entry check, so an order that looks like both is
protective:

| Signal | Verdict | Cancelled? |
|---|---|---|
| `parentId != 0` (bracket child) | protective | no |
| non-empty `ocaGroup` | protective | no |
| orderType ∈ {STP, STP LMT, TRAIL, TRAIL LIMIT, MIT, LIT, …} | protective | no |
| orderType not in {LMT, MKT, MIDPRICE, REL} | unknown | no |
| **positions unavailable** | unknown | no |
| SELL against a long / BUY against a short (any size) | reduce_only | no |
| plain LMT/MKT that opens or increases exposure | **entry** | **yes** |

Two of those deserve emphasis:

- **Positions unavailable ⇒ cancel nothing.** Reading "no positions" as "flat" would make
  every SELL look like an entry — the most dangerous misreading available here. `_positions_probe_ok`
  separates *known-flat* from *could-not-read*.
- **An oversized opposing order is still reduce_only, not an entry.** A flip's closing half
  is an exit; treating it as an entry would strip the close.

## Two defects the tests caught during the build

Both were the exact failure class this tool exists to avoid, and both are now pinned:

1. **`complete` on zero cancels.** The outcome was `executed and not failed and
   len(cancelled) == len(cancellable)`. With nothing cancellable that is `0 == 0` → the
   report said **"complete"** after cancelling nothing. Now `complete` requires that
   something was actually cancelled; zero cancels is always `nothing_cancelled`, with a
   loud note that it is *"NOT proof the book is clear"*.
2. **Unparseable orders vanishing.** A broker object of unexpected shape sailed through the
   `getattr` defaults into an all-zero `OrderRow`. It was still left alone (safe), but it
   disappeared from `orders_unparseable`, so an operator would never learn the broker
   returned something the tool did not understand. Now shape-checked and recorded.

## Gates

| Gate | Behaviour |
|---|---|
| `CHAD_EXECUTION_MODE` ∉ {paper, dry_run} | **abort**, even for a plan |
| `CHAD_HOLD_CANCEL_ENTRIES` ≠ 1 | `--execute` aborts; **`--plan` still works** |
| default mode | `--plan` (dry run) |
| Channel-1 | added to `.claude/hooks/chad-order-guard.sh` — operator-invoked in the terminal, never agent-invoked |

Planning deliberately does **not** require the flag: an operator must always be able to see
what *would* be cancelled without arming anything.

## Evidence artifact

`runtime/hold_cancel_report.json` (`hold_cancel_report.v1`): every order seen, its verdict
and the reason for it, per-order action and result, `client_ids_seen`, explicit
`not_cancelled` reasons, and `loud_notes`. A partial or zero-cancel outcome is loud by
construction.

## How to use

```bash
# 1. Always plan first. No flag needed.
cd /home/ubuntu/chad_finale && source venv/bin/activate
python3 -m ops.hold_cancel_entries --plan

# 2. Read runtime/hold_cancel_report.json. Confirm:
#    - client_ids_seen covers every client you expect to have working orders
#    - every `cancellable: true` row is genuinely an entry
#    - not_cancelled reasons look right

# 3. Only then, for THIS hold application:
CHAD_HOLD_CANCEL_ENTRIES=1 python3 -m ops.hold_cancel_entries --execute
```

## Explicitly out of scope

No change to hold semantics, `live_gate`, `operator_intent`, or the exit/protective path.
This *adds* an action to hold application; it does not alter what a hold means.

**D7 stands: entry-only.** Cancelling protective orders on `DENY_ALL` is how a hold turns
into an incident. If that is ever wanted it should be its own explicit decision, not an
extension of this one.

## Rollback

Delete `ops/hold_cancel_entries.py` and revert the one-line guard change. Nothing else
references it — it is invoked only by an operator, never on a timer and never automatically.

## Evidence

| Claim | Source |
|---|---|
| Hold persists / is consulted | `chad/ops/operator_intent_refresher.py:347-359`; `chad/core/live_gate.py:420-426` |
| Hard brake is operator-side | `audits/INCIDENT_20260723_DRILL_EXHAUST_FALSE_FLAT.md:154-193` |
| Indiscriminate cancel mechanic | `chad/core/paper_position_closer.py:359-367` |
| clientId-scoped enumeration trap | `paper_position_closer.py:360` uses `ib.openOrders()`; standing IBKR probe methodology |
| Cross-client contract precedent | `scripts/flatten_all.py:262-271` (not read — chad-order-guard blocked access, which is the guard working correctly; reimplemented from the documented contract) |
| CHAD sets no `orderRef` | repo-wide search of `chad/execution/`, zero hits |
| Tests | `chad/tests/test_w6b_hold_cancel_entries.py` (35 tests, most asserting something is NOT cancelled) |
