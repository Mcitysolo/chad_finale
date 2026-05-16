# CHAD Phase C Status Lock — 2026-05-16

Documentation + audit checkpoint capturing the delta in Phase C work
after CHAD Unified SSOT v9.2 (commit `5392fc7`). This file does not
authorize any new code, runtime, or deploy change. It records which
Phase C items are live, which are scaffold-only, which are blocked,
and what the single next authorized action is.

---

## 1. Baseline

- Latest HEAD commit: `c9da3b9 Add Kraken Futures authenticated smoke scaffold`
- SSOT v9.2 anchor commit: `5392fc7 Docs: CHAD Unified SSOT v9.2 …`
- Working tree at start: clean (`git status --short` empty)
- Test baseline: **1934 passed** in 71.76s
  (`python3 -m pytest chad/tests/ -q`, with `CHAD_SKIP_IB_CONNECT=1`)
- `full_cycle_preview` result: clean — orders_count=0, total_notional=0.00,
  futures_orders=0, no broker calls (logical preview only)
- Trading posture: PAPER (CHAD_EXECUTION_MODE=paper)

---

## 2. Phase C commit delta after SSOT v9.2

Five Phase C commits landed since SSOT v9.2 (most recent first):

| Commit  | Subject                                                  | Item  |
|---------|----------------------------------------------------------|-------|
| c9da3b9 | Add Kraken Futures authenticated smoke scaffold          | C1C   |
| 69d3a95 | Add Phase C Kraken Futures adapter scaffold              | C1B   |
| 759fd32 | Align options Greeks TTL with daily refresh cadence      | maint |
| 0204cba | Add Phase C Kraken Futures public intel publisher        | C1A   |
| 42daa1f | Document Phase C IBKR DOM entitlement blocker            | C3    |

Net additions from this window:

- `chad/market_data/kraken_futures_intel_publisher.py`
- `chad/exchanges/kraken_futures_client.py`
- `chad/execution/kraken_futures_adapter.py`
- `chad/tools/kraken_futures_auth_smoke.py`
- `chad/tests/test_phase_c_item1_kraken_futures_intel.py`
- `chad/tests/test_kraken_futures_client.py`
- `chad/tests/test_kraken_futures_adapter.py`
- `chad/tests/test_kraken_futures_auth_smoke.py`
- `docs/PHASE_C_C1B_KRAKEN_FUTURES_ADAPTER_SCAFFOLD.md`
- `docs/PHASE_C_C1C_KRAKEN_FUTURES_AUTH_SMOKE.md`
- `docs/PHASE_C_C3_IBKR_DOM_BLOCKED_2026-05-15.md`
- One maintenance edit to `chad/utils/options_greeks_gate.py`
  (TTL aligned to 90000 s for daily refresh cadence).

---

## 3. C1 Kraken Futures status

### A. Public intelligence (C1A) — LIVE / operational

- Module: `chad/market_data/kraken_futures_intel_publisher.py`
- Runtime artifact: `runtime/kraken_futures_intel.json`
- Universe: broad 306-perp coverage from Kraken public derivatives feed.
- Cadence: 5-minute systemd timer
  (`chad-kraken-futures-intel-refresh.timer` — confirmed active and
  scheduled at the time of this checkpoint).
- Endpoint: read-only **public** Kraken Futures endpoint. No
  credentials touched. No order or private API calls.
- Status: live, healthy, intelligence-only. **No strategy consumer is
  wired to this feed yet** — it is intelligence layer only.

### B. Adapter scaffold (C1B) — SCAFFOLD ONLY / not wired

- Modules:
  - `chad/exchanges/kraken_futures_client.py` (class `KrakenFuturesClient`)
  - `chad/execution/kraken_futures_adapter.py` (class `KrakenFuturesAdapter`)
- Default posture: `dry_run=True`. No live routing path active.
- Importers in live code: **none.**
  Verified by grep over `chad/core`, `chad/strategies`,
  `chad/execution/` (excluding the adapter file itself): no live
  module imports `KrakenFuturesAdapter` or `KrakenFuturesClient`. The
  only references come from the dedicated test files and the
  auth-smoke tool.
- Systemd: **none.** No service or timer references the adapter.
- Order placement: not exercised. Adapter exposes payload-validation
  and symbol-normalisation logic, gated behind `dry_run`.

### C. Authenticated smoke scaffold (C1C) — FAIL-CLOSED, no creds

- Module: `chad/tools/kraken_futures_auth_smoke.py`
- Behaviour:
  - Detects whether `KRAKEN_FUTURES_API_KEY` /
    `KRAKEN_FUTURES_API_SECRET` are present in env (or fallback env
    file).
  - Dry-run path: constructs `KrakenFuturesClient(credentials=None,
    dry_run=True)` and exits ok.
  - Missing-credentials path: fails closed — does not attempt any
    private endpoint call.
  - Refuses speculative private endpoint calls when no certified
    read-only endpoint exists.
- Status: scaffold present; safe to invoke; cannot promote anything
  to live by itself.

### D. Live-trading unlock conditions (all must hold)

**Kraken Futures live trading is blocked for the current Canadian
deployment.** See
`docs/PHASE_C_KRAKEN_FUTURES_CANADA_BLOCKED_2026-05-16.md` for the
jurisdictional decision. Kraken spot remains allowed; C1A public intel
remains allowed as read-only market data; C1B adapter scaffold remains
dormant; C1C auth smoke scaffold must not be used against the
operator's Canadian Kraken account.

Future unlock is only possible with a legally eligible entity, account,
and jurisdiction plus explicit operator approval. Possession of API
keys alone is not sufficient. In addition to that eligibility gate, the
following technical conditions must all hold before any consideration
of live Kraken Futures routing:

1. `KRAKEN_FUTURES_API_KEY` present in the operator environment.
2. `KRAKEN_FUTURES_API_SECRET` present in the operator environment.
3. An authenticated **read-only** Kraken Futures endpoint certified
   end-to-end against real credentials.
4. The no-order smoke test passes against that certified endpoint.
5. Order payload validation passes against real-account expectations
   (symbol map, size precision, side, type, tif, reduce-only).
6. Risk-manager approval path designed and reviewed — sizing,
   max-gross, per-strategy caps, leverage cap.
7. Execution-pipeline integration explicitly authorized by operator
   (no implicit wiring through strategy imports).
8. Kill-switch and reconciliation rules defined: position truth
   source, fill harvester, drift detector, and trade-closer behaviour
   for Kraken Futures fills.

Until the jurisdictional block is lifted and all eight technical
conditions are satisfied, the adapter and client remain scaffold-only
and isolated from live signal flow.

---

## 4. C2 Coinglass status — BLOCKED

- Public Coinglass endpoints returned HTTP 500 / deprecation responses
  during probe.
- No Coinglass API key is provisioned.
- Implementation requires a paid Coinglass API plan.
- **No Coinglass implementation is authorized.** No publisher, no
  scaffold, no test surface to be added until a paid key exists and a
  stable endpoint contract is documented.

---

## 5. C3 IBKR DOM status — BLOCKED / DEFERRED

- Live probe returned IBKR **Error 354** ("Requested market data is not
  subscribed").
- For MES and MNQ, `domBids` / `domAsks` returned length 0.
- Root cause: missing IBKR Level 2 (market depth) entitlement on the
  paper account.
- See `docs/PHASE_C_C3_IBKR_DOM_BLOCKED_2026-05-15.md` for the probe
  detail.
- **Implementation is not authorized.** Unlock condition: MES and MNQ
  return non-empty DOM bids and asks with no Error 354 against the
  trading account in use.

---

## 6. Phase B runtime publisher status (current healthy set)

For context — these Phase B publishers are installed and healthy
alongside Phase C C1A:

- `news_intel` — Phase B headline / sentiment feed
- `relative_strength` — sector + symbol RS publisher
- `volume_scan` — intraday RVOL scanner
  (`chad-volume-scan.timer` confirmed active)
- `crypto_derivatives` — crypto derivatives intel
  (`chad-crypto-derivatives-refresh.timer` confirmed active)
- `futures_roll_state` — futures roll calendar gate
  (`chad-futures-roll-refresh.timer` confirmed active)
- `options_greeks` — options Greeks metadata
  (`chad-options-greeks-refresh.timer` confirmed active, TTL 90000 s)
- `kraken_futures_intel` — Phase C C1A
  (`chad-kraken-futures-intel-refresh.timer` confirmed active)

All seven artifacts are present in `runtime/`.

---

## 7. Notable benign issues (no action required)

- `earnings_state.json` and `sector_rotation.json` exist as stale
  bootstrap leftovers with no active consumer; safe to ignore until
  a real earnings/analyst publisher lands.
- `options_greeks` TTL is now aligned to 90000 s to match the daily
  refresh cadence (commit `759fd32`). Intentional, not a regression.
- `kraken_futures_intel` is intelligence-only — there is currently
  no strategy consumer of this feed. Intentional during scaffold
  phase.

---

## 8. Next authorized action

Kraken Futures live trading is blocked for the current Canadian
deployment (see
`docs/PHASE_C_KRAKEN_FUTURES_CANADA_BLOCKED_2026-05-16.md`). The
authenticated smoke test must not be used against the operator's
Canadian Kraken account, and option **A** as previously framed
(provisioning Kraken Futures credentials and running the auth smoke)
is therefore not authorized under this deployment. Public Kraken
Futures intel (C1A) continues to run as read-only market data.

Next safe action:

- **B.** Build the FMP earnings/analyst publisher from the already
  scaffolded FMP stable-endpoint client (commit `29e2699`) — strictly
  read-only, intelligence-only, no execution surface.

Recommended choice: **B**, because the Canadian jurisdiction block
removes the Kraken Futures private-endpoint path from the menu for
this deployment, and B is read-only intelligence work that does not
touch live execution.

---

## 9. Forbidden actions until unlock

The following are not permitted in this checkpoint window:

- no Kraken Futures live orders (jurisdiction-blocked for the current
  Canadian deployment — see
  `docs/PHASE_C_KRAKEN_FUTURES_CANADA_BLOCKED_2026-05-16.md`).
- no strategy routing to `KrakenFuturesAdapter` /
  `KrakenFuturesClient`.
- no use of `chad/tools/kraken_futures_auth_smoke.py` against the
  operator's Canadian Kraken account.
- no IBKR DOM implementation (blocked on Level 2 entitlement).
- no Coinglass implementation (blocked on paid API key and stable
  endpoint contract).
- no private endpoint calls without credentials AND a certified
  read-only smoke test AND a legally eligible Kraken Futures entity.

---

## 10. Verification evidence

- `git status --short` at start: empty (clean working tree).
- `git log --oneline -12` head matches the five Phase C commits
  enumerated in §2 above, on top of SSOT v9.2 anchor `5392fc7`.
- `python3 -m pytest chad/tests/ -q` (with
  `CHAD_SKIP_IB_CONNECT=1`, `PYTHONPATH=/home/ubuntu/chad_finale`):
  **1934 passed in 71.76s**. Matches the documented baseline; no
  regressions.
- `python3 -m chad.core.full_cycle_preview`: completed; latest
  runtime artifact present with `orders_count=0`,
  `total_notional=0`, `futures_orders=0`, intents_count=0; alpha
  futures smoke `executed=False`, `mode=disabled`, `ok=True`. Final
  line confirms: "No broker calls were made. This is a logical
  preview only."
- Timer audit: `chad-kraken-futures-intel-refresh.timer`,
  `chad-crypto-derivatives-refresh.timer`,
  `chad-futures-roll-refresh.timer`, and
  `chad-options-greeks-refresh.timer` are all installed and
  scheduled.
- Runtime audit: `runtime/kraken_futures_intel.json`,
  `runtime/crypto_derivatives.json`, `runtime/futures_roll_state.json`,
  and `runtime/options_greeks.json` are all present.
- Importer audit: no module under `chad/core`, `chad/strategies`,
  or `chad/execution` (other than the adapter module itself) imports
  `KrakenFuturesAdapter` or `KrakenFuturesClient`. Adapter remains
  fully isolated from live signal flow.

End of status lock.
