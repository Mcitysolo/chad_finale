# CHAD FMP Earnings Intelligence Status — 2026-05-16

## 1. Title
CHAD FMP Earnings Intelligence Status — 2026-05-16

## 2. Status
**LIVE INTELLIGENCE PUBLISHER — INSTALLED**

The FMP earnings / analyst intelligence publisher has been committed
(`a98a165 Add FMP earnings and analyst intelligence publisher`) and is
running on its installed systemd timer. Output is being written to
`runtime/earnings_intel.json` with a healthy `partial` status as expected
by design.

## 3. Runtime Artifact
- Path: `runtime/earnings_intel.json`
- `schema_version`: `earnings_intel.v1`
- Cadence: every 6 hours
- TTL: `21600` seconds (6 hours)
- Last write (observed): `2026-05-16T17:01:52Z`
- Freshness: within TTL at audit time

## 4. Source
- Provider: FMP stable endpoints
- Endpoints in use:
  - `earnings-calendar`
  - `price-target-consensus`
  - `analyst-estimates` (annual)
  - `sec-filings-search`
- FMP news intentionally not used because the current FMP plan
  restricts it. Do not enable the FMP news endpoint without an explicit
  plan-level decision.

## 5. Installed Unit
- Service: `chad-fmp-earnings-intel-refresh.service`
- Timer:   `chad-fmp-earnings-intel-refresh.timer`
- Execution: `/home/ubuntu/chad_finale/venv/bin/python3 -m chad.market_data.fmp_earnings_intel_publisher`

## 6. Current Behavior
- status partial is expected and acceptable when ETFs or unsupported
  tickers (e.g., SPY, QQQ, AVGO in current symbol set) lack full FMP
  analyst/earnings coverage. A `status=partial` result is by design,
  not a publisher failure.
- Per-symbol `provider_errors` arrays are allowed and fail-open;
  individual endpoint failures degrade gracefully and do not abort the
  publisher run.
- Old `runtime/earnings_state.json` remains untouched and should be
  treated as bootstrap legacy output. The new publisher does not mutate
  it.

## 7. Consumer Status
- **`earnings_intel.json` is now surfaced read-only** in the
  strategy-intelligence / dashboard context via
  `chad.intel.strategy_intelligence._load_earnings_intel_context` and the
  `earnings_intel` block on `chad.dashboard.api.StateBuilder._intelligence()`.
- **No strategy gates are wired.** No file under `chad/strategies/`
  references the helper or `earnings_intel.json` (enforced by
  `chad/tests/test_earnings_intel_context.py`).
- **No confidence modifiers are wired.** No adjustment is applied to
  any `TradeSignal.confidence` or to any sizing path.
- **No execution routing consumes it.** No file under `chad/execution/`
  or `chad/risk/` references the helper or `earnings_intel.json`
  (enforced by the same test module).
- The old `earnings_state.json` remains untouched bootstrap legacy.
- **Observation period remains one full collection week** before any
  strategy wiring is considered. Re-audit before advancing to metadata
  enrichment (Option B) or a soft confidence modifier (Option C).

## 8. Future Use Cases
- Earnings proximity guard (e.g., block new entries within N days of
  `next_earnings_date`).
- Analyst target context (use `price_target_consensus` and
  `annual_eps_avg_estimate` for risk-aware sizing or filter rules).
- SEC filing context (recent 8-K / SD / 424B2 awareness via
  `latest_filing_date` and `latest_filing_type`).
- Post-earnings continuation scoring.
- Avoid-entry-before-earnings policy, if later approved.

## 9. Forbidden Actions
- Do **not** delete `earnings_state.json` yet.
- Do **not** wire strategy gates against `earnings_intel.json` without a
  separate audit and explicit approval.
- Do **not** enable or call the FMP news endpoint on the current plan.
- Do **not** treat `status=partial` as a publisher failure — it is the
  designed, fail-open behavior.

## 10. Verification Evidence

### Git baseline (audit start)
- `git status --short`: clean (no output)
- `git log --oneline -8` (top):
  - `a98a165 Add FMP earnings and analyst intelligence publisher`
  - `9eaad5f Docs: mark Kraken Futures blocked for Canadian deployment`
  - `2dd0f24 Docs: lock Phase C status after Kraken Futures scaffold`
  - `c9da3b9 Add Kraken Futures authenticated smoke scaffold`
  - `69d3a95 Add Phase C Kraken Futures adapter scaffold`
  - `759fd32 Align options Greeks TTL with daily refresh cadence`
  - `0204cba Add Phase C Kraken Futures public intel publisher`
  - `42daa1f Document Phase C IBKR DOM entitlement blocker`

### Timer state
- `LoadState=loaded`
- `ActiveState=active`
- `SubState=waiting`
- `LastTriggerUSec=Sat 2026-05-16 17:01:34 UTC`
- Next scheduled run (from `list-timers`):
  `Sat 2026-05-16 23:01:46 UTC` (~4h 34min away at audit time)
- Service last run: `code=exited, status=0/SUCCESS`

### Service log summary (last run)
```
[fmp_earnings_intel_publisher] wrote /home/ubuntu/chad_finale/runtime/earnings_intel.json
  status=partial
  provider_status=partial
  symbols_requested=25
  symbols_processed=25
  with_next_earnings=1 with_price_targets=10 with_analyst_estimates=10 with_sec_filings=21
```

### Runtime summary (`runtime/earnings_intel.json`)
- `schema_version`: `earnings_intel.v1`
- `status`: `partial`
- `ttl_seconds`: `21600`
- `source.provider`: `fmp_stable`
- `source.provider_status`: `partial`
- `source.endpoints`: `earnings-calendar`, `price-target-consensus`,
  `analyst-estimates annual`, `sec-filings-search`
- `window`: `date_from=2026-05-02`, `date_to=2026-06-30`,
  `forward_days=45`, `lookback_days=14`
- `summary`:
  - `symbols_requested=25`
  - `symbols_processed=25`
  - `symbols_with_next_earnings=1`
  - `symbols_with_price_targets=10`
  - `symbols_with_analyst_estimates=10`
  - `symbols_with_sec_filings=21`
- `ts_utc`: `2026-05-16T17:01:52Z` (fresh within TTL)

### Example symbol coverage
- AAPL: data_available=True, price_target_consensus=324.21,
  annual_eps_avg_estimate=8.71903, latest_filing_type=4 (2026-05-12),
  provider_errors=[]
- MSFT: data_available=True, price_target_consensus=556.88,
  annual_eps_avg_estimate=16.77989, latest_filing_type=8-K (2026-05-14),
  provider_errors=[]
- NVDA: data_available=True, next_earnings_date=2026-05-20
  (days_to_next_earnings=4), price_target_consensus=276.75,
  annual_eps_avg_estimate=8.31929, latest_filing_type=SD (2026-05-15),
  provider_errors=[]
- BAC: data_available=True, price_target_consensus=61.13,
  annual_eps_avg_estimate=4.46107, sec_filings_count=100,
  latest_filing_type=424B2 (2026-05-15), provider_errors=[]
- SPY: data_available=False, provider_errors=[`price-target-consensus`,
  `analyst-estimates`, `sec-filings`] — expected (ETF, unsupported by
  analyst/filings endpoints).
- QQQ: data_available=False, provider_errors=[`price-target-consensus`,
  `analyst-estimates`, `sec-filings`] — expected (ETF).
- AVGO: data_available=False, provider_errors=[`price-target-consensus`,
  `analyst-estimates`, `sec-filings`] — provider-side gap; fail-open
  behavior preserves the run.

### Consumer audit
- `grep -Rni "earnings_intel" chad/`: only the publisher itself.
- `grep -Rni "earnings_state" chad/`: only a docstring reference in the
  new publisher explaining it is not mutated.
- Deploy unit reference confirmed:
  `deploy/chad-fmp-earnings-intel-refresh.service` invokes
  `python3 -m chad.market_data.fmp_earnings_intel_publisher`.
- Conclusion: publisher is **intelligence-only**; no live strategy is
  reading either `earnings_intel.json` or stale `earnings_state.json`.

### Audit-end git status
- Working tree contains only the new file
  `docs/CHAD_FMP_EARNINGS_INTEL_STATUS_2026-05-16.md`.
- No commit created by this audit.
