# PA — CRYPTO-TRUST: Kraken Trusted Paper-Fill Engine + Margin Chokepoint + Min-Size Handling

- **Date:** 2026-07-12
- **Status:** BUILT (repo-only, committed) — **NOT DEPLOYED.** Activates at the next `chad-live-loop` restart (operator action).
- **Posture:** paper. No live Kraken API order calls; the real Kraken `validate=true` endpoint remains only as the pre-check it already is.
- **Kill-switch:** `CHAD_KRAKEN_TRUSTED_FILLS` — **default ON** (mirrors the L1-CLD `CHAD_EXECUTION_OWN_CONNECTION` idiom: truthy unless `∈ {0,false,no,off}`). Set `=0` to fall back to the legacy untrusted `validate_only` evidence path.

## Mission framing
The Kraken paper wallet holds ~$185 (`runtime/kraken_balances.json`: `usd_equivalent≈184.58`, `cad_equivalent≈252.85`). Small size is a FEATURE: percentage returns are size-independent and fee drag is visible, not amortized. This build makes what happens WHEN alpha_crypto trades honest (real touch, real fee, realized PnL, honest labels). It does **not** change WHETHER it trades — alpha_crypto stays regime-gated (no `regime_activation_matrix.json` edits).

---

## U1 — TRUSTED PAPER-FILL ENGINE

### Problem it fixes
The paper-Kraken lane (`chad/core/kraken_execution.py:_write_paper_kraken_evidence`) marks fills at `intent.price` (`_resolve_paper_evidence_price:166-189`) — NOT the live tape — and stamps every row `validate_only=True` + `pnl_untrusted=True` + `pnl=0.0`. The Stage-2 adapter (`chad/validation/trade_log_adapter.py:trust_exclusion:261-295`) therefore REJECTS every Kraken row (validate_only / pnl_untrusted exclusions). Crypto contributes zero admissible evidence.

### Design
New module `chad/core/kraken_trusted_fill_engine.py` (pure core, injectable tick source; no import of live broker/WS objects):

1. **Live-tick marking.** Reads the existing Kraken WS feed's on-disk snapshot `runtime/kraken_prices.json` → `ticks[<symbol>].{bid,ask,last,ts_utc}` (writer: `chad/market_data/kraken_ws_client.py:301`; cross-process, atomic, 10 s cadence, `ttl_seconds=30`). Freshness is enforced: a stale/missing touch (age > ttl) fails **closed** — the engine declines to mint a trusted fill and the lane falls back to the untrusted evidence row (never a fabricated trusted PnL). The tick source is injected in tests (`tmp_path`), never the live file.
2. **Slippage model** (documented; a CHAD modeling assumption, not a Kraken-published value): marketable orders cross the spread (taker) and pay a bps impact floor.
   - `mid = (bid + ask) / 2`
   - `half_spread_frac = (ask - bid) / 2 / mid`  (≥ 0)
   - `impact_frac = slippage_impact_floor_bps / 1e4`
   - **buy** `fill_price = mid * (1 + half_spread_frac + impact_frac)`  (= ask + mid·impact_frac)
   - **sell** `fill_price = mid * (1 - half_spread_frac - impact_frac)`
   - `slippage_bps = (half_spread_frac + impact_frac) * 1e4` (cost vs mid, recorded on the fill)
   - Default `slippage_impact_floor_bps = 5.0` (config `config/kraken_trading.json`). Rationale: at $185 size real market impact is ~0, but a floor keeps the drag honest and visible rather than zero.
3. **Real Kraken fee.** `fee = fill_notional * taker_bps / 1e4`, taker rate from `config/kraken_trading.json`. **Source:** Kraken Pro published **spot taker 0.26 %** (26 bps) for the base 30-day-volume tier (≤ $10k) — consistent with the existing repo constant `paper_exec_evidence_writer.py:_CRYPTO_FEE_TAKER_PCT=0.0026`. **Marked `operator_verify: true`** in the config because the live per-account `TradeVolume` tier cannot be queried offline. NOT invented — it is the documented public base-tier rate and matches the code already in the tree.
4. **Round-trip lifecycle (FIFO).** A per-`(strategy, canonical_symbol)` lot book: an opening fill (long = buy-open, short = sell-open) pushes a lot; an opposing fill FIFO-matches against open lots and produces a **realized round-trip** with real net PnL:
   - long: `gross = (exit_price - entry_price) * matched_qty`; short: `gross = (entry_price - exit_price) * matched_qty`
   - `realized_pnl = gross - entry_fee_alloc - exit_fee_alloc` (both legs' taker fees are subtracted — this is where the ~$185 fee drag shows up)
   - The book persists open lots in the **existing** consolidated store (see Store resolution) so lifecycle survives restarts; state I/O is injected to `tmp_path` in tests.
5. **Evidence flow (same pipeline as IBKR).**
   - Every fill (open + close) → `write_paper_exec_evidence` (FILLS_/FEES_/EXECUTION_METRICS_), trusted labels.
   - Every **close** → `log_trade_result` → `data/trades/trade_history_*.ndjson` (the store Stage-2 reads) with the realized `pnl`, `broker="kraken_paper"`, `asset_class="crypto"`, positive qty/price/notional, and **no** validate_only / pnl_untrusted / placeholder flags → admitted.

### Provenance & trust contract (end-to-end)
New collision-free constants (defined beside `_FEE_MODEL_TAG`, `paper_exec_evidence_writer.py:1643`), carried in `ev.extra` and mirrored onto the TradeResult:
- `fee_model = "kraken_paper_v1"`
- `provenance = "SIMULATED_AGAINST_LIVE_TICKS"`

The crypto fee-modeling branch in `_model_ibkr_commission` (`paper_exec_evidence_writer.py:1702-1703`) currently stamps `ibkr_fixed_v1` on crypto; trusted Kraken fills carry their fee + `fee_model=kraken_paper_v1` pre-set so the writer's `_apply_modeled_commission` idempotency marker (`ev.extra["_fee_modeled"]`) suppresses the IBKR re-stamp — no mislabeling.

**Honesty guard.** Trusted rows carry `provenance=SIMULATED_AGAINST_LIVE_TICKS` all the way to Stage-2, and the Stage-2 manifest reports **simulated-crypto counts separately** (`admitted_by_provenance` / `admitted_by_instrument_class` — new tally in `run_adapter`, `trade_log_adapter.py:662-664`). No verdict silently mixes simulated-crypto with broker-confirmed evidence.

### Store resolution (GAP-021)
**Honesty note:** the "GAP-021" in `docs/` refers to the ib_async Phase-1 migration (Box-038), and **no `exec_state.v1` schema string exists anywhere in the tree** — that premise is stale. The real Kraken exec-state fragmentation is two SQLite stores:
- **#4 (LIVE):** `runtime/exec_state_paper.sqlite3` — `IdempotencyStore` table `kraken_paper_evidence`, the store the paper lane already uses for dedup.
- **#5 (DEAD):** `data/exec_state/exec_state.sqlite3` — `ExecStateStore` (`kraken_executor.py:133-238`), the "rich" per-order state machine. **Unwritten for the paper lane since 2026-02-17**; `paper_kraken` never dispatches through `ExecStateStore.claim`.

**Resolution — formally RETIRE the dead exec_state store (#5) for the paper lane** and make `runtime/exec_state_paper.sqlite3` (#4) the single Kraken-paper exec-state/idempotency store; the trusted round-trip lot book adds a table to **that same store** (not a third store). Reviving #5 would mean re-plumbing the whole validate-only dispatcher onto `ExecStateStore.claim` with zero current readers — larger and riskier, no benefit. #5's live txid-writeback path is left intact for a future *live* lane. Trade evidence remains the standard two-surface CHAD pipeline (`trade_history` for Stage-2 + hash-chained `FILLS_*`), identical to IBKR — that duplication is by-design, not Kraken fragmentation.

---

## U2 — MARGIN GATE CRYPTO CHOKEPOINT (CRYPTO-2)

- **Narrowest chokepoint:** `KrakenExecutor.execute_with_risk` (`chad/execution/kraken_executor.py:298`) — the risk-gated `_submit_intent` analog and the sole caller of `KrakenTradeRouter.execute → KrakenClient.add_order` (the only real-submit funnel). Both entry sites (live-loop `kraken_execution.py:426`, roundtrip runner) converge here.
- **Wiring (mirrors G3C):** add `margin_gate: Optional[MarginShadowGate]=None` to `KrakenExecutor.__init__`; at the TOP of `execute_with_risk`, **before** the risk check, the validate-only `router.execute` (:328) and the live `store.claim` (:333), call `gate.evaluate(order_view, now_epoch)` then `gate.should_block(verdict)`. SHADOW ⇒ `should_block` always False ⇒ proceed; any exception ⇒ fail-OPEN (proceed) with a `MARGIN_GATE_ERROR` marker. Blocks nothing; emits `MARGIN_SHADOW` markers + evidence to `data/margin_shadow/margin_shadow_YYYYMMDD.ndjson`. Default `None` ⇒ byte-identical legacy behavior.
- **New glue** (`chad/execution/kraken_margin_gate.py`): `kraken_order_view_from_intent(intent, order_id)` maps the Kraken `StrategyTradeIntent` (`pair/volume/price/side`, no `asset_class`) to the dict `margin_block.decide()` reads, with **`asset_class="crypto"`** set explicitly (routes to `_decide_crypto`, `margin_block.py:657-661/750-804`), `symbol` from pair, `qty` from `volume`, `notional` from `notional_estimate`, `currency="USD"` (notional is USD; `_order_notional_cad` converts to the CAD balance basis). `build_default_kraken_shadow_gate(...)` constructs `MarginShadowGate(config, snapshot_source=<kraken>, open_orders_source=None, evidence_path=…)` — `build_default_shadow_gate` is IBKR-only, so this is a parallel builder. The snapshot source reads `runtime/kraken_balances.json` → `KrakenBuyingPowerSnapshot` (`kraken_bp_provider.parse_kraken_balances`; `available_cad` is the capacity field) with a **longer TTL (600 s)** because the balance producer refreshes ~300 s (the 30 s default would spuriously STALE→fail-closed).
- **Same config, no new thresholds.** `config/margin_block.json mode=shadow` drives both lanes; `crypto_unlevered=1.00` rate row already present. No Kraken section added (the strict loader rejects unknown keys).
- **CI invariant (mirrors `test_margin_shadow_gate.py:325-358`):** (a) a spy-gate proves every `execute_with_risk` (validate + live) evaluates the gate; (b) source-inspection proves the gate call precedes `self._store.claim(` and `self._router.execute(` — no Kraken submit path bypasses the gate.

---

## U3 — LUNCH-MONEY SIZING

- **Config table** `config/kraken_trading.json` → `min_order_size_by_pair` (from the repo's existing `execution_pipeline.py:_KRAKEN_MIN_VOLUMES` — XBTUSD 0.0001, ETHUSD 0.001, SOLUSD 0.05, XBTCAD/ETHCAD — carried forward, **`operator_verify: true`** because live `AssetPairs.ordermin` cannot be queried offline). Not invented; the repo's own documented minima.
- **Decision (`chad/execution/kraken_min_size.py`, pure):** `decide_min_size(pair, computed_volume, price, min_volume, available_notional, risk_cap_notional) → {action, final_volume, marker, reason}`:
  - `computed_volume ≥ min_volume` → **PASS** (unchanged, no marker).
  - below min, `min_notional = min_volume*price` affordable (`≤ available_notional` AND `≤ risk_cap_notional`) → **BUMP** to `min_volume`, marker `CRYPTO_MIN_SIZE_BUMP`.
  - below min, not affordable → **SKIP** (loud `logger.warning` with the marker + pair/computed/min/notional/cap), marker `CRYPTO_BELOW_MIN_SKIP`. Never a silent starve.
- **Insertion:** `execution_pipeline.py:1546-1548` (replaces the silent `return None`). All of `pair/volume_dec/price_dec/attenuated_notional` + per-pair min are in scope. Markers ride via a new `markers: Tuple[str,...]=()` field on `StrategyTradeIntent` (backward-compatible) → read into the evidence `tags`/`extra` at `kraken_execution.py:335` and the trusted engine.
- **SCR sizing (honesty fix):** SCR `sizing_factor` is currently applied to the **IBKR lane only** (`live_loop.py:2244-2247`), never crypto. This build adds a crypto-lane read (mirroring that idiom) so SCR 0.1 actually attenuates Kraken size — otherwise the "$185 × SCR 0.1" matrix would be a no-op. `PAUSED`/zero factor blocks; otherwise `computed_volume *= factor` before the min check.
- **Test matrix:** $185 account × SCR 0.1 across BTC/ETH/SOL — each of BUMP / PASS(afford) / SKIP proven.

---

## New markers
`CRYPTO_MIN_SIZE_BUMP`, `CRYPTO_BELOW_MIN_SKIP` (sizing), plus the trust labels `fee_model=kraken_paper_v1`, `provenance=SIMULATED_AGAINST_LIVE_TICKS`. Existing `MARGIN_SHADOW` / `MARGIN_GATE_ERROR` reused for U2.

## Activation plan
1. Commit repo-only (done). No deploy.
2. Next operator `chad-live-loop` restart loads the trusted path (kill-switch default ON) + the Kraken margin shadow gate.
3. Watch `data/trades/trade_history_*.ndjson` for `broker=kraken_paper` rows with real `pnl` + `provenance=SIMULATED_AGAINST_LIVE_TICKS`; `data/margin_shadow/*.ndjson` for `MARGIN_SHADOW` crypto verdicts; logs for `CRYPTO_MIN_SIZE_BUMP` / `CRYPTO_BELOW_MIN_SKIP`.
4. Stage-2 (`trade_log_adapter --stage stage2`) manifest should show `admitted_by_provenance.SIMULATED_AGAINST_LIVE_TICKS > 0`, separated from broker-confirmed.

## Rollback
- `CHAD_KRAKEN_TRUSTED_FILLS=0` (drop-in env) → legacy untrusted `_write_paper_kraken_evidence` path; margin gate stays shadow (never blocks) and can be dropped by passing `margin_gate=None`.
- `git revert` of the build commits. No runtime/config state was mutated by the build.
