Here is the unified final list. Everything in one place.

---

# CHAD Elite Completion Checklist
## "Offense as good as defense"

---

## PART 1 — OFFENSE: Elite Trading Capabilities

### Signal Quality (Phase A)
| # | Item | Status |
|---|---|---|
| A1 | Stop-distance sizing — entry only when stop is valid and sized | ✅ Built |
| A2 | Time-of-day session zones — no trades pre-open or post-close | ✅ Built |
| A3 | R:R gate — minimum 1.5:1 risk-reward required before entry | ✅ Built |
| A4 | Trade setup tagging — every signal carries setup_family label | ✅ Built |
| A5 | Float-aware liquidity gate — thin stocks blocked before entry | ✅ Built |

### Intelligence Feeds (Phase B)
| # | Item | Status |
|---|---|---|
| B1 | Catalyst/news publisher — confirmed_gate_relevant scoring, 30-min cadence | ✅ Live |
| B2 | Relative strength publisher — RS vs SPY, strong/neutral/weak, daily | ✅ Live |
| B3 | RVOL scanner — real-time volume vs 20-day average, 5-min cadence | ✅ Live |
| B4 | Futures roll calendar — roll proximity warnings, daily cadence | ✅ Live |
| B5 | Options Greeks publisher — synthetic B-S delta/theo, daily cadence | ✅ Live |
| B6 | FMP earnings intel — next earnings date, analyst targets, 6-hr cadence | ✅ Live |

### Exchange Connections (Phase C)
| # | Item | Status |
|---|---|---|
| C1A | Kraken Futures public intel — 306 perps, OI/funding/crowding, 5-min | ✅ Live |
| C1B | Kraken Futures adapter scaffold — dry-run only | ✅ Built |
| C1C | Kraken Futures live trading (alpha_crypto_perps) | 🔒 Blocked — Canadian jurisdiction |
| C2 | Coinglass liquidation heatmap | ⬜ Decision needed — paid key OR build Binance replacement (free) |
| C2B | Binance perpetuals publisher — free replacement for Coinglass OI/funding | ⬜ Not yet built |
| C3 | IBKR DOM / Level 2 consumer — CME Real-Time subscribed | ⏸ Pending live DOM test |

### Architecture (Phase D)
| # | Item | Status |
|---|---|---|
| D1 | Dynamic universe scanner — 0.35×RS + 0.30×RVOL + 0.25×catalyst + 0.10×liquidity | ✅ Live (observation mode) |
| D1B | Dynamic scanner → active universe promotion (v2) | ⬜ After 5-day observation window |
| D2T1 | Typed OptionsSpreadSpec — validated BAG contract, replaces stringly-typed dict | ✅ Built |
| D2T2 | BAG LMT discipline — MKT coerced to LMT, no-price BAG hard-blocked | ✅ Built |
| D2T3 | Offline quote check — synthetic mid vs estimated debit, tolerance gate | ✅ Built |
| D2T3B | Live quote probe — read-only IBKR snapshot, real SPY prices confirmed | ✅ Built |
| D2T3C | BAG limit-price unit normalisation — per-share contract preserved at adapter (no ÷100 conversion needed); ÷100 applied only at paper-fill `notional` boundary | ✅ Built (BOX-051 OFFICIAL: contract documented in `runtime/completion_matrix_evidence/BOX-051_OFFICIAL_BAG_Tier_3C_limit_price_unit_normalization.md`; pinned by `chad/tests/test_box051_official_bag_lmt_unit_normalization.py` + 6 prior BAG test files, 76 tests total) |
| D2T4 | BAG bracket/OCA or failsafe exit — unmanaged position protection | ⬜ Pre-live requirement |
| D2T5 | spread_id-aware position guard — prevents concurrent BAG collision | ⬜ Pre-live requirement |
| D2T6 | Live BAG fill harness — broker-confirmed round-trip test | ⬜ Pre-live requirement |

### ML Intelligence
| # | Item | Status |
|---|---|---|
| ML1 | XGB veto gate — 71.1% accuracy, shadow-only mode, blocks bad signals | ✅ Live |
| ML2 | XGB model promotion workflow — metric-gated, no dirty git on Sundays | ✅ Built |
| ML3 | XGB model health review — 99.1% loss prob on MES suspicious (GAP-019) | ⬜ Manual review needed |

---

## PART 2 — DEFENSE: System Integrity

### P0 — Block live promotion
| # | Gap | Fix |
|---|---|---|
| P0-1 | Delta strategy emitting $100 placeholder fills for SPY (GAP-001) | Trace and fix root cause in delta.py |
| P0-2 | Prometheus shows -$90K loss, SCR shows +$9.9K profit — same gauge (GAP-002) | Split into chad_paper_raw_* and chad_paper_lean_* |

### P1 — High priority system truth
| # | Gap | Fix |
|---|---|---|
| P1-1 | Strategy registry 3-way mismatch: enum=18, tier_manager=17, weights=16 (GAP-003/008/026) | Single registry.py, validated at startup |
| P1-2 | 3 production files still import ib_insync (GAP-004) | Migrate paper_shadow_runner, paper_position_closer, ibkr_broker_events_collector |
| P1-3 | drawdown_state.json 10 days stale, TTL=300s (GAP-005) | Find and fix the orphaned publisher |
| P1-4 | ibkr_watchdog_last.json 44 days stale (GAP-006) | Find and fix the orphaned publisher |
| P1-5 | Account equity: 3 sources, 3 values — $172K vs $183K vs $183K (GAP-007) | Designate portfolio_snapshot.json as canonical |
| P1-6 | position_guard_drift.json not found despite reconciliation timer firing (GAP-015/017) | Trace publisher, fix write path |
| P1-7 | Stop-bus / live-loop process predates fix commit bbe7525 (GAP-039) | Restart live-loop into fixed build |
| P1-8 | Stale PendingSubmit rows in ibkr_adapter_state.sqlite3 (GAP-036) | Clean stale rows, verify lifecycle transitions |

### P2 — Important hardening
| # | Gap | Fix |
|---|---|---|
| P2-1 | same_side_position_open still blocking valid entries — 156 hits/day (GAP-009) | Weekday regression audit, fix guard logic |
| P2-2 | operator_intent=ALLOW_LIVE while system not live-ready (GAP-016) | Default to DENY_ALL in paper, require explicit flip |
| P2-3 | HALT_BOOST_SUPPRESSED every cycle for delta and gamma_futures (GAP-018) | Reconcile winner_multipliers with allocator clamp |
| P2-4 | XGB veto 99.1% loss prob on MES — model drift or feature error (GAP-019) | Run model_doctor.py, review feature distributions |
| P2-5 | Ports 9618/9619/9620 bound to 0.0.0.0 (GAP-013) | Verify EC2 security group, restrict if exposed |
| P2-6 | legend_top_stocks.json 47 days stale (GAP-014) | Fix daily legend pipeline writer |
| P2-7 | Telegram bot urllib3 RemoteDisconnected loop (GAP-010/023) | Pin urllib3<2 or upgrade telegram-bot to 20.x |
| P2-8 | APScheduler pkg_resources deprecation warning (GAP-022) | Upgrade to APScheduler 4.x |
| P2-9 | options-chain-refresh.service silently failing (NEW-GAP-044) | Fix service, add loud alert on failure |
| P2-10 | live-readiness reads upstream GREEN instead of resolved reconciliation (NEW-GAP-041) | Fix readiness gate logic |
| P2-11 | Dual ledger ambiguity — ibkr_paper_ledger vs ibkr_paper_ledger_state (NEW-GAP-052) | Declare authority, document which is canonical |

### P3 — Hygiene
| # | Gap | Fix |
|---|---|---|
| P3-1 | 1573 Telegram dedupe JSON files in runtime/ (GAP-012) | Move to runtime/dedupe/, add cleanup policy |
| P3-2 | legacy dynamic_caps backup files in runtime/ (GAP-011) | Archive to runtime/archive/ |
| P3-3 | Logs not log-rotated (GAP-027) | Add /etc/logrotate.d/chad, 14-day retention |
| P3-4 | Stale memory entries claiming placeholder fills fixed and snapshot stale (GAP-024/025) | Update/delete stale memory entries |
| P3-5 | Zero-fill strategies not classified — dormant vs broken vs regime-silent | Audit and classify each |

---

## PART 3 — LIVE PROMOTION PATH

### Prerequisites (must be done before flipping live)
| # | Item |
|---|---|
| L1 | All P0 gaps closed with evidence |
| L2 | All P1 gaps closed with evidence |
| L3 | GAP-039 deployed — live-loop running on fixed build |
| L4 | Weekday re-audit — equity strategies routing bars to all 16 strategies confirmed |
| L5 | Binance perpetuals publisher built (C2B) — or Coinglass paid plan activated |
| L6 | C3 IBKR DOM live test passed — MES/MNQ domBids+domAsks > 0, no Error 354 |
| L7 | BAG D2T3C unit normalisation built and tested |
| L8 | BAG D2T4 bracket/failsafe exit built |
| L9 | BAG D2T5 spread_id-aware guard built |
| L10 | BAG D2T6 live fill harness built and tested |
| L11 | Dynamic universe scanner 5-day observation period complete — v2 promotion decision |
| L12 | XGB model health reviewed — shadow mode confirmed intentional or model replaced |
| L13 | Sustained clean paper soak — defined duration, no P0/P1, stable SCR, clean fills |
| L14 | Final P0/P1 sign-off — all boxes checked with runtime/log/test evidence |
| L15 | Operator explicit GO — canary plan approved, first live trade size decided |
| L16 | Flip ready_for_live=true |

---

## Summary

| Section | Total items | Done | Remaining |
|---|---|---|---|
| Offense — Phase A signals | 5 | 5 | 0 |
| Offense — Phase B feeds | 6 | 6 | 0 |
| Offense — Phase C exchanges | 6 | 2 | 4 |
| Offense — Phase D architecture | 11 | 6 | 5 |
| Offense — ML intelligence | 3 | 2 | 1 |
| Defense — P0 gaps | 2 | 0 | 2 |
| Defense — P1 gaps | 8 | 0 | 8 |
| Defense — P2 gaps | 11 | 0 | 11 |
| Defense — P3 hygiene | 5 | 0 | 5 |
| Live promotion path | 16 | 0 | 16 |
| **Total** | **73** | **21** | **52** |

21 items complete. 52 remaining. The offense is strong and largely built. The defense has known gaps that need closing. Live promotion is gated behind both.

**The most valuable next three moves:**
1. Fix P0-1 (delta $100 fills) — the only fill-quality regression
2. Fix P0-2 (metrics split) — so operator dashboards tell the truth
3. Restart live-loop into fixed build (L3/GAP-039) — so CHAD is running on the correct code
