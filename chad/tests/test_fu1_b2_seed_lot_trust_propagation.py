"""FU1/B2 — seed-lot untrust must propagate end-to-end.

ULTRA_CLOSE_AUDIT_2026-07-17 B-2: the Epoch-3 reconciler seeds adopted broker
positions with a FABRICATED cost basis and marks the lot
``meta={"pnl_untrusted": True, "scoring_excluded": True, ...}``. That marker
reached the closed_trade.v1 payload under ``payload["meta"]`` only — while SCR
(``trade_stats_engine``) reads ``payload["extra"]``/``tags`` and Stage-2
(``trade_log_adapter.trust_exclusion``) read the same two. Both missed. The
``fill_ids`` backstop missed too: ``RECON_ADOPT_*`` ids appear in zero fills
files, so the 4,782-id exclusion set matched none of them.

Net effect the flip would have had: closing a seed lot banks invented alpha into
SCR *and* Stage-2 as clean evidence — the one harm not undone by flipping back.

These tests pin all three ends of the fix.
"""

from __future__ import annotations

import json
from pathlib import Path

from chad.execution.trade_closer import ClosedTrade
from chad.analytics.trade_stats_engine import _is_untrusted
from chad.validation.trade_log_adapter import trust_exclusion
from chad.utils.quarantine import (
    get_exclusion_sets,
    get_untrusted_fill_ids_from_fifo_lots,
)

# The exact meta block scripts/reconcile_ledger_to_broker.py stamps on a seed lot.
SEED_LOT_META = {
    "reconciled": True,
    "provenance": "UNATTRIBUTED_EPOCH3_ACCUMULATION",
    "pnl_untrusted": True,
    "scoring_excluded": True,
    "source": "reconcile_ledger_to_broker",
    "seeded_from": "broker_truth",
}


def _seed_closed_trade(**overrides) -> ClosedTrade:
    """The live UNH seed lot from the audit: 273 @ 420.71 fabricated basis."""
    kwargs = dict(
        strategy="gamma",
        symbol="UNH",
        side="BUY",
        entry_price=420.71,
        exit_price=378.64,
        quantity=273.0,
        entry_time_utc="2026-07-15T16:25:17+00:00",
        exit_time_utc="2026-07-17T14:38:34+00:00",
        pnl=-11485.11,
        contract_multiplier=1.0,
        fill_ids=["RECON_ADOPT_UNH_20260621T000000Z", "fill-close-1"],
        meta=dict(SEED_LOT_META),
    )
    kwargs.update(overrides)
    return ClosedTrade(**kwargs)


def _clean_closed_trade() -> ClosedTrade:
    return _seed_closed_trade(
        symbol="SPY",
        entry_price=500.0,
        exit_price=505.0,
        quantity=15.0,
        pnl=75.0,
        fill_ids=["fill-open-clean", "fill-close-clean"],
        meta={"setup_family": "ORB", "stop_width_usd": 1.25},
    )


# --------------------------------------------------------------------------- #
# (a) trade_closer propagates meta -> extra / tags
# --------------------------------------------------------------------------- #
class TestPayloadPropagation:
    def test_seed_lot_payload_carries_extra_markers(self):
        payload = _seed_closed_trade().to_payload()
        assert payload["extra"]["pnl_untrusted"] is True
        assert payload["extra"]["scoring_excluded"] is True

    def test_seed_lot_payload_carries_tag_markers(self):
        tags = _seed_closed_trade().to_payload()["tags"]
        assert "pnl_untrusted" in tags
        assert "scoring_excluded" in tags
        # The pre-existing tags must survive — attribution still reads them.
        assert {"paper", "closed", "gamma"} <= set(tags)

    def test_seed_lot_extra_carries_provenance_for_the_auditor(self):
        extra = _seed_closed_trade().to_payload()["extra"]
        assert extra["provenance"] == "UNATTRIBUTED_EPOCH3_ACCUMULATION"
        assert "seed_lot_fabricated_cost_basis" in extra["pnl_untrusted_reason"]

    def test_meta_block_is_still_emitted_unchanged(self):
        # B2 mirrors meta; it must not consume or relocate it (Gap-4 readers
        # like setup_family_expectancy_updater still read payload["meta"]).
        payload = _seed_closed_trade().to_payload()
        assert payload["meta"]["provenance"] == "UNATTRIBUTED_EPOCH3_ACCUMULATION"

    def test_clean_trade_payload_shape_is_unchanged(self):
        # A clean trade must not grow an `extra` key — B2 is surgical, not a
        # schema change for the 99% case.
        payload = _clean_closed_trade().to_payload()
        assert "extra" not in payload
        assert payload["tags"] == ["paper", "closed", "gamma"]

    def test_clean_trade_meta_survives_without_trust_markers(self):
        payload = _clean_closed_trade().to_payload()
        assert payload["meta"]["setup_family"] == "ORB"


# --------------------------------------------------------------------------- #
# (b) SCR + Stage-2 treat meta OR extra as exclusion truth
# --------------------------------------------------------------------------- #
class TestScrExclusion:
    def test_meta_only_marker_excludes(self):
        # The pre-B2 on-disk shape: marker on meta, nothing on tags/extra.
        assert _is_untrusted([], {}, SEED_LOT_META) is True

    def test_extra_only_marker_excludes(self):
        assert _is_untrusted([], {"scoring_excluded": True}, None) is True

    def test_tag_only_marker_excludes(self):
        assert _is_untrusted(["scoring_excluded"], {}, None) is True

    def test_pnl_untrusted_still_excludes_via_every_carrier(self):
        assert _is_untrusted(["pnl_untrusted"], {}, None) is True
        assert _is_untrusted([], {"pnl_untrusted": True}, None) is True
        assert _is_untrusted([], {}, {"pnl_untrusted": True}) is True

    def test_clean_row_is_not_excluded(self):
        assert _is_untrusted(["paper", "closed", "gamma"], {}, {"setup_family": "ORB"}) is False

    def test_unreadable_trust_block_fails_closed(self):
        class Hostile:
            def get(self, _key, _default=None):
                raise RuntimeError("unreadable")

            def __bool__(self):
                return True

        assert _is_untrusted([], Hostile(), None) is True


class TestStage2Exclusion:
    def test_seed_lot_round_trip_is_refused_end_to_end(self):
        """The spec's acceptance test: closing a seed lot yields a non-None reason."""
        payload = _seed_closed_trade().to_payload()
        assert trust_exclusion({"payload": payload}) is not None

    def test_meta_only_row_is_refused(self):
        # Rows written BEFORE the trade_closer mirror carry meta alone.
        payload = {"broker": "paper_exec", "meta": dict(SEED_LOT_META)}
        assert trust_exclusion({"payload": payload}) == "pnl_untrusted"

    def test_scoring_excluded_alone_is_refused(self):
        payload = {"broker": "paper_exec", "meta": {"scoring_excluded": True}}
        assert trust_exclusion({"payload": payload}) == "scoring_excluded"

    def test_scoring_excluded_via_extra_is_refused(self):
        payload = {"broker": "paper_exec", "extra": {"scoring_excluded": True}}
        assert trust_exclusion({"payload": payload}) == "scoring_excluded"

    def test_clean_round_trip_is_still_admitted(self):
        # The gate must stay permissive for real evidence — a fix that excluded
        # everything would silently end Stage-2 rather than protect it.
        payload = _clean_closed_trade().to_payload()
        assert trust_exclusion({"payload": payload}) is None


# --------------------------------------------------------------------------- #
# (c) RECON_ADOPT_* ids register in the fill_ids backstop
# --------------------------------------------------------------------------- #
def _write_state(runtime: Path, lots: list) -> None:
    (runtime / "trade_closer_state.json").write_text(
        json.dumps({"queues": [{"strategy": "gamma", "symbol": "UNH", "lots": lots}]}),
        encoding="utf-8",
    )


class TestFifoLotBackstop:
    def test_marked_seed_lot_id_is_pinned(self, tmp_path):
        _write_state(tmp_path, [
            {"fill_id": "RECON_ADOPT_UNH_20260621T000000Z", "meta": dict(SEED_LOT_META)},
        ])
        assert get_untrusted_fill_ids_from_fifo_lots(runtime_dir=tmp_path) == {
            "RECON_ADOPT_UNH_20260621T000000Z"
        }

    def test_prefix_fallback_pins_unmarked_seed_lot(self, tmp_path):
        # Belt-and-braces: a seed lot minted before the markers existed.
        _write_state(tmp_path, [{"fill_id": "RECON_ADOPT_V_20260621T000000Z"}])
        assert "RECON_ADOPT_V_20260621T000000Z" in get_untrusted_fill_ids_from_fifo_lots(
            runtime_dir=tmp_path
        )

    def test_clean_lot_is_not_pinned(self, tmp_path):
        _write_state(tmp_path, [
            {"fill_id": "fill-real-1", "meta": {"setup_family": "ORB"}},
        ])
        assert get_untrusted_fill_ids_from_fifo_lots(runtime_dir=tmp_path) == set()

    def test_missing_state_file_is_fail_safe(self, tmp_path):
        assert get_untrusted_fill_ids_from_fifo_lots(runtime_dir=tmp_path) == set()

    def test_corrupt_state_file_is_fail_safe(self, tmp_path):
        (tmp_path / "trade_closer_state.json").write_text("{not json", encoding="utf-8")
        assert get_untrusted_fill_ids_from_fifo_lots(runtime_dir=tmp_path) == set()

    def test_seed_ids_reach_the_union_exclusion_set(self, tmp_path):
        # The audit's measured failure: get_exclusion_sets() returned 4,782 ids
        # and ZERO matched a RECON_ADOPT_* lot. This is that gap, closed.
        runtime = tmp_path / "runtime"
        fills = tmp_path / "fills"
        runtime.mkdir()
        fills.mkdir()
        _write_state(runtime, [
            {"fill_id": "RECON_ADOPT_SVXY_20260621T000000Z", "meta": dict(SEED_LOT_META)},
        ])
        fill_ids, _hashes = get_exclusion_sets(runtime_dir=runtime, fills_dir=fills, trades_dir=fills)
        assert "RECON_ADOPT_SVXY_20260621T000000Z" in fill_ids
