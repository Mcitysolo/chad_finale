"""PositionTruthEngine — Decision 1 / R1 closeout.

Reads ``runtime/positions_snapshot.json`` (periodic IBKR poll) and
``runtime/ibkr_paper_ledger_state.json`` (event-driven append log),
merges them per the 5 rules defined in
``docs/design/POSITION_TRUTH_V2_ENGINE_DESIGN_2026-05-27.md`` §9, and
emits a ``position_truth_v2.v1`` document.

This is the **stub** delivery — the engine is pure-function and the
CLI is ``--check`` only. Production wiring (writing
``runtime/position_truth_v2.json`` on a timer; repointing downstream
consumers) is the migration plan in §12 of the design, separately
operator-authorized.

Safety contract:
  - The engine **never** mutates any source file.
  - The engine **never** calls the broker.
  - The engine **never** reads any other ``runtime/*.json``.
  - The engine **never** sends Telegram.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chad.schemas.position_truth_v2 import (
    AD_FAIL_CLOSED,
    AD_LEDGER,
    AD_SNAPSHOT,
    DEFAULT_TTL_SECONDS,
    ENGINE_VERSION,
    HEALTH_GREEN,
    HEALTH_RED,
    LEDGER_TTL_SECONDS,
    PositionEntry,
    PositionTruthV2,
    ProvenanceEntry,
    RULE_M1,
    RULE_M2,
    RULE_M3,
    RULE_M4,
    RULE_M5,
    SNAPSHOT_TTL_SECONDS,
    SourceArtifact,
    VS_BOTH,
    VS_DISAGREEMENT,
    VS_FAIL_CLOSED,
    VS_LEDGER,
    VS_SNAPSHOT,
    derive_side,
    health_from_rules,
    serialize_to_dict,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT_PATH = REPO_ROOT / "runtime" / "positions_snapshot.json"
DEFAULT_LEDGER_PATH = REPO_ROOT / "runtime" / "ibkr_paper_ledger_state.json"

# 2 × snapshot cadence (5-min poll → 10-min window for M2 ledger-lead tolerance).
SNAPSHOT_CADENCE_SECONDS = 300
M2_MAX_LEAD_SECONDS = 2 * SNAPSHOT_CADENCE_SECONDS


# ---------------------------------------------------------------------------
# IO helpers (READ-ONLY against sources)
# ---------------------------------------------------------------------------

def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso_utc(s: str | None) -> datetime | None:
    if not s:
        return None
    txt = s.strip()
    # Tolerate trailing Z + microseconds
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _sha256_of_file(p: Path) -> str:
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PositionTruthEngine:
    """Merge snapshot + ledger into a ``position_truth_v2.v1`` document.

    The engine is a **pure function** of its constructor arguments
    plus the contents of the source files at ``run()`` time. Running
    it twice in succession with no source change produces identical
    output (modulo ``ts_utc`` of the engine itself).
    """

    def __init__(
        self,
        snapshot_path: Path | str = DEFAULT_SNAPSHOT_PATH,
        ledger_path: Path | str = DEFAULT_LEDGER_PATH,
        snapshot_ttl: int = SNAPSHOT_TTL_SECONDS,
        ledger_ttl: int = LEDGER_TTL_SECONDS,
    ) -> None:
        self.snapshot_path = Path(snapshot_path)
        self.ledger_path = Path(ledger_path)
        self.snapshot_ttl = int(snapshot_ttl)
        self.ledger_ttl = int(ledger_ttl)

    # ---- source loaders ----------------------------------------------------

    def read_snapshot(self) -> tuple[dict[str, Any] | None, str | None]:
        """Return (parsed_doc, error_or_None). Never raises on missing file."""
        if not self.snapshot_path.is_file():
            return None, f"snapshot_missing:{self.snapshot_path}"
        try:
            return json.loads(self.snapshot_path.read_text(encoding="utf-8")), None
        except json.JSONDecodeError as exc:
            return None, f"snapshot_parse_error:{exc}"

    def read_ledger(self) -> tuple[dict[str, Any] | None, str | None]:
        """Return (parsed_doc, error_or_None). Never raises on missing file."""
        if not self.ledger_path.is_file():
            return None, f"ledger_missing:{self.ledger_path}"
        try:
            return json.loads(self.ledger_path.read_text(encoding="utf-8")), None
        except json.JSONDecodeError as exc:
            return None, f"ledger_parse_error:{exc}"

    # ---- per-source collapse ----------------------------------------------

    def _collapse_snapshot_to_symbol(
        self, snap: dict[str, Any] | None
    ) -> tuple[dict[str, float], dict[str, list[ProvenanceEntry]], str | None]:
        """Collapse the snapshot ``positions`` list into ``{symbol: qty}``.

        Returns (qty_by_symbol, provenance_by_symbol, snapshot_ts_utc).
        """
        if not snap:
            return {}, {}, None
        ts = snap.get("ts_utc")
        positions_field = snap.get("positions")
        if not isinstance(positions_field, list):
            return {}, {}, ts
        qty_map: dict[str, float] = {}
        prov_map: dict[str, list[ProvenanceEntry]] = {}
        for row in positions_field:
            if not isinstance(row, dict):
                continue
            sym = row.get("symbol")
            qty = row.get("position")
            if not isinstance(sym, str) or qty is None:
                continue
            qty_map[sym] = qty_map.get(sym, 0.0) + float(qty)
            prov_map.setdefault(sym, []).append(
                ProvenanceEntry(
                    surface="snapshot",
                    ref=str(row.get("conId", sym)),
                    ts_utc=ts,
                )
            )
        return qty_map, prov_map, ts

    def _collapse_ledger_to_symbol(
        self, led: dict[str, Any] | None
    ) -> tuple[dict[str, float], dict[str, list[ProvenanceEntry]], str | None, list[str]]:
        """Collapse the ledger (hash-keyed dict) into ``{symbol: qty}``.

        The ledger has no global ts_utc — we derive it as
        ``max(opened_at_utc)`` across entries. Entries with unparseable
        timestamps are dropped from the timestamp derivation (but
        retained in the qty sum) with a warning.
        """
        warnings: list[str] = []
        if not led:
            return {}, {}, None, warnings
        qty_map: dict[str, float] = {}
        prov_map: dict[str, list[ProvenanceEntry]] = {}
        max_ts: datetime | None = None
        for key, entry in led.items():
            if not isinstance(entry, dict):
                continue
            sym = entry.get("symbol")
            qty = entry.get("qty")
            if not isinstance(sym, str) or qty is None:
                continue
            qty_map[sym] = qty_map.get(sym, 0.0) + float(qty)
            ts = entry.get("opened_at_utc")
            prov_map.setdefault(sym, []).append(
                ProvenanceEntry(surface="ledger", ref=str(key)[:16], ts_utc=ts)
            )
            parsed = _parse_iso_utc(ts)
            if parsed is None and ts is not None:
                warnings.append(f"ledger_entry_unparseable_ts:{key[:8]}")
            elif parsed is not None and (max_ts is None or parsed > max_ts):
                max_ts = parsed
        ledger_ts = max_ts.strftime("%Y-%m-%dT%H:%M:%SZ") if max_ts else None
        return qty_map, prov_map, ledger_ts, warnings

    # ---- merge rules (design §9) ------------------------------------------

    def _rule_m1_both_agree(
        self, *, s: float | None, l: float | None
    ) -> bool:
        """M1 trigger: both surfaces present and qty + sign agree."""
        if s is None or l is None:
            return False
        return float(s) == float(l)

    def _rule_m2_ledger_newer(
        self,
        *,
        s: float | None,
        l: float | None,
        snap_ts: datetime | None,
        ledger_ts: datetime | None,
    ) -> bool:
        """M2 trigger: ledger newer than snapshot, integer-bounded delta,
        within 2× snapshot cadence."""
        if s is None or l is None or snap_ts is None or ledger_ts is None:
            return False
        if ledger_ts <= snap_ts:
            return False
        delta = float(l) - float(s)
        if abs(delta) != float(int(abs(delta))):
            return False
        if (ledger_ts - snap_ts).total_seconds() > M2_MAX_LEAD_SECONDS:
            return False
        # Side must not be flipped — sign change is M4 territory.
        if (float(s) > 0 and float(l) < 0) or (float(s) < 0 and float(l) > 0):
            return False
        return True

    def _rule_m3_snapshot_newer(
        self,
        *,
        s: float | None,
        l: float | None,
        snap_ts: datetime | None,
        ledger_ts: datetime | None,
    ) -> bool:
        """M3 trigger: snapshot newer than ledger, integer-bounded delta."""
        if s is None or l is None or snap_ts is None or ledger_ts is None:
            return False
        if snap_ts <= ledger_ts:
            return False
        delta = float(l) - float(s)
        if abs(delta) != float(int(abs(delta))):
            return False
        if (float(s) > 0 and float(l) < 0) or (float(s) < 0 and float(l) > 0):
            return False
        return True

    def _rule_m4_structural_mismatch(
        self, *, s: float | None, l: float | None
    ) -> bool:
        """M4 trigger: fractional delta OR side mismatch OR sign flip."""
        if s is None or l is None:
            return False
        if float(s) == float(l):
            return False
        delta = float(l) - float(s)
        if abs(delta) != float(int(abs(delta))):
            return True
        if (float(s) > 0 and float(l) < 0) or (float(s) < 0 and float(l) > 0):
            return True
        return False

    def _rule_m5_stale_or_missing(
        self,
        *,
        s: float | None,
        l: float | None,
        snap_age: float | None,
        ledger_age: float | None,
    ) -> bool:
        """M5 trigger: source stale beyond TTL OR one-sided absence."""
        if s is None or l is None:
            return True
        if snap_age is None or ledger_age is None:
            return True
        if snap_age > self.snapshot_ttl:
            return True
        if ledger_age > self.ledger_ttl:
            return True
        return False

    # ---- per-symbol classifier --------------------------------------------

    def _classify_symbol(
        self,
        *,
        sym: str,
        s: float | None,
        l: float | None,
        snap_ts: datetime | None,
        ledger_ts: datetime | None,
        snap_age: float | None,
        ledger_age: float | None,
        chain: list[ProvenanceEntry],
        engine_ts_iso: str,
    ) -> PositionEntry:
        """Apply the 5 rules in the precedence order documented in §9."""
        delta: float | None = None
        if s is not None and l is not None:
            delta = float(l) - float(s)

        # Precedence (§9): M5 → M4 → M1 → M2 → M3
        if self._rule_m5_stale_or_missing(s=s, l=l, snap_age=snap_age, ledger_age=ledger_age):
            reason = (
                "missing_in_one_source"
                if (s is None) != (l is None)
                else "stale_source"
            )
            return PositionEntry(
                qty=None,
                side="UNKNOWN",
                value_source=VS_FAIL_CLOSED,
                snapshot_value=s,
                ledger_value=l,
                agreement=False,
                delta=delta,
                delta_reason=reason,
                merge_rule=RULE_M5,
                authority_decision=AD_FAIL_CLOSED,
                fail_closed=True,
                last_reconciled_utc=engine_ts_iso,
                provenance_chain=chain,
            )
        if self._rule_m4_structural_mismatch(s=s, l=l):
            return PositionEntry(
                qty=None,
                side="UNKNOWN",
                value_source=VS_DISAGREEMENT,
                snapshot_value=s,
                ledger_value=l,
                agreement=False,
                delta=delta,
                delta_reason="structural_mismatch",
                merge_rule=RULE_M4,
                authority_decision=AD_FAIL_CLOSED,
                fail_closed=True,
                last_reconciled_utc=engine_ts_iso,
                provenance_chain=chain,
            )
        if self._rule_m1_both_agree(s=s, l=l):
            return PositionEntry(
                qty=l,  # equal to s by definition
                side=derive_side(l),
                value_source=VS_BOTH,
                snapshot_value=s,
                ledger_value=l,
                agreement=True,
                delta=0,
                delta_reason="in_agreement",
                merge_rule=RULE_M1,
                authority_decision=AD_LEDGER,  # arbitrary; both agree
                fail_closed=False,
                last_reconciled_utc=engine_ts_iso,
                provenance_chain=chain,
            )
        if self._rule_m2_ledger_newer(s=s, l=l, snap_ts=snap_ts, ledger_ts=ledger_ts):
            return PositionEntry(
                qty=l,
                side=derive_side(l),
                value_source=VS_LEDGER,
                snapshot_value=s,
                ledger_value=l,
                agreement=False,
                delta=delta,
                delta_reason="ledger_lag_within_cadence",
                merge_rule=RULE_M2,
                authority_decision=AD_LEDGER,
                fail_closed=False,
                last_reconciled_utc=engine_ts_iso,
                provenance_chain=chain,
            )
        if self._rule_m3_snapshot_newer(s=s, l=l, snap_ts=snap_ts, ledger_ts=ledger_ts):
            return PositionEntry(
                qty=s,
                side=derive_side(s),
                value_source=VS_SNAPSHOT,
                snapshot_value=s,
                ledger_value=l,
                agreement=False,
                delta=delta,
                delta_reason="missed_fill_event",
                merge_rule=RULE_M3,
                authority_decision=AD_SNAPSHOT,
                fail_closed=True,
                last_reconciled_utc=engine_ts_iso,
                provenance_chain=chain,
            )
        # No rule fired — defensive FAIL_CLOSED (should not happen).
        return PositionEntry(
            qty=None,
            side="UNKNOWN",
            value_source=VS_FAIL_CLOSED,
            snapshot_value=s,
            ledger_value=l,
            agreement=False,
            delta=delta,
            delta_reason="no_rule_matched_defensive_fail_closed",
            merge_rule=RULE_M5,
            authority_decision=AD_FAIL_CLOSED,
            fail_closed=True,
            last_reconciled_utc=engine_ts_iso,
            provenance_chain=chain,
        )

    # ---- top-level merge ---------------------------------------------------

    def apply_merge_rules(self) -> PositionTruthV2:
        engine_dt = _utc_now()
        engine_ts_iso = engine_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        snap, snap_err = self.read_snapshot()
        led, led_err = self.read_ledger()

        snap_qty, snap_prov, snap_ts_str = self._collapse_snapshot_to_symbol(snap)
        led_qty, led_prov, led_ts_str, led_warnings = self._collapse_ledger_to_symbol(led)

        snap_ts = _parse_iso_utc(snap_ts_str)
        ledger_ts = _parse_iso_utc(led_ts_str)
        snap_age = (engine_dt - snap_ts).total_seconds() if snap_ts else None
        ledger_age = (engine_dt - ledger_ts).total_seconds() if ledger_ts else None

        snapshot_artifact = SourceArtifact(
            path=str(self.snapshot_path),
            ts_utc=snap_ts_str,
            sha256=_sha256_of_file(self.snapshot_path),
            age_seconds=snap_age,
        )
        ledger_artifact = SourceArtifact(
            path=str(self.ledger_path),
            ts_utc=led_ts_str,
            sha256=_sha256_of_file(self.ledger_path),
            age_seconds=ledger_age,
        )

        all_symbols = sorted(set(snap_qty.keys()) | set(led_qty.keys()))
        positions: dict[str, PositionEntry] = {}
        for sym in all_symbols:
            chain = snap_prov.get(sym, []) + led_prov.get(sym, [])
            entry = self._classify_symbol(
                sym=sym,
                s=snap_qty.get(sym),
                l=led_qty.get(sym),
                snap_ts=snap_ts,
                ledger_ts=ledger_ts,
                snap_age=snap_age,
                ledger_age=ledger_age,
                chain=chain,
                engine_ts_iso=engine_ts_iso,
            )
            positions[sym] = entry

        rules = [p.merge_rule for p in positions.values()]
        global_health = health_from_rules(rules)
        fail_closed_symbols = sorted(
            [s for s, p in positions.items() if p.fail_closed]
        )
        errors: list[str] = []
        if snap_err:
            errors.append(snap_err)
            global_health = HEALTH_RED
        if led_err:
            errors.append(led_err)
            global_health = HEALTH_RED

        return PositionTruthV2(
            ts_utc=engine_ts_iso,
            source_artifacts={"snapshot": snapshot_artifact, "ledger": ledger_artifact},
            positions=positions,
            global_authority_health=global_health,
            fail_closed_symbols=fail_closed_symbols,
            warnings=led_warnings,
            errors=errors,
        )

    def run(self) -> PositionTruthV2:
        """Compute the merged truth. Does **not** write any file."""
        return self.apply_merge_rules()

    def write_truth_v2(self, output_path: Path | str) -> None:
        """Atomically write the merged truth to disk (tmp + os.replace).

        This is the migration §12 Step-1 surface — exposed in code but
        not invoked by any production scheduler in this phase.
        """
        doc = self.run()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(
            json.dumps(serialize_to_dict(doc), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        os.replace(tmp, out)


# ---------------------------------------------------------------------------
# CLI (--check only; never writes)
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="chad.core.position_truth_engine",
        description="Run the position_truth_v2 engine in --check (read-only) mode.",
    )
    p.add_argument("--check", action="store_true", help="Run engine and print merged view (default).")
    p.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT_PATH), help="Path to positions_snapshot.json")
    p.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH), help="Path to ibkr_paper_ledger_state.json")
    p.add_argument("--snapshot-ttl", type=int, default=SNAPSHOT_TTL_SECONDS)
    p.add_argument("--ledger-ttl", type=int, default=LEDGER_TTL_SECONDS)
    args = p.parse_args(argv)

    eng = PositionTruthEngine(
        snapshot_path=args.snapshot,
        ledger_path=args.ledger,
        snapshot_ttl=args.snapshot_ttl,
        ledger_ttl=args.ledger_ttl,
    )
    doc = eng.run()
    print(json.dumps(serialize_to_dict(doc), indent=2, sort_keys=True, default=str))
    # Exit non-zero on RED so the CLI is wireable into a future health pipeline.
    return 0 if doc.global_authority_health != HEALTH_RED else 2


if __name__ == "__main__":
    sys.exit(main())
