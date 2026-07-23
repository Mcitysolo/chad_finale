"""
chad/analytics/excursion_recorder.py — W5A E3 MAE/MFE excursion.

Max Adverse / Max Favorable Excursion per open position, persisted per lap AT
CLOSE. Per rider R4, this does NOT add a second position iterator: the exit
overlay already walks every position every cycle, and W5A-3 extends that ONE
walk with true high/low WATERMARK fields (`hwm`/`lwm`) — kept SEPARATE from the
overlay's `peak`/`trough`, which are load-bearing for the trailing ATR stop and
must never move (observer-class). At the overlay's existing confirmed-close
detection (`_save_anchors`, before the prune), the accumulated watermarks are
snapshotted here into `data/exit_overlay/excursion_YYYYMMDD.ndjson`
(`mae_mfe.v1`) — a serialization at the walk's close edge, not a new walker.

Why watermarks and not the point sample: the overlay samples a POINT price each
~72s cycle, so an intracycle extreme between marks is invisible. Folding the
latest bar's high/low gives a true (bar-resolution) watermark. Granularity is
bounded by the daily bar cache + cycle cadence — `excursion_source` says so.

R1 honest nulls: MAE/MFE is null-with-reason when entry or the relevant
watermark is unknown — never a fabricated 0.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE_DIR = REPO_ROOT / "data" / "exit_overlay"

EXCURSION_SCHEMA_VERSION = "mae_mfe.v1"
E3_EXCURSION_ENV = "CHAD_E3_EXCURSION"

MODE_OFF = "off"
MODE_SIDECAR = "sidecar"
MODE_STAMP = "stamp"
_VALID_MODES = frozenset({MODE_OFF, MODE_SIDECAR, MODE_STAMP})


def e3_mode(env: Optional[Mapping[str, str]] = None) -> str:
    """CHAD_E3_EXCURSION: off | sidecar | stamp (garbage → off).
    off ⇒ no watermark tracking, no sidecar (byte-identical anchor state);
    sidecar ⇒ track hwm/lwm + write the close-time excursion row;
    stamp ⇒ additionally stamp mae/mfe onto the closed_trade (W5A-4)."""
    src = env if env is not None else os.environ
    raw = str(src.get(E3_EXCURSION_ENV, "")).strip().lower()
    return raw if raw in _VALID_MODES else MODE_OFF


def excursion_tracking_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    return e3_mode(env) in (MODE_SIDECAR, MODE_STAMP)


def _f(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def latest_bar_hilo(bars: Any) -> "tuple[Optional[float], Optional[float]]":
    """(high, low) of the most-recent bar carrying valid high/low, else
    (None, None). The same bar high/low the overlays' ATR already reads."""
    for b in reversed(list(bars or [])):
        if not isinstance(b, Mapping):
            continue
        hi = _f(b.get("high"))
        lo = _f(b.get("low"))
        if hi > 0.0 or lo > 0.0:
            return (hi if hi > 0.0 else None, lo if lo > 0.0 else None)
    return (None, None)


def update_watermarks(
    prior: Optional[Mapping[str, Any]],
    price: float,
    bar_high: Optional[float],
    bar_low: Optional[float],
) -> Dict[str, Any]:
    """Extend the running high/low watermark with this cycle's point price AND
    the latest bar's true high/low (R4). Returns {hwm, lwm, excursion_source}.
    Kept separate from peak/trough by the caller."""
    prior_hwm = _f(prior.get("hwm")) if isinstance(prior, Mapping) else 0.0
    prior_lwm = _f(prior.get("lwm")) if isinstance(prior, Mapping) else 0.0
    hi_cands = [v for v in (prior_hwm, price, _f(bar_high)) if v and v > 0.0]
    lo_cands = [v for v in (prior_lwm, price, _f(bar_low)) if v and v > 0.0]
    hwm = max(hi_cands) if hi_cands else price
    lwm = min(lo_cands) if lo_cands else price
    used_bar = bool((_f(bar_high) > 0.0) or (_f(bar_low) > 0.0))
    return {
        "hwm": hwm,
        "lwm": lwm,
        "excursion_source": "watermark_bar_hilo" if used_bar else "watermark_point_only",
    }


def compute_mae_mfe(
    *,
    entry_price: float,
    hwm: float,
    lwm: float,
    side: str,
    quantity: float,
    contract_multiplier: float = 1.0,
) -> Dict[str, Any]:
    """MAE ≤ 0 (worst adverse), MFE ≥ 0 (best favorable), entry-relative.
    Long: MFE at hwm, MAE at lwm. Short: MFE at lwm, MAE at hwm.
    Honest nulls (R1) when entry or the relevant watermark is unknown."""
    e = _f(entry_price)
    h = _f(hwm)
    l = _f(lwm)
    q = abs(_f(quantity))
    m = _f(contract_multiplier) or 1.0
    s = str(side or "").strip().upper()

    out: Dict[str, Any] = {
        "mae_pct": None, "mae_usd": None, "mfe_pct": None, "mfe_usd": None,
        "mae_reason": "resolved", "mfe_reason": "resolved",
    }
    if e <= 0.0:
        out.update(mae_reason="no_entry_price", mfe_reason="no_entry_price")
        return out

    if s == "BUY":
        fav_wm, adv_wm = h, l
    elif s == "SELL":
        fav_wm, adv_wm = l, h
    else:
        out.update(mae_reason="unknown_side", mfe_reason="unknown_side")
        return out

    # MFE (favorable): long→(hwm-e), short→(e-lwm); ≥0.
    if fav_wm > 0.0:
        fav = (fav_wm - e) if s == "BUY" else (e - fav_wm)
        out["mfe_pct"] = round(fav / e, 6)
        out["mfe_usd"] = round(fav * q * m, 6) if q else None
        if not q:
            out["mfe_reason"] = "no_quantity_usd_only"
    else:
        out["mfe_reason"] = "no_favorable_watermark"

    # MAE (adverse): long→(lwm-e), short→(e-hwm); ≤0.
    if adv_wm > 0.0:
        adv = (adv_wm - e) if s == "BUY" else (e - adv_wm)
        out["mae_pct"] = round(adv / e, 6)
        out["mae_usd"] = round(adv * q * m, 6) if q else None
        if not q:
            out["mae_reason"] = "no_quantity_usd_only"
    else:
        out["mae_reason"] = "no_adverse_watermark"

    return out


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_excursion_row(
    position_key: str,
    anchor: Mapping[str, Any],
    *,
    lane: str,
    closed_detect_utc: Optional[str] = None,
) -> Dict[str, Any]:
    """One `mae_mfe.v1` row from a closing anchor. Join key for the harness:
    (strategy, symbol, opened_at_utc)."""
    strategy = position_key.split("|", 1)[0] if "|" in position_key else ""
    symbol = position_key.split("|", 1)[1] if "|" in position_key else position_key
    side = str(anchor.get("side") or "").upper()
    entry = _f(anchor.get("entry_price"))
    hwm = _f(anchor.get("hwm"))
    lwm = _f(anchor.get("lwm"))
    qty = _f(anchor.get("qty"))
    mm = compute_mae_mfe(
        entry_price=entry, hwm=hwm, lwm=lwm, side=side, quantity=qty,
        contract_multiplier=_f(anchor.get("contract_multiplier")) or 1.0,
    )
    return {
        "schema_version": EXCURSION_SCHEMA_VERSION,
        "ts_utc": closed_detect_utc or _now_iso(),
        "lane": lane,
        "position_key": position_key,
        "strategy": strategy,
        "symbol": symbol.upper(),
        "side": side,
        "entry_price": entry or None,
        "hwm": hwm or None,
        "lwm": lwm or None,
        "quantity": qty or None,
        "opened_at_utc": anchor.get("opened_at_utc"),
        "closed_detect_utc": closed_detect_utc or _now_iso(),
        "excursion_source": anchor.get("excursion_source"),
        **mm,
    }


def record_excursion_at_close(
    position_key: str,
    anchor: Mapping[str, Any],
    *,
    lane: str,
    evidence_dir: Optional[Path] = None,
    now_iso: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Append the close-time excursion row. Best-effort — never raises upward
    (a measurement write must never affect the overlay). Returns the row, or
    None on write failure."""
    row = build_excursion_row(position_key, anchor, lane=lane, closed_detect_utc=now_iso)
    target_dir = Path(evidence_dir) if evidence_dir is not None else DEFAULT_EVIDENCE_DIR
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        date = (row["ts_utc"] or _now_iso())[:10].replace("-", "")
        path = target_dir / f"excursion_{date}.ndjson"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        return row
    except Exception as exc:  # noqa: BLE001 — measurement is best-effort
        LOG.warning("excursion_write_failed key=%s err=%s", position_key, exc)
        return None


__all__ = [
    "E3_EXCURSION_ENV",
    "EXCURSION_SCHEMA_VERSION",
    "MODE_OFF",
    "MODE_SIDECAR",
    "MODE_STAMP",
    "build_excursion_row",
    "compute_mae_mfe",
    "e3_mode",
    "excursion_tracking_enabled",
    "latest_bar_hilo",
    "record_excursion_at_close",
    "update_watermarks",
]
