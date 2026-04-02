from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def iter_ndjson(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def is_trusted_kraken_pnl(payload: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Trust policy (strict, deterministic):
    - Only Kraken rows are handled here.
    - We treat realized pnl as trusted only when:
        * side == SELL
        * pnl is finite and non-zero
        * record is marked enriched (kraken_enriched True) OR status closed
    Everything else is untrusted (entry-only, pending, or zero pnl).
    """
    broker = str(payload.get("broker", "")).lower().strip()
    if broker != "kraken":
        return False, "non_kraken_row"

    side = str(payload.get("side", "")).upper().strip()
    pnl = payload.get("pnl", None)
    pnl_v = safe_float(pnl, default=float("nan"))
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
    status = str(extra.get("status", payload.get("status", ""))).lower().strip()
    enriched = bool(extra.get("kraken_enriched", False))

    if side == "SELL" and math.isfinite(pnl_v) and abs(pnl_v) > 1e-12 and (enriched or status == "closed"):
        return True, "sell_realized_nonzero"

    # Common cases: BUY entries, pnl=0, or pending
    if side == "BUY":
        return False, "entry_only_buy"
    if math.isfinite(pnl_v) and abs(pnl_v) <= 1e-12:
        return False, "pnl_zero_or_unrealized"
    return False, "untrusted_or_incomplete"


def build_trusted_record(obj: Dict[str, Any]) -> Dict[str, Any]:
    payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
    trusted, reason = is_trusted_kraken_pnl(payload)

    # Do not mutate original payload; create a shallow copy + copy extra.
    payload2 = dict(payload)
    extra = payload2.get("extra") if isinstance(payload2.get("extra"), dict) else {}
    extra2 = dict(extra)

    extra2["pnl_trusted"] = bool(trusted)
    if not trusted:
        extra2["pnl_untrusted_reason"] = str(reason)
    else:
        extra2.pop("pnl_untrusted_reason", None)

    extra2["trusted_patch_at_utc"] = utc_now()
    extra2["trusted_patch_version"] = "kraken_pnl_trust_patch.v1"

    payload2["extra"] = extra2

    # Wrap as a new append-only “derived” record with its own hash, but preserve original hashes for traceability.
    orig_hash = str(obj.get("record_hash") or "")
    derived_payload = {
        "source": "kraken_pnl_trust_patch",
        "derived_at_utc": utc_now(),
        "original_record_hash": orig_hash,
        "payload": payload2,
    }
    rec_hash = sha256_hex(json.dumps(derived_payload, sort_keys=True, separators=(",", ":")))
    return {
        "record_hash": rec_hash,
        "original_record_hash": orig_hash,
        "timestamp_utc": utc_now(),
        "payload": payload2,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Create trusted Kraken PnL ledger (SELL non-zero pnl only).")
    ap.add_argument("--in", dest="inp", default="data/trades/trade_history_enriched.ndjson")
    ap.add_argument("--out", dest="outp", default="data/trades/trade_history_enriched_trusted.ndjson")
    ap.add_argument("--max-rows", type=int, default=1_000_000)
    args = ap.parse_args()

    inp = Path(args.inp).expanduser().resolve()
    outp = Path(args.outp).expanduser().resolve()

    if not inp.exists():
        raise SystemExit(f"missing_input: {inp}")

    n = 0
    trusted = 0
    untrusted = 0
    lines: list[str] = []

    for obj in iter_ndjson(inp):
        payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
        if str(payload.get("broker", "")).lower().strip() != "kraken":
            continue

        rec = build_trusted_record(obj)
        # Count trust
        extra = rec.get("payload", {}).get("extra", {}) if isinstance(rec.get("payload"), dict) else {}
        if isinstance(extra, dict) and bool(extra.get("pnl_trusted", False)):
            trusted += 1
        else:
            untrusted += 1

        lines.append(json.dumps(rec, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        n += 1
        if n >= int(args.max_rows):
            break

    atomic_write(outp, "\n".join(lines) + ("\n" if lines else ""))

    print(json.dumps({
        "ts_utc": utc_now(),
        "input": str(inp),
        "output": str(outp),
        "kraken_rows_written": n,
        "trusted": trusted,
        "untrusted": untrusted,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
