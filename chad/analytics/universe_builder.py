#!/usr/bin/env python3
"""
chad/analytics/universe_builder.py

CHAD — Universe Builder (Phase 11)
----------------------------------
A read-only, audit-first "universe refresh" module intended to be run by systemd:

    python3 -m chad.analytics.universe_builder

What it does
------------
- Builds a deterministic list of tickers ("universe") from *existing* on-disk artifacts:
  1) config/universe.json                     (preferred, operator-controlled)
  2) runtime/full_execution_cycle_last.json   (fallback: summary.tick_symbols)
  3) control/polygon_universe.txt             (fallback: one ticker per line)

- Writes runtime/universe.json (atomic replace, stable shape).

Hard guarantees
---------------
- No broker calls, no network calls.
- No strategy logic.
- No order generation.
- Safe for timers: **never crashes the service loop due to missing inputs**.
  If inputs are missing/invalid, it still writes an "ok=false" universe artifact and exits 0.

Determinism rules
-----------------
- If config/universe.json exists and contains tickers, those define the primary order.
- Additional sources only append *new* symbols, preserving existing order.
- Symbols are normalized and validated (uppercase, safe charset).
- Output is capped to max_symbols.

Environment (optional)
----------------------
- CHAD_REPO_DIR            default: /home/ubuntu/chad_finale
- CHAD_RUNTIME_DIR         default: <repo>/runtime
- CHAD_CONFIG_DIR          default: <repo>/config
- CHAD_CONTROL_DIR         default: <repo>/control
- CHAD_UNIVERSE_MAX_SYMBOLS default: 25

Exit codes
----------
- Always 0 after writing runtime/universe.json (even when ok=false).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SYMBOL_RE = re.compile(r"^[A-Z0-9.\-]{1,20}$")


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if not v:
        return int(default)
    try:
        return int(str(v).strip())
    except Exception:
        return int(default)


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="strict")).hexdigest()


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    try:
        if not path.is_file():
            return None, f"missing:{path}"
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"parse_error:{path.name}:{type(exc).__name__}:{exc}"


def _safe_str(x: Any, default: str = "") -> str:
    try:
        return str(x)
    except Exception:
        return default


def _normalize_symbol(sym: str) -> str:
    s = _safe_str(sym).strip().upper()
    return s


def _valid_symbol(sym: str) -> bool:
    if not sym:
        return False
    if not SYMBOL_RE.fullmatch(sym):
        return False
    return True


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in items:
        sym = _normalize_symbol(raw)
        if not _valid_symbol(sym):
            continue
        if sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _load_symbols_from_config_universe(path: Path) -> Tuple[List[str], Optional[str]]:
    """
    Accepts any of:
      {"tickers":[...]} | {"symbols":[...]} | {"universe":[...]} | [...]
    """
    obj, err = _read_json(path)
    if err:
        return [], err
    if isinstance(obj, list):
        return _dedupe_preserve_order([_safe_str(x) for x in obj]), None
    if isinstance(obj, dict):
        for key in ("tickers", "symbols", "universe"):
            v = obj.get(key)
            if isinstance(v, list):
                return _dedupe_preserve_order([_safe_str(x) for x in v]), None
        return [], "config_universe_no_list_keys"
    return [], "config_universe_invalid_shape"


def _load_symbols_from_plan(path: Path) -> Tuple[List[str], Optional[str]]:
    """
    Reads runtime/full_execution_cycle_last.json and attempts:
      - summary.tick_symbols (preferred)
      - summary.tickSymbols (legacy)
      - any list under keys named tick_symbols/tickers/symbols (best effort)
    """
    obj, err = _read_json(path)
    if err:
        return [], err
    if not isinstance(obj, dict):
        return [], "plan_invalid_shape"
    summ = obj.get("summary")
    if isinstance(summ, dict):
        for key in ("tick_symbols", "tickSymbols"):
            v = summ.get(key)
            if isinstance(v, list):
                return _dedupe_preserve_order([_safe_str(x) for x in v]), None

    # Best-effort scan (bounded)
    candidates: List[str] = []
    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                lk = str(k).lower()
                if lk in ("tick_symbols", "tickers", "symbols", "universe") and isinstance(v, list):
                    candidates.extend([_safe_str(xx) for xx in v])
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)
    out = _dedupe_preserve_order(candidates)
    if out:
        return out, None
    return [], "plan_no_symbols_found"


def _load_symbols_from_text_file(path: Path) -> Tuple[List[str], Optional[str]]:
    """
    control/polygon_universe.txt format: one ticker per line, allows comments '#'
    """
    if not path.is_file():
        return [], f"missing:{path}"
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        items: List[str] = []
        for raw in lines:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            items.append(s)
        return _dedupe_preserve_order(items), None
    except Exception as exc:
        return [], f"text_read_error:{path.name}:{type(exc).__name__}:{exc}"


@dataclass(frozen=True)
class Paths:
    repo_dir: Path
    runtime_dir: Path
    config_dir: Path
    control_dir: Path

    @staticmethod
    def from_env() -> "Paths":
        repo_dir = Path(os.environ.get("CHAD_REPO_DIR", "/home/ubuntu/chad_finale")).resolve()
        runtime_dir = Path(os.environ.get("CHAD_RUNTIME_DIR", str(repo_dir / "runtime"))).resolve()
        config_dir = Path(os.environ.get("CHAD_CONFIG_DIR", str(repo_dir / "config"))).resolve()
        control_dir = Path(os.environ.get("CHAD_CONTROL_DIR", str(repo_dir / "control"))).resolve()
        return Paths(repo_dir=repo_dir, runtime_dir=runtime_dir, config_dir=config_dir, control_dir=control_dir)


def build_universe(*, paths: Paths, max_symbols: int) -> Dict[str, Any]:
    ts = utc_now_iso()
    max_symbols = int(max(1, max_symbols))

    cfg_universe_path = paths.config_dir / "universe.json"
    plan_path = paths.runtime_dir / "full_execution_cycle_last.json"
    polygon_universe_path = paths.control_dir / "polygon_universe.txt"
    out_path = paths.runtime_dir / "universe.json"

    warnings: List[str] = []
    sources: Dict[str, Any] = {}

    # 1) config/universe.json (preferred)
    cfg_syms, cfg_err = _load_symbols_from_config_universe(cfg_universe_path)
    sources["config_universe"] = {"path": str(cfg_universe_path), "count": len(cfg_syms), "error": cfg_err}
    if cfg_err:
        warnings.append(f"config_universe:{cfg_err}")

    # 2) runtime plan tick_symbols (fallback)
    plan_syms, plan_err = _load_symbols_from_plan(plan_path)
    sources["plan_tick_symbols"] = {"path": str(plan_path), "count": len(plan_syms), "error": plan_err}
    if plan_err:
        warnings.append(f"plan_tick_symbols:{plan_err}")

    # 3) polygon universe txt (fallback)
    pol_syms, pol_err = _load_symbols_from_text_file(polygon_universe_path)
    sources["polygon_universe_txt"] = {"path": str(polygon_universe_path), "count": len(pol_syms), "error": pol_err}
    if pol_err:
        warnings.append(f"polygon_universe_txt:{pol_err}")

    # Merge in deterministic order:
    # - primary: config universe if present, else plan if present, else polygon txt.
    primary: List[str] = cfg_syms or plan_syms or pol_syms
    merged: List[str] = list(primary)

    # Append extras without disturbing primary order
    for extra in (cfg_syms, plan_syms, pol_syms):
        for sym in extra:
            if sym not in merged:
                merged.append(sym)

    merged = merged[:max_symbols]
    ok = bool(merged)

    payload: Dict[str, Any] = {
        "schema_version": "universe.v1",
        "ts_utc": ts,
        "ok": ok,
        "max_symbols": max_symbols,
        "symbols": merged,
        "counts": {
            "symbols": len(merged),
        },
        "sources": sources,
        "warnings": warnings,
        "artifact": {
            "out_path": str(out_path),
            "symbols_sha256": _sha256_hex(_canonical_json(merged)),
        },
    }

    if not ok:
        payload["reason"] = "no_valid_symbols_from_sources"

    return payload


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CHAD Universe Builder (read-only)")
    ap.add_argument("--max-symbols", type=int, default=_env_int("CHAD_UNIVERSE_MAX_SYMBOLS", 25))
    ap.add_argument("--out", type=str, default="", help="Override output path (default runtime/universe.json)")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    paths = Paths.from_env()

    payload = build_universe(paths=paths, max_symbols=int(args.max_symbols))

    out_path = Path(args.out).expanduser().resolve() if str(args.out).strip() else (paths.runtime_dir / "universe.json")
    # Write artifact
    payload["artifact"]["out_path"] = str(out_path)
    _atomic_write_json(out_path, payload)

    # Print minimal stable stdout for systemd logs
    summary = {
        "ts_utc": payload.get("ts_utc"),
        "ok": payload.get("ok"),
        "symbols_count": payload.get("counts", {}).get("symbols"),
        "out_path": str(out_path),
        "symbols_sha256": payload.get("artifact", {}).get("symbols_sha256"),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
