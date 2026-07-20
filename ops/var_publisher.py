#!/usr/bin/env python3
"""
ops/var_publisher.py

GAP-015A — report-only portfolio VaR publisher.

Reads local-only inputs (positions, prices, daily bars), computes a 1-day
parametric VaR via ``chad.risk.portfolio_var.compute_portfolio_var`` and
writes ``runtime/var_state.json`` (schema ``var_state.v1``).

Contract:
* Never calls a broker.
* Never modifies live mode or risk caps.
* Never wires into ``live_loop.py``.
* Always writes a valid schema, even on insufficient data (zeros).
* Emits a JSON summary on stdout for orchestration scripts.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chad.risk.portfolio_var import compute_portfolio_var, report_to_state_dict  # noqa: E402

RUNTIME_DIR = Path(os.environ.get("CHAD_RUNTIME_DIR", str(REPO_ROOT / "runtime"))).resolve()
VAR_STATE_PATH = RUNTIME_DIR / "var_state.json"
DEFAULT_TTL_SECONDS = 3600

# Market-data input freshness bound. A timer that recomputes VaR hourly would,
# without this gate, stamp ts_utc=now even against stale bars/prices — the
# "stale-as-fresh" failure re-emerging one layer down (invisible to EXS1 and the
# A4 metrics guard, which inspect only the artifact's own ts_utc). Default 48h
# tolerates normal weekend / market-closed gaps; override via env.
DEFAULT_INPUT_MAX_AGE_SECONDS = 172800  # 48h
DATA_BARS_1D_DIR = REPO_ROOT / "data" / "bars" / "1d"


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _input_max_age_from_env() -> int:
    raw = os.environ.get("CHAD_VAR_INPUT_MAX_AGE_SECONDS")
    if raw is None:
        return DEFAULT_INPUT_MAX_AGE_SECONDS
    try:
        v = int(float(raw))
        return v if v > 0 else DEFAULT_INPUT_MAX_AGE_SECONDS
    except Exception:
        return DEFAULT_INPUT_MAX_AGE_SECONDS


def compute_input_freshness(
    *,
    runtime_dir: Path,
    bars_dir: Optional[Path],
    symbols_used: list,
    max_age_seconds: int,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Return {inputs_fresh, oldest_input_age_seconds, oldest_input} for the
    market-data inputs that actually drive the VaR magnitude.

    Gated inputs: the positions file the loader would use (truth preferred, else
    snapshot), ``price_cache.json``, and the per-symbol daily bars in
    ``symbols_used``. Purely mtime-based; reads no file contents; never raises.

    Fail-closed: on any unexpected error, ``inputs_fresh=False`` so a stale read
    is never mistaken for fresh (observability only — VaR enforces nothing).
    When there are no gateable inputs on disk (degenerate/empty book), returns
    fresh with a null age, since the VaR ``status`` already reflects missing data
    as ``insufficient_data`` and there is no market data to be stale about.
    """
    try:
        now_ts = now if now is not None else time.time()
        base_bars = Path(bars_dir).resolve() if bars_dir else DATA_BARS_1D_DIR
        candidates = []  # list[(label, Path)]

        truth = runtime_dir / "positions_truth.json"
        snap = runtime_dir / "positions_snapshot.json"
        if truth.exists():
            candidates.append(("positions_truth.json", truth))
        elif snap.exists():
            candidates.append(("positions_snapshot.json", snap))

        price_cache = runtime_dir / "price_cache.json"
        if price_cache.exists():
            candidates.append(("price_cache.json", price_cache))

        for sym in symbols_used:
            bp = base_bars / f"{sym}.json"
            if bp.exists():
                candidates.append((f"bars/1d/{sym}.json", bp))

        oldest_age: Optional[float] = None
        oldest_label: Optional[str] = None
        for label, path in candidates:
            try:
                age = now_ts - path.stat().st_mtime
            except Exception:
                continue
            if age < 0:
                age = 0.0
            if oldest_age is None or age > oldest_age:
                oldest_age = age
                oldest_label = label

        if oldest_age is None:
            return {"inputs_fresh": True, "oldest_input_age_seconds": None, "oldest_input": None}

        return {
            "inputs_fresh": oldest_age <= max_age_seconds,
            "oldest_input_age_seconds": oldest_age,
            "oldest_input": oldest_label,
        }
    except Exception:
        return {
            "inputs_fresh": False,
            "oldest_input_age_seconds": None,
            "oldest_input": "<freshness_error>",
        }


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        pass


def publish_var(
    *,
    runtime_dir: Optional[Path] = None,
    bars_dir: Optional[Path] = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    allow_stale_inputs: bool = False,
    input_max_age_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute the VaR report and atomically write var_state.json. Returns the state dict."""
    rt = Path(runtime_dir).resolve() if runtime_dir else RUNTIME_DIR
    out_path = rt / "var_state.json"

    report = compute_portfolio_var(
        positions_truth_path=rt / "positions_truth.json",
        positions_snapshot_path=rt / "positions_snapshot.json",
        price_cache_path=rt / "price_cache.json",
        portfolio_snapshot_path=rt / "portfolio_snapshot.json",
        pnl_state_path=rt / "pnl_state.json",
        bars_dir=bars_dir,
    )

    max_age = (
        int(input_max_age_seconds)
        if input_max_age_seconds is not None
        else _input_max_age_from_env()
    )
    freshness = compute_input_freshness(
        runtime_dir=rt,
        bars_dir=bars_dir,
        symbols_used=list(report.symbols_used),
        max_age_seconds=max_age,
    )

    # Downgrade only the "ok" path: stale market data behind an otherwise-valid
    # VaR is the stale-as-fresh hazard. insufficient_data already reads not-ok to
    # the A4 consumer, so leave it untouched. --allow-stale-inputs suppresses the
    # downgrade for manual diagnostic runs but never masks the recorded fields.
    effective_status = report.status
    if report.status == "ok" and not freshness["inputs_fresh"] and not allow_stale_inputs:
        effective_status = "stale_inputs"

    state = report_to_state_dict(
        report,
        ts_utc=utc_now_iso(),
        ttl_seconds=ttl_seconds,
        status_override=effective_status,
        inputs_fresh=freshness["inputs_fresh"],
        oldest_input_age_seconds=freshness["oldest_input_age_seconds"],
        oldest_input=freshness["oldest_input"],
        input_max_age_seconds=max_age,
    )
    atomic_write_json(out_path, state)
    return state


def main(argv: Optional[list] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Report-only portfolio VaR publisher (writes runtime/var_state.json)."
    )
    parser.add_argument(
        "--allow-stale-inputs",
        action="store_true",
        help=(
            "Do not downgrade status to stale_inputs when market-data inputs "
            "exceed the freshness bound (diagnostic runs only; the recorded "
            "inputs_fresh/oldest_input fields are never masked)."
        ),
    )
    args = parser.parse_args(argv)

    state = publish_var(allow_stale_inputs=args.allow_stale_inputs)
    summary = {
        "ok": True,
        "artifact": str(VAR_STATE_PATH),
        "schema_version": state.get("schema_version"),
        "status": state.get("status"),
        "var_95_1day_usd": state.get("var_95_1day_usd"),
        "var_99_1day_usd": state.get("var_99_1day_usd"),
        "symbol_count": state.get("symbol_count"),
        "enforcement_active": state.get("enforcement_active", False),
        "inputs_fresh": state.get("inputs_fresh"),
        "oldest_input": state.get("oldest_input"),
        "oldest_input_age_seconds": state.get("oldest_input_age_seconds"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
