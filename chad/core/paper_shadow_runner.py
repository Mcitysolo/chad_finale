#!/usr/bin/env python3
"""
CHAD Paper Shadow Runner — Production-Grade (IBKR-truth, fail-closed)

Purpose
-------
This runner is CHAD's "paper execution truth engine" for SCR.

It:
- Reads the orchestrator plan (runtime/full_execution_cycle_last.json by default).
- Selects ONE deterministic BUY intent.
- Enforces operator intent gates (DENY_ALL / EXIT_ONLY / ALLOW_LIVE).
- Enforces IBKR-truth market gate (holiday + session aware) using contractDetails.
- Executes ONE paper round-trip (BUY -> HOLD -> SELL) when allowed.
- Writes ONLY realized results to CHAD's trade ledger (append-only NDJSON + hash chain).
- Writes a structured run report to reports/shadow/ (safe to delete).
- Provides best-effort exactly-once semantics for ledger writes via sqlite idempotency store.

Safety invariants
-----------------
- Preview mode: no connect, no orders, no ledger writes.
- Execute mode: requires explicit --execute flag.
- Operator DENY_ALL: hard block.
- Operator EXIT_ONLY: blocks new entries (this runner only opens a position). Hard block.
- Market CLOSED (IBKR liquidHours): hard block, no orders.
- No ledger writes unless a full realized round-trip occurs.
- One run = at most one round-trip attempt.
- Exactly-once ledger: duplicate trade_id suppresses second write.

Plan semantics
--------------
The plan may contain only SELL intents (e.g. portfolio closing).
This runner requires a BUY intent. If none exists, it can optionally generate a
deterministic canary BUY intent using summary.tick_symbols.

Environment
-----------
CHAD_RUNTIME_DIR                (default: <repo_root>/runtime)
CHAD_PLANNED_ORDERS_PATH        (default: <runtime_dir>/full_execution_cycle_last.json)
CHAD_SHADOW_HOLD_SECONDS        (default from CLI, capped)
CHAD_SHADOW_AUTO_CANCEL_SECONDS (default from CLI)
CHAD_SHADOW_SIZE_MULTIPLIER     (default from CLI)
CHAD_SHADOW_CANARY_FALLBACK     ("1"/"0", default: 1)
CHAD_SHADOW_CANARY_MAX_QTY      (default: 1.0)
IBKR_HOST / IBKR_PORT / IBKR_CLIENT_ID

Exit codes
----------
0  success (including clean blocks)
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from chad.execution.idempotency_store import IdempotencyStore, default_paper_db_path

LOGGER = logging.getLogger("chad.paper_shadow_runner")
# ##ARM_GATING_COMPAT_V1##
# ---------------------------------------------------------------------------
# Backward-compatible arming + execute gating API (TEST CONTRACT)
# ---------------------------------------------------------------------------
# These symbols are intentionally lightweight and must NOT import ib_insync.
# They exist so safety gating remains auditable and stable across refactors.

ARM_ENV_NAME = "CHAD_PAPER_SHADOW_ARM"
ARM_PHRASE = "I_UNDERSTAND_PAPER_CAN_PLACE_ORDERS"


@dataclass(frozen=True)
class PaperShadowConfig:
    """
    Minimal config surface used by gating tests.
    enabled: feature flag
    armed:   operator-configured arming flag
    """
    enabled: bool = True
    armed: bool = False


def is_armed() -> bool:
    """
    True only when env var is set to the exact arming phrase.
    Fail-safe default is False.
    """
    v = os.environ.get(ARM_ENV_NAME, "")
    return str(v).strip() == ARM_PHRASE


def should_place_paper_orders(cfg: PaperShadowConfig) -> tuple[bool, list[str]]:
    """
    Public gating: used by tests and callers to decide if paper orders may be placed.
    Does not touch brokers, does not import ib_insync.
    """
    reasons: list[str] = []
    if not bool(cfg.enabled):
        reasons.append("enabled is false")
        return False, reasons

    if not is_armed():
        reasons.append(f"{ARM_ENV_NAME} not set to arm phrase")
        return False, reasons

    reasons.append("env-armed")
    return True, reasons


def _should_execute_paper(cfg: PaperShadowConfig) -> tuple[bool, list[str]]:
    """
    Stricter gate for EXECUTE mode: requires cfg.enabled AND cfg.armed AND env phrase.
    """
    reasons: list[str] = []
    if not bool(cfg.enabled):
        reasons.append("enabled is false")
        return False, reasons

    if not bool(cfg.armed):
        reasons.append("armed is false")
        return False, reasons

    if not is_armed():
        reasons.append("arm phrase missing")
        return False, reasons

    reasons.append("env-armed")
    return True, reasons


def _live_gate_allows_paper() -> tuple[bool, str]:
    """
    Best-effort LiveGate probe.
    MUST NOT raise; fail-safe returns (False, reason).
    """
    try:
        import json
        from urllib.request import Request, urlopen

        url = os.environ.get("CHAD_LIVE_GATE_URL", "http://127.0.0.1:9618/live-gate")
        req = Request(url, headers={"User-Agent": "chad-paper-shadow-gate/1.0"})
        with urlopen(req, timeout=2) as resp:
            raw = resp.read()
        obj = json.loads(raw.decode("utf-8"))
        allow = bool(obj.get("allow_ibkr_paper", False))
        return allow, "ok"
    except Exception as e:
        return False, f"live_gate_unreachable:{type(e).__name__}"

MAX_HOLD_SECONDS = 10 * 60
DEFAULT_CONTRACT_CACHE_TTL_SECONDS = 60 * 10  # 10 minutes
DEFAULT_CONNECT_RETRIES = 2
DEFAULT_DETAILS_RETRIES = 2


# =============================================================================
# Small utilities (safe, deterministic)
# =============================================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat(timespec="microseconds").replace("+00:00", "Z")


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        f = float(x)
        if f != f or f in (float("inf"), float("-inf")):
            return default
        return f
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def fsync_append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def jitter_sleep(base_s: float, *, jitter_frac: float = 0.25) -> None:
    j = base_s * jitter_frac
    time.sleep(max(0.0, base_s + random.uniform(-j, j)))


# =============================================================================
# Config + IO contracts
# =============================================================================

@dataclass(frozen=True)
class RunnerConfig:
    repo_root: Path
    runtime_dir: Path
    trades_dir: Path
    reports_dir: Path

    planned_path: Path

    preview: bool
    execute: bool

    hold_seconds: int
    auto_cancel_seconds: float
    size_multiplier: float

    canary_fallback: bool
    canary_max_qty: float

    ib_host: str
    ib_port: int
    ib_client_id: int

    contract_cache_ttl_seconds: int
    connect_retries: int
    details_retries: int

    @staticmethod
    def from_env_args(args: argparse.Namespace) -> "RunnerConfig":
        repo_root = Path(__file__).resolve().parents[2]
        runtime_dir = Path(os.environ.get("CHAD_RUNTIME_DIR", str(repo_root / "runtime"))).resolve()
        trades_dir = (repo_root / "data" / "trades").resolve()
        reports_dir = (repo_root / "reports" / "shadow").resolve()

        planned_path = Path(
            os.environ.get(
                "CHAD_PLANNED_ORDERS_PATH",
                str(runtime_dir / "full_execution_cycle_last.json"),
            )
        ).resolve()

        hold_s = int(
            max(
                1,
                min(
                    int(safe_int(os.environ.get("CHAD_SHADOW_HOLD_SECONDS", args.hold_seconds), args.hold_seconds)),
                    MAX_HOLD_SECONDS,
                ),
            )
        )
        cancel_s = float(max(1.0, safe_float(os.environ.get("CHAD_SHADOW_AUTO_CANCEL_SECONDS", args.auto_cancel_seconds), args.auto_cancel_seconds)))
        mult = float(max(0.0, safe_float(os.environ.get("CHAD_SHADOW_SIZE_MULTIPLIER", args.size_multiplier), args.size_multiplier)))

        canary_fallback = env_bool("CHAD_SHADOW_CANARY_FALLBACK", True)
        canary_max_qty = float(max(0.0, safe_float(os.environ.get("CHAD_SHADOW_CANARY_MAX_QTY", "1.0"), 1.0)))
        if canary_max_qty <= 0:
            canary_max_qty = 1.0

        ib_host = os.environ.get("IBKR_HOST", "127.0.0.1").strip()
        ib_port = safe_int(os.environ.get("IBKR_PORT", "4002"), 4002)
        ib_client_id = safe_int(os.environ.get("IBKR_CLIENT_ID", "9013"), 9013)

        ttl = safe_int(os.environ.get("CHAD_CONTRACT_CACHE_TTL_SECONDS", str(DEFAULT_CONTRACT_CACHE_TTL_SECONDS)), DEFAULT_CONTRACT_CACHE_TTL_SECONDS)
        c_retries = safe_int(os.environ.get("CHAD_IBKR_CONNECT_RETRIES", str(DEFAULT_CONNECT_RETRIES)), DEFAULT_CONNECT_RETRIES)
        d_retries = safe_int(os.environ.get("CHAD_IBKR_DETAILS_RETRIES", str(DEFAULT_DETAILS_RETRIES)), DEFAULT_DETAILS_RETRIES)

        return RunnerConfig(
            repo_root=repo_root,
            runtime_dir=runtime_dir,
            trades_dir=trades_dir,
            reports_dir=reports_dir,
            planned_path=planned_path,
            preview=bool(args.preview) or (not bool(args.execute)),
            execute=bool(args.execute),
            hold_seconds=hold_s,
            auto_cancel_seconds=cancel_s,
            size_multiplier=mult,
            canary_fallback=bool(canary_fallback),
            canary_max_qty=float(canary_max_qty),
            ib_host=ib_host,
            ib_port=ib_port,
            ib_client_id=ib_client_id,
            contract_cache_ttl_seconds=int(max(30, ttl)),
            connect_retries=int(max(0, c_retries)),
            details_retries=int(max(0, d_retries)),
        )


@dataclass(frozen=True)
class PlannedIntent:
    strategy: str
    symbol: str
    side: str
    quantity: float
    exchange: str
    currency: str
    sec_type: str

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PlannedIntent":
        return PlannedIntent(
            strategy=str(d.get("strategy") or "beta").strip().lower() or "beta",
            symbol=str(d.get("symbol") or "").strip().upper(),
            side=str(d.get("side") or "BUY").strip().upper(),
            quantity=float(safe_float(d.get("quantity"), 0.0)),
            exchange=str(d.get("exchange") or "SMART").strip().upper(),
            currency=str(d.get("currency") or "USD").strip().upper(),
            sec_type=str(d.get("sec_type") or "STK").strip().upper(),
        )


@dataclass(frozen=True)
class TradeResult:
    strategy: str
    symbol: str
    quantity: float
    entry_fill_price: float
    exit_fill_price: float
    realized_pnl: float
    ts_entry_utc: str
    ts_exit_utc: str
    broker: str
    is_live: bool
    order_id_entry: Optional[int]
    order_id_exit: Optional[int]
    perm_id_entry: Optional[int]
    perm_id_exit: Optional[int]
    extra: Dict[str, Any]

    def trade_id(self) -> str:
        payload = {
            "strategy": self.strategy,
            "symbol": self.symbol,
            "qty": round(float(self.quantity), 8),
            "entry_ts": self.ts_entry_utc,
            "exit_ts": self.ts_exit_utc,
            "entry_fp": round(float(self.entry_fill_price), 6),
            "exit_fp": round(float(self.exit_fill_price), 6),
        }
        return sha256_hex(canonical_json(payload))


# =============================================================================
# Ledger (append-only + hash chain + exactly-once gate)
# =============================================================================

class TradeLedger:
    def __init__(self, *, trades_dir: Path, repo_root: Path) -> None:
        self._dir = trades_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._repo_root = repo_root

    def _path_for_day(self, now: datetime) -> Path:
        return self._dir / f"trade_history_{now.strftime('%Y%m%d')}.ndjson"

    def _prev_hash(self, path: Path) -> str:
        if not path.exists():
            return "0" * 64
        try:
            with path.open("rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                if size <= 0:
                    return "0" * 64
                f.seek(max(0, size - 8192), os.SEEK_SET)
                chunk = f.read().decode("utf-8", errors="ignore")
            lines = [ln for ln in chunk.splitlines() if ln.strip()]
            if not lines:
                return "0" * 64
            last = json.loads(lines[-1])
            rh = str(last.get("record_hash") or ("0" * 64))
            return rh if len(rh) == 64 else ("0" * 64)
        except Exception:
            return "0" * 64

    def append_trade_result(self, tr: TradeResult) -> Tuple[Path, bool]:
        """
        Append a realized trade record to the ledger.

        Returns: (ledger_path, wrote_bool)
          - wrote_bool=False means duplicate trade_id was suppressed.
        """
        now = utc_now()
        path = self._path_for_day(now)
        prev = self._prev_hash(path)

        payload = {
            "broker": tr.broker,
            "is_live": tr.is_live,
            "strategy": tr.strategy,
            "symbol": tr.symbol,
            "side": "SELL",  # realized close event
            "quantity": float(tr.quantity),
            "entry_fill_price": float(tr.entry_fill_price),
            "exit_fill_price": float(tr.exit_fill_price),
            "realized_pnl": float(tr.realized_pnl),
            "entry_time_utc": tr.ts_entry_utc,
            "exit_time_utc": tr.ts_exit_utc,
            "order_id_entry": tr.order_id_entry,
            "order_id_exit": tr.order_id_exit,
            "perm_id_entry": tr.perm_id_entry,
            "perm_id_exit": tr.perm_id_exit,
            "trade_id": tr.trade_id(),
            "extra": tr.extra,
        }

        rec = {
            "timestamp_utc": iso_utc(now),
            "payload": payload,
            "prev_hash": prev,
        }
        rec_hash = sha256_hex(canonical_json(rec))
        rec["record_hash"] = rec_hash

        # Exactly-once gate: skip ONLY on proven duplicate.
        # If store errors, fail-open and write.
        try:
            store = IdempotencyStore(default_paper_db_path(self._repo_root))
            payload_hash = sha256_hex(canonical_json(payload))
            trade_id = str(payload.get("trade_id") or "")
            m = store.mark_once(trade_id, payload_hash, meta={"source": "paper_shadow_runner", "ledger": str(path)})
            if getattr(m, "reason", "") == "duplicate":
                return path, False
        except Exception:
            pass

        fsync_append_line(path, canonical_json(rec) + "\n")
        return path, True


# =============================================================================
# Operator intent gate (read-only)
# =============================================================================

def operator_mode(runtime_dir: Path) -> str:
    """
    Fail-closed operator mode read. Defaults to EXIT_ONLY if missing/corrupt.
    """
    p = runtime_dir / "operator_intent.json"
    if not p.exists():
        return "EXIT_ONLY"
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return "EXIT_ONLY"
        mode = str(obj.get("operator_mode", obj.get("mode", "EXIT_ONLY"))).upper().strip()
        if mode == "ALLOW":
            mode = "ALLOW_LIVE"
        if mode not in ("EXIT_ONLY", "ALLOW_LIVE", "DENY_ALL"):
            return "EXIT_ONLY"
        return mode
    except Exception:
        return "EXIT_ONLY"


# =============================================================================
# Plan load + intent selection
# =============================================================================

def load_plan(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def iter_ibkr_intents(plan: Dict[str, Any]) -> Iterable[PlannedIntent]:
    intents = plan.get("ibkr_intents")
    if isinstance(intents, list):
        for d in intents:
            if isinstance(d, dict):
                it = PlannedIntent.from_dict(d)
                if it.symbol:
                    yield it


def choose_one_buy_intent(plan: Dict[str, Any]) -> Optional[PlannedIntent]:
    """
    Deterministic intent selection:
    - Only BUY
    - Only positive qty
    - Prefer smallest notional estimate if present (safer), then alpha sort
    """
    candidates: List[Tuple[float, str, PlannedIntent]] = []
    for it in iter_ibkr_intents(plan):
        if it.side != "BUY":
            continue
        if it.quantity <= 0:
            continue
        # try to infer notional estimate from raw dict (if present)
        notional = 0.0
        try:
            raw_list = plan.get("ibkr_intents") or []
            for d in raw_list:
                if isinstance(d, dict) and str(d.get("symbol", "")).upper() == it.symbol and str(d.get("side", "")).upper() == "BUY":
                    notional = safe_float(d.get("notional_estimate"), 0.0)
                    break
        except Exception:
            notional = 0.0
        key_notional = notional if notional > 0 else 1e18
        candidates.append((key_notional, it.symbol, it))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]


def canary_fallback_intent(plan: Dict[str, Any], *, max_qty: float) -> Optional[PlannedIntent]:
    """
    Build a deterministic tiny BUY intent from summary.tick_symbols.
    Deterministic selection: first symbol in sorted tick_symbols.
    """
    try:
        summary = plan.get("summary")
        if not isinstance(summary, dict):
            return None
        ts = summary.get("tick_symbols")
        if not isinstance(ts, list):
            return None
        syms = [str(x).strip().upper() for x in ts if str(x).strip()]
        if not syms:
            return None
        symbol = sorted(syms)[0]
        qty = float(max(0.0, min(float(max_qty), 1.0e9)))
        if qty <= 0:
            qty = 1.0
        return PlannedIntent(
            strategy="paper_shadow_canary",
            symbol=symbol,
            side="BUY",
            quantity=float(qty),
            exchange="SMART",
            currency="USD",
            sec_type="STK",
        )
    except Exception:
        return None


# =============================================================================
# IBKR facade (lazy imports, bounded retries, contract-hours cache)
# =============================================================================

class IBKRClient:
    def __init__(self, host: str, port: int, client_id: int) -> None:
        self._host = host
        self._port = port
        self._client_id = client_id
        self.ib = None

    def connect(self, *, retries: int) -> None:
        from ib_insync import IB  # type: ignore

        last: Optional[Exception] = None
        for attempt in range(max(0, retries) + 1):
            ib = IB()
            try:
                ib.connect(self._host, int(self._port), clientId=int(self._client_id), timeout=10)
                if not ib.isConnected():
                    raise RuntimeError("ibkr_connected_false")
                self.ib = ib
                return
            except Exception as e:
                last = e
                try:
                    ib.disconnect()
                except Exception:
                    pass
                if attempt >= retries:
                    raise
                jitter_sleep(0.6 * (2 ** attempt), jitter_frac=0.25)
        if last:
            raise last

    def disconnect(self) -> None:
        try:
            if self.ib is not None:
                self.ib.disconnect()
        except Exception:
            pass
        self.ib = None


def _cache_path(runtime_dir: Path) -> Path:
    return runtime_dir / "calendar_state.json"


def _load_contract_cache(runtime_dir: Path) -> Dict[str, Any]:
    p = _cache_path(runtime_dir)
    if not p.exists():
        return {"ts_utc": "", "items": {}}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {"ts_utc": "", "items": {}}
    except Exception:
        return {"ts_utc": "", "items": {}}


def _save_contract_cache(runtime_dir: Path, obj: Dict[str, Any]) -> None:
    atomic_write_json(_cache_path(runtime_dir), obj)


def ibkr_is_liquid_open_now(
    ib: Any,
    *,
    runtime_dir: Path,
    symbol: str,
    exchange: str,
    currency: str,
    now: datetime,
    details_retries: int,
    cache_ttl_s: int,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Uses IBKR contractDetails to determine if liquidHours/tradingHours are open now.
    Cached in runtime/calendar_state.json (TTL).
    """
    dbg: Dict[str, Any] = {"symbol": symbol, "exchange": exchange, "currency": currency}
    key = f"{symbol}::{exchange}::{currency}"

    cache = _load_contract_cache(runtime_dir)
    items = cache.get("items") if isinstance(cache.get("items"), dict) else {}
    dbg["cache_keys"] = len(items) if isinstance(items, dict) else 0

    # Cache freshness check (per-key)
    if isinstance(items, dict) and key in items:
        it = items.get(key)
        if isinstance(it, dict):
            ts = str(it.get("fetched_ts_utc") or "")
            try:
                fetched = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
                age = (now.astimezone(timezone.utc) - fetched).total_seconds()
                if age <= float(cache_ttl_s):
                    # Use cached liquidHours string
                    lh = str(it.get("liquidHours") or "")
                    dbg["cache_hit"] = True
                    ok = _hours_open_now(liquid_hours=lh, now=now, dbg=dbg)
                    return ok, ("OPEN" if ok else "CLOSED"), dbg
            except Exception:
                pass

    dbg["cache_hit"] = False

    # Fetch contractDetails
    from ib_insync import Stock  # type: ignore

    contract = Stock(symbol, exchange, currency)
    last: Optional[Exception] = None
    details = None
    for attempt in range(max(0, details_retries) + 1):
        try:
            details_list = ib.reqContractDetails(contract)
            if not details_list:
                raise RuntimeError("contractDetails_empty")
            details = details_list[0]
            break
        except Exception as e:
            last = e
            if attempt >= details_retries:
                raise
            jitter_sleep(0.6 * (2 ** attempt), jitter_frac=0.25)
    if details is None and last:
        raise last

    # Extract hours
    liquid = ""
    trading = ""
    try:
        liquid = str(getattr(details, "liquidHours", "") or "")
        trading = str(getattr(details, "tradingHours", "") or "")
    except Exception:
        liquid = ""
        trading = ""

    dbg["liquidHours"] = liquid
    dbg["tradingHours"] = trading

    # Update cache
    try:
        if not isinstance(items, dict):
            items = {}
        items[key] = {
            "fetched_ts_utc": iso_utc(now),
            "liquidHours": liquid,
            "tradingHours": trading,
        }
        cache["items"] = items
        cache["ts_utc"] = iso_utc(now)
        _save_contract_cache(runtime_dir, cache)
    except Exception:
        pass

    ok = _hours_open_now(liquid_hours=liquid or trading, now=now, dbg=dbg)
    return ok, ("OPEN" if ok else "CLOSED"), dbg


def _hours_open_now(*, liquid_hours: str, now: datetime, dbg: Dict[str, Any]) -> bool:
    """
    IBKR hours format: "YYYYMMDD:HHMM-YYYYMMDD:HHMM;YYYYMMDD:CLOSED;..."
    We treat any range that covers now as open.
    """
    s = str(liquid_hours or "").strip()
    dbg["hours_len"] = len(s)

    if not s or s.upper() == "CLOSED":
        return False

    now_utc = now.astimezone(timezone.utc)
    ymd = now_utc.strftime("%Y%m%d")
    # Find the segment for today
    seg = None
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        if part.startswith(ymd + ":"):
            seg = part
            break

    if not seg:
        # If no today segment, fail closed
        dbg["hours_today_segment"] = None
        return False

    dbg["hours_today_segment"] = seg
    rhs = seg.split(":", 1)[1].strip()
    if rhs.upper() == "CLOSED":
        return False

    # Can have multiple windows separated by ","
    for win in rhs.split(","):
        win = win.strip()
        if not win:
            continue
        if "-" not in win:
            continue
        a, b = win.split("-", 1)
        a = a.strip()
        b = b.strip()
        # Sometimes IBKR returns HHMM-HHMM (same day). Sometimes full ymd:HHMM.
        try:
            start = _parse_hours_point(ymd, a)
            end = _parse_hours_point(ymd, b)
            if start <= now_utc <= end:
                return True
        except Exception:
            continue

    return False


def _parse_hours_point(today_ymd: str, token: str) -> datetime:
    t = token.strip()
    if ":" in t:
        ymd, hhmm = t.split(":", 1)
        ymd = ymd.strip()
        hhmm = hhmm.strip()
    else:
        ymd, hhmm = today_ymd, t
    hhmm = hhmm.replace(" ", "")
    hh = int(hhmm[:2])
    mm = int(hhmm[2:4])
    return datetime(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8]), hh, mm, tzinfo=timezone.utc)


# =============================================================================
# IBKR order helpers (paper only)
# =============================================================================

def place_mkt_order(ib: Any, *, symbol: str, exchange: str, currency: str, side: str, qty: float) -> Any:
    from ib_insync import Stock, MarketOrder  # type: ignore

    c = Stock(symbol, exchange, currency)
    order = MarketOrder(side.upper(), float(qty))
    trade = ib.placeOrder(c, order)
    return trade


def wait_fill_or_cancel(ib: Any, trade: Any, *, timeout_s: float) -> Tuple[bool, float, str, float]:
    """
    Wait for trade fill; cancel at timeout.
    Returns: (filled?, filled_qty, status, avg_fill_price)
    """
    t0 = time.time()
    filled_qty = 0.0
    avg_price = 0.0
    status = ""

    while True:
        status = str(getattr(trade, "orderStatus", None).status if getattr(trade, "orderStatus", None) else "")
        filled_qty = safe_float(getattr(trade, "orderStatus", None).filled if getattr(trade, "orderStatus", None) else 0.0, 0.0)
        avg_price = safe_float(getattr(trade, "orderStatus", None).avgFillPrice if getattr(trade, "orderStatus", None) else 0.0, 0.0)

        if status.upper() == "FILLED" and filled_qty > 0.0:
            return True, float(filled_qty), status, float(avg_price)

        if time.time() - t0 >= float(timeout_s):
            try:
                ib.cancelOrder(trade.order)
            except Exception:
                pass
            return False, float(filled_qty), status or "TIMEOUT", float(avg_price)

        time.sleep(0.25)


# =============================================================================
# Report model
# =============================================================================

@dataclass
class RunReport:
    ts_utc: str
    safety: Dict[str, Any]
    actions: Dict[str, Any]
    notes: List[Dict[str, Any]]


# =============================================================================
# Runner
# =============================================================================

class PaperShadowRunner:
    def __init__(self, cfg: RunnerConfig, *, ledger: TradeLedger) -> None:
        self._cfg = cfg
        self._ledger = ledger

    def run_once(self) -> RunReport:
        now = utc_now()

        report = RunReport(
            ts_utc=iso_utc(now),
            safety={},
            actions={
                "orders_submitted": 0,
                "orders_cancelled": 0,
                "trade_results_logged": 0,
                "ledger_path": None,
            },
            notes=[],
        )

        # Preview is absolute: no connect, no orders, no ledger.
        if self._cfg.preview or not self._cfg.execute:
            report.safety.update({"mode": "PREVIEW", "no_orders": True, "no_ledger": True})
            report.notes.append({"event": "preview_only", "reason": "missing_execute_flag_or_preview"})
            return report

        # Operator gate
        op = operator_mode(self._cfg.runtime_dir)
        report.safety["operator_mode"] = op
        if op == "DENY_ALL":
            report.safety.update({"blocked": True, "no_orders": True, "no_ledger": True})
            report.notes.append({"event": "blocked", "reason": "operator_deny_all"})
            return report
        if op == "EXIT_ONLY":
            report.safety.update({"blocked": True, "no_orders": True, "no_ledger": True})
            report.notes.append({"event": "blocked", "reason": "operator_exit_only_blocks_entries"})
            return report

        # Plan -> BUY intent selection
        planned = load_plan(self._cfg.planned_path)
        it = choose_one_buy_intent(planned)

        if it is None and self._cfg.canary_fallback:
            it = canary_fallback_intent(planned, max_qty=self._cfg.canary_max_qty)
            if it is not None:
                report.safety["canary_fallback"] = True
                report.notes.append({"event": "canary_fallback", "reason": "no_valid_buy_intent", "symbol": it.symbol})

        if it is None:
            report.safety.update({"no_plan": True, "no_orders": True, "no_ledger": True})
            report.notes.append({"event": "no_plan", "reason": "no_valid_buy_intent"})
            return report

        qty = float(it.quantity) * float(self._cfg.size_multiplier)
        if qty <= 0.0:
            report.safety.update({"blocked": True, "no_orders": True, "no_ledger": True})
            report.notes.append({"event": "blocked", "reason": "qty<=0_after_multiplier"})
            return report

        # Connect + IBKR-truth market gate (holiday/session)
        ibc = IBKRClient(self._cfg.ib_host, self._cfg.ib_port, self._cfg.ib_client_id)
        try:
            ibc.connect(retries=self._cfg.connect_retries)
            report.notes.append({"event": "ib_connect_ok"})

            ok, why, dbg = ibkr_is_liquid_open_now(
                ibc.ib,
                runtime_dir=self._cfg.runtime_dir,
                symbol=it.symbol,
                exchange=it.exchange,
                currency=it.currency,
                now=now,
                details_retries=self._cfg.details_retries,
                cache_ttl_s=self._cfg.contract_cache_ttl_seconds,
            )
            report.safety["ibkr_market_open"] = bool(ok)
            report.safety["ibkr_market_reason"] = why
            report.safety["ibkr_market_debug"] = dbg

            if not ok:
                report.safety.update({"blocked": True, "no_orders": True, "no_ledger": True})
                report.notes.append({"event": "blocked", "reason": "market_closed"})
                return report

            # ENTRY
            entry_ts = utc_now()
            entry_trade = place_mkt_order(
                ibc.ib,
                symbol=it.symbol,
                exchange=it.exchange,
                currency=it.currency,
                side="BUY",
                qty=float(qty),
            )
            report.actions["orders_submitted"] += 1
            report.notes.append({"event": "order_submitted", "side": "BUY", "symbol": it.symbol, "qty": float(qty)})

            entry_filled, entry_filled_qty, entry_status, entry_fp = wait_fill_or_cancel(
                ibc.ib, entry_trade, timeout_s=self._cfg.auto_cancel_seconds
            )
            if not entry_filled:
                report.actions["orders_cancelled"] += 1
                report.safety["no_ledger"] = True
                report.notes.append({"event": "entry_not_filled", "status": entry_status, "filled_qty": float(entry_filled_qty), "entry_fp": float(entry_fp)})
                return report

            if entry_filled_qty <= 0.0 or entry_fp <= 0.0:
                report.safety["no_ledger"] = True
                report.notes.append({"event": "entry_bad_fill", "filled_qty": float(entry_filled_qty), "entry_fp": float(entry_fp)})
                return report

            # HOLD
            time.sleep(int(self._cfg.hold_seconds))

            # EXIT
            exit_ts = utc_now()
            exit_trade = place_mkt_order(
                ibc.ib,
                symbol=it.symbol,
                exchange=it.exchange,
                currency=it.currency,
                side="SELL",
                qty=float(entry_filled_qty),
            )
            report.actions["orders_submitted"] += 1
            report.notes.append({"event": "order_submitted", "side": "SELL", "symbol": it.symbol, "qty": float(entry_filled_qty)})

            exit_filled, exit_filled_qty, exit_status, exit_fp = wait_fill_or_cancel(
                ibc.ib, exit_trade, timeout_s=self._cfg.auto_cancel_seconds
            )
            if not exit_filled:
                report.actions["orders_cancelled"] += 1
                report.safety["no_ledger"] = True
                report.notes.append({"event": "exit_not_filled", "status": exit_status, "filled_qty": float(exit_filled_qty), "exit_fp": float(exit_fp)})
                return report

            if exit_filled_qty <= 0.0 or exit_fp <= 0.0:
                report.safety["no_ledger"] = True
                report.notes.append({"event": "exit_bad_fill", "filled_qty": float(exit_filled_qty), "exit_fp": float(exit_fp)})
                return report

            realized = (float(exit_fp) - float(entry_fp)) * float(exit_filled_qty)

            tr = TradeResult(
                strategy=it.strategy,
                symbol=it.symbol,
                quantity=float(exit_filled_qty),
                entry_fill_price=float(entry_fp),
                exit_fill_price=float(exit_fp),
                realized_pnl=float(realized),
                ts_entry_utc=iso_utc(entry_ts),
                ts_exit_utc=iso_utc(exit_ts),
                broker="ibkr",
                is_live=False,
                order_id_entry=safe_int(getattr(getattr(entry_trade, "order", None), "orderId", None), 0) or None,
                order_id_exit=safe_int(getattr(getattr(exit_trade, "order", None), "orderId", None), 0) or None,
                perm_id_entry=safe_int(getattr(getattr(entry_trade, "order", None), "permId", None), 0) or None,
                perm_id_exit=safe_int(getattr(getattr(exit_trade, "order", None), "permId", None), 0) or None,
                extra={
                    "exchange": it.exchange,
                    "currency": it.currency,
                    "sec_type": it.sec_type,
                    "source": "paper_shadow_runner",
                    "planned_path": str(self._cfg.planned_path),
                    "hold_seconds": int(self._cfg.hold_seconds),
                    "canary_fallback": bool(report.safety.get("canary_fallback", False)),
                },
            )

            ledger_path, wrote = self._ledger.append_trade_result(tr)
            report.actions["trade_results_logged"] += 1 if wrote else 0
            report.actions["ledger_path"] = str(ledger_path)
            report.notes.append({"event": "trade_result_logged", "trade_id": tr.trade_id(), "realized_pnl": float(realized), "ledger_path": str(ledger_path), "wrote": bool(wrote)})

            return report

        finally:
            ibc.disconnect()


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CHAD Paper Shadow Runner (production-grade)")
    ap.add_argument("--preview", action="store_true", help="Preview only: no connect, no orders, no ledger.")
    ap.add_argument("--execute", action="store_true", help="Execute one paper round-trip if fully allowed.")
    ap.add_argument("--hold-seconds", type=int, default=5, help="Hold time between entry and exit.")
    ap.add_argument("--auto-cancel-seconds", type=float, default=30.0, help="Cancel unfilled orders after timeout.")
    ap.add_argument("--size-multiplier", type=float, default=1.0, help="Multiplier applied to planned quantity.")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = build_parser().parse_args(argv)
    cfg = RunnerConfig.from_env_args(args)
    ledger = TradeLedger(trades_dir=cfg.trades_dir, repo_root=cfg.repo_root)
    runner = PaperShadowRunner(cfg, ledger=ledger)

    report = runner.run_once()

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    rp = cfg.reports_dir / f"PAPER_SHADOW_RUN_{utc_now().strftime('%Y%m%dT%H%M%SZ')}.json"
    atomic_write_json(rp, dataclasses.asdict(report))

    # Stable stdout for systemd/wrapper parsing
    print(json.dumps({"report_path": str(rp), "actions": report.actions, "safety": report.safety}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
