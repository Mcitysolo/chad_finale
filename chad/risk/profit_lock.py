"""
An advanced, production‑grade profit‑lock engine for the CHAD automated trading system.

This module analyses realised profits and account equity to determine when to
reduce risk, lock in profits or stop trading for the day. It is designed with
professional engineering principles: modularity, clear separation of concerns,
asynchronous I/O, dependency injection and extensibility via strategy/factory
patterns.  The engine reads trade ledgers (as NDJSON files) and account
equity from configurable sources, computes profit ratios and produces a
state object with sizing factors and flags for upstream risk managers.  The
state is written atomically to JSON to ensure downstream consumers always see
either the old or the new state, never a partial write.

Key features:

* **Asynchronous I/O**: File reads and PnL aggregation run concurrently using
  `asyncio` and `aiofiles` to minimise blocking and improve performance on
  systems with many trade ledger files.  The engine can therefore scale
  gracefully as the number of files grows.

* **Dependency Injection**: Equity and PnL providers are pluggable via
  abstract base classes.  New providers can be added without modifying the
  core engine (e.g. reading equity from a REST API or a database).

* **Strategy Pattern**: The profit‑lock decision logic is encapsulated in
  dedicated methods and uses a finite set of modes represented by an
  enumeration.  Thresholds and sizing factors are read from environment
  variables or defaults, making behaviour easy to tune.

* **Factory for Providers**: A simple factory function creates composite
  equity providers from a list of possible sources.  If the first source
  fails to produce a valid equity value, the next one is tried.

* **Extensive Error Handling**: All file and JSON operations are wrapped in
  try/except blocks.  Numeric conversions are hardened against NaNs and
  infinities.  The engine fails closed by treating unknown equity as
  ineligible for profit locking unless explicitly allowed.

* **Atomic Writes**: The JSON state is written via a temporary file and a
  rename to guarantee atomicity, preventing readers from seeing truncated
  output.

* **Command‑Line Interface**: Users can run the engine from the command line
  to generate the current profit‑lock state.  Options allow overriding the
  output path, selecting equity providers and specifying how many days back
  to look for trade ledgers.

Example usage:

```bash
python improved_profit_lock.py --repo /home/ubuntu/chad_finale --out /home/ubuntu/chad_finale/runtime/profit_lock_state.json
```

Environment variables (all optional):

* `CHAD_PROFIT_LOCK_TTL_SECONDS`: expiry time in seconds for the state file.
* `CHAD_PROFIT_LOCK_WARN_PCT`: warning threshold (%) of equity for profit lock.
* `CHAD_PROFIT_LOCK_1_PCT`, `CHAD_PROFIT_LOCK_2_PCT`, `CHAD_PROFIT_LOCK_3_PCT`,
  `CHAD_PROFIT_LOCK_HARD_STOP_PCT`: percentage thresholds for increasing
  severity profit locks.
* `CHAD_PROFIT_LOCK_1_FACTOR`, `CHAD_PROFIT_LOCK_2_FACTOR`,
  `CHAD_PROFIT_LOCK_3_FACTOR`, `CHAD_PROFIT_LOCK_HARD_STOP_FACTOR`: sizing
  factors corresponding to the above thresholds.
* `CHAD_PROFIT_LOCK_ENABLE_ON_UNKNOWN_EQUITY`: set to a truthy value to
  activate profit locking even when equity cannot be determined.

This module does not depend on any part of the existing CHAD codebase and can
be integrated into other systems.  It only requires a POSIX filesystem,
Python 3.8+ (for `asyncio` and type hints) and the `aiofiles` package for
async file I/O.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum, auto
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import aiofiles  # type: ignore
except ImportError:
    aiofiles = None  # fallback to synchronous I/O if aiofiles isn't available

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Return the current UTC datetime without microseconds."""
    return datetime.now(UTC).replace(microsecond=0)


def _today_yyyymmdd() -> str:
    """Return today's date in YYYYMMDD format (UTC)."""
    return _utc_now().strftime("%Y%m%d")


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float safely, guarding against NaN and infinities."""
    try:
        if value is None or isinstance(value, bool):
            return default
        out = float(value)  # type: ignore[arg-type]
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


class ProfitLockMode(Enum):
    """
    Discrete profit‑lock states.

    The engine escalates through these modes as realised profit relative to
    equity crosses configured thresholds.  In the `HARD_STOP` mode the system
    should cease opening new positions entirely.
    """

    NORMAL = auto()
    WARN = auto()
    LOCK1 = auto()
    LOCK2 = auto()
    LOCK3 = auto()
    HARD_STOP = auto()
    INACTIVE_EQUITY_UNKNOWN = auto()


@dataclass(frozen=True)
class ProfitLockDecision:
    """Outcome of the profit‑lock decision logic."""

    mode: ProfitLockMode
    sizing_factor: float
    stop_new_entries: bool
    explain: str


@dataclass(frozen=True)
class ProfitLockConfig:
    """
    Configuration for profit locking.

    Threshold percentages and sizing factors correspond 1:1.  When the
    realised PnL exceeds a threshold (expressed as a percentage of equity), the
    corresponding sizing factor limits position sizes.  A hard stop halts new
    entries entirely.  TTL specifies how long the generated state file is
    considered valid.
    """

    ttl_seconds: int = 60
    warn_profit_pct: float = 1.5
    lock1_profit_pct: float = 3.0
    lock2_profit_pct: float = 5.0
    lock3_profit_pct: float = 8.0
    hard_stop_profit_pct: float = 10.0
    lock1_sizing_factor: float = 0.50
    lock2_sizing_factor: float = 0.25
    lock3_sizing_factor: float = 0.10
    hard_stop_sizing_factor: float = 0.00
    enable_on_negative_equity_unknown: bool = False
    # Daily loss limit: halt new entries if realized PnL today drops below
    # -daily_loss_limit_pct% of equity. Additive to profit-lock tiers above.
    daily_loss_limit_pct: float = 3.0

    @classmethod
    def from_env(cls) -> "ProfitLockConfig":
        """Construct a configuration from environment variables."""
        def getenv_float(name: str, default: float) -> float:
            raw = os.getenv(name)
            return _safe_float(raw, default) if raw is not None else default

        def getenv_bool(name: str, default: bool = False) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            return str(raw).strip().lower() in {"1", "true", "yes", "on"}

        return cls(
            ttl_seconds=int(os.getenv("CHAD_PROFIT_LOCK_TTL_SECONDS", cls.ttl_seconds)),
            warn_profit_pct=getenv_float("CHAD_PROFIT_LOCK_WARN_PCT", cls.warn_profit_pct),
            lock1_profit_pct=getenv_float("CHAD_PROFIT_LOCK_1_PCT", cls.lock1_profit_pct),
            lock2_profit_pct=getenv_float("CHAD_PROFIT_LOCK_2_PCT", cls.lock2_profit_pct),
            lock3_profit_pct=getenv_float("CHAD_PROFIT_LOCK_3_PCT", cls.lock3_profit_pct),
            hard_stop_profit_pct=getenv_float("CHAD_PROFIT_LOCK_HARD_STOP_PCT", cls.hard_stop_profit_pct),
            lock1_sizing_factor=getenv_float("CHAD_PROFIT_LOCK_1_FACTOR", cls.lock1_sizing_factor),
            lock2_sizing_factor=getenv_float("CHAD_PROFIT_LOCK_2_FACTOR", cls.lock2_sizing_factor),
            lock3_sizing_factor=getenv_float("CHAD_PROFIT_LOCK_3_FACTOR", cls.lock3_sizing_factor),
            hard_stop_sizing_factor=getenv_float("CHAD_PROFIT_LOCK_HARD_STOP_FACTOR", cls.hard_stop_sizing_factor),
            enable_on_negative_equity_unknown=getenv_bool(
                "CHAD_PROFIT_LOCK_ENABLE_ON_UNKNOWN_EQUITY",
                cls.enable_on_negative_equity_unknown,
            ),
            daily_loss_limit_pct=getenv_float(
                "CHAD_DAILY_LOSS_LIMIT_PCT", cls.daily_loss_limit_pct
            ),
        )


class EquityProvider:
    """Abstract base class for account equity providers."""

    async def get_equity(self, repo_root: Path) -> Tuple[Optional[float], str]:
        """Return account equity and its source path, or (None, source) if unknown."""
        raise NotImplementedError


class DynamicCapsEquityProvider(EquityProvider):
    """
    Read equity from the CHAD dynamic caps JSON file.

    This provider expects a file named `dynamic_caps.json` in the runtime
    directory under the repo root.  The file must contain a `total_equity`
    field with a positive numeric value.
    """

    def __init__(self, filename: str = "dynamic_caps.json", key: str = "total_equity") -> None:
        self.filename = filename
        self.key = key

    async def get_equity(self, repo_root: Path) -> Tuple[Optional[float], str]:
        runtime = repo_root / "runtime"
        path = runtime / self.filename
        if not path.exists():
            logger.debug("DynamicCapsEquityProvider: %s does not exist", path)
            return None, str(path)
        try:
            # read synchronously off main thread to avoid blocking event loop
            content = await asyncio.to_thread(path.read_text, encoding="utf-8")
            data = json.loads(content)
        except Exception as exc:
            logger.warning("DynamicCapsEquityProvider: failed to read %s: %s", path, exc)
            return None, str(path)
        value = _safe_float(data.get(self.key), default=-1.0)
        if value > 0.0:
            return value, str(path)
        return None, str(path)


class FileEquityProvider(EquityProvider):
    """
    Generic equity provider reading from a JSON file and searching for candidate keys.

    This provider can be configured to read from any JSON file (e.g.
    `positions_snapshot.json` or `portfolio_state.json`) and will attempt to
    extract the first positive value from a list of candidate keys.  It can be
    used as a fallback when the primary provider fails.
    """

    def __init__(self, filename: str, keys: Sequence[str]) -> None:
        self.filename = filename
        self.keys = list(keys)

    async def get_equity(self, repo_root: Path) -> Tuple[Optional[float], str]:
        path = repo_root / "runtime" / self.filename
        if not path.exists():
            logger.debug("FileEquityProvider: %s does not exist", path)
            return None, str(path)
        try:
            content = await asyncio.to_thread(path.read_text, encoding="utf-8")
            data = json.loads(content)
        except Exception as exc:
            logger.warning("FileEquityProvider: failed to read %s: %s", path, exc)
            return None, str(path)
        # search top level first
        for key in self.keys:
            if key in data:
                val = _safe_float(data.get(key), default=-1.0)
                if val > 0.0:
                    return val, str(path)
        # search nested common structures
        for container_key in ("account", "balances"):
            container = data.get(container_key)
            if isinstance(container, Mapping):
                for key in self.keys:
                    if key in container:
                        val = _safe_float(container.get(key), default=-1.0)
                        if val > 0.0:
                            return val, str(path)
        return None, str(path)


class CompositeEquityProvider(EquityProvider):
    """
    Try multiple equity providers in sequence.

    The first provider returning a positive equity value is used.  If none
    succeed, the last provider's source path is returned with None for the
    equity.
    """

    def __init__(self, providers: Sequence[EquityProvider]) -> None:
        self.providers = list(providers)

    async def get_equity(self, repo_root: Path) -> Tuple[Optional[float], str]:
        last_source = ""
        for provider in self.providers:
            value, source = await provider.get_equity(repo_root)
            last_source = source
            if value is not None and value > 0.0:
                return value, source
        return None, last_source


class PnlProvider:
    """Abstract base class for realised PnL providers."""

    async def get_realized_pnl(self, repo_root: Path, days: int = 0) -> Tuple[float, int, List[str]]:
        """Return the sum of realised PnL, count of trades and list of file sources."""
        raise NotImplementedError


class NdjsonPnlProvider(PnlProvider):
    """
    Compute realised PnL from NDJSON trade ledgers.

    By default this provider looks for files matching `*YYYYMMDD*.ndjson` in
    the `data/trades` directory under the repo root.  The `days` argument
    allows scanning multiple days back (0 means only today).  All files for
    each day in the range are processed concurrently to improve throughput.
    """

    def __init__(self) -> None:
        # keys for realised pnl extraction; order matters for speed
        self.pnl_keys: Sequence[str] = (
            "realized_pnl",
            "realizedPnl",
            "pnl_realized",
            "closed_pnl",
            "closedPnl",
            "net_pnl",
            "netPnl",
            "pnl",
            "profit_loss",
            "profitLoss",
        )
        # keys that disable counting a trade as realised (paper trades, previews)
        self.preview_keys: Sequence[str] = (
            "paper",
            "is_paper",
            "what_if",
            "preview_only",
        )

    async def get_realized_pnl(self, repo_root: Path, days: int = 0) -> Tuple[float, int, List[str]]:
        trades_dir = repo_root / "data" / "trades"
        fills_dir = repo_root / "data" / "fills"
        # Quarantine awareness: load invalid IDs once per call as the
        # union of runtime/quarantine_manifest_*.json and a live scan
        # of data/fills/FILLS_*.ndjson rows tagged pnl_untrusted, so
        # derived closed trades that reference an untrusted opening
        # or closing fill (even when the closed-trade row itself
        # carries no marker) do not re-enter pnl_state.
        try:
            from chad.utils.quarantine import get_exclusion_sets
            invalid_fill_ids, invalid_trade_hashes = get_exclusion_sets(
                runtime_dir=repo_root / "runtime",
                fills_dir=fills_dir,
            )
        except Exception:
            invalid_fill_ids, invalid_trade_hashes = set(), set()
        # compute date strings to search
        day_strings = []
        today = _utc_now().date()
        for offset in range(days + 1):
            dt = today - timedelta(days=offset)
            day_strings.append(dt.strftime("%Y%m%d"))
        # gather files from both data/trades/ and data/fills/
        files: List[Path] = []
        for scan_dir in (trades_dir, fills_dir):
            if not scan_dir.exists():
                continue
            for ds in day_strings:
                pattern = f"*{ds}*.ndjson"
                files.extend(sorted(scan_dir.glob(pattern)))
        if not files:
            return 0.0, 0, []
        # process files concurrently
        tasks = [
            asyncio.create_task(
                self._process_file(fp, invalid_fill_ids, invalid_trade_hashes)
            )
            for fp in files
        ]
        totals: List[Tuple[float, int, bool]] = await asyncio.gather(*tasks)
        total_pnl = 0.0
        total_count = 0
        used_sources: List[str] = []
        for pnl, count, used in totals:
            total_pnl += pnl
            total_count += count
            if used:
                used_sources.append(str(files[totals.index((pnl, count, used))]))
        return round(total_pnl, 8), total_count, used_sources

    async def _process_file(
        self,
        path: Path,
        invalid_fill_ids: Optional[set] = None,
        invalid_trade_hashes: Optional[set] = None,
    ) -> Tuple[float, int, bool]:
        """Return (pnl, trade_count, file_used)."""
        pnl_sum = 0.0
        trade_count = 0
        file_used = False
        bad_fills = invalid_fill_ids or set()
        bad_hashes = invalid_trade_hashes or set()
        try:
            if aiofiles:
                async with aiofiles.open(path, mode="r", encoding="utf-8") as handle:  # type: ignore[attr-defined]
                    async for raw in handle:
                        rec = self._parse_line(raw)
                        if rec is None:
                            continue
                        if self._is_quarantined(rec, bad_fills, bad_hashes):
                            continue
                        if not self._is_effective_trade(rec):
                            continue
                        pnl_value = self._extract_pnl(rec)
                        if pnl_value is not None:
                            pnl_sum += pnl_value
                            trade_count += 1
                            file_used = True
            else:
                # fallback synchronous reader
                def read_sync() -> None:
                    nonlocal pnl_sum, trade_count, file_used
                    for raw in path.open("r", encoding="utf-8"):
                        rec = self._parse_line(raw)
                        if rec is None:
                            continue
                        if self._is_quarantined(rec, bad_fills, bad_hashes):
                            continue
                        if not self._is_effective_trade(rec):
                            continue
                        pnl_value = self._extract_pnl(rec)
                        if pnl_value is not None:
                            pnl_sum += pnl_value
                            trade_count += 1
                            file_used = True
                await asyncio.to_thread(read_sync)
        except Exception as exc:
            logger.warning("NdjsonPnlProvider: failed to read %s: %s", path, exc)
        return pnl_sum, trade_count, file_used

    @staticmethod
    def _is_quarantined(
        record: Mapping[str, Any],
        invalid_fill_ids: set,
        invalid_trade_hashes: set,
    ) -> bool:
        """Skip records listed in runtime/quarantine_manifest_*.json
        or referenced via the untrusted-fill set built from
        data/fills/FILLS_*.ndjson. Closed-trade records expose their
        opening + closing fill IDs in ``payload.fill_ids``; any
        intersection with *invalid_fill_ids* drops the trade so the
        derived realized PnL never reaches pnl_state."""
        rh = record.get("record_hash")
        if isinstance(rh, str) and rh in invalid_trade_hashes:
            return True
        payload = record.get("payload")
        if isinstance(payload, Mapping):
            fid = payload.get("fill_id")
            if isinstance(fid, str) and fid in invalid_fill_ids:
                return True
            fids = payload.get("fill_ids")
            if isinstance(fids, list) and any(
                isinstance(f, str) and f in invalid_fill_ids for f in fids
            ):
                return True
        return False

    def _parse_line(self, raw: str) -> Optional[Mapping[str, Any]]:
        """Parse a JSON line, ignoring invalid entries."""
        line = raw.strip()
        if not line:
            return None
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, Mapping):
            return None
        return obj

    def _is_effective_trade(self, record: Mapping[str, Any]) -> bool:
        """
        Determine if a record represents a real trade with realised PnL.

        We skip paper trades, previews and other non‑executed activities.  The
        logic checks both top‑level flags and nested payload structures.
        """
        payload = record.get("payload")
        def is_preview(container: Mapping[str, Any]) -> bool:
            for key in self.preview_keys:
                if container.get(key) is True:
                    return True
            if container.get("pnl_untrusted") is True:
                return True
            status = str(container.get("status") or container.get("trade_status") or "").strip().lower()
            if status in {"preview_only", "paper_only", "what_if"}:
                return True
            return False
        # check top level
        if is_preview(record):
            return False
        # check nested payload
        if isinstance(payload, Mapping) and is_preview(payload):
            return False
        return True

    def _extract_pnl(self, record: Mapping[str, Any]) -> Optional[float]:
        """
        Extract realised PnL from a record.

        The function searches multiple locations: top level, nested `payload`
        structure and common result substructures like `result`, `execution`,
        `performance`, `ledger` or `trade`.  The first matching field with a
        numeric value is returned.
        """
        def search_container(container: Mapping[str, Any]) -> Optional[float]:
            for key in self.pnl_keys:
                if key in container:
                    val = _safe_float(container.get(key), default=0.0)
                    return val
            return None
        # search top level
        pnl = search_container(record)
        if pnl is not None:
            return pnl
        payload = record.get("payload")
        if isinstance(payload, Mapping):
            pnl = search_container(payload)
            if pnl is not None:
                return pnl
        # search common nested keys
        for nested_key in ("result", "trade_result", "execution", "performance", "ledger", "trade"):
            nested = None
            if isinstance(payload, Mapping):
                nested = payload.get(nested_key)
            elif nested_key in record:
                nested = record.get(nested_key)
            if isinstance(nested, Mapping):
                pnl = search_container(nested)
                if pnl is not None:
                    return pnl
        # fill records from data/fills/ have fill_price but no explicit pnl —
        # count them as executed trades with zero realised pnl (entry, not exit)
        fp_container = payload if isinstance(payload, Mapping) else record
        if "fill_price" in fp_container and _safe_float(fp_container.get("fill_price"), default=0.0) > 0.0:
            return 0.0
        return None


class ProfitLockEngine:
    """
    Core engine coordinating PnL aggregation, equity retrieval and lock decisions.
    """

    def __init__(
        self,
        repo_root: Path,
        equity_provider: EquityProvider,
        pnl_provider: PnlProvider,
        config: ProfitLockConfig,
    ) -> None:
        self.repo_root = repo_root
        self.equity_provider = equity_provider
        self.pnl_provider = pnl_provider
        self.config = config

    async def compute_decision(self, days: int = 0) -> Tuple[ProfitLockDecision, dict[str, Any]]:
        """
        Compute the profit‑lock decision and assemble diagnostic inputs.

        Returns a tuple of (decision, inputs) where `inputs` contains
        intermediate values such as realised PnL, trade count, account equity and
        percentage gain.
        """
        pnl_realized, trade_count, pnl_sources = await self.pnl_provider.get_realized_pnl(
            self.repo_root, days
        )
        equity, equity_source = await self.equity_provider.get_equity(self.repo_root)
        equity_known = equity is not None and equity > 0.0
        pnl_pct = 0.0
        if equity_known and equity:
            pnl_pct = round((pnl_realized / equity) * 100.0, 8)
        decision = self._decide(pnl_pct, equity_known)
        inputs = {
            "realized_pnl": float(pnl_realized),
            "trade_count": int(trade_count),
            "pnl_sources": pnl_sources,
            "account_equity": float(equity) if equity_known else None,
            "equity_source": equity_source,
            "equity_known": equity_known,
            "pnl_pct_of_equity": float(pnl_pct),
        }
        return decision, inputs

    def _decide(self, pnl_pct: float, equity_known: bool) -> ProfitLockDecision:
        """
        Determine the profit‑lock mode based on realised profit percentage and equity state.

        Follows a strict order from most severe (hard stop) down to no action.  If
        equity is unknown and `enable_on_negative_equity_unknown` is False, the
        decision will be `INACTIVE_EQUITY_UNKNOWN` regardless of PnL.
        """
        cfg = self.config
        if not equity_known and not cfg.enable_on_negative_equity_unknown:
            return ProfitLockDecision(
                mode=ProfitLockMode.INACTIVE_EQUITY_UNKNOWN,
                sizing_factor=1.0,
                stop_new_entries=False,
                explain="account equity unavailable; profit lock inactive",
            )
        if pnl_pct >= cfg.hard_stop_profit_pct:
            return ProfitLockDecision(
                mode=ProfitLockMode.HARD_STOP,
                sizing_factor=cfg.hard_stop_sizing_factor,
                stop_new_entries=True,
                explain="hard profit stop threshold reached",
            )
        if pnl_pct >= cfg.lock3_profit_pct:
            return ProfitLockDecision(
                mode=ProfitLockMode.LOCK3,
                sizing_factor=cfg.lock3_sizing_factor,
                stop_new_entries=False,
                explain="profit lock level 3 engaged",
            )
        if pnl_pct >= cfg.lock2_profit_pct:
            return ProfitLockDecision(
                mode=ProfitLockMode.LOCK2,
                sizing_factor=cfg.lock2_sizing_factor,
                stop_new_entries=False,
                explain="profit lock level 2 engaged",
            )
        if pnl_pct >= cfg.lock1_profit_pct:
            return ProfitLockDecision(
                mode=ProfitLockMode.LOCK1,
                sizing_factor=cfg.lock1_sizing_factor,
                stop_new_entries=False,
                explain="profit lock level 1 engaged",
            )
        if pnl_pct >= cfg.warn_profit_pct:
            return ProfitLockDecision(
                mode=ProfitLockMode.WARN,
                sizing_factor=1.0,
                stop_new_entries=False,
                explain="profit warning threshold reached",
            )
        return ProfitLockDecision(
            mode=ProfitLockMode.NORMAL,
            sizing_factor=1.0,
            stop_new_entries=False,
            explain="profit lock inactive",
        )

    async def build_state(self, days: int = 0) -> Mapping[str, Any]:
        """Assemble the complete profit‑lock state object."""
        decision, inputs = await self.compute_decision(days=days)

        realized_today = float(inputs.get("realized_pnl", 0.0))
        equity = inputs.get("account_equity")
        equity_known = bool(inputs.get("equity_known", False))
        limit_pct = float(self.config.daily_loss_limit_pct)
        daily_loss_today = realized_today if realized_today < 0.0 else 0.0
        loss_limit_dollars = (
            float(equity) * (limit_pct / 100.0)
            if equity_known and isinstance(equity, (int, float))
            else None
        )
        daily_loss_limit_hit = False
        if loss_limit_dollars is not None and realized_today <= -loss_limit_dollars:
            daily_loss_limit_hit = True
            logger.warning(
                "DAILY_LOSS_LIMIT: realized_today=$%.2f limit=-$%.2f",
                realized_today,
                loss_limit_dollars,
            )

        final_stop_new_entries = bool(decision.stop_new_entries) or daily_loss_limit_hit
        explain = decision.explain
        if daily_loss_limit_hit:
            explain = f"{explain} | daily_loss_limit_hit"

        state = {
            "ts_utc": _utc_now().isoformat().replace("+00:00", "Z"),
            "ttl_seconds": int(self.config.ttl_seconds),
            "mode": decision.mode.name,
            "sizing_factor": float(decision.sizing_factor),
            "stop_new_entries": final_stop_new_entries,
            "profit_lock_active": bool(
                decision.mode
                not in {
                    ProfitLockMode.NORMAL,
                    ProfitLockMode.WARN,
                    ProfitLockMode.INACTIVE_EQUITY_UNKNOWN,
                }
            ),
            "explain": explain,
            "daily_loss_today": float(daily_loss_today),
            "daily_loss_limit_pct": limit_pct,
            "daily_loss_limit_dollars": (
                float(loss_limit_dollars) if loss_limit_dollars is not None else None
            ),
            "daily_loss_limit_hit": bool(daily_loss_limit_hit),
            "inputs": inputs,
            "thresholds_pct": {
                "warn": float(self.config.warn_profit_pct),
                "lock1": float(self.config.lock1_profit_pct),
                "lock2": float(self.config.lock2_profit_pct),
                "lock3": float(self.config.lock3_profit_pct),
                "hard_stop": float(self.config.hard_stop_profit_pct),
                "daily_loss_limit": limit_pct,
            },
            "factors": {
                "lock1": float(self.config.lock1_sizing_factor),
                "lock2": float(self.config.lock2_sizing_factor),
                "lock3": float(self.config.lock3_sizing_factor),
                "hard_stop": float(self.config.hard_stop_sizing_factor),
            },
            "config": asdict(self.config),
        }
        return state

    async def write_state(self, out_path: Path, days: int = 0) -> Mapping[str, Any]:
        """
        Compute and write the profit‑lock state to a file atomically.

        Returns the state dictionary for convenience.
        """
        _previous_mode = ""
        try:
            if out_path.is_file():
                _prev_raw = json.loads(out_path.read_text(encoding="utf-8"))
                _previous_mode = str(_prev_raw.get("mode") or "")
        except Exception:
            _previous_mode = ""

        state = await self.build_state(days=days)
        # ensure directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(out_path.name + ".tmp")
        # write to tmp file in thread to avoid blocking event loop
        await asyncio.to_thread(self._atomic_write_json, tmp_path, out_path, state)

        try:
            new_mode = str(state.get("mode") or "")
            if new_mode and _previous_mode and new_mode != _previous_mode:
                from chad.utils.telegram_notify import notify
                _daily_pnl = float(state.get("daily_loss_today", 0.0) or 0.0)
                _loss_limit = float(state.get("daily_loss_limit_dollars") or 0.0)
                _severity = "critical" if new_mode == "HARD_STOP" else "warning"
                notify(
                    f"⚠️ PROFIT LOCK — {new_mode}\n"
                    f"Daily P&L: ${_daily_pnl:.2f} / Limit: ${_loss_limit:.2f}",
                    severity=_severity,
                    dedupe_key=f"profit_lock_{new_mode}",
                )
        except Exception:
            pass

        # Side-write canonical pnl_state.json from the same computation
        inputs = state.get("inputs", {})

        # BOX-034A Inc 3 Step 1b: additively propagate currency tags onto the
        # operator-facing account_equity. We co-locate a read of the SAME
        # dynamic_caps.json that DynamicCapsEquityProvider sources from (matched
        # by the equity_source string), so this needs no EquityProvider ABC
        # change. Additive only — NO reader assertions here. Fail-closed False
        # on any ambiguity (unknown equity, fallback source, absent key, read
        # error). The currency uses the dynamic_caps tag when the equity came
        # from there, else the configured base.
        base_currency = os.environ.get("CHAD_BASE_CURRENCY", "CAD").strip().upper() or "CAD"
        _equity_source = inputs.get("equity_source")
        _equity_known = bool(inputs.get("equity_known", False))
        _account_equity = inputs.get("account_equity")
        _dynamic_caps_path = self.repo_root / "runtime" / "dynamic_caps.json"
        _from_dynamic_caps = _equity_source == str(_dynamic_caps_path)

        account_equity_currency = base_currency
        account_equity_currency_ok = False
        if _from_dynamic_caps:
            try:
                _dc = json.loads(_dynamic_caps_path.read_text(encoding="utf-8"))
            except Exception:
                _dc = {}
            if isinstance(_dc, dict):
                _dc_currency = _dc.get("total_equity_currency")
                if isinstance(_dc_currency, str) and _dc_currency.strip():
                    account_equity_currency = _dc_currency.strip().upper()
                account_equity_currency_ok = (
                    _equity_known
                    and _account_equity is not None
                    and _dc.get("total_equity_currency_ok") is True
                )

        pnl_state = {
            "schema_version": "pnl_state.v1",
            "ts_utc": state["ts_utc"],
            "ttl_seconds": int(self.config.ttl_seconds),
            "realized_pnl": float(inputs.get("realized_pnl", 0.0)),
            "trade_count": int(inputs.get("trade_count", 0)),
            "pnl_pct_of_equity": float(inputs.get("pnl_pct_of_equity", 0.0)),
            "account_equity": inputs.get("account_equity"),
            "account_equity_currency": account_equity_currency,
            "account_equity_currency_ok": bool(account_equity_currency_ok),
            "equity_known": bool(inputs.get("equity_known", False)),
            "pnl_sources": list(inputs.get("pnl_sources", [])),
            "source": "profit_lock_engine",
        }
        pnl_path = out_path.parent / "pnl_state.json"
        pnl_tmp = pnl_path.with_name(pnl_path.name + ".tmp")
        await asyncio.to_thread(self._atomic_write_json, pnl_tmp, pnl_path, pnl_state)

        return state

    @staticmethod
    def _atomic_write_json(tmp_path: Path, final_path: Path, data: Mapping[str, Any]) -> None:
        """
        Write JSON atomically: write to a tmp file then rename to final.

        This synchronous helper is intended to run in a thread via
        `asyncio.to_thread` to avoid blocking the event loop during file IO.
        """
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, final_path)


def _build_default_equity_provider() -> EquityProvider:
    """
    Construct a default composite equity provider suitable for the CHAD system.

    The providers are tried in order: dynamic caps, positions snapshot,
    portfolio state, positions truth.  Additional providers can be added here
    without modifying other code.
    """
    providers: List[EquityProvider] = [
        DynamicCapsEquityProvider(),
        FileEquityProvider("positions_snapshot.json", ("net_liquidation", "netLiquidation", "equity", "total_equity")),
        FileEquityProvider("portfolio_state.json", ("equity", "total_equity", "account_equity")),
        FileEquityProvider("positions_truth.json", ("equity", "total_equity", "net_liquidation")),
    ]
    return CompositeEquityProvider(providers)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Parse command line arguments for the profit lock tool.

    Options:

    * `--repo`: repository root (defaults to current working directory)
    * `--out`: output path for the state JSON (defaults to `<repo>/runtime/profit_lock_state.json`)
    * `--days`: how many days back to include in PnL calculation (default 0 = today only)
    """
    parser = argparse.ArgumentParser(description="Compute CHAD profit lock state")
    parser.add_argument(
        "--repo",
        default=".",
        help="Repository root containing runtime/ and data/ directories",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output file for JSON state; default: <repo>/runtime/profit_lock_state.json",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=0,
        help="Number of days back (inclusive) for PnL aggregation (0 = today)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase logging verbosity",
    )
    return parser.parse_args(argv)


async def main_async(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    repo_root = Path(args.repo).expanduser().resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        logger.error("Repository root %s does not exist", repo_root)
        return 1
    config = ProfitLockConfig.from_env()
    equity_provider = _build_default_equity_provider()
    pnl_provider = NdjsonPnlProvider()
    engine = ProfitLockEngine(repo_root, equity_provider, pnl_provider, config)
    out_path = Path(args.out).expanduser().resolve() if args.out else repo_root / "runtime" / "profit_lock_state.json"
    try:
        state = await engine.write_state(out_path, days=args.days)
    except Exception as exc:
        logger.exception("Failed to compute or write profit lock state: %s", exc)
        return 1
    # Refresh stop_state.json TTL — preserve existing operator intent.
    # Stop file is the highest-authority kill switch; we never override its
    # value here, only re-stamp the TTL so it does not go stale between
    # explicit operator changes.
    try:
        from chad.core.stop_state import STOP_PATH, write_stop_state
        if STOP_PATH.is_file():
            raw = json.loads(STOP_PATH.read_text(encoding="utf-8"))
            existing_stop = bool(raw.get("stop", False))
            existing_reason = str(raw.get("reason") or "unspecified")
            write_stop_state(
                stop=existing_stop,
                reason=f"ttl_refresh:{existing_reason}",
            )
    except Exception as exc:
        logger.warning("stop_state TTL refresh skipped: %s", exc)

    # Print summary for shell usage
    decision = state.get("mode", "")
    sizing = state.get("sizing_factor", 1.0)
    stop = state.get("stop_new_entries", False)
    print(f"wrote={out_path}")
    print(f"mode={decision}")
    print(f"sizing_factor={sizing}")
    print(f"stop_new_entries={stop}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Synchronous entry point wrapping the async main."""
    try:
        rc = asyncio.run(main_async(argv))
    except KeyboardInterrupt:
        rc = 130
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
