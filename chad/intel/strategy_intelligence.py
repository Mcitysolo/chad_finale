from __future__ import annotations

"""
CHAD Phase 9 — Strategy Intelligence (Claude-Powered)

Provides real-time intelligence to strategy execution:
- Confidence bias adjustments per symbol/strategy
- Universe filtering for binary risk events
- Regime profile classification

All outputs are fail-closed: errors return neutral bias (0.0), no filtering,
and "normal" regime. Never blocks execution.

Outputs written to runtime/strategy_intelligence.json for audit.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("chad.intel.strategy_intelligence")

# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

CONFIDENCE_BIAS_MIN = -0.15
CONFIDENCE_BIAS_MAX = 0.10
CONFIDENCE_CACHE_TTL_SEC = 300       # 5 minutes
REGIME_CACHE_TTL_SEC = 900           # 15 minutes
MAX_CALL_TIMEOUT_SEC = 5.0


@dataclass
class ConfidenceBias:
    symbol: str
    strategy: str
    adjustment: float
    reason: str
    macro_risk: str
    regime: str
    ts_utc: str


@dataclass
class UniverseFilter:
    avoid_symbols: List[str]
    prefer_symbols: List[str]
    reasons: Dict[str, str]
    ts_utc: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _neutral_bias(symbol: str, strategy: str, reason: str = "neutral_default") -> ConfidenceBias:
    return ConfidenceBias(
        symbol=symbol,
        strategy=strategy,
        adjustment=0.0,
        reason=reason,
        macro_risk="unknown",
        regime="unknown",
        ts_utc=_utc_now_iso(),
    )


def _neutral_universe_filter() -> UniverseFilter:
    return UniverseFilter(
        avoid_symbols=[],
        prefer_symbols=[],
        reasons={},
        ts_utc=_utc_now_iso(),
    )


# ---------------------------------------------------------------------------
# Strategy Intelligence
# ---------------------------------------------------------------------------


class StrategyIntelligence:
    """
    Claude-powered strategy intelligence layer.

    Provides confidence bias, universe filtering, and regime classification.
    All methods are fail-closed and time-bounded.
    """

    def __init__(
        self,
        claude_client: Any,
        runtime_dir: Path,
    ) -> None:
        self._client = claude_client
        self._runtime_dir = Path(runtime_dir)

        # In-memory caches
        self._confidence_cache: Dict[str, Dict[str, Any]] = {}  # key: "symbol|strategy"
        self._regime_cache: Dict[str, Dict[str, Any]] = {}      # key: strategy_name
        self._cache_path = self._runtime_dir / "strategy_intelligence_cache.json"
        self._output_path = self._runtime_dir / "strategy_intelligence.json"

        # Load persisted cache
        self._load_cache()

    # ------------------------------------------------------------------ #
    # Cache management
    # ------------------------------------------------------------------ #

    def _load_cache(self) -> None:
        """Load persisted cache from disk."""
        try:
            data = _read_json(self._cache_path)
            self._confidence_cache = data.get("confidence", {})
            self._regime_cache = data.get("regime", {})
        except Exception:
            pass

    def _save_cache(self) -> None:
        """Persist cache to disk."""
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "confidence": self._confidence_cache,
                "regime": self._regime_cache,
                "last_updated_utc": _utc_now_iso(),
            }
            tmp = self._cache_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            tmp.replace(self._cache_path)
        except Exception:
            pass

    def _write_audit(self, section: str, data: Any) -> None:
        """Append audit entry to runtime/strategy_intelligence.json."""
        try:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            existing = _read_json(self._output_path) if self._output_path.exists() else {}
            if not isinstance(existing, dict):
                existing = {}

            entries = existing.get(section, [])
            if not isinstance(entries, list):
                entries = []

            entry = data if isinstance(data, dict) else asdict(data)
            entries.append(entry)

            # Keep last 100 per section
            if len(entries) > 100:
                entries = entries[-100:]

            existing[section] = entries
            existing["last_updated_utc"] = _utc_now_iso()

            tmp = self._output_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
            tmp.replace(self._output_path)
        except Exception:
            pass

    def _is_cache_fresh(self, cache_entry: Dict[str, Any], ttl_sec: float) -> bool:
        """Check if a cache entry is still fresh."""
        ts_str = cache_entry.get("ts_utc", "")
        if not ts_str:
            return False
        try:
            cached_time = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - cached_time).total_seconds()
            return age < ttl_sec
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Context builders
    # ------------------------------------------------------------------ #

    def _load_market_context(self) -> Dict[str, Any]:
        """Load macro_state.json and execution_quality.json."""
        macro = _read_json(self._runtime_dir / "macro_state.json")
        exec_quality = _read_json(self._runtime_dir / "execution_quality.json")
        return {
            "macro_state": macro,
            "execution_quality": exec_quality,
        }

    def _load_news_headlines(self, symbols: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """Load recent Yahoo Finance news headlines."""
        try:
            from chad.market_data.yahoo_news_provider import YahooNewsProvider
            provider = YahooNewsProvider()
            items = provider.get_headlines(symbols=symbols or [], limit=5)
            return [
                {"headline": item.headline, "published_utc": item.published_utc}
                for item in items[:5]
            ]
        except Exception:
            return []

    # ------------------------------------------------------------------ #
    # get_confidence_bias
    # ------------------------------------------------------------------ #

    def get_confidence_bias(
        self,
        symbol: str,
        strategy_name: str,
        base_confidence: float,
        market_context: Optional[Dict[str, Any]] = None,
    ) -> ConfidenceBias:
        """
        Get Claude-powered confidence bias for a symbol/strategy pair.

        Returns adjustment in range [-0.15, +0.10] (asymmetric — easier to reduce).
        Cached per symbol per 5 minutes. Fail-closed: returns 0.0 on any error.
        """
        cache_key = f"{symbol}|{strategy_name}"

        # Check cache
        cached = self._confidence_cache.get(cache_key, {})
        if self._is_cache_fresh(cached, CONFIDENCE_CACHE_TTL_SEC):
            return ConfidenceBias(**{k: cached[k] for k in ConfidenceBias.__dataclass_fields__})

        try:
            return self._fetch_confidence_bias(symbol, strategy_name, base_confidence, market_context)
        except Exception as exc:
            LOG.warning("Confidence bias failed for %s/%s: %s", symbol, strategy_name, exc)
            return _neutral_bias(symbol, strategy_name, f"error:{type(exc).__name__}")

    def _fetch_confidence_bias(
        self,
        symbol: str,
        strategy_name: str,
        base_confidence: float,
        market_context: Optional[Dict[str, Any]],
    ) -> ConfidenceBias:
        """Fetch confidence bias from Claude."""
        if market_context is None:
            market_context = self._load_market_context()

        headlines = self._load_news_headlines([symbol])

        prompt = f"""Analyze confidence bias for trading signal.

Symbol: {symbol}
Strategy: {strategy_name}
Base confidence: {base_confidence}

Market context:
{json.dumps(market_context, indent=2, default=str)[:4000]}

Recent headlines:
{json.dumps(headlines, default=str)[:2000]}

Respond with JSON:
{{
  "adjustment": <float between -0.15 and +0.10>,
  "reason": "<one sentence>",
  "macro_risk": "<low|medium|high|extreme>",
  "regime": "<risk_on|neutral|risk_off|crisis>"
}}

Rules:
- Negative bias (reduce confidence) for: earnings within 48h, FOMC, geopolitical escalation, extreme VIX
- Positive bias (boost confidence) for: strong trend confirmation with low vol
- Default to 0.0 if uncertain
- Bias MUST be between -0.15 and +0.10"""

        t0 = time.monotonic()
        result = self._client.chat_json(
            prompt,
            system="You are a quantitative risk analyst. Respond only with the requested JSON.",
            task_type="routine",
        )
        elapsed = time.monotonic() - t0

        if elapsed > MAX_CALL_TIMEOUT_SEC:
            LOG.warning("Confidence bias took %.1fs (>%.1fs limit)", elapsed, MAX_CALL_TIMEOUT_SEC)

        # Parse and clamp
        adjustment = _clamp(float(result.get("adjustment", 0.0)), CONFIDENCE_BIAS_MIN, CONFIDENCE_BIAS_MAX)
        reason = str(result.get("reason", ""))[:200]
        macro_risk = str(result.get("macro_risk", "unknown"))
        regime = str(result.get("regime", "neutral"))

        bias = ConfidenceBias(
            symbol=symbol,
            strategy=strategy_name,
            adjustment=adjustment,
            reason=reason,
            macro_risk=macro_risk,
            regime=regime,
            ts_utc=_utc_now_iso(),
        )

        # Update cache
        self._confidence_cache[f"{symbol}|{strategy_name}"] = asdict(bias)
        self._save_cache()
        self._write_audit("confidence_bias", bias)

        return bias

    # ------------------------------------------------------------------ #
    # get_universe_filter
    # ------------------------------------------------------------------ #

    def get_universe_filter(
        self,
        strategy_name: str,
        proposed_symbols: List[str],
    ) -> UniverseFilter:
        """
        Filter proposed trading universe for binary risk events.

        Called once per orchestrator cycle. Checks news for earnings, FOMC,
        geopolitical events. Fail-closed: returns empty filter on error.
        """
        try:
            return self._fetch_universe_filter(strategy_name, proposed_symbols)
        except Exception as exc:
            LOG.warning("Universe filter failed for %s: %s", strategy_name, exc)
            return _neutral_universe_filter()

    def _fetch_universe_filter(
        self,
        strategy_name: str,
        proposed_symbols: List[str],
    ) -> UniverseFilter:
        """Fetch universe filter from Claude."""
        headlines = self._load_news_headlines(proposed_symbols)
        market_ctx = self._load_market_context()

        prompt = f"""Analyze proposed trading universe for binary risk events.

Strategy: {strategy_name}
Proposed symbols: {json.dumps(proposed_symbols)}

Market context:
{json.dumps(market_ctx, indent=2, default=str)[:3000]}

Recent headlines:
{json.dumps(headlines, default=str)[:3000]}

Respond with JSON:
{{
  "avoid_symbols": ["<symbol>", ...],
  "prefer_symbols": ["<symbol>", ...],
  "reasons": {{"<symbol>": "<reason>"}}
}}

Rules:
- Avoid symbols with: earnings within 24h, pending regulatory decisions, extreme event risk
- Prefer symbols with: clean technicals, no binary events, liquid markets
- If no clear reason to avoid, return empty avoid list
- Only include symbols from the proposed list"""

        t0 = time.monotonic()
        result = self._client.chat_json(
            prompt,
            system="You are a quantitative risk analyst. Respond only with the requested JSON.",
            task_type="routine",
        )
        elapsed = time.monotonic() - t0

        if elapsed > MAX_CALL_TIMEOUT_SEC:
            LOG.warning("Universe filter took %.1fs (>%.1fs limit)", elapsed, MAX_CALL_TIMEOUT_SEC)

        avoid = result.get("avoid_symbols", [])
        prefer = result.get("prefer_symbols", [])
        reasons = result.get("reasons", {})

        if not isinstance(avoid, list):
            avoid = []
        if not isinstance(prefer, list):
            prefer = []
        if not isinstance(reasons, dict):
            reasons = {}

        # Only include symbols that were actually proposed
        proposed_set = set(proposed_symbols)
        avoid = [s for s in avoid if s in proposed_set]
        prefer = [s for s in prefer if s in proposed_set]

        uf = UniverseFilter(
            avoid_symbols=avoid,
            prefer_symbols=prefer,
            reasons={k: str(v) for k, v in reasons.items()},
            ts_utc=_utc_now_iso(),
        )

        self._write_audit("universe_filter", asdict(uf))
        return uf

    # ------------------------------------------------------------------ #
    # get_regime_profile
    # ------------------------------------------------------------------ #

    def get_regime_profile(self, strategy_name: str) -> str:
        """
        Classify current regime and return pre-approved parameter profile.

        Returns "normal" or "conservative". Cached 15 minutes.
        Fail-closed: returns "normal" on any error.
        """
        # Check cache
        cached = self._regime_cache.get(strategy_name, {})
        if self._is_cache_fresh(cached, REGIME_CACHE_TTL_SEC):
            return str(cached.get("profile", "normal"))

        try:
            return self._fetch_regime_profile(strategy_name)
        except Exception as exc:
            LOG.warning("Regime profile failed for %s: %s", strategy_name, exc)
            return "normal"

    def _fetch_regime_profile(self, strategy_name: str) -> str:
        """Fetch regime profile from Claude."""
        market_ctx = self._load_market_context()

        # Extract key metrics
        macro = market_ctx.get("macro_state", {})
        vix_val = macro.get("vix", macro.get("vix_close", "unknown"))
        risk_label = macro.get("risk_label", "unknown")

        prompt = f"""Classify current market regime for strategy parameter selection.

Strategy: {strategy_name}
VIX: {vix_val}
Macro risk label: {risk_label}

Full macro context:
{json.dumps(macro, indent=2, default=str)[:3000]}

Respond with JSON:
{{
  "profile": "<normal|conservative>",
  "reasoning": "<one sentence>"
}}

Rules:
- "conservative" if: VIX > 25, or drawdown > 5%, or macro risk is high/extreme, or crisis regime
- "normal" otherwise
- Default to "normal" if data is insufficient"""

        t0 = time.monotonic()
        result = self._client.chat_json(
            prompt,
            system="You are a quantitative risk analyst. Respond only with the requested JSON.",
            task_type="routine",
        )
        elapsed = time.monotonic() - t0

        if elapsed > MAX_CALL_TIMEOUT_SEC:
            LOG.warning("Regime profile took %.1fs (>%.1fs limit)", elapsed, MAX_CALL_TIMEOUT_SEC)

        profile = str(result.get("profile", "normal")).strip().lower()
        if profile not in ("normal", "conservative"):
            profile = "normal"

        reasoning = str(result.get("reasoning", ""))[:200]

        # Update cache
        cache_entry = {
            "profile": profile,
            "reasoning": reasoning,
            "ts_utc": _utc_now_iso(),
        }
        self._regime_cache[strategy_name] = cache_entry
        self._save_cache()
        self._write_audit("regime_profile", {
            "strategy": strategy_name,
            **cache_entry,
        })

        return profile
