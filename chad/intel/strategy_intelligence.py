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

    @staticmethod
    def _freshness_status(payload: Dict[str, Any]) -> str:
        """
        Classify a published runtime payload as 'real', 'stale', or 'unavailable'.

        Treats missing/empty payloads as unavailable, payloads older than
        ts_utc + ttl_seconds as stale, otherwise real. ts_utc and ttl_seconds
        are SSOT v5 contract fields written by every runtime publisher.
        """
        if not payload:
            return "unavailable"
        ts_str = str(payload.get("ts_utc") or "")
        ttl = payload.get("ttl_seconds")
        if not ts_str or ttl is None:
            return "unavailable"
        try:
            ttl_sec = float(ttl)
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            if age <= ttl_sec:
                return "real"
            return "stale"
        except Exception:
            return "unavailable"

    def _load_vix_context(self) -> Dict[str, Any]:
        """
        Load VIX (and equity-vol cousins) from runtime/price_cache.json with
        explicit freshness metadata. Never invents a number — if the cache is
        missing/stale or the symbol is absent, returns provider_status that
        downstream prompts and consumers can read.
        """
        price_cache = _read_json(self._runtime_dir / "price_cache.json")
        status = self._freshness_status(price_cache)
        prices = price_cache.get("prices", {}) if isinstance(price_cache, dict) else {}

        vix_val: Optional[float] = None
        vix_source: Optional[str] = None
        for key in ("VIX", "VIXY", "VXX", "UVXY"):
            raw = prices.get(key)
            if raw is None:
                continue
            try:
                vix_val = float(raw)
                vix_source = key
                break
            except (TypeError, ValueError):
                continue

        if vix_val is None:
            return {
                "vix": None,
                "vix_source": None,
                "provider_status": "unavailable" if status != "real" else "unavailable_no_symbol",
                "ts_utc": price_cache.get("ts_utc") if isinstance(price_cache, dict) else None,
                "ttl_seconds": price_cache.get("ttl_seconds") if isinstance(price_cache, dict) else None,
            }
        return {
            "vix": vix_val,
            "vix_source": vix_source,
            "provider_status": status,  # real | stale | unavailable
            "ts_utc": price_cache.get("ts_utc"),
            "ttl_seconds": price_cache.get("ttl_seconds"),
        }

    def _load_event_context(self) -> Dict[str, Any]:
        """
        Load runtime/event_risk.json with freshness + provider status.

        Prefers structured next_event (real economic-calendar provider).
        Marks provider_status='placeholder_or_unavailable' if the publisher
        emitted a time-of-day MarketHoursRiskProvider stub or no real event.
        """
        event_risk = _read_json(self._runtime_dir / "event_risk.json")
        status = self._freshness_status(event_risk)

        if not event_risk:
            return {
                "provider_status": "unavailable",
                "severity": None,
                "next_event": None,
                "elevated_risk": False,
                "source_provider": None,
            }

        provider = ""
        source = event_risk.get("source")
        if isinstance(source, dict):
            provider = str(source.get("provider") or "")

        next_event = event_risk.get("next_event")
        is_real_provider = provider == "EconomicCalendarRiskProvider" and isinstance(next_event, dict)
        is_placeholder = provider in ("MarketHoursRiskProvider", "MarketHoursRiskProvider(fallback)", "error")

        if is_placeholder or not is_real_provider:
            ps = "placeholder_or_unavailable"
        elif status == "stale":
            ps = "stale"
        else:
            ps = status  # real | unavailable

        return {
            "provider_status": ps,
            "severity": event_risk.get("severity"),
            "next_event": next_event if isinstance(next_event, dict) else None,
            "elevated_risk": bool(event_risk.get("elevated_risk", False)),
            "source_provider": provider or None,
            "ts_utc": event_risk.get("ts_utc"),
            "ttl_seconds": event_risk.get("ttl_seconds"),
        }

    def _load_market_context(self) -> Dict[str, Any]:
        """
        Load macro / VIX / event-risk runtime state with freshness+source metadata.

        Each subsection carries provider_status so prompts and audit consumers
        can distinguish real, stale, unavailable, and placeholder data — no
        section is silently passed through as 'unknown'.
        """
        macro = _read_json(self._runtime_dir / "macro_state.json")
        exec_quality = _read_json(self._runtime_dir / "execution_quality.json")
        macro_status = self._freshness_status(macro)
        macro_meta = {
            "provider_status": macro_status,
            "ts_utc": macro.get("ts_utc") if isinstance(macro, dict) else None,
            "ttl_seconds": macro.get("ttl_seconds") if isinstance(macro, dict) else None,
            "provider": (macro.get("source") or {}).get("provider") if isinstance(macro.get("source"), dict) else None,
        } if isinstance(macro, dict) else {"provider_status": "unavailable"}

        return {
            "macro_state": macro,
            "macro_meta": macro_meta,
            "vix": self._load_vix_context(),
            "event_risk": self._load_event_context(),
            "execution_quality": exec_quality,
        }

    def _get_trends_adjustment(self, symbol: str, strategy_name: str) -> float:
        """
        Get confidence adjustment from Google Trends data.

        HIGH interest (ratio > 1.5):
          +0.05 for momentum strategies (alpha, gamma)
          -0.05 for reversion strategies (gamma_reversion)
        LOW interest (ratio < 0.5):
          -0.05 for momentum strategies
          +0.05 for reversion strategies
        Returns 0.0 on any error or NEUTRAL signal.
        """
        try:
            trends_state = _read_json(self._runtime_dir / "trends_state.json")
            signals = trends_state.get("signals", {})
            sig = signals.get(symbol)
            if not sig or not isinstance(sig, dict):
                return 0.0

            signal = sig.get("signal", "NEUTRAL")
            if signal == "NEUTRAL":
                return 0.0

            momentum_strategies = {"alpha", "gamma", "alpha_futures", "gamma_futures"}
            reversion_strategies = {"gamma_reversion"}

            strat = strategy_name.lower()

            if signal == "HIGH":
                if strat in momentum_strategies:
                    return 0.05
                if strat in reversion_strategies:
                    return -0.05
            elif signal == "LOW":
                if strat in momentum_strategies:
                    return -0.05
                if strat in reversion_strategies:
                    return 0.05

            return 0.0
        except Exception:
            return 0.0

    def _get_reddit_adjustment(self, symbol: str, strategy_name: str) -> float:
        """
        Get confidence adjustment from Reddit sentiment data.

        HYPE (>50 mentions, positive):
          +0.05 for momentum strategies (crowded trade may continue)
          -0.08 for reversion strategies (crowded = risky to fade)
        BEARISH with high mentions:
          -0.05 for momentum (negative pressure)
          +0.03 for reversion (oversold sentiment precedes bounces)
        Returns 0.0 on any error or NEUTRAL/low-mention signals.
        """
        try:
            state = _read_json(self._runtime_dir / "reddit_sentiment.json")
            signals = state.get("signals", {})
            sig = signals.get(symbol)
            if not sig or not isinstance(sig, dict):
                return 0.0

            signal = sig.get("signal", "NEUTRAL")
            if signal == "NEUTRAL":
                return 0.0

            momentum_strategies = {"alpha", "gamma", "alpha_futures", "gamma_futures"}
            reversion_strategies = {"gamma_reversion"}
            strat = strategy_name.lower()

            if signal == "HYPE":
                if strat in momentum_strategies:
                    return 0.05
                if strat in reversion_strategies:
                    return -0.08
            elif signal == "BEARISH":
                mentions = sig.get("mention_count", 0)
                if mentions >= 10:  # meaningful volume of discussion
                    if strat in momentum_strategies:
                        return -0.05
                    if strat in reversion_strategies:
                        return 0.03

            return 0.0
        except Exception:
            return 0.0

    def _get_short_interest_adjustment(self, symbol: str, strategy_name: str) -> float:
        """
        Get confidence adjustment from short interest data.

        EXTREME short interest + price uptrend (squeeze):
          +0.08 for momentum strategies (ride the squeeze)
        HIGH short interest + price downtrend:
          +0.04 for momentum (shorts pushing price lower)
        Returns 0.0 on any error or LOW/MODERATE signals.
        """
        try:
            state = _read_json(self._runtime_dir / "short_interest.json")
            signals = state.get("signals", {})
            sig = signals.get(symbol)
            if not sig or not isinstance(sig, dict):
                return 0.0

            signal = sig.get("signal", "LOW")
            squeeze = sig.get("squeeze_risk", False)

            momentum_strategies = {"alpha", "gamma", "alpha_futures", "gamma_futures"}
            strat = strategy_name.lower()

            if signal == "EXTREME" and squeeze and strat in momentum_strategies:
                return 0.08
            if signal == "HIGH" and not squeeze and strat in momentum_strategies:
                return 0.04

            return 0.0
        except Exception:
            return 0.0

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

        # Parse and clamp — include alternative data adjustments
        base_adj = float(result.get("adjustment", 0.0))
        trends_adj = self._get_trends_adjustment(symbol, strategy_name)
        reddit_adj = self._get_reddit_adjustment(symbol, strategy_name)
        short_adj = self._get_short_interest_adjustment(symbol, strategy_name)
        adjustment = _clamp(base_adj + trends_adj + reddit_adj + short_adj,
                            CONFIDENCE_BIAS_MIN, CONFIDENCE_BIAS_MAX)
        reason = str(result.get("reason", ""))[:200]
        adj_tags = []
        if trends_adj != 0.0:
            adj_tags.append(f"trends:{'+' if trends_adj > 0 else ''}{trends_adj:.2f}")
        if reddit_adj != 0.0:
            adj_tags.append(f"reddit:{'+' if reddit_adj > 0 else ''}{reddit_adj:.2f}")
        if short_adj != 0.0:
            adj_tags.append(f"short:{'+' if short_adj > 0 else ''}{short_adj:.2f}")
        if adj_tags:
            reason = f"{reason} [{','.join(adj_tags)}]"
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

        macro = market_ctx.get("macro_state", {}) or {}
        macro_meta = market_ctx.get("macro_meta", {}) or {}
        vix_ctx = market_ctx.get("vix", {}) or {}
        event_ctx = market_ctx.get("event_risk", {}) or {}

        macro_status = macro_meta.get("provider_status", "unavailable")
        risk_label = (
            str(macro.get("risk_label") or macro.get("macro_label") or "unavailable")
            if macro_status == "real" else "unavailable"
        )

        vix_status = vix_ctx.get("provider_status", "unavailable")
        if vix_status == "real" and vix_ctx.get("vix") is not None:
            vix_line = (
                f"VIX: {vix_ctx['vix']} (source={vix_ctx.get('vix_source')}, status=real)"
            )
        elif vix_status == "stale" and vix_ctx.get("vix") is not None:
            vix_line = (
                f"VIX: {vix_ctx['vix']} (source={vix_ctx.get('vix_source')}, status=stale — DEGRADE CONSERVATIVELY)"
            )
        else:
            vix_line = f"VIX: unavailable (status={vix_status} — do not infer; degrade conservatively)"

        event_status = event_ctx.get("provider_status", "unavailable")
        next_event = event_ctx.get("next_event") or {}
        if event_status == "real" and next_event:
            event_line = (
                f"Event risk: severity={event_ctx.get('severity')}; "
                f"next_event={next_event.get('name')} in {next_event.get('hours_until')}h "
                f"(severity={next_event.get('severity')}, source={next_event.get('source')}, status=real)"
            )
        else:
            event_line = (
                f"Event risk: status={event_status} — no real structured next_event; "
                f"do not treat as neutral; degrade conservatively"
            )

        prompt = f"""Classify current market regime for strategy parameter selection.

Strategy: {strategy_name}
{vix_line}
Macro risk label: {risk_label} (status={macro_status})
{event_line}

Full macro context:
{json.dumps(macro, indent=2, default=str)[:2500]}

Respond with JSON:
{{
  "profile": "<normal|conservative>",
  "reasoning": "<one sentence>"
}}

Rules:
- "conservative" if: VIX > 25, or drawdown > 5%, or macro risk is high/extreme, or crisis regime,
  or any of (VIX status, macro status, event_risk status) is stale/unavailable/placeholder_or_unavailable
- "normal" otherwise
- Default to "conservative" if any input is missing or marked unavailable — never treat 'unavailable' as neutral"""

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
