#!/usr/bin/env python3
"""
chad/risk/allocator_v3.py

PHASE 11.5 — Allocator V3
=========================
Correlation-aware, volatility-targeted, Kelly-constrained, regime-sensitive capital engine.

This module produces *strategy weight multipliers* and adjusted raw weights suitable for feeding into
DynamicRiskAllocator (which remains the final cap builder).

Inputs (read-only, no brokers)
------------------------------
- runtime/intel_cache/brain_returns_1m_active.ndjson   (shared active-minute return grid)
- runtime/intel_cache/brain_returns_state.json         (coverage + included strategies)
- runtime/intel_cache/macro_state.json                 (risk_label: risk_on|risk_off|neutral)

Outputs (optional; orchestrator may persist)
-------------------------------------------
- runtime/allocator_v3_state.json  (audit receipt; deterministic)

Core ideas
----------
1) Base weights: StrategyAllocation (env/defaults).
2) Regime tilt: mild multipliers based on macro risk_label.
3) Vol targeting: inverse volatility weights (risk parity style) across non-zero variance strategies.
4) Correlation penalty: shrink weights in correlated clusters using a covariance-based risk contribution proxy.
5) Kelly constraint: compute kelly_raw ~ mean/variance on active-minute returns; if kelly_raw <= 0 => clamp to 0.
   If positive, clamp to [0, kelly_max]. Kelly acts as a *ceiling*, never a booster beyond caps.
6) Smoothing: limit how fast weights change (max_step) and apply EMA to prevent thrash.
7) Always bounded + deterministic + fail-closed.

Safety rules
------------
- Never increases total risk budget (daily_risk_fraction unchanged).
- Never resurrects a strategy with base weight 0.
- If data missing/invalid => returns neutral multipliers and a state receipt explaining why.
- Zero-variance strategies are excluded from vol/corr/kelly math, but can still keep base weight.

Env knobs (sane defaults)
-------------------------
- CHAD_ALLOC_V3_ENABLED            (default: 1)
- CHAD_ALLOC_V3_MIN_VARIANCE       (default: 1e-12)
- CHAD_ALLOC_V3_KELLY_MAX          (default: 0.25)   # max fraction allowed by Kelly
- CHAD_ALLOC_V3_REGIME_ON          (default: 1.05)
- CHAD_ALLOC_V3_REGIME_OFF         (default: 0.95)
- CHAD_ALLOC_V3_CORR_PENALTY       (default: 0.35)   # strength of correlation penalty (0..1)
- CHAD_ALLOC_V3_VOL_POWER          (default: 1.0)    # inverse-vol exponent
- CHAD_ALLOC_V3_MAX_STEP           (default: 0.20)   # max relative change per cycle per strategy
- CHAD_ALLOC_V3_EMA_ALPHA          (default: 0.35)   # smoothing (0..1), higher = more responsive

Notes
-----
- Correlation computed only among non-zero variance strategies, on ACTIVE minutes.
- Covariance and penalty are computed on standardized returns for stability.
"""

from __future__ import annotations

import time
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

# Paths are derived from repo root by orchestrator; we accept repo_root injection.
DEFAULT_TTL_SECONDS = 300


# -----------------------------
# Env helpers
# -----------------------------

def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return float(default)
    try:
        x = float(str(v).strip())
        return float(x) if math.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _safe_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.is_file():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class AllocV3Config:
    enabled: bool
    min_variance: float
    kelly_max: float
    regime_on: float
    regime_off: float
    corr_penalty: float
    vol_power: float
    max_step: float
    ema_alpha: float
    ttl_seconds: int

    @classmethod
    def from_env(cls) -> "AllocV3Config":
        enabled = _env_bool("CHAD_ALLOC_V3_ENABLED", True)
        min_variance = _env_float("CHAD_ALLOC_V3_MIN_VARIANCE", 1e-12)
        kelly_max = _env_float("CHAD_ALLOC_V3_KELLY_MAX", 0.25)
        regime_on = _env_float("CHAD_ALLOC_V3_REGIME_ON", 1.05)
        regime_off = _env_float("CHAD_ALLOC_V3_REGIME_OFF", 0.95)
        corr_penalty = _env_float("CHAD_ALLOC_V3_CORR_PENALTY", 0.35)
        vol_power = _env_float("CHAD_ALLOC_V3_VOL_POWER", 1.0)
        max_step = _env_float("CHAD_ALLOC_V3_MAX_STEP", 0.20)
        ema_alpha = _env_float("CHAD_ALLOC_V3_EMA_ALPHA", 0.35)
        ttl = int(_env_float("CHAD_ALLOC_V3_TTL_SECONDS", DEFAULT_TTL_SECONDS))

        min_variance = max(0.0, float(min_variance))
        kelly_max = _clamp(float(kelly_max), 0.0, 1.0)
        corr_penalty = _clamp(float(corr_penalty), 0.0, 1.0)
        vol_power = _clamp(float(vol_power), 0.0, 4.0)
        max_step = _clamp(float(max_step), 0.0, 5.0)
        ema_alpha = _clamp(float(ema_alpha), 0.0, 1.0)
        ttl = max(30, int(ttl))

        # Regime tilts must be positive
        if not (math.isfinite(regime_on) and regime_on > 0.0):
            regime_on = 1.05
        if not (math.isfinite(regime_off) and regime_off > 0.0):
            regime_off = 0.95

        return cls(
            enabled=enabled,
            min_variance=min_variance,
            kelly_max=kelly_max,
            regime_on=float(regime_on),
            regime_off=float(regime_off),
            corr_penalty=corr_penalty,
            vol_power=vol_power,
            max_step=max_step,
            ema_alpha=ema_alpha,
            ttl_seconds=ttl,
        )


# -----------------------------
# Math helpers
# -----------------------------

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _var(xs: List[float], mu: float) -> float:
    if not xs:
        return 0.0
    return sum((x - mu) * (x - mu) for x in xs) / len(xs)


def _stdev(xs: List[float], mu: float) -> float:
    v = _var(xs, mu)
    return math.sqrt(v) if v > 0 else 0.0


def _cov(xs: List[float], ys: List[float], mx: float, my: float) -> float:
    n = min(len(xs), len(ys))
    if n == 0:
        return 0.0
    return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / n


def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, float(v)) for v in w.values())
    if s <= 0:
        return {k: 0.0 for k in w}
    return {k: max(0.0, float(v)) / s for k, v in w.items()}


def _apply_max_step(prev: Dict[str, float], target: Dict[str, float], max_step: float) -> Dict[str, float]:
    """
    Limit relative step per key:
      new = prev * clamp(target/prev, 1-max_step, 1+max_step)
    Handles prev=0 safely: if prev==0, allow target but still bounded via absolute clamp.
    """
    out: Dict[str, float] = {}
    for k, t in target.items():
        p = float(prev.get(k, 0.0))
        t = float(t)
        if p <= 0.0:
            # If previously zero, allow some activation only if target positive,
            # but do not exceed target (and do not resurrect if caller already zeroed)
            out[k] = max(0.0, t)
            continue
        if max_step <= 0.0:
            out[k] = max(0.0, t)
            continue
        ratio = t / p if p > 0 else 1.0
        ratio = _clamp(ratio, 1.0 - max_step, 1.0 + max_step)
        out[k] = max(0.0, p * ratio)
    return out


def _ema(prev: Dict[str, float], cur: Dict[str, float], alpha: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in cur.keys():
        p = float(prev.get(k, cur[k]))
        c = float(cur[k])
        out[k] = (1.0 - alpha) * p + alpha * c
    return out


# -----------------------------
# Core compute
# -----------------------------

def _load_active_returns(repo_root: Path) -> Tuple[Dict[str, List[float]], int, Optional[str]]:
    p = repo_root / "runtime" / "intel_cache" / "brain_returns_1m_active.ndjson"
    if not p.is_file():
        return {}, 0, f"missing:{p}"

    rets: Dict[str, List[float]] = {}
    rows = 0
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                r = obj.get("returns", {})
                if not isinstance(r, dict):
                    continue
                for k, v in r.items():
                    if k not in rets:
                        rets[k] = []
                    try:
                        fv = float(v)
                        if not math.isfinite(fv):
                            fv = 0.0
                    except Exception:
                        fv = 0.0
                    rets[k].append(float(fv))
                rows += 1
        return rets, rows, None
    except Exception as exc:
        return {}, 0, f"parse_error:{exc}"


def _macro_risk_label(repo_root: Path) -> str:
    obj = _safe_json(repo_root / "runtime" / "intel_cache" / "macro_state.json")
    if not obj:
        obj = _safe_json(repo_root / "runtime" / "macro_state.json")
    if not obj:
        return "unknown"
    payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else obj
    lab = str(payload.get("risk_label") or "unknown").strip().lower()
    return lab if lab in ("risk_on", "risk_off", "neutral") else "unknown"


def compute_allocator_v3(
    *,
    repo_root: Path,
    base_weights: Mapping[str, float],
    cfg: AllocV3Config,
    prev_state: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Returns (adjusted_raw_weights, state_receipt).
    adjusted_raw_weights are *raw* weights (not normalized), suitable for StrategyAllocation(weights=...).
    """
    state: Dict[str, Any] = {
        "schema_version": "allocator_v3_state.v1",
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ttl_seconds": int(cfg.ttl_seconds),
        "enabled": bool(cfg.enabled),
        "notes": [],
        "inputs": {
            "base_weights": {k: float(base_weights[k]) for k in sorted(base_weights.keys())},
            "macro_risk_label": _macro_risk_label(repo_root),
        },
        "metrics": {},
        "outputs": {},
    }

    # If disabled, return base
    if not cfg.enabled:
        state["notes"].append("disabled: returning base weights")
        return dict(base_weights), state

    rets, rows, err = _load_active_returns(repo_root)
    if err is not None or rows < 50:
        state["notes"].append(f"returns_unavailable_or_too_short:{err}:rows={rows}")
        return dict(base_weights), state

    # Compute mean/var per strategy (active minutes)
    mu: Dict[str, float] = {}
    var: Dict[str, float] = {}
    sd: Dict[str, float] = {}
    for s, xs in rets.items():
        m = _mean(xs)
        v = _var(xs, m)
        mu[s] = float(m)
        var[s] = float(v)
        sd[s] = float(math.sqrt(v) if v > 0 else 0.0)

    # Determine eligible strategies (non-zero variance, present in base weights, base weight > 0)
    eligible = []
    for s in sorted(base_weights.keys()):
        bw = float(base_weights.get(s, 0.0))
        if bw <= 0.0:
            continue
        if s not in sd:
            continue
        if var.get(s, 0.0) <= cfg.min_variance:
            continue
        eligible.append(s)

    state["metrics"]["rows_active"] = int(rows)
    state["metrics"]["eligible_strategies"] = eligible
    state["metrics"]["per_strategy"] = {s: {"mean": mu.get(s, 0.0), "sd": sd.get(s, 0.0), "var": var.get(s, 0.0)} for s in sorted(base_weights.keys()) if s in mu}

    if not eligible:
        state["notes"].append("no_eligible_strategies_nonzero_variance: returning base")
        return dict(base_weights), state

    # Regime tilt (mild)
    risk_label = state["inputs"]["macro_risk_label"]
    regime_mult: Dict[str, float] = {s: 1.0 for s in base_weights.keys()}
    if risk_label == "risk_off":
        for s in regime_mult:
            if s in ("omega",):
                regime_mult[s] = cfg.regime_on
            elif s in ("beta",):
                regime_mult[s] = 1.02
            else:
                regime_mult[s] = cfg.regime_off
    elif risk_label == "risk_on":
        for s in regime_mult:
            if s in ("omega",):
                regime_mult[s] = cfg.regime_off
            else:
                regime_mult[s] = cfg.regime_on

    # Vol targeting weights on eligible: inv_vol^power
    vol_w: Dict[str, float] = {}
    for s in eligible:
        inv = 1.0 / max(sd[s], 1e-12)
        vol_w[s] = float((inv ** cfg.vol_power))

    vol_w = _normalize_weights(vol_w)

    # Correlation penalty: compute corr among eligible, then penalize crowdedness
    # crowd(s) = average positive correlation with others (0..1), then factor = 1 - corr_penalty*crowd
    crowd: Dict[str, float] = {s: 0.0 for s in eligible}
    for i, a in enumerate(eligible):
        xa = rets[a]
        ma = mu[a]
        sa = sd[a]
        if sa <= 0:
            continue
        pos_sum = 0.0
        cnt = 0
        for j, b in enumerate(eligible):
            if b == a:
                continue
            xb = rets[b]
            mb = mu[b]
            sb = sd[b]
            if sb <= 0:
                continue
            c = _cov(xa, xb, ma, mb) / (sa * sb)
            if math.isfinite(c):
                if c > 0:
                    pos_sum += float(c)
                cnt += 1
        crowd[a] = (pos_sum / cnt) if cnt > 0 else 0.0

    corr_factor: Dict[str, float] = {}
    for s in eligible:
        cf = 1.0 - cfg.corr_penalty * _clamp(crowd[s], 0.0, 1.0)
        corr_factor[s] = float(_clamp(cf, 0.10, 1.0))

    # Kelly constraint (ceiling): k = mu/var. If <=0 => 0. If >0 => clamp to kelly_max.
    kelly_cap: Dict[str, float] = {}
    for s in eligible:
        v = var[s]
        k = (mu[s] / v) if v > 0 else 0.0
        if not math.isfinite(k) or k <= 0.0:
            kelly_cap[s] = 0.0
        else:
            kelly_cap[s] = float(_clamp(k, 0.0, cfg.kelly_max))

    # Combine:
    # target_weight(s) ∝ base_weight(s) * regime_mult(s) * vol_w(s) * corr_factor(s) * (1 + kelly_cap(s))
    # BUT Kelly never boosts if cap=0; it only allows limited scaling when positive.
    raw_target: Dict[str, float] = {}
    for s in eligible:
        bw = float(base_weights[s])
        k_boost = 1.0 + float(kelly_cap[s])
        raw_target[s] = bw * float(regime_mult.get(s, 1.0)) * float(vol_w.get(s, 0.0)) * float(corr_factor[s]) * k_boost

    # Normalize within eligible, then reapply to full strategy set by scaling base sum
    norm_target = _normalize_weights(raw_target)

    # Convert to raw weights that keep approximate base weight mass for eligible strategies
    base_mass = sum(float(base_weights[s]) for s in eligible)
    adjusted: Dict[str, float] = {k: float(base_weights.get(k, 0.0)) for k in base_weights.keys()}
    for s in eligible:
        adjusted[s] = float(norm_target[s] * base_mass)

    # Apply max-step + EMA smoothing using prev_state if present
    prev_adj = None
    if isinstance(prev_state, dict):
        prev_adj = prev_state.get("outputs", {}).get("adjusted_raw_weights")
        if isinstance(prev_adj, dict):
            prev_adj = {k: float(prev_adj.get(k, 0.0)) for k in adjusted.keys()}
        else:
            prev_adj = None

    if prev_adj is not None:
        stepped = _apply_max_step(prev_adj, adjusted, cfg.max_step)
        smoothed = _ema(prev_adj, stepped, cfg.ema_alpha)
        adjusted = smoothed
        state["notes"].append("applied_max_step_and_ema_smoothing")
    else:
        state["notes"].append("no_prev_state: no_smoothing_applied")

    # Never resurrect base zero weights
    for k in list(adjusted.keys()):
        if float(base_weights.get(k, 0.0)) <= 0.0:
            adjusted[k] = 0.0

    state["metrics"]["regime_mult"] = {k: float(regime_mult.get(k, 1.0)) for k in sorted(base_weights.keys())}
    state["metrics"]["vol_weights"] = {k: float(vol_w.get(k, 0.0)) for k in eligible}
    state["metrics"]["crowd"] = {k: float(crowd.get(k, 0.0)) for k in eligible}
    state["metrics"]["corr_factor"] = {k: float(corr_factor.get(k, 0.0)) for k in eligible}
    state["metrics"]["kelly_cap"] = {k: float(kelly_cap.get(k, 0.0)) for k in eligible}
    state["outputs"]["adjusted_raw_weights"] = {k: float(adjusted.get(k, 0.0)) for k in sorted(adjusted.keys())}

    return adjusted, state


def write_state_atomic(repo_root: Path, state: Mapping[str, Any]) -> Path:
    out = repo_root / "runtime" / "allocator_v3_state.json"
    tmp = out.with_suffix(out.suffix + f".tmp.{os.getpid()}")
    payload = dict(state)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out)
    try:
        dfd = os.open(str(out.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        pass
    return out
