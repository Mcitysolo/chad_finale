from __future__ import annotations

from fastapi import APIRouter, HTTPException

from chad.portfolio.portfolio_engine import PortfolioEngine

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.get("/active")
def portfolio_active() -> dict:
    eng = PortfolioEngine()
    out = eng.load_active_positions()
    if not out.get("ok"):
        raise HTTPException(status_code=503, detail=out.get("error") or "active_positions_error")
    return out


@router.get("/targets/{profile}")
def portfolio_targets(profile: str) -> dict:
    eng = PortfolioEngine()
    out = eng.get_targets(profile)
    if not out.get("ok"):
        raise HTTPException(status_code=404, detail=out.get("error") or "unknown_profile")
    return out


@router.get("/rebalance/latest")
def portfolio_rebalance_latest(profile: str = "BALANCED") -> dict:
    eng = PortfolioEngine()
    out = eng.get_rebalance_latest(profile)
    if not out.get("ok"):
        raise HTTPException(status_code=503, detail=out.get("error") or "rebalance_error")
    return out
