from dataclasses import dataclass
from typing import Dict


@dataclass
class FuturesSpec:
    point_value: float


FUTURES_SPECS: Dict[str, FuturesSpec] = {
    "MES": FuturesSpec(point_value=5.0),
    "MNQ": FuturesSpec(point_value=2.0),
    "MCL": FuturesSpec(point_value=100.0),
    "MGC": FuturesSpec(point_value=10.0),
}


def compute_contract_size(
    symbol: str,
    price: float,
    atr: float,
    equity: float,
    risk_pct: float = 0.01,
    max_contracts: int = 5,
) -> int:
    """
    Professional-grade futures position sizing

    Uses:
    - ATR-based risk
    - % of equity risk
    """

    spec = FUTURES_SPECS.get(symbol)
    if not spec:
        return 0

    risk_dollars = equity * risk_pct

    # Risk per contract
    risk_per_contract = atr * spec.point_value

    if risk_per_contract <= 0:
        return 0

    contracts = int(risk_dollars / risk_per_contract)

    return max(1, min(contracts, max_contracts))
