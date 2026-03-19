"""
Motivated Seller Scoring Engine

Each indicator type carries a base weight. Multiple indicators compound via
the "at least one" probability model:

    score = 100 * (1 - product(1 - effective_weight_i))

This gives diminishing returns: adding more indicators always increases the score
but never causes a simple sum to exceed 100.

Modifiers applied per indicator:
- Recency decay: older indicators carry less weight.
- Amount boost: large tax debt / lien amounts raise the effective weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.indicator import PropertyIndicator

# ---------------------------------------------------------------------------
# Configuration tables
# ---------------------------------------------------------------------------

# Base weights out of 100 — tuned empirically, stored here so they can be
# adjusted without a DB migration.
BASE_WEIGHTS: dict[str, float] = {
    "foreclosure":      65,
    "pre_foreclosure":  45,
    "probate":          35,
    "tax_delinquent":   25,
    "lien":             20,
    "eviction":         20,
    "expired_listing":  15,
    "code_violation":   15,
    "vacant":           15,
    "price_reduction":  10,
    "absentee_owner":    8,
    "days_on_market":    8,
}

# Recency decay — multiplier applied based on how old the indicator is
RECENCY_DECAY: list[tuple[int, float]] = [
    # (max_days, multiplier) — checked in order, first match wins
    (30,   1.00),
    (90,   0.85),
    (180,  0.65),
    (365,  0.40),
    (99999, 0.15),  # anything older
]

# Amount thresholds for tax_delinquent and lien — boost multiplier
AMOUNT_BOOST_THRESHOLDS: list[tuple[int, float]] = [
    # (min_cents, multiplier)
    (50_000_00, 1.4),   # > $50k
    (20_000_00, 1.25),  # > $20k
    (10_000_00, 1.15),  # > $10k
    (5_000_00,  1.05),  # > $5k
    (0,         1.00),  # below $5k
]

# DOM thresholds — boost for days_on_market indicator
DOM_BOOST_THRESHOLDS: list[tuple[int, float]] = [
    (365, 1.5),
    (180, 1.3),
    (90,  1.1),
    (0,   1.0),
]

HOT_THRESHOLD  = 60.0
WARM_THRESHOLD = 35.0


# ---------------------------------------------------------------------------
# Helper dataclass for lightweight scoring (avoids ORM dependency in tests)
# ---------------------------------------------------------------------------

@dataclass
class IndicatorSnapshot:
    """Minimal view of an indicator needed for scoring."""
    indicator_type: str
    status: str
    filing_date: date | None
    amount_cents: int | None = None
    days_on_market: int | None = None  # for days_on_market indicator


def _days_old(filing_date: date | None) -> int:
    if filing_date is None:
        return 0
    today = datetime.now(timezone.utc).date()
    return max(0, (today - filing_date).days)


def _recency_multiplier(filing_date: date | None) -> float:
    age = _days_old(filing_date)
    for max_days, multiplier in RECENCY_DECAY:
        if age <= max_days:
            return multiplier
    return 0.15


def _amount_boost(indicator_type: str, amount_cents: int | None) -> float:
    if indicator_type not in {"tax_delinquent", "lien"} or amount_cents is None:
        return 1.0
    for threshold, multiplier in AMOUNT_BOOST_THRESHOLDS:
        if amount_cents >= threshold:
            return multiplier
    return 1.0


def _dom_boost(days_on_market: int | None) -> float:
    if days_on_market is None:
        return 1.0
    for threshold, multiplier in DOM_BOOST_THRESHOLDS:
        if days_on_market >= threshold:
            return multiplier
    return 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_score(indicators: list[IndicatorSnapshot]) -> dict:
    """
    Calculate the motivated seller score for a property.

    Returns a dict with:
      - total_score: float 0-100
      - tier: 'hot' | 'warm' | 'cold'
      - breakdown: dict mapping indicator_type -> effective weight (0-100)
      - indicator_count: int
    """
    active = [i for i in indicators if i.status == "active"]
    if not active:
        return {"total_score": 0.0, "tier": "cold", "breakdown": {}, "indicator_count": 0}

    breakdown: dict[str, float] = {}

    for ind in active:
        base = BASE_WEIGHTS.get(ind.indicator_type, 5)
        recency = _recency_multiplier(ind.filing_date)
        amount = _amount_boost(ind.indicator_type, ind.amount_cents)
        dom = _dom_boost(ind.days_on_market) if ind.indicator_type == "days_on_market" else 1.0

        effective = min(base * recency * amount * dom, 55.0)  # hard cap at 55 per signal

        # If the same indicator_type appears more than once, keep the highest
        prev = breakdown.get(ind.indicator_type, 0.0)
        breakdown[ind.indicator_type] = max(prev, effective)

    # Compound via "at least one" probability model
    compound = 1.0
    for w in breakdown.values():
        compound *= 1.0 - (w / 100.0)

    total_score = round((1.0 - compound) * 100.0, 2)
    tier = "hot" if total_score >= HOT_THRESHOLD else "warm" if total_score >= WARM_THRESHOLD else "cold"

    return {
        "total_score": total_score,
        "tier": tier,
        "breakdown": {k: round(v, 1) for k, v in breakdown.items()},
        "indicator_count": len(breakdown),
    }


def score_from_orm(orm_indicators: list["PropertyIndicator"]) -> dict:
    """Convenience wrapper that converts ORM objects to IndicatorSnapshot."""
    snapshots = [
        IndicatorSnapshot(
            indicator_type=ind.indicator_type,
            status=ind.status,
            filing_date=ind.filing_date,
            amount_cents=ind.amount_cents,
        )
        for ind in orm_indicators
    ]
    return calculate_score(snapshots)
