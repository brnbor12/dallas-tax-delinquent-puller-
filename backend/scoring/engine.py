"""
Motivated Seller Scoring Engine v2

Additive model with stacking bonuses and timing decay.

Tiers:
  HOT     >= 80  (call immediately)
  WARM    >= 60  (multi-touch campaign)
  NURTURE >= 40  (drip automation)
  cold    <  40  (batch / skip)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.indicator import PropertyIndicator

# ---------------------------------------------------------------------------
# Distress signal weights (max 60 pts from this bucket)
# ---------------------------------------------------------------------------
DISTRESS_WEIGHTS: dict[str, float] = {
    "foreclosure":     40.0,
    "pre_foreclosure": 35.0,
    "probate":         20.0,
    "tax_delinquent":  20.0,
    "lien":            18.0,
    "eviction":        15.0,
    "code_violation":  15.0,
    "active_listing":  10.0,
    "expired_listing": 10.0,
    "vacant":          10.0,
    "price_reduction":  5.0,
}

# Ownership signal weights (max 25 pts)
OWNERSHIP_WEIGHTS: dict[str, float] = {
    "absentee_owner":     10.0,
    "out_of_state_owner":  8.0,
    "no_homestead":        7.0,
}

# Stacking bonuses — applied on top
STACK_BONUS: dict[int, float] = {
    2: 10.0,   # 2 distress signals
    3: 20.0,   # 3+ distress signals (use highest that applies)
}

# Ownership combo bonus
ABSENTEE_OOS_BONUS = 5.0   # absentee + out_of_state together

# Timing bonus per indicator (applied once to the freshest signal)
TIMING_BONUS: list[tuple[int, float]] = [
    (30,   5.0),
    (90,   3.0),
    (365,  1.0),
    (99999, 0.0),
]

# Amount boost for tax_delinquent / lien
AMOUNT_BOOST: list[tuple[int, float]] = [
    (50_000_00, 1.40),
    (20_000_00, 1.25),
    (10_000_00, 1.15),
    ( 5_000_00, 1.05),
    (0,         1.00),
]

# Tier thresholds
TIERS = [
    (80.0, "hot"),
    (60.0, "warm"),
    (40.0, "nurture"),
    (0.0,  "cold"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class IndicatorSnapshot:
    indicator_type: str
    status: str
    filing_date: date | None
    amount_cents: int | None = None


def _days_old(filing_date: date | None) -> int:
    if filing_date is None:
        return 999
    return max(0, (datetime.now(timezone.utc).date() - filing_date).days)


def _timing_bonus(filing_date: date | None) -> float:
    age = _days_old(filing_date)
    for max_days, bonus in TIMING_BONUS:
        if age <= max_days:
            return bonus
    return 0.0


def _amount_boost(indicator_type: str, amount_cents: int | None) -> float:
    if indicator_type not in {"tax_delinquent", "lien"} or not amount_cents:
        return 1.0
    for threshold, mult in AMOUNT_BOOST:
        if amount_cents >= threshold:
            return mult
    return 1.0


def _tier(score: float) -> str:
    for threshold, label in TIERS:
        if score >= threshold:
            return label
    return "cold"


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def calculate_score(indicators: list[IndicatorSnapshot]) -> dict:
    """
    Calculate motivated seller score.

    Returns:
      total_score: float 0-100
      tier: hot | warm | nurture | cold
      breakdown: dict of component scores
      indicator_count: int
    """
    active = [i for i in indicators if i.status == "active"]
    if not active:
        return {"total_score": 0.0, "tier": "cold", "breakdown": {}, "indicator_count": 0}

    distress_pts = 0.0
    ownership_pts = 0.0
    distress_types: set[str] = set()
    ownership_types: set[str] = set()
    freshest_date: date | None = None
    breakdown: dict[str, float] = {}

    # Deduplicate — keep best per type (highest amount_boost wins)
    best: dict[str, IndicatorSnapshot] = {}
    for ind in active:
        prev = best.get(ind.indicator_type)
        if prev is None:
            best[ind.indicator_type] = ind
        else:
            # Keep the one with more amount or more recent date
            curr_days = _days_old(ind.filing_date)
            prev_days = _days_old(prev.filing_date)
            if curr_days < prev_days:
                best[ind.indicator_type] = ind

    for itype, ind in best.items():
        boost = _amount_boost(itype, ind.amount_cents)

        if itype in DISTRESS_WEIGHTS:
            pts = min(DISTRESS_WEIGHTS[itype] * boost, DISTRESS_WEIGHTS[itype] * 1.4)
            distress_pts += pts
            distress_types.add(itype)
            breakdown[itype] = round(pts, 1)
        elif itype in OWNERSHIP_WEIGHTS:
            pts = OWNERSHIP_WEIGHTS[itype]
            ownership_pts += pts
            ownership_types.add(itype)
            breakdown[itype] = round(pts, 1)

        # Track freshest signal for timing bonus
        if ind.filing_date:
            if freshest_date is None or ind.filing_date > freshest_date:
                freshest_date = ind.filing_date

    # Cap buckets
    distress_pts = min(distress_pts, 60.0)
    ownership_pts = min(ownership_pts, 25.0)

    # Stacking bonus on distress signals
    stack_bonus = 0.0
    n_distress = len(distress_types)
    if n_distress >= 3:
        stack_bonus = STACK_BONUS[3]
    elif n_distress >= 2:
        stack_bonus = STACK_BONUS[2]

    # Ownership combo bonus
    combo_bonus = 0.0
    if "absentee_owner" in ownership_types and "out_of_state_owner" in ownership_types:
        combo_bonus = ABSENTEE_OOS_BONUS

    # Timing bonus (based on freshest signal)
    timing_pts = _timing_bonus(freshest_date)

    total = distress_pts + ownership_pts + stack_bonus + combo_bonus + timing_pts
    total = min(round(total, 2), 100.0)

    breakdown["_stack_bonus"] = stack_bonus
    breakdown["_combo_bonus"] = combo_bonus
    breakdown["_timing"] = timing_pts

    return {
        "total_score": total,
        "tier": _tier(total),
        "breakdown": breakdown,
        "indicator_count": len([k for k in breakdown if not k.startswith("_")]),
    }


def score_from_orm(orm_indicators: list["PropertyIndicator"]) -> dict:
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
