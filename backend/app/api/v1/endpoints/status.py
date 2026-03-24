from __future__ import annotations
import asyncio
from fastapi import APIRouter, Depends
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db, AsyncSessionLocal
from app.models.property import Property
from app.models.indicator import PropertyIndicator
from app.models.score import PropertyScore

router = APIRouter(prefix="/status", tags=["status"])


async def _run_query(query):
    """Run a query in its own session so multiple can execute concurrently."""
    async with AsyncSessionLocal() as session:
        return await session.execute(query)


@router.get("")
async def get_status(db: AsyncSession = Depends(get_db)):
    # Run all independent queries concurrently
    (
        total_res,
        state_res,
        ind_res,
        score_res,
        top_res,
        null_res,
        loc_res,
        dup_res,
    ) = await asyncio.gather(
        _run_query(select(func.count(Property.id))),
        _run_query(
            select(Property.address_state, func.count(Property.id).label("cnt"))
            .group_by(Property.address_state)
            .order_by(func.count(Property.id).desc())
            .limit(20)
        ),
        _run_query(
            select(PropertyIndicator.indicator_type, func.count(PropertyIndicator.id).label("cnt"))
            .group_by(PropertyIndicator.indicator_type)
            .order_by(func.count(PropertyIndicator.id).desc())
        ),
        _run_query(
            select(PropertyScore.score_tier, func.count(PropertyScore.id).label("cnt"))
            .group_by(PropertyScore.score_tier)
            .order_by(func.count(PropertyScore.id).desc())
        ),
        _run_query(
            select(PropertyScore.total_score, func.count(PropertyScore.id).label("cnt"))
            .where(PropertyScore.total_score > 50)
            .group_by(PropertyScore.total_score)
            .order_by(PropertyScore.total_score.desc())
            .limit(10)
        ),
        _run_query(
            select(func.count(Property.id))
            .where((Property.address_state == None) | (Property.address_state == ""))
        ),
        _run_query(
            select(func.count(Property.id)).where(Property.location.isnot(None))
        ),
        _run_query(
            text("SELECT COUNT(*) FROM (SELECT address_normalized, county_id FROM properties WHERE address_normalized IS NOT NULL GROUP BY address_normalized, county_id HAVING COUNT(*) > 1) sub")
        ),
    )

    total_props = total_res.scalar() or 0
    state_rows = state_res.all()
    ind_rows = ind_res.all()
    score_rows = score_res.all()
    top_scores = top_res.all()
    null_state = null_res.scalar() or 0
    with_location = loc_res.scalar() or 0
    dup_count = dup_res.scalar() or 0
    total_indicators = sum(r.cnt for r in ind_rows)

    return {
        "totals": {"properties": total_props, "indicators": total_indicators, "with_location": with_location, "null_state": null_state, "duplicate_addresses": dup_count},
        "by_state": [{"state": r.address_state or "(none)", "count": r.cnt} for r in state_rows],
        "by_indicator": [{"type": r.indicator_type, "count": r.cnt} for r in ind_rows],
        "by_tier": [{"tier": r.score_tier or "unscored", "count": r.cnt} for r in score_rows],
        "top_scores": [{"score": float(r.total_score), "count": r.cnt} for r in top_scores],
        "scrapers": [
            {"key": "tx_dallas_code_enforcement", "county": "Dallas, TX", "signal": "code_violation", "freq": "daily", "status": "running", "last_result": "6,288 found · 5,606 upserted"},
            {"key": "tx_dallas_lis_pendens", "county": "Dallas, TX", "signal": "pre_foreclosure", "freq": "daily", "status": "running", "last_result": "116 found · 116 upserted"},
            {"key": "tx_dallas_foreclosure", "county": "Dallas, TX", "signal": "foreclosure", "freq": "weekly", "status": "pending", "last_result": "—"},
            {"key": "tx_dallas_tax_delinquent", "county": "Dallas, TX", "signal": "tax_delinquent", "freq": "weekly", "status": "pending", "last_result": "—"},
            {"key": "tx_dallas_supabase_leads", "county": "Dallas, TX", "signal": "multi", "freq": "weekly", "status": "running", "last_result": "GCP sync"},
            {"key": "tx_harris_absentee_owner", "county": "Harris, TX", "signal": "absentee_owner", "freq": "monthly", "status": "pending", "last_result": "HCAD bulk"},
            {"key": "fl_hillsborough_eviction", "county": "Hillsborough, FL", "signal": "eviction", "freq": "daily", "status": "running", "last_result": "—"},
            {"key": "fl_hillsborough_probate", "county": "Hillsborough, FL", "signal": "probate", "freq": "daily", "status": "running", "last_result": "—"},
            {"key": "fl_hillsborough_lis_pendens", "county": "Hillsborough, FL", "signal": "pre_foreclosure", "freq": "daily", "status": "running", "last_result": "—"},
            {"key": "fl_hillsborough_foreclosure", "county": "Hillsborough, FL", "signal": "foreclosure", "freq": "daily", "status": "running", "last_result": "—"},
            {"key": "fl_polk_absentee_owner", "county": "Polk, FL", "signal": "absentee_owner", "freq": "weekly", "status": "running", "last_result": "FTP bulk"},
            {"key": "fl_polk_code_enforcement", "county": "Polk, FL", "signal": "code_violation", "freq": "daily", "status": "running", "last_result": "Accela POLKCO"},
            {"key": "fl_polk_out_of_state", "county": "Polk, FL", "signal": "out_of_state_owner", "freq": "weekly", "status": "running", "last_result": "FTP bulk"},
            {"key": "fl_polk_official_records", "county": "Polk, FL", "signal": "lien,probate,pre_foreclosure", "freq": "daily", "status": "running", "last_result": "Clerk OR API via proxy"},
            {"key": "fl_polk_eviction", "county": "Polk, FL", "signal": "eviction", "freq": "daily", "status": "running", "last_result": "PRO via 2captcha + proxy"},
            {"key": "fl_polk_foreclosure", "county": "Polk, FL", "signal": "foreclosure", "freq": "daily", "status": "running", "last_result": "PRO via 2captcha + proxy"},
            {"key": "tx_dallas_probate", "county": "Dallas, TX", "signal": "probate", "freq": "weekly", "status": "broken", "last_result": "found=0 · Odyssey returns empty"},
            {"key": "tx_dallas_eviction_fed", "county": "Dallas, TX", "signal": "eviction", "freq": "daily", "status": "disabled", "last_result": "IP blocked"},
            {"key": "tx_dallas_divorce", "county": "Dallas, TX", "signal": "divorce", "freq": "weekly", "status": "disabled", "last_result": "IP blocked"},
        ],
    }
