"""Market Intelligence endpoint — reads from pre-computed materialized view."""
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db

router = APIRouter(prefix="/market", tags=["market"])


@router.get("/stats")
async def market_stats(db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(text("""
        SELECT * FROM mv_market_stats
        ORDER BY state, county, hot DESC, zip
    """))).fetchall()

    result = {}
    for row in rows:
        state = row.state or "Unknown"
        county = row.county or "Unknown"
        if state not in result:
            result[state] = {}
        if county not in result[state]:
            result[state][county] = {
                "fips": row.fips_code,
                "total": 0, "hot": 0, "warm": 0, "cold": 0,
                "zips": []
            }
        c = result[state][county]
        c["total"] += row.total or 0
        c["hot"] += row.hot or 0
        c["warm"] += row.warm or 0
        c["cold"] += row.cold or 0
        c["zips"].append({
            "zip": row.zip,
            "total": row.total or 0,
            "hot": row.hot or 0,
            "warm": row.warm or 0,
            "cold": row.cold or 0,
            "avg_score": float(row.avg_score or 0),
            "tax_delinquent": row.tax_delinquent or 0,
            "foreclosure": row.foreclosure or 0,
            "pre_foreclosure": row.pre_foreclosure or 0,
            "probate": row.probate or 0,
            "code_violation": row.code_violation or 0,
            "absentee": row.absentee or 0,
            "eviction": row.eviction or 0,
            "lien": row.lien or 0,
        })
    return result


@router.post("/refresh")
async def refresh_market_stats(db: AsyncSession = Depends(get_db)):
    """Refresh the materialized view. Call after ingestion runs."""
    await db.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY mv_market_stats"))
    await db.commit()
    return {"status": "refreshed"}
