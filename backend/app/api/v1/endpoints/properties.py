"""Property list and detail API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from geoalchemy2.functions import ST_X, ST_Y
from geoalchemy2.shape import to_shape

from app.core.database import get_db
from app.models.county import County
from app.models.property import Property
from app.models.indicator import PropertyIndicator
from app.models.score import PropertyScore
from app.schemas.property import (
    FilterParams,
    PropertyDetail,
    PropertyListItem,
    PropertyListResponse,
    ScoreSchema,
    IndicatorSchema,
    OwnerSchema,
    ListingSchema,
    CountySchema,
)

router = APIRouter(prefix="/properties", tags=["properties"])
counties_router = APIRouter(prefix="/counties", tags=["counties"])


@counties_router.get("")
async def list_counties(db: AsyncSession = Depends(get_db)):
    """Return counties that have at least one property, with counts."""
    stmt = (
        select(
            County.fips_code,
            County.name,
            County.state_abbr,
            func.count(Property.id).label("property_count"),
        )
        .join(Property, Property.county_id == County.id)
        .group_by(County.fips_code, County.name, County.state_abbr)
        .having(func.count(Property.id) > 0)
        .order_by(func.count(Property.id).desc())
    )
    result = await db.execute(stmt)
    rows = result.all()
    return [
        {"fips": r.fips_code, "name": r.name, "state_abbr": r.state_abbr, "count": r.property_count}
        for r in rows
    ]


@router.get("", response_model=PropertyListResponse)
async def list_properties(
    params: FilterParams = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """
    List properties with filtering, sorting, and pagination.
    Returns lightweight PropertyListItem objects suitable for the map/list view.
    """
    filters = params.to_filter_dict()
    offset = (params.page - 1) * params.page_size

    # Build the base query (async version)
    stmt = _build_list_query(filters, limit=params.page_size, offset=offset)
    count_stmt = _build_count_query(filters)

    result = await db.execute(stmt)
    properties = result.scalars().unique().all()

    count_result = await db.execute(count_stmt)
    total = count_result.scalar() or 0

    items = [_to_list_item(p) for p in properties]

    return PropertyListResponse(
        total=total,
        page=params.page,
        page_size=params.page_size,
        results=items,
    )


@router.get("/{property_id}", response_model=PropertyDetail)
async def get_property(property_id: int, db: AsyncSession = Depends(get_db)):
    """Get full property detail including all indicators, owners, listings."""
    stmt = (
        select(Property)
        .options(
            selectinload(Property.score),
            selectinload(Property.indicators),
            selectinload(Property.owners),
            selectinload(Property.listings),
            selectinload(Property.county),
        )
        .where(Property.id == property_id)
    )
    result = await db.execute(stmt)
    prop = result.scalar_one_or_none()

    if prop is None:
        raise HTTPException(status_code=404, detail="Property not found")

    return _to_detail(prop)


@router.get("/map/points")
async def map_points(
    params: FilterParams = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """
    Lightweight endpoint for map marker rendering.
    Returns only id, lat, lng, and score_tier for up to 25000 properties.
    Used by MapLibre GL to render markers efficiently.
    """
    filters = params.to_filter_dict()

    stmt = (
        select(
            Property.id,
            ST_X(Property.location).label("lng"),
            ST_Y(Property.location).label("lat"),
            Property.address_line1,
            Property.address_city,
            Property.address_state,
            Property.address_zip,
            PropertyScore.score_tier,
            PropertyScore.total_score,
        )
        .outerjoin(Property.score)
        .where(Property.location.isnot(None))
        .limit(25000)
    )

    conditions = _build_conditions(filters, stmt)
    if conditions:
        from sqlalchemy import and_
        stmt = stmt.where(and_(*conditions))

    result = await db.execute(stmt)
    rows = result.all()

    def _addr(row) -> str:
        city_state = ", ".join(filter(None, [row.address_city, row.address_state, row.address_zip]))
        parts = [p for p in [row.address_line1, city_state] if p]
        return " — ".join(parts)

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [row.lng, row.lat]},
                "properties": {
                    "id": row.id,
                    "tier": row.score_tier or "cold",
                    "score": float(row.total_score or 0),
                    "address": _addr(row),
                },
            }
            for row in rows
            if row.lat and row.lng
        ],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_list_query(filters: dict, limit: int, offset: int):
    from sqlalchemy import and_
    from app.models.county import County

    stmt = (
        select(Property)
        .options(
            selectinload(Property.score),
            selectinload(Property.indicators),
            selectinload(Property.county),
        )
    )

    conditions = _build_conditions(filters, stmt)
    if conditions:
        stmt = stmt.where(and_(*conditions))

    stmt = stmt.outerjoin(Property.score).order_by(
        PropertyScore.total_score.desc().nullslast()
    )

    return stmt.limit(limit).offset(offset)


def _build_count_query(filters: dict):
    from sqlalchemy import and_

    stmt = select(func.count(Property.id))
    conditions = _build_conditions(filters, stmt)
    if conditions:
        stmt = stmt.where(and_(*conditions))
    return stmt


def _build_conditions(filters: dict, stmt) -> list:
    from app.models.county import County

    conditions = []

    if state := filters.get("state"):
        conditions.append(Property.address_state == state.upper())

    if county_fips := filters.get("county_fips"):
        fips_list = county_fips if isinstance(county_fips, list) else [county_fips]
        county_subq = (
            select(County.id).where(County.fips_code.in_(fips_list)).scalar_subquery()
        )
        conditions.append(Property.county_id.in_(county_subq))

    if zip_codes := filters.get("zip_codes"):
        conditions.append(Property.address_zip.in_(zip_codes))

    if prop_type := filters.get("property_type"):
        conditions.append(Property.property_type == prop_type)

    if indicator_types := filters.get("indicator_types"):
        for itype in indicator_types:
            subq = (
                select(PropertyIndicator.property_id)
                .where(
                    PropertyIndicator.indicator_type == itype,
                    PropertyIndicator.status == "active",
                )
                .scalar_subquery()
            )
            conditions.append(Property.id.in_(subq))

    if score_min := filters.get("score_min"):
        subq = select(PropertyScore.property_id).where(PropertyScore.total_score >= score_min).scalar_subquery()
        conditions.append(Property.id.in_(subq))

    if score_tier := filters.get("score_tier"):
        subq = select(PropertyScore.property_id).where(PropertyScore.score_tier == score_tier).scalar_subquery()
        conditions.append(Property.id.in_(subq))


    return conditions


def _extract_coords(location) -> tuple[float | None, float | None]:
    if location is None:
        return None, None
    try:
        pt = to_shape(location)
        return pt.y, pt.x  # lat, lng
    except Exception:
        return None, None


def _to_list_item(prop: Property) -> PropertyListItem:
    active_types = [i.indicator_type for i in prop.indicators if i.status == "active"]
    lat, lng = _extract_coords(prop.location)
    return PropertyListItem(
        id=prop.id,
        address_raw=prop.address_raw,
        address_line1=prop.address_line1,
        address_city=prop.address_city,
        address_state=prop.address_state,
        address_zip=prop.address_zip,
        property_type=prop.property_type,
        assessed_value=prop.assessed_value,
        score=ScoreSchema.model_validate(prop.score) if prop.score else None,
        active_indicator_types=active_types,
        lat=lat,
        lng=lng,
    )


def _to_detail(prop: Property) -> PropertyDetail:
    lat, lng = _extract_coords(prop.location)
    return PropertyDetail(
        id=prop.id,
        apn=prop.apn,
        address_raw=prop.address_raw,
        address_line1=prop.address_line1,
        address_city=prop.address_city,
        address_state=prop.address_state,
        address_zip=prop.address_zip,
        property_type=prop.property_type,
        year_built=prop.year_built,
        sqft=prop.sqft,
        bedrooms=prop.bedrooms,
        bathrooms=float(prop.bathrooms) if prop.bathrooms else None,
        lot_size_sqft=prop.lot_size_sqft,
        zoning=prop.zoning,
        assessed_value=prop.assessed_value,
        market_value=prop.market_value,
        last_sale_date=prop.last_sale_date,
        last_sale_price=prop.last_sale_price,
        data_source=prop.data_source,
        created_at=prop.created_at,
        updated_at=prop.updated_at,
        county=CountySchema.model_validate(prop.county) if prop.county else None,
        owners=[OwnerSchema.model_validate(o) for o in prop.owners],
        indicators=[IndicatorSchema.model_validate(i) for i in prop.indicators],
        score=ScoreSchema.model_validate(prop.score) if prop.score else None,
        listings=[ListingSchema.model_validate(l) for l in prop.listings],
        lat=lat,
        lng=lng,
    )
