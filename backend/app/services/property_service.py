"""
Property query service — builds SQLAlchemy queries from filter dicts.
Used by both the API endpoints and the export Celery task.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import and_, or_, select, func
from sqlalchemy.orm import Session, selectinload
from geoalchemy2.functions import ST_DWithin, ST_MakePoint

from app.models.indicator import PropertyIndicator
from app.models.property import Property
from app.models.owner import Owner
from app.models.score import PropertyScore


def build_property_query(session: Session, filters: dict, limit: int = 100, offset: int = 0):
    """
    Build and return a SQLAlchemy query for properties matching the given filters.

    Supported filter keys:
      state:           str — e.g. "CA"
      county_fips:     list[str]
      zip_codes:       list[str]
      indicator_types: list[str]
      score_min:       float
      score_tier:      str — hot|warm|cold
      property_type:   str
      year_built_min:  int
      year_built_max:  int
      assessed_min:    int (dollars)
      assessed_max:    int (dollars)
      bedrooms_min:    int
      dom_min:         int (days on market)
      out_of_state_only: bool
      lat/lng/radius_km: float — geographic radius search
    """
    stmt = (
        select(Property)
        .options(
            selectinload(Property.score),
            selectinload(Property.indicators),
            selectinload(Property.owners),
            selectinload(Property.listings),
            selectinload(Property.county),
        )
    )

    conditions = []

    if state := filters.get("state"):
        conditions.append(Property.address_state == state.upper())

    if county_fips := filters.get("county_fips"):
        from app.models.county import County
        stmt = stmt.join(Property.county)
        conditions.append(County.fips_code.in_(county_fips))

    if zip_codes := filters.get("zip_codes"):
        conditions.append(Property.address_zip.in_(zip_codes))

    if prop_type := filters.get("property_type"):
        conditions.append(Property.property_type == prop_type)

    if year_min := filters.get("year_built_min"):
        conditions.append(Property.year_built >= year_min)

    if year_max := filters.get("year_built_max"):
        conditions.append(Property.year_built <= year_max)

    if assessed_min := filters.get("assessed_min"):
        conditions.append(Property.assessed_value >= assessed_min * 100)

    if assessed_max := filters.get("assessed_max"):
        conditions.append(Property.assessed_value <= assessed_max * 100)

    if beds := filters.get("bedrooms_min"):
        conditions.append(Property.bedrooms >= beds)

    # Filter by indicator types — property must have at least one active indicator of each type
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

    # Score filters — join property_scores
    if score_min := filters.get("score_min"):
        stmt = stmt.join(Property.score)
        conditions.append(PropertyScore.total_score >= score_min)

    if score_tier := filters.get("score_tier"):
        if not filters.get("score_min"):
            stmt = stmt.join(Property.score)
        conditions.append(PropertyScore.score_tier == score_tier)

    # Absentee/out-of-state filter
    if filters.get("out_of_state_only"):
        subq = (
            select(Owner.property_id)
            .where(Owner.is_out_of_state == True)  # noqa: E712
            .scalar_subquery()
        )
        conditions.append(Property.id.in_(subq))

    # Geographic radius search
    if lat := filters.get("lat"):
        lng = filters.get("lng")
        radius_km = filters.get("radius_km", 10)
        if lng and Property.location is not None:
            # ST_DWithin with geography type uses meters
            conditions.append(
                ST_DWithin(
                    func.cast(Property.location, type_=None),
                    ST_MakePoint(lng, lat),
                    radius_km * 1000,
                )
            )

    if conditions:
        stmt = stmt.where(and_(*conditions))

    # Sort by score desc by default
    sort_by = filters.get("sort_by", "score_desc")
    if sort_by == "score_desc":
        stmt = stmt.outerjoin(Property.score).order_by(PropertyScore.total_score.desc().nullslast())
    elif sort_by == "filing_date_desc":
        pass  # Complex — handle in future iteration
    elif sort_by == "assessed_value_asc":
        stmt = stmt.order_by(Property.assessed_value.asc().nullslast())

    stmt = stmt.limit(limit).offset(offset)
    return session.execute(stmt).scalars().unique()
