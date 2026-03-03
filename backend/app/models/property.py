from datetime import date, datetime
from sqlalchemy import (
    BigInteger, Date, DateTime, ForeignKey, Index, Integer,
    Numeric, SmallInteger, String, Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from geoalchemy2 import Geometry

from app.core.database import Base


class Property(Base):
    __tablename__ = "properties"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    apn: Mapped[str | None] = mapped_column(Text)  # Assessor Parcel Number
    county_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("counties.id"))

    # Address
    address_raw: Mapped[str] = mapped_column(Text, nullable=False)
    address_normalized: Mapped[str | None] = mapped_column(Text)  # For pg_trgm fuzzy matching
    address_line1: Mapped[str | None] = mapped_column(Text)
    address_city: Mapped[str | None] = mapped_column(Text)
    address_state: Mapped[str | None] = mapped_column(String(2))
    address_zip: Mapped[str | None] = mapped_column(String(5))

    # Spatial
    location: Mapped[object | None] = mapped_column(Geometry("POINT", srid=4326))
    parcel_geometry: Mapped[object | None] = mapped_column(Geometry("POLYGON", srid=4326))

    # Property details
    property_type: Mapped[str | None] = mapped_column(Text)  # SFR, MFR, Commercial, Land
    year_built: Mapped[int | None] = mapped_column(SmallInteger)
    sqft: Mapped[int | None] = mapped_column(Integer)
    bedrooms: Mapped[int | None] = mapped_column(SmallInteger)
    bathrooms: Mapped[float | None] = mapped_column(Numeric(3, 1))
    lot_size_sqft: Mapped[int | None] = mapped_column(Integer)
    zoning: Mapped[str | None] = mapped_column(Text)

    # Financials (stored in cents to avoid float issues)
    assessed_value: Mapped[int | None] = mapped_column(BigInteger)
    market_value: Mapped[int | None] = mapped_column(BigInteger)
    last_sale_date: Mapped[date | None] = mapped_column(Date)
    last_sale_price: Mapped[int | None] = mapped_column(BigInteger)

    # Raw source data
    raw_data: Mapped[dict | None] = mapped_column(JSONB)
    data_source: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    county: Mapped["County"] = relationship(back_populates="properties")  # noqa: F821
    owners: Mapped[list["Owner"]] = relationship(back_populates="property", cascade="all, delete-orphan")  # noqa: F821
    indicators: Mapped[list["PropertyIndicator"]] = relationship(back_populates="property", cascade="all, delete-orphan")  # noqa: F821
    score: Mapped["PropertyScore | None"] = relationship(back_populates="property", cascade="all, delete-orphan", uselist=False)  # noqa: F821
    listings: Mapped[list["ListingData"]] = relationship(back_populates="property", cascade="all, delete-orphan")  # noqa: F821

    __table_args__ = (
        Index("idx_properties_location", "location", postgresql_using="gist"),
        Index("idx_properties_county", "county_id"),
        Index("idx_properties_zip", "address_zip"),
        Index("idx_properties_updated", "updated_at"),
        # Unique per APN + county
        Index("idx_properties_apn_county", "apn", "county_id", unique=True,
              postgresql_where="apn IS NOT NULL"),
    )
