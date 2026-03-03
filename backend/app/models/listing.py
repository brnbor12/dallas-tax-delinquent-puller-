from datetime import date, datetime
from sqlalchemy import BigInteger, Date, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base


class ListingData(Base):
    __tablename__ = "listing_data"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    property_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("properties.id", ondelete="CASCADE"), nullable=False
    )
    source: Mapped[str] = mapped_column(Text, nullable=False)  # attom, redfin, zillow
    listing_id: Mapped[str | None] = mapped_column(Text)
    list_price: Mapped[int | None] = mapped_column(BigInteger)  # cents
    original_price: Mapped[int | None] = mapped_column(BigInteger)  # cents
    days_on_market: Mapped[int | None] = mapped_column(Integer)
    price_reductions: Mapped[int] = mapped_column(Integer, default=0)
    listing_status: Mapped[str | None] = mapped_column(Text)  # active|pending|expired|sold
    listed_date: Mapped[date | None] = mapped_column(Date)
    last_price_cut: Mapped[date | None] = mapped_column(Date)
    description: Mapped[str | None] = mapped_column(Text)
    raw_data: Mapped[dict | None] = mapped_column(JSONB)
    scraped_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    property: Mapped["Property"] = relationship(back_populates="listings")  # noqa: F821

    __table_args__ = (
        Index("idx_listing_property", "property_id"),
        Index("idx_listing_source_id", "source", "listing_id", unique=True,
              postgresql_where="listing_id IS NOT NULL"),
    )
