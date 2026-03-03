from datetime import date, datetime
from sqlalchemy import BigInteger, Date, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base

# Allowed indicator types
INDICATOR_TYPES = {
    "pre_foreclosure",   # NOD filed
    "foreclosure",       # Auction scheduled
    "tax_delinquent",    # Unpaid property taxes
    "code_violation",    # Municipal code violation
    "probate",           # Estate/probate filing
    "lien",              # Mechanic's, HOA, judgment lien
    "eviction",          # Unlawful detainer filing
    "vacant",            # Vacancy confirmed
    "price_reduction",   # MLS price drop
    "days_on_market",    # Extended listing
    "expired_listing",   # Listing expired without sale
    "absentee_owner",    # Owner mailing != property address
}


class PropertyIndicator(Base):
    __tablename__ = "property_indicators"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    property_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("properties.id", ondelete="CASCADE"), nullable=False
    )
    indicator_type: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="active")
    # active | resolved | expired

    source: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text)
    amount_cents: Mapped[int | None] = mapped_column(BigInteger)  # tax debt, lien amt
    filing_date: Mapped[date | None] = mapped_column(Date)
    expiry_date: Mapped[date | None] = mapped_column(Date)
    case_number: Mapped[str | None] = mapped_column(String(100))
    raw_data: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    property: Mapped["Property"] = relationship(back_populates="indicators")  # noqa: F821

    __table_args__ = (
        Index("idx_indicators_property", "property_id"),
        Index("idx_indicators_type", "indicator_type"),
        Index("idx_indicators_status", "status"),
        Index("idx_indicators_filing_date", "filing_date"),
    )
