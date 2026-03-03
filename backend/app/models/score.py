from datetime import datetime
from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base


class PropertyScore(Base):
    __tablename__ = "property_scores"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    property_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("properties.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    total_score: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False, default=0)
    indicator_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    score_breakdown: Mapped[dict | None] = mapped_column(JSONB)
    score_tier: Mapped[str | None] = mapped_column(Text)  # hot | warm | cold
    last_scored_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    property: Mapped["Property"] = relationship(back_populates="score")  # noqa: F821

    __table_args__ = (
        Index("idx_scores_total", "total_score"),
        Index("idx_scores_tier", "score_tier"),
    )
