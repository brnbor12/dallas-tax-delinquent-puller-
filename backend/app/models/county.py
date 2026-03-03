from datetime import datetime
from sqlalchemy import BigInteger, Boolean, DateTime, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.core.database import Base


class County(Base):
    __tablename__ = "counties"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    fips_code: Mapped[str] = mapped_column(String(5), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    state_abbr: Mapped[str] = mapped_column(String(2), nullable=False)
    state_fips: Mapped[str] = mapped_column(String(2), nullable=False)
    has_scraper: Mapped[bool] = mapped_column(Boolean, default=False)
    scraper_config: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    properties: Mapped[list["Property"]] = relationship(back_populates="county")  # noqa: F821
    scrape_jobs: Mapped[list["ScrapeJob"]] = relationship(back_populates="county")  # noqa: F821
