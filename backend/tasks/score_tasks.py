"""Celery tasks for score calculation."""

from __future__ import annotations

import structlog

from tasks.celery_app import app

logger = structlog.get_logger(__name__)


@app.task(bind=True, max_retries=3)
def recalculate_property_score(self, property_id: int):
    """Recalculate motivated seller score for a single property."""
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import sessionmaker
    from app.core.config import settings
    from app.models.indicator import PropertyIndicator
    from app.models.score import PropertyScore
    from scoring.engine import score_from_orm

    sync_url = settings.database_url.replace("+asyncpg", "+psycopg2")
    engine = create_engine(sync_url)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        indicators = session.execute(
            select(PropertyIndicator).where(PropertyIndicator.property_id == property_id)
        ).scalars().all()

        result = score_from_orm(list(indicators))

        score_row = session.execute(
            select(PropertyScore).where(PropertyScore.property_id == property_id)
        ).scalar_one_or_none()

        if score_row is None:
            score_row = PropertyScore(property_id=property_id)
            session.add(score_row)

        score_row.total_score = result["total_score"]
        score_row.indicator_count = result["indicator_count"]
        score_row.score_breakdown = result["breakdown"]
        score_row.score_tier = result["tier"]
        session.commit()

    logger.info("score_recalculated", property_id=property_id, score=result["total_score"])


@app.task
def nightly_score_decay():
    """
    Recalculate scores for all properties with active indicators.
    Runs nightly to apply recency decay as indicators age.
    """
    from sqlalchemy import create_engine, select, distinct
    from sqlalchemy.orm import sessionmaker
    from app.core.config import settings
    from app.models.indicator import PropertyIndicator

    sync_url = settings.database_url.replace("+asyncpg", "+psycopg2")
    engine = create_engine(sync_url)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        property_ids = session.execute(
            select(distinct(PropertyIndicator.property_id)).where(
                PropertyIndicator.status == "active"
            )
        ).scalars().all()

    logger.info("nightly_score_decay_started", property_count=len(property_ids))

    for pid in property_ids:
        recalculate_property_score.delay(pid)

    logger.info("nightly_score_decay_queued", count=len(property_ids))
