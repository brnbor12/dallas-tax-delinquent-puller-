"""Celery tasks for running scrapers."""

from __future__ import annotations

import asyncio
import structlog
from datetime import datetime, timezone

from celery import Task

from tasks.celery_app import app

logger = structlog.get_logger(__name__)


class ScrapeTask(Task):
    """Base task class with DB session management."""
    _session = None

    @property
    def session(self):
        if self._session is None:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            from app.core.config import settings

            # Use sync engine for Celery (not async)
            sync_url = settings.database_url.replace("+asyncpg", "+psycopg2")
            engine = create_engine(sync_url, pool_size=2, max_overflow=0)
            self._session = sessionmaker(bind=engine)()
        return self._session


@app.task(bind=True, base=ScrapeTask, max_retries=3, default_retry_delay=300)
def run_county_api_scraper(self, scraper_key: str, job_id: int | None = None, **kwargs):
    """
    Run a county API scraper by key.
    Fetches records and passes them to the ingestor.
    """
    from scrapers.registry import get_scraper
    from scrapers.ingestor import ingest_record, IngestResult

    result = IngestResult()

    try:
        scraper = get_scraper(scraper_key, config=kwargs.get("config"))
        logger.info("[%s] scraper started", scraper_key)

        # Tax roll scrapers have APNs for every record — skip geocoding to avoid
        # blocking for hours on tens-of-thousands of new addresses. A separate
        # batch geocode job fills in coordinates afterwards.
        skip_geocode = "tax" in scraper_key

        # Run async scraper in a sync context
        async def _run():
            async for record in scraper.fetch_records(**kwargs):
                result.found += 1
                success = ingest_record(self.session, record, geocode=not skip_geocode)
                if success:
                    result.upserted += 1
                else:
                    result.failed += 1

                # Commit every 100 records to avoid large transactions
                if result.found % 100 == 0:
                    self.session.commit()
                    logger.info("[%s] progress: %d found", scraper_key, result.found)

        asyncio.run(_run())
        self.session.commit()

        logger.info(
            "[%s] completed — found=%d upserted=%d failed=%d",
            scraper_key, result.found, result.upserted, result.failed,
        )

        # Trigger nightly score recalculation for affected properties
        nightly_score_decay.delay()

        return {"found": result.found, "upserted": result.upserted, "failed": result.failed}

    except Exception as exc:
        self.session.rollback()
        logger.error("[%s] failed: %s", scraper_key, exc)
        raise self.retry(exc=exc)


@app.task(bind=True, base=ScrapeTask, max_retries=3, default_retry_delay=300)
def run_web_scraper(self, scraper_key: str, **kwargs):
    """Alias for browser-based scrapers (routed to web_scrape queue)."""
    return run_county_api_scraper.apply(args=[scraper_key], kwargs=kwargs)


@app.task(bind=True, base=ScrapeTask, max_retries=3, default_retry_delay=300)
def run_mls_scraper(self, scraper_key: str, **kwargs):
    """Alias for MLS scrapers (routed to mls queue, stricter rate limits)."""
    return run_county_api_scraper.apply(args=[scraper_key], kwargs=kwargs)


from tasks.score_tasks import nightly_score_decay  # noqa: E402 (circular-safe)
