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


def _start_run(session, scraper_key: str, celery_task_id: str | None = None) -> int:
    """Create a scrape_runs record and return its id."""
    from app.models.scrape_job import ScrapeRun
    run = ScrapeRun(
        scraper_key=scraper_key,
        celery_task_id=celery_task_id,
        status="running",
        started_at=datetime.now(timezone.utc),
    )
    session.add(run)
    session.commit()
    return run.id


def _complete_run(session, run_id: int, found: int, upserted: int, failed: int):
    """Mark a scrape_runs record as completed."""
    from app.models.scrape_job import ScrapeRun
    run = session.get(ScrapeRun, run_id)
    if run:
        run.status = "completed"
        run.records_found = found
        run.records_upserted = upserted
        run.records_failed = failed
        run.completed_at = datetime.now(timezone.utc)
        session.commit()


def _fail_run(session, run_id: int, error: str, found: int = 0, upserted: int = 0, failed: int = 0):
    """Mark a scrape_runs record as failed."""
    from app.models.scrape_job import ScrapeRun
    run = session.get(ScrapeRun, run_id)
    if run:
        run.status = "failed"
        run.error_message = error[:2000]
        run.records_found = found
        run.records_upserted = upserted
        run.records_failed = failed
        run.completed_at = datetime.now(timezone.utc)
        session.commit()


@app.task(bind=True, base=ScrapeTask, max_retries=3, default_retry_delay=300)
def run_county_api_scraper(self, scraper_key: str, job_id: int | None = None, **kwargs):
    """
    Run a county API scraper by key.
    Fetches records and passes them to the ingestor.
    Bulk-rescores affected properties once at the end (not per-record).
    """
    from scrapers.registry import get_scraper
    from scrapers.ingestor import ingest_record, IngestResult
    from tasks.score_tasks import bulk_rescore_properties

    result = IngestResult()
    run_id = _start_run(self.session, scraper_key, celery_task_id=self.request.id)

    try:
        scraper = get_scraper(scraper_key, config=kwargs.get("config"))
        logger.info("[%s] scraper started", scraper_key)

        _NO_GEOCODE_KEYS = {
            "tx_dallas_lis_pendens",
            "tx_dallas_probate",
            "tx_dallas_divorce",
            "tx_dallas_eviction_fed",
            "fl_hillsborough_lis_pendens",
            "fl_polk_lis_pendens",
            "fl_polk_official_records",
            "tx_dallas_tax_delinquent",
            "ca_la_tax_delinquent",
            "fl_pinellas_lis_pendens",
            "fl_pasco_lis_pendens",
            "fl_pasco_official_records",
            "fl_pinellas_court_records",
        }
        skip_geocode = scraper_key in _NO_GEOCODE_KEYS

        async def _run():
            affected_pids = set()

            async for record in scraper.fetch_records(**kwargs):
                result.found += 1
                property_id = ingest_record(self.session, record, geocode=not skip_geocode)
                if property_id is not None:
                    result.upserted += 1
                    affected_pids.add(property_id)
                else:
                    result.failed += 1

                if result.found % 100 == 0:
                    self.session.commit()
                    logger.info("[%s] progress: %d found", scraper_key, result.found)

            return affected_pids

        affected_pids = asyncio.run(_run())
        self.session.commit()

        logger.info(
            "[%s] completed — found=%d upserted=%d failed=%d",
            scraper_key, result.found, result.upserted, result.failed,
        )
        _complete_run(self.session, run_id, result.found, result.upserted, result.failed)

        # Bulk rescore only if we actually ingested new/updated records
        if result.upserted > 0 and affected_pids:
            bulk_rescore_properties.delay(list(affected_pids))
            logger.info("[%s] enqueued bulk rescore for %d properties", scraper_key, len(affected_pids))
        else:
            logger.info("[%s] no new records — skipping rescore", scraper_key)

        # Chain enrichment after OR scraper finishes
        if scraper_key == "fl_polk_official_records" and result.upserted > 0:
            from tasks.enrichment_tasks import enrich_polk_official_records
            enrich_polk_official_records.delay()
            logger.info("[%s] chained enrichment task", scraper_key)

        return {"found": result.found, "upserted": result.upserted, "failed": result.failed}

    except Exception as exc:
        self.session.rollback()
        _fail_run(self.session, run_id, str(exc), result.found, result.upserted, result.failed)
        logger.error("[%s] failed: %s", scraper_key, exc)
        raise self.retry(exc=exc)


@app.task(bind=True, base=ScrapeTask, queue="web_scrape", max_retries=3, default_retry_delay=300)
def run_web_scraper(self, scraper_key: str, **kwargs):
    """Browser-based scrapers — runs on playwright-worker via web_scrape queue."""
    from scrapers.registry import get_scraper
    from scrapers.ingestor import ingest_record, IngestResult
    from tasks.score_tasks import bulk_rescore_properties

    result = IngestResult()
    run_id = _start_run(self.session, scraper_key, celery_task_id=self.request.id)

    try:
        scraper = get_scraper(scraper_key, config=kwargs.get("config"))
        logger.info("[%s] scraper started", scraper_key)

        import asyncio
        async def _run():
            affected_pids = set()

            async for record in scraper.fetch_records(**kwargs):
                result.found += 1
                property_id = ingest_record(self.session, record, geocode=True)
                if property_id is not None:
                    result.upserted += 1
                    affected_pids.add(property_id)
                else:
                    result.failed += 1
                if result.found % 100 == 0:
                    self.session.commit()
                    logger.info("[%s] progress: %d found", scraper_key, result.found)

            return affected_pids

        affected_pids = asyncio.run(_run())
        self.session.commit()

        logger.info("[%s] completed — found=%d upserted=%d failed=%d",
                    scraper_key, result.found, result.upserted, result.failed)
        _complete_run(self.session, run_id, result.found, result.upserted, result.failed)

        if result.upserted > 0 and affected_pids:
            bulk_rescore_properties.delay(list(affected_pids))
            logger.info("[%s] enqueued bulk rescore for %d properties", scraper_key, len(affected_pids))
        else:
            logger.info("[%s] no new records — skipping rescore", scraper_key)

        return {"found": result.found, "upserted": result.upserted, "failed": result.failed}

    except Exception as exc:
        self.session.rollback()
        _fail_run(self.session, run_id, str(exc), result.found, result.upserted, result.failed)
        logger.error("[%s] failed: %s", scraper_key, exc)
        raise self.retry(exc=exc)

@app.task(bind=True, base=ScrapeTask, max_retries=3, default_retry_delay=300)
def run_mls_scraper(self, scraper_key: str, **kwargs):
    """Alias for MLS scrapers (routed to mls queue, stricter rate limits)."""
    return run_county_api_scraper.apply(args=[scraper_key], kwargs=kwargs)
