"""Admin endpoints for managing scrape jobs."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.core.database import get_db
from app.models.scrape_job import ScrapeJob, ScrapeRun

router = APIRouter(prefix="/admin/scrape-jobs", tags=["admin"])


@router.get("")
async def list_jobs(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ScrapeJob).order_by(ScrapeJob.created_at.desc()))
    jobs = result.scalars().all()
    return [
        {
            "id": j.id,
            "job_name": j.job_name,
            "job_type": j.job_type,
            "is_active": j.is_active,
            "schedule_cron": j.schedule_cron,
            "last_run_at": j.last_run_at,
        }
        for j in jobs
    ]


@router.post("/{job_id}/trigger")
async def trigger_job(job_id: int, db: AsyncSession = Depends(get_db)):
    """Manually trigger a scrape job immediately."""
    result = await db.execute(select(ScrapeJob).where(ScrapeJob.id == job_id))
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    from tasks.scrape_tasks import run_county_api_scraper
    scraper_key = job.config.get("scraper_key") if job.config else None
    if not scraper_key:
        raise HTTPException(status_code=400, detail="Job has no scraper_key in config")

    task = run_county_api_scraper.delay(scraper_key=scraper_key, job_id=job_id)
    return {"task_id": task.id, "status": "queued"}


@router.get("/{job_id}/runs")
async def list_runs(job_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ScrapeRun)
        .where(ScrapeRun.job_id == job_id)
        .order_by(ScrapeRun.started_at.desc())
        .limit(20)
    )
    runs = result.scalars().all()
    return [
        {
            "id": r.id,
            "status": r.status,
            "records_found": r.records_found,
            "records_upserted": r.records_upserted,
            "records_failed": r.records_failed,
            "error_message": r.error_message,
            "started_at": r.started_at,
            "completed_at": r.completed_at,
        }
        for r in runs
    ]
