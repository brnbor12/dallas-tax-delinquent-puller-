"""Celery tasks for generating CSV exports."""

from __future__ import annotations

import csv
import io
import logging

from tasks.celery_app import app

logger = logging.getLogger(__name__)


@app.task(bind=True, max_retries=2, time_limit=300)
def generate_export(self, user_id: str, filters: dict, export_id: str):
    """
    Generate a CSV export for the given filter set.
    Stores the result in Redis with a 1-hour TTL, keyed by export_id.
    """
    import redis
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.core.config import settings
    from app.services.property_service import build_property_query

    sync_url = settings.database_url.replace("+asyncpg", "+psycopg2")
    engine = create_engine(sync_url)
    Session = sessionmaker(bind=engine)

    try:
        with Session() as session:
            query = build_property_query(session, filters, limit=10_000)
            rows = query.all()

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Address", "City", "State", "Zip", "APN", "County",
            "Score", "Tier", "Indicators", "Assessed Value",
            "Owner Name", "Owner Mailing", "Is Absentee",
            "Days On Market", "List Price",
        ])

        for row in rows:
            writer.writerow([
                row.address_line1 or row.address_raw,
                row.address_city, row.address_state, row.address_zip,
                row.apn, row.county.name if row.county else "",
                row.score.total_score if row.score else "",
                row.score.score_tier if row.score else "",
                ",".join(i.indicator_type for i in row.indicators if i.status == "active"),
                f"${row.assessed_value / 100:,.0f}" if row.assessed_value else "",
                row.owners[0].name_raw if row.owners else "",
                row.owners[0].mailing_address if row.owners else "",
                "Yes" if row.owners and row.owners[0].is_absentee else "No",
                row.listings[0].days_on_market if row.listings else "",
                f"${row.listings[0].list_price / 100:,.0f}" if row.listings and row.listings[0].list_price else "",
            ])

        csv_bytes = output.getvalue().encode()

        r = redis.from_url(settings.redis_url)
        r.setex(f"export:{export_id}", 3600, csv_bytes)

        logger.info("export_completed", export_id=export_id, rows=len(rows))
        return {"export_id": export_id, "rows": len(rows)}

    except Exception as exc:
        logger.error("export_failed", export_id=export_id, error=str(exc))
        raise self.retry(exc=exc)
