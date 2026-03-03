"""Export endpoint — triggers async CSV generation via Celery."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.property import FilterParams

router = APIRouter(prefix="/export", tags=["export"])


@router.post("")
async def request_export(
    params: FilterParams = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger an async CSV export. Returns an export_id.
    Poll GET /export/{export_id} to retrieve the file when ready.
    """
    from tasks.export_tasks import generate_export

    export_id = str(uuid.uuid4())
    generate_export.delay(
        user_id="anonymous",  # TODO: wire up real user_id from auth
        filters=params.to_filter_dict(),
        export_id=export_id,
    )
    return {"export_id": export_id, "status": "pending"}


@router.get("/{export_id}")
async def download_export(export_id: str):
    """Download a previously requested CSV export."""
    import redis
    from app.core.config import settings

    r = redis.from_url(settings.redis_url)
    csv_bytes = r.get(f"export:{export_id}")

    if csv_bytes is None:
        raise HTTPException(status_code=404, detail="Export not ready or expired")

    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="motivated_sellers_{export_id[:8]}.csv"'
        },
    )
