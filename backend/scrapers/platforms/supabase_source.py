"""
Supabase REST API paged reader.

Used by scrapers that pull from an existing Supabase project rather than
scraping a county website directly.  Handles pagination, retries, and
optional incremental sync via a `since` timestamp filter.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import httpx
import structlog

logger = structlog.get_logger(__name__)

_PAGE_SIZE = 1000
_TIMEOUT = 30.0
_MAX_RETRIES = 3


class SupabasePagedReader:
    """
    Async paged reader for a Supabase REST API table or view.

    Usage::

        reader = SupabasePagedReader(url, key, "gov_leads")
        async for row in reader.paginate(source_type="eq.tax_delinquent"):
            ...  # row is a plain dict
    """

    def __init__(self, url: str, key: str, table: str, page_size: int = _PAGE_SIZE):
        self.url = url.rstrip("/")
        self.key = key
        self.table = table
        self.page_size = page_size
        self._headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    async def paginate(
        self,
        since: datetime | None = None,
        since_field: str = "first_seen_at",
        order_field: str = "id",
        **filters: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Yield all rows matching the given filters, paginating automatically.

        Args:
            since:       If set, only return rows where `since_field >= since`.
            since_field: Column to filter on for incremental sync.
            order_field: Column to order by (must be indexed for stable pagination).
            **filters:   PostgREST filter params, e.g. source_type="eq.tax_delinquent".
        """
        params: dict[str, str] = {
            "order": f"{order_field}.asc",
            "limit": str(self.page_size),
            **filters,
        }
        if since is not None:
            iso = since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
            params[since_field] = f"gte.{iso}"

        offset = 0
        total_yielded = 0

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            while True:
                params["offset"] = str(offset)
                rows = await self._fetch_page(client, params)
                if not rows:
                    break

                for row in rows:
                    yield row
                    total_yielded += 1

                if len(rows) < self.page_size:
                    break  # last page

                offset += self.page_size
                await asyncio.sleep(0.1)  # be polite

        logger.info(
            "supabase_paginate_complete",
            table=self.table,
            total_yielded=total_yielded,
        )

    async def _fetch_page(
        self, client: httpx.AsyncClient, params: dict[str, str]
    ) -> list[dict[str, Any]]:
        endpoint = f"{self.url}/rest/v1/{self.table}"
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = await client.get(endpoint, headers=self._headers, params=params)
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                logger.warning(
                    "supabase_fetch_error",
                    table=self.table,
                    attempt=attempt,
                    error=str(exc),
                )
                if attempt == _MAX_RETRIES:
                    raise
                await asyncio.sleep(2**attempt)
        return []
