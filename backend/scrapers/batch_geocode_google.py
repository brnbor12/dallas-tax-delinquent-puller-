"""
Google Maps batch geocoder — fills in coordinates for properties
that the Census Bureau batch geocoder could not resolve.

Runs 50 concurrent requests with a brief delay between batches.
At ~50 req/s this processes 28k addresses in under 10 minutes.

Usage:
    docker compose exec api python -m scrapers.batch_geocode_google
    docker compose exec api python -m scrapers.batch_geocode_google --county-fips 48113
    docker compose exec api python -m scrapers.batch_geocode_google --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import os
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import httpx
import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.core.config import settings

logger = structlog.get_logger(__name__)

CONCURRENCY = 50          # parallel Google requests
BATCH_DB_SIZE = 500       # rows committed per DB write
REDIS_TTL = 30 * 24 * 3600


def _get_redis():
    import redis
    return redis.from_url(settings.redis_url, decode_responses=True)


def _cache_key(address: str) -> str:
    h = hashlib.md5(address.lower().strip().encode()).hexdigest()
    return f"geocode:{h}"


async def _geocode_one(
    client: httpx.AsyncClient,
    prop_id: int,
    address: str,
    api_key: str,
) -> tuple[int, float | None, float | None]:
    """Geocode one address via Google Maps. Returns (prop_id, lat, lng)."""
    try:
        resp = await client.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": address, "key": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("results"):
            result = data["results"][0]
            loc_type = result.get("geometry", {}).get("location_type", "")
            if loc_type in ("ROOFTOP", "RANGE_INTERPOLATED"):
                loc = result["geometry"]["location"]
                return prop_id, loc["lat"], loc["lng"]
    except Exception as exc:
        logger.debug("google_geocode_one_failed", prop_id=prop_id, error=str(exc)[:80])
    return prop_id, None, None


async def run_google_batch(
    rows: list[tuple[int, str]],
    api_key: str,
    concurrency: int = CONCURRENCY,
) -> dict[int, tuple[float, float]]:
    """Geocode a list of (prop_id, address) rows concurrently."""
    semaphore = asyncio.Semaphore(concurrency)
    results: dict[int, tuple[float, float]] = {}

    async def bounded(prop_id, address):
        async with semaphore:
            pid, lat, lng = await _geocode_one(client, prop_id, address, api_key)
            if lat is not None:
                results[pid] = (lat, lng)

    async with httpx.AsyncClient() as client:
        tasks = [bounded(pid, addr) for pid, addr in rows]
        await asyncio.gather(*tasks)

    return results


def run_batch_geocode_google(
    session: Session,
    county_fips: str | None = None,
    dry_run: bool = False,
) -> dict:
    api_key = settings.google_maps_api_key
    if not api_key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is not set in settings")

    stats = {"total": 0, "geocoded": 0, "failed": 0}

    county_filter = "AND c.fips_code = :fips" if county_fips else ""
    query = text(f"""
        SELECT p.id, p.address_raw
        FROM properties p
        JOIN counties c ON c.id = p.county_id
        WHERE p.location IS NULL
          AND p.address_raw IS NOT NULL
          AND p.address_raw NOT LIKE 'Parcel%'
          AND length(p.address_raw) > 10
          {county_filter}
        ORDER BY p.id
    """)

    params: dict = {}
    if county_fips:
        params["fips"] = county_fips

    all_rows = session.execute(query, params).fetchall()
    stats["total"] = len(all_rows)
    logger.info("google_batch_start", total=stats["total"], dry_run=dry_run)

    if dry_run:
        print(f"[DRY RUN] Would geocode {stats['total']} addresses via Google Maps")
        return stats

    try:
        redis_client = _get_redis()
    except Exception:
        redis_client = None
        logger.warning("google_batch_redis_unavailable")

    # Process in chunks to allow incremental DB commits
    chunk_size = BATCH_DB_SIZE * 5  # geocode 2500 at a time, commit every 500

    for chunk_start in range(0, len(all_rows), chunk_size):
        chunk = all_rows[chunk_start: chunk_start + chunk_size]
        chunk_num = chunk_start // chunk_size + 1
        total_chunks = (len(all_rows) + chunk_size - 1) // chunk_size

        logger.info(
            "google_batch_chunk",
            chunk=chunk_num,
            of=total_chunks,
            size=len(chunk),
        )

        geocoded = asyncio.run(run_google_batch(chunk, api_key))

        # Cache results in Redis
        if redis_client and geocoded:
            id_to_addr = {row[0]: row[1] for row in chunk}
            pipe = redis_client.pipeline()
            for pid, (lat, lng) in geocoded.items():
                addr = id_to_addr.get(pid, "")
                if addr:
                    pipe.setex(
                        _cache_key(addr), REDIS_TTL,
                        json.dumps({"lat": lat, "lng": lng}),
                    )
            pipe.execute()

        # Bulk UPDATE in sub-batches
        geocoded_items = list(geocoded.items())
        for db_start in range(0, len(geocoded_items), BATCH_DB_SIZE):
            sub = geocoded_items[db_start: db_start + BATCH_DB_SIZE]
            if not sub:
                continue
            value_rows = ", ".join(
                f"({pid}, ST_SetSRID(ST_MakePoint({lng}, {lat}), 4326))"
                for pid, (lat, lng) in sub
            )
            session.execute(
                text(f"""
                    UPDATE properties AS p
                    SET location = v.geom
                    FROM (VALUES {value_rows}) AS v(id, geom)
                    WHERE p.id = v.id
                """)
            )
            session.commit()

        stats["geocoded"] += len(geocoded)
        stats["failed"] += len(chunk) - len(geocoded)
        pct = stats["geocoded"] / max(stats["total"], 1) * 100
        logger.info(
            "google_batch_chunk_done",
            geocoded_this_chunk=len(geocoded),
            total_geocoded=stats["geocoded"],
            pct=f"{pct:.1f}%",
        )

    logger.info("google_batch_complete", **stats)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google Maps batch geocoder")
    parser.add_argument("--county-fips", type=str, default=None,
                        help="Limit to a specific county FIPS code")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count addresses without geocoding")
    args = parser.parse_args()

    db_url = str(settings.database_url).replace("+asyncpg", "")
    engine = create_engine(db_url, pool_size=2, max_overflow=0)

    with Session(engine) as session:
        stats = run_batch_geocode_google(
            session,
            county_fips=args.county_fips,
            dry_run=args.dry_run,
        )
        pct = stats["geocoded"] / max(stats["total"], 1) * 100
        print(f"\nResults: {stats}")
        print(f"Geocoded {stats['geocoded']}/{stats['total']} ({pct:.1f}%)")
