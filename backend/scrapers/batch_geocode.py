"""
Batch geocoder — fills in coordinates for properties imported without geocoding.

Uses the US Census Bureau batch geocoding endpoint (free, no key, no rate limit):
  https://geocoding.geo.census.gov/geocoder/locations/addressbatch

The Census API accepts up to 10,000 addresses per POST as a CSV file and returns
matched coordinates in bulk. For 79k properties this is ~8 requests (~10 min total).

Usage:
    docker compose exec api python -m scrapers.batch_geocode
    docker compose exec api python -m scrapers.batch_geocode --batch-size 5000
    docker compose exec api python -m scrapers.batch_geocode --county-fips 48113
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import os
import time
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import httpx
import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.core.config import settings

logger = structlog.get_logger(__name__)

CENSUS_BATCH_URL = "https://geocoding.geo.census.gov/geocoder/locations/addressbatch"
BATCH_SIZE = 9_500  # Census max is 10,000; leave a small buffer
REDIS_TTL = 30 * 24 * 3600  # 30 days


def _get_redis():
    import redis
    return redis.from_url(settings.redis_url, decode_responses=True)


def _cache_key(address: str) -> str:
    h = hashlib.md5(address.lower().strip().encode()).hexdigest()
    return f"geocode:{h}"


def _cache_results(redis_client, results: dict[int, tuple[float, float]],
                   id_to_address: dict[int, str]) -> None:
    """Cache geocode results in Redis so the regular geocoder can reuse them."""
    pipe = redis_client.pipeline()
    for prop_id, (lat, lng) in results.items():
        address = id_to_address.get(prop_id, "")
        if address:
            key = _cache_key(address)
            pipe.setex(key, REDIS_TTL, json.dumps({"lat": lat, "lng": lng}))
    pipe.execute()


def geocode_batch_census(
    rows: list[tuple[int, str]]
) -> dict[int, tuple[float, float]]:
    """
    Submit one batch to the Census geocoder.

    rows: list of (property_id, address_raw)
    Returns: {property_id: (lat, lng)} for successfully geocoded rows.
    """
    # Build CSV: ID, street, city, state, zip
    # We pass the full address as the street field (one-line address works fine)
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    for prop_id, address in rows:
        # Census batch CSV format: Unique ID, Street address, City, State, ZIP
        # We can pass the full address in the street column; city/state/zip can be blank
        writer.writerow([prop_id, address, "", "", ""])
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    results: dict[int, tuple[float, float]] = {}
    try:
        with httpx.Client(timeout=httpx.Timeout(connect=30, read=300, write=60, pool=30)) as client:
            resp = client.post(
                CENSUS_BATCH_URL,
                data={
                    "benchmark": "Public_AR_Current",
                    "returntype": "locations",
                },
                files={"addressFile": ("addresses.csv", csv_bytes, "text/csv")},
            )
            resp.raise_for_status()

        # Parse response CSV:
        # ID,InputAddress,MatchIndicator,MatchType,OutputAddress,Coords,TigerLineID,Side
        reader = csv.reader(io.StringIO(resp.text))
        for row in reader:
            if len(row) < 6:
                continue
            prop_id_str, _, match_indicator, _, _, coords_str = row[:6]
            if match_indicator.strip().upper() != "MATCH":
                continue
            try:
                prop_id = int(prop_id_str.strip())
                lng_str, lat_str = coords_str.strip().split(",")
                results[prop_id] = (float(lat_str), float(lng_str))
            except (ValueError, IndexError):
                continue

    except httpx.HTTPError as exc:
        logger.error("census_batch_failed", error=str(exc))

    return results


def run_batch_geocode(
    session: Session,
    batch_size: int = BATCH_SIZE,
    county_fips: str | None = None,
    limit: int | None = None,
) -> dict:
    stats = {"total": 0, "geocoded": 0, "failed": 0, "batches": 0}

    # Build WHERE clause
    county_filter = ""
    if county_fips:
        county_filter = "AND c.fips_code = :fips"

    query = text(f"""
        SELECT p.id, p.address_raw
        FROM properties p
        JOIN counties c ON c.id = p.county_id
        WHERE p.location IS NULL
          AND p.address_raw IS NOT NULL
          AND length(p.address_raw) > 5
          {county_filter}
        ORDER BY p.id
        {"LIMIT :lim" if limit else ""}
    """)

    params: dict = {}
    if county_fips:
        params["fips"] = county_fips
    if limit:
        params["lim"] = limit

    all_rows = session.execute(query, params).fetchall()
    stats["total"] = len(all_rows)
    logger.info("batch_geocode_start", total=stats["total"], batch_size=batch_size)

    try:
        redis_client = _get_redis()
    except Exception:
        redis_client = None
        logger.warning("batch_geocode_redis_unavailable")

    id_to_address = {row[0]: row[1] for row in all_rows}

    for batch_start in range(0, len(all_rows), batch_size):
        batch = all_rows[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(all_rows) + batch_size - 1) // batch_size

        logger.info(
            "batch_geocode_submitting",
            batch=batch_num,
            of=total_batches,
            size=len(batch),
        )

        results = geocode_batch_census(batch)
        stats["batches"] += 1

        if not results:
            logger.warning("batch_geocode_no_results", batch=batch_num)
            stats["failed"] += len(batch)
            time.sleep(2)
            continue

        # Cache in Redis
        if redis_client:
            try:
                _cache_results(redis_client, results, id_to_address)
            except Exception as exc:
                logger.warning("batch_geocode_cache_failed", error=str(exc))

        # Update DB in a single statement
        # Build a VALUES list for a bulk UPDATE
        value_rows = ", ".join(
            f"({prop_id}, ST_SetSRID(ST_MakePoint({lng}, {lat}), 4326))"
            for prop_id, (lat, lng) in results.items()
        )
        if value_rows:
            session.execute(
                text(f"""
                    UPDATE properties AS p
                    SET location = v.geom
                    FROM (VALUES {value_rows}) AS v(id, geom)
                    WHERE p.id = v.id
                """)
            )
            session.commit()

        geocoded_count = len(results)
        failed_count = len(batch) - geocoded_count
        stats["geocoded"] += geocoded_count
        stats["failed"] += failed_count

        logger.info(
            "batch_geocode_batch_done",
            batch=batch_num,
            geocoded=geocoded_count,
            failed=failed_count,
            total_geocoded=stats["geocoded"],
        )

        # Brief pause between batches to be polite to the Census API
        if batch_start + batch_size < len(all_rows):
            time.sleep(2)

    logger.info("batch_geocode_complete", **stats)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch geocode ungeooded properties")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Addresses per Census API request (max 10000, default {BATCH_SIZE})")
    parser.add_argument("--county-fips", type=str, default=None,
                        help="Limit to a specific county FIPS code (e.g. 48113)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap total addresses to process (for testing)")
    args = parser.parse_args()

    db_url = str(settings.database_url).replace("+asyncpg", "")
    engine = create_engine(db_url, pool_size=2, max_overflow=0)

    with Session(engine) as session:
        stats = run_batch_geocode(
            session,
            batch_size=args.batch_size,
            county_fips=args.county_fips,
            limit=args.limit,
        )
        pct = stats["geocoded"] / max(stats["total"], 1) * 100
        print(f"\nResults: {stats}")
        print(f"Geocoded {stats['geocoded']}/{stats['total']} ({pct:.1f}%)")
