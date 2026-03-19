"""
Dallas Obituary → DCAD Owner Match Scraper  (Texas - FIPS 48113)
=================================================================

Source:  obituary_raw table — pre-fetched by the local fetch_obits.py
         script running on a residential IP machine.
Signal:  "probate" — deceased property owner → estate / motivated heirs.

How it works:
  1. Read pending rows from obituary_raw (uploaded by local fetcher).
  2. Normalize each deceased name to LAST FIRST order.
  3. Fuzzy-match against owners.name_raw for Dallas County properties
     using pg_trgm similarity (threshold 0.72).
  4. Yield a "probate" RawIndicatorRecord for each matched property.
  5. Mark matched rows in obituary_raw as processed.

The fetch step is intentionally split to a local machine because
obituary sites block datacenter IPs. The local fetch_obits.py script
uploads raw obituary rows, and this VPS-side job does the matching.
"""

from __future__ import annotations

import re
import unicodedata
import structlog
from datetime import date, datetime, timedelta
from typing import AsyncIterator

from sqlalchemy import text

from app.core.database import AsyncSessionLocal
from scrapers.base import BaseCountyScraper, RawIndicatorRecord

logger = structlog.get_logger(__name__)

COUNTY_FIPS = "48113"
LOOKBACK_DAYS = 30
NAME_SIMILARITY_THRESHOLD = 0.72
MAX_MATCHES_PER_OBIT = 2


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v", "esq", "md", "phd"}
_HONORIFICS = {"mr", "mrs", "ms", "dr", "rev", "pastor"}


def _normalize_name(raw: str) -> str:
    """Normalize a name for pg_trgm matching. Returns uppercase, strips punctuation/suffixes."""
    name = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode()
    name = re.sub(r"[^\w\s]", " ", name)
    tokens = [t.lower() for t in name.split() if t]
    tokens = [t for t in tokens if t not in _SUFFIXES and t not in _HONORIFICS]
    return " ".join(tokens).upper()


def _name_variants(first: str, last: str, full: str) -> list[str]:
    """Return name variants to try against the DCAD owner table."""
    first = first.upper().strip()
    last = last.upper().strip()
    norm = _normalize_name(full) if full else ""
    variants = []
    if last and first:
        variants.append(f"{last} {first}")    # SMITH JOHN  (DCAD typical format)
        variants.append(f"{first} {last}")    # JOHN SMITH
    if norm and norm not in variants:
        variants.append(norm)
    return list(dict.fromkeys(variants))


def _parse_date(raw: str) -> date | None:
    if not raw or not raw.strip():
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y"):
        try:
            return datetime.strptime(raw.strip()[:19], fmt).date()
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# DB queries
# ---------------------------------------------------------------------------

_FETCH_PENDING_SQL = text("""
    SELECT id, decedent_name_raw, first_name, last_name, death_date,
           obituary_publish_date, city, source_url, source
    FROM obituary_raw
    WHERE match_status = 'pending'
      AND scraped_at >= NOW() - INTERVAL ':days days'
    ORDER BY scraped_at DESC
    LIMIT 500
""")

_MATCH_SQL = text("""
    SELECT
        p.id            AS property_id,
        p.address_raw,
        p.apn,
        o.name_raw      AS owner_name,
        similarity(o.name_raw, :search_name) AS sim
    FROM properties p
    JOIN owners o ON o.property_id = p.id
    JOIN counties c ON c.id = p.county_id
    WHERE c.fips_code = :fips
      AND similarity(o.name_raw, :search_name) >= :threshold
    ORDER BY sim DESC
    LIMIT :limit
""")

_MARK_PROCESSED_SQL = text("""
    UPDATE obituary_raw
    SET match_status = :status, matched_at = NOW()
    WHERE id = ANY(:ids)
""")


async def _fetch_pending(session, days_back: int) -> list[dict]:
    result = await session.execute(
        text("""
            SELECT id, decedent_name_raw, first_name, last_name, death_date,
                   obituary_publish_date, city, source_url, source
            FROM obituary_raw
            WHERE match_status = 'pending'
              AND scraped_at >= NOW() - (:days || ' days')::interval
            ORDER BY scraped_at DESC
            LIMIT 500
        """),
        {"days": days_back},
    )
    return [dict(r._mapping) for r in result.fetchall()]


async def _match_owner(session, search_name: str,
                       threshold: float, limit: int) -> list[dict]:
    result = await session.execute(
        _MATCH_SQL,
        {"search_name": search_name, "fips": COUNTY_FIPS,
         "threshold": threshold, "limit": limit},
    )
    return [dict(r._mapping) for r in result.fetchall()]


# ---------------------------------------------------------------------------
# Scraper class
# ---------------------------------------------------------------------------

class DallasObituaryMatchScraper(BaseCountyScraper):
    """
    Reads pre-fetched obituary rows from obituary_raw table and fuzzy-matches
    deceased names against DCAD property owner records.
    Yields a "probate" indicator for each matched property.

    Requires the local fetch_obits.py script to have uploaded rows first.
    """

    county_fips = COUNTY_FIPS
    source_name = "Dallas Obituary → DCAD Owner Match"
    indicator_types = ["probate"]
    rate_limit_per_minute = 120

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        days_back = self.config.get("lookback_days", LOOKBACK_DAYS)
        threshold = self.config.get("similarity_threshold", NAME_SIMILARITY_THRESHOLD)
        max_matches = self.config.get("max_matches_per_obit", MAX_MATCHES_PER_OBIT)

        total_matched = 0
        total_no_match = 0
        processed_ids: list[int] = []

        async with AsyncSessionLocal() as session:
            obituaries = await _fetch_pending(session, days_back)

            if not obituaries:
                logger.info("dallas_obituary_match_no_pending",
                            tip="Run fetch_obits.py locally to upload obituary data")
                return

            logger.info("dallas_obituary_match_start",
                        pending=len(obituaries), threshold=threshold)

            for obit in obituaries:
                obit_id = obit["id"]
                first = obit.get("first_name") or ""
                last = obit.get("last_name") or ""
                full = obit.get("decedent_name_raw") or f"{first} {last}".strip()

                variants = _name_variants(first, last, full)
                seen_property_ids: set[int] = set()
                obit_matched = False

                for search_name in variants:
                    if len(seen_property_ids) >= max_matches:
                        break
                    try:
                        rows = await _match_owner(session, search_name, threshold, max_matches)
                    except Exception as exc:
                        logger.warning("dallas_obituary_db_error",
                                       name=search_name, error=str(exc)[:120])
                        continue

                    for row in rows:
                        pid = row["property_id"]
                        if pid in seen_property_ids or len(seen_property_ids) >= max_matches:
                            continue
                        seen_property_ids.add(pid)
                        obit_matched = True

                        filing_date = obit.get("death_date") or obit.get("obituary_publish_date")

                        record = RawIndicatorRecord(
                            indicator_type="probate",
                            address_raw=row["address_raw"],
                            county_fips=COUNTY_FIPS,
                            apn=row["apn"],
                            owner_name=full,
                            filing_date=filing_date,
                            case_number=f"OBIT-{obit_id}",
                            source_url=obit.get("source_url") or "",
                            raw_payload={
                                "obituary_raw_id": obit_id,
                                "deceased_name": full,
                                "death_date": filing_date.isoformat() if filing_date else None,
                                "city": obit.get("city"),
                                "source": obit.get("source"),
                                "source_url": obit.get("source_url"),
                                "matched_owner": row["owner_name"],
                                "similarity": round(float(row["sim"]), 3),
                                "search_name_used": search_name,
                                "signal_source": "obituary_match",
                            },
                        )

                        if await self.validate_record(record):
                            yield record
                            total_matched += 1
                            logger.info("dallas_obituary_matched",
                                        deceased=full,
                                        owner=row["owner_name"],
                                        address=row["address_raw"],
                                        sim=round(float(row["sim"]), 3))

                processed_ids.append(obit_id)
                if not obit_matched:
                    total_no_match += 1

            # Mark all processed rows (matched or not) so we don't reprocess
            if processed_ids:
                await session.execute(
                    _MARK_PROCESSED_SQL,
                    {"status": "processed", "ids": processed_ids},
                )
                await session.commit()

        logger.info("dallas_obituary_match_complete",
                    total_matched=total_matched,
                    total_no_match=total_no_match,
                    obits_processed=len(processed_ids))
