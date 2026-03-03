"""
One-time backfill: link pre_foreclosure indicators to tax_delinquent properties
using pg_trgm fuzzy address matching (house-number-aware).

Background
----------
The TRW tax-roll import stored 83k+ tax_delinquent indicators with APNs and
normalized addresses ("8687 N CENTRAL EXPY DALLAS").

The NOTS PDF scraper stored pre_foreclosure indicators under *separate* property
rows because:
  - Address strings differed (DRIVE vs DR, ROAD vs RD, etc.)
  - No APN in pre_foreclosure records so APN dedup didn't fire

This script:
1. Backfills address_normalized for any rows that missed the migration.
2. Finds all pre_foreclosure-only properties (no APN) that match a
   tax_delinquent property (with APN) using a two-phase match:
     a. Exact house number (first numeric token must match exactly)
     b. Street name fuzzy similarity >= 0.65 via pg_trgm
3. Moves pre_foreclosure indicators + owner records to the matched property.
4. Deletes the orphaned property rows.
5. Marks affected properties for score recalculation.

Run after migration a1c4e2d9f803:
    docker compose exec api python -m scrapers.backfill_address_merge
    docker compose exec api python -m scrapers.backfill_address_merge --dry-run
    docker compose exec api python -m scrapers.backfill_address_merge --threshold 0.7
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.core.config import settings

logger = structlog.get_logger(__name__)


def _build_match_query(threshold: float) -> str:
    """
    Build the address-match SQL. Uses similarity() >= threshold directly
    (avoids needing SET pg_trgm.similarity_threshold to persist across
    SQLAlchemy execute() calls). The exact house-number join reduces
    the candidate set so a sequential scan on the street portion is fast.
    """
    return f"""
WITH pre_fc AS (
    SELECT DISTINCT
        p.id          AS pre_id,
        p.address_raw AS pre_addr,
        p.address_normalized AS pre_norm,
        p.county_id
    FROM properties p
    JOIN property_indicators pi ON pi.property_id = p.id
    WHERE pi.indicator_type = 'pre_foreclosure'
      AND p.apn IS NULL
      AND p.address_normalized IS NOT NULL
      AND p.address_normalized ~ '^[0-9]{{1,6}} '
      AND p.address_normalized NOT LIKE '%COMMERCE ST%'
      AND p.address_normalized NOT LIKE '/MAILING%'
),
tax_del AS (
    SELECT DISTINCT
        p.id          AS tax_id,
        p.address_raw AS tax_addr,
        p.address_normalized AS tax_norm,
        p.apn,
        p.county_id
    FROM properties p
    JOIN property_indicators pi ON pi.property_id = p.id
    WHERE pi.indicator_type = 'tax_delinquent'
      AND p.apn IS NOT NULL
      AND p.address_normalized IS NOT NULL
),
matches AS (
    SELECT
        pre.pre_id,
        pre.pre_addr,
        tax.tax_id,
        tax.tax_addr,
        tax.apn,
        similarity(
            regexp_replace(pre.pre_norm, '^[0-9]+\\s*', ''),
            regexp_replace(tax.tax_norm, '^[0-9]+\\s*', '')
        ) AS street_sim,
        ROW_NUMBER() OVER (
            PARTITION BY pre.pre_id
            ORDER BY similarity(
                regexp_replace(pre.pre_norm, '^[0-9]+\\s*', ''),
                regexp_replace(tax.tax_norm, '^[0-9]+\\s*', '')
            ) DESC
        ) AS rn
    FROM pre_fc pre
    JOIN tax_del tax ON
        tax.county_id = pre.county_id
        AND substring(pre.pre_norm FROM '^[0-9]+') = substring(tax.tax_norm FROM '^[0-9]+')
        AND similarity(
            regexp_replace(pre.pre_norm, '^[0-9]+\\s*', ''),
            regexp_replace(tax.tax_norm, '^[0-9]+\\s*', '')
        ) >= {threshold}
)
SELECT pre_id, pre_addr, tax_id, tax_addr, apn, round(street_sim::numeric, 3) AS sim
FROM matches
WHERE rn = 1
ORDER BY sim DESC
"""


def run_backfill(session: Session, dry_run: bool = False, threshold: float = 0.65) -> dict:
    stats = {"matched": 0, "moved": 0, "skipped": 0, "errors": 0}

    # Step 1: Backfill address_normalized for any rows missing it
    result = session.execute(
        text(
            """
            UPDATE properties
            SET address_normalized = TRIM(
                REGEXP_REPLACE(
                    REGEXP_REPLACE(
                        REGEXP_REPLACE(UPPER(address_raw), '[.,#'';]', ' ', 'g'),
                        '\\m[A-Z]{2}\\s+\\d{5}(-\\d{4})?\\M\\s*$', '', 'g'
                    ),
                    '\\s+', ' ', 'g'
                )
            )
            WHERE address_normalized IS NULL
            """
        )
    )
    if not dry_run:
        session.commit()
    logger.info("backfill_normalized_updated", rows=result.rowcount)

    # Step 2: Find all matches in one query
    rows = session.execute(text(_build_match_query(threshold))).fetchall()

    logger.info("backfill_matches_found", count=len(rows), dry_run=dry_run)
    stats["matched"] = len(rows)

    for pre_id, pre_addr, tax_id, tax_addr, apn, sim in rows:
        logger.info(
            "backfill_match",
            pre_addr=pre_addr,
            tax_addr=tax_addr,
            apn=apn,
            similarity=float(sim),
        )

        if dry_run:
            continue

        try:
            # Move indicators: pre_foreclosure → tax property
            session.execute(
                text(
                    "UPDATE property_indicators SET property_id = :target "
                    "WHERE property_id = :src AND indicator_type = 'pre_foreclosure'"
                ),
                {"target": tax_id, "src": pre_id},
            )

            # Move owner records
            session.execute(
                text(
                    "UPDATE owners SET property_id = :target "
                    "WHERE property_id = :src"
                ),
                {"target": tax_id, "src": pre_id},
            )

            # Delete orphaned property (all its indicators have been moved)
            session.execute(
                text(
                    "DELETE FROM property_scores WHERE property_id = :src; "
                    "DELETE FROM properties WHERE id = :src"
                ),
                {"src": pre_id},
            )

            # Mark target property score as stale (total_score = -1 signals recalc needed)
            session.execute(
                text(
                    """
                    INSERT INTO property_scores (property_id, total_score, indicator_count, score_tier, last_scored_at)
                    VALUES (:target, -1, 0, 'cold', now())
                    ON CONFLICT (property_id) DO UPDATE SET total_score = -1
                    """
                ),
                {"target": tax_id},
            )

            session.commit()
            stats["moved"] += 1

        except Exception as exc:
            logger.error("backfill_row_error", pre_id=pre_id, tax_id=tax_id, error=str(exc))
            session.rollback()
            stats["errors"] += 1

    logger.info("backfill_complete", **stats)
    return stats


def trigger_score_recalc(session: Session) -> int:
    """Queue score recalculation for all properties marked stale (score = -1)."""
    try:
        from tasks.score_tasks import recalculate_property_score

        rows = session.execute(
            text(
                "SELECT property_id FROM property_scores WHERE total_score = -1"
            )
        ).fetchall()

        for (prop_id,) in rows:
            recalculate_property_score.delay(prop_id)

        logger.info("score_recalc_queued", count=len(rows))
        return len(rows)
    except Exception as exc:
        logger.warning("score_recalc_queue_failed", error=str(exc))
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill address-based property merges")
    parser.add_argument("--dry-run", action="store_true", help="Show matches without modifying DB")
    parser.add_argument(
        "--threshold", type=float, default=0.65,
        help="pg_trgm street-name similarity threshold (default: 0.65)"
    )
    args = parser.parse_args()

    db_url = str(settings.database_url).replace("+asyncpg", "")
    engine = create_engine(db_url, pool_size=2, max_overflow=0)

    with Session(engine) as session:
        stats = run_backfill(session, dry_run=args.dry_run, threshold=args.threshold)
        print(f"\nResults: {stats}")

        if not args.dry_run and stats["moved"] > 0:
            queued = trigger_score_recalc(session)
            print(f"Score recalc queued for {queued} properties")
