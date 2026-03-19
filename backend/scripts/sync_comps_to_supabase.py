"""
Sync Rentcast listing data from local Postgres to Supabase as comp records.

Creates/updates two Supabase tables:
  1. comp_sales — individual property listings with address, price, sqft, status
  2. zip_price_data — updates $/sqft aggregates with Rentcast source data

Run after Rentcast scrapers complete:
    python scripts/sync_comps_to_supabase.py

Requires: SUPABASE_URL and SUPABASE_KEY in .env
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime
from collections import defaultdict

import httpx
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.config import settings

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
SB_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "resolution=merge-duplicates",
}


def get_db_session():
    sync_url = settings.database_url.replace("+asyncpg", "+psycopg2")
    engine = create_engine(sync_url, pool_size=2)
    return sessionmaker(bind=engine)()


def ensure_comp_sales_table():
    """Create comp_sales table in Supabase if it doesn't exist.
    Uses Supabase RPC — requires a migration function or manual creation.
    Falls back to upserting rows which auto-discovers the table.
    """
    # Test if table exists
    resp = httpx.get(
        f"{SUPABASE_URL}/rest/v1/comp_sales?limit=0",
        headers=SB_HEADERS,
    )
    if resp.status_code == 200:
        print("comp_sales table exists")
        return True

    print(f"comp_sales table not found (HTTP {resp.status_code})")
    print("Please create the table in Supabase SQL Editor:")
    print("""
CREATE TABLE IF NOT EXISTS comp_sales (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    address text NOT NULL,
    city text,
    state text,
    zip text NOT NULL,
    county_fips text,
    price numeric,
    price_per_sqft numeric,
    sqft integer,
    bedrooms integer,
    bathrooms numeric,
    property_type text,
    listing_status text,
    days_on_market integer,
    listed_date date,
    last_seen_date date,
    mls_number text,
    mls_name text,
    data_source text DEFAULT 'rentcast',
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),
    UNIQUE(address, data_source)
);

CREATE INDEX idx_comp_sales_zip ON comp_sales(zip);
CREATE INDEX idx_comp_sales_status ON comp_sales(listing_status);
CREATE INDEX idx_comp_sales_county ON comp_sales(county_fips);
    """)
    return False


def pull_listing_data() -> list[dict]:
    """Pull Rentcast listing records from local DB."""
    session = get_db_session()
    try:
        rows = session.execute(text("""
            SELECT
                p.address_line1, p.address_city, p.address_state, p.address_zip,
                p.sqft, p.bedrooms, p.bathrooms, p.property_type,
                pi.raw_data,
                c.fips_code AS county_fips
            FROM property_indicators pi
            JOIN properties p ON p.id = pi.property_id
            LEFT JOIN counties c ON c.id = p.county_id
            WHERE pi.indicator_type = 'active_listing'
              AND pi.status = 'active'
              AND pi.raw_data IS NOT NULL
        """)).fetchall()

        records = []
        for r in rows:
            payload = r.raw_data if isinstance(r.raw_data, dict) else json.loads(r.raw_data or "{}")
            price = payload.get("price")
            sqft = payload.get("sqft") or r.sqft
            price_per_sqft = None
            if price and sqft and sqft > 0:
                price_per_sqft = round(float(price) / float(sqft), 2)

            addr = r.address_line1 or ""
            if not addr:
                continue

            full_addr = f"{addr}, {r.address_city or ''}, {r.address_state or ''} {r.address_zip or ''}".strip(", ")

            records.append({
                "address": full_addr,
                "city": r.address_city,
                "state": r.address_state,
                "zip": (r.address_zip or "")[:5],
                "county_fips": r.county_fips,
                "price": float(price) if price else None,
                "price_per_sqft": price_per_sqft,
                "sqft": int(sqft) if sqft else None,
                "bedrooms": int(payload.get("bedrooms") or r.bedrooms or 0) or None,
                "bathrooms": float(payload.get("bathrooms") or r.bathrooms or 0) or None,
                "property_type": payload.get("property_type") or r.property_type,
                "listing_status": payload.get("status", "Active"),
                "days_on_market": payload.get("days_on_market"),
                "listed_date": payload.get("listing_date") or None,
                "last_seen_date": payload.get("last_seen") or None,
                "mls_number": payload.get("mls_number") or None,
                "mls_name": payload.get("mls_name") or None,
                "data_source": "rentcast",
                "updated_at": datetime.utcnow().isoformat(),
            })

        print(f"Pulled {len(records)} listing records from local DB")
        return records
    finally:
        session.close()


def push_comp_sales(records: list[dict]):
    """Upsert comp records to Supabase."""
    if not records:
        return

    # Batch upsert 200 at a time
    batch_size = 200
    upserted = 0
    failed = 0

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        resp = httpx.post(
            f"{SUPABASE_URL}/rest/v1/comp_sales",
            headers=SB_HEADERS,
            json=batch,
            timeout=30.0,
        )
        if resp.status_code in (200, 201):
            upserted += len(batch)
        else:
            failed += len(batch)
            if i == 0:  # Print first error for debugging
                print(f"Supabase error: {resp.status_code} — {resp.text[:300]}")

    print(f"comp_sales: upserted={upserted}, failed={failed}")


def update_zip_price_data(records: list[dict]):
    """Aggregate listing data by zip and update zip_price_data with Rentcast source."""
    zip_stats: dict[str, list[float]] = defaultdict(list)
    zip_counts: dict[str, int] = defaultdict(int)

    for r in records:
        zc = r.get("zip", "")
        ppsf = r.get("price_per_sqft")
        if zc and len(zc) == 5 and ppsf and ppsf > 0:
            zip_stats[zc].append(ppsf)
            zip_counts[zc] += 1

    if not zip_stats:
        print("No zip-level data to update")
        return

    updates = []
    for zc, prices in zip_stats.items():
        median_ppsf = sorted(prices)[len(prices) // 2]
        updates.append({
            "zip": zc,
            "market": _infer_market(zc),
            "price_per_sqft": round(median_ppsf, 1),
            "source_count": zip_counts[zc],
            "pull_date": date.today().isoformat(),
        })

    # Upsert to zip_price_data
    resp = httpx.post(
        f"{SUPABASE_URL}/rest/v1/zip_price_data",
        headers=SB_HEADERS,
        json=updates,
        timeout=30.0,
    )
    if resp.status_code in (200, 201):
        print(f"zip_price_data: updated {len(updates)} zips with Rentcast $/sqft")
    else:
        print(f"zip_price_data error: {resp.status_code} — {resp.text[:300]}")

    # Also update market_pricing_ddba1844 blended values
    for u in updates:
        zc = u["zip"]
        # Update the blended price — merge with existing sources
        resp = httpx.get(
            f"{SUPABASE_URL}/rest/v1/market_pricing_ddba1844?zip_code=eq.{zc}&limit=1",
            headers=SB_HEADERS,
        )
        if resp.status_code == 200 and resp.json():
            existing = resp.json()[0]
            # Average with existing if available, otherwise set
            old_ppsf = existing.get("price_per_sqft") or 0
            new_ppsf = u["price_per_sqft"]
            blended = round((old_ppsf + new_ppsf) / 2, 1) if old_ppsf > 0 else new_ppsf

            httpx.patch(
                f"{SUPABASE_URL}/rest/v1/market_pricing_ddba1844?zip_code=eq.{zc}",
                headers=SB_HEADERS,
                json={
                    "price_per_sqft": blended,
                    "confidence_level": "High" if u["source_count"] >= 5 else "Medium",
                    "source_count": u["source_count"],
                    "last_updated": datetime.utcnow().isoformat(),
                    "notes": f"Blended with Rentcast ({u['source_count']} listings)",
                },
                timeout=10.0,
            )


def _infer_market(zip_code: str) -> str:
    prefix = zip_code[:3]
    if prefix.startswith("75"):
        return "dallas_tx"
    elif prefix.startswith("33"):
        return "tampa_fl"
    elif prefix.startswith("338") or prefix.startswith("348"):
        return "polk_fl"
    return "unknown"


def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: SUPABASE_URL and SUPABASE_KEY required in .env")
        sys.exit(1)

    print(f"Supabase: {SUPABASE_URL[:40]}...")

    # Check table exists
    table_ok = ensure_comp_sales_table()

    # Pull from local DB
    records = pull_listing_data()
    if not records:
        print("No listing records to sync")
        return

    # Push to Supabase (try even if table check failed — might be RLS)
    push_comp_sales(records)

    # Update zip-level aggregates
    update_zip_price_data(records)

    print("Done!")


if __name__ == "__main__":
    main()
