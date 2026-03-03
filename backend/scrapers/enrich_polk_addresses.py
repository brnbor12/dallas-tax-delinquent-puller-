"""
Polk County FL address enrichment.

Looks up situs (physical) addresses from the Polk County Property Appraiser
website (polkflpa.gov) for properties imported via the tax deed scraper, which
stores only the strap number as a placeholder address.

Flow per property:
  1. Two-step session init on polkflpa.gov (GET ×2 for cookie dance)
  2. POST CamaSearch.aspx with strap number → parse Site Address from results table
  3. GET CamaDisplay.aspx for the matched parcel → parse Physical Street Address
     and Postal City/Zip from the detail page
  4. UPDATE properties table with full address components

Usage:
    docker compose exec api python -m scrapers.enrich_polk_addresses
    docker compose exec api python -m scrapers.enrich_polk_addresses --dry-run
    docker compose exec api python -m scrapers.enrich_polk_addresses --limit 10
"""

from __future__ import annotations

import argparse
import re
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import httpx
import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.core.config import settings

logger = structlog.get_logger(__name__)

BASE_URL = "https://www.polkflpa.gov"
SEARCH_URL = f"{BASE_URL}/CamaSearch.aspx"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

# Seconds between requests (polite rate limit)
REQUEST_DELAY = 1.0
# Re-init session after this many lookups (avoids session timeouts)
SESSION_REFRESH_EVERY = 50


def _extract_form_fields(html: str) -> dict:
    """Extract all <input> name/value pairs from an HTML form."""
    fields: dict[str, str] = {}
    for m in re.finditer(
        r'<input[^>]+name="([^"]+)"[^>]*value="([^"]*)"', html, re.IGNORECASE
    ):
        fields[m.group(1)] = m.group(2)
    return fields


def _init_session(client: httpx.Client) -> dict:
    """
    Two-step cookie dance: first GET triggers the cookie-test redirect,
    second GET loads the real form with a populated ViewState.
    Returns the extracted form fields (ViewState, EventValidation, etc.).
    """
    client.get(SEARCH_URL, headers=HEADERS)          # sets ASP.NET_SessionId cookie
    r = client.get(SEARCH_URL, headers=HEADERS)      # loads real form
    fields = _extract_form_fields(r.text)
    logger.debug("polk_enrich_session_init", viewstate_len=len(fields.get("__VIEWSTATE", "")))
    return fields


def _search_strap(
    client: httpx.Client,
    form_fields: dict,
    strap: str,
) -> tuple[str | None, str | None, dict]:
    """
    POST a parcel ID search and parse the results table.

    Returns:
        (site_address_street, display_url, new_form_fields)
        site_address_street: e.g. "1099 CLUBHOUSE RD" or None
        display_url: full URL to CamaDisplay.aspx for the property, or None
        new_form_fields: ViewState etc. from the POST response (for next call)
    """
    post_data = dict(form_fields)
    post_data["ctl00$mainCopy$rblRealTangible"] = "Real"
    post_data["ctl00$mainCopy$searchRE_id"] = strap
    post_data["ctl00$mainCopy$submitSearch"] = "Search for Property"

    r = client.post(
        SEARCH_URL,
        data=post_data,
        headers={
            **HEADERS,
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": SEARCH_URL,
        },
    )
    time.sleep(REQUEST_DELAY)

    html = r.text
    new_form_fields = _extract_form_fields(html)

    # Parse the RESearchResults GridView table
    # Columns: [#, Owner Name, Parcel ID, Site Address, Last Sale Date]
    table_m = re.search(
        r'<table[^>]+id="RESearchResults"[^>]*>(.*?)</table>', html, re.DOTALL
    )
    if not table_m:
        return None, None, new_form_fields

    table_html = table_m.group(1)
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.DOTALL)

    for row in rows[1:]:  # skip header row
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
        cells = [
            re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", c)).strip()
            for c in cells
        ]
        # Expect: [row_num, owner_name, parcel_id, site_address, sale_date]
        if len(cells) >= 4 and cells[0].isdigit():
            site_address = cells[3].strip()
            if site_address:
                break
    else:
        return None, None, new_form_fields

    # Extract CamaDisplay link for this parcel
    display_m = re.search(
        r'href="(/CamaDisplay\.aspx\?OutputMode=Display[^"]+)"', html, re.IGNORECASE
    )
    display_url = (BASE_URL + display_m.group(1)) if display_m else None

    return site_address, display_url, new_form_fields


def _get_full_address(client: httpx.Client, display_url: str) -> tuple[str | None, str | None]:
    """
    Fetch the CamaDisplay detail page and extract:
      - Physical street address
      - Postal city + state + zip (as a single string like "WINTER HAVEN FL 33884")

    The page contains commented-out label TDs; strip comments before parsing.
    """
    r = client.get(display_url, headers=HEADERS)
    time.sleep(REQUEST_DELAY)

    # Strip HTML comments to avoid matching commented-out label cells
    html = re.sub(r"<!--.*?-->", "", r.text, flags=re.DOTALL)

    def _extract_first_td(section_heading: str) -> str | None:
        m = re.search(
            rf"{re.escape(section_heading)}.*?<td[^>]*>(.*?)</td>",
            html,
            re.DOTALL | re.IGNORECASE,
        )
        if not m:
            return None
        raw = re.sub(r"<[^>]+>", " ", m.group(1))
        raw = re.sub(r"&nbsp;", " ", raw)
        raw = re.sub(r"\s+", " ", raw).strip()
        return raw if len(raw) >= 4 else None

    street = _extract_first_td("Physical Street Address")
    city_state_zip = _extract_first_td("Postal City and Zip")

    return street, city_state_zip


def _parse_city_state_zip(raw: str) -> tuple[str | None, str | None, str | None]:
    """
    Parse "WINTER HAVEN FL 33884" → (city, state, zip).
    Handles spacing and &nbsp; already replaced by caller.
    """
    m = re.match(r"^(.*?)\s+([A-Z]{2})\s+(\d{5})\s*$", raw.strip(), re.IGNORECASE)
    if m:
        return m.group(1).strip().upper(), m.group(2).upper(), m.group(3)
    return None, None, None


def lookup_strap_address(
    client: httpx.Client,
    form_fields: dict,
    strap: str,
) -> tuple[str | None, str | None, str | None, str | None, dict]:
    """
    Full address lookup for one strap number.

    Returns: (line1, city, state, zip, new_form_fields)
    All address components may be None if not found.
    """
    street, display_url, new_fields = _search_strap(client, form_fields, strap)

    if not street:
        return None, None, None, None, new_fields

    city: str | None = None
    state: str | None = "FL"
    zipcode: str | None = None

    if display_url:
        disp_street, city_state_zip = _get_full_address(client, display_url)
        # Prefer display page street (it's the official situs line)
        if disp_street:
            street = disp_street
        if city_state_zip:
            city, state, zipcode = _parse_city_state_zip(city_state_zip)

    return street, city, state or "FL", zipcode, new_fields


def run_polk_address_enrichment(
    session: Session,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict:
    """Enrich Polk County properties that have placeholder 'Parcel XXXX' addresses."""
    stats = {"total": 0, "enriched": 0, "not_found": 0, "failed": 0}

    # Find Polk properties with placeholder addresses
    query = text("""
        SELECT p.id, p.apn, p.address_raw
        FROM properties p
        JOIN counties c ON c.id = p.county_id
        WHERE c.fips_code = '12105'
          AND p.address_raw LIKE 'Parcel%'
          AND p.apn IS NOT NULL
        ORDER BY p.id
        {}
    """.format("LIMIT :lim" if limit else ""))

    params: dict = {}
    if limit:
        params["lim"] = limit

    rows = session.execute(query, params).fetchall()
    stats["total"] = len(rows)
    logger.info("polk_enrich_start", total=stats["total"], dry_run=dry_run)

    if dry_run:
        print(f"[DRY RUN] Would enrich {stats['total']} Polk County properties")
        return stats

    if stats["total"] == 0:
        logger.info("polk_enrich_nothing_to_do")
        return stats

    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(connect=15, read=30, write=15, pool=15)) as client:
        form_fields = _init_session(client)

        for i, row in enumerate(rows):
            prop_id = row[0]
            strap = str(row[1]).strip()
            address_raw = row[2]

            # Refresh session periodically
            if i > 0 and i % SESSION_REFRESH_EVERY == 0:
                logger.info("polk_enrich_session_refresh", at=i)
                form_fields = _init_session(client)

            logger.debug("polk_enrich_lookup", prop_id=prop_id, strap=strap)

            try:
                line1, city, state, zipcode, form_fields = lookup_strap_address(
                    client, form_fields, strap
                )
            except Exception as exc:
                logger.warning("polk_enrich_error", prop_id=prop_id, strap=strap, error=str(exc)[:120])
                stats["failed"] += 1
                continue

            if not line1:
                logger.debug("polk_enrich_not_found", prop_id=prop_id, strap=strap)
                stats["not_found"] += 1
                continue

            # Build full address_raw
            if city and zipcode:
                new_address_raw = f"{line1}, {city}, {state} {zipcode}"
            elif city:
                new_address_raw = f"{line1}, {city}, {state}"
            else:
                new_address_raw = f"{line1}, Polk County, {state}"

            session.execute(
                text("""
                    UPDATE properties
                    SET address_raw   = :addr_raw,
                        address_line1 = :line1,
                        address_city  = :city,
                        address_state = :state,
                        address_zip   = :zipcode,
                        updated_at    = NOW()
                    WHERE id = :prop_id
                """),
                {
                    "addr_raw": new_address_raw,
                    "line1": line1,
                    "city": city,
                    "state": state,
                    "zipcode": zipcode,
                    "prop_id": prop_id,
                },
            )
            session.commit()
            stats["enriched"] += 1

            logger.info(
                "polk_enrich_updated",
                prop_id=prop_id,
                strap=strap,
                address=new_address_raw,
            )

    pct = stats["enriched"] / max(stats["total"], 1) * 100
    logger.info("polk_enrich_complete", **stats, pct=f"{pct:.1f}%")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich Polk County FL property addresses")
    parser.add_argument("--dry-run", action="store_true", help="Count without updating")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of properties")
    args = parser.parse_args()

    db_url = str(settings.database_url).replace("+asyncpg", "")
    engine = create_engine(db_url, pool_size=2, max_overflow=0)

    with Session(engine) as session:
        stats = run_polk_address_enrichment(
            session,
            dry_run=args.dry_run,
            limit=args.limit,
        )
        pct = stats["enriched"] / max(stats["total"], 1) * 100
        print(f"\nResults: {stats}")
        print(f"Enriched {stats['enriched']}/{stats['total']} ({pct:.1f}%)")
