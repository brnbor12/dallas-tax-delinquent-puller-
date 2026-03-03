"""
Hillsborough County Judicial Foreclosure Auction Scraper (Florida - FIPS 12057)

Source: hillsborough.realforeclose.com  (RealForeclose platform, ColdFusion)
Data:   Court-ordered foreclosure auction listings — properties with a final
        judgment of foreclosure scheduled for sale at the online auction.
        Auctions are held multiple days per week (Mon–Fri).

Signal: indicator_type = "pre_foreclosure"
        Foreclosure auctions represent distressed properties where the owner
        still has the right of redemption up until the sale date, creating a
        window where they may sell to avoid public auction.

Auth:   None required for listing data.  The platform uses a session-based
        auction-date selector.  We mimic the two-step browser flow:
          1. GET /index.cfm?zaction=AUCTION&zmethod=PREVIEW&AuctionDate=<date>
             (sets the server-side session to the desired auction date)
          2. GET /index.cfm?zaction=AUCTION&Zmethod=UPDATE&FNC=LOAD&AREA=C&...
             (returns JSON {retHTML, rlist} with all auction items for that date)

Parsing: retHTML uses abbreviated HTML tags (@A/@B/... → div open/close,
         @H/@F/@G → tr/td open/close, etc.).  Property data is embedded as
         literal text between the abbreviated tags and can be extracted with
         simple regex patterns matching the label-value rows.

Rate limit: conservative — one date per 3 seconds.
"""

from __future__ import annotations

import re
import time
import json
from datetime import date, datetime, timedelta
from typing import AsyncIterator

import httpx
import structlog

from scrapers.base import BaseCountyScraper, RawIndicatorRecord

logger = structlog.get_logger(__name__)

COUNTY_FIPS = "12057"
BASE_URL = "https://hillsborough.realforeclose.com"

# How many calendar days ahead (and behind) to scan for auction dates
DAYS_LOOKAHEAD = 90
DAYS_LOOKBACK = 7

# AID (auction item ID) from container div
_AID_RE = re.compile(r'aid="(\d+)"')

# Parcel ID embedded in the hcpafl.org link URL
_PARCEL_URL_RE = re.compile(r'ParcelID=([A-Z0-9]+)', re.IGNORECASE)

# Generic label→value pair:  ">LABEL:@F...@CAD_DTA">VALUE@G"
_PAIR_RE = re.compile(
    r'>([^>@]+?):\s*@F[^>]*@CAD_DTA">(.*?)@G',
    re.DOTALL,
)

# All data cells (labeled and unlabeled): @CAD_DTA">VALUE@G
_CELL_RE = re.compile(r'@CAD_DTA">(.*?)@G', re.DOTALL)


def _clean(text: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_auction_item(html_fragment: str) -> dict | None:
    """
    Extract structured fields from a single auction item's retHTML block.

    The template uses abbreviated HTML tags:
      @H/@F/@G → <tr ...>/<td ...>/</tr>
      @CAD_DTA → class="AD_DTA" (data cell)
      @CAD_LBL → class="AD_LBL" (label cell)

    Parcel ID is always embedded in the hcpafl.org link URL
    (ParcelID=<value>), which is safer than parsing the link text.

    The city/state/zip appears in an unlabeled row immediately after
    "Property Address:", so we collect all data-cell values in order
    and reconstruct the full address from them.
    """
    aid_m = _AID_RE.search(html_fragment)
    aid = aid_m.group(1) if aid_m else None

    data: dict = {"aid": aid}

    # --- Named label→value pairs ---
    for label, value in _PAIR_RE.findall(html_fragment):
        key = label.strip().lower().replace(" ", "_").rstrip(":")
        data[key] = _clean(value)

    # --- Parcel ID: prefer URL-embedded value (from hcpafl.org link) over link text ---
    # The link text can be "Property Appraiser" or other non-parcel text.
    parcel_m = _PARCEL_URL_RE.search(html_fragment)
    if parcel_m:
        data["parcel_id"] = parcel_m.group(1)
    elif "parcel_id" in data:
        # Validate: Hillsborough parcel IDs are alphanumeric, no spaces
        if " " in data["parcel_id"] or not re.match(r'^[A-Z0-9]+$', data["parcel_id"]):
            del data["parcel_id"]

    # --- City/state/zip: unlabeled row right after "Property Address:" ---
    # Collect all data-cell values in document order; the address cells
    # are consecutive: street address, then city/state/zip
    if "property_address" in data:
        street = data["property_address"]
        all_cells = [_clean(c) for c in _CELL_RE.findall(html_fragment)]
        try:
            idx = all_cells.index(street)
            if idx + 1 < len(all_cells):
                city_state = all_cells[idx + 1]
                # Looks like "WIMAUMA, FL- 33598" → normalise to "WIMAUMA, FL 33598"
                city_state = city_state.replace("- ", "").strip()
                data["city_state_zip"] = city_state
        except ValueError:
            pass

    return data if "parcel_id" in data else None


class HillsboroughForeclosureScraper(BaseCountyScraper):
    county_fips = COUNTY_FIPS
    source_name = "Hillsborough County Foreclosure Auctions (RealForeclose)"
    indicator_types = ["pre_foreclosure"]
    rate_limit_per_minute = 20  # 3 s between requests

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        """
        Scan upcoming (and very recent) auction dates, fetch each auction's
        item list, and yield one RawIndicatorRecord per property.
        """
        auction_dates = self._get_scan_dates()
        logger.info(
            "hillsborough_fc_scan_start",
            dates=len(auction_dates),
            first=str(auction_dates[0]),
            last=str(auction_dates[-1]),
        )

        seen_aids: set[str] = set()

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15, read=30, write=15, pool=15),
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/121 Safari/537.36"
                )
            },
        ) as client:
            for auction_date in auction_dates:
                date_str = auction_date.strftime("%m/%d/%Y")
                items = await self._fetch_auction_items(client, date_str)
                if not items:
                    continue

                logger.info(
                    "hillsborough_fc_date_loaded",
                    date=date_str,
                    items=len(items),
                )

                for item in items:
                    aid = item.get("aid")
                    if aid in seen_aids:
                        continue
                    if aid:
                        seen_aids.add(aid)

                    record = self._build_record(item, auction_date)
                    if record and await self.validate_record(record):
                        yield record

        logger.info(
            "hillsborough_fc_complete",
            total_yielded=len(seen_aids),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_scan_dates(self) -> list[date]:
        """Return candidate auction dates: lookback + lookahead window."""
        today = date.today()
        start = today - timedelta(days=DAYS_LOOKBACK)
        end = today + timedelta(days=DAYS_LOOKAHEAD)
        dates = []
        d = start
        while d <= end:
            # Skip weekends — Hillsborough auctions are weekdays only
            if d.weekday() < 5:
                dates.append(d)
            d += timedelta(days=1)
        return dates

    async def _fetch_auction_items(
        self, client: httpx.AsyncClient, date_str: str
    ) -> list[dict]:
        """
        Two-step fetch:
          1. Set auction date in server session via the PREVIEW page.
          2. Load auction items via the JSON UPDATE endpoint.
        Returns a list of parsed item dicts.
        """
        try:
            # Step 1: Set session date
            await client.get(
                f"{BASE_URL}/index.cfm",
                params={"zaction": "AUCTION", "zmethod": "PREVIEW", "AuctionDate": date_str},
            )
            await self._rate_limit_sleep()

            # Step 2: Load items for this date
            ts = int(time.time() * 1000)
            resp = await client.get(
                f"{BASE_URL}/index.cfm",
                params={
                    "zaction": "AUCTION",
                    "Zmethod": "UPDATE",
                    "FNC": "LOAD",
                    "AREA": "C",
                    "PageDir": "0",
                    "doR": "1",
                    "tx": str(ts),
                    "bypassPage": "0",
                },
            )
            await self._rate_limit_sleep()

            # The response has leading whitespace/HTML before the JSON line
            raw = resp.text.strip()
            # Find the last line that starts with '{' — the JSON payload
            json_line = next(
                (line for line in reversed(raw.splitlines()) if line.strip().startswith("{")),
                None,
            )
            if not json_line:
                return []

            payload = json.loads(json_line)
            ret_html = payload.get("retHTML", "")
            if not ret_html:
                return []

            # Split into individual auction item blocks by AITEM_ div
            blocks = re.split(r'(?=<div id="AITEM_)', ret_html)
            items = []
            for block in blocks:
                if not block.strip():
                    continue
                parsed = _parse_auction_item(block)
                if parsed:
                    items.append(parsed)
            return items

        except (httpx.HTTPError, json.JSONDecodeError, StopIteration) as exc:
            logger.warning(
                "hillsborough_fc_fetch_error",
                date=date_str,
                error=str(exc)[:200],
            )
            return []

    def _build_record(self, item: dict, auction_date: date) -> RawIndicatorRecord | None:
        """Convert a parsed auction item dict to a RawIndicatorRecord."""
        parcel_id = item.get("parcel_id", "").strip()
        if not parcel_id:
            return None

        # Address: street + city/state/zip parsed from consecutive data cells
        addr_line1 = item.get("property_address", "").strip()
        city_state = item.get("city_state_zip", "").strip()

        if addr_line1 and city_state:
            address_raw = f"{addr_line1}, {city_state}"
        elif addr_line1:
            address_raw = f"{addr_line1}, Hillsborough County, FL"
        else:
            # Without an address we can still record the parcel; geocoder will skip it
            address_raw = f"Parcel {parcel_id}, Hillsborough County, FL"

        # Dollar amounts
        amount: float | None = None
        raw_amt = item.get("final_judgment_amount", "") or item.get("opening_bid", "")
        try:
            amount = float(re.sub(r"[,$]", "", raw_amt)) if raw_amt else None
        except ValueError:
            pass

        case_number = item.get("case_#", "") or item.get("case_number", "")
        auction_type = item.get("auction_type", "FORECLOSURE")

        return RawIndicatorRecord(
            indicator_type="pre_foreclosure",
            address_raw=address_raw,
            county_fips=self.county_fips,
            apn=parcel_id,
            amount=amount,
            filing_date=auction_date,
            case_number=case_number.strip() or None,
            source_url=f"{BASE_URL}/index.cfm?zaction=AUCTION&zmethod=PREVIEW",
            raw_payload={
                "aid": item.get("aid"),
                "auction_type": auction_type,
                "case_number": case_number,
                "parcel_id": parcel_id,
                "assessed_value": item.get("assessed_value", ""),
                "final_judgment_amount": item.get("final_judgment_amount", ""),
                "auction_date": str(auction_date),
            },
        )
