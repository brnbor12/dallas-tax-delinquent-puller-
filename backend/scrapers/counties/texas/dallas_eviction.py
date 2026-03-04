"""
Dallas County Eviction (Forcible Entry and Detainer) Scraper (Texas - FIPS 48113)

Source: Dallas County Justice of the Peace Courts
Portal: Dallas County JP Courts Odyssey system
URL:    https://dcdjp.dallascounty.org  (JP-specific portal, primary)
        Fallback: https://portal.co.dallas.tx.us  (county clerk portal)

Data:   Active FED (Forcible Entry and Detainer) case filings — evictions
        filed by landlords against tenants in Justice of the Peace courts.

Signal: indicator_type = "eviction"
        Active eviction = landlord-tenant conflict on a rental property.
        Indicates cash flow problems — the landlord may be motivated to sell
        to exit a problem tenancy.

Strategy:
        Uses TylerOdysseyPlaywrightScraper (Playwright-based) to establish a
        real browser session on the JP portal, then POSTs to the case search
        API using the browser's cookie jar.  Tries multiple FED case type
        codes in order.  Falls back to the county clerk portal if the JP
        portal returns no results.

Address: The defendant (tenant) address in FED cases IS the rental property
        address.  Extracted from case detail parties.
        Falls back to "Eviction {case_number}, Dallas County, TX" placeholder.

Rate: 15 req/min.
"""

from __future__ import annotations

import re
import structlog
from datetime import date, datetime
from typing import AsyncIterator

from scrapers.base import RawIndicatorRecord
from scrapers.counties.texas.tyler_odyssey_base import TylerOdysseyPlaywrightScraper

logger = structlog.get_logger(__name__)

COUNTY_FIPS = "48113"
JP_PORTAL_BASE = "https://dcdjp.dallascounty.org"
CC_PORTAL_BASE = "https://courtsportal.dallascounty.org/DALLASPROD"
LOOKBACK_DAYS = 90  # evictions resolve quickly

# Possible case type codes for FED across Odyssey versions
_FED_CASE_TYPES = ["FED", "FEDU", "EVICTION", "FORCIBLE", "FE"]

# Party roles that identify the defendant (tenant = property address)
_DEFENDANT_ROLES = {"DEFENDANT", "RESPONDENT", "TENANT", "DEF", "RESP"}


def _coerce(d: dict, *keys: str, default: str = "") -> str:
    for k in keys:
        v = d.get(k)
        if v is not None:
            return str(v).strip()
    return default


def _extract_address(parties: list[dict], role_set: set[str]) -> str | None:
    for party in parties or []:
        role = _coerce(
            party, "PartyType", "partyType", "party_type", "RoleDesc", "roleDesc"
        ).upper()
        if not any(r in role for r in role_set):
            continue
        addr = _coerce(party, "Address", "address", "streetAddress")
        city = _coerce(party, "City", "city")
        state = _coerce(party, "State", "state", default="TX")
        zip_code = _coerce(party, "Zip", "zip", "ZipCode", "zipCode")
        if addr and addr[0].isdigit():
            full = f"{addr}, {city or 'Dallas'}, {state}"
            if zip_code:
                full += f" {zip_code}"
            return full
    return None


class DallasEvictionScraper(TylerOdysseyPlaywrightScraper):
    portal_base = JP_PORTAL_BASE  # set per-attempt in fetch_records
    county_fips = COUNTY_FIPS
    source_name = "Dallas County JP Court Evictions FED (Tyler Odyssey Playwright)"
    indicator_types = ["eviction"]
    rate_limit_per_minute = 15

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        pairs = await self._try_portals()

        if not pairs:
            logger.warning(
                "dallas_eviction_api_exhausted",
                hint="CaseSearch API failed — falling back to Hearing Search UI",
            )
            async for record in self._fetch_via_hearing_search():
                yield record
            return

        total = 0
        for case, detail in pairs:
            record = self._build_record(case, detail)
            if record and await self.validate_record(record):
                yield record
                total += 1

        logger.info("dallas_eviction_complete", total_yielded=total)

    async def _fetch_via_hearing_search(self) -> AsyncIterator[RawIndicatorRecord]:
        """Fallback: scrape Hearing Search UI for JP court FED hearings."""
        self.portal_base = CC_PORTAL_BASE
        cases = await self._get_cases_via_hearing_search(
            court_location="Justice of the Peace Courts",
            lookback_days=30,
            lookahead_days=60,
        )
        if not cases:
            logger.warning("dallas_eviction_hearing_search_empty")
            return

        total = 0
        for row in cases:
            record = self._build_record_from_hearing_row(row)
            if record and await self.validate_record(record):
                yield record
                total += 1
                await self._rate_limit_sleep()

        logger.info("dallas_eviction_hearing_complete", total_yielded=total)

    def _build_record_from_hearing_row(self, row: dict) -> RawIndicatorRecord | None:
        """Build RawIndicatorRecord from a Hearing Search table row."""
        case_number = ""
        for k in row:
            if re.search(r"case.?no|case.?number|casenum", k, re.I):
                case_number = str(row[k]).strip()
                break

        style_name = ""
        for k in row:
            if re.search(r"style|name|caption|parties", k, re.I):
                style_name = str(row[k]).strip()
                break

        if not case_number:
            return None

        case_url = ""
        for k in row:
            if k.endswith("_href") and row[k]:
                case_url = str(row[k])
                break

        hearing_date_str = ""
        for k in row:
            if re.search(r"hearing.?date|date|scheduled", k, re.I) and not k.endswith("_href"):
                hearing_date_str = str(row[k]).strip()
                break

        filing_date: date | None = None
        if hearing_date_str:
            for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y"):
                try:
                    filing_date = datetime.strptime(hearing_date_str[:10], fmt).date()
                    break
                except ValueError:
                    continue

        # In FED cases, defendant (tenant) = rental property address — use style as proxy
        plaintiff = ""
        if " VS " in style_name.upper():
            plaintiff = style_name.upper().split(" VS ")[0].strip().title()

        address_raw = case_url or f"Eviction {case_number}, Dallas County, TX"

        return RawIndicatorRecord(
            indicator_type="eviction",
            address_raw=address_raw,
            county_fips=self.county_fips,
            owner_name=plaintiff or None,
            filing_date=filing_date,
            case_number=case_number,
            source_url=f"{CC_PORTAL_BASE}/",
            raw_payload={
                "case_number": case_number,
                "style_name": style_name,
                "case_url": case_url,
                "source": "hearing_search",
                **{k: v for k, v in row.items() if not k.startswith("_")},
            },
        )

    async def _try_portals(self) -> list[tuple[dict, dict]]:
        """
        Try JP portal first, then county clerk portal fallback.
        Tries multiple node IDs and FED case type codes on each portal.
        Dallas ePortal uses full location strings like "Justice of the Peace Courts".
        """
        jp_node_ids = ["Justice of the Peace Courts", "JP", "JPC", ""]
        cc_node_ids = ["Justice of the Peace Courts", "JP", "JPC", ""]

        for portal_base, node_ids in [
            (JP_PORTAL_BASE, jp_node_ids),
            (CC_PORTAL_BASE, cc_node_ids),
        ]:
            self.portal_base = portal_base
            for node_id in node_ids:
                for case_type in _FED_CASE_TYPES:
                    pairs = await self._get_cases_with_details(
                        category="CV",
                        node_id=node_id,
                        status_type="A",
                        lookback_days=LOOKBACK_DAYS,
                        case_type_id=case_type,
                    )
                    if pairs:
                        logger.info(
                            "dallas_eviction_portal_hit",
                            portal=portal_base,
                            node_id=node_id,
                            case_type=case_type,
                            count=len(pairs),
                        )
                        return pairs

        return []

    def _build_record(self, case: dict, detail: dict) -> RawIndicatorRecord | None:
        case_number = _coerce(case, "CaseNumber", "caseNumber", "case_number")
        if not case_number:
            return None

        style_name = _coerce(case, "StyleName", "styleName", "style_name")
        case_id = _coerce(case, "CaseID", "caseId", "caseID", "id")
        raw_date = _coerce(case, "FilingDate", "filingDate", "filing_date")

        filing_date: date | None = None
        if raw_date:
            try:
                filing_date = datetime.fromisoformat(raw_date[:10]).date()
            except ValueError:
                pass

        parties = (
            detail.get("parties")
            or detail.get("Parties")
            or detail.get("caseParties")
            or []
        )
        address_raw = _extract_address(parties, _DEFENDANT_ROLES)
        if not address_raw:
            address_raw = f"Eviction {case_number}, Dallas County, TX"

        # Plaintiff = landlord = property owner
        plaintiff = _coerce(case, "PlaintiffName", "plaintiffName", "plaintiff_name")
        owner_name = plaintiff.title() if plaintiff else None

        return RawIndicatorRecord(
            indicator_type="eviction",
            address_raw=address_raw,
            county_fips=self.county_fips,
            owner_name=owner_name,
            filing_date=filing_date,
            case_number=case_number,
            source_url=f"{self.portal_base}/",
            raw_payload={
                "case_number": case_number,
                "style_name": style_name,
                "case_id": case_id,
                "court_id": _coerce(case, "CourtID", "courtId", "court_id"),
                "case_type": _coerce(case, "CaseTypeDesc", "caseTypeDesc"),
                "status": _coerce(case, "StatusDesc", "statusDesc", default="Active"),
                "filing_date": raw_date,
                "plaintiff": plaintiff,
            },
        )
