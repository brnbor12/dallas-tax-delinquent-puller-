"""
Dallas County Eviction (Forcible Entry and Detainer) Scraper (Texas - FIPS 48113)

Source: Dallas County Justice of the Peace Courts
Portal: Dallas County JP Courts Odyssey system
URL:    https://dcdjp.dallascounty.org  (JP-specific portal)
        Fallback: https://portal.co.dallas.tx.us  (county clerk portal)

Data:   Active FED (Forcible Entry and Detainer) case filings — evictions
        filed by landlords against tenants in Justice of the Peace courts.
        Dallas County has JP courts in Precincts 1–5 (multiple places each).

Signal: indicator_type = "eviction"
        Active eviction = landlord-tenant conflict on a rental property.
        Indicates cash flow problems — the landlord may be motivated to sell
        to exit a problem tenancy. Strongest when combined with other signals
        (tax delinquency, code violations, absentee owner).

Session: JP court portal may differ from the county clerk Odyssey portal.
         This scraper attempts:
           1. https://dcdjp.dallascounty.org  (JP-specific Odyssey instance)
           2. Falls back to the standard county clerk portal with JP category

         The two-step flow mirrors the probate scraper:
           1. GET portal homepage → session cookie
           2. POST to case search with category="JP", caseTypeId="FED"

Address: Eviction cases list the defendant's (tenant's) address, which IS
         the rental property address. Extracted from case parties.
         Falls back to "Eviction {case_number}, Dallas County, TX" placeholder.

Rate limit: 15 req/min — conservative for court-operated infrastructure.

Note:   If this scraper returns 0 records, the JP court may use a different
        portal or case category code. Check:
          - https://dcdjp.dallascounty.org  (JP portal)
          - Dallas County District Clerk at portal.co.dallas.tx.us
          - Case type codes: FED, FEDU, FORCIBLE, DETAINER (vary by county)
        A Playwright-based scraper may be required if no REST API is available.
"""

from __future__ import annotations

import structlog
from datetime import date, datetime, timedelta
from typing import AsyncIterator

import httpx

from scrapers.base import BaseCountyScraper, RawIndicatorRecord

logger = structlog.get_logger(__name__)

COUNTY_FIPS = "48113"

# JP-specific Odyssey portal (primary target)
JP_PORTAL_BASE = "https://dcdjp.dallascounty.org"
JP_PORTAL_INIT = f"{JP_PORTAL_BASE}/PortalService/api/portal/"
JP_CASE_SEARCH = f"{JP_PORTAL_BASE}/PortalService/api/Case/CaseSearch"
JP_CASE_DETAIL = f"{JP_PORTAL_BASE}/PortalService/api/Case/CaseDetail"

# County clerk portal (fallback)
CC_PORTAL_BASE = "https://portal.co.dallas.tx.us"
CC_CASE_SEARCH = f"{CC_PORTAL_BASE}/PortalService/api/Case/CaseSearch"
CC_CASE_DETAIL = f"{CC_PORTAL_BASE}/PortalService/api/Case/CaseDetail"

# Active evictions only — recent window (evictions resolve quickly)
LOOKBACK_DAYS = 90

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Content-Type": "application/json",
}

# Possible case type codes for evictions across Tyler Odyssey versions
_FED_CASE_TYPES = ["FED", "FEDU", "EVICTION", "FORCIBLE", "FE"]

# Party roles that hold the rental property address (defendant = tenant = address)
_DEFENDANT_ROLES = {
    "DEFENDANT", "RESPONDENT", "TENANT", "DEF", "RESP",
}


def _coerce(d: dict, *keys: str, default: str = "") -> str:
    for k in keys:
        v = d.get(k)
        if v is not None:
            return str(v).strip()
    return default


class DallasEvictionScraper(BaseCountyScraper):
    county_fips = COUNTY_FIPS
    source_name = "Dallas County JP Court Evictions (FED)"
    indicator_types = ["eviction"]
    rate_limit_per_minute = 15

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        """
        Attempt eviction scraping from the JP portal, then fall back to
        the county clerk portal if the JP portal is unreachable or returns
        no results.
        """
        # Try JP-specific portal first
        cases = await self._fetch_from_portal(
            JP_PORTAL_BASE, JP_PORTAL_INIT, JP_CASE_SEARCH, "JP"
        )

        if not cases:
            # Fall back: county clerk portal, search JP court type
            logger.info(
                "dallas_eviction_fallback_to_cc_portal",
                reason="JP portal returned no results or is unreachable",
            )
            cases = await self._fetch_from_portal(
                CC_PORTAL_BASE, None, CC_CASE_SEARCH, "JP"
            )

        if not cases:
            logger.warning(
                "dallas_eviction_no_cases",
                hint=(
                    "Neither JP portal nor county clerk portal returned FED cases. "
                    "Check dcdjp.dallascounty.org in a browser and verify "
                    "case type codes and nodeId values."
                ),
            )
            return

        logger.info("dallas_eviction_cases_found", count=len(cases))
        total = 0

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(connect=15, read=30, write=15, pool=15),
        ) as detail_client:
            # Establish session for detail calls
            try:
                base = JP_PORTAL_BASE if cases else CC_PORTAL_BASE
                detail_url = JP_CASE_DETAIL if cases else CC_CASE_DETAIL
                await detail_client.get(base + "/", headers={**_HEADERS, "Content-Type": "text/html"})
                await self._rate_limit_sleep()
            except httpx.HTTPError:
                detail_url = JP_CASE_DETAIL

            for case in cases:
                record = await self._build_record(detail_client, detail_url, case)
                if record and await self.validate_record(record):
                    yield record
                    total += 1

        logger.info("dallas_eviction_complete", total_yielded=total)

    async def _fetch_from_portal(
        self,
        portal_base: str,
        init_url: str | None,
        search_url: str,
        node_id: str,
    ) -> list[dict]:
        """
        Attempt a case search against the given portal URL.
        Returns a list of case dicts, or [] on failure.
        """
        headers = {**_HEADERS, "Origin": portal_base, "Referer": f"{portal_base}/"}

        start_date = (date.today() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        end_date = date.today().strftime("%Y-%m-%d")

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(connect=15, read=30, write=15, pool=15),
        ) as client:
            # Session init
            try:
                await client.get(portal_base + "/", headers={**headers, "Content-Type": "text/html"})
                await self._rate_limit_sleep()
                if init_url:
                    await client.get(init_url, headers=headers)
                    await self._rate_limit_sleep()
            except httpx.HTTPError as exc:
                logger.debug("dallas_eviction_portal_unreachable", portal=portal_base, error=str(exc)[:100])
                return []

            # Try each possible FED case type code until one returns results
            for case_type in _FED_CASE_TYPES:
                search_payload = {
                    "nodeId": node_id,
                    "category": "CV",          # Civil
                    "caseTypeId": case_type,
                    "statusType": "A",          # Active cases only
                    "filingDateStart": start_date,
                    "filingDateEnd": end_date,
                    "lastName": "",
                    "firstName": "",
                    "caseNumber": "",
                }

                try:
                    resp = await client.post(search_url, json=search_payload, headers=headers)
                    await self._rate_limit_sleep()

                    if resp.status_code not in (200, 201):
                        continue

                    data = resp.json()
                except (httpx.HTTPError, ValueError):
                    continue

                cases = (
                    data
                    if isinstance(data, list)
                    else (
                        data.get("cases")
                        or data.get("result")
                        or data.get("Results")
                        or data.get("data")
                        or []
                    )
                )

                if cases:
                    logger.info(
                        "dallas_eviction_portal_hit",
                        portal=portal_base,
                        case_type=case_type,
                        count=len(cases),
                    )
                    return cases

        return []

    async def _build_record(
        self, client: httpx.AsyncClient, detail_url: str, case: dict
    ) -> RawIndicatorRecord | None:
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

        # Get defendant (tenant) address = the rental property address
        address_raw: str | None = None
        if case_id:
            address_raw = await self._get_defendant_address(client, detail_url, case_id)

        if not address_raw:
            address_raw = f"Eviction {case_number}, Dallas County, TX"

        # Plaintiff name (landlord) = owner; defendant name = tenant
        plaintiff = _coerce(case, "PlaintiffName", "plaintiffName", "plaintiff_name")
        owner_name = plaintiff.title() if plaintiff else None

        return RawIndicatorRecord(
            indicator_type="eviction",
            address_raw=address_raw,
            county_fips=self.county_fips,
            owner_name=owner_name,
            filing_date=filing_date,
            case_number=case_number,
            source_url=f"{JP_PORTAL_BASE}/",
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

    async def _get_defendant_address(
        self, client: httpx.AsyncClient, detail_url: str, case_id: str
    ) -> str | None:
        """
        Fetch case detail and extract the defendant (tenant) address,
        which is the rental property address in FED cases.
        """
        try:
            resp = await client.post(
                detail_url,
                json={"caseId": case_id},
                headers=_HEADERS,
            )
            await self._rate_limit_sleep()
            if resp.status_code != 200:
                return None
            detail = resp.json()
        except (httpx.HTTPError, ValueError):
            return None

        parties = (
            detail.get("parties")
            or detail.get("Parties")
            or detail.get("caseParties")
            or []
        )

        for party in parties:
            role = _coerce(
                party, "PartyType", "partyType", "party_type", "RoleDesc", "roleDesc"
            ).upper()

            if not any(r in role for r in _DEFENDANT_ROLES):
                continue

            addr = _coerce(party, "Address", "address", "streetAddress")
            city = _coerce(party, "City", "city")
            state = _coerce(party, "State", "state", default="TX")
            zip_code = _coerce(party, "Zip", "zip", "ZipCode", "zipCode")

            if addr and addr[0].isdigit():  # looks like a real street address
                full = f"{addr}, {city or 'Dallas'}, {state}"
                if zip_code:
                    full += f" {zip_code}"
                return full

        return None
