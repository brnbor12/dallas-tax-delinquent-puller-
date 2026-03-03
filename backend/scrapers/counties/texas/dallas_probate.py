"""
Dallas County Probate Scraper (Texas - FIPS 48113)

Source: Dallas County Clerk — Tyler Odyssey Case Management Portal
URL:    https://portal.co.dallas.tx.us

Data:   Active probate cases from Dallas County Probate Courts No. 1–4.
        Covers independent/dependent estate administration, guardianships,
        and will probate applications where real property is commonly sold.

Signal: indicator_type = "probate"
        Active probate = property owner deceased; heirs often need to
        liquidate real estate quickly to settle debts or divide assets.
        Dependent administrations (court-supervised) are highest-value
        because a forced sale is more likely.

Address resolution:
        The case search API does not return property addresses. This scraper
        makes a second API call per case to fetch the case detail/parties,
        then uses the decedent/testator's address as the property address.
        Falls back to a "Probate {case_number}" placeholder if unavailable
        (ingestor stores the indicator but geocoder will skip it).

Session: Tyler Odyssey is a JavaScript SPA. We mimic the browser flow:
  1. GET portal homepage → obtains session cookie
  2. GET /PortalService/api/portal/ → portal config (may include node list)
  3. POST /PortalService/api/Case/CaseSearch → active probate case list
  4. POST /PortalService/api/Case/CaseDetail (per case) → party addresses

Filter: statusType="A" (Active only). Lookback window: 2 years to capture
        long-running estate administrations still in progress.

Rate limit: 15 req/min — conservative for court-operated infrastructure.

Note:   If this scraper returns 0 records, verify the Odyssey endpoint paths
        by inspecting Network tab at portal.co.dallas.tx.us in a browser.
        Field names may vary between Odyssey versions (PascalCase vs camelCase).
"""

from __future__ import annotations

import structlog
from datetime import date, datetime, timedelta
from typing import AsyncIterator

import httpx

from scrapers.base import BaseCountyScraper, RawIndicatorRecord

logger = structlog.get_logger(__name__)

COUNTY_FIPS = "48113"
PORTAL_BASE = "https://portal.co.dallas.tx.us"
PORTAL_INIT_URL = f"{PORTAL_BASE}/PortalService/api/portal/"
CASE_SEARCH_URL = f"{PORTAL_BASE}/PortalService/api/Case/CaseSearch"
CASE_DETAIL_URL = f"{PORTAL_BASE}/PortalService/api/Case/CaseDetail"

LOOKBACK_DAYS = 730  # 2 years of active cases

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Content-Type": "application/json",
    "Origin": PORTAL_BASE,
    "Referer": f"{PORTAL_BASE}/",
}

# Party role names that indicate the decedent/property owner in probate
_DECEDENT_ROLES = {
    "DECEDENT", "TESTATOR", "DECEASED", "ESTATE", "WARD",
    "DECEDANT",  # common typo in court systems
}


def _coerce(d: dict, *keys: str, default: str = "") -> str:
    """Try multiple key names (PascalCase, camelCase, snake_case) and return first hit."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            return str(v).strip()
    return default


class DallasProbateScraper(BaseCountyScraper):
    county_fips = COUNTY_FIPS
    source_name = "Dallas County Probate Courts (Tyler Odyssey)"
    indicator_types = ["probate"]
    rate_limit_per_minute = 15

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        """
        1. Establish Odyssey session.
        2. Search active probate cases over rolling 2-year window.
        3. For each case, fetch party detail to extract property address.
        4. Yield one RawIndicatorRecord per active estate.
        """
        total = 0

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(connect=15, read=30, write=15, pool=15),
        ) as client:
            # --- Step 1: Session init ---
            try:
                await client.get(PORTAL_BASE + "/", headers=_HEADERS)
                await self._rate_limit_sleep()
            except httpx.HTTPError as exc:
                logger.error("dallas_probate_session_failed", error=str(exc))
                return

            # --- Step 2: Portal config (best-effort; session cookie already set) ---
            try:
                await client.get(PORTAL_INIT_URL, headers=_HEADERS)
                await self._rate_limit_sleep()
            except httpx.HTTPError:
                pass  # non-fatal; session cookie from step 1 may suffice

            # --- Step 3: Case search ---
            start_date = (date.today() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
            end_date = date.today().strftime("%Y-%m-%d")

            search_payload = {
                "nodeId": "PROBATE",
                "nodeDesc": "Probate",
                "category": "PR",
                "statusType": "A",          # Active cases only
                "filingDateStart": start_date,
                "filingDateEnd": end_date,
                "lastName": "",
                "firstName": "",
                "caseNumber": "",
                "dateOfBirth": "",
            }

            try:
                resp = await client.post(
                    CASE_SEARCH_URL,
                    json=search_payload,
                    headers=_HEADERS,
                )
                await self._rate_limit_sleep()
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                logger.error(
                    "dallas_probate_search_failed",
                    error=str(exc)[:300],
                    hint="Verify endpoint at portal.co.dallas.tx.us via browser DevTools",
                )
                return

            # Odyssey may wrap results in {"cases": [...]} or return a bare list
            if isinstance(data, list):
                cases = data
            elif isinstance(data, dict):
                cases = (
                    data.get("cases")
                    or data.get("result")
                    or data.get("Results")
                    or data.get("data")
                    or []
                )
            else:
                cases = []

            if not cases:
                logger.warning(
                    "dallas_probate_no_results",
                    response_type=type(data).__name__,
                    response_keys=list(data.keys()) if isinstance(data, dict) else "N/A",
                    hint="Check statusType, nodeId, and date range in search_payload",
                )
                return

            logger.info("dallas_probate_cases_found", count=len(cases))

            # --- Step 4: Enrich each case with party address ---
            for case in cases:
                record = await self._build_record(client, case)
                if record and await self.validate_record(record):
                    yield record
                    total += 1

        logger.info("dallas_probate_complete", total_yielded=total)

    async def _build_record(
        self, client: httpx.AsyncClient, case: dict
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

        # Attempt to get decedent address from case detail parties
        address_raw: str | None = None
        if case_id:
            address_raw = await self._get_decedent_address(client, case_id)

        if not address_raw:
            # Placeholder — record is stored but won't geocode until enriched
            address_raw = f"Probate {case_number}, Dallas County, TX"

        # Parse "ESTATE OF JOHN DOE" → owner name "John Doe"
        owner_name: str | None = None
        upper = style_name.upper()
        if upper.startswith("ESTATE OF "):
            owner_name = style_name[10:].strip().title()
        elif style_name:
            owner_name = style_name.title()

        return RawIndicatorRecord(
            indicator_type="probate",
            address_raw=address_raw,
            county_fips=self.county_fips,
            owner_name=owner_name,
            filing_date=filing_date,
            case_number=case_number,
            source_url=f"{PORTAL_BASE}/",
            raw_payload={
                "case_number": case_number,
                "style_name": style_name,
                "case_id": case_id,
                "court_id": _coerce(case, "CourtID", "courtId", "court_id"),
                "case_type": _coerce(case, "CaseTypeDesc", "caseTypeDesc", "case_type_desc"),
                "status": _coerce(case, "StatusDesc", "statusDesc", "status_desc", default="Active"),
                "filing_date": raw_date,
            },
        )

    async def _get_decedent_address(
        self, client: httpx.AsyncClient, case_id: str
    ) -> str | None:
        """
        Fetch case detail and extract the decedent/testator's street address.
        Returns a formatted address string or None.
        """
        try:
            resp = await client.post(
                CASE_DETAIL_URL,
                json={"caseId": case_id, "nodeId": "PROBATE"},
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

            if not any(r in role for r in _DECEDENT_ROLES):
                continue

            addr = _coerce(party, "Address", "address", "streetAddress")
            city = _coerce(party, "City", "city")
            state = _coerce(party, "State", "state", default="TX")
            zip_code = _coerce(party, "Zip", "zip", "ZipCode", "zipCode")

            if addr and city:
                full = f"{addr}, {city}, {state}"
                if zip_code:
                    full += f" {zip_code}"
                return full
            elif addr:
                return f"{addr}, Dallas, TX"

        return None
