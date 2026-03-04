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

Strategy:
        Uses TylerOdysseyPlaywrightScraper which navigates to the portal
        as a real browser (setting the required session cookie), then uses
        the browser's own cookie jar to POST to the internal case search API.
        This is more reliable than direct httpx calls, which fail when the
        SPA requires a session established by the initial page load.

Address: The case search API does not include property addresses. The
        _get_cases_with_details() method fetches each case's party list and
        extracts the decedent/testator's mailing address as a proxy for the
        property address.

Rate: 15 req/min via BaseCountyScraper._rate_limit_sleep().
"""

from __future__ import annotations

import structlog
from datetime import date, datetime
from typing import AsyncIterator

from scrapers.base import RawIndicatorRecord
from scrapers.counties.texas.tyler_odyssey_base import TylerOdysseyPlaywrightScraper

logger = structlog.get_logger(__name__)

COUNTY_FIPS = "48113"
PORTAL_BASE = "https://courtsportal.dallascounty.org/DALLASPROD"
LOOKBACK_DAYS = 730  # 2 years — probate estates are long-running

# nodeId values to try (Dallas County ePortal uses full strings from location dropdown)
_NODE_IDS = ["County Courts - Probate", "PROBATE", "PR", "", "29"]

# Party role substrings that identify the decedent / property owner
_DECEDENT_ROLES = {
    "DECEDENT", "TESTATOR", "DECEASED", "ESTATE", "WARD", "DECEDANT",
}


def _coerce(d: dict, *keys: str, default: str = "") -> str:
    for k in keys:
        v = d.get(k)
        if v is not None:
            return str(v).strip()
    return default


def _extract_address(parties: list[dict], role_set: set[str]) -> str | None:
    """Return the formatted address of the first party whose role matches."""
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
        if addr and city:
            full = f"{addr}, {city}, {state}"
            if zip_code:
                full += f" {zip_code}"
            return full
        if addr:
            return f"{addr}, Dallas, TX"
    return None


class DallasProbateScraper(TylerOdysseyPlaywrightScraper):
    portal_base = PORTAL_BASE
    county_fips = COUNTY_FIPS
    source_name = "Dallas County Probate Courts (Tyler Odyssey Playwright)"
    indicator_types = ["probate"]
    rate_limit_per_minute = 15

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        pairs = await self._try_search()

        if not pairs:
            logger.warning(
                "dallas_probate_no_cases",
                hint=(
                    "No probate cases found on any nodeId/category combination. "
                    "Open courtsportal.dallascounty.org/DALLASPROD in a browser, "
                    "search a known probate case, and check DevTools → Network for "
                    "the actual CaseSearch request payload to verify nodeId and category."
                ),
            )
            return

        total = 0
        for case, detail in pairs:
            record = self._build_record(case, detail)
            if record and await self.validate_record(record):
                yield record
                total += 1

        logger.info("dallas_probate_complete", total_yielded=total)

    async def _try_search(self) -> list[tuple[dict, dict]]:
        """Try multiple nodeId and category combinations until one returns results."""
        for node_id in _NODE_IDS:
            for category in ["PR", "CV", "FAM"]:
                pairs = await self._get_cases_with_details(
                    category=category,
                    node_id=node_id,
                    status_type="A",
                    lookback_days=LOOKBACK_DAYS,
                    detail_node_id=node_id,
                )
                if pairs:
                    logger.info(
                        "dallas_probate_hit",
                        node_id=node_id,
                        category=category,
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

        # Extract decedent address from detail parties
        parties = (
            detail.get("parties")
            or detail.get("Parties")
            or detail.get("caseParties")
            or []
        )
        address_raw = _extract_address(parties, _DECEDENT_ROLES)
        if not address_raw:
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
                "case_type": _coerce(case, "CaseTypeDesc", "caseTypeDesc"),
                "status": _coerce(case, "StatusDesc", "statusDesc", default="Active"),
                "filing_date": raw_date,
            },
        )
