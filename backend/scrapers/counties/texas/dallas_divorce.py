"""
Dallas County Divorce Scraper (Texas - FIPS 48113)

Source: Dallas County District Clerk — Tyler Odyssey Case Management Portal
URL:    https://portal.co.dallas.tx.us

Data:   Active divorce/family law cases involving real property disposition.
        Divorce proceedings frequently result in forced home sales when spouses
        cannot agree on how to divide equity — court may order a sale.

Signal: indicator_type = "lien"  (used as a distress/litigation signal)
        Active contested divorce with real property = motivated seller signal.
        Strongest when combined with other indicators (tax delinquency,
        code violations, absentee owner on one spouse's mailing address).

Strategy:
        Uses TylerOdysseyPlaywrightScraper.  The Dallas District Clerk shares
        the same Tyler Odyssey portal as the County Clerk (portal.co.dallas.tx.us).
        Divorce/family cases are in category "FAM" (Family Law).

        Case type codes to try, in order:
            DIV   — Divorce (most common)
            DIVM  — Divorce with minor children
            FAM   — General family (broader, may include custody-only)

        The scraper captures petitioner/respondent names and the petitioner's
        address as a proxy for the property address.  Many divorce filings
        include the property address in the party detail.

Address: Extracted from the petitioner's party record in case detail.
        Falls back to "Divorce {case_number}, Dallas County, TX" placeholder.

Lookback: 180 days — divorce cases run 6-18 months; we want active ones
        that are likely to result in a sale in the near term.

Rate: 10 req/min — conservative for District Clerk portal.

Note:   If this scraper returns 0 results, verify the nodeId and category
        by opening portal.co.dallas.tx.us, searching for a known divorce case,
        and checking the Network tab for the CaseSearch request payload.
        The nodeId may be "DISTRICT" or "DCCV" depending on the Odyssey config.
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
PORTAL_BASE = "https://courtsportal.dallascounty.org/DALLASPROD"
LOOKBACK_DAYS = 180

# nodeId values to try — Dallas ePortal uses full location strings from dropdown
_NODE_IDS = ["District Courts - Family", "DISTRICT", "DCCV", "DC", "DISTRICTCLERK", ""]

# Case type codes to try for divorce/family
_DIVORCE_CASE_TYPES = ["DIV", "DIVM", "FAM", "DIVP"]

# Party roles for the petitioner (property owner / motivated seller)
_PETITIONER_ROLES = {"PETITIONER", "PLAINTIFF", "PET", "PLF"}


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


class DallasDivorceScraper(TylerOdysseyPlaywrightScraper):
    portal_base = PORTAL_BASE
    county_fips = COUNTY_FIPS
    source_name = "Dallas County District Clerk Divorce (Tyler Odyssey Playwright)"
    indicator_types = ["lien"]
    rate_limit_per_minute = 10

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        pairs = await self._try_search()

        if not pairs:
            logger.warning(
                "dallas_divorce_api_exhausted",
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

        logger.info("dallas_divorce_complete", total_yielded=total)

    async def _fetch_via_hearing_search(self) -> AsyncIterator[RawIndicatorRecord]:
        """Fallback: scrape Hearing Search UI for family/divorce court hearings."""
        cases = await self._get_cases_via_hearing_search(
            court_location="District Courts - Family",
            lookback_days=30,
            lookahead_days=60,
        )
        if not cases:
            logger.warning("dallas_divorce_hearing_search_empty")
            return

        total = 0
        for row in cases:
            record = self._build_record_from_hearing_row(row)
            if record and await self.validate_record(record):
                yield record
                total += 1
                await self._rate_limit_sleep()

        logger.info("dallas_divorce_hearing_complete", total_yielded=total)

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

        # Petitioner from "SMITH VS JONES" style
        petitioner_name: str | None = None
        if " VS " in style_name.upper():
            petitioner_name = style_name.upper().split(" VS ")[0].strip().title()
        elif style_name:
            petitioner_name = style_name.title()

        address_raw = case_url or f"Divorce {case_number}, Dallas County, TX"

        return RawIndicatorRecord(
            indicator_type="lien",
            address_raw=address_raw,
            county_fips=self.county_fips,
            owner_name=petitioner_name,
            filing_date=filing_date,
            case_number=case_number,
            source_url=f"{PORTAL_BASE}/",
            raw_payload={
                "case_number": case_number,
                "style_name": style_name,
                "case_url": case_url,
                "source": "hearing_search",
                "signal": "divorce_active",
                **{k: v for k, v in row.items() if not k.startswith("_")},
            },
        )

    async def _try_search(self) -> list[tuple[dict, dict]]:
        """Try multiple nodeId + case type combinations until one returns results."""
        for node_id in _NODE_IDS:
            for case_type in _DIVORCE_CASE_TYPES:
                pairs = await self._get_cases_with_details(
                    category="FAM",
                    node_id=node_id,
                    status_type="A",
                    lookback_days=LOOKBACK_DAYS,
                    case_type_id=case_type,
                    detail_node_id=node_id,
                )
                if pairs:
                    logger.info(
                        "dallas_divorce_hit",
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
        address_raw = _extract_address(parties, _PETITIONER_ROLES)
        if not address_raw:
            address_raw = f"Divorce {case_number}, Dallas County, TX"

        # Petitioner is typically the party initiating the divorce
        petitioner_name: str | None = None
        for party in parties or []:
            role = _coerce(
                party, "PartyType", "partyType", "party_type", "RoleDesc", "roleDesc"
            ).upper()
            if any(r in role for r in _PETITIONER_ROLES):
                petitioner_name = _coerce(
                    party, "FullName", "fullName", "name", "Name"
                ).title() or None
                break
        if not petitioner_name and style_name:
            # "SMITH, JOHN vs SMITH, JANE" → extract first party
            vs_parts = style_name.split(" VS ")
            if vs_parts:
                petitioner_name = vs_parts[0].strip().title()

        return RawIndicatorRecord(
            indicator_type="lien",
            address_raw=address_raw,
            county_fips=self.county_fips,
            owner_name=petitioner_name,
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
                "signal": "divorce_active",
            },
        )
