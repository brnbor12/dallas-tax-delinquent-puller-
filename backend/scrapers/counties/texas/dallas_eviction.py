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
    portal_base = CC_PORTAL_BASE
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
        """Fallback: scrape Hearing Search UI for JP court FED hearings.

        Evictions are filed by landlords — many of whom are business entities
        (LLCs, property management companies). BusinessName search works without
        requiring FirstName, so we sweep A–Z to cover business plaintiffs.

        PartyName is also attempted with empty FirstName to catch individual
        landlords (though this usually fails server-side validation).
        """
        import string
        seen_case_numbers: set[str] = set()
        cases: list[dict] = []

        # BusinessName sweep A–Z — catches LLC/corporate landlords
        biz_hit = False
        for letter in string.ascii_lowercase:
            letter_cases = await self._get_cases_via_hearing_search(
                court_location="Justice of the Peace Courts",
                lookback_days=LOOKBACK_DAYS,
                search_by_type="BusinessName",
                search_value=letter,
            )
            if letter_cases:
                logger.info("dallas_eviction_hearing_biz_hit",
                            letter=letter, count=len(letter_cases))
                cases.extend(letter_cases)
                biz_hit = True

        if not biz_hit:
            logger.info("dallas_eviction_hearing_biz_sweep_empty",
                        hint="No BusinessName results for JP courts")

        # PartyName probe — catches individual landlords if server accepts empty FirstName
        probe = await self._get_cases_via_hearing_search(
            court_location="Justice of the Peace Courts",
            lookback_days=LOOKBACK_DAYS,
            search_by_type="PartyName",
            search_value="smith",
            first_name="",
        )
        if probe:
            cases.extend(probe)
            for letter in string.ascii_lowercase:
                if letter == "s":
                    continue
                letter_cases = await self._get_cases_via_hearing_search(
                    court_location="Justice of the Peace Courts",
                    lookback_days=LOOKBACK_DAYS,
                    search_by_type="PartyName",
                    search_value=letter,
                    first_name="",
                )
                if letter_cases:
                    cases.extend(letter_cases)

        if not cases:
            logger.warning("dallas_eviction_hearing_search_empty")
            return

        total = 0
        for row in cases:
            case_number = str(row.get("CaseNumber") or "").strip()
            if not case_number:
                import re as _re
                for k in row:
                    if _re.search(r"case.?no|case.?number|casenum", k, _re.I):
                        case_number = str(row[k]).strip()
                        break
            if case_number and case_number in seen_case_numbers:
                continue
            if case_number:
                seen_case_numbers.add(case_number)

            record = self._build_record_from_hearing_row(row)
            if record and await self.validate_record(record):
                yield record
                total += 1
                await self._rate_limit_sleep()

        logger.info("dallas_eviction_hearing_complete", total_yielded=total)

    def _build_record_from_hearing_row(self, row: dict) -> RawIndicatorRecord | None:
        """Build RawIndicatorRecord from a Hearing Search JSON row (Read endpoint)."""
        # JSON format from HearingResults/Read: CaseNumber, Style, HearingDate, CaseLoadUrl
        case_number = str(row.get("CaseNumber") or "").strip()
        if not case_number:
            for k in row:
                if re.search(r"case.?no|case.?number|casenum", k, re.I):
                    case_number = str(row[k]).strip()
                    break

        if not case_number:
            return None

        style_name = str(row.get("Style") or row.get("SortStyleOrDefendant") or "").strip()
        if not style_name:
            for k in row:
                if re.search(r"style|defendant|caption|parties", k, re.I):
                    val = row[k]
                    if isinstance(val, str):
                        style_name = val.strip()
                        break

        case_url = str(row.get("CaseLoadUrl") or "").strip()
        if not case_url:
            enc = row.get("EncryptedCaseId") or ""
            case_id = row.get("CaseId") or ""
            if enc:
                case_url = f"{CC_PORTAL_BASE}/Case/CaseDetail?eid={enc}"
            elif case_id:
                case_url = f"{CC_PORTAL_BASE}/Case/CaseDetail?caseId={case_id}"

        hearing_date_str = str(row.get("HearingDate") or "").strip()

        filing_date: date | None = None
        if hearing_date_str:
            m = re.search(r"/Date\((\d+)\)/", hearing_date_str)
            if m:
                import datetime as _dt
                filing_date = _dt.datetime.fromtimestamp(
                    int(m.group(1)) / 1000, tz=_dt.timezone.utc
                ).date()
            else:
                for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y"):
                    try:
                        filing_date = datetime.strptime(hearing_date_str[:10], fmt).date()
                        break
                    except ValueError:
                        continue

        # Plaintiff = landlord. Style format: "LANDLORD VS TENANT" or "TENANT, LANDLORD"
        plaintiff = ""
        upper = style_name.upper()
        if " VS " in upper:
            plaintiff = style_name[:upper.index(" VS ")].strip().title()

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
        """Try multiple node IDs and FED case type codes on the county clerk portal."""
        node_ids = ["Justice of the Peace Courts", "JP", "JPC", ""]

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
