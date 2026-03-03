"""
Dallas County Foreclosure Notice Scraper (Texas - FIPS 48113)

Source: Dallas County Clerk Recording Division
Data:   Notice of Trustee's Sale (NOTS) — Texas non-judicial foreclosure notices
URL:    https://www.dallascounty.org/government/county-clerk/recording/foreclosures.php

How Texas foreclosure works:
  Texas uses non-judicial (deed-of-trust) foreclosure. The lender files a Notice
  of Trustee's Sale (NOTS) at least 21 days before the auction. Auctions happen
  on the first Tuesday of every month at the George Allen Courts Building in
  downtown Dallas. The county clerk posts the NOTS as PDF files organized by
  month and city.

What we capture:
  - indicator_type = "pre_foreclosure"  (weight 40 in scoring engine)
  - Property address, owner/grantor name, sale date, trustee sale number

Rate limit: 10 req/min — conservative for the county website.
"""

from __future__ import annotations

import io
import structlog
import re
from datetime import date, datetime
from typing import AsyncIterator
from urllib.parse import urljoin

import httpx
import pdfplumber
from bs4 import BeautifulSoup

from scrapers.address_utils import is_blocklisted_address, looks_like_property_address
from scrapers.base import BaseCountyScraper, RawIndicatorRecord

logger = structlog.get_logger(__name__)

BASE_URL = "https://www.dallascounty.org"
FORECLOSURE_PAGE = f"{BASE_URL}/government/county-clerk/recording/foreclosures.php"

# Cities in Dallas County — used as fallback address detection
DALLAS_COUNTY_CITIES = (
    "Dallas|Garland|Irving|Richardson|Mesquite|Grand Prairie|Carrollton|"
    "Plano|Rowlett|Cedar Hill|DeSoto|Duncanville|Balch Springs|Farmers Branch|"
    "Addison|Lancaster|Glenn Heights|Hutchins|Seagoville|Sunnyvale|"
    "Cockrell Hill|Highland Park|University Park|Wilmer|Combine|Coppell"
)


class DallasCountyForeclosureScraper(BaseCountyScraper):
    county_fips = "48113"
    source_name = "Dallas County Clerk — Notice of Trustee's Sale"
    indicator_types = ["pre_foreclosure"]
    rate_limit_per_minute = 10

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        """
        Scrape the foreclosure notices page, download each PDF, and parse NOTS records.

        The county posts PDFs organized by month and city:
          /department/countyclerk/media/foreclosure/{Month}/{City}_{N}.pdf
        """
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            pdf_urls = await self._get_pdf_urls(client)
            logger.info("[dallas_foreclosure] found %d PDFs", len(pdf_urls))

            for pdf_url in pdf_urls:
                try:
                    records = await self._download_and_parse_pdf(client, pdf_url)
                    for record in records:
                        if await self.validate_record(record):
                            yield record
                    await self._rate_limit_sleep()
                except Exception as exc:
                    logger.error(
                        "dallas_foreclosure_pdf_failed", url=pdf_url, error=str(exc)
                    )

    async def _get_pdf_urls(self, client: httpx.AsyncClient) -> list[str]:
        """Fetch the foreclosure notices page and extract all PDF links."""
        try:
            resp = await client.get(FORECLOSURE_PAGE)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("dallas_foreclosure_page_failed", error=str(exc))
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        pdf_urls: list[str] = []

        for link in soup.find_all("a", href=True):
            href: str = link["href"]
            # Match county clerk foreclosure PDF pattern
            if "/foreclosure/" in href.lower() and href.lower().endswith(".pdf"):
                full_url = urljoin(BASE_URL, href)
                pdf_urls.append(full_url)

        # Deduplicate while preserving order
        return list(dict.fromkeys(pdf_urls))

    async def _download_and_parse_pdf(
        self, client: httpx.AsyncClient, pdf_url: str
    ) -> list[RawIndicatorRecord]:
        """Download a foreclosure PDF and extract one NOTS record per page."""
        try:
            resp = await client.get(pdf_url)
        except httpx.HTTPError as exc:
            logger.warning("dallas_foreclosure_download_failed", url=pdf_url, error=str(exc))
            return []

        if resp.status_code != 200:
            logger.warning("dallas_foreclosure_pdf_not_found", url=pdf_url, status=resp.status_code)
            return []

        records: list[RawIndicatorRecord] = []

        try:
            with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text() or ""
                        record = self._parse_nots_page(text, pdf_url)
                        if record:
                            records.append(record)
                    except Exception as exc:
                        logger.warning(
                            "dallas_foreclosure_page_parse_failed",
                            url=pdf_url,
                            page=page_num,
                            error=str(exc),
                        )
        except Exception as exc:
            logger.error("dallas_foreclosure_pdf_open_failed", url=pdf_url, error=str(exc))

        return records

    def _parse_nots_page(self, text: str, source_url: str) -> RawIndicatorRecord | None:
        """
        Parse text from a single Notice of Trustee's Sale page.

        Texas Property Code §51.002 requires NOTS to include:
          - Grantor (borrower/owner) name
          - Property description (legal + street address)
          - Sale date (first Tuesday of month)
          - Trustee name and contact
        """
        if not text or "TRUSTEE" not in text.upper():
            return None

        # Require some minimum content to reduce false positives
        if len(text.strip()) < 100:
            return None

        owner_name = self._extract_grantor(text)
        address_raw = self._extract_address(text)
        filing_date = self._extract_sale_date(text)
        case_number = self._extract_case_number(text)
        amount = self._extract_amount(text)

        if not address_raw:
            return None

        address_raw = re.sub(r"\s+", " ", address_raw).strip().rstrip(".,;")

        # Reject courthouse / auction venue addresses that appear in the "sale location"
        # section of the NOTS — these are NOT the property being foreclosed on.
        if is_blocklisted_address(address_raw):
            logger.debug("dallas_foreclosure_blocklisted_address", address=address_raw)
            return None

        # Require address to look like a real street address (starts with a number)
        if not looks_like_property_address(address_raw):
            logger.debug("dallas_foreclosure_invalid_address", address=address_raw)
            return None

        return RawIndicatorRecord(
            indicator_type="pre_foreclosure",
            address_raw=address_raw,
            county_fips=self.county_fips,
            owner_name=owner_name,
            amount=amount,
            filing_date=filing_date,
            case_number=case_number,
            source_url=source_url,
            raw_payload={"text_preview": text[:600], "source_url": source_url},
        )

    def _extract_grantor(self, text: str) -> str | None:
        """Extract the grantor (borrower/property owner) name."""
        patterns = [
            r"Grantor[s]?:\s*(.+?)(?:\n|Trustor|Borrower|Property|Legal|Said|Lender)",
            r"Trustor[s]?:\s*(.+?)(?:\n|Grantor|Borrower|Property|Legal|Said)",
            r"Borrower[s]?:\s*(.+?)(?:\n|Grantor|Trustor|Property|Legal|Said)",
            # Some TX NOTS put borrower in preamble: "...to [TRUSTEE], as Trustee for [OWNER]..."
            r"(?:executed\s+by|made\s+by)\s+(.+?)(?:,|\n|to\s+\w+\s+Trustee)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                val = match.group(1).strip()
                # Sanity check: owner name shouldn't be a whole paragraph
                if 3 < len(val) < 120 and "\n\n" not in val:
                    return val
        return None

    def _extract_address(self, text: str) -> str | None:
        """Extract the property street address."""
        # Try labeled address fields first
        labeled_patterns = [
            r"[Cc]ommonly [Kk]nown [Aa]s:?\s*(.+?)(?:\n|Legal|Said|Parcel|APN|$)",
            r"[Pp]roperty [Aa]ddress:?\s*(.+?)(?:\n|Legal|Said|Parcel|APN|$)",
            r"[Ss]treet [Aa]ddress:?\s*(.+?)(?:\n|Legal|Said|Parcel|APN|$)",
            r"[Ll]ocated at:?\s*(.+?)(?:\n|Legal|Said|Parcel|APN|$)",
        ]
        for pattern in labeled_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                val = match.group(1).strip()
                if 10 < len(val) < 200:
                    return val

        # Fallback: find any street address mentioning a Dallas County city
        return self._find_dallas_address(text)

    def _find_dallas_address(self, text: str) -> str | None:
        """Regex to match a Dallas County street address anywhere in the text."""
        pattern = (
            rf"\d{{2,6}}\s+[\w\s]{{5,50}},"
            rf"\s*(?:{DALLAS_COUNTY_CITIES})"
            rf"(?:,\s*(?:TX|Texas))?"
            rf"(?:\s+\d{{5}})?"
        )
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
        return None

    def _extract_sale_date(self, text: str) -> date | None:
        """Extract the trustee sale date."""
        patterns = [
            r"[Dd]ate of [Ss]ale:?\s*(\w+\s+\d{1,2},?\s*\d{4})",
            r"[Ss]ale [Dd]ate:?\s*(\w+\s+\d{1,2},?\s*\d{4})",
            r"will\s+be\s+sold\s+on\s+\w+,?\s+(\w+\s+\d{1,2},?\s*\d{4})",
            r"(\w+\s+\d{1,2},\s*\d{4})\s+at\s+10:00",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    raw = re.sub(r"\s+", " ", match.group(1)).strip().replace(",", "")
                    return datetime.strptime(raw, "%B %d %Y").date()
                except ValueError:
                    pass
        return None

    def _extract_case_number(self, text: str) -> str | None:
        """Extract the trustee sale number or case number."""
        patterns = [
            r"(?:Trustee Sale No\.?|TS No\.?|T\.S\. No\.?|Sale No\.?)\s*[:#]?\s*([A-Z0-9\-]+)",
            r"(?:File No\.?|Loan No\.?)\s*[:#]?\s*([A-Z0-9\-]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _extract_amount(self, text: str) -> float | None:
        """Extract the loan or outstanding debt amount."""
        patterns = [
            r"[Oo]riginal\s+[Ll]oan\s+[Aa]mount\s*:?\s*\$?([\d,]+(?:\.\d{2})?)",
            r"[Uu]npaid\s+[Bb]alance\s*:?\s*\$?([\d,]+(?:\.\d{2})?)",
            r"[Oo]bligations?\s+(?:in\s+the\s+sum\s+of|totaling|of)\s+\$?([\d,]+(?:\.\d{2})?)",
            r"\$\s*([\d,]{4,}(?:\.\d{2})?)\s+(?:plus|with|including)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1).replace(",", ""))
                except ValueError:
                    pass
        return None
