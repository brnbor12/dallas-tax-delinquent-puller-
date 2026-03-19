"""
Hillsborough County Lis Pendens Scraper (Florida - FIPS 12057)

Source: Hillsborough County Clerk of Circuit Court — Official Records Daily Index
Portal: https://publicrec.hillsclerk.com/OfficialRecords/DailyIndexes/

Data:   Lis Pendens (LP) instruments recorded in Official Records.
        In Florida, foreclosure actions begin with a Lis Pendens filing.
        LP records put the world on notice that the property is subject to
        pending litigation (most commonly foreclosure or title dispute).

Signal: indicator_type = "lien"

Strategy:
        Three pipe-delimited files are published per business day:
          D{YYYYMMDD}01id.29 — Document file: one row per recorded instrument
              Fields: Action|County|InstrumentNumber|DocType|DocDesc|LegalDesc|
                      BookType|BookNumber|PageNumber|Filler|PageCount|
                      DateRecorded|TimeRecorded|ConsiderationAmount
          P{YYYYMMDD}01id.29 — Party file: grantors and grantees per instrument
              Fields: Action|County|Sequence|From/To|PartyName
          M{YYYYMMDD}01d.29  — Map file: DocType → FACC code (tiny, ignored)

        We download D+P files for the last LOOKBACK_DAYS business days,
        filter D rows for DocType "LP" (Lis Pendens), join party names,
        and yield one record per LP instrument.

Address: LPs carry a legal description, not a street address.
         We store a placeholder using the instrument number and legal description
         snippet. Downstream address enrichment can match via parcel lookup.

Rate:   Files are ~150-300KB each — simple HTTP downloads.
"""

from __future__ import annotations

import re
import structlog
from datetime import date, datetime, timedelta
from typing import AsyncIterator

import httpx

from scrapers.base import BaseCountyScraper, RawIndicatorRecord

logger = structlog.get_logger(__name__)

COUNTY_FIPS = "12057"
OR_BASE = "https://publicrec.hillsclerk.com/OfficialRecords/DailyIndexes"
SOURCE_URL = "https://publicrec.hillsclerk.com/OfficialRecords/DailyIndexes/"

LOOKBACK_DAYS = 90

# Document types to capture (lis pendens family)
_LP_DOC_TYPES = {"LP", "NLP", "LPE", "LPNOT", "LPREL"}


def _parse_date(raw: str) -> date | None:
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _parse_pipe_file(content: str) -> list[list[str]]:
    """Parse pipe-delimited lines, returning list of field lists."""
    rows = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            rows.append(line.split("|"))
    return rows


async def _download_text(client: httpx.AsyncClient, url: str) -> str | None:
    try:
        resp = await client.get(url, timeout=30.0, follow_redirects=True)
        if resp.status_code == 200 and len(resp.content) > 200:
            return resp.content.decode("latin-1", errors="replace")
        return None
    except Exception as exc:
        logger.debug("hillsborough_lp_download_failed", url=url[-60:], error=str(exc)[:60])
        return None


class HillsboroughLisPendensScraper(BaseCountyScraper):
    county_fips = COUNTY_FIPS
    source_name = "Hillsborough County Clerk Official Records — Lis Pendens"
    indicator_types = ["lien"]
    rate_limit_per_minute = 60

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        today = date.today()
        cutoff = today - timedelta(days=LOOKBACK_DAYS)

        # Collect LP instruments: instrument_number -> {doc fields + parties}
        instruments: dict[str, dict] = {}

        async with httpx.AsyncClient(
            headers={"User-Agent": "Mozilla/5.0 (compatible; motivated-seller-bot/1.0)"},
        ) as client:
            for days_back in range(0, LOOKBACK_DAYS + 5):
                ref_date = today - timedelta(days=days_back)
                if ref_date.weekday() >= 5:  # skip weekends
                    continue
                if ref_date < cutoff:
                    break

                date_str = ref_date.strftime("%Y%m%d")
                doc_url = f"{OR_BASE}/D{date_str}01id.29"
                party_url = f"{OR_BASE}/P{date_str}01id.29"

                # Download document file first
                doc_content = await _download_text(client, doc_url)
                if not doc_content:
                    continue  # file not published yet

                # Parse document rows and filter for LP types
                day_instruments: dict[str, dict] = {}
                for fields in _parse_pipe_file(doc_content):
                    if len(fields) < 12:
                        continue
                    action = fields[0].strip()
                    if action == "DDD":  # deleted record
                        continue
                    instrument_num = fields[2].strip()
                    doc_type = fields[3].strip().upper()
                    doc_desc = fields[4].strip()
                    legal_desc = fields[5].strip()
                    date_recorded_raw = fields[11].strip() if len(fields) > 11 else ""
                    consideration_raw = fields[13].strip() if len(fields) > 13 else ""

                    if doc_type not in _LP_DOC_TYPES:
                        continue
                    if not instrument_num:
                        continue

                    rec_date = _parse_date(date_recorded_raw)
                    if rec_date and rec_date < cutoff:
                        continue

                    amount = None
                    try:
                        amount = float(consideration_raw) if consideration_raw else None
                    except ValueError:
                        pass

                    day_instruments[instrument_num] = {
                        "instrument_num": instrument_num,
                        "doc_type": doc_type,
                        "doc_desc": doc_desc,
                        "legal_desc": legal_desc[:300],
                        "date_recorded": date_recorded_raw,
                        "rec_date": rec_date,
                        "amount": amount,
                        "grantors": [],  # from = grantor (borrower/property owner)
                        "grantees": [],  # to = grantee (lender)
                    }

                if not day_instruments:
                    continue

                logger.debug("hillsborough_lp_day_instruments",
                             date=date_str, lp_count=len(day_instruments))

                # Download party file and attach names
                party_content = await _download_text(client, party_url)
                if party_content:
                    # Party file has sequence numbers that map to instruments
                    # by position order matching D file rows. We track by sequence.
                    # Fields: Action|County|Sequence|From/To|PartyName
                    # Sequence is NOT the instrument number directly.
                    # We need to match parties to instruments by sequence order.
                    # The D file and P file share the same County+Sequence pairing.
                    # However, the sequence in P file matches the order of D file rows.
                    # For simplicity, we collect all grantors and associate by
                    # building a sequence->instrument mapping from the D file position.

                    # Actually, per the README: Party file has Sequence field that
                    # corresponds to the row ordering in the D file for that day.
                    # We build an ordered list of instrument numbers from the D file.
                    doc_sequence = []
                    for fields in _parse_pipe_file(doc_content):
                        if len(fields) < 3:
                            continue
                        if fields[0].strip() == "DDD":
                            continue
                        instr = fields[2].strip()
                        if instr:
                            doc_sequence.append(instr)

                    for fields in _parse_pipe_file(party_content):
                        if len(fields) < 5:
                            continue
                        if fields[0].strip() == "DPD":
                            continue
                        try:
                            seq = int(fields[2].strip()) - 1  # 1-based to 0-based
                        except (ValueError, IndexError):
                            continue
                        from_to = fields[3].strip().upper()  # FROM=grantor, TO=grantee
                        party_name = fields[4].strip().title() if len(fields) > 4 else ""

                        if seq < len(doc_sequence):
                            instr = doc_sequence[seq]
                            if instr in day_instruments:
                                if from_to == "FROM":
                                    day_instruments[instr]["grantors"].append(party_name)
                                elif from_to == "TO":
                                    day_instruments[instr]["grantees"].append(party_name)

                # Merge into master dict (skip duplicates)
                for instr, data in day_instruments.items():
                    if instr not in instruments:
                        instruments[instr] = data

        logger.info("hillsborough_lp_instruments_found", count=len(instruments))

        total = 0
        for instr, data in instruments.items():
            record = self._build_record(data)
            if record and await self.validate_record(record):
                yield record
                total += 1

        if total == 0:
            logger.warning("hillsborough_lp_no_records")
        else:
            logger.info("hillsborough_lp_complete", total_yielded=total)

    def _build_record(self, data: dict) -> RawIndicatorRecord | None:
        instr = data["instrument_num"]
        if not instr:
            return None

        grantors = data.get("grantors", [])
        grantees = data.get("grantees", [])
        legal_desc = data.get("legal_desc", "")

        # Primary grantor = property owner (borrower filing LP or being foreclosed upon)
        owner_name = grantors[0] if grantors else None

        # LP filings use legal descriptions, not street addresses
        # We store a placeholder for downstream enrichment
        address_raw = f"LP {instr}, Hillsborough County, FL"
        if legal_desc:
            address_raw = f"LP {instr} — {legal_desc[:80]}, Hillsborough County, FL"

        rec_date = data.get("rec_date")

        return RawIndicatorRecord(
            indicator_type="lien",
            address_raw=address_raw,
            county_fips=self.county_fips,
            owner_name=owner_name,
            filing_date=rec_date,
            case_number=instr,
            amount=data.get("amount"),
            source_url=SOURCE_URL,
            raw_payload={
                "instrument_number": instr,
                "doc_type": data.get("doc_type", "LP"),
                "doc_desc": data.get("doc_desc", "LIS PENDENS"),
                "legal_description": legal_desc,
                "date_recorded": data.get("date_recorded", ""),
                "grantors": grantors[:5],
                "grantees": grantees[:5],
            },
        )
