"""
Dallas County Foreclosure Notice Scraper
=========================================
Source: https://www.dallascounty.org/departments/dallascounty/foreclosures.php

In Texas, Trustee's Sale notices (foreclosure) are posted with the county clerk
and published monthly. Dallas County publishes PDF lists of properties scheduled
for the first Tuesday foreclosure auction.

This scraper:
  1. Fetches the foreclosure listing page and finds the most recent PDF link(s).
  2. Downloads and parses each PDF using pdfplumber.
  3. Upserts records into gov_pull_raw and gov_leads (source_type=foreclosure_notice).

Env vars required:
  SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
Optional:
  GCS_BUCKET — enables CSV export
"""

import io
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from core import (
    chunks,
    dedupe_key,
    export_clean_to_gcs,
    gcs_upload_bytes,
    normalize_zip,
    supabase_upsert,
    write_csv_bytes,
)

SOURCE_TYPE = "foreclosure_notice"
FORECLOSURE_PAGE = "https://www.dallascounty.org/departments/dallascounty/foreclosures.php"
BASE_URL = "https://www.dallascounty.org"

# Dallas County cities for address parsing
_DALLAS_CITIES = re.compile(
    r"\b(DALLAS|IRVING|GARLAND|MESQUITE|BALCH SPRINGS|SEAGOVILLE|HUTCHINS|WILMER|"
    r"FERRIS|LANCASTER|DESOTO|CEDAR HILL|DUNCANVILLE|GRAND PRAIRIE|ROWLETT|"
    r"SUNNYVALE|SACHSE|FARMERS BRANCH|CARROLLTON|ADDISON|RICHARDSON|"
    r"UNIVERSITY PARK|HIGHLAND PARK)\b",
    re.IGNORECASE,
)


# -----------------------------
# Helpers
# -----------------------------

def _find_pdf_links(html: str) -> List[str]:
    """Return all PDF URLs found on the foreclosure page."""
    soup = BeautifulSoup(html, "lxml")
    links: List[str] = []
    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        if not href.lower().endswith(".pdf"):
            continue
        if href.startswith("http"):
            links.append(href)
        else:
            links.append(BASE_URL + "/" + href.lstrip("/"))
    return links


def _prefer_current(links: List[str]) -> List[str]:
    """Sort PDF links to prefer the current year/month."""
    now = datetime.now(timezone.utc)
    yr, mo = str(now.year), f"{now.month:02d}"

    def score(u: str) -> int:
        s = 0
        if yr in u:
            s += 2
        if mo in u:
            s += 1
        return s

    return sorted(links, key=score, reverse=True)


def _parse_pdf(pdf_bytes: bytes) -> List[Dict[str, str]]:
    """
    Parse a Dallas County foreclosure sale PDF.
    Returns a list of raw record dicts with keys:
      seq, owner, address, legal, trustee, sale_date_raw
    """
    if pdfplumber is None:
        raise RuntimeError("pdfplumber not installed — add it to requirements.txt")

    records: List[Dict[str, str]] = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    for row in table:
                        if not row or not any(cell for cell in row if cell):
                            continue
                        cells = [str(c or "").strip() for c in row]
                        # Skip obvious header rows
                        first = cells[0].lower() if cells else ""
                        if any(kw in first for kw in ["seq", "#", "no.", "property", "owner"]):
                            continue
                        records.append({
                            "seq":          cells[0] if len(cells) > 0 else "",
                            "owner":        cells[1] if len(cells) > 1 else "",
                            "address":      cells[2] if len(cells) > 2 else "",
                            "legal":        cells[3] if len(cells) > 3 else "",
                            "trustee":      cells[4] if len(cells) > 4 else "",
                            "sale_date_raw": cells[5] if len(cells) > 5 else "",
                        })
            else:
                # Fallback: text extraction — parse line-by-line blocks
                text = page.extract_text() or ""
                for block in re.split(r"\n{2,}", text):
                    block = block.strip()
                    if not block:
                        continue
                    lines = [l.strip() for l in block.splitlines() if l.strip()]
                    # Heuristic: first line with a digit and street keyword is the address
                    address = ""
                    owner = lines[0] if lines else ""
                    for line in lines:
                        if re.search(r"\d+\s+\w", line) and re.search(
                            r"\b(ST|AVE|DR|LN|BLVD|RD|CT|PL|WAY|CIR|TRL|HWY|FWY|PKY|PKWY)\b",
                            line, re.IGNORECASE
                        ):
                            address = line
                            break
                    if address:
                        records.append({
                            "seq": "",
                            "owner": owner,
                            "address": address,
                            "legal": "",
                            "trustee": "",
                            "sale_date_raw": "",
                        })

    return records


def _normalize(rec: Dict[str, str], pdf_url: str, sale_date_str: str) -> Optional[Dict[str, Any]]:
    """
    Clean up a raw PDF record into a normalized address record.
    Returns None if we can't extract a usable street address.
    """
    raw_address = rec.get("address", "").strip()
    if not raw_address:
        return None

    # Strip leading sequence numbers  ("1.", "12)", etc.)
    raw_address = re.sub(r"^\d+[.)]\s*", "", raw_address)

    # Try to extract ZIP
    zip5 = ""
    zip_m = re.search(r"\b(\d{5})(?:-\d{4})?\b", raw_address)
    if zip_m:
        zip5 = zip_m.group(1)
        raw_address = raw_address[: zip_m.start()].strip().rstrip(",")

    # Try to identify city
    city = "DALLAS"
    state = "TX"
    city_m = _DALLAS_CITIES.search(raw_address)
    if city_m:
        city = city_m.group(1).upper()
        raw_address = raw_address[: city_m.start()].strip().rstrip(",")

    street = re.sub(r"\s+", " ", raw_address).strip()
    if not street or not re.search(r"\d", street):
        return None

    return {
        "street": street,
        "city": city,
        "state": state,
        "zip": normalize_zip(zip5),
        "owner": rec.get("owner", "").strip(),
        "legal": rec.get("legal", "").strip(),
        "trustee": rec.get("trustee", "").strip(),
        "seq": rec.get("seq", "").strip(),
        "sale_date": sale_date_str,
        "pdf_url": pdf_url,
    }


def _extract_sale_date(pdf_url: str, html: str) -> str:
    """Best-effort extraction of the sale date from URL or page HTML."""
    # Try URL pattern  e.g.  2026-01  or  Jan2026  or  01_2026
    m = re.search(r"(\d{4}[-_/]\d{2}|\d{2}[-_/]\d{4})", pdf_url)
    if m:
        return m.group(1).replace("_", "-").replace("/", "-")
    # Try page HTML near the PDF link
    idx = html.find(os.path.basename(pdf_url))
    if idx > 0:
        snippet = html[max(0, idx - 200): idx + 200]
        m2 = re.search(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b", snippet, re.IGNORECASE)
        if m2:
            return m2.group(0)
    return ""


# -----------------------------
# Public entry point
# -----------------------------

def run(
    batch_size: int = 500,
    export_mode: str = "none",
    debug: bool = False,
    max_pdfs: int = 1,
    **_kwargs,
) -> Dict[str, Any]:
    """
    Scrape Dallas County foreclosure sale notices and upsert to Supabase.

    Args:
        batch_size:  Supabase upsert batch size.
        export_mode: "none" | "clean" | "both"
        debug:       Include sample_lead in return value.
        max_pdfs:    How many PDFs to process (default 1 = most recent).
    """
    # 1. Fetch listing page
    resp = requests.get(
        FORECLOSURE_PAGE,
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0 (compatible; DallasLeadScraper/1.0)"},
    )
    resp.raise_for_status()
    html = resp.text

    pdf_links = _find_pdf_links(html)
    if not pdf_links:
        return {
            "ok": False,
            "source": SOURCE_TYPE,
            "error": "No PDF links found on foreclosure page. Site layout may have changed.",
        }

    pdf_links = _prefer_current(pdf_links)[:max_pdfs]

    raw_rows: List[Dict[str, Any]] = []
    lead_rows: List[Dict[str, Any]] = []
    clean_rows: List[Dict[str, Any]] = []
    seen_keys: set = set()
    total_parsed = 0

    for pdf_url in pdf_links:
        # 2. Download PDF
        pdf_resp = requests.get(
            pdf_url,
            timeout=180,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DallasLeadScraper/1.0)"},
        )
        pdf_resp.raise_for_status()

        # 3. Parse PDF records
        raw_records = _parse_pdf(pdf_resp.content)
        total_parsed += len(raw_records)
        sale_date_str = _extract_sale_date(pdf_url, html)
        now_iso = datetime.now(tz=timezone.utc).isoformat()

        for i, rec in enumerate(raw_records):
            norm = _normalize(rec, pdf_url, sale_date_str)
            if not norm:
                continue

            street = norm["street"]
            city = norm["city"]
            state = norm["state"]
            zip5 = norm["zip"]
            dk = dedupe_key(street, city, state, zip5)
            if dk in seen_keys:
                continue
            seen_keys.add(dk)

            source_key = norm["seq"] or f"{pdf_url}#{i}"

            raw_rows.append({
                "source_type": SOURCE_TYPE,
                "source_key": source_key,
                "source_event_date": sale_date_str or now_iso[:10],
                "payload": {
                    "owner": norm["owner"],
                    "address": street,
                    "city": city,
                    "state": state,
                    "zip": zip5,
                    "legal_description": norm["legal"],
                    "trustee": norm["trustee"],
                    "sale_date": sale_date_str,
                    "seq": norm["seq"],
                    "pdf_url": pdf_url,
                },
            })

            lead_rows.append({
                "source_type": SOURCE_TYPE,
                "last_seen_at": now_iso,
                "address": street.upper(),
                "city": city.upper(),
                "state": state,
                "zip": zip5,
                "tags": ["foreclosure"],
                "source_event_date": sale_date_str or now_iso[:10],
                "raw_source_key": source_key,
                "source_url": pdf_url,
                "dedupe_key": dk,
            })

            clean_rows.append({"Street Address": street, "City": city, "State": state, "Zip": zip5})

    # 4. Upsert
    for chunk in chunks(raw_rows, batch_size):
        supabase_upsert("gov_pull_raw", chunk, "source_type,source_key")
    for chunk in chunks(lead_rows, batch_size):
        supabase_upsert("gov_leads", chunk, "dedupe_key")

    # 5. GCS export
    exports: Dict[str, str] = {}
    bucket = os.getenv("GCS_BUCKET", "").strip()
    if export_mode != "none" and bucket:
        exports = export_clean_to_gcs(
            bucket,
            prefix="dallas-foreclosure",
            stable_name="dallas-foreclosure/foreclosure_latest.csv",
            clean_rows=clean_rows,
        )

    return {
        "ok": True,
        "source": SOURCE_TYPE,
        "pdfs_processed": len(pdf_links),
        "records_parsed": total_parsed,
        "raw_rows": len(raw_rows),
        "leads_upserted": len(lead_rows),
        "exports": exports,
        "sample_lead": lead_rows[0] if debug and lead_rows else None,
    }
