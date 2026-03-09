"""
Dallas County Probate Court Scraper
=====================================
Source: Dallas County Clerk online records search
  https://countyclerk.dallascounty.org/

Searches for probate-related instrument types filed within the last N days:
  - PROBATE WILL
  - LETTERS TESTAMENTARY
  - LETTERS OF ADMINISTRATION
  - MUNIMENT OF TITLE
  - AFFIDAVIT OF HEIRSHIP

Probate records are useful for:
  - Locating properties in estate that may need to be sold
  - Finding heirs who may want to sell inherited real estate

Stores to:
  gov_pull_raw  (source_type=probate_filing)
  gov_leads     (source_type=probate_filing)

Env vars required:
  SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
Optional:
  GCS_BUCKET
  PROBATE_LOOKBACK_DAYS  (default: 30)

NOTE: If the county clerk portal is JavaScript-rendered, swap the
requests + BeautifulSoup approach here for Playwright. Install with:
  pip install playwright && playwright install chromium
"""

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from core import (
    chunks,
    dedupe_key,
    export_clean_to_gcs,
    normalize_zip,
    supabase_upsert,
)

SOURCE_TYPE = "probate_filing"

# Dallas County Clerk online records portal
CLERK_SEARCH_URL = "https://countyclerk.dallascounty.org/cgi-bin/cty_public_qry_03.cgi"
CLERK_BASE = "https://countyclerk.dallascounty.org"

# Instrument types that indicate a probate estate
PROBATE_INSTRUMENT_TYPES = [
    "PROBATE",
    "LETTERS TESTAMENTARY",
    "LETTERS OF ADMINISTRATION",
    "MUNIMENT OF TITLE",
    "AFFIDAVIT OF HEIRSHIP",
    "SMALL ESTATE AFFIDAVIT",
]

_DALLAS_CITIES = re.compile(
    r"\b(DALLAS|IRVING|GARLAND|MESQUITE|BALCH SPRINGS|SEAGOVILLE|HUTCHINS|WILMER|"
    r"FERRIS|LANCASTER|DESOTO|CEDAR HILL|DUNCANVILLE|GRAND PRAIRIE|ROWLETT|"
    r"SUNNYVALE|SACHSE|FARMERS BRANCH|CARROLLTON|ADDISON|RICHARDSON)\b",
    re.IGNORECASE,
)


# -----------------------------
# Helpers
# -----------------------------

def _build_search_params(instrument_type: str, date_from: str, date_to: str) -> Dict[str, str]:
    """Build POST form parameters for the county clerk search."""
    return {
        "directSearch": "N",
        "searchType": "D",          # Document type search
        "instrCode": instrument_type,
        "dateFrom": date_from,      # MM/DD/YYYY
        "dateTo": date_to,
        "maxRows": "500",
        "submit": "Search",
    }


def _search_instrument(
    session: requests.Session,
    instrument_type: str,
    date_from: str,
    date_to: str,
) -> List[Dict[str, str]]:
    """
    POST to county clerk search and parse the HTML results table.
    Returns a list of raw record dicts.
    """
    params = _build_search_params(instrument_type, date_from, date_to)
    try:
        resp = session.post(
            CLERK_SEARCH_URL,
            data=params,
            timeout=60,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DallasLeadScraper/1.0)"},
        )
        resp.raise_for_status()
    except Exception as e:
        return [{"_error": str(e), "_instrument": instrument_type}]

    return _parse_results_table(resp.text, instrument_type)


def _parse_results_table(html: str, instrument_type: str) -> List[Dict[str, str]]:
    """Parse the HTML results table from the clerk's search portal."""
    soup = BeautifulSoup(html, "lxml")
    records: List[Dict[str, str]] = []

    # The clerk portal typically renders results in a <table> with rows of data.
    # Column order (typical): Instrument#, Book/Page, Date, Grantor, Grantee, Legal
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        # Try to identify header row
        header_cells = [th.get_text(strip=True).upper() for th in rows[0].find_all(["th", "td"])]
        if not any(kw in " ".join(header_cells) for kw in ["INSTRUMENT", "DATE", "GRANTOR", "NAME"]):
            continue
        # Map column names to indices
        col_map: Dict[str, int] = {}
        for i, h in enumerate(header_cells):
            for key, keywords in {
                "instrument_num": ["INSTRUMENT", "INSTR", "DOC"],
                "filed_date":     ["DATE", "FILED"],
                "grantor":        ["GRANTOR", "DECEDENT", "NAME", "FROM"],
                "grantee":        ["GRANTEE", "TO", "HEIR"],
                "legal":          ["LEGAL", "DESCRIPTION", "PROPERTY"],
                "address":        ["ADDRESS", "ADDR"],
            }.items():
                if any(kw in h for kw in keywords) and key not in col_map:
                    col_map[key] = i
        for row in rows[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            if not cells:
                continue
            rec = {
                "instrument_type": instrument_type,
                "instrument_num":  cells[col_map["instrument_num"]] if "instrument_num" in col_map and len(cells) > col_map["instrument_num"] else "",
                "filed_date":      cells[col_map["filed_date"]]     if "filed_date"     in col_map and len(cells) > col_map["filed_date"]     else "",
                "grantor":         cells[col_map["grantor"]]        if "grantor"        in col_map and len(cells) > col_map["grantor"]        else "",
                "grantee":         cells[col_map["grantee"]]        if "grantee"        in col_map and len(cells) > col_map["grantee"]        else "",
                "legal":           cells[col_map["legal"]]          if "legal"          in col_map and len(cells) > col_map["legal"]          else "",
                "address":         cells[col_map["address"]]        if "address"        in col_map and len(cells) > col_map["address"]        else "",
            }
            if rec["grantor"] or rec["instrument_num"]:
                records.append(rec)

    return records


def _extract_address_from_legal(legal: str) -> Optional[str]:
    """
    Try to pull a street address from a legal description.
    Probate records often include the property address in the legal field.
    """
    if not legal:
        return None
    m = re.search(
        r"\d+\s+[A-Z][A-Za-z0-9\s]+\b(?:ST|AVE|DR|LN|BLVD|RD|CT|PL|WAY|CIR|TRL|HWY|PKWY|PKY)\b[^\n,]*",
        legal,
        re.IGNORECASE,
    )
    return m.group(0).strip() if m else None


def _parse_address(raw: str) -> Dict[str, str]:
    """Parse raw address string into components."""
    city, state, zip5 = "DALLAS", "TX", ""

    zip_m = re.search(r"\b(\d{5})(?:-\d{4})?\b", raw)
    if zip_m:
        zip5 = zip_m.group(1)
        raw = raw[: zip_m.start()].strip().rstrip(",")

    city_m = _DALLAS_CITIES.search(raw)
    if city_m:
        city = city_m.group(1).upper()
        raw = raw[: city_m.start()].strip().rstrip(",")

    street = re.sub(r"\s+", " ", raw).strip()
    return {"street": street, "city": city, "state": state, "zip": normalize_zip(zip5)}


# -----------------------------
# Public entry point
# -----------------------------

def run(
    batch_size: int = 500,
    export_mode: str = "none",
    debug: bool = False,
    lookback_days: Optional[int] = None,
    **_kwargs,
) -> Dict[str, Any]:
    """
    Search Dallas County Clerk for recent probate instrument filings and
    upsert leads into Supabase.

    Args:
        batch_size:    Supabase upsert batch size.
        export_mode:   "none" | "clean"
        debug:         Include sample_lead in return value.
        lookback_days: How many days back to search (default: env PROBATE_LOOKBACK_DAYS or 30).
    """
    if lookback_days is None:
        lookback_days = int(os.getenv("PROBATE_LOOKBACK_DAYS", "30"))

    now = datetime.now(timezone.utc)
    date_to = now.strftime("%m/%d/%Y")
    date_from = (now - timedelta(days=lookback_days)).strftime("%m/%d/%Y")

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; DallasLeadScraper/1.0)"})

    all_records: List[Dict[str, str]] = []
    for instr in PROBATE_INSTRUMENT_TYPES:
        recs = _search_instrument(session, instr, date_from, date_to)
        all_records.extend(recs)

    now_iso = now.isoformat()
    raw_rows: List[Dict[str, Any]] = []
    lead_rows: List[Dict[str, Any]] = []
    clean_rows: List[Dict[str, Any]] = []
    seen_keys: set = set()

    for rec in all_records:
        if "_error" in rec:
            continue  # log errors but continue

        # Try address field first, then extract from legal description
        raw_address = rec.get("address", "").strip()
        if not raw_address:
            raw_address = _extract_address_from_legal(rec.get("legal", "")) or ""

        # For probate, property address is often not in the clerk record —
        # we still store the grantor (decedent) name as a lead without full address.
        # If no address, create a partial lead keyed by instrument number.
        instrument_num = rec.get("instrument_num", "").strip()
        filed_date = rec.get("filed_date", "").strip()
        grantor = rec.get("grantor", "").strip()
        instr_type = rec.get("instrument_type", SOURCE_TYPE)

        if raw_address and re.search(r"\d", raw_address):
            addr = _parse_address(raw_address)
            street = addr["street"]
            city = addr["city"]
            state = addr["state"]
            zip5 = addr["zip"]
        else:
            # No parseable address — skip as a lead but store in raw
            street = city = state = zip5 = ""

        source_key = instrument_num or f"probate_{grantor}_{filed_date}"

        payload = {
            "instrument_type": instr_type,
            "instrument_num": instrument_num,
            "filed_date": filed_date,
            "decedent_name": grantor,
            "grantee": rec.get("grantee", "").strip(),
            "legal_description": rec.get("legal", "").strip(),
            "address": street,
            "city": city,
            "state": state,
            "zip": zip5,
            "search_date_from": date_from,
            "search_date_to": date_to,
        }

        raw_rows.append({
            "source_type": SOURCE_TYPE,
            "source_key": source_key,
            "source_event_date": filed_date or now_iso[:10],
            "payload": payload,
        })

        if street and city:
            dk = dedupe_key(street, city, state, zip5)
            if dk not in seen_keys:
                seen_keys.add(dk)
                lead_rows.append({
                    "source_type": SOURCE_TYPE,
                    "last_seen_at": now_iso,
                    "address": street.upper(),
                    "city": city.upper(),
                    "state": state,
                    "zip": zip5,
                    "tags": ["probate"],
                    "source_event_date": filed_date or now_iso[:10],
                    "raw_source_key": source_key,
                    "source_url": CLERK_SEARCH_URL,
                    "dedupe_key": dk,
                })
                clean_rows.append({"Street Address": street, "City": city, "State": state, "Zip": zip5})

    # Upsert
    for chunk in chunks(raw_rows, batch_size):
        supabase_upsert("gov_pull_raw", chunk, "source_type,source_key")
    for chunk in chunks(lead_rows, batch_size):
        supabase_upsert("gov_leads", chunk, "dedupe_key")

    # GCS export
    exports: Dict[str, str] = {}
    bucket = os.getenv("GCS_BUCKET", "").strip()
    if export_mode != "none" and bucket and clean_rows:
        exports = export_clean_to_gcs(
            bucket,
            prefix="dallas-probate",
            stable_name="dallas-probate/probate_latest.csv",
            clean_rows=clean_rows,
        )

    return {
        "ok": True,
        "source": SOURCE_TYPE,
        "lookback_days": lookback_days,
        "instrument_types_searched": len(PROBATE_INSTRUMENT_TYPES),
        "raw_records": len(all_records),
        "raw_rows": len(raw_rows),
        "leads_upserted": len(lead_rows),
        "exports": exports,
        "sample_lead": lead_rows[0] if debug and lead_rows else None,
    }
