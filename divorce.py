"""
Dallas County Divorce / Family Court Scraper
=============================================
Source: Dallas County District Clerk case search
  https://www.dallascounty.org/departments/districtclerk/

Searches for recently filed divorce and family law cases in Dallas County
District Courts (Family Law divisions: Courts 301-308).

Case types targeted:
  - DIVORCE
  - DIVORCE W/CHILDREN
  - SUIT AFFECTING PARENT CHILD RELATIONSHIP (SAPCR)

Why this matters for real estate leads:
  - Divorcing couples frequently need to sell the family home
  - Marital dissolution often triggers forced property sales
  - Court records are public and include party addresses

Stores to:
  gov_pull_raw  (source_type=divorce_filing)
  gov_leads     (source_type=divorce_filing)

Env vars required:
  SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
Optional:
  GCS_BUCKET
  DIVORCE_LOOKBACK_DAYS  (default: 30)

NOTE: Dallas County courts may use Tyler Technologies Odyssey portal or
their own system. If this endpoint changes, update DISTRICT_CLERK_SEARCH_URL
and _build_search_params() below.
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

SOURCE_TYPE = "divorce_filing"

# Dallas County District Clerk online case access
DISTRICT_CLERK_SEARCH_URL = "https://www.dallascounty.org/departments/districtclerk/case-search.php"
DISTRICT_CLERK_BASE = "https://www.dallascounty.org"

# Family law case type codes (adjust to match the clerk portal's values)
DIVORCE_CASE_TYPES = [
    "DIVORCE",
    "DIVORCE W/CHILDREN",
    "SAPCR",
]

# Family law court numbers in Dallas County
FAMILY_COURTS = ["301", "302", "303", "304", "305", "306", "307", "308"]

_DALLAS_CITIES = re.compile(
    r"\b(DALLAS|IRVING|GARLAND|MESQUITE|BALCH SPRINGS|SEAGOVILLE|HUTCHINS|WILMER|"
    r"FERRIS|LANCASTER|DESOTO|CEDAR HILL|DUNCANVILLE|GRAND PRAIRIE|ROWLETT|"
    r"SUNNYVALE|SACHSE|FARMERS BRANCH|CARROLLTON|ADDISON|RICHARDSON)\b",
    re.IGNORECASE,
)


# -----------------------------
# Helpers
# -----------------------------

def _build_search_params(
    case_type: str,
    date_from: str,
    date_to: str,
    court: str = "",
) -> Dict[str, str]:
    """
    Build POST/GET parameters for the district clerk case search.
    Adjust field names to match the actual portal form.
    """
    params = {
        "caseType":    case_type,
        "dateFrom":    date_from,     # MM/DD/YYYY
        "dateTo":      date_to,
        "court":       court,
        "searchType":  "F",           # F = Filed date range
        "maxResults":  "500",
        "submit":      "Search",
    }
    return params


def _search_case_type(
    session: requests.Session,
    case_type: str,
    date_from: str,
    date_to: str,
) -> List[Dict[str, str]]:
    """Search for one case type and return parsed records."""
    params = _build_search_params(case_type, date_from, date_to)
    try:
        # Try GET first; some portals use GET with query params
        resp = session.get(
            DISTRICT_CLERK_SEARCH_URL,
            params=params,
            timeout=60,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DallasLeadScraper/1.0)"},
        )
        if resp.status_code == 405:
            # Fall back to POST
            resp = session.post(
                DISTRICT_CLERK_SEARCH_URL,
                data=params,
                timeout=60,
                headers={"User-Agent": "Mozilla/5.0 (compatible; DallasLeadScraper/1.0)"},
            )
        resp.raise_for_status()
    except Exception as e:
        return [{"_error": str(e), "_case_type": case_type}]

    return _parse_case_table(resp.text, case_type)


def _parse_case_table(html: str, case_type: str) -> List[Dict[str, str]]:
    """
    Parse HTML case search results table.
    Typical columns: Case Number, Filed Date, Petitioner, Respondent, Court, Status
    """
    soup = BeautifulSoup(html, "lxml")
    records: List[Dict[str, str]] = []

    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue

        header_cells = [th.get_text(strip=True).upper() for th in rows[0].find_all(["th", "td"])]
        if not any(kw in " ".join(header_cells) for kw in ["CASE", "PETITIONER", "FILED", "PARTY"]):
            continue

        col_map: Dict[str, int] = {}
        for i, h in enumerate(header_cells):
            for key, keywords in {
                "case_num":    ["CASE NO", "CASE NUM", "CAUSE"],
                "filed_date":  ["FILED", "DATE"],
                "petitioner":  ["PETITIONER", "PLAINTIFF", "PARTY1"],
                "respondent":  ["RESPONDENT", "DEFENDANT", "PARTY2"],
                "court":       ["COURT"],
                "status":      ["STATUS", "DISP"],
                "address":     ["ADDRESS", "ADDR"],
            }.items():
                if any(kw in h for kw in keywords) and key not in col_map:
                    col_map[key] = i

        for row in rows[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            if not cells:
                continue
            rec = {
                "case_type":   case_type,
                "case_num":    cells[col_map["case_num"]]   if "case_num"   in col_map and len(cells) > col_map["case_num"]   else "",
                "filed_date":  cells[col_map["filed_date"]] if "filed_date" in col_map and len(cells) > col_map["filed_date"] else "",
                "petitioner":  cells[col_map["petitioner"]] if "petitioner" in col_map and len(cells) > col_map["petitioner"] else "",
                "respondent":  cells[col_map["respondent"]] if "respondent" in col_map and len(cells) > col_map["respondent"] else "",
                "court":       cells[col_map["court"]]      if "court"      in col_map and len(cells) > col_map["court"]      else "",
                "status":      cells[col_map["status"]]     if "status"     in col_map and len(cells) > col_map["status"]     else "",
                "address":     cells[col_map["address"]]    if "address"    in col_map and len(cells) > col_map["address"]    else "",
            }
            if rec["case_num"] or rec["petitioner"]:
                records.append(rec)

    return records


def _fetch_case_detail(
    session: requests.Session,
    case_num: str,
) -> Optional[str]:
    """
    Optionally fetch individual case detail page to get party addresses.
    Returns the raw HTML or None on failure.
    """
    detail_url = f"{DISTRICT_CLERK_BASE}/departments/districtclerk/case-detail.php"
    try:
        resp = session.get(
            detail_url,
            params={"caseNum": case_num},
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DallasLeadScraper/1.0)"},
        )
        if resp.ok:
            return resp.text
    except Exception:
        pass
    return None


def _extract_address_from_detail(html: str) -> Optional[str]:
    """Parse a party's mailing address from the case detail page."""
    soup = BeautifulSoup(html, "lxml")
    # Look for address patterns near "Address:" labels
    for label in soup.find_all(string=re.compile(r"address", re.IGNORECASE)):
        parent = label.parent
        if parent:
            sibling = parent.find_next_sibling()
            if sibling:
                text = sibling.get_text(strip=True)
                if re.search(r"\d+\s+\w", text):
                    return text
    # Fallback: find any address-like string in the page
    text = soup.get_text(" ")
    m = re.search(
        r"\d+\s+[A-Z][A-Za-z\s]+\b(?:ST|AVE|DR|LN|BLVD|RD|CT|PL|WAY|CIR|TRL|HWY|PKWY)\b[^,\n]*",
        text,
        re.IGNORECASE,
    )
    return m.group(0).strip() if m else None


def _parse_address(raw: str) -> Dict[str, str]:
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
    fetch_details: bool = False,
    **_kwargs,
) -> Dict[str, Any]:
    """
    Search Dallas County District Clerk for recent divorce filings and upsert to Supabase.

    Args:
        batch_size:     Supabase upsert batch size.
        export_mode:    "none" | "clean"
        debug:          Include sample_lead in return value.
        lookback_days:  Days back to search (default: env DIVORCE_LOOKBACK_DAYS or 30).
        fetch_details:  If True, fetch individual case pages to extract party addresses.
                        Slower but provides richer data. Default False.
    """
    if lookback_days is None:
        lookback_days = int(os.getenv("DIVORCE_LOOKBACK_DAYS", "30"))

    now = datetime.now(timezone.utc)
    date_to = now.strftime("%m/%d/%Y")
    date_from = (now - timedelta(days=lookback_days)).strftime("%m/%d/%Y")

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; DallasLeadScraper/1.0)"})

    all_records: List[Dict[str, str]] = []
    for ct in DIVORCE_CASE_TYPES:
        recs = _search_case_type(session, ct, date_from, date_to)
        all_records.extend(recs)

    now_iso = now.isoformat()
    raw_rows: List[Dict[str, Any]] = []
    lead_rows: List[Dict[str, Any]] = []
    clean_rows: List[Dict[str, Any]] = []
    seen_keys: set = set()

    for rec in all_records:
        if "_error" in rec:
            continue

        case_num = rec.get("case_num", "").strip()
        filed_date = rec.get("filed_date", "").strip()
        petitioner = rec.get("petitioner", "").strip()
        respondent = rec.get("respondent", "").strip()
        raw_address = rec.get("address", "").strip()

        # Optionally fetch case detail for party address
        if fetch_details and case_num and not raw_address:
            detail_html = _fetch_case_detail(session, case_num)
            if detail_html:
                raw_address = _extract_address_from_detail(detail_html) or ""

        source_key = case_num or f"divorce_{petitioner}_{filed_date}"

        if raw_address and re.search(r"\d", raw_address):
            addr = _parse_address(raw_address)
            street = addr["street"]
            city = addr["city"]
            state = addr["state"]
            zip5 = addr["zip"]
        else:
            street = city = state = zip5 = ""

        payload = {
            "case_type":    rec.get("case_type", "DIVORCE"),
            "case_num":     case_num,
            "filed_date":   filed_date,
            "petitioner":   petitioner,
            "respondent":   respondent,
            "court":        rec.get("court", "").strip(),
            "status":       rec.get("status", "").strip(),
            "address":      street,
            "city":         city,
            "state":        state,
            "zip":          zip5,
            "search_date_from": date_from,
            "search_date_to":   date_to,
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
                    "tags": ["divorce"],
                    "source_event_date": filed_date or now_iso[:10],
                    "raw_source_key": source_key,
                    "source_url": DISTRICT_CLERK_SEARCH_URL,
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
            prefix="dallas-divorce",
            stable_name="dallas-divorce/divorce_latest.csv",
            clean_rows=clean_rows,
        )

    return {
        "ok": True,
        "source": SOURCE_TYPE,
        "lookback_days": lookback_days,
        "case_types_searched": len(DIVORCE_CASE_TYPES),
        "raw_records": len(all_records),
        "raw_rows": len(raw_rows),
        "leads_upserted": len(lead_rows),
        "exports": exports,
        "sample_lead": lead_rows[0] if debug and lead_rows else None,
    }
