"""
Manual data import endpoint.

Accepts CSV uploads with property/lead data and ingests them through the
standard ingestor pipeline (dedup, normalize address, create indicator).

Use this for data sources that are IP-blocked from the VPS:
  - Dallas County probate / eviction / divorce (Tyler Odyssey portal exports)
  - Polk County court data (apps.polkcountyclerk.net exports)
  - Any PropStream / ATTOM / skip-trace CSV export

POST /api/v1/import
  Form fields:
    file          — CSV or TSV file
    county_fips   — 5-digit FIPS (e.g. "48113" for Dallas TX)
    indicator_type — e.g. "probate", "eviction", "divorce", "absentee_owner"

Column auto-detection (case-insensitive, flexible):
  Address  : address, property_address, situs, site_address, prop_addr, street
  Owner    : owner, owner_name, grantor, grantee, name, party
  Amount   : amount, balance, tax_due, debt, lien_amount, value
  Date     : date, filing_date, recorded_date, file_date, case_date
  Case #   : case, case_number, case_no, doc_number, instrument
"""
from __future__ import annotations

import csv
import io
import re
from datetime import date, datetime
from typing import Optional

import structlog
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from scrapers.base import RawIndicatorRecord
from scrapers.ingestor import IngestResult, ingest_record

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/import", tags=["import"])

MAX_ROWS = 10_000

# Column name aliases — first match wins
_COL_MAP = {
    "address": ["address", "property_address", "situs", "site_address", "prop_addr",
                "street", "address1", "full_address", "str_address"],
    "owner":   ["owner_name", "owner", "grantor", "grantee", "name", "party",
                "taxpayer", "mailing_name"],
    "amount":  ["amount", "balance", "tax_due", "debt", "lien_amount", "value",
                "assessed_value", "amt"],
    "date":    ["filing_date", "date", "recorded_date", "file_date", "case_date",
                "recorded", "filed"],
    "case":    ["case_number", "case_no", "case", "doc_number", "instrument",
                "doc_no", "file_number"],
    "zip":     ["zip", "zip_code", "zipcode", "postal_code"],
    "city":    ["city", "prop_city", "site_city"],
    "state":   ["state", "prop_state"],
}


def _detect_columns(headers: list[str]) -> dict[str, str | None]:
    """Map logical field names to actual CSV column headers."""
    lower = {h.lower().strip(): h for h in headers}
    result: dict[str, str | None] = {}
    for field, aliases in _COL_MAP.items():
        result[field] = next(
            (lower[a] for a in aliases if a in lower), None
        )
    return result


def _parse_amount(raw: str) -> float | None:
    if not raw:
        return None
    cleaned = re.sub(r"[^\d.]", "", raw)
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def _parse_date(raw: str) -> date | None:
    if not raw:
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y", "%m/%d/%y", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _build_address(row: dict, col: dict, county_fips: str) -> str:
    """Assemble the best possible address string from available columns."""
    addr = (row.get(col["address"] or "", "") or "").strip()
    if not addr:
        return ""

    # Append city/state/zip if not already in address string
    city = (row.get(col["city"] or "", "") or "").strip()
    state = (row.get(col["state"] or "", "") or "").strip()
    zip_code = (row.get(col["zip"] or "", "") or "").strip()

    # Infer state from FIPS if not in row
    if not state:
        state_by_fips = {"48": "TX", "12": "FL", "06": "CA", "36": "NY"}
        state = state_by_fips.get(county_fips[:2], "")

    suffix_parts = [p for p in [city, state, zip_code] if p]
    suffix = ", ".join(suffix_parts)

    if suffix and suffix.lower() not in addr.lower():
        addr = f"{addr}, {suffix}"

    return addr


@router.post("")
async def import_file(
    file: UploadFile = File(...),
    county_fips: str = Form(...),
    indicator_type: str = Form(...),
    source_label: Optional[str] = Form(None),
):
    """
    Ingest a CSV file as motivated seller leads.
    Returns a summary of records processed/created/updated.
    """
    if not county_fips or not indicator_type:
        raise HTTPException(400, "county_fips and indicator_type are required")

    # Read file content
    content = await file.read()
    try:
        text = content.decode("utf-8-sig")  # handle BOM from Excel exports
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    # Parse CSV — try comma then tab delimiter
    sample = text[:4096]
    dialect = "excel-tab" if sample.count("\t") > sample.count(",") else "excel"
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)

    try:
        headers = reader.fieldnames or []
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV headers: {e}")

    if not headers:
        raise HTTPException(400, "File has no headers or is empty")

    col = _detect_columns(list(headers))

    if not col["address"]:
        raise HTTPException(
            400,
            f"Could not detect an address column. Found: {list(headers)}. "
            "Rename your address column to 'address' or 'property_address'."
        )

    # Build a sync DB session (ingestor is synchronous)
    sync_url = settings.database_url.replace("+asyncpg", "+psycopg2")
    engine = create_engine(sync_url, pool_size=2, max_overflow=0)
    Session = sessionmaker(bind=engine)
    session = Session()

    result = IngestResult()
    source_url = source_label or f"manual_import:{file.filename}"

    try:
        for i, row in enumerate(reader):
            if i >= MAX_ROWS:
                break

            addr = _build_address(row, col, county_fips)
            if not addr:
                result.failed += 1
                continue

            record = RawIndicatorRecord(
                indicator_type=indicator_type,
                address_raw=addr,
                county_fips=county_fips,
                owner_name=(row.get(col["owner"] or "", "") or "").strip() or None,
                amount=_parse_amount(row.get(col["amount"] or "", "") or ""),
                filing_date=_parse_date(row.get(col["date"] or "", "") or ""),
                case_number=(row.get(col["case"] or "", "") or "").strip() or None,
                source_url=source_url,
                raw_payload={k: v for k, v in row.items() if v},
            )

            result.found += 1
            ok = ingest_record(session, record, geocode=False)
            if ok:
                result.upserted += 1
            else:
                result.failed += 1

            if result.found % 200 == 0:
                session.commit()
                logger.info(
                    "import_progress",
                    file=file.filename,
                    found=result.found,
                    upserted=result.upserted,
                )

        session.commit()

        # Trigger score recalculation
        from tasks.score_tasks import nightly_score_decay
        nightly_score_decay.delay()

    except Exception as exc:
        session.rollback()
        logger.error("import_failed", error=str(exc))
        raise HTTPException(500, f"Import failed: {exc}")
    finally:
        session.close()
        engine.dispose()

    logger.info(
        "import_complete",
        file=file.filename,
        county_fips=county_fips,
        indicator_type=indicator_type,
        **result.__dict__,
    )

    return {
        "status": "ok",
        "file": file.filename,
        "county_fips": county_fips,
        "indicator_type": indicator_type,
        "rows_read": result.found,
        "imported": result.upserted,
        "failed": result.failed,
        "column_mapping": {k: v for k, v in col.items() if v},
    }


@router.get("/columns")
async def detect_columns_preview(
    file: UploadFile = File(...),
):
    """Preview what columns were auto-detected in the uploaded file."""
    content = await file.read(8192)
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    sample = text[:4096]
    dialect = "excel-tab" if sample.count("\t") > sample.count(",") else "excel"
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)

    headers = list(reader.fieldnames or [])
    col = _detect_columns(headers)

    rows = []
    for i, row in enumerate(reader):
        if i >= 5:
            break
        rows.append(dict(row))

    return {
        "headers": headers,
        "detected_columns": col,
        "preview_rows": rows,
    }