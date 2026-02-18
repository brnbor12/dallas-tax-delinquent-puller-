import os
import re
import io
import csv
import gzip
import zipfile
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

import requests
from fastapi import FastAPI, HTTPException, Query

try:
    from google.cloud import storage  # requires google-cloud-storage
except Exception:
    storage = None  # allow running without GCS deps if not exporting

app = FastAPI()

DALLAS_TAX_ROLL_PAGE = "https://www.dallascounty.org/departments/tax/tax-roll.php"


# -----------------------------
# Utilities
# -----------------------------
def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def normalize_zip(z: str) -> str:
    digits = re.sub(r"\D+", "", (z or ""))
    return digits[:5] if len(digits) >= 5 else digits


def parse_fixed(line: str, start_col_1: int, length: int) -> str:
    """Fixed-width substring. start_col_1 is 1-based."""
    start = max(0, start_col_1 - 1)
    return line[start : start + length].strip()


def parse_yyyymmdd(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not re.fullmatch(r"\d{8}", s):
        return None
    y = int(s[0:4])
    m = int(s[4:6])
    d = int(s[6:8])
    return datetime(y, m, d, tzinfo=timezone.utc)


def numeric_amount(s: str) -> int:
    # TRW amounts are integer-like. Strip anything not digit or minus.
    s = re.sub(r"[^\d-]", "", (s or ""))
    if not s:
        return 0
    try:
        return int(s)
    except Exception:
        return 0


def dedupe_key(address: str, city: str, state: str, zip5: str) -> str:
    norm = lambda x: re.sub(r"\s+", " ", (x or "").strip().lower())
    return f"{norm(address)}|{norm(city)}|{(state or '').upper()}|{normalize_zip(zip5)}"


def find_trw_zip_url(html: str) -> Optional[str]:
    """
    Looks for:
    https://www.dallascounty.org/Assets/uploads/docs/tax/trw/trwfile.<...>.zip
    """
    matches = re.findall(
        r"https?://www\.dallascounty\.org/Assets/uploads/docs/tax/trw/trwfile\.[^\"'\s]+\.zip",
        html,
        flags=re.IGNORECASE,
    )
    if not matches:
        return None
    # prefer non-sample
    for u in matches:
        if "sample" not in u.lower():
            return u
    return matches[0]


def supabase_upsert(table: str, rows: List[Dict[str, Any]], on_conflict: str) -> None:
    if not rows:
        return
    supabase_url = require_env("SUPABASE_URL").rstrip("/")
    service_key = require_env("SUPABASE_SERVICE_ROLE_KEY")

    endpoint = f"{supabase_url}/rest/v1/{table}"
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }
    params = {"on_conflict": on_conflict}

    r = requests.post(endpoint, headers=headers, params=params, json=rows, timeout=180)
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase upsert failed {table}: {r.status_code} {r.text[:800]}")


def chunks(lst: List[Any], n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# -----------------------------
# Address logic (fixes STE-only bug)
# -----------------------------
_UNIT_ONLY_RE = re.compile(r"^\s*(#\s*\w+|(ste|suite|apt|unit)\b\s*[\w-]+)\s*$", re.IGNORECASE)

def is_unit_only_line(line: str) -> bool:
    if not line:
        return False
    return bool(_UNIT_ONLY_RE.match(line.strip()))


def make_street(owner1: str, a2: str, a3: str, a4: str) -> str:
    """
    TRW mailing address lines are often:
    owner1, addr2, addr3, addr4
    Some cases have unit-only lines like "STE 200" that include digits.
    We avoid choosing a unit-only line as the street.
    """
    parts = [p.strip() for p in [owner1, a2, a3, a4] if p and p.strip()]
    if not parts:
        return ""

    # Find first line with a digit that is NOT unit-only (e.g., "123 MAIN ST" ok; "STE 200" not ok)
    idx = -1
    for i, p in enumerate(parts):
        if re.search(r"\d", p) and not is_unit_only_line(p):
            idx = i
            break

    # If we found a plausible street line, use that and anything after it.
    # Otherwise, drop owner1 and keep remaining address lines.
    if idx >= 0:
        street_parts = parts[idx:]
    else:
        street_parts = parts[1:] if len(parts) > 1 else parts

    # Remove any leading unit-only line if it somehow slipped in
    while street_parts and is_unit_only_line(street_parts[0]):
        street_parts = street_parts[1:]

    street = re.sub(r"\s+", " ", " ".join(street_parts)).strip()
    return street


# -----------------------------
# GCS exports
# -----------------------------
def gcs_upload_bytes(bucket_name: str, object_name: str, data: bytes, content_type: str) -> str:
    if storage is None:
        raise RuntimeError("google-cloud-storage not available. Add it to requirements.txt.")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_string(data, content_type=content_type)
    return f"gs://{bucket_name}/{object_name}"


def flatten_payload_rows(raw_rows: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    raw_rows elements look like:
      {"source_type":..., "source_key":..., "source_event_date":..., "payload": {...}}
    We produce:
      columns = ["source_type","source_key","source_event_date", ...payload keys sorted...]
      flattened rows dicts
    """
    payload_keys = set()
    for r in raw_rows:
        payload = r.get("payload") or {}
        payload_keys.update(payload.keys())

    cols = ["source_type", "source_key", "source_event_date"] + sorted(payload_keys)

    flat: List[Dict[str, Any]] = []
    for r in raw_rows:
        payload = r.get("payload") or {}
        out = {
            "source_type": r.get("source_type", ""),
            "source_key": r.get("source_key", ""),
            "source_event_date": r.get("source_event_date", ""),
        }
        for k in payload_keys:
            v = payload.get(k)
            out[k] = "" if v is None else v
        flat.append(out)

    return cols, flat


def write_csv_bytes(fieldnames: List[str], rows: List[Dict[str, Any]]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for row in rows:
        w.writerow(row)
    return buf.getvalue().encode("utf-8")


def export_to_gcs(
    bucket_name: str,
    run_ts: str,
    raw_rows: List[Dict[str, Any]],
    clean_rows: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Writes:
      - FULL: dallas-taxroll-full/YYYY-MM/dallas_tax_delinquent_full_<ts>.csv.gz
      - CLEAN: dallas-taxroll-clean/weekly_clean_<ts>.csv
    """
    dt = datetime.now(timezone.utc)
    ym = dt.strftime("%Y-%m")

    # FULL
    full_cols, full_flat = flatten_payload_rows(raw_rows)
    full_csv = write_csv_bytes(full_cols, full_flat)
    full_gz = gzip.compress(full_csv)

    full_obj = f"dallas-taxroll-full/{ym}/dallas_tax_delinquent_full_{run_ts}.csv.gz"
    full_uri = gcs_upload_bytes(bucket_name, full_obj, full_gz, "application/gzip")

    # CLEAN (keep exactly your current 4 columns)
    clean_cols = ["Street Address", "City", "State", "Zip"]
    clean_csv = write_csv_bytes(clean_cols, clean_rows)
    clean_obj = f"dallas-taxroll-clean/weekly_clean_{run_ts}.csv"
    clean_uri = gcs_upload_bytes(bucket_name, clean_obj, clean_csv, "text/csv")

    # Also write/update a stable pointer file for convenience (optional)
    # weekly_clean.csv
    stable_clean_uri = gcs_upload_bytes(bucket_name, "weekly_clean.csv", clean_csv, "text/csv")

    return {"full": full_uri, "clean": clean_uri, "clean_latest": stable_clean_uri}


# -----------------------------
# Main endpoint
# -----------------------------
@app.get("/run")
def run(
    debug: bool = Query(False),
    max_accounts: int = Query(5000, ge=1, le=50000),
    mailing_zips: str = Query("", description="Comma-separated 5-digit zips to filter MAILING address zip"),
    batch_size: int = Query(500, ge=50, le=2000),
    export_mode: str = Query("both", description="none|clean|full|both. Requires GCS_BUCKET env var for exports."),
):
    """
    Pull Dallas County TRW tax roll, extract delinquent accounts, and upsert into:
      - gov_pull_raw (source_type=tax_delinquent_trw)
      - gov_leads    (source_type=tax_delinquent_trw)

    Optionally exports to GCS:
      - full payload CSV.gz
      - clean 4-col CSV
    """
    try:
        export_mode = (export_mode or "both").strip().lower()
        if export_mode not in {"none", "clean", "full", "both"}:
            raise RuntimeError("export_mode must be one of: none, clean, full, both")

        # 1) Find latest TRW zip link
        html = requests.get(DALLAS_TAX_ROLL_PAGE, timeout=60).text
        trw_zip_url = find_trw_zip_url(html)
        if not trw_zip_url:
            raise RuntimeError("Could not locate TRW ZIP link on tax-roll.php")

        # 2) Download ZIP
        z = requests.get(trw_zip_url, timeout=180)
        z.raise_for_status()

        # 3) Unzip: pick biggest non-readme file
        zf = zipfile.ZipFile(io.BytesIO(z.content))
        names = [n for n in zf.namelist() if "readme" not in n.lower()]
        if not names:
            raise RuntimeError("ZIP had no usable files")
        names.sort(key=lambda n: zf.getinfo(n).file_size, reverse=True)
        data_name = names[0]

        wanted_zips = [normalize_zip(x.strip()) for x in mailing_zips.split(",") if normalize_zip(x.strip())]
        today = datetime.now(tz=timezone.utc)

        # Aggregate by account
        by_account: Dict[str, Dict[str, Any]] = {}

        with zf.open(data_name, "r") as fh:
            for raw in fh:
                try:
                    line = raw.decode("utf-8", errors="ignore")
                except Exception:
                    continue
                if not line.strip():
                    continue

                account = parse_fixed(line, 1, 34)
                if not account:
                    continue

                due_date = parse_yyyymmdd(parse_fixed(line, 101, 8))
                levy_bal = numeric_amount(parse_fixed(line, 111, 11))
                tot_due = numeric_amount(parse_fixed(line, 490, 11))

                is_delinquent = (due_date is not None) and (due_date < today) and (levy_bal > 0)
                if not is_delinquent:
                    continue

                city = parse_fixed(line, 386, 40)
                state = (parse_fixed(line, 426, 2) or "TX").upper()
                zip5 = normalize_zip(parse_fixed(line, 428, 12))

                if wanted_zips and zip5 and zip5 not in wanted_zips:
                    continue

                owner1 = parse_fixed(line, 226, 40)
                addr2 = parse_fixed(line, 266, 40)
                addr3 = parse_fixed(line, 306, 40)
                addr4 = parse_fixed(line, 346, 40)
                tax_unit_acct = parse_fixed(line, 43, 34)

                rec = by_account.get(account)
                if not rec:
                    by_account[account] = {
                        "account": account,
                        "tax_unit_acct": tax_unit_acct,
                        "due_date": due_date,
                        "levy_balance": levy_bal,
                        "tot_amt_due": tot_due,
                        "owner1": owner1,
                        "addr2": addr2,
                        "addr3": addr3,
                        "addr4": addr4,
                        "city": city,
                        "state": state,
                        "zip": zip5,
                        "lines": 1,
                    }
                else:
                    rec["lines"] += 1
                    if due_date and (rec["due_date"] is None or due_date < rec["due_date"]):
                        rec["due_date"] = due_date
                    rec["levy_balance"] = max(rec["levy_balance"], levy_bal)
                    rec["tot_amt_due"] = max(rec["tot_amt_due"], tot_due)
                    if not rec["city"] and city:
                        rec["city"] = city
                    if not rec["zip"] and zip5:
                        rec["zip"] = zip5

                if len(by_account) >= max_accounts:
                    break

        now_iso = datetime.now(tz=timezone.utc).isoformat()

        raw_rows: List[Dict[str, Any]] = []
        lead_rows: List[Dict[str, Any]] = []
        clean_rows: List[Dict[str, Any]] = []

        for acc, r in by_account.items():
            street = make_street(r["owner1"], r["addr2"], r["addr3"], r["addr4"])
            city = (r["city"] or "").strip()
            state = (r["state"] or "TX").strip().upper()
            zip5 = normalize_zip(r["zip"] or "")

            # Minimum viable for outbound + clean export
            if not street or not city or not zip5:
                continue

            payload = {
                "account": r["account"],
                "tax_unit_acct": r["tax_unit_acct"],
                "due_date": r["due_date"].date().isoformat() if r["due_date"] else None,
                "levy_balance": r["levy_balance"],
                "tot_amt_due": r["tot_amt_due"],
                "owner_line1": r["owner1"],
                "address2": r["addr2"],
                "address3": r["addr3"],
                "address4": r["addr4"],
                "city": city,
                "state": state,
                "zip": zip5,
                "jurisdiction_lines": r["lines"],
                "trw_zip_url": trw_zip_url,
                "trw_file_name": data_name,
            }

            raw_rows.append(
                {
                    "source_type": "tax_delinquent_trw",
                    "source_key": acc,
                    "source_event_date": r["due_date"].isoformat() if r["due_date"] else None,
                    "payload": payload,
                }
            )

            lead_rows.append(
                {
                    "source_type": "tax_delinquent_trw",
                    "last_seen_at": now_iso,
                    "address": street,
                    "city": city.upper(),
                    "state": state,
                    "zip": zip5,
                    "tags": ["tax_delinquent"],
                    "source_event_date": r["due_date"].isoformat() if r["due_date"] else None,
                    "raw_source_key": acc,
                    "source_url": trw_zip_url,
                    "dedupe_key": dedupe_key(street, city, state, zip5),
                }
            )

            clean_rows.append(
                {"Street Address": street, "City": city, "State": state, "Zip": zip5}
            )

        # Dedup leads in-memory by dedupe_key
        dedup_map: Dict[str, Dict[str, Any]] = {}
        for lr in lead_rows:
            dedup_map[lr["dedupe_key"]] = lr
        lead_rows = list(dedup_map.values())

        # Upsert in batches
        for chunk in chunks(raw_rows, batch_size):
            supabase_upsert("gov_pull_raw", chunk, "source_type,source_key")
        for chunk in chunks(lead_rows, batch_size):
            supabase_upsert("gov_leads", chunk, "dedupe_key")

        exports: Dict[str, str] = {}
        bucket = os.getenv("GCS_BUCKET", "").strip()

        # GCS exports (optional)
        if export_mode != "none":
            if not bucket:
                exports = {"skipped": "GCS_BUCKET not set"}
            else:
                run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                if export_mode == "clean":
                    # Upload only clean + stable pointer
                    if storage is None:
                        raise RuntimeError("google-cloud-storage not available. Add it to requirements.txt.")
                    clean_cols = ["Street Address", "City", "State", "Zip"]
                    clean_csv = write_csv_bytes(clean_cols, clean_rows)
                    clean_obj = f"dallas-taxroll-clean/weekly_clean_{run_ts}.csv"
                    clean_uri = gcs_upload_bytes(bucket, clean_obj, clean_csv, "text/csv")
                    clean_latest_uri = gcs_upload_bytes(bucket, "weekly_clean.csv", clean_csv, "text/csv")
                    exports = {"clean": clean_uri, "clean_latest": clean_latest_uri}
                elif export_mode == "full":
                    if not bucket:
                        exports = {"skipped": "GCS_BUCKET not set"}
                    else:
                        # Build only FULL artifact
                        full_cols, full_flat = flatten_payload_rows(raw_rows)
                        full_csv = write_csv_bytes(full_cols, full_flat)
                        full_gz = gzip.compress(full_csv)
                        dt = datetime.now(timezone.utc)
                        ym = dt.strftime("%Y-%m")
                        full_obj = f"dallas-taxroll-full/{ym}/dallas_tax_delinquent_full_{run_ts}.csv.gz"
                        full_uri = gcs_upload_bytes(bucket, full_obj, full_gz, "application/gzip")
                        exports = {"full": full_uri}
                else:
                    # both
                    exports = export_to_gcs(bucket, run_ts, raw_rows, clean_rows)

        return {
            "ok": True,
            "source": "tax_delinquent_trw",
            "trw_zip_url": trw_zip_url,
            "trw_file": data_name,
            "accounts_parsed": len(by_account),
            "raw_rows": len(raw_rows),
            "leads_upserted": len(lead_rows),
            "exports": exports,
            "sample_lead": lead_rows[0] if debug and lead_rows else None,
            "note": "These are MAILING addresses from TRW (not guaranteed property/situs address).",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
