import io
import os
import re
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, Query

from core import (
    chunks,
    dedupe_key,
    export_clean_to_gcs,
    export_full_to_gcs,
    flatten_payload_rows,
    gcs_upload_bytes,
    is_unit_only_line,
    make_street,
    normalize_zip,
    numeric_amount,
    parse_fixed,
    parse_yyyymmdd,
    require_env,
    supabase_upsert,
    write_csv_bytes,
)

app = FastAPI()

DALLAS_TAX_ROLL_PAGE = "https://www.dallascounty.org/departments/tax/tax-roll.php"


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
    for u in matches:
        if "sample" not in u.lower():
            return u
    return matches[0]


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

        if export_mode != "none":
            if not bucket:
                exports = {"skipped": "GCS_BUCKET not set"}
            else:
                run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                if export_mode == "clean":
                    clean_cols = ["Street Address", "City", "State", "Zip"]
                    clean_csv = write_csv_bytes(clean_cols, clean_rows)
                    clean_uri = gcs_upload_bytes(bucket, f"dallas-taxroll-clean/weekly_clean_{run_ts}.csv", clean_csv, "text/csv")
                    clean_latest_uri = gcs_upload_bytes(bucket, "weekly_clean.csv", clean_csv, "text/csv")
                    exports = {"clean": clean_uri, "clean_latest": clean_latest_uri}
                elif export_mode == "full":
                    full_uri = export_full_to_gcs(bucket, "dallas-taxroll-full", raw_rows)
                    exports = {"full": full_uri}
                else:
                    # both
                    exports = export_clean_to_gcs(
                        bucket,
                        prefix="dallas-taxroll-clean",
                        stable_name="weekly_clean.csv",
                        clean_rows=clean_rows,
                    )
                    full_uri = export_full_to_gcs(bucket, "dallas-taxroll-full", raw_rows)
                    exports["full"] = full_uri

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
