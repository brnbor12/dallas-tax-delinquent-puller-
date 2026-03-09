"""
Shared utilities for all Dallas County scrapers.
"""
import csv
import gzip
import io
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from google.cloud import storage
except Exception:
    storage = None


# -----------------------------
# Basic utilities
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
    return line[start: start + length].strip()


def parse_yyyymmdd(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not re.fullmatch(r"\d{8}", s):
        return None
    y, m, d = int(s[0:4]), int(s[4:6]), int(s[6:8])
    return datetime(y, m, d, tzinfo=timezone.utc)


def numeric_amount(s: str) -> int:
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


def chunks(lst: List[Any], n: int):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


# -----------------------------
# Address logic
# -----------------------------

_UNIT_ONLY_RE = re.compile(r"^\s*(#\s*\w+|(ste|suite|apt|unit)\b\s*[\w-]+)\s*$", re.IGNORECASE)


def is_unit_only_line(line: str) -> bool:
    if not line:
        return False
    return bool(_UNIT_ONLY_RE.match(line.strip()))


def make_street(owner1: str, a2: str, a3: str, a4: str) -> str:
    parts = [p.strip() for p in [owner1, a2, a3, a4] if p and p.strip()]
    if not parts:
        return ""
    idx = -1
    for i, p in enumerate(parts):
        if re.search(r"\d", p) and not is_unit_only_line(p):
            idx = i
            break
    if idx >= 0:
        street_parts = parts[idx:]
    else:
        street_parts = parts[1:] if len(parts) > 1 else parts
    while street_parts and is_unit_only_line(street_parts[0]):
        street_parts = street_parts[1:]
    return re.sub(r"\s+", " ", " ".join(street_parts)).strip()


# -----------------------------
# Supabase
# -----------------------------

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
    r = requests.post(endpoint, headers=headers, params={"on_conflict": on_conflict}, json=rows, timeout=180)
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase upsert failed [{table}]: {r.status_code} {r.text[:800]}")


# -----------------------------
# GCS exports
# -----------------------------

def gcs_upload_bytes(bucket_name: str, object_name: str, data: bytes, content_type: str) -> str:
    if storage is None:
        raise RuntimeError("google-cloud-storage not available.")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_string(data, content_type=content_type)
    return f"gs://{bucket_name}/{object_name}"


def write_csv_bytes(fieldnames: List[str], rows: List[Dict[str, Any]]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for row in rows:
        w.writerow(row)
    return buf.getvalue().encode("utf-8")


def flatten_payload_rows(raw_rows: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    payload_keys: set = set()
    for r in raw_rows:
        payload_keys.update((r.get("payload") or {}).keys())
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


def export_clean_to_gcs(bucket_name: str, prefix: str, stable_name: str, clean_rows: List[Dict[str, Any]]) -> Dict[str, str]:
    """Upload a clean 4-column CSV to GCS with a timestamped file and a stable pointer."""
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    clean_cols = ["Street Address", "City", "State", "Zip"]
    clean_csv = write_csv_bytes(clean_cols, clean_rows)
    ts_uri = gcs_upload_bytes(bucket_name, f"{prefix}/{prefix}_{run_ts}.csv", clean_csv, "text/csv")
    stable_uri = gcs_upload_bytes(bucket_name, stable_name, clean_csv, "text/csv")
    return {"clean": ts_uri, "clean_latest": stable_uri}


def export_full_to_gcs(bucket_name: str, prefix: str, raw_rows: List[Dict[str, Any]]) -> str:
    """Upload a gzipped full CSV to GCS."""
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ym = datetime.now(timezone.utc).strftime("%Y-%m")
    full_cols, full_flat = flatten_payload_rows(raw_rows)
    full_csv = write_csv_bytes(full_cols, full_flat)
    full_gz = gzip.compress(full_csv)
    obj = f"{prefix}/{ym}/{prefix}_{run_ts}.csv.gz"
    return gcs_upload_bytes(bucket_name, obj, full_gz, "application/gzip")
