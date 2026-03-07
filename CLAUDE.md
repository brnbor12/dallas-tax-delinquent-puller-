# CLAUDE.md

## Project Overview

Dallas County tax delinquent property data puller. Scrapes the TRW (Tax Roll Workfile) from the Dallas County website, parses fixed-width records, identifies delinquent accounts, and upserts results to Supabase. Optionally exports to Google Cloud Storage.

Deployed as a **Google Cloud Run Job** via Docker.

## Architecture

```
runner.py          → Batch entry point (reads env vars, calls main.run())
main.py            → FastAPI app with single GET /run endpoint
Dockerfile         → python:3.11-slim container
requirements.txt   → Python dependencies
```

This is a monolithic two-file application. All business logic lives in `main.py`.

## Data Flow

1. Fetch Dallas County tax roll page, extract TRW ZIP URL
2. Download and unzip the TRW file
3. Parse fixed-width text records (1-based column positions)
4. Filter for delinquent accounts (due_date < today AND levy_balance > 0)
5. Normalize addresses, deduplicate by address components
6. Batch upsert to Supabase tables: `gov_pull_raw` and `gov_leads`
7. Optionally export to GCS (full gzipped CSV and/or clean 4-column CSV)

## Environment Variables

### Required
- `SUPABASE_URL` — Supabase project REST API base URL
- `SUPABASE_SERVICE_ROLE_KEY` — Supabase service role key

### Optional
- `GCS_BUCKET` — GCS bucket name (required if export_mode != "none")
- `EXPORT_MODE` — `none|clean|full|both` (default: `both`)
- `MAX_ACCOUNTS` — Max accounts to parse (default: `50000`)
- `MAILING_ZIPS` — Comma-separated ZIP codes to filter by mailing address
- `BATCH_SIZE` — Supabase upsert batch size (default: `500`, range: 50–2000)

## Key Technical Details

### Fixed-Width Parsing
The TRW file uses fixed-width columns (1-based positions). Key fields:
- Account: cols 1–34
- Due date: cols 101–108 (YYYYMMDD)
- Levy balance: cols 111–121
- Owner/address: cols 226–385 (four 40-char lines)
- City: cols 386–425
- State: cols 426–427
- ZIP: cols 428–439
- Total due: cols 490–500

### Address Logic
The `make_street()` function handles TRW address lines (owner1, addr2, addr3, addr4) by finding the first line with a digit that isn't a unit-only pattern (STE, SUITE, APT, UNIT, #). This avoids selecting "STE 200" as the street address.

### Deduplication
Leads are deduplicated in-memory using `dedupe_key()` which normalizes and joins address, city, state, and ZIP.

### Supabase Tables
- `gov_pull_raw` — Full parsed records with payload JSON. Conflict key: `source_type,source_key`
- `gov_leads` — Normalized lead data. Conflict key: `dedupe_key`

### GCS Export Paths
- Full: `dallas-taxroll-full/{YYYY-MM}/dallas_tax_delinquent_full_{timestamp}.csv.gz`
- Clean: `dallas-taxroll-clean/weekly_clean_{timestamp}.csv`
- Latest pointer: `weekly_clean.csv` (overwritten each run)

## Development

### Run locally
```bash
pip install -r requirements.txt

# As FastAPI server:
uvicorn main:app --reload
# Then visit: http://localhost:8000/run?debug=true&max_accounts=100

# As batch job:
python runner.py
```

### Build and run with Docker
```bash
docker build -t dallas-tax-puller .
docker run --env-file .env dallas-tax-puller
```

### Code Style
- Python 3.11
- Type hints throughout (Optional, List, Dict, Any, Tuple)
- Snake_case for functions and variables
- No linter or formatter configured — keep style consistent with existing code
- Minimal dependencies; `google-cloud-storage` gracefully degrades if not installed

### Testing
No test suite exists. When adding tests, use `pytest`. Key functions worth testing:
- `parse_fixed()`, `parse_yyyymmdd()`, `normalize_zip()`
- `is_unit_only_line()`, `make_street()`
- `dedupe_key()`

## Common Pitfalls

- `is_unit_only_line()` is defined twice in `main.py` (lines 4–9 and 127–130). The second definition (with compiled regex) is the one used at runtime.
- The `google-cloud-storage` import is wrapped in try/except — GCS features silently degrade if the package is missing.
- TRW file format can change without notice; column positions are hardcoded.
- All amounts are integer cents (no decimal handling).

---

# Real Estate Lead Scraper — Project Context

## Goal
Build a modular Python scraper that pulls motivated seller leads from public
county records and stores them in Supabase.

## Target Counties (Phase 1)
- Dallas County, TX
- Polk County, FL
- Hillsborough County, FL

## Target List Types (Priority Order)
1. Tax Delinquent
2. Lis Pendens / Pre-Foreclosure
3. Code Violations
4. Probate Filings
5. Evictions

## Output Schema (Supabase table: leads)
- id (uuid)
- first_name
- last_name
- mailing_address
- mailing_city
- mailing_state
- mailing_zip
- property_address
- property_city
- property_state
- property_zip
- county
- list_type (tax_delinquent, lis_pendens, etc.)
- source_url
- distress_signals (array — for stacking)
- score (int — number of stacked signals)
- date_pulled (timestamp)
- raw_data (jsonb — store original record)

## Tech Stack
- Python 3.11+
- requests + BeautifulSoup for HTML scraping
- playwright for JS-heavy sites
- supabase-py for database writes
- python-dotenv for credentials
- loguru for logging

## Project Structure
real-estate-scraper/
├── CLAUDE.md
├── .env
├── main.py
├── config.py
├── scrapers/
│   ├── __init__.py
│   ├── base_scraper.py
│   ├── dallas/
│   │   ├── tax_delinquent.py
│   │   ├── lis_pendens.py
│   │   └── code_violations.py
│   ├── polk/
│   │   ├── tax_delinquent.py
│   │   └── lis_pendens.py
│   └── hillsborough/
│       ├── tax_delinquent.py
│       └── lis_pendens.py
├── processors/
│   ├── deduplicator.py
│   └── scorer.py
├── db/
│   └── supabase_client.py
└── logs/

## Rules for Claude Code
- Build one scraper module at a time
- Always test with a small sample before bulk pulling
- Never hardcode credentials — use .env
- Every scraper inherits from base_scraper.py
- Log all errors with loguru, never silently fail
- After each scrape, run deduplicator before writing to Supabase
- Score each lead based on how many distress signals stack
