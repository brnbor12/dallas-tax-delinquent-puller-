# Motivated Seller Finder

A national web app that aggregates motivated seller signals from public records, court filings, and listing data. Search by location, filter by signal type, view results on a map, and export leads to CSV.

## Stack

| Layer | Tech |
|---|---|
| Frontend | React 18 + TypeScript + Vite + Tailwind |
| Map | MapLibre GL JS |
| Backend API | Python + FastAPI |
| Task queue | Celery + Redis |
| Database | PostgreSQL 16 + PostGIS |
| Map tiles | Martin tile server |

## Quick Start (Docker)

```bash
# 1. Copy env file and fill in your keys
cp .env.example .env

# 2. Start all services
docker compose up --build

# 3. Run database migrations (first time only)
docker compose exec api alembic -c migrations/alembic.ini upgrade head

# 4. Frontend dev server (separate terminal)
cd frontend && npm install && npm run dev
```

Services:
- Frontend: http://localhost:5173
- API: http://localhost:8000
- API docs: http://localhost:8000/docs
- Celery monitor: http://localhost:5555
- Map tiles: http://localhost:3001

## Running a Scraper Manually

```bash
# Trigger LA County tax delinquent scraper via API
curl -X POST http://localhost:8000/api/v1/admin/scrape-jobs/1/trigger

# Or run directly in Python
docker compose exec worker python -c "
from tasks.scrape_tasks import run_county_api_scraper
run_county_api_scraper.apply(args=['ca_la_tax_delinquent'])
"
```

## Running Tests

```bash
docker compose exec api python -m pytest tests/unit/ -v
```

## Project Structure

```
motivated-seller-app/
  backend/
    app/
      api/v1/endpoints/   # FastAPI route handlers
      core/               # Config, database, logging
      models/             # SQLAlchemy ORM models
      schemas/            # Pydantic request/response
      services/           # Business logic
    scrapers/
      base.py             # Abstract scraper interface
      ingestor.py         # Geocode → deduplicate → upsert
      geocoder.py         # Address → lat/lng (Nominatim or Google)
      registry.py         # Scraper lookup table
      counties/           # Per-county scraper implementations
        california/
          la_county_tax.py   # LA County tax delinquent (Socrata)
          ca_nod_statewide.py # CA NOD via ATTOM Data API
        platforms/
          socrata_api.py     # Generic Socrata scraper (config-driven)
    scoring/
      engine.py           # Motivated seller scoring algorithm
    tasks/
      celery_app.py       # Celery configuration + Beat schedule
      scrape_tasks.py     # Background scraping tasks
      score_tasks.py      # Score recalculation tasks
      export_tasks.py     # CSV export tasks
    tests/
      unit/test_scoring.py
    migrations/           # Alembic migrations
  frontend/
    src/
      components/
        map/MapView.tsx        # MapLibre GL map
        filters/FilterPanel.tsx # Filter sidebar
        property/
          PropertyList.tsx     # Scrollable list
          PropertyCard.tsx     # List item
          PropertyDetail.tsx   # Slide-in detail panel
          ScoreBadge.tsx       # Score chip
          IndicatorPill.tsx    # Signal tag
        layout/Header.tsx
      hooks/useProperties.ts   # TanStack Query hooks
      stores/
        filterStore.ts         # Zustand filter state
        mapStore.ts            # Map viewport state
      types/property.ts
      types/filter.ts
  docker-compose.yml
  docker/Dockerfile.api

## Adding a New County Scraper

1. Create `backend/scrapers/counties/<state>/<name>.py`
2. Implement `BaseCountyScraper.fetch_records()` yielding `RawIndicatorRecord`
3. Register it in `backend/scrapers/registry.py`
4. Add a row to `scrape_jobs` table with `config.scraper_key`

No other code changes needed.

## Indicator Types

| Signal | Description |
|---|---|
| `pre_foreclosure` | Notice of Default filed |
| `foreclosure` | Auction scheduled |
| `tax_delinquent` | Unpaid property taxes |
| `probate` | Estate/probate filing |
| `lien` | Mechanic's, HOA, judgment lien |
| `eviction` | Unlawful detainer filing |
| `code_violation` | Municipal code violation |
| `vacant` | Property unoccupied |
| `absentee_owner` | Owner mailing ≠ property address |
| `price_reduction` | MLS price drop |
| `expired_listing` | Listing expired without sale |
| `days_on_market` | Extended listing duration |

## Scoring

Scores are 0–100 using a compound probability model (diminishing returns).
- **Hot** ≥ 60 (red)
- **Warm** 30–59 (orange)
- **Cold** < 30 (blue)

Scores decay over time as indicators age. Recalculated nightly.

## Legal Notes

- **Public county records**: Generally OK — government data is public under FOIA/state equivalents. Always respect robots.txt and rate limits.
- **Zillow/MLS**: Do NOT scrape directly. Use ATTOM Data Solutions API or BatchData for licensed listing data.
- **FCRA**: Platform is for investment research only. State this in your ToS.
- **Privacy**: Include a CCPA-compliant privacy policy and data removal form.
```
