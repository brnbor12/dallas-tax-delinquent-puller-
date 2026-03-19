"""
Celery application configuration.

Queues:
  county_api   -- Fast API-based scrapers (Socrata, direct REST) -- `worker` service
  web_scrape   -- Browser-based scrapers (Playwright) -- `playwright-worker` service
  mls          -- MLS/listing data scrapers (rate-limited) -- `worker` service
  default      -- Score recalculation, exports, notifications -- `worker` service

Worker split (see docker-compose.yml):
  worker            -> -Q county_api,mls,default
  playwright-worker -> -Q web_scrape  (--concurrency=2, Dockerfile.playwright)
"""

import structlog
from celery import Celery
from celery.signals import worker_process_init
from kombu import Exchange, Queue

from app.core.config import settings


@worker_process_init.connect
def configure_worker_logging(**kwargs):
    """Configure structlog for each forked Celery worker process."""
    from app.core.logging import setup_logging
    setup_logging()

app = Celery(
    "motivated_seller",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "tasks.scrape_tasks",
        "tasks.score_tasks",
        "tasks.export_tasks",
        "tasks.notification_tasks",
    ],
)

# Queue definitions
default_exchange = Exchange("default", type="direct")

app.conf.task_queues = (
    Queue("county_api",  default_exchange, routing_key="county_api"),
    Queue("web_scrape",  default_exchange, routing_key="web_scrape"),
    Queue("mls",         default_exchange, routing_key="mls"),
    Queue("default",     default_exchange, routing_key="default"),
)

app.conf.task_default_queue = "default"
app.conf.task_default_exchange = "default"
app.conf.task_default_routing_key = "default"

app.conf.task_routes = {
    "tasks.scrape_tasks.run_county_api_scraper":   {"queue": "county_api"},
    "tasks.scrape_tasks.run_web_scraper":           {"queue": "web_scrape"},
    "tasks.scrape_tasks.run_mls_scraper":           {"queue": "mls"},
    "tasks.score_tasks.recalculate_property_score": {"queue": "default"},
    "tasks.score_tasks.nightly_score_decay":        {"queue": "default"},
    "tasks.export_tasks.generate_export":           {"queue": "default"},
    "tasks.notification_tasks.send_alert":          {"queue": "default"},
}

app.conf.task_serializer = "json"
app.conf.result_serializer = "json"
app.conf.accept_content = ["json"]
app.conf.timezone = "UTC"
app.conf.enable_utc = True
app.conf.task_acks_late = True
app.conf.task_reject_on_worker_lost = True
app.conf.task_max_retries = 3
app.conf.result_expires = 86400

# ---------------------------------------------------------------------------
# Celery Beat Schedule
# ---------------------------------------------------------------------------
# Scrapers marked IP_BLOCKED fail from DigitalOcean IPs (Tyler Odyssey portals).
# Use the manual import endpoint (/api/v1/import) to upload court data CSVs.
# ---------------------------------------------------------------------------

app.conf.beat_schedule = {

    # -- Dallas, TX ----------------------------------------------------------

    # DCAD bulk data — ownership changes, absentee owners, value signals
    "tx-dcad-bulk-monthly": {
        "task": "tasks.scrape_tasks.run_county_api_scraper",
        "schedule": 2592000,  # monthly
        "kwargs": {"scraper_key": "tx_dcad_bulk"},
        "options": {"queue": "county_api"},
    },
    # Foreclosure notices -- Playwright, monthly PDF, run weekly
    "tx-dallas-foreclosure-weekly": {
        "task": "tasks.scrape_tasks.run_web_scraper",
        "schedule": 604800,
        "kwargs": {"scraper_key": "tx_dallas_foreclosure"},
        "options": {"queue": "web_scrape"},
    },
    # Lis Pendens -- Playwright, recorded daily, 90-day lookback
    "tx-dallas-lis-pendens-daily": {
        "task": "tasks.scrape_tasks.run_web_scraper",
        "schedule": 86400,
        "kwargs": {"scraper_key": "tx_dallas_lis_pendens"},
        "options": {"queue": "web_scrape"},
    },
    # Tax delinquent roll -- open data API, updated weekly on Fridays
    "tx-dallas-tax-weekly": {
        "task": "tasks.scrape_tasks.run_county_api_scraper",
        "schedule": 604800,
        "kwargs": {"scraper_key": "tx_dallas_tax_delinquent"},
        "options": {"queue": "county_api"},
    },
    # Code enforcement violations -- open data API (311 dataset), updated daily
    "tx-dallas-code-enforcement-daily": {
        "task": "tasks.scrape_tasks.run_county_api_scraper",
        "schedule": 86400,
        "kwargs": {"scraper_key": "tx_dallas_code_enforcement"},
        "options": {"queue": "county_api"},
    },
    # GCP/Supabase leads sync -- pulls from gov_leads table, run weekly
    "tx-dallas-supabase-leads-weekly": {
        "task": "tasks.scrape_tasks.run_county_api_scraper",
        "schedule": 604800,
        "kwargs": {"scraper_key": "tx_dallas_supabase_leads"},
        "options": {"queue": "county_api"},
    },
    # GCP/Supabase foreclosure notices sync -- run weekly
    "tx-dallas-supabase-foreclosure-weekly": {
        "task": "tasks.scrape_tasks.run_county_api_scraper",
        "schedule": 604800,
        "kwargs": {"scraper_key": "tx_dallas_supabase_foreclosure"},
        "options": {"queue": "county_api"},
    },

    # -- Harris County, TX (Houston metro) -----------------------------------
    # Source: HCAD bulk CAMA data -- annual release (~spring). ~200MB download.
    # Run monthly so we pick up the new file when HCAD refreshes it each year.
    "tx-harris-absentee-owner-monthly": {
        "task": "tasks.scrape_tasks.run_county_api_scraper",
        "schedule": 2592000,
        "kwargs": {"scraper_key": "tx_harris_absentee_owner"},
        "options": {"queue": "county_api"},
    },

    # -- Hillsborough County, FL ---------------------------------------------
    # Source: publicrec.hillsclerk.com bulk CSVs -- no IP blocking, free daily

    "fl-hillsborough-eviction-daily": {
        "task": "tasks.scrape_tasks.run_county_api_scraper",
        "schedule": 86400,
        "kwargs": {"scraper_key": "fl_hillsborough_eviction"},
        "options": {"queue": "county_api"},
    },
    "fl-hillsborough-probate-daily": {
        "task": "tasks.scrape_tasks.run_county_api_scraper",
        "schedule": 86400,
        "kwargs": {"scraper_key": "fl_hillsborough_probate"},
        "options": {"queue": "county_api"},
    },
    "fl-hillsborough-lis-pendens-daily": {
        "task": "tasks.scrape_tasks.run_county_api_scraper",
        "schedule": 86400,
        "kwargs": {"scraper_key": "fl_hillsborough_lis_pendens"},
        "options": {"queue": "county_api"},
    },
    "fl-hillsborough-foreclosure-daily": {
        "task": "tasks.scrape_tasks.run_county_api_scraper",
        "schedule": 86400,
        "kwargs": {"scraper_key": "fl_hillsborough_foreclosure"},
        "options": {"queue": "county_api"},
    },

    # -- Polk County, FL -----------------------------------------------------
    # Source: polkflpa.gov FTP bulk data -- nightly refresh, no IP blocking

    "fl-polk-tax-delinquent-pa-weekly": {
        "task": "tasks.scrape_tasks.run_county_api_scraper",
        "schedule": 604800,
        "kwargs": {"scraper_key": "fl_polk_tax_delinquent_pa"},
        "options": {"queue": "county_api"},
    },
    "fl-polk-absentee-owner-weekly": {
        "task": "tasks.scrape_tasks.run_county_api_scraper",
        "schedule": 604800,
        "kwargs": {"scraper_key": "fl_polk_absentee_owner"},
        "options": {"queue": "county_api"},
    },
    "fl-polk-probate-weekly": {
        "task": "tasks.scrape_tasks.run_county_api_scraper",
        "schedule": 604800,
        "kwargs": {"scraper_key": "fl_polk_probate"},
        "options": {"queue": "county_api"},
    },
    "fl-polk-eviction-weekly": {
        "task": "tasks.scrape_tasks.run_county_api_scraper",
        "schedule": 604800,
        "kwargs": {"scraper_key": "fl_polk_eviction"},
        "options": {"queue": "county_api"},
    },
    "fl-polk-foreclosure-weekly": {
        "task": "tasks.scrape_tasks.run_county_api_scraper",
        "schedule": 604800,
        "kwargs": {"scraper_key": "fl_polk_foreclosure"},
        "options": {"queue": "county_api"},
    },
    "fl-polk-lis-pendens-weekly": {
        "task": "tasks.scrape_tasks.run_county_api_scraper",
        "schedule": 604800,
        "kwargs": {"scraper_key": "fl_polk_lis_pendens"},
        "options": {"queue": "county_api"},
    },

    # -- Rentcast Listings (all counties) ------------------------------------
    # Source: Rentcast API — active/pending sale listings with prices
    # Free tier = 50 calls/month; each returns up to 500 listings.
    # Run weekly — covers active inventory changes.

    "rentcast-dallas-weekly": {
        "task": "tasks.scrape_tasks.run_county_api_scraper",
        "schedule": 604800,
        "kwargs": {"scraper_key": "rentcast_dallas"},
        "options": {"queue": "county_api"},
    },
    "rentcast-hillsborough-weekly": {
        "task": "tasks.scrape_tasks.run_county_api_scraper",
        "schedule": 604800,
        "kwargs": {"scraper_key": "rentcast_hillsborough"},
        "options": {"queue": "county_api"},
    },
    "rentcast-polk-weekly": {
        "task": "tasks.scrape_tasks.run_county_api_scraper",
        "schedule": 604800,
        "kwargs": {"scraper_key": "rentcast_polk"},
        "options": {"queue": "county_api"},
    },

    # -- Maintenance ---------------------------------------------------------

    "nightly-score-decay": {
        "task": "tasks.score_tasks.nightly_score_decay",
        "schedule": 86400,
        "options": {"queue": "default"},
    },

    # -- DISABLED (IP_BLOCKED from DigitalOcean VPS) -------------------------
    # These Tyler Odyssey portals block DigitalOcean IPs at TCP level.
    # Use the manual import endpoint to upload court data CSVs instead:
    #   POST /api/v1/import  (county_fips, indicator_type, file upload)
    #
    # "tx-dallas-probate-weekly":  tx_dallas_probate   -- courtsportal.dallascounty.org
    # "tx-dallas-eviction-daily":  tx_dallas_eviction_fed -- courtsportal.dallascounty.org
    # "tx-dallas-divorce-weekly":  tx_dallas_divorce   -- courtsportal.dallascounty.org
    # "fl-polk-tax-deed-weekly":   fl_polk_tax_deed    -- apps.polkcountyclerk.net
}
