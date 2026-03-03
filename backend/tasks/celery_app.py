"""
Celery application configuration.

Queues:
  county_api   — Fast API-based scrapers (Socrata, direct REST) — `worker` service
  web_scrape   — Browser-based scrapers (Playwright) — `playwright-worker` service
  mls          — MLS/listing data scrapers (rate-limited) — `worker` service
  default      — Score recalculation, exports, notifications — `worker` service

Worker split (see docker-compose.yml):
  worker            → -Q county_api,mls,default
  playwright-worker → -Q web_scrape  (--concurrency=2, Dockerfile.playwright)
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
priority_exchange = Exchange("priority", type="direct")

app.conf.task_queues = (
    Queue("county_api",  default_exchange, routing_key="county_api"),
    Queue("web_scrape",  default_exchange, routing_key="web_scrape"),
    Queue("mls",         default_exchange, routing_key="mls"),
    Queue("default",     default_exchange, routing_key="default"),
)

app.conf.task_default_queue = "default"
app.conf.task_default_exchange = "default"
app.conf.task_default_routing_key = "default"

# Route tasks to appropriate queues
app.conf.task_routes = {
    "tasks.scrape_tasks.run_county_api_scraper":  {"queue": "county_api"},
    "tasks.scrape_tasks.run_web_scraper":          {"queue": "web_scrape"},
    "tasks.scrape_tasks.run_mls_scraper":          {"queue": "mls"},
    "tasks.score_tasks.recalculate_property_score": {"queue": "default"},
    "tasks.score_tasks.nightly_score_decay":        {"queue": "default"},
    "tasks.export_tasks.generate_export":           {"queue": "default"},
    "tasks.notification_tasks.send_alert":          {"queue": "default"},
}

# Serialization
app.conf.task_serializer = "json"
app.conf.result_serializer = "json"
app.conf.accept_content = ["json"]
app.conf.timezone = "UTC"
app.conf.enable_utc = True

# Retry policy defaults
app.conf.task_acks_late = True
app.conf.task_reject_on_worker_lost = True
app.conf.task_max_retries = 3

# Result expiry (keep results for 24 hours)
app.conf.result_expires = 86400

# Celery Beat schedule
app.conf.beat_schedule = {
    # Dallas County — foreclosure notices (monthly PDF, run weekly to catch new filings)
    "tx-dallas-foreclosure-weekly": {
        "task": "tasks.scrape_tasks.run_web_scraper",
        "schedule": 604800,  # every 7 days
        "kwargs": {"scraper_key": "tx_dallas_foreclosure"},
        "options": {"queue": "web_scrape"},
    },
    # Dallas County — Lis Pendens (recorded daily; run daily, 90-day lookback)
    "tx-dallas-lis-pendens-daily": {
        "task": "tasks.scrape_tasks.run_web_scraper",
        "schedule": 86400,  # every 24 hours
        "kwargs": {"scraper_key": "tx_dallas_lis_pendens"},
        "options": {"queue": "web_scrape"},
    },
    # Dallas County — probate (active estates, 2-year lookback, run weekly)
    "tx-dallas-probate-weekly": {
        "task": "tasks.scrape_tasks.run_web_scraper",
        "schedule": 604800,
        "kwargs": {"scraper_key": "tx_dallas_probate"},
        "options": {"queue": "web_scrape"},
    },
    # Dallas County — eviction FED (active filings, 90-day lookback, run daily)
    "tx-dallas-eviction-daily": {
        "task": "tasks.scrape_tasks.run_web_scraper",
        "schedule": 86400,
        "kwargs": {"scraper_key": "tx_dallas_eviction_fed"},
        "options": {"queue": "web_scrape"},
    },
    # Dallas County — divorce (active FAM cases, 180-day lookback, run weekly)
    "tx-dallas-divorce-weekly": {
        "task": "tasks.scrape_tasks.run_web_scraper",
        "schedule": 604800,
        "kwargs": {"scraper_key": "tx_dallas_divorce"},
        "options": {"queue": "web_scrape"},
    },
    # Dallas County — tax delinquent roll (updated weekly on Fridays)
    "tx-dallas-tax-weekly": {
        "task": "tasks.scrape_tasks.run_county_api_scraper",
        "schedule": 604800,  # every 7 days
        "kwargs": {"scraper_key": "tx_dallas_tax_delinquent"},
        "options": {"queue": "county_api"},
    },
    # California — kept for reference, can disable if focused on TX
    "la-county-tax-weekly": {
        "task": "tasks.scrape_tasks.run_county_api_scraper",
        "schedule": 604800,  # every 7 days
        "kwargs": {"scraper_key": "ca_la_tax_delinquent"},
        "options": {"queue": "county_api"},
    },
    "nightly-score-decay": {
        "task": "tasks.score_tasks.nightly_score_decay",
        "schedule": 86400,  # every 24 hours
        "options": {"queue": "default"},
    },
}
