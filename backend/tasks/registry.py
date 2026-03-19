"""
Scraper registry — maps scraper keys (stored in scrape_jobs.config) to classes.

To add a new scraper:
  1. Implement BaseCountyScraper in scrapers/counties/<state>/<name>.py
  2. Import it here and add an entry to SCRAPER_REGISTRY.
  3. Insert a row into scrape_jobs with config.scraper_key matching your key.
  No other code changes needed.
"""

from scrapers.base import BaseCountyScraper
from scrapers.counties.california.la_county_tax import LACountyTaxDelinquentScraper
from scrapers.counties.california.ca_nod_statewide import CANODStatewideHelper
from scrapers.counties.florida.hillsborough_foreclosure import HillsboroughForeclosureScraper
from scrapers.counties.florida.hillsborough_eviction import HillsboroughEvictionScraper
from scrapers.counties.florida.hillsborough_probate import HillsboroughProbateScraper
from scrapers.counties.florida.hillsborough_lis_pendens import HillsboroughLisPendensScraper
from scrapers.counties.florida.polk_tax_deed import PolkTaxDeedScraper
from scrapers.counties.florida.polk_absentee_owner import PolkAbsenteeOwnerScraper
from scrapers.counties.platforms.socrata_api import SocrataAPIScraper
from scrapers.counties.texas.dallas_foreclosure import DallasCountyForeclosureScraper
from scrapers.counties.texas.dallas_tax import DallasCountyTaxRollScraper
from scrapers.counties.texas.dallas_code_enforcement import DallasCodeEnforcementScraper
from scrapers.counties.texas.dallas_probate import DallasProbateScraper
from scrapers.counties.texas.dallas_eviction import DallasEvictionScraper
from scrapers.counties.texas.dallas_lis_pendens import DallasLisPendensScraper
from scrapers.counties.texas.dallas_divorce import DallasDivorceScraper
from scrapers.counties.texas.dallas_supabase_leads import (
    DallasGovLeadsScraper,
    DallasForeclosureNoticeSupabaseScraper,
)
from scrapers.counties.texas.harris_absentee_owner import HarrisAbsenteeOwnerScraper

SCRAPER_REGISTRY: dict[str, type[BaseCountyScraper]] = {
    # Texas — Dallas County
    "tx_dallas_foreclosure":        DallasCountyForeclosureScraper,
    "tx_dallas_tax_delinquent":     DallasCountyTaxRollScraper,
    "tx_dallas_code_enforcement":   DallasCodeEnforcementScraper,
    "tx_dallas_probate":            DallasProbateScraper,
    "tx_dallas_eviction_fed":       DallasEvictionScraper,
    "tx_dallas_lis_pendens":        DallasLisPendensScraper,
    "tx_dallas_divorce":            DallasDivorceScraper,
    # Texas — Dallas County (Supabase/GCP pipeline)
    "tx_dallas_supabase_leads":         DallasGovLeadsScraper,
    "tx_dallas_supabase_foreclosure":   DallasForeclosureNoticeSupabaseScraper,
    # Texas — Harris County (Houston metro)
    "tx_harris_absentee_owner":         HarrisAbsenteeOwnerScraper,
    # Florida — Hillsborough County
    "fl_hillsborough_foreclosure":  HillsboroughForeclosureScraper,
    "fl_hillsborough_eviction":     HillsboroughEvictionScraper,
    "fl_hillsborough_probate":      HillsboroughProbateScraper,
    "fl_hillsborough_lis_pendens":  HillsboroughLisPendensScraper,
    # Florida — Polk County
    "fl_polk_tax_deed":            PolkTaxDeedScraper,
    "fl_polk_absentee_owner":      PolkAbsenteeOwnerScraper,
    # California
    "ca_la_tax_delinquent":    LACountyTaxDelinquentScraper,
    "ca_nod_statewide":        CANODStatewideHelper,
    # Generic platform templates
    "socrata_generic":         SocrataAPIScraper,
}


def get_scraper(scraper_key: str, config: dict | None = None) -> BaseCountyScraper:
    cls = SCRAPER_REGISTRY.get(scraper_key)
    if cls is None:
        raise KeyError(f"No scraper registered for key: {scraper_key!r}")
    return cls(config=config)
