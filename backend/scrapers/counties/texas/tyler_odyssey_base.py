"""
Tyler Odyssey Court Portal — Playwright Base Scraper

Tyler Technologies' Odyssey is a JavaScript SPA used by hundreds of counties
for case management.  Direct REST API calls require session cookies set by the
SPA on first load, plus occasional CSRF tokens.  Playwright handles all of this
transparently by navigating the portal as a real browser, then using the
browser's own cookie jar to POST to the internal API.

Usage
-----
Subclass TylerOdysseyPlaywrightScraper and set `portal_base`.  Call
`_get_cases()` with the desired case category/type, and optionally
`_get_case_detail()` per case to enrich with party addresses.

    class MyProbateScraper(TylerOdysseyPlaywrightScraper):
        portal_base = "https://portal.co.dallas.tx.us"
        county_fips = "48113"
        ...
        async def fetch_records(self, **kwargs):
            cases = await self._get_cases(category="PR", node_id="PROBATE")
            for case in cases:
                detail = await self._get_case_detail(case_id)
                yield self._build_record(case, detail)

Architecture
------------
- Navigate to portal homepage → Odyssey SPA bootstraps and sets session cookie
- Use `context.request.post()` (Playwright's APIRequestContext) which sends
  requests using the BrowserContext's cookie jar — so the POST arrives with
  the same session cookie the browser just received.
- This is more reliable than intercepting XHR responses because it gives us
  direct control over the payload and doesn't depend on the SPA's routing.
"""

from __future__ import annotations

import asyncio
import json
import structlog
from datetime import date, timedelta
from typing import Any

from scrapers.playwright_base import PlaywrightBaseScraper

logger = structlog.get_logger(__name__)

# Odyssey API paths (same across all Tyler deployments)
_SEARCH_PATH = "/PortalService/api/Case/CaseSearch"
_DETAIL_PATH = "/PortalService/api/Case/CaseDetail"
_PORTAL_INIT_PATH = "/PortalService/api/portal/"

_API_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Content-Type": "application/json",
}


def _extract_cases(data: Any) -> list[dict]:
    """Pull a case list out of various Odyssey response wrapper shapes."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("cases", "Cases", "result", "Results", "data", "Data", "records"):
            val = data.get(key)
            if isinstance(val, list):
                return val
    return []


class TylerOdysseyPlaywrightScraper(PlaywrightBaseScraper):
    """
    Abstract base for Tyler Odyssey court portal scrapers.

    Subclasses must set:
        portal_base  — e.g. "https://portal.co.dallas.tx.us"
    """

    portal_base: str = ""

    async def _get_cases(
        self,
        category: str,
        node_id: str,
        status_type: str = "A",
        lookback_days: int = 90,
        case_type_id: str | None = None,
        extra_payload: dict | None = None,
    ) -> list[dict]:
        """
        Establish an Odyssey session via Playwright, then POST to the case
        search API using the browser's cookie jar.

        Returns a list of raw case dicts from the Odyssey JSON response.
        Returns [] on any error (logs the reason).
        """
        if not self.portal_base:
            raise ValueError(f"{self.__class__.__name__} must set portal_base")

        start_date = (date.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        end_date = date.today().strftime("%Y-%m-%d")

        payload: dict = {
            "nodeId": node_id,
            "nodeDesc": node_id,
            "category": category,
            "statusType": status_type,
            "filingDateStart": start_date,
            "filingDateEnd": end_date,
            "lastName": "",
            "firstName": "",
            "caseNumber": "",
            "dateOfBirth": "",
        }
        if case_type_id:
            payload["caseTypeId"] = case_type_id
        if extra_payload:
            payload.update(extra_payload)

        headers = {
            **_API_HEADERS,
            "Origin": self.portal_base,
            "Referer": f"{self.portal_base}/",
        }

        async with self.browser_context() as ctx:
            page = await ctx.new_page()

            # Step 1: Navigate to portal homepage → sets session cookie
            try:
                await page.goto(
                    self.portal_base + "/",
                    wait_until="domcontentloaded",
                    timeout=30_000,
                )
                await asyncio.sleep(2)
            except Exception as exc:
                logger.error(
                    "odyssey_portal_unreachable",
                    portal=self.portal_base,
                    error=str(exc)[:200],
                )
                return []

            # Step 2: Optional portal init call (some Odyssey versions need it)
            try:
                await ctx.request.get(
                    self.portal_base + _PORTAL_INIT_PATH,
                    headers=headers,
                )
            except Exception:
                pass  # non-fatal

            # Step 3: POST case search using the browser's session cookie
            try:
                response = await ctx.request.post(
                    self.portal_base + _SEARCH_PATH,
                    data=json.dumps(payload),
                    headers=headers,
                )
            except Exception as exc:
                logger.error(
                    "odyssey_search_request_failed",
                    portal=self.portal_base,
                    error=str(exc)[:200],
                )
                return []

            if response.status != 200:
                body = await response.text()
                logger.warning(
                    "odyssey_search_bad_status",
                    portal=self.portal_base,
                    status=response.status,
                    category=category,
                    body=body[:300],
                    hint=(
                        "Open the portal in a browser, run a search, and check "
                        "DevTools → Network for the actual CaseSearch request "
                        "payload and nodeId value."
                    ),
                )
                return []

            try:
                data = await response.json()
            except Exception as exc:
                logger.error("odyssey_search_parse_error", error=str(exc)[:200])
                return []

            cases = _extract_cases(data)
            if not cases:
                logger.warning(
                    "odyssey_search_no_results",
                    portal=self.portal_base,
                    category=category,
                    case_type_id=case_type_id,
                    response_keys=list(data.keys()) if isinstance(data, dict) else type(data).__name__,
                    hint="Try adjusting nodeId, category, or statusType in the payload",
                )

            return cases

    async def _get_case_detail(
        self,
        case_id: str,
        node_id: str = "",
    ) -> dict:
        """
        Fetch case detail for a single case, returning the raw detail dict.
        Returns {} on failure (caller should treat missing fields gracefully).

        Note: Opens a new browser context per call — callers should batch
        detail lookups or override this to reuse a context if performance
        matters.
        """
        if not case_id:
            return {}

        payload = {"caseId": case_id}
        if node_id:
            payload["nodeId"] = node_id

        headers = {
            **_API_HEADERS,
            "Origin": self.portal_base,
            "Referer": f"{self.portal_base}/",
        }

        async with self.browser_context() as ctx:
            page = await ctx.new_page()

            # Establish session
            try:
                await page.goto(
                    self.portal_base + "/",
                    wait_until="domcontentloaded",
                    timeout=30_000,
                )
                await asyncio.sleep(1)
            except Exception:
                return {}

            try:
                response = await ctx.request.post(
                    self.portal_base + _DETAIL_PATH,
                    data=json.dumps(payload),
                    headers=headers,
                )
                if response.status != 200:
                    return {}
                return await response.json()
            except Exception as exc:
                logger.debug(
                    "odyssey_detail_failed",
                    case_id=case_id,
                    error=str(exc)[:100],
                )
                return {}

    async def _get_cases_with_details(
        self,
        category: str,
        node_id: str,
        status_type: str = "A",
        lookback_days: int = 90,
        case_type_id: str | None = None,
        detail_node_id: str = "",
    ) -> list[tuple[dict, dict]]:
        """
        Convenience method: fetch cases + detail for each in a single browser
        context (efficient — one session establishment, one page, N API calls).

        Returns list of (case_dict, detail_dict) tuples.
        """
        if not self.portal_base:
            raise ValueError(f"{self.__class__.__name__} must set portal_base")

        start_date = (date.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        end_date = date.today().strftime("%Y-%m-%d")

        search_payload: dict = {
            "nodeId": node_id,
            "nodeDesc": node_id,
            "category": category,
            "statusType": status_type,
            "filingDateStart": start_date,
            "filingDateEnd": end_date,
            "lastName": "",
            "firstName": "",
            "caseNumber": "",
            "dateOfBirth": "",
        }
        if case_type_id:
            search_payload["caseTypeId"] = case_type_id

        headers = {
            **_API_HEADERS,
            "Origin": self.portal_base,
            "Referer": f"{self.portal_base}/",
        }

        results: list[tuple[dict, dict]] = []

        async with self.browser_context() as ctx:
            page = await ctx.new_page()

            # Establish session once
            try:
                await page.goto(
                    self.portal_base + "/",
                    wait_until="domcontentloaded",
                    timeout=30_000,
                )
                await asyncio.sleep(2)
            except Exception as exc:
                logger.error(
                    "odyssey_portal_unreachable",
                    portal=self.portal_base,
                    error=str(exc)[:200],
                )
                return []

            # Case search
            try:
                resp = await ctx.request.post(
                    self.portal_base + _SEARCH_PATH,
                    data=json.dumps(search_payload),
                    headers=headers,
                )
                if resp.status != 200:
                    logger.warning("odyssey_search_bad_status", status=resp.status)
                    return []
                data = await resp.json()
            except Exception as exc:
                logger.error("odyssey_search_failed", error=str(exc)[:200])
                return []

            cases = _extract_cases(data)
            logger.info(
                "odyssey_cases_found",
                portal=self.portal_base,
                category=category,
                count=len(cases),
            )

            # Fetch detail for each case in the same browser context
            for case in cases:
                case_id = (
                    case.get("CaseID")
                    or case.get("caseId")
                    or case.get("caseID")
                    or case.get("id")
                    or ""
                )
                detail: dict = {}
                if case_id:
                    try:
                        detail_payload: dict = {"caseId": str(case_id)}
                        if detail_node_id:
                            detail_payload["nodeId"] = detail_node_id
                        detail_resp = await ctx.request.post(
                            self.portal_base + _DETAIL_PATH,
                            data=json.dumps(detail_payload),
                            headers=headers,
                        )
                        if detail_resp.status == 200:
                            detail = await detail_resp.json()
                        await self._rate_limit_sleep()
                    except Exception as exc:
                        logger.debug(
                            "odyssey_detail_failed",
                            case_id=case_id,
                            error=str(exc)[:100],
                        )

                results.append((case, detail))

        return results
