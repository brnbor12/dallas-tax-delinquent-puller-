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
        portal_base = "https://courtsportal.dallascounty.org/DALLASPROD"
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
import re
import structlog
from datetime import date, timedelta
from typing import Any

from bs4 import BeautifulSoup

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

            # Step 2b: Navigate to the case search page to set session context
            # Some Odyssey deployments require visiting a search page before
            # the API will accept requests.
            try:
                await page.goto(
                    self.portal_base + "/Home/Dashboard/29",
                    wait_until="domcontentloaded",
                    timeout=15_000,
                )
                await asyncio.sleep(1)
            except Exception:
                pass  # non-fatal — Smart Search page, best-effort

            # Case search — try browser-native fetch() first (sends proper
            # Sec-Fetch-* headers that ASP.NET MVC checks), fall back to
            # ctx.request.post() if the page.evaluate approach fails.
            data = None
            try:
                search_url = self.portal_base + _SEARCH_PATH
                result = await page.evaluate(
                    """async ([url, payload]) => {
                        try {
                            const resp = await fetch(url, {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json',
                                    'Accept': 'application/json, text/plain, */*',
                                },
                                body: JSON.stringify(payload),
                                credentials: 'include',
                            });
                            const text = await resp.text();
                            return {status: resp.status, body: text};
                        } catch(e) {
                            return {status: 0, body: String(e)};
                        }
                    }""",
                    [search_url, search_payload],
                )
                status = result.get("status", 0)
                body_text = result.get("body", "")
                if status == 200:
                    import json as _json
                    data = _json.loads(body_text)
                else:
                    body_snippet = body_text[body_text.find("<h2>"):body_text.find("<h2>") + 400] if "<h2>" in body_text else body_text[:400]
                    logger.warning(
                        "odyssey_search_bad_status",
                        portal=self.portal_base,
                        status=status,
                        node_id=search_payload.get("nodeId"),
                        category=category,
                        body=body_snippet,
                        method="fetch",
                    )
            except Exception as exc:
                logger.debug("odyssey_fetch_fallback", error=str(exc)[:100])
                # Fall back to ctx.request.post()
                try:
                    resp = await ctx.request.post(
                        self.portal_base + _SEARCH_PATH,
                        data=json.dumps(search_payload),
                        headers=headers,
                    )
                    if resp.status != 200:
                        body = await resp.text()
                        body_snippet = body[body.find("<h2>"):body.find("<h2>") + 400] if "<h2>" in body else body[:400]
                        logger.warning(
                            "odyssey_search_bad_status",
                            portal=self.portal_base,
                            status=resp.status,
                            node_id=search_payload.get("nodeId"),
                            category=category,
                            body=body_snippet,
                            method="ctx_request",
                        )
                        return []
                    data = await resp.json()
                except Exception as exc2:
                    logger.error("odyssey_search_failed", error=str(exc2)[:200])
                    return []

            if data is None:
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

    # ------------------------------------------------------------------
    # Hearing Search UI fallback
    # ------------------------------------------------------------------

    async def _get_cases_via_hearing_search(
        self,
        court_location: str,
        lookback_days: int = 30,
        lookahead_days: int = 90,
        max_cases: int = 200,
        search_value: str = "",
        first_name: str = "",
        search_by_type: str = "PartyName",
    ) -> list[dict]:
        """
        Fallback scraper: interacts with Hearing Search (Dashboard/26) via
        Playwright browser UI instead of the broken REST API.

        Dashboard/26 has CaptchaEnabled=False so no reCAPTCHA is needed.
        Results come from an HTML table rendered after form submission.

        Returns list of dicts:
            case_number, case_style, case_url, hearing_date, court_location
        """
        if not self.portal_base:
            raise ValueError("portal_base must be set")

        search_url = f"{self.portal_base}/Home/Dashboard/26"
        date_from = (date.today() - timedelta(days=lookback_days)).strftime("%m/%d/%Y")
        date_to = (date.today() + timedelta(days=lookahead_days)).strftime("%m/%d/%Y")

        async with self.browser_context() as ctx:
            page = await ctx.new_page()

            # Establish portal session first
            try:
                await page.goto(
                    self.portal_base + "/",
                    wait_until="domcontentloaded",
                    timeout=30_000,
                )
                await asyncio.sleep(1)
            except Exception as exc:
                logger.error(
                    "odyssey_portal_unreachable",
                    portal=self.portal_base,
                    error=str(exc)[:200],
                )
                return []

            # Navigate to Hearing Search
            try:
                await page.goto(search_url, wait_until="domcontentloaded", timeout=30_000)
                await asyncio.sleep(2)
            except Exception as exc:
                logger.error(
                    "odyssey_hearing_load_failed",
                    url=search_url,
                    error=str(exc)[:200],
                )
                return []

            logger.info(
                "odyssey_hearing_page_loaded",
                title=await page.title(),
                url=page.url,
            )

            # Dump all select elements and their options for diagnosis on first call
            try:
                select_info = await page.evaluate("""
                    () => Array.from(document.querySelectorAll('select')).map(s => ({
                        id: s.id, name: s.name, count: s.options.length,
                        options: Array.from(s.options).map(o => ({v: o.value, t: o.text})).slice(0, 20)
                    }))
                """)
                logger.info("odyssey_hearing_selects", selects=select_info)
            except Exception:
                pass

            # Try to set court location via select (handles hidden Kendo selects too)
            location_set = False
            try:
                # Use JavaScript to find and set the location select with matching option
                location_set = await page.evaluate(
                    """([target]) => {
                        for (const sel of document.querySelectorAll('select')) {
                            for (const opt of sel.options) {
                                if (opt.value === target || opt.text.trim() === target) {
                                    sel.value = opt.value;
                                    sel.dispatchEvent(new Event('change', {bubbles: true}));
                                    // Notify Kendo if present
                                    if (window.$ && $(sel).data('kendoDropDownList')) {
                                        $(sel).data('kendoDropDownList').value(opt.value);
                                        $(sel).data('kendoDropDownList').trigger('change');
                                    }
                                    return true;
                                }
                            }
                        }
                        return false;
                    }""",
                    [court_location],
                )
            except Exception as exc:
                logger.debug("odyssey_hearing_location_js_error", error=str(exc)[:100])

            logger.info(
                "odyssey_hearing_location_set",
                court_location=court_location,
                success=location_set,
            )

            # Skip if no search criteria
            if search_by_type == "PartyName" and not search_value and not first_name:
                logger.debug("odyssey_hearing_skip_empty_party_search")
                return []

            # CRITICAL: set SearchByType first — reveals hidden name fields via JS
            try:
                await page.select_option(
                    "select[name='SearchCriteria.SearchByType']",
                    value=search_by_type,
                )
                await asyncio.sleep(1.5)
                logger.debug("odyssey_hearing_searchby_set", search_by_type=search_by_type)
            except Exception as exc:
                logger.warning("odyssey_hearing_searchby_failed", error=str(exc)[:100])

            # Fill LastName — force visible via JS then fill
            if search_value and search_by_type == "PartyName":
                await page.evaluate(
                    """([val]) => {
                        const el = document.getElementById('txtHSLastName');
                        if (el) {
                            el.style.display = '';
                            el.removeAttribute('hidden');
                            el.value = val;
                            el.dispatchEvent(new Event('input', {bubbles:true}));
                            el.dispatchEvent(new Event('change', {bubbles:true}));
                        }
                    }""",
                    [search_value],
                )
                logger.debug("odyssey_hearing_name_forced", value=search_value)

            # First name is required by portal for Party Name searches — default to "%" wildcard
            _fn = first_name if first_name else "%"
            try:
                await page.evaluate(
                    """([val]) => {
                        const el = document.getElementById('txtHSFirstName');
                        if (el) { el.value = val; el.dispatchEvent(new Event('change', {bubbles:true})); }
                    }""",
                    [_fn],
                )
            except Exception:
                pass

            # Fill date range — confirmed field names
            try:
                await page.fill("input[name='SearchCriteria.DateFrom']", date_from)
                await page.fill("input[name='SearchCriteria.DateTo']", date_to)
                logger.debug("odyssey_hearing_dates_filled", date_from=date_from, date_to=date_to)
            except Exception as exc:
                logger.warning("odyssey_hearing_date_fill_failed", error=str(exc)[:100])

            # Submit — confirmed: input[name='Search'] id='btnHSSubmit'
            submitted = False
            for sel in ["input[name='Search']", "input[id='btnHSSubmit']", "input[type='submit']", "button[type='submit']"]:
                try:
                    if await page.locator(sel).count() > 0:
                        await page.locator(sel).first.click()
                        submitted = True
                        logger.info("odyssey_hearing_submitted", selector=sel)
                        break
                except Exception:
                    continue

            if not submitted:
                logger.warning("odyssey_hearing_no_submit_found")
                return []

            # Wait for results to load
            try:
                await page.wait_for_load_state("networkidle", timeout=20_000)
            except Exception:
                await asyncio.sleep(4)

            current_url = page.url
            logger.info("odyssey_hearing_after_submit_url", url=current_url)

            # Get results HTML
            html = await page.content()

            cases = self._parse_hearing_table(html, court_location)
            logger.info(
                "odyssey_hearing_cases_parsed",
                count=len(cases),
                court_location=court_location,
            )
            return cases[:max_cases]

    def _parse_hearing_table(self, html: str, court_location: str) -> list[dict]:
        """
        Parse Hearing Search result HTML into a list of case dicts.
        Handles various Tyler ePortal table structures.
        """
        soup = BeautifulSoup(html, "lxml")
        results: list[dict] = []

        # Look for any table with case-related headers
        for table in soup.find_all("table"):
            ths = table.find_all("th")
            headers = [th.get_text(strip=True) for th in ths]
            if not headers:
                # Try to infer from first row
                first_row = table.find("tr")
                if first_row:
                    headers = [td.get_text(strip=True) for td in first_row.find_all(["td", "th"])]

            h_lower = " ".join(h.lower() for h in headers)
            # Only process tables that look like case/hearing results
            if not any(kw in h_lower for kw in ["case", "style", "party", "hearing", "number"]):
                continue

            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if not cells:
                    continue
                row_data: dict = {"_court_location": court_location}
                for i, cell in enumerate(cells):
                    col = headers[i] if i < len(headers) else f"col_{i}"
                    row_data[col] = cell.get_text(strip=True)
                    # Grab href from case number link
                    link = cell.find("a", href=True)
                    if link:
                        href = link["href"]
                        if not href.startswith("http"):
                            href = self.portal_base.rstrip("/") + "/" + href.lstrip("/")
                        row_data[f"{col}_href"] = href
                        # Prefer link text as value for case number fields
                        if not row_data[col]:
                            row_data[col] = link.get_text(strip=True)

                # Only keep rows that have something that looks like a case number
                combined_text = " ".join(str(v) for v in row_data.values())
                if re.search(r"\d{2}-\d{4,}", combined_text) or re.search(r"[A-Z]{2,}-\d{2}", combined_text):
                    results.append(row_data)

        # If no structured table found, try a Kendo grid data extraction via JSON
        if not results:
            # Tyler ePortal sometimes embeds grid data as JSON in a script tag
            for script in soup.find_all("script"):
                text = script.string or ""
                if '"CaseNumber"' in text or '"caseNumber"' in text:
                    try:
                        # Extract JSON array from script
                        m = re.search(r'\[(\{[^;]+\})\]', text, re.DOTALL)
                        if m:
                            data = json.loads(f"[{m.group(1)}]")
                            results = [dict(r, _court_location=court_location) for r in data]
                            break
                    except Exception:
                        pass

        return results

    async def _get_address_from_case_page(self, case_url: str) -> str | None:
        """
        Navigate to a case detail page URL and attempt to extract a property address.
        Returns the first address-like string found in party sections, or None.
        """
        if not case_url:
            return None

        async with self.browser_context() as ctx:
            page = await ctx.new_page()
            try:
                await page.goto(case_url, wait_until="domcontentloaded", timeout=30_000)
                await asyncio.sleep(2)
                html = await page.content()
            except Exception as exc:
                logger.debug("odyssey_case_page_failed", url=case_url, error=str(exc)[:100])
                return None

        soup = BeautifulSoup(html, "lxml")

        # Look for address-like text in party sections
        addr_pattern = re.compile(r"\d+\s+\w+.{0,60}(?:ST|AVE|BLVD|DR|LN|RD|WAY|CIR|CT|PKWY|HWY)[,\s]", re.I)

        # Try: tables, divs with class/id containing "party" or "address"
        for container in soup.find_all(
            ["div", "section", "table", "p"],
            class_=re.compile(r"party|address|contact", re.I),
        ):
            text = container.get_text(" ", strip=True)
            m = addr_pattern.search(text)
            if m:
                return m.group(0).strip()

        # Fallback: scan all text nodes
        full_text = soup.get_text(" ")
        m = addr_pattern.search(full_text)
        if m:
            return m.group(0).strip()

        return None
