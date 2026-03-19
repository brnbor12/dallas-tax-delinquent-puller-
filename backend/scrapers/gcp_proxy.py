"""
GCP Cloud Function proxy for bypassing WAF/IP blocks.

The proxy function is deployed at GCP_PROXY_URL and accepts:
    GET {GCP_PROXY_URL}?url={encoded_target_url}

Authentication: Uses gcloud identity token from the host machine.
The token is refreshed automatically when expired (~1 hour lifetime).

Usage:
    from scrapers.gcp_proxy import make_proxied_client

    async with make_proxied_client() as client:
        # This transparently routes through GCP:
        resp = await client.get("https://polkflpa.gov/some/path")
"""

from __future__ import annotations

import os
import subprocess
import time
import structlog
from urllib.parse import quote

import httpx

logger = structlog.get_logger(__name__)

GCP_PROXY_URL = os.environ.get("GCP_PROXY_URL", "")

# Cache the identity token (valid ~1 hour)
_token_cache: dict = {"token": "", "expires": 0}


def _get_identity_token() -> str:
    """Get a GCP identity token from env var or gcloud CLI."""
    # First: check env var (set by host cron or .env)
    env_token = os.environ.get("GCP_PROXY_TOKEN", "")
    if env_token:
        return env_token

    # Fallback: try gcloud CLI (only works if gcloud is installed)
    now = time.time()
    if _token_cache["token"] and _token_cache["expires"] > now:
        return _token_cache["token"]

    try:
        result = subprocess.run(
            ["gcloud", "auth", "print-identity-token"],
            capture_output=True, text=True, timeout=10,
        )
        token = result.stdout.strip()
        if token and not token.startswith("ERROR"):
            _token_cache["token"] = token
            _token_cache["expires"] = now + 3000
            return token
    except Exception as exc:
        logger.warning("gcp_proxy_token_error", error=str(exc)[:80])

    return _token_cache.get("token", "")


def proxy_url(target_url: str) -> str:
    """Convert a direct URL into a proxied URL."""
    return f"{GCP_PROXY_URL}?url={quote(target_url, safe='%')}"


def proxy_headers() -> dict[str, str]:
    """Return auth headers for the GCP proxy."""
    token = _get_identity_token()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


class GCPProxiedTransport(httpx.AsyncBaseTransport):
    """
    httpx transport that routes all requests through the GCP proxy.
    Drop-in replacement — scrapers don't need to change their URLs.
    """

    def __init__(self):
        self._inner = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        from urllib.parse import unquote
        # Decode percent-encoded chars so httpx re-encodes them exactly once
        original_url = unquote(str(request.url))

        # Only keep safe headers — drop host/content-length from original request
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }
        headers.update(proxy_headers())

        new_request = httpx.Request(
            method="GET",
            url=GCP_PROXY_URL,
            params={"url": original_url},
            headers=headers,
        )

        return await self._inner.handle_async_request(new_request)

    async def aclose(self) -> None:
        await self._inner.aclose()


def make_proxied_client(**kwargs) -> httpx.AsyncClient:
    """
    Create an httpx.AsyncClient that routes through GCP proxy.
    Falls back to direct requests if GCP_PROXY_URL is not set.
    """
    if GCP_PROXY_URL:
        logger.info("gcp_proxy_enabled", proxy=GCP_PROXY_URL[:60])
        kwargs["transport"] = GCPProxiedTransport()
        # Increase timeout since requests go through an extra hop
        if "timeout" not in kwargs:
            kwargs["timeout"] = 180.0
    return httpx.AsyncClient(**kwargs)
