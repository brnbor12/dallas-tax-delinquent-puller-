"""
Geocoding service — converts address strings to (lat, lng).

Supports two backends:
  - Nominatim (free, US addresses, rate-limited to 1 req/sec)
  - Google Maps Geocoding API (paid, more accurate)

Results are cached in Redis with a 30-day TTL to minimize API calls.
"""

from __future__ import annotations

import hashlib
import json
import structlog
import time

import httpx

from app.core.config import settings

logger = structlog.get_logger(__name__)

_redis_client = None


def _get_redis():
    global _redis_client
    if _redis_client is None:
        import redis
        _redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis_client


def _cache_key(address: str) -> str:
    h = hashlib.md5(address.lower().strip().encode()).hexdigest()
    return f"geocode:{h}"


def geocode_address(address: str) -> tuple[float | None, float | None]:
    """
    Returns (lat, lng) or (None, None) if geocoding fails.
    Results are cached in Redis for 30 days.
    """
    if not address:
        return None, None

    cache_key = _cache_key(address)

    try:
        cached = _get_redis().get(cache_key)
        if cached:
            result = json.loads(cached)
            return result["lat"], result["lng"]
    except Exception:
        pass  # Redis unavailable — proceed without cache

    lat, lng = _geocode_fresh(address)

    if lat is not None:
        try:
            _get_redis().setex(
                cache_key,
                30 * 24 * 3600,  # 30 days
                json.dumps({"lat": lat, "lng": lng}),
            )
        except Exception:
            pass

    return lat, lng


def _geocode_fresh(address: str) -> tuple[float | None, float | None]:
    if settings.geocoder == "google" and settings.google_maps_api_key:
        return _geocode_google(address)
    # Census Bureau geocoder: free, no API key, no rate limits, US only
    result = _geocode_census(address)
    if result[0] is not None:
        return result
    # Nominatim fallback
    return _geocode_nominatim(address)


def _geocode_census(address: str) -> tuple[float | None, float | None]:
    """
    US Census Bureau Geocoding API — free, no key, no strict rate limit.
    https://geocoding.geo.census.gov/geocoder/
    """
    try:
        resp = httpx.get(
            "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress",
            params={
                "address": address,
                "benchmark": "Public_AR_Current",
                "format": "json",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        matches = data.get("result", {}).get("addressMatches", [])
        if matches:
            coords = matches[0]["coordinates"]
            return float(coords["y"]), float(coords["x"])  # lat, lng
    except Exception as exc:
        logger.warning("census_geocode_failed", address=address, error=str(exc))
    return None, None


def _geocode_nominatim(address: str) -> tuple[float | None, float | None]:
    """OpenStreetMap Nominatim — fallback only. Rate limit: 1 req/sec."""
    try:
        time.sleep(1.1)  # Nominatim ToS: max 1 req/sec
        resp = httpx.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": address, "format": "json", "limit": 1, "countrycodes": "us"},
            headers={"User-Agent": f"MotivatedSellerApp/1.0 ({settings.nominatim_email})"},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception as exc:
        logger.warning("nominatim_geocode_failed", address=address, error=str(exc))
    return None, None


def _geocode_google(address: str) -> tuple[float | None, float | None]:
    """Google Maps Geocoding API. Requires GOOGLE_MAPS_API_KEY.

    Only accepts ROOFTOP and RANGE_INTERPOLATED results — skips city/zip-level
    approximations that would place all unmatched addresses at a city centroid.
    """
    try:
        resp = httpx.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": address, "key": settings.google_maps_api_key},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("results"):
            result = data["results"][0]
            location_type = result.get("geometry", {}).get("location_type", "")
            if location_type in ("ROOFTOP", "RANGE_INTERPOLATED"):
                loc = result["geometry"]["location"]
                return loc["lat"], loc["lng"]
    except Exception as exc:
        logger.warning("google_geocode_failed", address=address, error=str(exc))
    return None, None
