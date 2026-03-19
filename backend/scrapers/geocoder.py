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
import re
import structlog
import time

import httpx

from app.core.config import settings

logger = structlog.get_logger(__name__)

# Approximate bounding boxes (lat_min, lat_max, lng_min, lng_max) per US state
_STATE_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    "AL": (30.14, 35.01, -88.47, -84.89), "AK": (54.56, 71.39, -168.00, -129.99),
    "AZ": (31.33, 37.00, -114.82, -109.04), "AR": (33.00, 36.50, -94.62, -89.64),
    "CA": (32.53, 42.01, -124.41, -114.13), "CO": (36.99, 41.00, -109.06, -102.04),
    "CT": (40.98, 42.05, -73.73, -71.79), "DC": (38.79, 38.99, -77.12, -76.91),
    "DE": (38.45, 39.84, -75.79, -75.05), "FL": (24.54, 31.00, -87.63, -80.03),
    "GA": (30.36, 35.00, -85.61, -80.84), "HI": (18.91, 22.24, -160.24, -154.81),
    "ID": (41.99, 49.00, -117.24, -111.04), "IL": (36.97, 42.51, -91.51, -87.02),
    "IN": (37.77, 41.77, -88.10, -84.78), "IA": (40.37, 43.50, -96.64, -90.14),
    "KS": (36.99, 40.00, -102.05, -94.59), "KY": (36.50, 39.15, -89.57, -81.96),
    "LA": (28.93, 33.02, -94.04, -88.82), "ME": (43.06, 47.46, -71.08, -66.95),
    "MD": (37.91, 39.72, -79.49, -75.05), "MA": (41.24, 42.89, -73.50, -69.93),
    "MI": (41.70, 48.18, -90.42, -82.41), "MN": (43.50, 49.38, -97.24, -89.49),
    "MS": (30.17, 35.01, -91.65, -88.10), "MO": (35.99, 40.61, -95.77, -89.10),
    "MT": (44.36, 49.00, -116.05, -104.04), "NE": (40.00, 43.00, -104.05, -95.31),
    "NV": (35.00, 42.00, -120.00, -114.04), "NH": (42.70, 45.31, -72.56, -70.61),
    "NJ": (38.93, 41.36, -75.56, -73.89), "NM": (31.33, 37.00, -109.05, -103.00),
    "NY": (40.47, 45.02, -79.76, -71.79), "NC": (33.84, 36.59, -84.32, -75.46),
    "ND": (45.93, 49.00, -104.05, -96.55), "OH": (38.40, 41.98, -84.82, -80.52),
    "OK": (33.62, 37.00, -103.00, -94.43), "OR": (41.99, 46.24, -124.57, -116.46),
    "PA": (39.72, 42.27, -80.52, -74.69), "RI": (41.15, 42.02, -71.91, -71.12),
    "SC": (32.04, 35.22, -83.36, -78.56), "SD": (42.48, 45.94, -104.06, -96.44),
    "TN": (34.98, 36.68, -90.31, -81.65), "TX": (25.83, 36.50, -106.65, -93.51),
    "UT": (36.99, 42.00, -114.05, -109.04), "VT": (42.73, 45.02, -73.44, -71.46),
    "VA": (36.54, 39.47, -83.68, -75.17), "WA": (45.54, 49.00, -124.68, -116.92),
    "WV": (37.20, 40.64, -82.65, -77.72), "WI": (42.49, 46.96, -92.89, -86.25),
    "WY": (40.99, 45.01, -111.05, -104.05),
}

_ZIP_RE = re.compile(r",?\s*\d{5}(-\d{4})?$")
_STATE_RE = re.compile(r",\s*([A-Z]{2})\s*(?:\d{5}|,|$)")


def _extract_state(address: str) -> str | None:
    m = _STATE_RE.search(address.upper())
    return m.group(1) if m else None


def _in_state_bounds(lat: float, lng: float, state: str) -> bool:
    bounds = _STATE_BOUNDS.get(state)
    if not bounds:
        return True
    lat_min, lat_max, lng_min, lng_max = bounds
    return lat_min <= lat <= lat_max and lng_min <= lng <= lng_max

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

    expected_state = _extract_state(address)

    # Census Bureau geocoder: free, no API key, no rate limits, US only
    result = _geocode_census(address)
    if result[0] is not None:
        if not expected_state or _in_state_bounds(result[0], result[1], expected_state):
            return result
        # Result landed in wrong state — likely a bad ZIP. Retry without it.
        logger.warning(
            "geocode_wrong_state",
            address=address,
            expected=expected_state,
            lat=result[0],
            lng=result[1],
        )
        address_no_zip = _ZIP_RE.sub("", address).strip()
        if address_no_zip != address:
            result = _geocode_census(address_no_zip)
            if result[0] is not None and _in_state_bounds(result[0], result[1], expected_state):
                return result

    # Nominatim removed — causes 429s when multiple workers run concurrently.
    # Properties without Census match are stored with NULL geometry.
    return None, None


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
