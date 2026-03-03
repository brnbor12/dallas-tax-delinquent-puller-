"""
Address normalization utilities for fuzzy matching across data sources.

The TRW file uses uppercase abbreviated forms:
    "8687 N CENTRAL EXPY, DALLAS, TX 75243"
while NOTS PDFs produce mixed-case expanded forms:
    "8687 North Central Expressway, Dallas TX"

normalize_address() converts both to the same canonical form so that
pg_trgm similarity queries can link records across scrapers.
"""

from __future__ import annotations

import re

# USPS standard street type abbreviations (expanded form → abbrev).
# Sorted longest-first to avoid partial replacements (e.g. EXPRESSWAY before EXPRESS).
_STREET_TYPES: list[tuple[str, str]] = sorted(
    [
        ("ALLEY", "ALY"),
        ("AVENUE", "AVE"),
        ("BOULEVARD", "BLVD"),
        ("BYPASS", "BYP"),
        ("CIRCLE", "CIR"),
        ("COURT", "CT"),
        ("CROSSING", "XING"),
        ("DRIVE", "DR"),
        ("EXPRESSWAY", "EXPY"),
        ("EXTENSION", "EXT"),
        ("FREEWAY", "FWY"),
        ("HIGHWAY", "HWY"),
        ("LANE", "LN"),
        ("PARKWAY", "PKWY"),
        ("PLACE", "PL"),
        ("PLAZA", "PLZ"),
        ("ROAD", "RD"),
        ("ROUTE", "RTE"),
        ("SQUARE", "SQ"),
        ("STREET", "ST"),
        ("TERRACE", "TER"),
        ("TRAIL", "TRL"),
        ("TURNPIKE", "TPKE"),
        ("WAY", "WAY"),
    ],
    key=lambda x: -len(x[0]),  # longest first
)

# Directional abbreviations (expanded → short)
_DIRECTIONALS: list[tuple[str, str]] = sorted(
    [
        ("NORTHEAST", "NE"),
        ("NORTHWEST", "NW"),
        ("SOUTHEAST", "SE"),
        ("SOUTHWEST", "SW"),
        ("NORTH", "N"),
        ("SOUTH", "S"),
        ("EAST", "E"),
        ("WEST", "W"),
    ],
    key=lambda x: -len(x[0]),
)

# Known non-property addresses to reject (auction venues, courthouses, etc.)
# These show up in NOTS PDFs as the sale location, not the property address.
_BLOCKLIST_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"600\s+Commerce\s+St",          # George Allen Courts Building, Dallas
        r"Dallas\s+County\s+Courthouse",
        r"George\s+Allen",
        r"Courts?\s+Building",
        r"\bCourthouse\b",
        r"First\s+Floor\s+Lobby",
        r"^George\s+Allen",              # Some PDFs start with this
    ]
]


def normalize_address(raw: str) -> str:
    """
    Produce a canonical address string suitable for pg_trgm fuzzy matching.

    Transformations applied:
    - Uppercase
    - Remove punctuation (commas, periods, #, apostrophes) → space
    - Expand directionals to abbreviations  (NORTH → N)
    - Abbreviate street types              (EXPRESSWAY → EXPY)
    - Strip trailing state + zip           (, TX 75201 or TX 75201-1234)
    - Collapse whitespace

    Examples:
        "8687 North Central Expressway, Dallas, TX 75243"
            → "8687 N CENTRAL EXPY DALLAS"
        "8687 N CENTRAL EXPY, Dallas TX"
            → "8687 N CENTRAL EXPY DALLAS"
        "123 West Main Street, #4B, Irving TX 75060"
            → "123 W MAIN ST 4B IRVING"
    """
    if not raw:
        return ""

    s = raw.upper()

    # Remove common punctuation that differs between sources
    s = re.sub(r"[.,#';]", " ", s)

    # Normalize multi-character directionals first (NORTHEAST before NORTH)
    for long, short in _DIRECTIONALS:
        s = re.sub(rf"\b{long}\b", short, s)

    # Normalize street types
    for long, short in _STREET_TYPES:
        s = re.sub(rf"\b{long}\b", short, s)

    # Strip trailing state abbrev + zip: "TX 75201" or "TX 75201-1234"
    s = re.sub(r"\b[A-Z]{2}\s+\d{5}(-\d{4})?\s*$", "", s)

    # Remove any remaining non-alphanumeric except hyphens and spaces
    s = re.sub(r"[^A-Z0-9\s-]", " ", s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s


def is_blocklisted_address(raw: str) -> bool:
    """
    Return True if the address matches a known non-property location
    (e.g. courthouse auction venue). Used to filter garbled NOTS records.
    """
    for pattern in _BLOCKLIST_PATTERNS:
        if pattern.search(raw):
            return True
    return False


def parse_address_components(raw: str) -> tuple[str | None, str | None, str | None, str | None]:
    """
    Extract (line1, city, state, zip) from a raw address string.

    Handles common formats:
      "1633 LEDGESTONE DR, BRANDON, FL33511"      → line1, "BRANDON", "FL", "33511"
      "3410 W ROGERS AVE, TAMPA, FL 33611"        → line1, "TAMPA", "FL", "33611"
      "8203 LAKE JUNE RD, DALLAS, TX 75217"       → line1, "DALLAS", "TX", "75217"
      "Parcel 12-34-56-789, Polk County, FL"      → None (not a street address)
    """
    if not raw:
        return None, None, None, None

    raw = raw.strip()

    # Trailing state+zip: "FL33511" (no space), "FL 33511", "TX 75217"
    m = re.search(
        r",\s*([^,]+),\s*([A-Z]{2})\s*(\d{5})\s*$",
        raw,
        re.IGNORECASE,
    )
    if not m:
        # Try format with state+zip glued: "CITY, FL33511"
        m = re.search(
            r",\s*([^,]+),\s*([A-Z]{2})(\d{5})\s*$",
            raw,
            re.IGNORECASE,
        )
    if m:
        city = m.group(1).strip().upper()
        state = m.group(2).upper()
        zip_code = m.group(3)
        # line1 is everything before the first comma match
        line1 = raw[: m.start()].strip()
        return line1 if line1 else None, city, state, zip_code

    # State only, no zip (e.g. "..., Hillsborough County, FL")
    m = re.search(r",\s*([A-Z]{2})\s*$", raw, re.IGNORECASE)
    if m:
        state = m.group(1).upper()
        prefix = raw[: m.start()]
        parts = [p.strip() for p in prefix.rsplit(",", 1)]
        city = parts[-1].upper() if len(parts) > 1 else None
        line1 = parts[0] if len(parts) > 1 else None
        return line1 or None, city or None, state, None

    return None, None, None, None


def looks_like_property_address(raw: str) -> bool:
    """
    Quick sanity check: does this string look like a real street address?
    Requires a leading house/unit number followed by at least one word.
    """
    if not raw or len(raw.strip()) < 8:
        return False
    # Must start with a number (house number)
    return bool(re.match(r"^\d{1,6}\s+\w", raw.strip()))
