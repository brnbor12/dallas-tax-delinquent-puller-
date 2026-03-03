export type ScoreTier = 'hot' | 'warm' | 'cold'

export interface Score {
  total_score: number
  score_tier: ScoreTier | null
  score_breakdown: Record<string, number> | null
  indicator_count: number
  last_scored_at: string
}

export interface Indicator {
  id: number
  indicator_type: string
  status: 'active' | 'resolved' | 'expired'
  source: string
  source_url: string | null
  amount_cents: number | null
  filing_date: string | null
  expiry_date: string | null
  case_number: string | null
  created_at: string
}

export interface Owner {
  name_raw: string
  mailing_address: string | null
  mailing_city: string | null
  mailing_state: string | null
  mailing_zip: string | null
  owner_type: string | null
  is_absentee: boolean | null
  is_out_of_state: boolean | null
}

export interface Listing {
  source: string
  list_price: number | null
  original_price: number | null
  days_on_market: number | null
  price_reductions: number
  listing_status: string | null
  listed_date: string | null
  last_price_cut: string | null
}

export interface County {
  fips_code: string
  name: string
  state_abbr: string
}

export interface PropertyListItem {
  id: number
  address_raw: string
  address_line1: string | null
  address_city: string | null
  address_state: string | null
  address_zip: string | null
  property_type: string | null
  assessed_value: number | null
  score: Score | null
  active_indicator_types: string[]
  lat: number | null
  lng: number | null
}

export interface PropertyDetail extends PropertyListItem {
  apn: string | null
  year_built: number | null
  sqft: number | null
  bedrooms: number | null
  bathrooms: number | null
  lot_size_sqft: number | null
  zoning: string | null
  market_value: number | null
  last_sale_date: string | null
  last_sale_price: number | null
  data_source: string | null
  created_at: string
  updated_at: string
  county: County | null
  owners: Owner[]
  indicators: Indicator[]
  listings: Listing[]
}

export interface PropertyListResponse {
  total: number
  page: number
  page_size: number
  results: PropertyListItem[]
}

export interface MapPoint {
  id: number
  lat: number
  lng: number
  tier: ScoreTier
  score: number
}

export interface GeoJsonFeatureCollection {
  type: 'FeatureCollection'
  features: Array<{
    type: 'Feature'
    geometry: { type: 'Point'; coordinates: [number, number] }
    properties: { id: number; tier: ScoreTier; score: number }
  }>
}
