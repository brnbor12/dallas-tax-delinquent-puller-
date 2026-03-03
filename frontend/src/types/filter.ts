export const INDICATOR_TYPES = [
  { value: 'pre_foreclosure',  label: 'Pre-Foreclosure / NOD' },
  { value: 'foreclosure',      label: 'Foreclosure' },
  { value: 'tax_delinquent',   label: 'Tax Delinquent' },
  { value: 'probate',          label: 'Probate' },
  { value: 'lien',             label: 'Lien' },
  { value: 'eviction',         label: 'Eviction' },
  { value: 'code_violation',   label: 'Code Violation' },
  { value: 'vacant',           label: 'Vacant' },
  { value: 'absentee_owner',   label: 'Absentee Owner' },
  { value: 'price_reduction',  label: 'Price Reduction (MLS)' },
  { value: 'expired_listing',  label: 'Expired Listing' },
  { value: 'days_on_market',   label: 'Extended Days on Market' },
] as const

export type IndicatorType = typeof INDICATOR_TYPES[number]['value']

export interface FilterState {
  state: string
  zip_codes: string[]
  indicator_types: IndicatorType[]
  score_min: number
  score_tier: 'hot' | 'warm' | 'cold' | ''
  property_type: string
  year_built_min: number | ''
  year_built_max: number | ''
  assessed_min: number | ''
  assessed_max: number | ''
  bedrooms_min: number | ''
  dom_min: number | ''
  out_of_state_only: boolean
  sort_by: 'score_desc' | 'filing_date_desc' | 'assessed_value_asc'
}

export const DEFAULT_FILTERS: FilterState = {
  state: '',
  zip_codes: [],
  indicator_types: [],
  score_min: 0,
  score_tier: '',
  property_type: '',
  year_built_min: '',
  year_built_max: '',
  assessed_min: '',
  assessed_max: '',
  bedrooms_min: '',
  dom_min: '',
  out_of_state_only: false,
  sort_by: 'score_desc',
}
