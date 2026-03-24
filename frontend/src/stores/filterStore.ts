import { create } from 'zustand'
import { FilterState, DEFAULT_FILTERS, IndicatorType } from '@/types/filter'

interface FilterStore {
  filters: FilterState
  setFilter: <K extends keyof FilterState>(key: K, value: FilterState[K]) => void
  toggleIndicator: (type: IndicatorType) => void
  resetFilters: () => void
  toQueryParams: () => Record<string, string>
}

export const useFilterStore = create<FilterStore>((set, get) => ({
  filters: { ...DEFAULT_FILTERS },

  setFilter: (key, value) =>
    set((state) => ({ filters: { ...state.filters, [key]: value } })),

  toggleIndicator: (type) =>
    set((state) => {
      const types = state.filters.indicator_types
      const next = types.includes(type)
        ? types.filter((t) => t !== type)
        : [...types, type]
      return { filters: { ...state.filters, indicator_types: next } }
    }),

  resetFilters: () => set({ filters: { ...DEFAULT_FILTERS } }),

  toQueryParams: () => {
    const { filters } = get()
    const params: Record<string, string> = {}
    if (filters.state) params.state = filters.state
    if (filters.zip_codes.length) params.zip_codes = filters.zip_codes.join(',')
    if (filters.indicator_types.length) params.indicator_types = filters.indicator_types.join(',')
    if (filters.score_min > 0) params.score_min = String(filters.score_min)
    if (filters.score_tier) params.score_tier = filters.score_tier
    if (filters.property_type) params.property_type = filters.property_type
    if (filters.year_built_min !== '') params.year_built_min = String(filters.year_built_min)
    if (filters.year_built_max !== '') params.year_built_max = String(filters.year_built_max)
    if (filters.assessed_min !== '') params.assessed_min = String(filters.assessed_min)
    if (filters.assessed_max !== '') params.assessed_max = String(filters.assessed_max)
    if (filters.bedrooms_min !== '') params.bedrooms_min = String(filters.bedrooms_min)
    if (filters.dom_min !== '') params.dom_min = String(filters.dom_min)
    if (filters.out_of_state_only) params.out_of_state_only = 'true'
    params.sort_by = filters.sort_by
    return params
  },
}))
