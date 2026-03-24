import { useFilterStore } from '@/stores/filterStore'
import { INDICATOR_TYPES, type IndicatorType } from '@/types/filter'

const US_STATES = [
  'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA',
  'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
  'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT',
  'VA','WA','WV','WI','WY',
]

export function FilterPanel() {
  const { filters, setFilter, toggleIndicator, resetFilters } = useFilterStore()

  return (
    <aside className="flex h-full flex-col overflow-hidden border-r border-gray-200 bg-white w-64">
      <div className="flex items-center justify-between border-b border-gray-200 px-4 py-3">
        <h2 className="text-sm font-semibold text-gray-900">Filters</h2>
        <button
          onClick={resetFilters}
          className="text-xs text-blue-600 hover:text-blue-800"
        >
          Reset
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-5">
        {/* Geography */}
        <section>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
            Geography
          </h3>
          <select
            value={filters.state}
            onChange={(e) => setFilter('state', e.target.value)}
            className="w-full rounded-md border border-gray-300 px-2.5 py-1.5 text-sm focus:border-blue-500 focus:outline-none"
          >
            <option value="">All States</option>
            {US_STATES.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </section>

        {/* Motivated Seller Signals */}
        <section>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
            Signals
          </h3>
          <div className="space-y-1.5">
            {INDICATOR_TYPES.map(({ value, label }) => (
              <label key={value} className="flex cursor-pointer items-center gap-2">
                <input
                  type="checkbox"
                  checked={filters.indicator_types.includes(value as IndicatorType)}
                  onChange={() => toggleIndicator(value as IndicatorType)}
                  className="h-3.5 w-3.5 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="text-sm text-gray-700">{label}</span>
              </label>
            ))}
          </div>
        </section>

        {/* Score */}
        <section>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
            Score
          </h3>
          <div className="space-y-2">
            <div>
              <label className="text-xs text-gray-500">Min score: {filters.score_min}</label>
              <input
                type="range"
                min={0}
                max={100}
                step={5}
                value={filters.score_min}
                onChange={(e) => setFilter('score_min', Number(e.target.value))}
                className="w-full accent-blue-600"
              />
            </div>
            <div className="flex gap-2">
              {(['hot', 'warm', 'nurture', 'cold'] as const).map((tier) => (
                <button
                  key={tier}
                  onClick={() => setFilter('score_tier', filters.score_tier === tier ? '' : tier)}
                  className={`flex-1 rounded py-1 text-xs font-medium capitalize ${
                    filters.score_tier === tier
                      ? tier === 'hot'
                        ? 'bg-red-500 text-white'
                        : tier === 'warm'
                        ? 'bg-orange-500 text-white'
                        : tier === 'nurture'
                        ? 'bg-yellow-500 text-white'
                        : 'bg-blue-500 text-white'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  {tier}
                </button>
              ))}
            </div>
          </div>
        </section>

        {/* Property */}
        <section>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
            Property
          </h3>
          <div className="space-y-2">
            <select
              value={filters.property_type}
              onChange={(e) => setFilter('property_type', e.target.value)}
              className="w-full rounded-md border border-gray-300 px-2.5 py-1.5 text-sm focus:border-blue-500 focus:outline-none"
            >
              <option value="">All Types</option>
              <option value="SFR">Single Family</option>
              <option value="MFR">Multi-Family</option>
              <option value="Commercial">Commercial</option>
              <option value="Land">Land</option>
            </select>

            <div className="grid grid-cols-2 gap-2">
              <input
                type="number"
                placeholder="Min beds"
                value={filters.bedrooms_min}
                onChange={(e) => setFilter('bedrooms_min', e.target.value === '' ? '' : Number(e.target.value))}
                className="rounded-md border border-gray-300 px-2.5 py-1.5 text-sm focus:border-blue-500 focus:outline-none"
              />
              <input
                type="number"
                placeholder="Min DOM"
                value={filters.dom_min}
                onChange={(e) => setFilter('dom_min', e.target.value === '' ? '' : Number(e.target.value))}
                className="rounded-md border border-gray-300 px-2.5 py-1.5 text-sm focus:border-blue-500 focus:outline-none"
              />
            </div>
          </div>
        </section>

        {/* Owner */}
        <section>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
            Owner
          </h3>
          <label className="flex cursor-pointer items-center gap-2">
            <input
              type="checkbox"
              checked={filters.out_of_state_only}
              onChange={(e) => setFilter('out_of_state_only', e.target.checked)}
              className="h-3.5 w-3.5 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span className="text-sm text-gray-700">Out-of-state owners only</span>
          </label>
        </section>

        {/* Sort */}
        <section>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
            Sort
          </h3>
          <select
            value={filters.sort_by}
            onChange={(e) => setFilter('sort_by', e.target.value as FilterPanel_SortBy)}
            className="w-full rounded-md border border-gray-300 px-2.5 py-1.5 text-sm focus:border-blue-500 focus:outline-none"
          >
            <option value="score_desc">Score (High → Low)</option>
            <option value="filing_date_desc">Filing Date (Newest)</option>
            <option value="assessed_value_asc">Assessed Value (Low → High)</option>
          </select>
        </section>
      </div>
    </aside>
  )
}

type FilterPanel_SortBy = 'score_desc' | 'filing_date_desc' | 'assessed_value_asc'
