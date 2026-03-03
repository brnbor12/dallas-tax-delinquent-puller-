import { ScoreBadge } from './ScoreBadge'
import { IndicatorPill } from './IndicatorPill'
import { useMapStore } from '@/stores/mapStore'
import type { PropertyListItem } from '@/types/property'
import { clsx } from 'clsx'

interface Props {
  property: PropertyListItem
  isSelected: boolean
}

export function PropertyCard({ property, isSelected }: Props) {
  const setSelectedProperty = useMapStore((s) => s.setSelectedProperty)
  const setHoveredProperty = useMapStore((s) => s.setHoveredProperty)

  const addressDisplay =
    property.address_line1 ||
    property.address_raw.split(',')[0]

  const cityStateZip = [
    property.address_city,
    property.address_state,
    property.address_zip,
  ]
    .filter(Boolean)
    .join(', ')

  const assessedDisplay = property.assessed_value
    ? `$${(property.assessed_value / 100).toLocaleString()}`
    : null

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => setSelectedProperty(property.id)}
      onKeyDown={(e) => e.key === 'Enter' && setSelectedProperty(property.id)}
      onMouseEnter={() => setHoveredProperty(property.id)}
      onMouseLeave={() => setHoveredProperty(null)}
      className={clsx(
        'cursor-pointer rounded-lg border p-3 transition-colors hover:bg-gray-50',
        isSelected ? 'border-blue-500 bg-blue-50' : 'border-gray-200 bg-white',
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <p className="truncate font-medium text-gray-900">{addressDisplay}</p>
          {cityStateZip && (
            <p className="truncate text-sm text-gray-500">{cityStateZip}</p>
          )}
        </div>
        {property.score && (
          <ScoreBadge
            score={property.score.total_score}
            tier={property.score.score_tier}
            size="sm"
          />
        )}
      </div>

      {property.active_indicator_types.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {property.active_indicator_types.slice(0, 4).map((type) => (
            <IndicatorPill key={type} type={type} />
          ))}
          {property.active_indicator_types.length > 4 && (
            <span className="text-xs text-gray-400">
              +{property.active_indicator_types.length - 4} more
            </span>
          )}
        </div>
      )}

      {assessedDisplay && (
        <p className="mt-1.5 text-xs text-gray-400">
          Assessed: {assessedDisplay}
        </p>
      )}
    </div>
  )
}
