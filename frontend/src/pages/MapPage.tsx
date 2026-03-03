import { FilterPanel } from '@/components/filters/FilterPanel'
import { PropertyList } from '@/components/property/PropertyList'
import { MapView } from '@/components/map/MapView'
import { PropertyDetail } from '@/components/property/PropertyDetail'

export function MapPage() {
  return (
    <div className="flex h-full flex-col">
      {/* Main content: filter | list | map */}
      <div className="flex flex-1 overflow-hidden">
        {/* Filter sidebar */}
        <FilterPanel />

        {/* Property list */}
        <div className="w-80 flex-shrink-0 overflow-hidden border-r border-gray-200">
          <PropertyList />
        </div>

        {/* Map area (relative so PropertyDetail can be positioned within) */}
        <div className="relative flex-1">
          <MapView />
          <PropertyDetail />
        </div>
      </div>
    </div>
  )
}
