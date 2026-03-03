import { useState } from 'react'
import { useProperties } from '@/hooks/useProperties'
import { useMapStore } from '@/stores/mapStore'
import { PropertyCard } from './PropertyCard'

export function PropertyList() {
  const [page, setPage] = useState(1)
  const { data, isLoading, isFetching, isError } = useProperties(page)
  const selectedPropertyId = useMapStore((s) => s.selectedPropertyId)

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center text-gray-400">
        Loading properties…
      </div>
    )
  }

  if (isError) {
    return (
      <div className="flex h-full items-center justify-center text-red-500">
        Failed to load properties
      </div>
    )
  }

  const { results = [], total = 0, page: currentPage, page_size } = data ?? {}

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="border-b border-gray-200 px-3 py-2">
        <p className="text-sm text-gray-500">
          {isFetching ? 'Updating…' : `${total.toLocaleString()} properties`}
        </p>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {results.length === 0 ? (
          <p className="text-center text-gray-400 mt-8">
            No properties match your filters
          </p>
        ) : (
          results.map((prop) => (
            <PropertyCard
              key={prop.id}
              property={prop}
              isSelected={prop.id === selectedPropertyId}
            />
          ))
        )}
      </div>

      {/* Pagination */}
      {total > (page_size ?? 50) && (
        <div className="border-t border-gray-200 flex items-center justify-between px-3 py-2">
          <button
            disabled={currentPage <= 1}
            onClick={() => setPage((p) => p - 1)}
            className="rounded px-2 py-1 text-sm text-gray-600 hover:bg-gray-100 disabled:opacity-40"
          >
            ← Prev
          </button>
          <span className="text-xs text-gray-400">
            Page {currentPage} of {Math.ceil(total / (page_size ?? 50))}
          </span>
          <button
            disabled={currentPage >= Math.ceil(total / (page_size ?? 50))}
            onClick={() => setPage((p) => p + 1)}
            className="rounded px-2 py-1 text-sm text-gray-600 hover:bg-gray-100 disabled:opacity-40"
          >
            Next →
          </button>
        </div>
      )}
    </div>
  )
}
