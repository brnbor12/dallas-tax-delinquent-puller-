import { create } from 'zustand'

interface MapBounds {
  north: number
  south: number
  east: number
  west: number
}

interface MapStore {
  selectedPropertyId: number | null
  hoveredPropertyId: number | null
  mapBounds: MapBounds | null
  mapCenter: [number, number]
  mapZoom: number
  showHeatmap: boolean
  setSelectedProperty: (id: number | null) => void
  setHoveredProperty: (id: number | null) => void
  setMapBounds: (bounds: MapBounds) => void
  setMapView: (center: [number, number], zoom: number) => void
  toggleHeatmap: () => void
}

export const useMapStore = create<MapStore>((set) => ({
  selectedPropertyId: null,
  hoveredPropertyId: null,
  mapBounds: null,
  mapCenter: [-96.7970, 32.7767],  // Default: Dallas TX
  mapZoom: 9,
  showHeatmap: false,

  setSelectedProperty: (id) => set({ selectedPropertyId: id }),
  setHoveredProperty: (id) => set({ hoveredPropertyId: id }),
  setMapBounds: (bounds) => set({ mapBounds: bounds }),
  setMapView: (center, zoom) => set({ mapCenter: center, mapZoom: zoom }),
  toggleHeatmap: () => set((state) => ({ showHeatmap: !state.showHeatmap })),
}))
