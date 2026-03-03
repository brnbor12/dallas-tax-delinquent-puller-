import { useEffect, useRef } from 'react'
import maplibregl from 'maplibre-gl'
import { useMapStore } from '@/stores/mapStore'
import { useMapPoints } from '@/hooks/useProperties'
import type { GeoJsonFeatureCollection } from '@/types/property'

const TIER_COLORS = {
  hot:  '#ef4444',
  warm: '#f97316',
  cold: '#3b82f6',
}

const EMPTY_GEOJSON: GeoJsonFeatureCollection = {
  type: 'FeatureCollection',
  features: [],
}

export function MapView() {
  const mapContainer = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)

  const { mapCenter, mapZoom, showHeatmap, setSelectedProperty, setMapBounds } = useMapStore()
  const { data: geoJson } = useMapPoints()

  // Initialize map once
  useEffect(() => {
    if (!mapContainer.current || mapRef.current) return

    const map = new maplibregl.Map({
      container: mapContainer.current,
      style: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
      center: mapCenter,
      zoom: mapZoom,
    })

    map.addControl(new maplibregl.NavigationControl(), 'top-right')
    map.addControl(new maplibregl.ScaleControl(), 'bottom-left')

    map.on('load', () => {
      // Add GeoJSON source for property points
      map.addSource('properties', {
        type: 'geojson',
        data: EMPTY_GEOJSON,
        cluster: true,
        clusterMaxZoom: 14,
        clusterRadius: 50,
      })

      // Cluster circles
      map.addLayer({
        id: 'clusters',
        type: 'circle',
        source: 'properties',
        filter: ['has', 'point_count'],
        paint: {
          'circle-color': [
            'step', ['get', 'point_count'],
            '#6b7280', 10,
            '#f97316', 50,
            '#ef4444',
          ],
          'circle-radius': [
            'step', ['get', 'point_count'],
            15, 10,
            20, 50,
            25,
          ],
          'circle-opacity': 0.85,
        },
      })

      // Cluster count labels
      map.addLayer({
        id: 'cluster-count',
        type: 'symbol',
        source: 'properties',
        filter: ['has', 'point_count'],
        layout: {
          'text-field': '{point_count_abbreviated}',
          'text-size': 12,
          'text-font': ['Open Sans Bold', 'Arial Unicode MS Bold'],
        },
        paint: { 'text-color': '#ffffff' },
      })

      // Individual property points
      map.addLayer({
        id: 'property-points',
        type: 'circle',
        source: 'properties',
        filter: ['!', ['has', 'point_count']],
        paint: {
          'circle-color': [
            'match', ['get', 'tier'],
            'hot',  TIER_COLORS.hot,
            'warm', TIER_COLORS.warm,
            /* default */ TIER_COLORS.cold,
          ],
          'circle-radius': 7,
          'circle-stroke-width': 1.5,
          'circle-stroke-color': '#ffffff',
        },
      })

      // Heatmap layer (hidden by default)
      map.addLayer({
        id: 'heatmap',
        type: 'heatmap',
        source: 'properties',
        maxzoom: 14,
        layout: { visibility: 'none' },
        paint: {
          'heatmap-weight': ['interpolate', ['linear'], ['get', 'score'], 0, 0, 100, 1],
          'heatmap-intensity': 1,
          'heatmap-color': [
            'interpolate', ['linear'], ['heatmap-density'],
            0,   'rgba(33,102,172,0)',
            0.2, 'rgb(103,169,207)',
            0.4, 'rgb(209,229,240)',
            0.6, 'rgb(253,219,199)',
            0.8, 'rgb(239,138,98)',
            1,   'rgb(178,24,43)',
          ],
          'heatmap-radius': 20,
        },
      })

      // Click handlers
      map.on('click', 'property-points', (e) => {
        const feature = e.features?.[0]
        if (feature?.properties?.id) {
          setSelectedProperty(feature.properties.id as number)
        }
      })

      map.on('click', 'clusters', (e) => {
        const features = map.queryRenderedFeatures(e.point, { layers: ['clusters'] })
        const clusterId = features[0]?.properties?.cluster_id
        const source = map.getSource('properties') as maplibregl.GeoJSONSource
        source.getClusterExpansionZoom(clusterId, (err, zoom) => {
          if (err || !zoom) return
          const coords = (features[0].geometry as GeoJSON.Point).coordinates as [number, number]
          map.easeTo({ center: coords, zoom })
        })
      })

      // Pointer cursor on hover
      map.on('mouseenter', 'property-points', () => {
        map.getCanvas().style.cursor = 'pointer'
      })
      map.on('mouseleave', 'property-points', () => {
        map.getCanvas().style.cursor = ''
      })

      // Track bounds for spatial queries
      map.on('moveend', () => {
        const bounds = map.getBounds()
        setMapBounds({
          north: bounds.getNorth(),
          south: bounds.getSouth(),
          east: bounds.getEast(),
          west: bounds.getWest(),
        })
      })
    })

    mapRef.current = map
    return () => {
      map.remove()
      mapRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Update GeoJSON data when query results change
  useEffect(() => {
    const map = mapRef.current
    if (!map || !map.isStyleLoaded()) return
    const source = map.getSource('properties') as maplibregl.GeoJSONSource | undefined
    if (source) {
      source.setData(geoJson ?? EMPTY_GEOJSON)
    }
  }, [geoJson])

  // Toggle heatmap visibility
  useEffect(() => {
    const map = mapRef.current
    if (!map || !map.isStyleLoaded()) return
    map.setLayoutProperty('heatmap', 'visibility', showHeatmap ? 'visible' : 'none')
    map.setLayoutProperty('property-points', 'visibility', showHeatmap ? 'none' : 'visible')
  }, [showHeatmap])

  return <div ref={mapContainer} className="h-full w-full" />
}
