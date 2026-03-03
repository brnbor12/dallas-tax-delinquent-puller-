import { useQuery, keepPreviousData } from '@tanstack/react-query'
import { fetchProperties, fetchPropertyDetail, fetchMapPoints } from '@/api/properties'
import { useFilterStore } from '@/stores/filterStore'

export function useProperties(page = 1) {
  const toQueryParams = useFilterStore((s) => s.toQueryParams)
  const params = toQueryParams()

  return useQuery({
    queryKey: ['properties', params, page],
    queryFn: () => fetchProperties(params, page),
    placeholderData: keepPreviousData,
    staleTime: 60_000,
  })
}

export function usePropertyDetail(id: number | null) {
  return useQuery({
    queryKey: ['property', id],
    queryFn: () => fetchPropertyDetail(id!),
    enabled: id !== null,
    staleTime: 120_000,
  })
}

export function useMapPoints() {
  const toQueryParams = useFilterStore((s) => s.toQueryParams)
  const params = toQueryParams()

  return useQuery({
    queryKey: ['mapPoints', params],
    queryFn: () => fetchMapPoints(params),
    staleTime: 60_000,
  })
}
