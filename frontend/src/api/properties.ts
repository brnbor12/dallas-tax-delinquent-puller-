import { apiClient } from './client'
import type {
  PropertyListResponse,
  PropertyDetail,
  GeoJsonFeatureCollection,
} from '@/types/property'

export async function fetchProperties(
  params: Record<string, string>,
  page = 1,
  pageSize = 50,
): Promise<PropertyListResponse> {
  const { data } = await apiClient.get<PropertyListResponse>('/properties', {
    params: { ...params, page, page_size: pageSize },
  })
  return data
}

export async function fetchPropertyDetail(id: number): Promise<PropertyDetail> {
  const { data } = await apiClient.get<PropertyDetail>(`/properties/${id}`)
  return data
}

export async function fetchMapPoints(
  params: Record<string, string>,
): Promise<GeoJsonFeatureCollection> {
  const { data } = await apiClient.get<GeoJsonFeatureCollection>('/properties/map/points', {
    params,
  })
  return data
}

export async function requestExport(params: Record<string, string>): Promise<{ export_id: string }> {
  const { data } = await apiClient.post<{ export_id: string }>('/export', null, { params })
  return data
}
