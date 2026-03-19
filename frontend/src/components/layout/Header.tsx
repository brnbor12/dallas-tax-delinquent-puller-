import { useMapStore } from '@/stores/mapStore'
import { useFilterStore } from '@/stores/filterStore'
import { useState } from 'react'
import { StatusPanel } from '@/components/layout/StatusPanel'

export function Header() {
  const { showHeatmap, toggleHeatmap } = useMapStore()
  const toQueryParams = useFilterStore((s) => s.toQueryParams)
  const [exporting, setExporting] = useState(false)
  const [showStatus, setShowStatus] = useState(false)

  async function handleExport() {
    setExporting(true)
    try {
      const params = toQueryParams()
      const query = new URLSearchParams(params).toString()
      const resp = await fetch(`/api/v1/export?${query}`, { method: 'POST' })
      if (!resp.ok) throw new Error('Export failed')
      const blob = await resp.blob()
      const a = document.createElement('a')
      a.href = URL.createObjectURL(blob)
      a.download = 'motivated_sellers.csv'
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(a.href)
    } catch (e) {
      console.error('Export error:', e)
    } finally {
      setExporting(false)
    }
  }

  return (
    <>
      <header className="flex items-center justify-between border-b border-gray-200 bg-white px-4 py-2.5 shadow-sm">
        <div className="flex items-center gap-3">
          <span className="text-base font-bold text-gray-900">Motivated Seller Finder</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowStatus(true)}
            className="rounded-md px-3 py-1.5 text-sm font-medium bg-gray-100 text-gray-600 hover:bg-gray-200 transition-colors"
          >
            Pipeline
          </button>
          <button
            onClick={toggleHeatmap}
            className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
              showHeatmap ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            Heatmap
          </button>
          <button
            onClick={handleExport}
            disabled={exporting}
            className="rounded-md bg-green-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-green-700 disabled:opacity-50"
          >
            {exporting ? 'Exporting…' : 'Export CSV'}
          </button>
        </div>
      </header>
      {showStatus && <StatusPanel onClose={() => setShowStatus(false)} />}
    </>
  )
}
