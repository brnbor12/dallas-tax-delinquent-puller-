import { useMapStore } from '@/stores/mapStore'
import { requestExport } from '@/api/properties'
import { useFilterStore } from '@/stores/filterStore'
import { useState } from 'react'

export function Header() {
  const { showHeatmap, toggleHeatmap } = useMapStore()
  const toQueryParams = useFilterStore((s) => s.toQueryParams)
  const [exporting, setExporting] = useState(false)

  async function handleExport() {
    setExporting(true)
    try {
      const { export_id } = await requestExport(toQueryParams())
      // Poll every 2 seconds for the export to be ready
      let attempts = 0
      const poll = setInterval(async () => {
        attempts++
        try {
          const resp = await fetch(`/api/v1/export/${export_id}`)
          if (resp.ok) {
            clearInterval(poll)
            const blob = await resp.blob()
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = `motivated_sellers_${export_id.slice(0, 8)}.csv`
            a.click()
            URL.revokeObjectURL(url)
            setExporting(false)
          }
        } catch {/* still processing */}
        if (attempts > 30) {
          clearInterval(poll)
          setExporting(false)
        }
      }, 2000)
    } catch {
      setExporting(false)
    }
  }

  return (
    <header className="flex items-center justify-between border-b border-gray-200 bg-white px-4 py-2.5 shadow-sm">
      <div className="flex items-center gap-3">
        <span className="text-base font-bold text-gray-900">Motivated Seller Finder</span>
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={toggleHeatmap}
          className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
            showHeatmap
              ? 'bg-blue-600 text-white'
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
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
  )
}
