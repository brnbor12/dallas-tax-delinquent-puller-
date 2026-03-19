import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/api/client'

interface ZipStats {
  zip: string
  total: number
  hot: number
  warm: number
  cold: number
  avg_score: number
  tax_delinquent: number
  foreclosure: number
  pre_foreclosure: number
  probate: number
  code_violation: number
  absentee: number
  eviction: number
  lien: number
}

interface CountyStats {
  fips: string
  total: number
  hot: number
  warm: number
  cold: number
  zips: ZipStats[]
}

type MarketData = Record<string, Record<string, CountyStats>>

const SIGNAL_COLORS: Record<string, string> = {
  tax_delinquent: 'bg-orange-100 text-orange-800',
  foreclosure: 'bg-red-100 text-red-800',
  pre_foreclosure: 'bg-red-100 text-red-700',
  probate: 'bg-purple-100 text-purple-800',
  code_violation: 'bg-yellow-100 text-yellow-800',
  absentee: 'bg-blue-100 text-blue-800',
  eviction: 'bg-pink-100 text-pink-800',
  lien: 'bg-gray-100 text-gray-700',
}

function TierBadge({ count, tier }: { count: number; tier: 'hot' | 'warm' | 'cold' }) {
  if (!count) return null
  const colors = { hot: 'bg-red-500 text-white', warm: 'bg-orange-400 text-white', cold: 'bg-gray-200 text-gray-600' }
  return (
    <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold ${colors[tier]}`}>
      {tier === 'hot' ? '🔥' : tier === 'warm' ? '🌡️' : ''} {count.toLocaleString()}
    </span>
  )
}

function SignalPill({ label, count }: { label: string; count: number }) {
  if (!count) return null
  return (
    <span className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-xs font-medium ${SIGNAL_COLORS[label] || 'bg-gray-100 text-gray-600'}`}>
      {label.replace(/_/g, ' ')}: {count.toLocaleString()}
    </span>
  )
}

function ZipRow({ zip, onExport }: { zip: ZipStats; onExport: (zip: string, tier: string) => void }) {
  const [expanded, setExpanded] = useState(false)
  const hotRate = zip.total ? Math.round((zip.hot / zip.total) * 100) : 0

  return (
    <div className="border-b border-gray-100 last:border-0">
      <div
        className="flex cursor-pointer items-center gap-3 px-4 py-2 hover:bg-gray-50"
        onClick={() => setExpanded(!expanded)}
      >
        <span className="w-16 font-mono text-sm font-semibold text-gray-900">{zip.zip}</span>
        <div className="flex flex-1 items-center gap-2">
          <TierBadge count={zip.hot} tier="hot" />
          <TierBadge count={zip.warm} tier="warm" />
          <span className="text-xs text-gray-400">{zip.total.toLocaleString()} total</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-1.5 w-24 rounded-full bg-gray-200">
            <div className="h-1.5 rounded-full bg-red-500" style={{ width: `${Math.min(hotRate * 3, 100)}%` }} />
          </div>
          <span className="w-8 text-right text-xs text-gray-500">{zip.avg_score}</span>
        </div>
        <div className="flex gap-1">
          <button
            onClick={(e) => { e.stopPropagation(); onExport(zip.zip, 'hot') }}
            className="rounded px-2 py-0.5 text-xs bg-red-50 text-red-600 hover:bg-red-100"
          >
            Hot CSV
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); onExport(zip.zip, 'hot,warm') }}
            className="rounded px-2 py-0.5 text-xs bg-orange-50 text-orange-600 hover:bg-orange-100"
          >
            All CSV
          </button>
        </div>
        <span className="text-gray-400">{expanded ? '▲' : '▼'}</span>
      </div>
      {expanded && (
        <div className="flex flex-wrap gap-1.5 bg-gray-50 px-4 py-2">
          <SignalPill label="tax_delinquent" count={zip.tax_delinquent} />
          <SignalPill label="foreclosure" count={zip.foreclosure} />
          <SignalPill label="pre_foreclosure" count={zip.pre_foreclosure} />
          <SignalPill label="probate" count={zip.probate} />
          <SignalPill label="code_violation" count={zip.code_violation} />
          <SignalPill label="absentee" count={zip.absentee} />
          <SignalPill label="eviction" count={zip.eviction} />
          <SignalPill label="lien" count={zip.lien} />
        </div>
      )}
    </div>
  )
}

export function MarketPage() {
  const { data, isLoading: loading } = useQuery<MarketData>({
    queryKey: ['marketStats'],
    queryFn: () => apiClient.get<MarketData>('/market/stats').then(r => r.data),
  })
  const [expandedStates, setExpandedStates] = useState<Record<string, boolean>>({})
  const [expandedCounties, setExpandedCounties] = useState<Record<string, boolean>>({})
  const [sortBy, setSortBy] = useState<'hot' | 'warm' | 'zip'>('hot')
  const [filterZip, setFilterZip] = useState('')

  const handleExport = async (zip: string, tiers: string) => {
    const tierParam = tiers.includes(',') ? 'hot' : tiers
    // Export hot+warm for a specific zip
    const query = new URLSearchParams({ zip_codes: zip, score_tier: tierParam }).toString()
    const resp = await fetch(`/api/v1/export?${query}`, { method: 'POST' })
    if (!resp.ok) return
    const blob = await resp.blob()
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `leads_${zip}_${tierParam}.csv`
    a.click()
    URL.revokeObjectURL(a.href)
  }

  const toggleState = (state: string) => setExpandedStates(s => ({ ...s, [state]: !s[state] }))
  const toggleCounty = (key: string) => setExpandedCounties(s => ({ ...s, [key]: !s[key] }))

  if (loading) return (
    <div className="flex h-full items-center justify-center">
      <div className="text-center">
        <div className="mb-2 text-2xl">📊</div>
        <div className="text-sm text-gray-500">Loading market data...</div>
      </div>
    </div>
  )

  if (!data) return <div className="p-8 text-gray-500">Failed to load market data</div>

  // Sort and filter zips
  const sortZips = (zips: ZipStats[]) => {
    let filtered = filterZip ? zips.filter(z => z.zip.includes(filterZip)) : zips
    return [...filtered].sort((a, b) => {
      if (sortBy === 'zip') return a.zip.localeCompare(b.zip)
      return (b[sortBy] || 0) - (a[sortBy] || 0)
    })
  }

  // Global totals
  let totalHot = 0, totalWarm = 0, totalProps = 0
  Object.values(data).forEach(counties =>
    Object.values(counties).forEach(c => {
      totalHot += c.hot; totalWarm += c.warm; totalProps += c.total
    })
  )

  return (
    <div className="flex h-full flex-col overflow-hidden bg-gray-50">
      {/* Top bar */}
      <div className="flex items-center gap-4 border-b border-gray-200 bg-white px-4 py-2.5">
        <div className="flex items-center gap-6">
          <div className="text-center">
            <div className="text-lg font-bold text-red-600">{totalHot.toLocaleString()}</div>
            <div className="text-xs text-gray-500">Hot Leads</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-orange-500">{totalWarm.toLocaleString()}</div>
            <div className="text-xs text-gray-500">Warm Leads</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-gray-600">{totalProps.toLocaleString()}</div>
            <div className="text-xs text-gray-500">Total Properties</div>
          </div>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <input
            type="text"
            placeholder="Filter by zip..."
            value={filterZip}
            onChange={e => setFilterZip(e.target.value)}
            className="rounded border border-gray-200 px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-blue-400"
          />
          <span className="text-xs text-gray-500">Sort:</span>
          {(['hot', 'warm', 'zip'] as const).map(s => (
            <button key={s} onClick={() => setSortBy(s)}
              className={`rounded px-2 py-1 text-xs font-medium transition-colors ${sortBy === s ? 'bg-gray-800 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}>
              {s}
            </button>
          ))}
        </div>
      </div>

      {/* Market tree */}
      <div className="flex-1 overflow-y-auto">
        {Object.entries(data).sort().map(([state, counties]) => {
          const stateHot = Object.values(counties).reduce((a, c) => a + c.hot, 0)
          const stateWarm = Object.values(counties).reduce((a, c) => a + c.warm, 0)
          const isStateOpen = expandedStates[state] !== false // default open

          return (
            <div key={state} className="border-b border-gray-200 bg-white">
              {/* State header */}
              <div
                className="flex cursor-pointer items-center gap-3 bg-gray-800 px-4 py-2 text-white hover:bg-gray-700"
                onClick={() => toggleState(state)}
              >
                <span className="text-sm font-bold">{state}</span>
                <TierBadge count={stateHot} tier="hot" />
                <TierBadge count={stateWarm} tier="warm" />
                <span className="ml-auto text-gray-400">{isStateOpen ? '▲' : '▼'}</span>
              </div>

              {isStateOpen && Object.entries(counties).sort().map(([county, stats]) => {
                const countyKey = `${state}-${county}`
                const isCountyOpen = expandedCounties[countyKey] !== false // default open

                return (
                  <div key={county} className="border-t border-gray-100">
                    {/* County header */}
                    <div
                      className="flex cursor-pointer items-center gap-3 bg-gray-50 px-4 py-2 hover:bg-gray-100"
                      onClick={() => toggleCounty(countyKey)}
                    >
                      <span className="text-sm font-semibold text-gray-700">{county}</span>
                      <TierBadge count={stats.hot} tier="hot" />
                      <TierBadge count={stats.warm} tier="warm" />
                      <span className="ml-1 text-xs text-gray-400">{stats.zips.length} zips · {stats.total.toLocaleString()} props</span>
                      <span className="ml-auto text-gray-400">{isCountyOpen ? '▲' : '▼'}</span>
                    </div>

                    {/* Zip rows */}
                    {isCountyOpen && (
                      <div>
                        {/* Column headers */}
                        <div className="flex items-center gap-3 border-b border-gray-100 bg-white px-4 py-1 text-xs font-medium text-gray-400">
                          <span className="w-16">ZIP</span>
                          <span className="flex-1">Tiers</span>
                          <span className="w-32 text-right">Hot Rate / Avg Score</span>
                          <span className="w-28 text-right">Export</span>
                          <span className="w-4" />
                        </div>
                        {sortZips(stats.zips).map(zip => (
                          <ZipRow key={zip.zip} zip={zip} onExport={handleExport} />
                        ))}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )
        })}
      </div>
    </div>
  )
}
