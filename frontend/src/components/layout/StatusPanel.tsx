import { useEffect, useState } from 'react'
import { apiClient } from '@/api/client'

interface StatusData {
  totals: { properties: number; indicators: number; with_location: number; null_state: number; duplicate_addresses: number }
  by_state: { state: string; count: number }[]
  by_indicator: { type: string; count: number }[]
  by_tier: { tier: string; count: number }[]
  top_scores: { score: number; count: number }[]
  scrapers: { key: string; county: string; signal: string; freq: string; status: string; last_result: string }[]
}

const BADGE: Record<string, string> = {
  running: 'bg-green-100 text-green-800', pending: 'bg-yellow-100 text-yellow-800',
  broken: 'bg-red-100 text-red-800', disabled: 'bg-gray-100 text-gray-500',
}
const IND_COLOR: Record<string, string> = {
  tax_delinquent: '#1e40af', absentee_owner: '#1d4ed8', code_violation: '#2563eb',
  eviction: '#3b82f6', lien: '#60a5fa', pre_foreclosure: '#f59e0b',
  probate: '#ef4444', foreclosure: '#dc2626',
}
const fmt = (n: number) => n.toLocaleString()

export function StatusPanel({ onClose }: { onClose: () => void }) {
  const [data, setData] = useState<StatusData | null>(null)
  const [loading, setLoading] = useState(true)
  const [tab, setTab] = useState<'overview' | 'scrapers'>('overview')
  const [showAllStates, setShowAllStates] = useState(false)

  useEffect(() => {
    apiClient.get<StatusData>('/status').then(r => { setData(r.data); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  const maxInd = data ? Math.max(...data.by_indicator.map(i => i.count)) : 1
  const targetStates = ['TX', 'FL']
  const isTarget = (s: string) => targetStates.includes(s.toUpperCase())
  const allStates = data?.by_state.filter(s => s.state !== '(none)') || []
  const noneEntry = data?.by_state.find(s => s.state === '(none)')
  const maxState = Math.max(...allStates.map(s => s.count), 1)

  return (
    <div className="absolute inset-0 z-50 bg-white flex flex-col overflow-hidden">
      <div className="flex items-center justify-between border-b border-gray-200 px-5 py-3 flex-shrink-0">
        <div className="flex items-center gap-4">
          <span className="text-sm font-semibold text-gray-900">Pipeline Status</span>
          <div className="flex gap-1 bg-gray-100 rounded-lg p-0.5">
            {(['overview', 'scrapers'] as const).map(t => (
              <button key={t} onClick={() => setTab(t)}
                className={`px-3 py-1 text-xs rounded-md font-medium capitalize transition-colors ${tab === t ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}>
                {t}
              </button>
            ))}
          </div>
        </div>
        <button onClick={onClose} className="w-7 h-7 flex items-center justify-center rounded-full text-gray-400 hover:bg-gray-100 hover:text-gray-600 text-base">✕</button>
      </div>

      {loading && <div className="flex-1 flex items-center justify-center text-sm text-gray-400 gap-2"><span className="animate-spin">⟳</span> Loading…</div>}
      {!loading && !data && <div className="flex-1 flex items-center justify-center text-sm text-red-500">Failed to load — is the API reachable?</div>}

      {data && tab === 'overview' && (
        <div className="flex-1 overflow-y-auto p-5 space-y-6">
          <div className="grid grid-cols-5 gap-3">
            {[
              { label: 'Properties', val: fmt(data.totals.properties), sub: 'total records' },
              { label: 'Indicators', val: fmt(data.totals.indicators), sub: '8 signal types' },
              { label: 'With location', val: fmt(data.totals.with_location), sub: 'geocoded' },
              { label: 'Missing state', val: fmt(data.totals.null_state), sub: 'needs backfill', warn: true },
              { label: 'Dup addresses', val: fmt(data.totals.duplicate_addresses), sub: 'to dedupe', warn: data.totals.duplicate_addresses > 0 },
            ].map(({ label, val, sub, warn }) => (
              <div key={label} className={`rounded-lg p-3 ${warn ? 'bg-amber-50 border border-amber-200' : 'bg-gray-50'}`}>
                <div className="text-xs text-gray-400 mb-1">{label}</div>
                <div className={`text-xl font-semibold font-mono ${warn ? 'text-amber-600' : 'text-gray-900'}`}>{val}</div>
                <div className="text-xs text-gray-400 mt-0.5">{sub}</div>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-2 gap-8">
            <div>
              <div className="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-3">Distress signals</div>
              <div className="space-y-2.5">
                {data.by_indicator.map(({ type, count }) => (
                  <div key={type} className="flex items-center gap-3">
                    <div className="w-28 text-xs text-gray-500 text-right truncate">{type.replace(/_/g, ' ')}</div>
                    <div className="flex-1 bg-gray-100 rounded-full h-2 overflow-hidden">
                      <div className="h-full rounded-full transition-all" style={{ width: `${Math.max(2, (count / maxInd) * 100)}%`, background: IND_COLOR[type] || '#6b7280' }} />
                    </div>
                    <div className="w-16 text-xs font-mono text-right text-gray-500">{fmt(count)}</div>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <div className="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-3">Properties by state</div>
              <div className="space-y-2.5">
                {allStates.filter(s => showAllStates || isTarget(s.state)).map(({ state, count }) => (
                  <div key={state} className="flex items-center gap-3">
                    <div className={`w-8 text-xs font-mono text-right ${isTarget(state) ? 'text-blue-600 font-semibold' : 'text-gray-400'}`}>{state}</div>
                    <div className="flex-1 bg-gray-100 rounded-full h-2 overflow-hidden">
                      <div className="h-full rounded-full" style={{ width: `${Math.max(2, (count / maxState) * 100)}%`, background: isTarget(state) ? '#2563eb' : '#9ca3af' }} />
                    </div>
                    <div className="w-16 text-xs font-mono text-right text-gray-500">{fmt(count)}</div>
                  </div>
                ))}
                {allStates.filter(s => !isTarget(s.state)).length > 0 && (
                  <button onClick={() => setShowAllStates(!showAllStates)}
                    className="text-xs text-gray-400 hover:text-gray-600 transition-colors">
                    {showAllStates ? 'Hide other states' : `+ ${allStates.filter(s => !isTarget(s.state)).length} other states`}
                  </button>
                )}
                {noneEntry && (
                  <div className="flex items-center gap-3">
                    <div className="w-8 text-xs font-mono text-right text-amber-500">—</div>
                    <div className="flex-1 bg-amber-50 rounded-full h-2 overflow-hidden">
                      <div className="h-full rounded-full bg-amber-400" style={{ width: `${Math.max(2, (noneEntry.count / maxState) * 100)}%` }} />
                    </div>
                    <div className="w-16 text-xs font-mono text-right text-amber-600">{fmt(noneEntry.count)} ⚠</div>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div>
            <div className="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-3">Score tiers</div>
            <div className="flex gap-3 mb-3">
              {data.by_tier.map(({ tier, count }) => (
                <div key={tier} className={`flex-1 rounded-lg p-3 text-center border ${tier === 'hot' ? 'bg-red-50 border-red-100' : tier === 'warm' ? 'bg-orange-50 border-orange-100' : tier === 'cold' ? 'bg-blue-50 border-blue-100' : 'bg-gray-50 border-gray-100'}`}>
                  <div className={`text-xs capitalize mb-1 ${tier === 'hot' ? 'text-red-500' : tier === 'warm' ? 'text-orange-500' : tier === 'cold' ? 'text-blue-500' : 'text-gray-400'}`}>{tier}</div>
                  <div className="text-xl font-semibold font-mono text-gray-900">{fmt(count)}</div>
                </div>
              ))}
            </div>
            {data.top_scores.length > 0 && (
              <div>
                <div className="text-xs text-gray-400 mb-2">Top stacked scores</div>
                <div className="flex flex-wrap gap-1.5">
                  {data.top_scores.map(({ score, count }) => (
                    <span key={score} className="text-xs font-mono bg-red-50 text-red-700 border border-red-100 px-2 py-0.5 rounded">
                      {score.toFixed(2)}{count > 1 ? ` ×${count}` : ''}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div>
            <div className="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-3">Open issues</div>
            <div className="space-y-2">
              {[
                { sev: 'bug', msg: `${fmt(data.totals.null_state)} properties missing state — address_raw parser didn't extract state on import.` },
                { sev: 'bug', msg: `${fmt(data.totals.duplicate_addresses)} duplicate address groups in property table — causes ~10% ingest failures.` },
                { sev: 'fix', msg: 'Dallas probate hearing search returns 0 results — portal form fields not matching expected selectors.' },
                { sev: 'blocked', msg: 'Tyler Odyssey portals (probate, eviction, divorce) TCP-blocked from DigitalOcean IPs. Need proxy or CSV import.' },
                { sev: 'gap', msg: 'scrape_runs table never written — no run history, success/fail counts, or audit trail.' },
              ].map(({ sev, msg }, i) => (
                <div key={i} className="flex gap-2 items-start p-2.5 rounded-lg bg-gray-50">
                  <span className={`text-xs font-medium px-2 py-0.5 rounded flex-shrink-0 ${sev === 'bug' ? 'bg-red-100 text-red-700' : sev === 'fix' ? 'bg-yellow-100 text-yellow-700' : sev === 'blocked' ? 'bg-gray-200 text-gray-600' : 'bg-blue-50 text-blue-600'}`}>{sev}</span>
                  <span className="text-xs text-gray-600">{msg}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {data && tab === 'scrapers' && (
        <div className="flex-1 overflow-auto">
          <table className="w-full text-xs border-collapse">
            <thead className="sticky top-0 bg-gray-50 border-b border-gray-200">
              <tr>
                {['Scraper', 'County', 'Signal', 'Freq', 'Last result', 'Status'].map(h => (
                  <th key={h} className="text-left px-4 py-2.5 text-gray-400 font-semibold uppercase tracking-wide whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.scrapers.map(s => (
                <tr key={s.key} className={`border-b border-gray-100 hover:bg-gray-50 ${s.status === 'disabled' ? 'opacity-40' : ''}`}>
                  <td className="px-4 py-2.5 font-mono text-gray-700 text-xs">{s.key}</td>
                  <td className="px-4 py-2.5 text-gray-600 whitespace-nowrap">{s.county}</td>
                  <td className="px-4 py-2.5 text-gray-400">{s.signal}</td>
                  <td className="px-4 py-2.5"><span className="bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded">{s.freq}</span></td>
                  <td className="px-4 py-2.5 font-mono text-gray-400 max-w-xs truncate">{s.last_result}</td>
                  <td className="px-4 py-2.5"><span className={`px-2 py-0.5 rounded text-xs font-medium ${BADGE[s.status] || 'bg-gray-100 text-gray-500'}`}>{s.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
