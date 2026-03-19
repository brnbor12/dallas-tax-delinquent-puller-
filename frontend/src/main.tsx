import React, { Suspense, useState } from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Header } from '@/components/layout/Header'
import './index.css'

const MapPage = React.lazy(() => import('@/pages/MapPage').then(m => ({ default: m.MapPage })))
const MarketPage = React.lazy(() => import('@/pages/MarketPage').then(m => ({ default: m.MarketPage })))

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, staleTime: 5 * 60_000, gcTime: 10 * 60_000 } },
})

function App() {
  const [tab, setTab] = useState<'map' | 'market'>('map')
  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <Header />
      {/* Tab bar */}
      <div className="flex border-b border-gray-200 bg-white px-4">
        <button
          onClick={() => setTab('map')}
          className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
            tab === 'map'
              ? 'border-gray-900 text-gray-900'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          🗺️ Map
        </button>
        <button
          onClick={() => setTab('market')}
          className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
            tab === 'market'
              ? 'border-gray-900 text-gray-900'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          📊 Markets
        </button>
      </div>
      <main className="flex-1 overflow-hidden">
        <Suspense fallback={<div className="flex h-full items-center justify-center text-sm text-gray-400">Loading...</div>}>
          {tab === 'map' ? <MapPage /> : <MarketPage />}
        </Suspense>
      </main>
    </div>
  )
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>,
)
