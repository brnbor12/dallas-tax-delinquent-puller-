import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Header } from '@/components/layout/Header'
import { MapPage } from '@/pages/MapPage'
import './index.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      staleTime: 60_000,
    },
  },
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <div className="flex h-screen flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-hidden">
          <MapPage />
        </main>
      </div>
    </QueryClientProvider>
  </React.StrictMode>,
)
