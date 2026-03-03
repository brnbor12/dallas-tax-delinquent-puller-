import { usePropertyDetail } from '@/hooks/useProperties'
import { useMapStore } from '@/stores/mapStore'
import { ScoreBadge } from './ScoreBadge'
import { IndicatorPill } from './IndicatorPill'

function formatMoney(cents: number | null | undefined): string {
  if (!cents) return '—'
  return `$${(cents / 100).toLocaleString()}`
}

function formatDate(d: string | null | undefined): string {
  if (!d) return '—'
  return new Date(d).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })
}

export function PropertyDetail() {
  const selectedId = useMapStore((s) => s.selectedPropertyId)
  const setSelectedProperty = useMapStore((s) => s.setSelectedProperty)
  const { data: property, isLoading } = usePropertyDetail(selectedId)

  if (!selectedId) return null

  return (
    <div className="absolute inset-y-0 right-0 z-10 w-96 flex flex-col bg-white shadow-xl border-l border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-gray-200 px-4 py-3">
        <h2 className="font-semibold text-gray-900 text-sm">Property Details</h2>
        <button
          onClick={() => setSelectedProperty(null)}
          className="rounded p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-600"
          aria-label="Close"
        >
          ✕
        </button>
      </div>

      {isLoading ? (
        <div className="flex flex-1 items-center justify-center text-gray-400">Loading…</div>
      ) : !property ? (
        <div className="flex flex-1 items-center justify-center text-red-400">Failed to load</div>
      ) : (
        <div className="flex-1 overflow-y-auto p-4 space-y-5">
          {/* Address + Score */}
          <div>
            <p className="font-semibold text-gray-900">
              {property.address_line1 || property.address_raw}
            </p>
            <p className="text-sm text-gray-500">
              {[property.address_city, property.address_state, property.address_zip].filter(Boolean).join(', ')}
            </p>
            {property.score && (
              <div className="mt-2 flex items-center gap-2">
                <ScoreBadge score={property.score.total_score} tier={property.score.score_tier} size="lg" />
                <span className="text-sm text-gray-500">Motivated Seller Score</span>
              </div>
            )}
          </div>

          {/* Active Indicators */}
          {property.indicators.filter((i) => i.status === 'active').length > 0 && (
            <section>
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
                Active Signals
              </h3>
              <div className="flex flex-wrap gap-1.5">
                {property.indicators
                  .filter((i) => i.status === 'active')
                  .map((i) => (
                    <div key={i.id} className="flex flex-col gap-0.5">
                      <IndicatorPill type={i.indicator_type} />
                      {i.filing_date && (
                        <span className="text-xs text-gray-400 pl-1">
                          Filed {formatDate(i.filing_date)}
                        </span>
                      )}
                      {i.amount_cents && (
                        <span className="text-xs text-gray-400 pl-1">
                          {formatMoney(i.amount_cents)}
                        </span>
                      )}
                    </div>
                  ))}
              </div>
            </section>
          )}

          {/* Property Details */}
          <section>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
              Property
            </h3>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-sm">
              {property.property_type && (
                <><dt className="text-gray-500">Type</dt><dd>{property.property_type}</dd></>
              )}
              {property.year_built && (
                <><dt className="text-gray-500">Built</dt><dd>{property.year_built}</dd></>
              )}
              {property.bedrooms && (
                <><dt className="text-gray-500">Beds</dt><dd>{property.bedrooms}</dd></>
              )}
              {property.bathrooms && (
                <><dt className="text-gray-500">Baths</dt><dd>{property.bathrooms}</dd></>
              )}
              {property.sqft && (
                <><dt className="text-gray-500">Sq Ft</dt><dd>{property.sqft.toLocaleString()}</dd></>
              )}
              {property.lot_size_sqft && (
                <><dt className="text-gray-500">Lot</dt><dd>{property.lot_size_sqft.toLocaleString()} sqft</dd></>
              )}
              {property.zoning && (
                <><dt className="text-gray-500">Zoning</dt><dd>{property.zoning}</dd></>
              )}
              {property.apn && (
                <><dt className="text-gray-500">APN</dt><dd className="font-mono text-xs">{property.apn}</dd></>
              )}
            </dl>
          </section>

          {/* Financials */}
          <section>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
              Financials
            </h3>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-sm">
              <dt className="text-gray-500">Assessed</dt>
              <dd>{formatMoney(property.assessed_value)}</dd>
              <dt className="text-gray-500">Market Est.</dt>
              <dd>{formatMoney(property.market_value)}</dd>
              <dt className="text-gray-500">Last Sale</dt>
              <dd>{formatMoney(property.last_sale_price)}</dd>
              <dt className="text-gray-500">Sale Date</dt>
              <dd>{formatDate(property.last_sale_date)}</dd>
            </dl>
          </section>

          {/* Owner */}
          {property.owners.length > 0 && (
            <section>
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
                Owner
              </h3>
              {property.owners.map((owner, i) => (
                <div key={i} className="text-sm space-y-0.5">
                  <p className="font-medium text-gray-900">{owner.name_raw}</p>
                  {owner.mailing_address && (
                    <p className="text-gray-500">
                      {owner.mailing_address}
                      {owner.mailing_city && `, ${owner.mailing_city}`}
                      {owner.mailing_state && `, ${owner.mailing_state}`}
                      {owner.mailing_zip && ` ${owner.mailing_zip}`}
                    </p>
                  )}
                  {owner.is_out_of_state && (
                    <span className="inline-block rounded bg-indigo-100 px-1.5 py-0.5 text-xs text-indigo-700">
                      Out-of-state owner
                    </span>
                  )}
                  {owner.is_absentee && !owner.is_out_of_state && (
                    <span className="inline-block rounded bg-yellow-100 px-1.5 py-0.5 text-xs text-yellow-700">
                      Absentee owner
                    </span>
                  )}
                </div>
              ))}
            </section>
          )}

          {/* MLS Listing */}
          {property.listings.length > 0 && (
            <section>
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
                Listing
              </h3>
              {property.listings.slice(0, 1).map((listing, i) => (
                <dl key={i} className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-sm">
                  <dt className="text-gray-500">List Price</dt>
                  <dd>{formatMoney(listing.list_price)}</dd>
                  <dt className="text-gray-500">Original</dt>
                  <dd>{formatMoney(listing.original_price)}</dd>
                  <dt className="text-gray-500">Days on Market</dt>
                  <dd>{listing.days_on_market ?? '—'}</dd>
                  <dt className="text-gray-500">Price Cuts</dt>
                  <dd>{listing.price_reductions}</dd>
                  <dt className="text-gray-500">Status</dt>
                  <dd className="capitalize">{listing.listing_status ?? '—'}</dd>
                  <dt className="text-gray-500">Listed</dt>
                  <dd>{formatDate(listing.listed_date)}</dd>
                </dl>
              ))}
            </section>
          )}

          {/* Indicator History */}
          <section>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
              Signal History
            </h3>
            <div className="space-y-1">
              {property.indicators.map((ind) => (
                <div key={ind.id} className="flex items-center gap-2 text-xs">
                  <span
                    className={`h-1.5 w-1.5 flex-shrink-0 rounded-full ${
                      ind.status === 'active' ? 'bg-green-500' : 'bg-gray-300'
                    }`}
                  />
                  <span className="text-gray-400">{formatDate(ind.filing_date)}</span>
                  <IndicatorPill type={ind.indicator_type} />
                  <span className="capitalize text-gray-400">[{ind.status}]</span>
                </div>
              ))}
            </div>
          </section>
        </div>
      )}
    </div>
  )
}
