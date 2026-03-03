import { clsx } from 'clsx'

const INDICATOR_LABELS: Record<string, string> = {
  pre_foreclosure:  'NOD',
  foreclosure:      'Foreclosure',
  tax_delinquent:   'Tax Delinquent',
  probate:          'Probate',
  lien:             'Lien',
  eviction:         'Eviction',
  code_violation:   'Code Violation',
  vacant:           'Vacant',
  absentee_owner:   'Absentee',
  price_reduction:  'Price Cut',
  expired_listing:  'Expired Listing',
  days_on_market:   'Extended DOM',
}

const INDICATOR_COLORS: Record<string, string> = {
  pre_foreclosure: 'bg-red-100 text-red-700',
  foreclosure:     'bg-red-200 text-red-800',
  tax_delinquent:  'bg-amber-100 text-amber-700',
  probate:         'bg-purple-100 text-purple-700',
  lien:            'bg-orange-100 text-orange-700',
  eviction:        'bg-yellow-100 text-yellow-700',
  code_violation:  'bg-pink-100 text-pink-700',
  vacant:          'bg-gray-100 text-gray-600',
  absentee_owner:  'bg-indigo-100 text-indigo-700',
  price_reduction: 'bg-green-100 text-green-700',
  expired_listing: 'bg-slate-100 text-slate-600',
  days_on_market:  'bg-teal-100 text-teal-700',
}

interface Props {
  type: string
}

export function IndicatorPill({ type }: Props) {
  return (
    <span
      className={clsx(
        'inline-block rounded px-1.5 py-0.5 text-xs font-medium',
        INDICATOR_COLORS[type] ?? 'bg-gray-100 text-gray-500',
      )}
    >
      {INDICATOR_LABELS[type] ?? type}
    </span>
  )
}
