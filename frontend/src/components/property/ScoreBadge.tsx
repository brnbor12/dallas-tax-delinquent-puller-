import type { ScoreTier } from '@/types/property'
import { clsx } from 'clsx'

interface Props {
  score: number
  tier: ScoreTier | null
  size?: 'sm' | 'md' | 'lg'
}

const TIER_CLASSES: Record<string, string> = {
  hot:  'bg-red-100 text-red-700 ring-red-600/20',
  warm: 'bg-orange-100 text-orange-700 ring-orange-600/20',
  cold: 'bg-blue-100 text-blue-700 ring-blue-600/20',
}

const SIZE_CLASSES = {
  sm: 'text-xs px-1.5 py-0.5',
  md: 'text-sm px-2.5 py-1',
  lg: 'text-base px-3 py-1.5 font-semibold',
}

export function ScoreBadge({ score, tier, size = 'md' }: Props) {
  const tierKey = tier ?? 'cold'
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 rounded-full font-medium ring-1 ring-inset',
        TIER_CLASSES[tierKey],
        SIZE_CLASSES[size],
      )}
    >
      <span className={clsx(
        'h-2 w-2 rounded-full',
        tier === 'hot' ? 'bg-red-500' : tier === 'warm' ? 'bg-orange-500' : 'bg-blue-500'
      )} />
      {Math.round(score)}
    </span>
  )
}
