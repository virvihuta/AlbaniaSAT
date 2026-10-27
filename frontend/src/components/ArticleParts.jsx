import { formatDate } from '../api'

// Article image. Without an image we show stripes in the land cover colors.
export function Cover({ article, className = '' }) {
  if (article.image_url) {
    return <img src={article.image_url} alt="" className={`object-cover ${className}`} />
  }
  return (
    <div
      className={`bg-[linear-gradient(135deg,var(--color-lc-cropland)_0_35%,var(--color-lc-builtup)_35%_45%,var(--color-lc-grassland)_45%_60%,var(--color-lc-forest)_60%)] ${className}`}
    />
  )
}

// Round badge with initials, then author name and date
export function Author({ article }) {
  const name = article.author || 'AlbaniaSAT Team'
  const initials = name.split(' ').map((part) => part[0]).join('').slice(0, 2)

  return (
    <div className="flex items-center gap-3">
      <span className="flex h-9 w-9 items-center justify-center rounded-full bg-background text-xs font-semibold">
        {initials}
      </span>
      <div className="text-sm">
        <p className="font-semibold">{name}</p>
        <p className="text-muted">{formatDate(article.published_at)}</p>
      </div>
    </div>
  )
}