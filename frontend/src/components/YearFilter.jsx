// Round buttons to filter by year: All, 2026 (12), 2025 (5) ...
// years comes from the API: [{ year: 2026, count: 12 }, ...]. value is the chosen year ('' = all).
export default function YearFilter({ years, value, onChange }) {
  if (years.length === 0) return null

  return (
    <div className="flex flex-wrap gap-2">
      {[{ year: '', count: null }, ...years].map((y) => (
        <button
          key={y.year || 'all'}
          onClick={() => onChange(String(y.year))}
          className={
            'rounded-full border px-4 py-1.5 text-sm font-medium ' +
            (String(y.year) === value ? 'border-primary bg-primary text-white' : 'border-border bg-surface hover:border-primary')
          }
        >
          {y.year || 'All'}
          {y.count !== null && <span className="ml-1 opacity-70">({y.count})</span>}
        </button>
      ))}
    </div>
  )
}