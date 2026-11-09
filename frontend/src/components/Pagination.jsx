// Page numbers to show, for example 1 … 4 5 6 … 12
function pageList(page, totalPages) {
  const pages = []
  for (let p = 1; p <= totalPages; p++) {
    if (p === 1 || p === totalPages || Math.abs(p - page) <= 1) pages.push(p)
    else if (pages[pages.length - 1] !== '…') pages.push('…')
  }
  return pages
}

// Buttons to move between pages: Previous, page numbers, Next
export default function Pagination({ page, totalPages, onChange }) {
  if (totalPages <= 1) return null

  const button = 'flex h-10 min-w-10 items-center justify-center rounded-lg px-3 text-sm font-medium '

  return (
    <nav aria-label="Pages" className="mt-12 flex flex-wrap items-center justify-center gap-1">
      <button
        onClick={() => onChange(page - 1)}
        disabled={page === 1}
        className={button + 'text-muted hover:text-foreground disabled:opacity-30'}
      >
        ← Previous
      </button>

      {pageList(page, totalPages).map((p, i) =>
        p === '…' ? (
          <span key={'gap' + i} className="px-2 text-muted">…</span>
        ) : (
          <button
            key={p}
            onClick={() => onChange(p)}
            aria-current={p === page ? 'page' : undefined}
            className={button + (p === page ? 'bg-primary text-white' : 'hover:bg-surface')}
          >
            {p}
          </button>
        ),
      )}

      <button
        onClick={() => onChange(page + 1)}
        disabled={page === totalPages}
        className={button + 'text-muted hover:text-foreground disabled:opacity-30'}
      >
        Next →
      </button>
    </nav>
  )
}