import { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { getJSON } from '../api'
import PublicLayout from '../components/PublicLayout'
import Pagination from '../components/Pagination'
import { Cover, Author } from '../components/ArticleParts'

const PAGE_SIZE = 9

// Public news page ("Story"): 9 stories per page, filter by year, search.
// Everything lives in the address, for example /news?year=2025&page=2&q=forest,
// so links and the back button work.
export default function News() {
  const [params, setParams] = useSearchParams()
  const page = Number(params.get('page')) || 1
  const q = params.get('q') || ''
  const year = params.get('year') || ''

  const [search, setSearch] = useState(q)
  const [data, setData] = useState({ items: [], total: 0 })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [years, setYears] = useState([])

  // Years that have stories, for the year buttons. Loaded once.
  useEffect(() => {
    getJSON('/articles/years').then(setYears).catch(() => setYears([]))
  }, [])

  // Builds the address: only the filters that are set, page 1 is left out
  function update(changes) {
    const next = { q, year, page, ...changes }
    setParams(Object.fromEntries(Object.entries(next).filter(([key, value]) => value && !(key === 'page' && value === 1))))
  }

  // Write the search into the address 300 ms after typing and start again at page 1
  useEffect(() => {
    const timer = setTimeout(() => {
      if (search !== q) update({ q: search, page: 1 })
    }, 300)
    return () => clearTimeout(timer)
  })

  // Load one page whenever page or search changes
  useEffect(() => {
    const yearFilter = year ? `&year=${year}` : ''
    getJSON(`/articles?q=${encodeURIComponent(q)}${yearFilter}&page=${page}&page_size=${PAGE_SIZE}`)
      .then((result) => {
        setData(result)
        setError('')
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [q, year, page])

  function goToPage(newPage) {
    update({ page: newPage })
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const totalPages = Math.ceil(data.total / PAGE_SIZE)
  // The newest story is big, but only on the first page without search
  const showFeatured = page === 1 && !q && !year && data.items.length > 0
  const [featured, ...rest] = showFeatured ? data.items : [null, ...data.items]
  const from = (page - 1) * PAGE_SIZE + 1
  const to = from + data.items.length - 1

  return (
    <PublicLayout>
      <p className="eyebrow">Story</p>
      <h1 className="mt-3 text-5xl">What the maps show</h1>
      <p className="mt-4 max-w-xl text-lg text-muted">
        Short reads on Albania's changing land, and on how AlbaniaSAT was built, written by the team.
      </p>

      <div className="mt-8 flex flex-wrap items-center justify-between gap-4">
        <input
          className="w-full max-w-md rounded-full border border-border bg-surface px-5 py-2.5 outline-none focus:border-primary"
          placeholder="Search stories..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        {years.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {[{ year: '', count: null }, ...years].map((y) => (
              <button
                key={y.year || 'all'}
                onClick={() => update({ year: String(y.year), page: 1 })}
                className={
                  'rounded-full border px-4 py-1.5 text-sm font-medium ' +
                  (String(y.year) === year ? 'border-primary bg-primary text-white' : 'border-border bg-surface hover:border-primary')
                }
              >
                {y.year || 'All'}
                {y.count !== null && <span className="ml-1 opacity-70">({y.count})</span>}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="mt-4">
        {data.total > 0 && (
          <p className="text-sm text-muted">
            Showing {from}–{to} of {data.total} {data.total === 1 ? 'story' : 'stories'}
          </p>
        )}
      </div>

      {error && <p className="mt-8 text-primary">{error}</p>}
      {loading && <p className="mt-8 text-muted">Loading...</p>}
      {!loading && !error && data.total === 0 && <p className="mt-8 text-muted">No stories found.</p>}

      {featured && (
        <Link to={`/news/${featured.slug}`} className="group mt-10 grid overflow-hidden rounded-2xl border border-border bg-surface md:grid-cols-5">
          <Cover article={featured} className="h-64 w-full md:col-span-3 md:h-full md:min-h-96" />
          <div className="flex flex-col justify-center gap-6 p-8 md:col-span-2 md:p-10">
            <h2 className="text-4xl group-hover:text-primary">{featured.title}</h2>
            {featured.summary && <p className="text-muted">{featured.summary}</p>}
            <Author article={featured} />
            <span className="font-semibold text-primary">Read the story →</span>
          </div>
        </Link>
      )}

      <div className="mt-6 grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {rest.map((article) => (
          <Link key={article.id} to={`/news/${article.slug}`} className="group overflow-hidden rounded-2xl border border-border bg-surface">
            <Cover article={article} className="h-48 w-full" />
            <div className="space-y-3 p-6">
              <h2 className="text-2xl group-hover:text-primary">{article.title}</h2>
              {article.summary && <p className="line-clamp-3 text-sm text-muted">{article.summary}</p>}
              <Author article={article} />
            </div>
          </Link>
        ))}
      </div>

      <Pagination page={page} totalPages={totalPages} onChange={goToPage} />
    </PublicLayout>
  )
}