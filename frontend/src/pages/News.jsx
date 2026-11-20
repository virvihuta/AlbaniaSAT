import { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { getJSON } from '../api'
import PublicLayout from '../components/PublicLayout'
import Pagination from '../components/Pagination'
import YearFilter from '../components/YearFilter'
import { Cover, Author } from '../components/ArticleParts'
import { categories } from '../categories'
import { useLanguage } from '../i18n'

const PAGE_SIZE = 9

// Public news page ("Story"): 9 stories per page, filter by topic and year, search.
// Everything lives in the address, for example /news?topic=forests&year=2025&page=2,
// so links and the back button work.
export default function News() {
  const { t } = useLanguage()
  const [params, setParams] = useSearchParams()
  const page = Number(params.get('page')) || 1
  const q = params.get('q') || ''
  const year = params.get('year') || ''
  const topic = params.get('topic') || ''

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
    const next = { topic, q, year, page, ...changes }
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
    const filters = (year ? `&year=${year}` : '') + (topic ? `&category=${topic}` : '')
    getJSON(`/articles?q=${encodeURIComponent(q)}${filters}&page=${page}&page_size=${PAGE_SIZE}`)
      .then((result) => {
        setData(result)
        setError('')
      })
      .catch(() => setError('common.loadError'))
      .finally(() => setLoading(false))
  }, [q, year, topic, page])

  function goToPage(newPage) {
    update({ page: newPage })
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const totalPages = Math.ceil(data.total / PAGE_SIZE)
  // The newest story is big, but only on the first page without search
  const showFeatured = page === 1 && !q && !year && !topic && data.items.length > 0
  const [featured, ...rest] = showFeatured ? data.items : [null, ...data.items]
  const from = (page - 1) * PAGE_SIZE + 1
  const to = from + data.items.length - 1

  return (
    <PublicLayout>
      <p className="eyebrow">{t('nav.story')}</p>
      <h1 className="mt-3 text-5xl">{t('news.title')}</h1>
      <p className="mt-4 max-w-xl text-lg text-muted">{t('news.intro')}</p>

      <nav className="mt-10 flex gap-6 overflow-x-auto border-b border-border">
        {[{ code: '' }, ...categories].map((c) => (
          <button
            key={c.code || 'all'}
            onClick={() => update({ topic: c.code, page: 1 })}
            className={
              '-mb-px shrink-0 border-b-2 pb-3 text-sm font-medium ' +
              (topic === c.code ? 'border-primary text-foreground' : 'border-transparent text-muted hover:text-foreground')
            }
          >
            {c.code ? t('category.' + c.code) : t('news.allTopics')}
          </button>
        ))}
      </nav>

      <div className="mt-6 flex flex-wrap items-center justify-between gap-4">
        <input
          className="w-full max-w-md rounded-full border border-border bg-surface px-5 py-2.5 outline-none focus:border-primary"
          placeholder={t('news.search')}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <YearFilter years={years} value={year} onChange={(y) => update({ year: y, page: 1 })} />
      </div>

      <div className="mt-4">
        {data.total > 0 && (
          <p className="text-sm text-muted">
            {data.total === 1 ? t('news.showingOne') : t('news.showingMany', { from, to, total: data.total })}
          </p>
        )}
      </div>

      {error && <p className="mt-8 text-primary">{t(error)}</p>}
      {loading && <p className="mt-8 text-muted">{t('common.loading')}</p>}
      {!loading && !error && data.total === 0 && <p className="mt-8 text-muted">{t('news.none')}</p>}

      {featured && (
        <Link to={`/news/${featured.slug}`} className="group mt-10 grid overflow-hidden rounded-2xl border border-border bg-surface md:grid-cols-5">
          <Cover article={featured} className="h-64 w-full md:col-span-3 md:h-full md:min-h-96" />
          <div className="flex flex-col justify-center gap-6 p-8 md:col-span-2 md:p-10">
            {featured.category && <p className="eyebrow -mb-3 text-primary">{t('category.' + featured.category)}</p>}
            <h2 className="text-4xl group-hover:text-primary">{featured.title}</h2>
            {featured.summary && <p className="text-muted">{featured.summary}</p>}
            <Author article={featured} />
            <span className="font-semibold text-primary">{t('news.read')}</span>
          </div>
        </Link>
      )}

      <div className="mt-6 grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {rest.map((article) => (
          <Link key={article.id} to={`/news/${article.slug}`} className="group overflow-hidden rounded-2xl border border-border bg-surface">
            <Cover article={article} className="h-48 w-full" />
            <div className="space-y-3 p-6">
              {article.category && <p className="eyebrow text-primary">{t('category.' + article.category)}</p>}
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