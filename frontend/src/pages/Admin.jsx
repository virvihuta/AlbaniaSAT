import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { adminRequest, formatDate } from '../api'
import AdminLayout from '../components/AdminLayout'
import Pagination from '../components/Pagination'
import { categoryName } from '../categories'

const PAGE_SIZE = 20

const tabs = [
  { status: '', label: 'All', count: 'all' },
  { status: 'published', label: 'Published', count: 'published' },
  { status: 'draft', label: 'Drafts', count: 'draft' },
]

// Start page of the private area: articles with tabs (all, published, drafts), search and pages
export default function Admin() {
  const [data, setData] = useState({ items: [], total: 0, counts: {} })
  const [status, setStatus] = useState('')
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const [error, setError] = useState('')

  const load = useCallback(() => {
    const statusFilter = status ? `&status=${status}` : ''
    adminRequest('GET', `/admin/articles?q=${encodeURIComponent(search)}${statusFilter}&page=${page}&page_size=${PAGE_SIZE}`)
      .then(setData)
      .catch((err) => setError(err.message))
  }, [search, status, page])

  // Load again 300 ms after typing, or when the tab or page changes
  useEffect(() => {
    const timer = setTimeout(load, 300)
    return () => clearTimeout(timer)
  }, [load])

  async function handleDelete(article) {
    if (!window.confirm(`Delete "${article.title}"?`)) return
    try {
      await adminRequest('DELETE', '/admin/articles/' + article.id)
      // If this was the last article on the page, go one page back
      if (data.items.length === 1 && page > 1) setPage(page - 1)
      else load()
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <AdminLayout>
      <div className="flex flex-wrap items-center justify-between gap-4">
        <h1 className="text-3xl">Articles</h1>
        <Link to="/admin/articles/new" className="rounded-lg bg-primary px-5 py-2 font-semibold text-white hover:opacity-90">
          New article
        </Link>
      </div>

      <div className="mt-8 flex flex-wrap items-end justify-between gap-4 border-b border-border">
        <nav className="flex gap-6">
          {tabs.map((tab) => (
            <button
              key={tab.label}
              onClick={() => {
                setStatus(tab.status)
                setPage(1)
              }}
              className={
                '-mb-px border-b-2 pb-3 text-sm font-medium ' +
                (status === tab.status ? 'border-primary text-foreground' : 'border-transparent text-muted hover:text-foreground')
              }
            >
              {tab.label}
              <span className="ml-2 rounded-full bg-background px-2 py-0.5 text-xs">{data.counts?.[tab.count] ?? 0}</span>
            </button>
          ))}
        </nav>
        <input
          className="mb-3 w-full max-w-xs rounded-lg border border-border bg-surface px-4 py-2 text-sm outline-none focus:border-primary"
          placeholder="Search by title..."
          value={search}
          onChange={(e) => {
            setSearch(e.target.value)
            setPage(1)
          }}
        />
      </div>

      {error && <p className="mt-6 text-primary">{error}</p>}

      <table className="mt-2 w-full text-left text-sm">
        <thead className="text-muted">
          <tr>
            <th className="py-3 font-medium">Title</th>
            <th className="font-medium">Status</th>
            <th className="font-medium">Last edited</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {data.items.map((article) => (
            <tr key={article.id} className="border-t border-border">
              <td className="py-3">
                <p className="font-semibold">{article.title}</p>
                {article.category && <p className="text-xs text-muted">{categoryName(article.category)}</p>}
              </td>
              <td>
                {article.status === 'published' ? (
                  <span className="rounded-full bg-lc-coniferous/10 px-2.5 py-1 text-xs font-semibold text-lc-coniferous">Published</span>
                ) : (
                  <span className="rounded-full bg-background px-2.5 py-1 text-xs font-semibold text-muted">Draft</span>
                )}
              </td>
              <td className="text-muted">{formatDate(article.updated_at)}</td>
              <td className="space-x-4 text-right whitespace-nowrap">
                {article.status === 'published' && (
                  <a href={`/news/${article.slug}`} target="_blank" rel="noreferrer" className="text-muted hover:text-foreground">View ↗</a>
                )}
                <Link to={`/admin/articles/${article.id}`} className="font-semibold text-primary hover:underline">Edit</Link>
                <button onClick={() => handleDelete(article)} className="text-muted hover:text-primary">Delete</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {!error && data.total === 0 && <p className="mt-6 text-muted">No articles found.</p>}

      <Pagination page={page} totalPages={Math.ceil(data.total / PAGE_SIZE)} onChange={setPage} />
    </AdminLayout>
  )
}