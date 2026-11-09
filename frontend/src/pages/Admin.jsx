import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { adminRequest, formatDate } from '../api'
import AdminLayout from '../components/AdminLayout'
import Pagination from '../components/Pagination'

const PAGE_SIZE = 20

// Start page of the private area: all articles with search, pages, edit and delete
export default function Admin() {
  const [data, setData] = useState({ items: [], total: 0 })
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const [error, setError] = useState('')

  const load = useCallback(() => {
    adminRequest('GET', `/admin/articles?q=${encodeURIComponent(search)}&page=${page}&page_size=${PAGE_SIZE}`)
      .then(setData)
      .catch((err) => setError(err.message))
  }, [search, page])

  // Load again 300 ms after typing or when the page changes
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
        <div>
          <h1 className="text-3xl">Articles</h1>
          <p className="mt-1 text-sm text-muted">{data.total} articles in total</p>
        </div>
        <Link to="/admin/articles/new" className="rounded-lg bg-primary px-5 py-2 font-semibold text-white hover:opacity-90">
          New article
        </Link>
      </div>

      <input
        className="mt-8 w-full max-w-sm rounded-lg border border-border bg-surface px-4 py-2 outline-none focus:border-primary"
        placeholder="Search by title..."
        value={search}
        onChange={(e) => {
          setSearch(e.target.value)
          setPage(1)
        }}
      />

      {error && <p className="mt-6 text-primary">{error}</p>}

      <table className="mt-6 w-full text-left text-sm">
        <thead className="border-b border-border text-muted">
          <tr>
            <th className="py-2">Title</th>
            <th>Status</th>
            <th>Published</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {data.items.map((article) => (
            <tr key={article.id} className="border-b border-border">
              <td className="py-3 font-semibold">{article.title}</td>
              <td>
                <span className={article.status === 'published' ? 'text-lc-coniferous' : 'text-muted'}>{article.status}</span>
              </td>
              <td className="text-muted">{formatDate(article.published_at)}</td>
              <td className="space-x-4 text-right">
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