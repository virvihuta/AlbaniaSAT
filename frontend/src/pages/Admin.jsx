import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { adminRequest, formatDate } from '../api'
import AdminLayout from '../components/AdminLayout'

// Start page of the private area: all articles with edit and delete
export default function Admin() {
  const [articles, setArticles] = useState([])
  const [error, setError] = useState('')

  function load() {
    adminRequest('GET', '/admin/articles').then(setArticles).catch((err) => setError(err.message))
  }

  useEffect(load, [])

  async function handleDelete(article) {
    if (!window.confirm(`Delete "${article.title}"?`)) return
    try {
      await adminRequest('DELETE', '/admin/articles/' + article.id)
      load()
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <AdminLayout>
      <div className="flex items-center justify-between">
        <h1 className="text-3xl">Articles</h1>
        <Link to="/admin/articles/new" className="rounded-lg bg-primary px-5 py-2 font-semibold text-white hover:opacity-90">
          New article
        </Link>
      </div>

      {error && <p className="mt-6 text-primary">{error}</p>}

      <table className="mt-8 w-full text-left text-sm">
        <thead className="border-b border-border text-muted">
          <tr>
            <th className="py-2">Title</th>
            <th>Status</th>
            <th>Published</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {articles.map((article) => (
            <tr key={article.id} className="border-b border-border">
              <td className="py-3 font-semibold">{article.title}</td>
              <td>
                <span className={article.status === 'published' ? 'text-lc-forest' : 'text-muted'}>{article.status}</span>
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
      {!error && articles.length === 0 && <p className="mt-6 text-muted">No articles yet.</p>}
    </AdminLayout>
  )
}