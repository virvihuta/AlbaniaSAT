import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getJSON, formatDate } from '../api'
import PublicLayout from '../components/PublicLayout'

// Public news page: list of published articles with a search field
export default function News() {
  const [search, setSearch] = useState('')
  const [articles, setArticles] = useState([])
  const [error, setError] = useState('')

  // Load the articles again whenever the search text changes (waits 300 ms after typing)
  useEffect(() => {
    const timer = setTimeout(() => {
      getJSON('/articles?q=' + encodeURIComponent(search))
        .then((data) => {
          setArticles(data)
          setError('')
        })
        .catch((err) => setError(err.message))
    }, 300)
    return () => clearTimeout(timer)
  }, [search])

  return (
    <PublicLayout>
      <h1 className="text-3xl">News</h1>
      <input
        className="mt-6 w-full max-w-md rounded-lg border border-border bg-surface px-4 py-2 outline-none focus:border-primary"
        placeholder="Search articles..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />

      {error && <p className="mt-8 text-primary">{error}</p>}

      <div className="mt-8 grid gap-4">
        {articles.map((article) => (
          <Link key={article.id} to={`/news/${article.slug}`} className="rounded-lg border border-border p-6 hover:border-primary">
            <p className="text-xs text-muted">{formatDate(article.published_at)}</p>
            <h2 className="mt-1 text-xl">{article.title}</h2>
            {article.summary && <p className="mt-2 text-muted">{article.summary}</p>}
          </Link>
        ))}
        {!error && articles.length === 0 && <p className="text-muted">No articles found.</p>}
      </div>
    </PublicLayout>
  )
}