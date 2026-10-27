import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getJSON } from '../api'
import PublicLayout from '../components/PublicLayout'
import { Cover, Author } from '../components/ArticleParts'

// Public news page ("Story"): newest article big on top, the others in a grid
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

  const [featured, ...rest] = articles

  return (
    <PublicLayout>
      <p className="eyebrow">Story</p>
      <h1 className="mt-3 text-5xl">What the maps show</h1>
      <p className="mt-4 max-w-xl text-lg text-muted">
        Short reads on Albania's changing land, and on how AlbaniaSAT was built, written by the team.
      </p>
      <input
        className="mt-8 w-full max-w-md rounded-full border border-border bg-surface px-5 py-2.5 outline-none focus:border-primary"
        placeholder="Search stories..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />

      {error && <p className="mt-8 text-primary">{error}</p>}
      {!error && articles.length === 0 && <p className="mt-8 text-muted">No stories found.</p>}

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

      <div className="mt-6 grid gap-6 md:grid-cols-3">
        {rest.map((article) => (
          <Link key={article.id} to={`/news/${article.slug}`} className="group overflow-hidden rounded-2xl border border-border bg-surface">
            <Cover article={article} className="h-48 w-full" />
            <div className="space-y-3 p-6">
              <h2 className="text-2xl group-hover:text-primary">{article.title}</h2>
              {article.summary && <p className="text-sm text-muted">{article.summary}</p>}
              <Author article={article} />
            </div>
          </Link>
        ))}
      </div>
    </PublicLayout>
  )
}