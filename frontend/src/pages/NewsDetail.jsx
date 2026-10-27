import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { getJSON } from '../api'
import PublicLayout from '../components/PublicLayout'
import { Cover, Author } from '../components/ArticleParts'

// One article with its full text. The address looks like /news/new-sentinel-2-dataset
export default function NewsDetail() {
  const { slug } = useParams()
  const [article, setArticle] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    getJSON('/articles/' + slug)
      .then(setArticle)
      .catch((err) => setError(err.message))
  }, [slug])

  return (
    <PublicLayout>
      <p className="mx-auto max-w-3xl text-sm">
        <Link to="/news" className="text-primary hover:underline">Story</Link>
        {article && <span className="text-muted"> › {article.title}</span>}
      </p>

      {error && <p className="mx-auto mt-8 max-w-3xl text-primary">{error}</p>}
      {!article && !error && <p className="mx-auto mt-8 max-w-3xl text-muted">Loading...</p>}

      {article && (
        <article>
          <header className="mx-auto mt-6 max-w-3xl">
            <h1 className="text-5xl leading-tight md:text-6xl">{article.title}</h1>
            {article.summary && <p className="mt-6 text-lg text-muted">{article.summary}</p>}
            <div className="mt-6">
              <Author article={article} />
            </div>
          </header>

          <Cover article={article} className="mt-10 h-72 w-full rounded-2xl md:h-[28rem]" />

          <p className="mx-auto mt-10 max-w-2xl whitespace-pre-line text-lg leading-relaxed">{article.content}</p>

          <div className="mx-auto mt-14 flex max-w-2xl flex-wrap items-center justify-between gap-4 rounded-2xl border border-border bg-surface p-6">
            <div>
              <p className="font-heading text-xl font-light">More from the team</p>
              <p className="text-sm text-muted">Read the other stories about Albania's land.</p>
            </div>
            <Link to="/news" className="rounded-lg bg-primary px-5 py-2.5 font-semibold text-white hover:opacity-90">
              All stories →
            </Link>
          </div>
        </article>
      )}
    </PublicLayout>
  )
}