import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { getJSON, formatDate } from '../api'
import PublicLayout from '../components/PublicLayout'

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
      <Link to="/news" className="text-sm font-semibold text-primary hover:underline">← All news</Link>

      {error && <p className="mt-8 text-primary">{error}</p>}
      {!article && !error && <p className="mt-8 text-muted">Loading...</p>}

      {article && (
        <article className="mt-6 max-w-2xl">
          <p className="text-sm text-muted">
            {formatDate(article.published_at)}
            {article.author && ` · ${article.author}`}
          </p>
          <h1 className="mt-2 text-4xl leading-tight">{article.title}</h1>
          {article.image_url && <img src={article.image_url} alt="" className="mt-6 w-full rounded-lg" />}
          <p className="mt-6 whitespace-pre-line leading-relaxed">{article.content}</p>
        </article>
      )}
    </PublicLayout>
  )
}