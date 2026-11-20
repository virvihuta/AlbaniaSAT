import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { getJSON } from '../api'
import PublicLayout from '../components/PublicLayout'
import { Cover, Author } from '../components/ArticleParts'
import { useLanguage } from '../i18n'

// One article with its full text. The address looks like /news/new-sentinel-2-dataset
export default function NewsDetail() {
  const { slug } = useParams()
  const { t } = useLanguage()
  const [article, setArticle] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    getJSON('/articles/' + slug)
      .then(setArticle)
      .catch((err) => setError(err.message === 'Not found' ? 'news.notFound' : 'common.loadError'))
  }, [slug])

  return (
    <PublicLayout>
      <p className="mx-auto max-w-3xl text-sm">
        <Link to="/news" className="text-primary hover:underline">{t('nav.story')}</Link>
        {article?.category && (
          <>
            <span className="text-muted"> › </span>
            <Link to={`/news?topic=${article.category}`} className="text-primary hover:underline">{t('category.' + article.category)}</Link>
          </>
        )}
        {article && <span className="text-muted"> › {article.title}</span>}
      </p>

      {error && <p className="mx-auto mt-8 max-w-3xl text-primary">{t(error)}</p>}
      {!article && !error && <p className="mx-auto mt-8 max-w-3xl text-muted">{t('common.loading')}</p>}

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
              <p className="font-heading text-xl font-light">{t('news.moreTitle')}</p>
              <p className="text-sm text-muted">{t('news.moreText')}</p>
            </div>
            <Link to="/news" className="rounded-lg bg-primary px-5 py-2.5 font-semibold text-white hover:opacity-90">{t('news.allStories')}</Link>
          </div>
        </article>
      )}
    </PublicLayout>
  )
}