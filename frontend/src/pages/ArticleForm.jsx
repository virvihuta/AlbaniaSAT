import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { adminRequest } from '../api'
import AdminLayout from '../components/AdminLayout'
import { Field, SubmitButton, ErrorText } from '../components/Form'
import { categories } from '../categories'
import { useLanguage } from '../i18n'

const empty = { title: '', slug: '', summary: '', content: '', image_url: '', category: '', status: 'draft' }

// "Farming in Korçë!" -> "farming-in-korce", used for the address of the article.
// normalize('NFD') splits "ë" into "e" + dots and "ç" into "c" + hook, then we remove the dots and hooks.
function toSlug(text) {
  return text
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
}

// One form for both: new article (/admin/articles/new) and edit (/admin/articles/5)
export default function ArticleForm() {
  const { id } = useParams()
  const { t } = useLanguage()
  const isNew = id === 'new'
  const navigate = useNavigate()
  const [form, setForm] = useState(empty)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  // When editing, load the article first
  useEffect(() => {
    if (isNew) return
    adminRequest('GET', '/admin/articles/' + id)
      .then((a) => setForm({ ...empty, ...a, summary: a.summary || '', image_url: a.image_url || '', category: a.category || '' }))
      .catch((err) => setError(err.message))
  }, [id, isNew])

  function handleChange(e) {
    const { name, value } = e.target
    // For a new article the slug follows the title automatically
    if (name === 'title' && isNew) setForm({ ...form, title: value, slug: toSlug(value) })
    else setForm({ ...form, [name]: value })
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    const data = { ...form, summary: form.summary || null, image_url: form.image_url || null, category: form.category || null }
    try {
      if (isNew) await adminRequest('POST', '/admin/articles', data)
      else await adminRequest('PUT', '/admin/articles/' + id, data)
      navigate('/admin')
    } catch (err) {
      setError(err.message)
      setLoading(false)
    }
  }

  const boxClass = 'w-full rounded-lg border border-border bg-surface px-4 py-2 outline-none focus:border-primary'

  return (
    <AdminLayout>
      <Link to="/admin" className="text-sm font-semibold text-primary hover:underline">{t('article.back')}</Link>
      <h1 className="mt-4 text-3xl">{isNew ? t('article.newTitle') : t('article.editTitle')}</h1>

      <form onSubmit={handleSubmit} className="mt-8 max-w-2xl">
        <Field label={t('article.title')} name="title" value={form.title} onChange={handleChange} />
        <Field label={t('article.slug')} name="slug" value={form.slug} onChange={handleChange} />
        <Field label={t('article.image')} name="image_url" value={form.image_url} onChange={handleChange} required={false} />

        <label className="mb-4 block">
          <span className="mb-1 block text-sm font-medium">{t('article.summary')}</span>
          <textarea name="summary" rows={2} value={form.summary} onChange={handleChange} className={boxClass} />
        </label>
        <label className="mb-4 block">
          <span className="mb-1 block text-sm font-medium">{t('article.content')}</span>
          <textarea name="content" rows={10} value={form.content} onChange={handleChange} className={boxClass} required />
        </label>
        <label className="mb-4 block">
          <span className="mb-1 block text-sm font-medium">{t('article.topic')}</span>
          <select name="category" value={form.category} onChange={handleChange} className={boxClass}>
            <option value="">{t('article.noTopic')}</option>
            {categories.map((c) => (
              <option key={c.code} value={c.code}>{t('category.' + c.code)}</option>
            ))}
          </select>
        </label>
        <label className="mb-4 block">
          <span className="mb-1 block text-sm font-medium">{t('article.status')}</span>
          <select name="status" value={form.status} onChange={handleChange} className={boxClass}>
            <option value="draft">{t('article.draft')}</option>
            <option value="published">{t('article.published')}</option>
          </select>
        </label>

        <ErrorText error={error} />
        <SubmitButton loading={loading}>{loading ? t('article.saving') : t('article.save')}</SubmitButton>
      </form>
    </AdminLayout>
  )
}