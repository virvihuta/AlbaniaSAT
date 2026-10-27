import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { adminRequest } from '../api'
import AdminLayout from '../components/AdminLayout'
import { Field, SubmitButton, ErrorText } from '../components/Form'

const empty = { title: '', slug: '', summary: '', content: '', image_url: '', status: 'draft' }

// "My New Article!" -> "my-new-article", used for the address of the article
function toSlug(text) {
  return text.toLowerCase().trim().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')
}

// One form for both: new article (/admin/articles/new) and edit (/admin/articles/5)
export default function ArticleForm() {
  const { id } = useParams()
  const isNew = id === 'new'
  const navigate = useNavigate()
  const [form, setForm] = useState(empty)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  // When editing, load the article first
  useEffect(() => {
    if (isNew) return
    adminRequest('GET', '/admin/articles/' + id)
      .then((a) => setForm({ ...empty, ...a, summary: a.summary || '', image_url: a.image_url || '' }))
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
    const data = { ...form, summary: form.summary || null, image_url: form.image_url || null }
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
      <Link to="/admin" className="text-sm font-semibold text-primary hover:underline">← All articles</Link>
      <h1 className="mt-4 text-3xl">{isNew ? 'New article' : 'Edit article'}</h1>

      <form onSubmit={handleSubmit} className="mt-8 max-w-2xl">
        <Field label="Title" name="title" value={form.title} onChange={handleChange} />
        <Field label="Slug (address: /news/...)" name="slug" value={form.slug} onChange={handleChange} />
        <Field label="Image URL (optional)" name="image_url" value={form.image_url} onChange={handleChange} required={false} />

        <label className="mb-4 block">
          <span className="mb-1 block text-sm font-medium">Summary (optional)</span>
          <textarea name="summary" rows={2} value={form.summary} onChange={handleChange} className={boxClass} />
        </label>
        <label className="mb-4 block">
          <span className="mb-1 block text-sm font-medium">Content</span>
          <textarea name="content" rows={10} value={form.content} onChange={handleChange} className={boxClass} required />
        </label>
        <label className="mb-4 block">
          <span className="mb-1 block text-sm font-medium">Status</span>
          <select name="status" value={form.status} onChange={handleChange} className={boxClass}>
            <option value="draft">Draft (only visible here)</option>
            <option value="published">Published (visible on the news page)</option>
          </select>
        </label>

        <ErrorText error={error} />
        <SubmitButton loading={loading}>{loading ? 'Saving...' : 'Save article'}</SubmitButton>
      </form>
    </AdminLayout>
  )
}