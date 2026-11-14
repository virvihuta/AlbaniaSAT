import { useCallback, useEffect, useState } from 'react'
import { adminRequest, formatDate } from '../api'
import AdminLayout from '../components/AdminLayout'

const empty = { question: '', answer: '', is_published: true }

// Private page (admins and editors): questions for the FAQ on the contact page.
// Add, edit, hide, delete and change the order with the arrow buttons.
export default function AdminFaqs() {
  const [faqs, setFaqs] = useState([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [editing, setEditing] = useState(null) // null = no form, 'new' = new question, or the id
  const [form, setForm] = useState(empty)
  const [saving, setSaving] = useState(false)

  const load = useCallback(() => {
    adminRequest('GET', '/admin/faqs')
      .then((data) => {
        setFaqs(data)
        setError('')
      })
      .catch((err) => setError(err.message === 'Not Found' ? 'FAQ management is not available yet.' : err.message))
      .finally(() => setLoading(false))
  }, [])

  useEffect(load, [load])

  function startNew() {
    setEditing('new')
    setForm(empty)
  }

  function startEdit(faq) {
    setEditing(faq.id)
    setForm({ question: faq.question, answer: faq.answer, is_published: faq.is_published })
  }

  function handleChange(e) {
    const { name, value, type, checked } = e.target
    setForm({ ...form, [name]: type === 'checkbox' ? checked : value })
  }

  async function handleSubmit(e) {
    e.preventDefault()
    if (form.question.trim().length < 5) return setError('The question needs at least 5 characters.')
    if (form.answer.trim().length < 10) return setError('The answer needs at least 10 characters.')

    setSaving(true)
    try {
      if (editing === 'new') await adminRequest('POST', '/admin/faqs', form)
      else await adminRequest('PUT', '/admin/faqs/' + editing, form)
      setEditing(null)
      load()
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  async function handleDelete(faq) {
    if (!window.confirm(`Delete "${faq.question}"?`)) return
    try {
      await adminRequest('DELETE', '/admin/faqs/' + faq.id)
      load()
    } catch (err) {
      setError(err.message)
    }
  }

  // Swap two questions and save the new order
  async function move(index, direction) {
    const next = [...faqs]
    const other = index + direction
    ;[next[index], next[other]] = [next[other], next[index]]
    setFaqs(next)
    try {
      await adminRequest('PUT', '/admin/faqs/order', { ids: next.map((f) => f.id) })
    } catch (err) {
      setError(err.message)
      load()
    }
  }

  const boxClass = 'mt-1 w-full rounded-lg border border-border bg-background px-4 py-2 outline-none focus:border-primary'

  const formCard = (
    <form onSubmit={handleSubmit} className="rounded-2xl border border-primary/40 bg-surface p-6">
      <p className="font-semibold">{editing === 'new' ? 'New question' : 'Edit question'}</p>
      <label className="mt-4 block text-sm font-medium">
        Question
        <input name="question" value={form.question} onChange={handleChange} maxLength={200} className={boxClass} />
      </label>
      <label className="mt-4 block text-sm font-medium">
        Answer
        <textarea name="answer" rows={4} value={form.answer} onChange={handleChange} maxLength={2000} className={boxClass} />
        <span className="mt-1 block text-right text-muted">{form.answer.length} / 2000</span>
      </label>
      <label className="mt-2 flex items-center gap-2 text-sm">
        <input type="checkbox" name="is_published" checked={form.is_published} onChange={handleChange} className="accent-[var(--color-primary)]" />
        Visible on the contact page
      </label>
      <div className="mt-6 flex gap-3">
        <button type="submit" disabled={saving} className="rounded-lg bg-primary px-5 py-2 font-semibold text-white hover:opacity-90 disabled:opacity-60">
          {saving ? 'Saving...' : 'Save'}
        </button>
        <button type="button" onClick={() => setEditing(null)} className="rounded-lg border border-border px-5 py-2 font-semibold hover:border-primary">
          Cancel
        </button>
      </div>
    </form>
  )

  return (
    <AdminLayout>
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl">FAQ</h1>
          <p className="mt-1 text-sm text-muted">Questions on the contact page, in this order</p>
        </div>
        {editing === null && (
          <button onClick={startNew} className="rounded-lg bg-primary px-5 py-2 font-semibold text-white hover:opacity-90">
            New question
          </button>
        )}
      </div>

      {error && <p className="mt-6 text-primary">{error}</p>}
      {loading && <p className="mt-6 text-muted">Loading...</p>}
      {!loading && !error && faqs.length === 0 && editing === null && <p className="mt-6 text-muted">No questions yet.</p>}

      <div className="mt-8 space-y-4">
        {editing === 'new' && formCard}

        {faqs.map((faq, index) =>
          editing === faq.id ? (
            <div key={faq.id}>{formCard}</div>
          ) : (
            <article key={faq.id} className="flex gap-4 rounded-2xl border border-border bg-surface p-6">
              <div className="flex flex-col gap-1">
                <button onClick={() => move(index, -1)} disabled={index === 0} aria-label="Move up"
                  className="h-8 w-8 rounded-lg text-muted hover:bg-background hover:text-foreground disabled:opacity-20">↑</button>
                <button onClick={() => move(index, 1)} disabled={index === faqs.length - 1} aria-label="Move down"
                  className="h-8 w-8 rounded-lg text-muted hover:bg-background hover:text-foreground disabled:opacity-20">↓</button>
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <p className="font-semibold">{faq.question}</p>
                  {!faq.is_published && (
                    <span className="rounded-full bg-background px-2.5 py-0.5 text-xs font-semibold text-muted">Hidden</span>
                  )}
                </div>
                <p className="mt-2 line-clamp-2 text-sm text-muted">{faq.answer}</p>
                {faq.updated_at && <p className="mt-2 text-xs text-muted">Last edited {formatDate(faq.updated_at)}</p>}
              </div>
              <div className="flex items-start gap-4 text-sm whitespace-nowrap">
                <button onClick={() => startEdit(faq)} className="font-semibold text-primary hover:underline">Edit</button>
                <button onClick={() => handleDelete(faq)} className="text-muted hover:text-primary">Delete</button>
              </div>
            </article>
          ),
        )}
      </div>
    </AdminLayout>
  )
}