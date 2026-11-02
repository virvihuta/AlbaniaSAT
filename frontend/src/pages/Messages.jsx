import { useEffect, useState } from 'react'
import { adminRequest, formatDate } from '../api'
import AdminLayout from '../components/AdminLayout'

// Private page: all messages from the contact form, newest first
export default function Messages() {
  const [messages, setMessages] = useState([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    adminRequest('GET', '/admin/contact')
      .then(setMessages)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  return (
    <AdminLayout>
      <h1 className="text-3xl">Messages</h1>
      <p className="mt-1 text-muted">Messages sent through the contact form.</p>

      {error && <p className="mt-6 text-primary">{error}</p>}
      {loading && <p className="mt-6 text-muted">Loading...</p>}
      {!loading && !error && messages.length === 0 && <p className="mt-6 text-muted">No messages yet.</p>}

      <div className="mt-8 space-y-4">
        {messages.map((m) => (
          <article key={m.id} className="rounded-2xl border border-border bg-surface p-6">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <p className="font-semibold">
                {m.name}{' '}
                <a href={`mailto:${m.email}`} className="font-normal text-primary hover:underline">{m.email}</a>
              </p>
              <p className="text-sm text-muted">{formatDate(m.created_at)}</p>
            </div>
            <p className="mt-3 whitespace-pre-line text-muted">{m.message}</p>
          </article>
        ))}
      </div>
    </AdminLayout>
  )
}