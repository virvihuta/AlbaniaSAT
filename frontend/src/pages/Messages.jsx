import { useEffect, useState } from 'react'
import { adminRequest, formatDate } from '../api'
import AdminLayout from '../components/AdminLayout'
import Pagination from '../components/Pagination'
import YearFilter from '../components/YearFilter'

const PAGE_SIZE = 20

// "November 2026" for the month headings
function monthName(value) {
  return new Date(value).toLocaleDateString('en-GB', { month: 'long', year: 'numeric' })
}

// Private page (admins only): contact messages, newest first, by year, month and page.
// New messages are marked with a red dot until someone marks them as read.
export default function Messages() {
  const [data, setData] = useState({ items: [], total: 0 })
  const [years, setYears] = useState([])
  const [year, setYear] = useState('')
  const [page, setPage] = useState(1)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    adminRequest('GET', '/admin/contact/years').then(setYears).catch(() => setYears([]))
  }, [])

  useEffect(() => {
    const yearFilter = year ? `&year=${year}` : ''
    adminRequest('GET', `/admin/contact?page=${page}&page_size=${PAGE_SIZE}${yearFilter}`)
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [year, page])

  // Mark one message as read or unread. The header badge listens for "messages-changed".
  async function toggleRead(message) {
    try {
      await adminRequest('PATCH', '/admin/contact/' + message.id, { is_read: !message.is_read })
      setData({ ...data, items: data.items.map((m) => (m.id === message.id ? { ...m, is_read: !m.is_read } : m)) })
      window.dispatchEvent(new Event('messages-changed'))
    } catch (err) {
      setError(err.message)
    }
  }

  function chooseYear(newYear) {
    setYear(newYear)
    setPage(1)
  }

  return (
    <AdminLayout>
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl">Messages</h1>
          <p className="mt-1 text-sm text-muted">{data.total} messages from the contact form</p>
        </div>
        <YearFilter years={years} value={year} onChange={chooseYear} />
      </div>

      {error && <p className="mt-6 text-primary">{error}</p>}
      {loading && <p className="mt-6 text-muted">Loading...</p>}
      {!loading && !error && data.total === 0 && <p className="mt-6 text-muted">No messages yet.</p>}

      <div className="mt-6 space-y-4">
        {data.items.map((m, i) => {
          // Show a month heading when the month changes
          const month = monthName(m.created_at)
          const newMonth = i === 0 || month !== monthName(data.items[i - 1].created_at)
          // Only messages the API marks as unread get the red dot (no is_read from the API = no dot)
          const unread = m.is_read != null && !m.is_read
          return (
            <div key={m.id}>
              {newMonth && <p className="eyebrow mb-3 mt-8">{month}</p>}
              <article
                className={
                  'rounded-2xl border bg-surface p-6 ' + (unread ? 'border-primary/40 shadow-sm' : 'border-border')
                }
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="flex items-center gap-2">
                    {unread && <span className="h-2 w-2 rounded-full bg-primary" aria-label="Unread" />}
                    <span className={unread ? 'font-bold' : 'font-medium'}>{m.name}</span>
                    <a href={`mailto:${m.email}`} className="text-primary hover:underline">{m.email}</a>
                  </p>
                  <div className="flex items-center gap-4 text-sm">
                    <span className="text-muted">{formatDate(m.created_at)}</span>
                    {m.is_read != null && (
                      <button onClick={() => toggleRead(m)} className="font-medium text-muted hover:text-primary">
                        {unread ? 'Mark as read' : 'Mark as unread'}
                      </button>
                    )}
                  </div>
                </div>
                <p className={'mt-3 whitespace-pre-line ' + (unread ? 'text-foreground' : 'text-muted')}>{m.message}</p>
              </article>
            </div>
          )
        })}
      </div>

      <Pagination page={page} totalPages={Math.ceil(data.total / PAGE_SIZE)} onChange={setPage} />
    </AdminLayout>
  )
}