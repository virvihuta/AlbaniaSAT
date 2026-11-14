import { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import { adminRequest } from '../api'
import { useAuth } from '../auth'
import Logo from './Logo'

// Header for the private area: user name and logout
export default function AdminLayout({ children }) {
  const { user, logout } = useAuth()
  const [unread, setUnread] = useState(0)

  // Number of unread messages for the badge (admins only). Updates when a message is marked.
  useEffect(() => {
    if (user.role !== 'admin') return
    const loadUnread = () =>
      adminRequest('GET', '/admin/contact/unread-count')
        .then((d) => setUnread(d.count))
        .catch(() => setUnread(0))
    loadUnread()
    window.addEventListener('messages-changed', loadUnread)
    return () => window.removeEventListener('messages-changed', loadUnread)
  }, [user.role])

  // Editors see articles and FAQ. Messages have personal data, so only admins see them.
  const links = [
    { to: '/admin', label: 'Articles' },
    { to: '/admin/faqs', label: 'FAQ' },
  ]
  if (user.role === 'admin') links.push({ to: '/admin/messages', label: 'Messages' })

  return (
    <div className="min-h-screen">
      <header className="flex items-center justify-between border-b border-border bg-surface px-8 py-4">
        <div className="flex items-center gap-3">
          <Logo to="/admin" big />
          <span className="rounded-full bg-primary px-2.5 py-0.5 text-xs font-semibold text-white">
            {user.role === 'admin' ? 'Admin' : 'Editor'}
          </span>
          <nav className="ml-6 flex gap-5 text-sm font-medium">
            {links.map((link) => (
              <NavLink key={link.to} to={link.to} end
                className={({ isActive }) => (isActive ? 'text-primary' : 'text-muted hover:text-foreground')}>
                {link.label}
                {link.to === '/admin/messages' && unread > 0 && (
                  <span className="ml-1.5 rounded-full bg-primary px-1.5 py-0.5 text-xs font-semibold text-white">{unread}</span>
                )}
              </NavLink>
            ))}
          </nav>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-sm text-muted">{user.name}</span>
          <button
            onClick={logout}
            className="rounded-lg border border-border px-4 py-2 text-sm font-semibold hover:border-primary hover:text-primary"
          >
            Logout
          </button>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-8 py-12">{children}</main>
    </div>
  )
}