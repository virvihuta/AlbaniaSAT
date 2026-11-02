import { NavLink } from 'react-router-dom'
import { useAuth } from '../auth'
import Logo from './Logo'

// Header for the private area: user name and logout
export default function AdminLayout({ children }) {
  const { user, logout } = useAuth()

  return (
    <div className="min-h-screen">
      <header className="flex items-center justify-between border-b border-border bg-surface px-8 py-4">
        <div className="flex items-center gap-3">
          <Logo to="/admin" big />
          <span className="rounded-full bg-primary px-2.5 py-0.5 text-xs font-semibold text-white">Admin</span>
          <nav className="ml-6 flex gap-5 text-sm font-medium">
            {[
              { to: '/admin', label: 'Articles' },
              { to: '/admin/messages', label: 'Messages' },
            ].map((link) => (
              <NavLink key={link.to} to={link.to} end
                className={({ isActive }) => (isActive ? 'text-primary' : 'text-muted hover:text-foreground')}>
                {link.label}
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