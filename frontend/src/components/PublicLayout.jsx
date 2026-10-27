import { Link, NavLink } from 'react-router-dom'

const links = [
  { to: '/', label: 'Map' },
  { to: '/news', label: 'News' },
]

// Header with navigation for all public pages
export default function PublicLayout({ children }) {
  return (
    <div className="min-h-screen">
      <header className="flex items-center justify-between border-b border-border px-8 py-4">
        <Link to="/" className="font-heading text-sm font-semibold uppercase tracking-[0.3em] text-primary">
          AlbaniaSAT
        </Link>
        <nav className="flex gap-6 text-sm font-semibold">
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === '/'}
              className={({ isActive }) => (isActive ? 'text-primary' : 'text-muted hover:text-foreground')}
            >
              {link.label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main className="px-8 py-12">{children}</main>
    </div>
  )
}