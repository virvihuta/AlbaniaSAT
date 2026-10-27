import { NavLink } from 'react-router-dom'
import Logo from './Logo'
import ThemeToggle from './ThemeToggle'
import Footer from './Footer'

const links = [
  { to: '/', label: 'Overview' },
  { to: '/news', label: 'Story' },
]

// Header with logo and navigation, the page content and the footer
export default function PublicLayout({ children }) {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-10 border-b border-border bg-surface">
        <div className="mx-auto flex h-18 max-w-7xl items-center justify-between gap-6 px-6 md:px-10">
          <Logo />
          <nav className="flex h-full gap-8">
            {links.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                end={link.to === '/'}
                className={({ isActive }) =>
                  'flex items-center border-b-2 ' +
                  (isActive ? 'border-primary text-foreground' : 'border-transparent text-muted hover:text-foreground')
                }
              >
                {link.label}
              </NavLink>
            ))}
          </nav>
          <ThemeToggle />
        </div>
      </header>

      <main className="mx-auto w-full max-w-7xl flex-1 px-6 py-16 md:px-10">{children}</main>

      <Footer />
    </div>
  )
}