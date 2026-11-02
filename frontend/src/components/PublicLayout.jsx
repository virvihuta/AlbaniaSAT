import { NavLink } from 'react-router-dom'
import Logo from './Logo'
import ThemeToggle from './ThemeToggle'
import Footer from './Footer'

const links = [
  { to: '/', label: 'Overview' },
  { to: '/news', label: 'Story' },
  { to: '/contact', label: 'Contact' },
]

// Header with logo and navigation, the page content and the footer
export default function PublicLayout({ children }) {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-10 border-b border-border bg-surface">
        <div className="mx-auto flex h-24 max-w-7xl items-center gap-3 px-4 sm:gap-6 sm:px-6 md:px-10">
          <Logo big />
          <nav className="flex h-full sm:ml-10 sm:gap-2">
            {links.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                end={link.to === '/'}
                className={({ isActive }) =>
                  // Red line under the link: full for the open page, grows in on hover for the others
                  'relative flex items-center px-2 text-sm font-medium transition-colors sm:px-4 sm:text-[15px] ' +
                  'after:absolute after:inset-x-2 after:bottom-0 sm:after:inset-x-4 after:h-0.5 after:bg-primary after:transition-transform ' +
                  (isActive ? 'text-foreground after:scale-x-100' : 'text-muted after:scale-x-0 hover:text-foreground hover:after:scale-x-100')
                }
              >
                {link.label}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto">
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-7xl flex-1 px-6 py-16 md:px-10">{children}</main>

      <Footer />
    </div>
  )
}