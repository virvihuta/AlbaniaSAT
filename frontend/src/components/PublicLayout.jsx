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
        <div className="mx-auto flex h-24 max-w-7xl items-center gap-6 px-6 md:px-10">
          <Logo big />
          <nav className="flex h-full gap-2 sm:ml-10">
            {links.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                end={link.to === '/'}
                className={({ isActive }) =>
                  // Red line under the link: full for the open page, grows in on hover for the others
                  'relative flex items-center px-4 text-[15px] font-medium transition-colors ' +
                  'after:absolute after:inset-x-4 after:bottom-0 after:h-0.5 after:bg-primary after:transition-transform ' +
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