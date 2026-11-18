import { NavLink } from 'react-router-dom'
import { useLanguage } from '../i18n'
import Logo from './Logo'
import ThemeToggle from './ThemeToggle'
import LanguageToggle from './LanguageToggle'
import Footer from './Footer'

const links = [
  { to: '/', label: 'nav.overview' },
  { to: '/explorer', label: 'nav.explorer' },
  { to: '/news', label: 'nav.story' },
  { to: '/contact', label: 'nav.contact' },
]

// Header with logo and navigation, the page content and the footer.
// full = true is for the map: the page uses the whole width and has no footer.
export default function PublicLayout({ children, full = false }) {
  const { t } = useLanguage()

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-[1100] border-b border-border bg-surface">
        <div className={'mx-auto flex h-24 items-center gap-3 px-4 sm:gap-6 sm:px-6 md:px-10 ' + (full ? '' : 'max-w-7xl')}>
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
                {t(link.label)}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-2">
            <LanguageToggle />
            <ThemeToggle />
          </div>
        </div>
      </header>

      {full ? (
        <main className="flex-1">{children}</main>
      ) : (
        <main className="mx-auto w-full max-w-7xl flex-1 px-6 py-16 md:px-10">{children}</main>
      )}

      {!full && <Footer />}
    </div>
  )
}