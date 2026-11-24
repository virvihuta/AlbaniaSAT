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
// wide = true is for the start page: sections go from edge to edge, the footer stays.
export default function PublicLayout({ children, full = false, wide = false }) {
  const { t } = useLanguage()

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-[1100] border-b border-border bg-surface">
        {/* On phones the menu gets its own row under the logo, so nothing is cut off */}
        <div className={'mx-auto flex flex-wrap items-center gap-x-3 px-4 pt-3 sm:h-24 sm:flex-nowrap sm:gap-6 sm:px-6 sm:pt-0 md:px-10 ' + (full ? '' : 'max-w-7xl')}>
          <Logo big />
          <nav className="order-last -mx-2 flex h-12 w-full overflow-x-auto sm:order-none sm:mx-0 sm:ml-10 sm:h-full sm:w-auto sm:gap-2">
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

      {full || wide ? (
        <main className="flex-1">{children}</main>
      ) : (
        <main className="mx-auto w-full max-w-7xl flex-1 px-6 py-16 md:px-10">{children}</main>
      )}

      {!full && <Footer />}
    </div>
  )
}