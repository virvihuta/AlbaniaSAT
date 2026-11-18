import { Link } from 'react-router-dom'
import { useLanguage } from '../i18n'
import Logo from './Logo'

// Footer for all public pages
export default function Footer() {
  const { t } = useLanguage()

  const columns = [
    {
      title: t('footer.platform'),
      items: [
        { label: t('nav.overview'), to: '/' },
        { label: t('nav.explorer'), to: '/explorer' },
        { label: t('nav.story'), to: '/news' },
        { label: t('nav.contact'), to: '/contact' },
      ],
    },
    {
      title: t('footer.sources'),
      items: [{ label: 'Copernicus Sentinel-2' }, { label: 'ESA WorldCover 2021' }, { label: 'CORINE Land Cover 2018' }],
    },
  ]

  return (
    <footer className="border-t border-border bg-surface">
      <div className="mx-auto grid max-w-7xl gap-10 px-6 py-14 md:grid-cols-4 md:px-10">
        <div className="md:col-span-2">
          <Logo />
          <p className="mt-1 text-sm text-primary">{t('tagline')}</p>
          <p className="mt-4 max-w-sm text-sm text-muted">{t('footer.about')}</p>
        </div>

        {columns.map((column) => (
          <div key={column.title}>
            <p className="eyebrow">{column.title}</p>
            <ul className="mt-4 space-y-3 text-sm">
              {column.items.map((item) => (
                <li key={item.label}>
                  {item.to ? <Link to={item.to} className="hover:text-primary">{item.label}</Link> : item.label}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      <div className="mx-auto flex max-w-7xl flex-wrap justify-between gap-2 border-t border-border px-6 py-6 text-xs text-muted md:px-10">
        <p>{t('footer.rights')}</p>
        <p>{t('footer.copernicus')}</p>
      </div>
    </footer>
  )
}