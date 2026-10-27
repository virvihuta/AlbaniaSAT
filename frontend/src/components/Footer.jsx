import { Link } from 'react-router-dom'
import Logo from './Logo'

const columns = [
  { title: 'Platform', items: [{ label: 'Overview', to: '/' }, { label: 'Story', to: '/news' }] },
  { title: 'Data sources', items: [{ label: 'Copernicus Sentinel-2' }, { label: 'ESA WorldCover 2021' }, { label: 'CORINE Land Cover 2018' }] },
]

// Footer for all public pages
export default function Footer() {
  return (
    <footer className="border-t border-border bg-surface">
      <div className="mx-auto grid max-w-7xl gap-10 px-6 py-14 md:grid-cols-4 md:px-10">
        <div className="md:col-span-2">
          <Logo />
          <p className="mt-1 text-sm text-primary">Europe couldn't see Albania. Now it can.</p>
          <p className="mt-4 max-w-sm text-sm text-muted">
            A Diplomarbeit project at HTL Shkodër, supervised by Prof. Reuf Kozlica. Land cover maps built from
            Copernicus Sentinel-2 imagery.
          </p>
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
      <p className="mx-auto max-w-7xl border-t border-border px-6 py-6 text-xs text-muted md:px-10">
        © 2026 AlbaniaSAT · HTL Shkodër
      </p>
    </footer>
  )
}