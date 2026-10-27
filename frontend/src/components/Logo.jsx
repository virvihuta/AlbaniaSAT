import { Link } from 'react-router-dom'

// Red logo mark with the name next to it. Clicking it goes to the start page.
export default function Logo({ to = '/' }) {
  return (
    <Link to={to} className="flex items-center gap-2">
      <img src="/logo.png" alt="" className="h-9" />
      <span className="hidden font-heading text-2xl font-light sm:inline">AlbaniaSAT</span>
    </Link>
  )
}