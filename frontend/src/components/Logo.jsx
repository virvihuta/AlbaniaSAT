import { Link } from 'react-router-dom'

// Red logo mark with the name next to it. Clicking it goes to the start page.
export default function Logo({ to = '/' }) {
  return (
    <Link to={to} className="flex items-center gap-3">
      <img src="/logo.png" alt="" className="h-11" />
      <span className="hidden font-heading text-[1.75rem] font-light tracking-tight sm:inline">AlbaniaSAT</span>
    </Link>
  )
}