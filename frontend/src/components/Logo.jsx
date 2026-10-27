import { Link } from 'react-router-dom'

// Red logo mark with the name next to it. Clicking it goes to the start page.
// big = true is for the header, the footer uses the normal size.
export default function Logo({ to = '/', big = false }) {
  return (
    <Link to={to} className="flex shrink-0 items-center gap-3">
      <img src="/logo.png" alt="AlbaniaSAT logo" className={big ? 'h-14 w-auto shrink-0' : 'h-11 w-auto shrink-0'} />
      <span className={`hidden font-heading font-light tracking-tight sm:inline ${big ? 'text-[2rem]' : 'text-[1.75rem]'}`}>
        AlbaniaSAT
      </span>
    </Link>
  )
}