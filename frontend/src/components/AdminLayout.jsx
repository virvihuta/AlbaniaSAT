import { useAuth } from '../auth'

// Header for the private area: user name and logout
export default function AdminLayout({ children }) {
  const { user, logout } = useAuth()

  return (
    <div className="min-h-screen">
      <header className="flex items-center justify-between border-b border-border bg-surface px-8 py-4">
        <p className="font-heading text-sm font-semibold uppercase tracking-[0.3em] text-primary">AlbaniaSAT Admin</p>
        <div className="flex items-center gap-4">
          <span className="text-sm text-muted">{user.name}</span>
          <button
            onClick={logout}
            className="rounded-lg border border-border px-4 py-2 text-sm font-semibold hover:bg-background"
          >
            Logout
          </button>
        </div>
      </header>
      <main className="px-8 py-12">{children}</main>
    </div>
  )
}