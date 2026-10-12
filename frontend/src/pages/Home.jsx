import { useAuth } from '../auth'

export default function Home() {
  // Logging out sets user to null, and ProtectedRoute sends you to /login by itself
  const { user, logout } = useAuth()

  return (
    <div className="min-h-screen">
      <header className="flex items-center justify-between border-b border-border px-8 py-4">
        <p className="font-heading text-sm font-semibold uppercase tracking-[0.3em] text-primary">AlbaniaSAT</p>
        <div className="flex items-center gap-4">
          <span className="text-sm text-muted">{user.name}</span>
          <button
            onClick={logout}
            className="rounded-lg border border-border px-4 py-2 text-sm font-semibold hover:bg-surface"
          >
            Logout
          </button>
        </div>
      </header>

      <main className="px-8 py-12">
        <h1 className="text-3xl">Welcome, {user.name}</h1>
        <p className="mt-2 text-muted">The map will live here.</p>
      </main>
    </div>
  )
}