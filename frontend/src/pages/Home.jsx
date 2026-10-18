// Public start page: everyone can see it without login.
// The interactive map comes here later (goal M-3.3).
export default function Home() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-border px-8 py-4">
        <p className="font-heading text-sm font-semibold uppercase tracking-[0.3em] text-primary">AlbaniaSAT</p>
      </header>

      <main className="px-8 py-12">
        <h1 className="max-w-2xl text-4xl leading-tight">Europe couldn't see Albania. Now it can.</h1>
        <p className="mt-4 max-w-xl text-muted">Land cover of Albania, classified from Sentinel-2 satellite images.</p>
        <div className="mt-10 flex h-96 items-center justify-center rounded-lg border border-border bg-surface text-muted">
          The map will live here.
        </div>
      </main>
    </div>
  )
}