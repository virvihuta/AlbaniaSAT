import PublicLayout from '../components/PublicLayout'

// Public start page. The interactive map comes here later (goal M-3.3).
export default function Home() {
  return (
    <PublicLayout>
      <h1 className="max-w-2xl text-4xl leading-tight">Europe couldn't see Albania. Now it can.</h1>
      <p className="mt-4 max-w-xl text-muted">Land cover of Albania, classified from Sentinel-2 satellite images.</p>
      <div className="mt-10 flex h-96 items-center justify-center rounded-lg border border-border bg-surface text-muted">
        The map will live here.
      </div>
    </PublicLayout>
  )
}