import { Link } from 'react-router-dom'
import PublicLayout from '../components/PublicLayout'

const classes = [
  { name: 'Forest', color: 'bg-lc-forest' },
  { name: 'Shrubland', color: 'bg-lc-shrubland' },
  { name: 'Grassland', color: 'bg-lc-grassland' },
  { name: 'Cropland', color: 'bg-lc-cropland' },
  { name: 'Built-up', color: 'bg-lc-builtup' },
  { name: 'Bare ground', color: 'bg-lc-bare' },
  { name: 'Water', color: 'bg-lc-water' },
]

const stats = [
  { value: '80.94%', text: 'Test accuracy of the AlbaniaSAT model on held-out Albanian patches' },
  { value: '19.08%', text: 'Accuracy of a EuroSAT-trained model on the same Albanian data' },
  { value: '7', text: 'Land cover classes, from forest to built-up land' },
  { value: '~7,800', text: 'Labelled 64 × 64 px Sentinel-2 patches' },
]

// Public start page. The interactive map comes into the card later (goal M-3.3).
export default function Home() {
  return (
    <PublicLayout>
      <section className="grid items-center gap-12 lg:grid-cols-2">
        <div>
          <p className="eyebrow flex items-center gap-3">
            <span className="h-px w-6 bg-primary" /> Land cover of Albania · Sentinel-2 · 2017–2024
          </p>
          <h1 className="mt-6 text-5xl leading-tight md:text-7xl">Europe couldn't see Albania. Now it can.</h1>
          <p className="mt-8 max-w-xl text-lg text-muted">
            AlbaniaSAT maps land cover across the whole country from Sentinel-2 satellite imagery, with a model trained
            on Albanian ground truth instead of borrowed European data.
          </p>
          <Link to="/news" className="mt-10 inline-block rounded-lg bg-primary px-6 py-3 font-semibold text-white hover:opacity-90">
            Read the stories →
          </Link>
        </div>

        <div className="flex gap-4 rounded-2xl border border-border bg-surface p-6">
          <div className="flex h-96 flex-1 items-center justify-center rounded-xl bg-background text-center text-sm text-muted">
            The interactive map comes here.
          </div>
          <div className="self-end">
            <p className="eyebrow mb-3">Classes</p>
            {classes.map((c) => (
              <p key={c.name} className="mb-2 flex items-center gap-2 text-sm">
                <span className={`h-3 w-3 rounded-sm ${c.color}`} /> {c.name}
              </p>
            ))}
          </div>
        </div>
      </section>

      <section className="mt-20 grid border-y border-border sm:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => (
          <div key={stat.value} className="border-border py-8 pr-6 lg:border-r lg:pl-6 lg:first:pl-0 lg:last:border-r-0">
            <p className="font-heading text-5xl font-light">{stat.value}</p>
            <p className="mt-3 text-sm text-muted">{stat.text}</p>
          </div>
        ))}
      </section>
    </PublicLayout>
  )
}