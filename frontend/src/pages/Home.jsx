import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getJSON } from '../api'
import PublicLayout from '../components/PublicLayout'
import { landCoverClasses } from '../landCover'

// 0.8094 -> "80.94%"
function percent(value) {
  return (value * 100).toFixed(2) + '%'
}

// Public start page. The card on the right leads to the map explorer.
// Accuracy and F1 score come from the API (GET /metrics), so they update when there is a new model.
export default function Home() {
  // Values of the current model. Used until the API answers, or if it cannot be reached.
  const [model, setModel] = useState({ accuracy: 0.8094, f1_macro: 0.8138 })

  useEffect(() => {
    getJSON('/metrics')
      .then((list) => {
        // The list has one entry per model version. The newest one has the highest id.
        const newest = list.reduce((a, b) => (b.id > a.id ? b : a), list[0])
        if (newest) setModel(newest)
      })
      .catch(() => {})
  }, [])

  const stats = [
    { value: percent(model.accuracy), text: `Test accuracy of the AlbaniaSAT model on held-out Albanian patches (macro F1 ${percent(model.f1_macro)})` },
    { value: '19.08%', text: 'Accuracy of a EuroSAT-trained model on the same Albanian data' },
    { value: '7', text: 'Land cover classes, from forest to built-up land' },
    { value: '~7,800', text: 'Labelled 64 × 64 px Sentinel-2 patches' },
  ]

  return (
    <PublicLayout>
      <section className="grid items-center gap-12 lg:grid-cols-2">
        <div>
          <p className="eyebrow flex items-center gap-3">
            <span className="h-px w-6 bg-primary" /> Land cover of Albania · Sentinel-2 · 2021, more years coming
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
          <Link
            to="/explorer"
            className="group flex h-96 flex-1 flex-col items-center justify-center gap-3 rounded-xl bg-background text-center"
          >
            <span className="font-heading text-2xl font-light group-hover:text-primary">Explore the map</span>
            <span className="text-sm text-muted">Regions, classes and Sentinel-2 patches</span>
            <span className="mt-2 rounded-lg bg-primary px-5 py-2.5 text-sm font-semibold text-white group-hover:opacity-90">Open the explorer →</span>
          </Link>
          <div className="self-end">
            <p className="eyebrow mb-3">Classes</p>
            {landCoverClasses.map((c) => (
              <p key={c.code} className="mb-2 flex items-center gap-2 text-sm">
                <span className="h-3 w-3 rounded-sm" style={{ backgroundColor: c.color }} /> {c.name}
              </p>
            ))}
          </div>
        </div>
      </section>

      <section className="mt-20 grid border-y border-border sm:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => (
          <div key={stat.text} className="border-border py-8 pr-6 lg:border-r lg:pl-6 lg:first:pl-0 lg:last:border-r-0">
            <p className="font-heading text-5xl font-light">{stat.value}</p>
            <p className="mt-3 text-sm text-muted">{stat.text}</p>
          </div>
        ))}
      </section>
    </PublicLayout>
  )
}