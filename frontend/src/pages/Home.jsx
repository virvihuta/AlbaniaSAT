import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getJSON } from '../api'
import { useLanguage } from '../i18n'
import PublicLayout from '../components/PublicLayout'
import { landCoverClasses, className } from '../landCover'

// 0.8094 -> "80.94%" in English, "80,94%" in Albanian and German
function percent(value, lang) {
  const text = (value * 100).toFixed(2) + '%'
  return lang === 'en' ? text : text.replace('.', ',')
}

// How many patches, written the way each language writes big numbers
const patchCount = { en: '~7,800', sq: '~7 800', de: '~7.800' }

// Public start page. The card on the right leads to the map explorer.
// Accuracy and F1 score come from the API (GET /metrics), so they update when there is a new model.
export default function Home() {
  const { lang, t } = useLanguage()
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
    { key: 'accuracy', value: percent(model.accuracy, lang), text: t('home.statAccuracy', { f1: percent(model.f1_macro, lang) }) },
    { key: 'eurosat', value: percent(0.1908, lang), text: t('home.statEuroSat') },
    { key: 'classes', value: '7', text: t('home.statClasses') },
    { key: 'patches', value: patchCount[lang], text: t('home.statPatches') },
  ]

  return (
    <PublicLayout>
      <section className="grid items-center gap-12 lg:grid-cols-2">
        <div>
          <p className="eyebrow flex items-center gap-3">
            <span className="h-px w-6 bg-primary" /> {t('home.eyebrow')}
          </p>
          <h1 className="mt-6 text-5xl leading-tight md:text-7xl">{t('tagline')}</h1>
          <p className="mt-8 max-w-xl text-lg text-muted">{t('home.intro')}</p>
          <Link to="/news" className="mt-10 inline-block rounded-lg bg-primary px-6 py-3 font-semibold text-white hover:opacity-90">
            {t('home.readStories')}
          </Link>
        </div>

        <div className="flex gap-4 rounded-2xl border border-border bg-surface p-6">
          <Link
            to="/explorer"
            className="group flex h-96 flex-1 flex-col items-center justify-center gap-3 rounded-xl bg-background text-center"
          >
            <span className="font-heading text-2xl font-light group-hover:text-primary">{t('home.mapTitle')}</span>
            <span className="text-sm text-muted">{t('home.mapText')}</span>
            <span className="mt-2 rounded-lg bg-primary px-5 py-2.5 text-sm font-semibold text-white group-hover:opacity-90">
              {t('home.mapButton')}
            </span>
          </Link>
          <div className="self-end">
            <p className="eyebrow mb-3">{t('home.classes')}</p>
            {landCoverClasses.map((c) => (
              <p key={c.code} className="mb-2 flex items-center gap-2 text-sm">
                <span className="h-3 w-3 rounded-sm" style={{ backgroundColor: c.color }} /> {className(c, lang)}
              </p>
            ))}
          </div>
        </div>
      </section>

      <section className="mt-20 grid border-y border-border sm:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => (
          <div key={stat.key} className="border-border py-8 pr-6 lg:border-r lg:pl-6 lg:first:pl-0 lg:last:border-r-0">
            <p className="font-heading text-5xl font-light">{stat.value}</p>
            <p className="mt-3 text-sm text-muted">{stat.text}</p>
          </div>
        ))}
      </section>
    </PublicLayout>
  )
}