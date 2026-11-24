import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getJSON } from '../api'
import { useLanguage } from '../i18n'
import PublicLayout from '../components/PublicLayout'
import Contours from '../components/Contours'
import AlbaniaOutline from '../components/AlbaniaOutline'
import { CountUp, Reveal, useInView } from '../components/Reveal'
import { Cover, Author } from '../components/ArticleParts'
import { landCoverClasses, className } from '../landCover'

const locales = { en: 'en-GB', sq: 'sq-AL', de: 'de-DE' }
const EUROSAT_ACCURACY = 0.1908

// Number in the style of the language: 80.94 in English, 80,94 in Albanian and German
function number(value, lang, decimals = 0) {
  return new Intl.NumberFormat(locales[lang], { minimumFractionDigits: decimals, maximumFractionDigits: decimals }).format(value)
}

// Small line icons for the cards and steps
const icons = {
  satellite: (
    <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="9" width="6" height="6" rx="1" />
      <rect x="16" y="9" width="6" height="6" rx="1" />
      <rect x="9.5" y="8.5" width="5" height="7" rx="1" />
      <path d="M8 12h1.5M14.5 12H16M12 8.5V6M9.5 4a3.5 3.5 0 0 1 5 0" />
    </svg>
  ),
  model: (
    <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <rect x="6" y="6" width="12" height="12" rx="2" />
      <path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4M10 10h4v4h-4z" />
    </svg>
  ),
  city: (
    <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 21h18M5 21V9l5-3v15M10 21V3l9 4v14M13 9h3M13 13h3M13 17h3M7 12h1M7 16h1" />
    </svg>
  ),
  fire: (
    <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 21c-3.9 0-7-2.8-7-6.5 0-3.7 3-5.6 4-9.5 2 1.5 3 3.5 3 5.5 1-1 1.5-2.2 1.5-3.5 2.6 2 4.5 4.6 4.5 7.5 0 3.7-2.6 6.5-6 6.5z" />
      <path d="M12 21c-1.7 0-3-1.2-3-2.8 0-1.8 1.5-2.6 2-4.2 1.4 1 3 2.4 3 4.2 0 1.6-.9 2.8-2 2.8z" />
    </svg>
  ),
  field: (
    <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 21V11M12 11c0-3 2-5 5-5 0 3-2 5-5 5zM12 14c0-2.5-1.8-4.5-4.5-4.5 0 2.5 1.8 4.5 4.5 4.5zM3 21h18" />
    </svg>
  ),
  map: (
    <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 4 3 6v14l6-2 6 2 6-2V4l-6 2-6-2zM9 4v14M15 6v14" />
    </svg>
  ),
}

// Bar that grows to its value when it comes into view
function GrowBar({ value, color, label, text }) {
  const [ref, inView] = useInView()
  return (
    <div ref={ref}>
      <div className="flex items-baseline justify-between gap-4">
        <span className="text-sm font-medium">{label}</span>
        <span className="font-heading text-3xl font-light">{text}</span>
      </div>
      <div className="mt-3 h-3 overflow-hidden rounded-full bg-background">
        <div
          className="h-full rounded-full transition-[width] duration-[1600ms] ease-out"
          style={{ width: inView ? value * 100 + '%' : '0%', backgroundColor: color }}
        />
      </div>
    </div>
  )
}

// Red line and small grey label above a section title
function Eyebrow({ children }) {
  return (
    <p className="eyebrow flex items-center gap-3">
      <span className="h-px w-6 bg-primary" /> {children}
    </p>
  )
}

// Public start page: hero with the map of Albania, key numbers, why the maps matter,
// why the model is better, how it works, data and method, the seven classes,
// who it is for, the newest stories and a last call to open the map.
export default function Home() {
  const { lang, t } = useLanguage()
  // Values of the current model. Used until the API answers, or if it cannot be reached.
  const [model, setModel] = useState({ accuracy: 0.8094, f1_macro: 0.8138 })
  const [stories, setStories] = useState([])

  useEffect(() => {
    // Accuracy and F1 come from the API, so they update when there is a new model
    getJSON('/metrics')
      .then((list) => {
        const newest = list.reduce((a, b) => (b.id > a.id ? b : a), list[0])
        if (newest) setModel(newest)
      })
      .catch(() => {})
    getJSON('/articles?page=1&page_size=3')
      .then((data) => setStories(data.items || []))
      .catch(() => setStories([]))
  }, [])

  // "Europe couldn't see Albania. Now it can." -> the second sentence is red
  const [taglineStart, ...taglineEnd] = t('tagline').split(/(?<=\.)\s+/)
  const times = number(model.accuracy / EUROSAT_ACCURACY, lang, 1)

  const stats = [
    { key: 'accuracy', value: model.accuracy * 100, format: (v) => number(v, lang, 2) + ' %', text: t('home.statAccuracyShort') },
    { key: 'f1', value: model.f1_macro * 100, format: (v) => number(v, lang, 2) + ' %', text: t('home.statF1') },
    { key: 'classes', value: 7, format: (v) => number(v, lang), text: t('home.statClasses') },
    { key: 'patches', value: 7800, format: (v) => '~' + number(Math.round(v / 100) * 100, lang), text: t('home.statPatches') },
  ]

  const changes = [
    { icon: icons.city, title: t('home.change1Title'), text: t('home.change1Text') },
    { icon: icons.fire, title: t('home.change2Title'), text: t('home.change2Text') },
    { icon: icons.field, title: t('home.change3Title'), text: t('home.change3Text') },
  ]

  const facts = [
    [t('home.factSatellite'), t('home.factSatelliteValue')],
    [t('home.factResolution'), t('home.factResolutionValue')],
    [t('home.factPatch'), t('home.factPatchValue')],
    [t('home.factLabels'), t('home.factLabelsValue')],
    [t('home.factYears'), t('home.factYearsValue')],
    [t('home.factModel'), model.label || 'AlbaniaSAT-SoftCon-7class-v1'],
  ]

  const users = [
    { title: t('home.user1Title'), text: t('home.user1Text') },
    { title: t('home.user2Title'), text: t('home.user2Text') },
    { title: t('home.user3Title'), text: t('home.user3Text') },
    { title: t('home.user4Title'), text: t('home.user4Text') },
  ]

  const steps = [
    { icon: icons.satellite, title: t('home.step1Title'), text: t('home.step1Text') },
    { icon: icons.model, title: t('home.step2Title'), text: t('home.step2Text') },
    { icon: icons.map, title: t('home.step3Title'), text: t('home.step3Text') },
  ]

  return (
    <PublicLayout wide>
      {/* 1. Hero: dark like a satellite photo at night, with the map of Albania */}
      <section className="relative overflow-hidden bg-neutral-950 text-white">
        <Contours className="opacity-[0.12]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_85%_30%,rgba(191,30,45,0.35),transparent_55%)]" />
        <div className="relative mx-auto grid max-w-7xl items-center gap-16 px-6 pt-20 pb-36 md:px-10 lg:grid-cols-[1.25fr_1fr] lg:pt-24">
          <div className="hero-in">
            <p className="flex items-center gap-3 text-xs font-medium uppercase tracking-[0.2em] text-white/60">
              <span className="h-px w-8 bg-primary" /> {t('home.heroBadge')}
            </p>
            <h1 className="mt-8 text-5xl leading-[1.05] tracking-tight md:text-7xl">
              {taglineStart} <span className="font-normal text-[#ff5a67]">{taglineEnd.join(' ')}</span>
            </h1>
            <p className="mt-8 max-w-xl text-lg leading-relaxed text-white/70">{t('home.intro')}</p>
            <div className="mt-10 flex flex-wrap gap-4">
              <Link
                to="/explorer"
                className="rounded-lg bg-primary px-6 py-3.5 font-semibold text-white shadow-lg shadow-primary/30 transition hover:-translate-y-0.5 hover:shadow-primary/50"
              >
                {t('home.mapButton')}
              </Link>
              <Link
                to="/news"
                className="rounded-lg border border-white/25 px-6 py-3.5 font-semibold text-white transition hover:border-white hover:bg-white/5"
              >
                {t('home.readStories')}
              </Link>
            </div>
          </div>
          <div className="hero-in flex justify-center [animation-delay:200ms] lg:justify-end">
            <AlbaniaOutline />
          </div>
        </div>
      </section>

      {/* 2. Key numbers on a card that overlaps the hero */}
      <section className="relative mx-auto -mt-20 max-w-7xl px-6 md:px-10">
        <div className="grid rounded-2xl border border-border bg-surface shadow-xl shadow-black/5 sm:grid-cols-2 lg:grid-cols-4">
          {stats.map((stat, i) => (
            <div
              key={stat.key}
              className={
                'border-border p-8 ' +
                (i > 0 ? 'border-t sm:border-t-0 ' : '') +
                (i % 2 === 1 ? 'sm:border-l ' : '') +
                (i >= 2 ? 'sm:border-t lg:border-t-0 ' : '') +
                (i === 2 ? 'lg:border-l' : '')
              }
            >
              <p className="font-heading text-4xl font-light md:text-5xl">
                <CountUp value={stat.value} format={stat.format} />
              </p>
              <p className="mt-3 text-sm text-muted">{stat.text}</p>
            </div>
          ))}
        </div>
      </section>

      {/* 3. Why it matters: Albania is changing and the maps are old */}
      <section className="mx-auto max-w-7xl px-6 pt-28 md:px-10">
        <Reveal className="max-w-3xl">
          <Eyebrow>{t('home.changeEyebrow')}</Eyebrow>
          <h2 className="mt-5 text-4xl leading-tight md:text-5xl">{t('home.changeTitle')}</h2>
          <p className="mt-6 text-lg leading-relaxed text-muted">{t('home.changeText')}</p>
        </Reveal>
        <div className="mt-14 grid gap-6 md:grid-cols-3">
          {changes.map((item, i) => (
            <Reveal key={item.title} delay={i * 120} className="h-full">
              <div className="h-full rounded-2xl border border-border bg-surface p-8 transition hover:-translate-y-1 hover:border-primary/40 hover:shadow-lg">
                <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10 text-primary">{item.icon}</div>
                <h3 className="mt-6 text-2xl font-normal">{item.title}</h3>
                <p className="mt-3 leading-relaxed text-muted">{item.text}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      {/* 4. Why AlbaniaSAT: the European model against ours */}
      <section className="mx-auto grid max-w-7xl items-center gap-16 px-6 py-28 md:px-10 lg:grid-cols-2">
        <Reveal>
          <Eyebrow>{t('home.gapEyebrow')}</Eyebrow>
          <h2 className="mt-5 text-4xl leading-tight md:text-5xl">{t('home.gapTitle')}</h2>
          <p className="mt-6 max-w-lg text-lg leading-relaxed text-muted">{t('home.gapText', { times })}</p>
        </Reveal>
        <Reveal delay={150} className="rounded-2xl border border-border bg-surface p-8 md:p-10">
          <div className="space-y-10">
            <GrowBar
              value={EUROSAT_ACCURACY}
              color="var(--color-muted)"
              label={t('home.gapEuroSat')}
              text={number(EUROSAT_ACCURACY * 100, lang, 2) + ' %'}
            />
            <GrowBar
              value={model.accuracy}
              color="var(--color-primary)"
              label={t('home.gapOurs')}
              text={number(model.accuracy * 100, lang, 2) + ' %'}
            />
          </div>
          <p className="mt-8 border-t border-border pt-5 text-xs uppercase tracking-[0.15em] text-muted">{t('home.gapNote')}</p>
        </Reveal>
      </section>

      {/* 5. How it works in three steps */}
      <section className="border-y border-border bg-surface">
        <div className="mx-auto max-w-7xl px-6 py-28 md:px-10">
          <Reveal>
            <Eyebrow>{t('home.howEyebrow')}</Eyebrow>
            <h2 className="mt-5 text-4xl md:text-5xl">{t('home.howTitle')}</h2>
          </Reveal>
          <div className="relative mt-16 grid gap-12 md:grid-cols-3 md:gap-8">
            {/* Line that connects the three steps */}
            <div className="absolute top-7 right-[16%] left-[16%] hidden h-px bg-gradient-to-r from-transparent via-border to-transparent md:block" />
            {steps.map((step, i) => (
              <Reveal key={step.title} delay={i * 150} className="relative">
                <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-border bg-background text-primary">
                  {step.icon}
                </div>
                <p className="mt-6 font-heading text-sm text-muted">0{i + 1}</p>
                <h3 className="mt-1 text-2xl font-normal">{step.title}</h3>
                <p className="mt-3 leading-relaxed text-muted">{step.text}</p>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* 6. Data and method in a short fact list */}
      <section className="mx-auto grid max-w-7xl gap-16 px-6 pt-28 md:px-10 lg:grid-cols-[1fr_1.2fr]">
        <Reveal>
          <Eyebrow>{t('home.dataEyebrow')}</Eyebrow>
          <h2 className="mt-5 text-4xl leading-tight md:text-5xl">{t('home.dataTitle')}</h2>
          <p className="mt-6 max-w-md text-lg leading-relaxed text-muted">{t('home.dataText')}</p>
        </Reveal>
        <Reveal delay={150}>
          <dl className="divide-y divide-border border-y border-border">
            {facts.map(([label, value]) => (
              <div key={label} className="grid gap-1 py-5 sm:grid-cols-[12rem_1fr] sm:gap-6">
                <dt className="text-sm text-muted">{label}</dt>
                <dd className="font-medium">{value}</dd>
              </div>
            ))}
          </dl>
        </Reveal>
      </section>

      {/* 7. The seven classes in their map colors */}
      <section className="mx-auto max-w-7xl px-6 py-28 md:px-10">
        <Reveal className="max-w-2xl">
          <Eyebrow>{t('home.classes')}</Eyebrow>
          <h2 className="mt-5 text-4xl md:text-5xl">{t('home.classesTitle')}</h2>
          <p className="mt-6 text-lg text-muted">{t('home.classesText')}</p>
        </Reveal>
        <div className="mt-14 grid grid-cols-2 gap-4 sm:grid-cols-4 lg:grid-cols-7">
          {landCoverClasses.map((c, i) => (
            <Reveal key={c.code} delay={i * 70}>
              <div className="group h-full overflow-hidden rounded-2xl border border-border bg-surface transition hover:-translate-y-1 hover:shadow-lg">
                <div className="h-28" style={{ backgroundColor: c.color }} />
                <p className="p-4 text-sm font-medium">{className(c, lang)}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      {/* 8. Who it is for */}
      <section className="border-t border-border bg-surface">
        <div className="mx-auto max-w-7xl px-6 py-28 md:px-10">
          <Reveal className="max-w-2xl">
            <Eyebrow>{t('home.usersEyebrow')}</Eyebrow>
            <h2 className="mt-5 text-4xl leading-tight md:text-5xl">{t('home.usersTitle')}</h2>
          </Reveal>
          <div className="mt-14 grid gap-px overflow-hidden rounded-2xl border border-border bg-border sm:grid-cols-2 lg:grid-cols-4">
            {users.map((user, i) => (
              <Reveal key={user.title} delay={i * 100} className="h-full">
                <div className="h-full bg-surface p-8">
                  <p className="font-heading text-sm text-primary">0{i + 1}</p>
                  <h3 className="mt-3 text-xl font-normal">{user.title}</h3>
                  <p className="mt-3 text-sm leading-relaxed text-muted">{user.text}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* 9. Newest stories (only if there are any) */}
      {stories.length > 0 && (
        <section className="border-t border-border bg-surface">
          <div className="mx-auto max-w-7xl px-6 py-28 md:px-10">
            <div className="flex flex-wrap items-end justify-between gap-6">
              <Reveal>
                <Eyebrow>{t('nav.story')}</Eyebrow>
                <h2 className="mt-5 text-4xl md:text-5xl">{t('home.storiesTitle')}</h2>
              </Reveal>
              <Link to="/news" className="font-semibold text-primary hover:underline">{t('news.allStories')}</Link>
            </div>
            <div className="mt-12 grid gap-6 md:grid-cols-3">
              {stories.map((article, i) => (
                <Reveal key={article.id} delay={i * 120} className="h-full">
                  <Link
                    to={`/news/${article.slug}`}
                    className="group block h-full overflow-hidden rounded-2xl border border-border bg-background transition hover:-translate-y-1 hover:shadow-lg"
                  >
                    <Cover article={article} className="h-48 w-full" />
                    <div className="space-y-3 p-6">
                      {article.category && <p className="eyebrow text-primary">{t('category.' + article.category)}</p>}
                      <h3 className="text-2xl group-hover:text-primary">{article.title}</h3>
                      {article.summary && <p className="line-clamp-3 text-sm text-muted">{article.summary}</p>}
                      <Author article={article} />
                    </div>
                  </Link>
                </Reveal>
              ))}
            </div>
          </div>
        </section>
      )}

      {/* 10. Last call: open the map */}
      <section className="relative overflow-hidden bg-neutral-950 text-white">
        <Contours className="opacity-[0.12]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_15%_100%,rgba(191,30,45,0.45),transparent_60%)]" />
        <Reveal className="relative mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-8 px-6 py-24 md:px-10">
          <div>
            <h2 className="text-4xl md:text-5xl">{t('home.ctaTitle')}</h2>
            <p className="mt-4 max-w-lg text-lg text-white/70">{t('home.ctaText')}</p>
          </div>
          <Link
            to="/explorer"
            className="rounded-lg bg-white px-7 py-4 font-semibold text-neutral-950 transition hover:-translate-y-0.5 hover:bg-primary hover:text-white"
          >
            {t('home.mapButton')}
          </Link>
        </Reveal>
      </section>
    </PublicLayout>
  )
}