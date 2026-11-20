import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { getJSON } from '../api'
import PublicLayout from '../components/PublicLayout'
import AlbaniaMap from '../components/AlbaniaMap'
import { landCoverClasses, className } from '../landCover'
import { useLanguage } from '../i18n'

// Reads a color from index.css, e.g. "var(--color-lc-water)" -> "#2b7fd4" (Leaflet needs real colors)
function realColor(cssVar) {
  const name = cssVar.slice(4, -1)
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#999999'
}

// 820 -> "0.8 s" in English, "0,8 s" in Albanian and German
function seconds(ms, lang) {
  const text = (ms / 1000).toFixed(1).replace('.0', '') + ' s'
  return lang === 'en' ? text : text.replace('.', ',')
}

function percent(value) {
  return value == null ? '–' : Math.round(value * 100) + '%'
}

// Public map page: click a region to load its patches, click a patch to see the image and the prediction
export default function Explorer() {
  const { lang, t } = useLanguage()
  const [regions, setRegions] = useState(null)
  const [classIds, setClassIds] = useState({}) // database id -> class code, from GET /classes
  const [years, setYears] = useState([])
  const [year, setYear] = useState(null)
  const [regionId, setRegionId] = useState(null)
  const [patches, setPatches] = useState(null)
  const [summary, setSummary] = useState(null)
  const [patchId, setPatchId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadTime, setLoadTime] = useState(null) // { ms, cached } of the last region load
  const [error, setError] = useState('')
  // Regions that were already loaded: "regionId-year" -> { patches, summary, ms }. A second click is instant.
  const cache = useRef(new Map())

  // Regions, classes and years are loaded once
  useEffect(() => {
    getJSON('/regions/geojson').then(setRegions).catch(() => setError('explorer.mapError'))
    getJSON('/classes')
      .then((list) => setClassIds(Object.fromEntries(list.map((c) => [c.id, c.code]))))
      .catch(() => setClassIds({}))
    // Years with region results. As long as there are none, take the years of the patches.
    getJSON('/years')
      .then((list) => (list.length > 0 ? list : getJSON('/patches/years')))
      .then((list) => {
        setYears(list)
        setYear(list[list.length - 1] ?? null)
      })
      .catch(() => setYears([]))
  }, [])

  // Patches and summary of the selected region (and year, if the API has years).
  // Each region is loaded only once and then taken from the cache. The load time is measured for the panel.
  useEffect(() => {
    setPatchId(null)
    setPatches(null)
    setSummary(null)
    setLoadTime(null)
    if (!regionId) return

    const key = `${regionId}-${year ?? 'all'}`
    const saved = cache.current.get(key)
    if (saved) {
      setPatches(saved.patches)
      setSummary(saved.summary)
      setLoadTime({ ms: saved.ms, cached: true })
      return
    }

    let cancelled = false // true when another region is chosen before this one has loaded
    const start = performance.now()
    setLoading(true)
    const yearFilter = year ? `&year=${year}` : ''
    const patchRequest = getJSON(`/patches/geojson?region_id=${regionId}${yearFilter}`)
    const summaryRequest = year ? getJSON(`/classification/${regionId}/${year}`).catch(() => null) : Promise.resolve(null)

    Promise.all([patchRequest, summaryRequest])
      .then(([patchData, summaryData]) => {
        const ms = performance.now() - start
        cache.current.set(key, { patches: patchData, summary: summaryData, ms })
        if (cancelled) return
        setPatches(patchData)
        setSummary(summaryData)
        setLoadTime({ ms, cached: false })
      })
      .catch(() => !cancelled && setError('explorer.patchError'))
      .finally(() => !cancelled && setLoading(false))

    return () => {
      cancelled = true
    }
  }, [regionId, year])

  // Class of a prediction: database id -> code -> our class with name and color
  const classOf = useCallback(
    (id) => landCoverClasses.find((c) => c.code === classIds[id]) || null,
    [classIds],
  )
  const colorOf = useCallback((id) => realColor(classOf(id)?.color || 'var(--color-muted)'), [classOf])

  const selectRegion = useCallback((id) => {
    setError('')
    setRegionId(id)
  }, [])
  const selectPatch = useCallback((id) => setPatchId(id), [])

  const regionNames = useMemo(
    () => (regions ? regions.features.map((f) => f.properties).sort((a, b) => a.name.localeCompare(b.name)) : []),
    [regions],
  )
  const regionName = regionNames.find((r) => r.id === regionId)?.name

  // Share of each class among the patches of the region
  const shares = useMemo(() => {
    if (!patches) return []
    const total = patches.features.length
    return landCoverClasses
      .map((c) => {
        const count = patches.features.filter((f) => classOf(f.properties.predicted_class_id)?.code === c.code).length
        return { ...c, count, share: total ? count / total : 0 }
      })
      .sort((a, b) => b.count - a.count)
  }, [patches, classOf])

  // Average model confidence of the loaded patches (only patches with a prediction)
  const patchConfidence = useMemo(() => {
    const values = (patches?.features || []).map((f) => f.properties.confidence).filter((v) => v != null)
    return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null
  }, [patches])

  const patch = patches?.features.find((f) => f.properties.id === patchId)?.properties

  return (
    <PublicLayout full>
      <div className="flex flex-col lg:h-[calc(100vh-6rem)] lg:flex-row">
        {/* Map with the region list, legend and years on top */}
        <div className="relative h-[60vh] flex-1 lg:h-full">
          <AlbaniaMap
            regions={regions}
            patches={patches}
            selectedRegionId={regionId}
            selectedPatchId={patchId}
            colorOf={colorOf}
            onSelectRegion={selectRegion}
            onSelectPatch={selectPatch}
          />

          <div className="absolute left-4 top-4 z-[1000] flex flex-col gap-3">
            <select
              value={regionId ?? ''}
              onChange={(e) => selectRegion(e.target.value ? Number(e.target.value) : null)}
              className="rounded-lg border border-border bg-surface px-4 py-2.5 text-sm font-medium shadow-sm outline-none focus:border-primary"
            >
              <option value="">{t('explorer.allAlbania')}</option>
              {regionNames.map((r) => (
                <option key={r.id} value={r.id}>{r.name}</option>
              ))}
            </select>

            <div className="hidden rounded-xl border border-border bg-surface p-4 shadow-sm sm:block">
              <p className="eyebrow mb-3">{t('explorer.landCover')}{year ? ' ' + year : ''}</p>
              {landCoverClasses.map((c) => (
                <p key={c.code} className="mb-1.5 flex items-center gap-2 text-sm">
                  <span className="h-3 w-3 rounded-sm" style={{ backgroundColor: c.color }} /> {className(c, lang)}
                </p>
              ))}
            </div>
          </div>

          {years.length > 0 && (
            <div className="absolute bottom-6 left-4 z-[1000] flex items-center gap-2 rounded-xl border border-border bg-surface p-2 shadow-sm">
              <span className="px-2 text-xs font-semibold uppercase tracking-wider text-muted">{t('explorer.year')}</span>
              {years.map((y) => (
                <button
                  key={y}
                  onClick={() => setYear(y)}
                  className={'rounded-lg px-3 py-1.5 text-sm font-medium ' + (y === year ? 'bg-primary text-white' : 'hover:bg-background')}
                >
                  {y}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Panel on the right: what is selected */}
        <aside className="w-full overflow-y-auto border-l border-border bg-surface p-8 lg:w-[26rem]">
          {error && <p className="mb-6 text-sm text-primary">{t(error)}</p>}

          {!regionId && (
            <>
              <p className="eyebrow">{t('nav.explorer')}</p>
              <h1 className="mt-3 text-4xl">{t('explorer.introTitle')}</h1>
              <p className="mt-4 text-muted">{t('explorer.introText')}</p>
              <p className="mt-6 text-sm text-muted">{t('explorer.introHint')}</p>
            </>
          )}

          {regionId && !patch && (
            <>
              <p className="eyebrow">{t('explorer.selected')}</p>
              <h1 className="mt-3 text-4xl">{regionName}</h1>
              <p className="mt-2 text-sm text-muted">
                {year ? year + ' · ' : ''}
                {loading ? t('explorer.loadingPatches') : t('explorer.patches', { count: patches?.features.length ?? 0 })}
              </p>
              {loadTime && (
                <p className="mt-1 text-xs text-muted">
                  {loadTime.cached
                    ? t('explorer.fromCache', { time: seconds(loadTime.ms, lang) })
                    : t('explorer.loadedIn', { time: seconds(loadTime.ms, lang) })}
                </p>
              )}

              {summary?.mean_confidence != null ? (
                <div className="mt-6 rounded-xl bg-background p-4">
                  <p className="text-xs uppercase tracking-wider text-muted">{t('explorer.avgConfidence')}</p>
                  <p className="mt-1 font-heading text-3xl font-light">{percent(summary.mean_confidence)}</p>
                </div>
              ) : (
                patchConfidence != null && (
                  <div className="mt-6 rounded-xl bg-background p-4">
                    <p className="text-xs uppercase tracking-wider text-muted">{t('explorer.sampleConfidence')}</p>
                    <p className="mt-1 font-heading text-3xl font-light">{percent(patchConfidence)}</p>
                  </div>
                )
              )}

              {patches && patches.features.length > 0 && (
                <div className="mt-8">
                  <p className="eyebrow mb-4">{t('explorer.byClass')}</p>
                  {shares.map((c) => (
                    <div key={c.code} className="mb-3">
                      <div className="flex justify-between text-sm">
                        <span className="flex items-center gap-2">
                          <span className="h-3 w-3 rounded-sm" style={{ backgroundColor: c.color }} /> {className(c, lang)}
                        </span>
                        <span className="text-muted">{c.count}</span>
                      </div>
                      <div className="mt-1.5 h-1.5 rounded-full bg-background">
                        <div className="h-full rounded-full" style={{ width: c.share * 100 + '%', backgroundColor: c.color }} />
                      </div>
                    </div>
                  ))}
                  <p className="mt-4 text-xs text-muted">{t('explorer.balancedNote')}</p>
                </div>
              )}

              {summary?.ai_summary && (
                <div className="mt-8 rounded-xl bg-background p-4">
                  <p className="eyebrow">{t('explorer.summary')}</p>
                  <p className="mt-2 text-sm">{summary.ai_summary}</p>
                </div>
              )}

              <button onClick={() => selectRegion(null)} className="mt-8 text-sm font-semibold text-primary hover:underline">
                {t('explorer.backAll')}
              </button>
            </>
          )}

          {patch && (
            <>
              <button onClick={() => setPatchId(null)} className="text-sm font-semibold text-primary hover:underline">
                ← {regionName}
              </button>
              <p className="eyebrow mt-6">{t('explorer.patch', { id: patch.id, year: patch.year })}</p>
              {patch.image_url && (
                <img
                  src={patch.image_url}
                  alt={t('explorer.imageAlt')}
                  className="mt-4 aspect-square w-full rounded-xl bg-background object-cover [image-rendering:pixelated]"
                />
              )}
              <p className="mt-2 text-xs text-muted">Sentinel-2 · 64 × 64 px · 640 × 640 m</p>

              <dl className="mt-6 divide-y divide-border border-y border-border text-sm">
                <div className="flex justify-between py-3">
                  <dt className="text-muted">{t('explorer.prediction')}</dt>
                  <dd className="flex items-center gap-2 font-semibold">
                    <span className="h-3 w-3 rounded-sm" style={{ backgroundColor: classOf(patch.predicted_class_id)?.color }} />
                    {classOf(patch.predicted_class_id) ? className(classOf(patch.predicted_class_id), lang) : t('explorer.noPrediction')}
                  </dd>
                </div>
                <div className="flex justify-between py-3">
                  <dt className="text-muted">{t('explorer.confidence')}</dt>
                  <dd className="font-semibold">{percent(patch.confidence)}</dd>
                </div>
                {patch.model_version && (
                  <div className="flex justify-between py-3">
                    <dt className="text-muted">{t('explorer.version')}</dt>
                    <dd className="font-semibold">{patch.model_version}</dd>
                  </div>
                )}
                <div className="flex justify-between py-3">
                  <dt className="text-muted">{t('explorer.reference')}</dt>
                  <dd className="flex items-center gap-2">
                    <span className="h-3 w-3 rounded-sm" style={{ backgroundColor: classOf(patch.reference_class_id)?.color }} />
                    {classOf(patch.reference_class_id) ? className(classOf(patch.reference_class_id), lang) : t('explorer.none')}
                  </dd>
                </div>
              </dl>

              {patch.reference_class_id != null && patch.predicted_class_id != null && (
                <p className={'mt-4 text-sm ' + (patch.reference_class_id === patch.predicted_class_id ? 'text-lc-broadleaf' : 'text-primary')}>
                  {patch.reference_class_id === patch.predicted_class_id
                    ? t('explorer.match')
                    : t('explorer.mismatch')}
                </p>
              )}
            </>
          )}
        </aside>
      </div>
    </PublicLayout>
  )
}