import { useCallback, useEffect, useMemo, useState } from 'react'
import { getJSON } from '../api'
import PublicLayout from '../components/PublicLayout'
import AlbaniaMap from '../components/AlbaniaMap'
import { landCoverClasses } from '../landCover'

// Reads a color from index.css, e.g. "var(--color-lc-water)" -> "#2b7fd4" (Leaflet needs real colors)
function realColor(cssVar) {
  const name = cssVar.slice(4, -1)
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#999999'
}

function percent(value) {
  return value == null ? '–' : Math.round(value * 100) + '%'
}

// Public map page: click a region to load its patches, click a patch to see the image and the prediction
export default function Explorer() {
  const [regions, setRegions] = useState(null)
  const [classIds, setClassIds] = useState({}) // database id -> class code, from GET /classes
  const [years, setYears] = useState([])
  const [year, setYear] = useState(null)
  const [regionId, setRegionId] = useState(null)
  const [patches, setPatches] = useState(null)
  const [summary, setSummary] = useState(null)
  const [patchId, setPatchId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // Regions, classes and years are loaded once
  useEffect(() => {
    getJSON('/regions/geojson').then(setRegions).catch(() => setError('The map data could not be loaded. Please try again in a minute.'))
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

  // Patches and summary of the selected region (and year, if the API has years)
  useEffect(() => {
    setPatches(null)
    setSummary(null)
    setPatchId(null)
    if (!regionId) return
    setLoading(true)
    const yearFilter = year ? `&year=${year}` : ''
    getJSON(`/patches/geojson?region_id=${regionId}${yearFilter}`)
      .then(setPatches)
      .catch(() => setError('The patches could not be loaded.'))
      .finally(() => setLoading(false))
    if (year) getJSON(`/classification/${regionId}/${year}`).then(setSummary).catch(() => setSummary(null))
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
              <option value="">All of Albania</option>
              {regionNames.map((r) => (
                <option key={r.id} value={r.id}>{r.name}</option>
              ))}
            </select>

            <div className="hidden rounded-xl border border-border bg-surface p-4 shadow-sm sm:block">
              <p className="eyebrow mb-3">Land cover{year ? ' ' + year : ''}</p>
              {landCoverClasses.map((c) => (
                <p key={c.code} className="mb-1.5 flex items-center gap-2 text-sm">
                  <span className="h-3 w-3 rounded-sm" style={{ backgroundColor: c.color }} /> {c.name}
                </p>
              ))}
            </div>
          </div>

          {years.length > 0 && (
            <div className="absolute bottom-6 left-4 z-[1000] flex items-center gap-2 rounded-xl border border-border bg-surface p-2 shadow-sm">
              <span className="px-2 text-xs font-semibold uppercase tracking-wider text-muted">Year</span>
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
          {error && <p className="mb-6 text-sm text-primary">{error}</p>}

          {!regionId && (
            <>
              <p className="eyebrow">Explorer</p>
              <h1 className="mt-3 text-4xl">Land cover of Albania</h1>
              <p className="mt-4 text-muted">
                Choose a region on the map or in the list. You will see the sample patches the model classified, each as a dot in the
                color of its class.
              </p>
              <p className="mt-6 text-sm text-muted">Click a dot to see the Sentinel-2 image of that patch and what the model predicted.</p>
            </>
          )}

          {regionId && !patch && (
            <>
              <p className="eyebrow">Selected region</p>
              <h1 className="mt-3 text-4xl">{regionName}</h1>
              <p className="mt-2 text-sm text-muted">
                {year ? year + ' · ' : ''}
                {loading ? 'Loading patches...' : `${patches?.features.length ?? 0} patches`}
              </p>

              {summary?.mean_confidence != null ? (
                <div className="mt-6 rounded-xl bg-background p-4">
                  <p className="text-xs uppercase tracking-wider text-muted">Average model confidence</p>
                  <p className="mt-1 font-heading text-3xl font-light">{percent(summary.mean_confidence)}</p>
                </div>
              ) : (
                patchConfidence != null && (
                  <div className="mt-6 rounded-xl bg-background p-4">
                    <p className="text-xs uppercase tracking-wider text-muted">Model confidence on the sample patches</p>
                    <p className="mt-1 font-heading text-3xl font-light">{percent(patchConfidence)}</p>
                  </div>
                )
              )}

              {patches && patches.features.length > 0 && (
                <div className="mt-8">
                  <p className="eyebrow mb-4">Sample patches by class</p>
                  {shares.map((c) => (
                    <div key={c.code} className="mb-3">
                      <div className="flex justify-between text-sm">
                        <span className="flex items-center gap-2">
                          <span className="h-3 w-3 rounded-sm" style={{ backgroundColor: c.color }} /> {c.name}
                        </span>
                        <span className="text-muted">{c.count}</span>
                      </div>
                      <div className="mt-1.5 h-1.5 rounded-full bg-background">
                        <div className="h-full rounded-full" style={{ width: c.share * 100 + '%', backgroundColor: c.color }} />
                      </div>
                    </div>
                  ))}
                  <p className="mt-4 text-xs text-muted">
                    The dataset has about the same number of patches for every class, so these numbers do not show the real land
                    cover of the region. The full map of the region follows.
                  </p>
                </div>
              )}

              {summary?.ai_summary && (
                <div className="mt-8 rounded-xl bg-background p-4">
                  <p className="eyebrow">Summary · generated from the data</p>
                  <p className="mt-2 text-sm">{summary.ai_summary}</p>
                </div>
              )}

              <button onClick={() => selectRegion(null)} className="mt-8 text-sm font-semibold text-primary hover:underline">
                ← All of Albania
              </button>
            </>
          )}

          {patch && (
            <>
              <button onClick={() => setPatchId(null)} className="text-sm font-semibold text-primary hover:underline">
                ← {regionName}
              </button>
              <p className="eyebrow mt-6">Patch {patch.id} · {patch.year}</p>
              {patch.image_url && (
                <img
                  src={patch.image_url}
                  alt="Sentinel-2 image of this patch"
                  className="mt-4 aspect-square w-full rounded-xl bg-background object-cover [image-rendering:pixelated]"
                />
              )}
              <p className="mt-2 text-xs text-muted">Sentinel-2 · 64 × 64 px · 640 × 640 m</p>

              <dl className="mt-6 divide-y divide-border border-y border-border text-sm">
                <div className="flex justify-between py-3">
                  <dt className="text-muted">Model prediction</dt>
                  <dd className="flex items-center gap-2 font-semibold">
                    <span className="h-3 w-3 rounded-sm" style={{ backgroundColor: classOf(patch.predicted_class_id)?.color }} />
                    {classOf(patch.predicted_class_id)?.name || 'No prediction'}
                  </dd>
                </div>
                <div className="flex justify-between py-3">
                  <dt className="text-muted">Model confidence</dt>
                  <dd className="font-semibold">{percent(patch.confidence)}</dd>
                </div>
                <div className="flex justify-between py-3">
                  <dt className="text-muted">Reference label</dt>
                  <dd className="flex items-center gap-2">
                    <span className="h-3 w-3 rounded-sm" style={{ backgroundColor: classOf(patch.reference_class_id)?.color }} />
                    {classOf(patch.reference_class_id)?.name || 'None'}
                  </dd>
                </div>
              </dl>

              {patch.reference_class_id != null && patch.predicted_class_id != null && (
                <p className={'mt-4 text-sm ' + (patch.reference_class_id === patch.predicted_class_id ? 'text-lc-broadleaf' : 'text-primary')}>
                  {patch.reference_class_id === patch.predicted_class_id
                    ? 'The prediction matches the reference label.'
                    : 'The prediction differs from the reference label.'}
                </p>
              )}
            </>
          )}
        </aside>
      </div>
    </PublicLayout>
  )
}