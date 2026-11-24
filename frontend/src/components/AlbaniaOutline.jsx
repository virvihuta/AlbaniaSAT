import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useLanguage } from '../i18n'

const WIDTH = 320

// The 12 regions (qarqe) of Albania, saved in frontend/public as a file.
// Source: geoBoundaries ALB ADM1 (public domain). Because the file is part of the website,
// the map appears at once, even when the backend is still waking up.
const REGIONS_FILE = '/albania-regions.json'

// Cities shown as dots on the map: [name, longitude, latitude]. Tirana is the capital.
const cities = [
  ['Shkodër', 19.5126, 42.0683],
  ['Tiranë', 19.8187, 41.3275],
  ['Durrës', 19.4414, 41.3231],
  ['Korçë', 20.7808, 40.6186],
  ['Vlorë', 19.4914, 40.4686],
]

// Turns longitude/latitude into x/y inside the drawing. Albania is small, so a simple
// projection is enough: the longitude is squeezed by cos(latitude) so the shape is not stretched.
function makeProjection(features) {
  const points = features.flatMap((f) => allPoints(f.geometry))
  const lons = points.map((p) => p[0])
  const lats = points.map((p) => p[1])
  const minLon = Math.min(...lons)
  const maxLon = Math.max(...lons)
  const minLat = Math.min(...lats)
  const maxLat = Math.max(...lats)
  const squeeze = Math.cos((((minLat + maxLat) / 2) * Math.PI) / 180)
  const scale = WIDTH / ((maxLon - minLon) * squeeze)
  const height = (maxLat - minLat) * scale
  return {
    height,
    minLat,
    maxLat,
    minLon,
    maxLon,
    x: (lon) => (lon - minLon) * squeeze * scale,
    y: (lat) => (maxLat - lat) * scale,
  }
}

// All [lon, lat] points of a Polygon or MultiPolygon
function allPoints(geometry) {
  if (geometry.type === 'Polygon') return geometry.coordinates.flat()
  if (geometry.type === 'MultiPolygon') return geometry.coordinates.flat(2)
  return []
}

// SVG path ("M x y L x y ... Z") of a Polygon or MultiPolygon
function toPath(geometry, proj) {
  const polygons = geometry.type === 'Polygon' ? [geometry.coordinates] : geometry.coordinates
  return polygons
    .flat()
    .map((ring) => 'M' + ring.map(([lon, lat]) => `${proj.x(lon).toFixed(1)} ${proj.y(lat).toFixed(1)}`).join('L') + 'Z')
    .join('')
}

// Outline of Albania with its regions for the start page.
// A red line scans over the country like a satellite. Hover shows the region name, a click opens the explorer.
export default function AlbaniaOutline() {
  const { t } = useLanguage()
  const navigate = useNavigate()
  const [regions, setRegions] = useState(null)
  const [hovered, setHovered] = useState(null)

  useEffect(() => {
    fetch(REGIONS_FILE)
      .then((res) => res.json())
      .then((data) => setRegions(data.features.filter((f) => f.geometry)))
      .catch(() => setRegions([]))
  }, [])

  const map = useMemo(() => {
    if (!regions || regions.length === 0) return null
    const proj = makeProjection(regions)
    const shapes = regions.map((f, i) => ({
      id: i,
      name: f.properties.shapeName || f.properties.name,
      d: toPath(f.geometry, proj),
    }))
    const dots = cities.map(([name, lon, lat]) => ({ name, x: proj.x(lon), y: proj.y(lat) }))
    // Grid lines every full degree, like on a real map
    const lats = []
    for (let lat = Math.ceil(proj.minLat); lat <= proj.maxLat; lat++) lats.push({ value: lat, y: proj.y(lat) })
    const lons = []
    for (let lon = Math.ceil(proj.minLon); lon <= proj.maxLon; lon++) lons.push({ value: lon, x: proj.x(lon) })
    return { height: proj.height, shapes, dots, lats, lons }
  }, [regions])

  if (!map) {
    // Space holder while the regions load, so the page does not jump
    return <div className="aspect-[1/2] w-full max-w-[16rem] animate-pulse rounded-3xl bg-white/5 sm:max-w-sm lg:max-w-md" />
  }

  const pad = 28
  const viewBox = `${-pad} ${-pad} ${WIDTH + pad * 2} ${map.height + pad * 2}`

  return (
    <div className="relative w-full max-w-[16rem] sm:max-w-sm lg:max-w-md">
      <svg viewBox={viewBox} className="h-auto w-full overflow-visible" role="img" aria-label="Albania">
        <defs>
          <clipPath id="albania-clip">
            {map.shapes.map((s) => (
              <path key={s.id} d={s.d} />
            ))}
          </clipPath>
          <linearGradient id="scan-gradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#bf1e2d" stopOpacity="0" />
            <stop offset="0.85" stopColor="#bf1e2d" stopOpacity="0.35" />
            <stop offset="1" stopColor="#ff4d5e" stopOpacity="0.9" />
          </linearGradient>
        </defs>

        {/* Grid with degree labels */}
        <g className="text-white/25" stroke="currentColor" strokeWidth="0.5" strokeDasharray="2 4">
          {map.lats.map((l) => (
            <line key={'lat' + l.value} x1={-pad} x2={WIDTH + pad} y1={l.y} y2={l.y} />
          ))}
          {map.lons.map((l) => (
            <line key={'lon' + l.value} y1={-pad} y2={map.height + pad} x1={l.x} x2={l.x} />
          ))}
        </g>
        <g className="fill-white/40 font-sans" fontSize="9" letterSpacing="1">
          {map.lats.map((l) => (
            <text key={'latl' + l.value} x={WIDTH + pad - 2} y={l.y - 4} textAnchor="end">{l.value}°N</text>
          ))}
          {map.lons.map((l) => (
            <text key={'lonl' + l.value} x={l.x + 4} y={map.height + pad - 4}>{l.value}°E</text>
          ))}
        </g>

        {/* Regions */}
        {map.shapes.map((s, i) => (
          <path
            key={s.id}
            d={s.d}
            className="region-in cursor-pointer transition-colors duration-200"
            style={{ animationDelay: `${i * 60}ms` }}
            fill={hovered?.id === s.id ? 'rgba(191, 30, 45, 0.55)' : 'rgba(255, 255, 255, 0.06)'}
            stroke="rgba(255, 255, 255, 0.55)"
            strokeWidth="0.8"
            strokeLinejoin="round"
            onMouseEnter={() => setHovered(s)}
            onMouseLeave={() => setHovered(null)}
            onClick={() => navigate('/explorer')}
          />
        ))}

        {/* Sea name on the left, like on a printed map */}
        <text
          x="-12"
          y={map.height * 0.62}
          className="fill-white/30 font-sans"
          fontSize="10"
          letterSpacing="4"
          fontStyle="italic"
          transform={`rotate(-90 -12 ${map.height * 0.62})`}
          textAnchor="middle"
        >
          ADRIATIC SEA
        </text>

        {/* Cities. Tirana as the capital gets a red pulsing dot. */}
        <g pointerEvents="none" className="font-sans">
          {map.dots.map((c) => {
            const capital = c.name === 'Tiranë'
            return (
              <g key={c.name}>
                {capital && <circle cx={c.x} cy={c.y} r="9" className="city-pulse" fill="#ff5a67" />}
                <circle cx={c.x} cy={c.y} r={capital ? 3.5 : 2.5} fill={capital ? '#ff5a67' : '#fff'} />
                <text
                  x={c.x + 7}
                  y={c.y + 3.5}
                  fontSize={capital ? 11 : 9.5}
                  fontWeight={capital ? 600 : 400}
                  className={capital ? 'fill-white' : 'fill-white/70'}
                >
                  {c.name}
                </text>
              </g>
            )
          })}
        </g>

        {/* The scan line, only visible inside the country */}
        <g clipPath="url(#albania-clip)" pointerEvents="none">
          <rect
            className="scan"
            x={-pad}
            y="-70"
            width={WIDTH + pad * 2}
            height="70"
            fill="url(#scan-gradient)"
            style={{ '--scan-distance': `${map.height + 70}px` }}
          />
        </g>
      </svg>

      <p className="mt-2 text-center text-xs uppercase tracking-[0.2em] text-white/50">
        {hovered ? <span className="text-white">{hovered.name}</span> : t('home.mapHint')}
      </p>
    </div>
  )
}