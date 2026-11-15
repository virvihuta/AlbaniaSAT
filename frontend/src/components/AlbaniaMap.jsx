import { useEffect, useRef } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

// The whole country, used when no region is selected
const ALBANIA = [[39.6, 19.2], [42.7, 21.1]]

// Background maps without labels clutter, one for light and one for dark mode
const TILES = {
  light: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
  dark: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
}

function currentTheme() {
  return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light'
}

// Leaflet map of Albania: region borders and the patches of the selected region as colored dots.
// The page decides what is selected, this component only draws it.
export default function AlbaniaMap({ regions, patches, selectedRegionId, selectedPatchId, colorOf, onSelectRegion, onSelectPatch }) {
  const boxRef = useRef(null)
  const mapRef = useRef(null)
  const regionLayerRef = useRef(null)
  const patchLayerRef = useRef(null)

  // 1. Create the map once
  useEffect(() => {
    const map = L.map(boxRef.current, { preferCanvas: true, zoomControl: false, minZoom: 6 })
    map.fitBounds(ALBANIA)
    L.control.zoom({ position: 'topright' }).addTo(map)

    const tiles = L.tileLayer(TILES[currentTheme()], {
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
      subdomains: 'abcd',
      maxZoom: 18,
    }).addTo(map)

    // Switch the background map when the theme button is clicked
    const observer = new MutationObserver(() => tiles.setUrl(TILES[currentTheme()]))
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })

    mapRef.current = map
    return () => {
      observer.disconnect()
      map.remove()
    }
  }, [])

  // 2. Draw the region borders. The selected region gets a red border.
  useEffect(() => {
    const map = mapRef.current
    if (!map || !regions) return
    regionLayerRef.current?.remove()

    const red = getComputedStyle(document.documentElement).getPropertyValue('--color-primary').trim() || '#bf1e2d'
    const style = (feature) => {
      const selected = feature.properties.id === selectedRegionId
      return { color: red, weight: selected ? 2.5 : 1, opacity: selected ? 1 : 0.7, fillColor: red, fillOpacity: selected ? 0.04 : 0.02 }
    }

    const layer = L.geoJSON(regions, {
      style,
      onEachFeature: (feature, shape) => {
        shape.bindTooltip(feature.properties.name, { sticky: true, className: 'map-tooltip' })
        shape.on('click', () => onSelectRegion(feature.properties.id))
        shape.on('mouseover', () => shape.setStyle({ weight: 2.5, opacity: 1 }))
        shape.on('mouseout', () => layer.resetStyle(shape))
      },
    }).addTo(map)
    regionLayerRef.current = layer

    // Zoom to the selected region, or show all of Albania
    const selected = layer.getLayers().find((shape) => shape.feature.properties.id === selectedRegionId)
    if (selected) map.flyToBounds(selected.getBounds(), { padding: [30, 30], duration: 0.6 })
    else map.flyToBounds(ALBANIA, { duration: 0.6 })
  }, [regions, selectedRegionId, onSelectRegion])

  // 3. Draw the patches as small dots in the color of their predicted class
  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    patchLayerRef.current?.remove()
    if (!patches) return

    const red = getComputedStyle(document.documentElement).getPropertyValue('--color-primary').trim() || '#bf1e2d'
    const layer = L.geoJSON(patches, {
      pointToLayer: (feature, latlng) => {
        const selected = feature.properties.id === selectedPatchId
        return L.circleMarker(latlng, {
          radius: selected ? 8 : 5,
          color: selected ? red : '#ffffff',
          weight: selected ? 3 : 0.8,
          fillColor: colorOf(feature.properties.predicted_class_id),
          fillOpacity: 0.95,
        })
      },
      onEachFeature: (feature, dot) => dot.on('click', () => onSelectPatch(feature.properties.id)),
    }).addTo(map)
    patchLayerRef.current = layer
  }, [patches, selectedPatchId, colorOf, onSelectPatch])

  return <div ref={boxRef} className="isolate h-full w-full" />
}