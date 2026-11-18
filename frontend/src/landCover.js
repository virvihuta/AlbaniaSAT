// The 7 land cover classes of the AlbaniaSAT model.
// "code" is the same as in the database (table land_cover_classes).
// The colors live in the design system (index.css), so we only point to them here.
// name is English, nameSq is Albanian, nameDe is German.
export const landCoverClasses = [
  { code: 'broadleaf_forest', name: 'Broadleaf forest', nameSq: 'Pyll gjethegjerë', nameDe: 'Laubwald', color: 'var(--color-lc-broadleaf)' },
  { code: 'coniferous_forest', name: 'Coniferous forest', nameSq: 'Pyll halor', nameDe: 'Nadelwald', color: 'var(--color-lc-coniferous)' },
  { code: 'cropland', name: 'Cropland', nameSq: 'Tokë bujqësore', nameDe: 'Ackerland', color: 'var(--color-lc-cropland)' },
  { code: 'grass_shrub', name: 'Grassland and shrubland', nameSq: 'Livadhe dhe shkurre', nameDe: 'Grünland und Buschland', color: 'var(--color-lc-grass-shrub)' },
  { code: 'olive_grove', name: 'Olive groves', nameSq: 'Ullishte', nameDe: 'Olivenhaine', color: 'var(--color-lc-olive)' },
  { code: 'built_up', name: 'Built-up', nameSq: 'Zonë e ndërtuar', nameDe: 'Bebaute Fläche', color: 'var(--color-lc-builtup)' },
  { code: 'water', name: 'Water', nameSq: 'Ujëra', nameDe: 'Wasser', color: 'var(--color-lc-water)' },
]

// Name of a class in the chosen language ('en', 'sq' or 'de')
export function className(landCoverClass, lang) {
  if (lang === 'sq') return landCoverClass.nameSq
  if (lang === 'de') return landCoverClass.nameDe
  return landCoverClass.name
}