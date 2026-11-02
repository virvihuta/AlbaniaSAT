// The 7 land cover classes of the AlbaniaSAT model.
// "code" is the same as in the database (table land_cover_classes).
// The colors live in the design system (index.css), so we only point to them here.
export const landCoverClasses = [
  { code: 'broadleaf_forest', name: 'Broadleaf forest', color: 'var(--color-lc-broadleaf)' },
  { code: 'coniferous_forest', name: 'Coniferous forest', color: 'var(--color-lc-coniferous)' },
  { code: 'cropland', name: 'Cropland', color: 'var(--color-lc-cropland)' },
  { code: 'grass_shrub', name: 'Grassland and shrubland', color: 'var(--color-lc-grass-shrub)' },
  { code: 'olive_grove', name: 'Olive groves', color: 'var(--color-lc-olive)' },
  { code: 'built_up', name: 'Built-up', color: 'var(--color-lc-builtup)' },
  { code: 'water', name: 'Water', color: 'var(--color-lc-water)' },
]