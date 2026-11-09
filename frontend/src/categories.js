// The topics a story can have. "code" is saved in the database (column articles.category).
export const categories = [
  { code: 'urban_growth', name: 'Urban growth' },
  { code: 'forests', name: 'Forests' },
  { code: 'coast', name: 'Coast' },
  { code: 'agriculture', name: 'Agriculture' },
  { code: 'methodology', name: 'Methodology' },
]

// "forests" -> "Forests". Unknown or empty code gives ''.
export function categoryName(code) {
  return categories.find((c) => c.code === code)?.name || ''
}