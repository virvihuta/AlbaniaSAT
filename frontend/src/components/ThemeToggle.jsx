import { useState } from 'react'

// Button that switches between light and dark mode and remembers the choice
export default function ThemeToggle() {
  const [theme, setTheme] = useState(document.documentElement.dataset.theme || 'light')

  function toggle() {
    const next = theme === 'dark' ? 'light' : 'dark'
    document.documentElement.dataset.theme = next
    try {
      localStorage.setItem('theme', next)
    } catch {
      // Private window: the choice is just not saved
    }
    setTheme(next)
  }

  return (
    <button
      onClick={toggle}
      aria-label="Switch light and dark mode"
      className="flex h-10 w-10 items-center justify-center rounded-lg bg-background text-lg hover:text-primary"
    >
      {theme === 'dark' ? '☀' : '☾'}
    </button>
  )
}