import { useLanguage } from '../i18n'

const languages = [
  { code: 'sq', label: 'SQ' },
  { code: 'en', label: 'EN' },
  { code: 'de', label: 'DE' },
]

// Small buttons SQ | EN | DE in the header. The chosen language is red.
export default function LanguageToggle() {
  const { lang, setLang, t } = useLanguage()

  return (
    <div role="group" aria-label={t('language.label')} className="flex rounded-lg bg-background p-1">
      {languages.map((l) => (
        <button
          key={l.code}
          onClick={() => setLang(l.code)}
          aria-pressed={lang === l.code}
          className={
            'h-8 rounded-md px-3 text-sm font-semibold ' +
            (lang === l.code ? 'bg-primary text-white' : 'text-muted hover:text-foreground')
          }
        >
          {l.label}
        </button>
      ))}
    </div>
  )
}