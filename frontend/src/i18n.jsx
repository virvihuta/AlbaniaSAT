// Language of the website: English, Albanian or German. The choice is saved in the browser.
/* eslint-disable react-refresh/only-export-components -- this file shares both a component and a hook on purpose */
import { createContext, useContext, useEffect, useState } from 'react'
import { translations as baseTranslations } from './translations'
import { homeTranslations } from './homeTranslations'

// All texts in one place: the texts of all pages plus the texts of the start page
const translations = Object.fromEntries(
  Object.keys(baseTranslations).map((lang) => [lang, { ...baseTranslations[lang], ...homeTranslations[lang] }]),
)

const LanguageContext = createContext(null)

// Language from the last visit, English if there is none
function savedLanguage() {
  try {
    const saved = localStorage.getItem('lang')
    return translations[saved] ? saved : 'en'
  } catch {
    return 'en'
  }
}

export function LanguageProvider({ children }) {
  const [lang, setLang] = useState(savedLanguage)

  // Remember the choice and tell the browser which language the page has
  useEffect(() => {
    document.documentElement.lang = lang
    try {
      localStorage.setItem('lang', lang)
    } catch {
      // Private window: the choice is just not saved
    }
  }, [lang])

  // t('nav.story') -> "Story", "Histori" or "Berichte". Values like {f1} are filled in from vars.
  function t(key, vars = {}) {
    const text = translations[lang][key] ?? translations.en[key] ?? key
    return text.replace(/\{(\w+)\}/g, (_, name) => vars[name] ?? '')
  }

  return <LanguageContext.Provider value={{ lang, setLang, t }}>{children}</LanguageContext.Provider>
}

export function useLanguage() {
  return useContext(LanguageContext)
}