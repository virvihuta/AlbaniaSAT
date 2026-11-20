import PublicLayout from '../components/PublicLayout'
import ContactForm from '../components/ContactForm'
import Faq from '../components/Faq'
import { useLanguage } from '../i18n'

// Public contact page: form on the left, short info on the right, FAQ below
export default function Contact() {
  const { t } = useLanguage()

  return (
    <PublicLayout>
      <p className="eyebrow">{t('nav.contact')}</p>
      <h1 className="mt-3 text-5xl">{t('contact.title')}</h1>
      <p className="mt-4 max-w-xl text-lg text-muted">{t('contact.intro')}</p>

      <section className="mt-10 grid gap-10 lg:grid-cols-5">
        <div className="lg:col-span-3">
          <ContactForm />
        </div>
        <div className="space-y-8 lg:col-span-2">
          <div>
            <p className="eyebrow">{t('contact.aboutTitle')}</p>
            <p className="mt-3 text-muted">{t('contact.aboutText')}</p>
          </div>
          <div>
            <p className="eyebrow">{t('contact.responseTitle')}</p>
            <p className="mt-3 text-muted">{t('contact.responseText')}</p>
          </div>
        </div>
      </section>

      <Faq />
    </PublicLayout>
  )
}