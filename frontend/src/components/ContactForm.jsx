import { useState } from 'react'
import { postJSON } from '../api'
import { useLanguage } from '../i18n'

const empty = { name: '', email: '', message: '' }

// Checks every field and returns an object with one error per wrong field.
// The error is a key from translations.js, so it changes with the language.
function validate(form) {
  const errors = {}
  if (form.name.trim().length < 2) errors.name = 'form.nameShort'
  if (form.name.length > 100) errors.name = 'form.nameLong'
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email)) errors.email = 'form.emailInvalid'
  if (form.message.trim().length < 10) errors.message = 'form.messageShort'
  if (form.message.length > 2000) errors.message = 'form.messageLong'
  return errors
}

// Contact form with checks in the browser before anything is sent
export default function ContactForm() {
  const { t } = useLanguage()
  const [form, setForm] = useState(empty)
  const [errors, setErrors] = useState({})
  const [status, setStatus] = useState('') // '', 'sending', 'sent' or 'failed'

  function handleChange(e) {
    setForm({ ...form, [e.target.name]: e.target.value })
    setErrors({ ...errors, [e.target.name]: '' })
  }

  async function handleSubmit(e) {
    e.preventDefault()
    const found = validate(form)
    setErrors(found)
    if (Object.keys(found).length > 0) return

    setStatus('sending')
    try {
      await postJSON('/contact', form)
      setForm(empty)
      setStatus('sent')
    } catch {
      setStatus('failed')
    }
  }

  const input = 'mt-1 w-full rounded-lg border bg-background px-4 py-2.5 outline-none focus:border-primary '

  return (
    <form onSubmit={handleSubmit} noValidate className="space-y-5 rounded-2xl border border-border bg-surface p-8">
      {[
        { name: 'name', label: t('form.name'), type: 'text' },
        { name: 'email', label: t('form.email'), type: 'email' },
      ].map((field) => (
        <label key={field.name} className="block text-sm font-medium">
          {field.label}
          <input name={field.name} type={field.type} value={form[field.name]} onChange={handleChange}
            className={input + (errors[field.name] ? 'border-primary' : 'border-border')} />
          {errors[field.name] && <span className="mt-1 block text-primary">{t(errors[field.name])}</span>}
        </label>
      ))}

      <label className="block text-sm font-medium">
        {t('form.message')}
        <textarea name="message" rows="6" value={form.message} onChange={handleChange}
          className={input + (errors.message ? 'border-primary' : 'border-border')} />
        <span className="mt-1 flex justify-between">
          <span className="text-primary">{errors.message ? t(errors.message) : ''}</span>
          <span className="text-muted">{form.message.length} / 2000</span>
        </span>
      </label>

      {status === 'sent' && <p className="rounded-lg bg-background px-4 py-3 text-sm text-lc-coniferous">{t('form.sent')}</p>}
      {status === 'failed' && <p className="rounded-lg bg-background px-4 py-3 text-sm text-primary">{t('form.failed')}</p>}

      <button type="submit" disabled={status === 'sending'}
        className="rounded-lg bg-primary px-6 py-3 font-semibold text-white hover:opacity-90 disabled:opacity-60">
        {status === 'sending' ? t('form.sending') : t('form.send')}
      </button>
    </form>
  )
}