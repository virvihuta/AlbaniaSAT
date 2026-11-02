import { useState } from 'react'
import { postJSON } from '../api'

const empty = { name: '', email: '', message: '' }

// Checks every field and returns an object with one error text per wrong field
function validate(form) {
  const errors = {}
  if (form.name.trim().length < 2) errors.name = 'Please enter your name.'
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email)) errors.email = 'Please enter a valid email address.'
  if (form.message.trim().length < 10) errors.message = 'Your message needs at least 10 characters.'
  if (form.message.length > 2000) errors.message = 'Your message can have at most 2000 characters.'
  return errors
}

// Contact form with checks in the browser before anything is sent
export default function ContactForm() {
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
        { name: 'name', label: 'Name', type: 'text' },
        { name: 'email', label: 'Email', type: 'email' },
      ].map((field) => (
        <label key={field.name} className="block text-sm font-medium">
          {field.label}
          <input name={field.name} type={field.type} value={form[field.name]} onChange={handleChange}
            className={input + (errors[field.name] ? 'border-primary' : 'border-border')} />
          {errors[field.name] && <span className="mt-1 block text-primary">{errors[field.name]}</span>}
        </label>
      ))}

      <label className="block text-sm font-medium">
        Message
        <textarea name="message" rows="6" value={form.message} onChange={handleChange}
          className={input + (errors.message ? 'border-primary' : 'border-border')} />
        <span className="mt-1 flex justify-between">
          <span className="text-primary">{errors.message}</span>
          <span className="text-muted">{form.message.length} / 2000</span>
        </span>
      </label>

      {status === 'sent' && <p className="rounded-lg bg-background px-4 py-3 text-sm text-lc-forest">Thank you! Your message was sent.</p>}
      {status === 'failed' && <p className="rounded-lg bg-background px-4 py-3 text-sm text-primary">The message could not be sent. Please try again later.</p>}

      <button type="submit" disabled={status === 'sending'}
        className="rounded-lg bg-primary px-6 py-3 font-semibold text-white hover:opacity-90 disabled:opacity-60">
        {status === 'sending' ? 'Sending...' : 'Send message →'}
      </button>
    </form>
  )
}