import { useState } from 'react'
import { register } from '../auth'
import AdminLayout from '../components/AdminLayout'
import { Field, SubmitButton, ErrorText } from '../components/Form'

const emptyForm = { name: '', email: '', password: '', confirm: '', role: 'editor' }

// Only admins see this page: they create accounts for new admins or editors
export default function Register() {
  const [form, setForm] = useState(emptyForm)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [loading, setLoading] = useState(false)

  // Updates the field that was typed in, using its name
  function handleChange(e) {
    setForm({ ...form, [e.target.name]: e.target.value })
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setSuccess('')

    if (form.password.length < 8) return setError('Password must be at least 8 characters')
    if (form.password !== form.confirm) return setError('Passwords do not match')

    setError('')
    setLoading(true)
    try {
      await register(form.name, form.email, form.password, form.role)
      setSuccess(`Account for ${form.name} created`)
      setForm(emptyForm)
    } catch (err) {
      setError(err.message)
    }
    setLoading(false)
  }

  return (
    <AdminLayout>
      <h1 className="text-3xl">New account</h1>
      <form onSubmit={handleSubmit} className="mt-8 max-w-md">
        <Field label="Name" name="name" value={form.name} onChange={handleChange} />
        <Field label="Email" name="email" type="email" value={form.email} onChange={handleChange} />
        <Field label="Password" name="password" type="password" value={form.password} onChange={handleChange} />
        <Field label="Confirm password" name="confirm" type="password" value={form.confirm} onChange={handleChange} />

        <label className="mb-4 block">
          <span className="mb-1 block text-sm font-medium">Role</span>
          <select name="role" value={form.role} onChange={handleChange} className="w-full rounded-lg border border-border bg-surface px-4 py-2">
            <option value="editor">Editor</option>
            <option value="admin">Admin</option>
          </select>
        </label>

        <ErrorText error={error} />
        {success && <p className="mb-4 rounded-lg bg-surface px-4 py-2 text-sm text-lc-forest">{success}</p>}
        <SubmitButton loading={loading}>{loading ? 'Creating account...' : 'Create account'}</SubmitButton>
      </form>
    </AdminLayout>
  )
}