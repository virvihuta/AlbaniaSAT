import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { register } from '../auth'
import AuthLayout from '../components/AuthLayout'
import { Field, SubmitButton, ErrorText } from '../components/Form'

export default function Register() {
  const navigate = useNavigate()
  const [form, setForm] = useState({ name: '', email: '', password: '', confirm: '' })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  // Updates the field that was typed in, using its name
  function handleChange(e) {
    setForm({ ...form, [e.target.name]: e.target.value })
  }

  async function handleSubmit(e) {
    e.preventDefault()

    if (form.password.length < 8) return setError('Password must be at least 8 characters')
    if (form.password !== form.confirm) return setError('Passwords do not match')

    setError('')
    setLoading(true)
    try {
      await register(form.name, form.email, form.password)
      navigate('/login')
    } catch (err) {
      setError(err.message)
    }
    setLoading(false)
  }

  return (
    <AuthLayout title="Create account" subtitle="Join AlbaniaSAT">
      <form onSubmit={handleSubmit}>
        <Field label="Name" name="name" value={form.name} onChange={handleChange} />
        <Field label="Email" name="email" type="email" value={form.email} onChange={handleChange} />
        <Field label="Password" name="password" type="password" value={form.password} onChange={handleChange} />
        <Field label="Confirm password" name="confirm" type="password" value={form.confirm} onChange={handleChange} />

        <ErrorText error={error} />
        <SubmitButton loading={loading}>{loading ? 'Creating account...' : 'Create account'}</SubmitButton>

        <p className="mt-6 text-center text-sm text-muted">
          Already have an account?{' '}
          <Link to="/login" className="font-semibold text-primary hover:underline">Login</Link>
        </p>
      </form>
    </AuthLayout>
  )
}