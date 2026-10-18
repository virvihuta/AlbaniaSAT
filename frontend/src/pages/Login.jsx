import { useState } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth'
import AuthLayout from '../components/AuthLayout'
import { Field, SubmitButton, ErrorText } from '../components/Form'

// Login for the private area (admins and editors only)
export default function Login() {
  const navigate = useNavigate()
  const { user, login } = useAuth()
  const [form, setForm] = useState({ email: '', password: '' })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  // Already logged in? Go straight to the private area
  if (user) return <Navigate to="/admin" replace />

  // Updates the field that was typed in, using its name
  function handleChange(e) {
    setForm({ ...form, [e.target.name]: e.target.value })
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await login(form.email, form.password)
      navigate('/admin')
    } catch (err) {
      setError(err.message)
      setLoading(false)
    }
  }

  return (
    <AuthLayout title="Admin login" subtitle="Only for the AlbaniaSAT team">
      <form onSubmit={handleSubmit}>
        <Field label="Email" name="email" type="email" value={form.email} onChange={handleChange} />
        <Field label="Password" name="password" type="password" value={form.password} onChange={handleChange} />

        <ErrorText error={error} />
        <SubmitButton loading={loading}>{loading ? 'Signing in...' : 'Sign in'}</SubmitButton>
      </form>
    </AuthLayout>
  )
}