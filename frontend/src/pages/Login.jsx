import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth'
import AuthLayout from '../components/AuthLayout'
import { Field, SubmitButton, ErrorText } from '../components/Form'

export default function Login() {
  const navigate = useNavigate()
  const { login } = useAuth()
  const [form, setForm] = useState({ email: '', password: '' })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

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
      navigate('/')
    } catch (err) {
      setError(err.message)
    }
    setLoading(false)
  }

  return (
    <AuthLayout title="Welcome back" subtitle="Sign in to AlbaniaSAT">
      <form onSubmit={handleSubmit}>
        <Field label="Email" name="email" type="email" value={form.email} onChange={handleChange} />
        <Field label="Password" name="password" type="password" value={form.password} onChange={handleChange} />

        <ErrorText error={error} />
        <SubmitButton loading={loading}>{loading ? 'Signing in...' : 'Sign in'}</SubmitButton>

        <p className="mt-6 text-center text-sm text-muted">
          No account yet?{' '}
          <Link to="/register" className="font-semibold text-primary hover:underline">Create one</Link>
        </p>
      </form>
    </AuthLayout>
  )
}