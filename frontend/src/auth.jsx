// Everything about logging in lives in this one file.
/* eslint-disable react-refresh/only-export-components -- this file shares both components and functions on purpose */
import { createContext, useContext, useState } from 'react'
import { Navigate } from 'react-router-dom'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const MOCK = import.meta.env.VITE_MOCK_AUTH === 'true'

// Sends data to the backend and returns its answer. The token proves who is logged in.
async function post(path, body) {
  // Test mode: wait half a second and pretend the backend said yes
  if (MOCK) {
    await new Promise((resolve) => setTimeout(resolve, 500))
    return { token: 'demo-token', user: { name: 'Demo Admin', email: body.email, role: 'admin' } }
  }

  const res = await fetch(API_URL + path, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: 'Bearer ' + localStorage.getItem('token'),
    },
    body: JSON.stringify(body),
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.detail || data.message || 'Something went wrong')
  return data
}

// Only an admin can create new accounts (role is "admin" or "editor")
export function register(name, email, password, role) {
  return post('/auth/register', { name, email, password, role })
}

// Shares the logged in user with every page
const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  // Read the user from the browser, so a refresh keeps you logged in
  const [user, setUser] = useState(() => JSON.parse(localStorage.getItem('user')))

  async function login(email, password) {
    const data = await post('/auth/login', { email, password })
    localStorage.setItem('token', data.token)
    localStorage.setItem('user', JSON.stringify(data.user))
    setUser(data.user)
  }

  function logout() {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    setUser(null)
  }

  return <AuthContext.Provider value={{ user, login, logout }}>{children}</AuthContext.Provider>
}

export function useAuth() {
  return useContext(AuthContext)
}

// Protects the private area. With role="admin" only admins get in.
export function ProtectedRoute({ role, children }) {
  const { user } = useAuth()
  if (!user) return <Navigate to="/admin/login" replace />
  if (role && user.role !== role) return <Navigate to="/admin" replace />
  return children
}