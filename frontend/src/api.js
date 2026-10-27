// Reads data from the backend (public endpoints, no login needed)
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export async function getJSON(path) {
  const res = await fetch(API_URL + path)
  if (!res.ok) throw new Error(res.status === 404 ? 'Not found' : 'Could not load data')
  return res.json()
}

// Turns "2026-10-24T14:00:00Z" into "24 Oct 2026"
export function formatDate(value) {
  return value ? new Date(value).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' }) : ''
}

// Like getJSON, but sends the login token. Used in the admin area.
export async function adminRequest(method, path, body) {
  const res = await fetch(API_URL + path, {
    method,
    headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + localStorage.getItem('token') },
    body: body ? JSON.stringify(body) : undefined,
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Please check the fields')
  return data
}