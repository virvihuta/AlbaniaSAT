import { useAuth } from '../auth'
import AdminLayout from '../components/AdminLayout'

// Start page of the private area. Article management comes here later (goal M-2.3).
export default function Admin() {
  const { user } = useAuth()

  return (
    <AdminLayout>
      <h1 className="text-3xl">Welcome, {user.name}</h1>
      <p className="mt-2 text-muted">You are logged in as {user.role}.</p>
    </AdminLayout>
  )
}