import { Routes, Route, Navigate } from 'react-router-dom'
import { ProtectedRoute } from './auth'
import Home from './pages/Home'
import Login from './pages/Login'
import Admin from './pages/Admin'
import Register from './pages/Register'

export default function App() {
  return (
    <Routes>
      {/* Public area: no login needed */}
      <Route path="/" element={<Home />} />

      {/* Private area: only for admins and editors */}
      <Route path="/admin/login" element={<Login />} />
      <Route path="/admin" element={<ProtectedRoute><Admin /></ProtectedRoute>} />
      <Route path="/admin/register" element={<ProtectedRoute role="admin"><Register /></ProtectedRoute>} />

      {/* Unknown address: back to the start page */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}