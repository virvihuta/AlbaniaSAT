import { Routes, Route, Navigate } from 'react-router-dom'
import { ProtectedRoute } from './auth'
import Home from './pages/Home'
import News from './pages/News'
import NewsDetail from './pages/NewsDetail'
import Login from './pages/Login'
import Admin from './pages/Admin'
import ArticleForm from './pages/ArticleForm'

export default function App() {
  return (
    <Routes>
      {/* Public area: no login needed */}
      <Route path="/" element={<Home />} />
      <Route path="/news" element={<News />} />
      <Route path="/news/:slug" element={<NewsDetail />} />

      {/* Private area: only for admins and editors */}
      <Route path="/admin/login" element={<Login />} />
      <Route path="/admin" element={<ProtectedRoute><Admin /></ProtectedRoute>} />
      <Route path="/admin/articles/:id" element={<ProtectedRoute><ArticleForm /></ProtectedRoute>} />

      {/* Unknown address: back to the start page */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}