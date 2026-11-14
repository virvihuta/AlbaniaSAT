import { useEffect, useState } from 'react'
import { getJSON } from '../api'

// Used until the FAQ endpoint exists in the backend, or if it cannot be reached
const fallback = [
  {
    id: 'f1',
    question: 'Where does the imagery come from?',
    answer: 'From the Copernicus Sentinel-2 mission, accessed through Google Earth Engine. Sentinel-2 photographs all of Albania every few days at 10 m resolution.',
  },
  {
    id: 'f2',
    question: 'How accurate is the map?',
    answer: 'The AlbaniaSAT model reaches 80.94% accuracy on held-out Albanian test patches. Single pixels can be wrong, but trends over larger areas are far more reliable.',
  },
  {
    id: 'f3',
    question: 'Why not use an existing European model?',
    answer: 'Models trained on European data such as EuroSAT reach only 19.08% on Albanian patches. Albanian land looks different, so we trained on Albanian ground truth.',
  },
  {
    id: 'f4',
    question: 'Which years can I compare?',
    answer: 'The classification covers 2017 to 2024, so you can compare how land cover changed over these years.',
  },
]

// FAQ section of the contact page. The questions come from the admin area (GET /faqs).
// Only one answer is open at a time. Without questions the whole section is hidden.
export default function Faq() {
  const [questions, setQuestions] = useState(null)
  const [open, setOpen] = useState(0)

  useEffect(() => {
    getJSON('/faqs')
      .then(setQuestions)
      .catch(() => setQuestions(fallback))
  }, [])

  if (!questions || questions.length === 0) return null

  return (
    <section className="mt-24 grid gap-10 lg:grid-cols-3">
      <div>
        <p className="eyebrow">FAQ</p>
        <h2 className="mt-3 text-4xl">Questions</h2>
        <p className="mt-4 text-muted">Something missing? Write to the team through the contact form.</p>
      </div>

      <div className="border-t border-border lg:col-span-2">
        {questions.map((item, index) => (
          <div key={item.id} className="border-b border-border">
            <button
              onClick={() => setOpen(open === index ? -1 : index)}
              aria-expanded={open === index}
              className="flex w-full items-center justify-between gap-6 py-6 text-left font-heading text-lg font-light hover:text-primary"
            >
              {item.question}
              <span className="text-2xl text-muted">{open === index ? '−' : '+'}</span>
            </button>
            {open === index && <p className="max-w-2xl whitespace-pre-line pb-6 text-muted">{item.answer}</p>}
          </div>
        ))}
      </div>
    </section>
  )
}