import { useState } from 'react'

const questions = [
  {
    q: 'Where does the imagery come from?',
    a: 'From the Copernicus Sentinel-2 mission, accessed through Google Earth Engine. Sentinel-2 photographs all of Albania every few days at 10 m resolution.',
  },
  {
    q: 'How accurate is the map?',
    a: 'The AlbaniaSAT model reaches 80.94% accuracy on held-out Albanian test patches. Single pixels can be wrong, but trends over larger areas are far more reliable.',
  },
  {
    q: 'Why not use an existing European model?',
    a: 'Models trained on European data such as EuroSAT reach only 19.08% on Albanian patches. Albanian land looks different, so we trained on Albanian ground truth.',
  },
  {
    q: 'Which years can I compare?',
    a: 'The classification covers 2017 to 2024, so you can compare how land cover changed over these years.',
  },
]

// Questions that open and close when you click them. Only one is open at a time.
export default function Faq() {
  const [open, setOpen] = useState(0)

  return (
    <div className="border-t border-border">
      {questions.map((item, index) => (
        <div key={item.q} className="border-b border-border">
          <button
            onClick={() => setOpen(open === index ? -1 : index)}
            aria-expanded={open === index}
            className="flex w-full items-center justify-between gap-6 py-6 text-left font-heading text-lg font-light hover:text-primary"
          >
            {item.q}
            <span className="text-2xl text-muted">{open === index ? '−' : '+'}</span>
          </button>
          {open === index && <p className="max-w-2xl pb-6 text-muted">{item.a}</p>}
        </div>
      ))}
    </div>
  )
}