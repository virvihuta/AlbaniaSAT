import PublicLayout from '../components/PublicLayout'
import ContactForm from '../components/ContactForm'
import Faq from '../components/Faq'

// Public contact page: form on the left, short info on the right, FAQ below
export default function Contact() {
  return (
    <PublicLayout>
      <p className="eyebrow">Contact</p>
      <h1 className="mt-3 text-5xl">Get in touch</h1>
      <p className="mt-4 max-w-xl text-lg text-muted">
        Questions about the maps, the data or the project? Send us a message and the team will answer.
      </p>

      <section className="mt-10 grid gap-10 lg:grid-cols-5">
        <div className="lg:col-span-3">
          <ContactForm />
        </div>
        <div className="space-y-8 lg:col-span-2">
          <div>
            <p className="eyebrow">What to write about</p>
            <p className="mt-3 text-muted">Feedback on the map, errors you found in a region, ideas for stories or questions about using the data.</p>
          </div>
          <div>
            <p className="eyebrow">Response time</p>
            <p className="mt-3 text-muted">We usually reply within a few days.</p>
          </div>
        </div>
      </section>

      <section className="mt-24 grid gap-10 lg:grid-cols-3">
        <div>
          <p className="eyebrow">FAQ</p>
          <h2 className="mt-3 text-4xl">Questions</h2>
          <p className="mt-4 text-muted">Something missing? Write to the team through the contact form.</p>
        </div>
        <div className="lg:col-span-2">
          <Faq />
        </div>
      </section>
    </PublicLayout>
  )
}