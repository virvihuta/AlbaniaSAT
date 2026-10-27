import Contours from './Contours'

const facts = [
  { value: '7', label: 'Land cover classes' },
  { value: '10 m', label: 'Resolution' },
  { value: '80.94 %', label: 'Accuracy' },
]

// The page for Login and Register: dark picture on the left, form on the right
export default function AuthLayout({ title, subtitle, children }) {
  return (
    <div className="flex min-h-screen">
      {/* Left half (hidden on phones) */}
      <div className="relative hidden overflow-hidden bg-neutral-950 md:flex md:w-1/2">
        <Contours />

        {/* Red glow in the bottom left corner */}
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_0%_100%,rgba(191,30,45,0.55),transparent_60%)]" />

        <div className="relative flex w-full flex-col justify-between p-12 text-white">
          <div className="flex items-center gap-3">
            <img src="/logo.png" alt="AlbaniaSAT logo" className="h-14 w-auto" />
            <span className="font-heading text-[2rem] font-light tracking-tight">AlbaniaSAT</span>
          </div>

          <div>
            <h2 className="max-w-md text-5xl leading-tight">Europe couldn't see Albania. Now it can.</h2>
            <div className="mt-10 flex gap-10">
              {facts.map((fact) => (
                <div key={fact.label}>
                  <p className="text-2xl font-bold">{fact.value}</p>
                  <p className="text-xs uppercase tracking-wider text-white/60">{fact.label}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Right half: the form */}
      <div className="flex w-full items-center justify-center px-6 md:w-1/2">
        <div className="w-full max-w-md">
          <img src="/logo.png" alt="AlbaniaSAT" className="mb-8 h-14 w-auto md:hidden" />
          <h1 className="text-3xl">{title}</h1>
          <p className="mt-1 mb-8 text-muted">{subtitle}</p>
          {children}
        </div>
      </div>
    </div>
  )
}