// Contour lines in the background, like a topographic map.
// How it works: random "terrain" noise is generated once, then only thin height bands
// are kept visible. Every 6th step in the list below is 1 (visible), the rest 0.
// The lines only move with a CSS transform (class "drift"), which is cheap for the browser.
// The noise itself is not animated any more, because recalculating it every frame made the page slow.
const bands = Array(10).fill('0 0 0 0 0 1').join(' ')

// className sets how strong the lines are, for example opacity-10 behind text.
export default function Contours({ className = 'opacity-30' }) {
  return (
    <svg
      className={`drift pointer-events-none absolute -inset-40 h-[calc(100%+20rem)] w-[calc(100%+20rem)] will-change-transform ${className}`}
      aria-hidden="true"
    >
      <filter id="topo" x="0" y="0" width="100%" height="100%">
        {/* The random terrain */}
        <feTurbulence type="fractalNoise" baseFrequency="0.003 0.005" numOctaves="2" seed="12" />
        {/* Make it white, using the terrain height as transparency */}
        <feColorMatrix type="matrix" values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  1 0 0 0 0" />
        {/* Keep only the thin bands, which become the lines */}
        <feComponentTransfer>
          <feFuncA type="discrete" tableValues={bands} />
        </feComponentTransfer>
      </filter>
      <rect width="100%" height="100%" filter="url(#topo)" />
    </svg>
  )
}