// Moving white contour lines, like a topographic map.
// How it works: random "terrain" noise is generated, then only thin height bands
// are kept visible. Every 6th step in the list below is 1 (visible), the rest 0.
const bands = Array(10).fill('0 0 0 0 0 1').join(' ')

export default function Contours() {
  return (
    <svg className="drift absolute -inset-40 h-[calc(100%+20rem)] w-[calc(100%+20rem)] opacity-30">
      <filter id="topo" x="0" y="0" width="100%" height="100%">
        {/* The random terrain. The animate tag slowly changes it so the lines flow. */}
        <feTurbulence type="fractalNoise" baseFrequency="0.003 0.005" numOctaves="3" seed="12">
          <animate
            attributeName="baseFrequency"
            values="0.003 0.005; 0.0036 0.0056; 0.003 0.005"
            dur="30s"
            repeatCount="indefinite"
          />
        </feTurbulence>
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