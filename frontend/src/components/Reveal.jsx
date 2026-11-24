import { useEffect, useRef, useState } from 'react'

// True as soon as the element is on the screen (once). Used to start animations when you scroll to them.
export function useInView() {
  const ref = useRef(null)
  const [inView, setInView] = useState(false)

  useEffect(() => {
    const element = ref.current
    if (!element || !('IntersectionObserver' in window)) {
      setInView(true)
      return
    }
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setInView(true)
          observer.disconnect()
        }
      },
      { threshold: 0.2 },
    )
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  return [ref, inView]
}

// Content that slides up softly when it comes into view
export function Reveal({ children, className = '', delay = 0 }) {
  const [ref, inView] = useInView()
  return (
    <div ref={ref} className={'reveal ' + (inView ? 'is-visible ' : '') + className} style={{ transitionDelay: `${delay}ms` }}>
      {children}
    </div>
  )
}

// Number that counts up from 0 when it comes into view.
// format turns the current number into text, for example 80.94 -> "80.94%".
export function CountUp({ value, format, duration = 1400 }) {
  const [ref, inView] = useInView()
  const [current, setCurrent] = useState(0)

  useEffect(() => {
    if (!inView) return
    // People who turned off animations in their system settings see the end value right away
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setCurrent(value)
      return
    }
    let frame
    const start = performance.now()
    const step = (now) => {
      const progress = Math.min((now - start) / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3) // fast at the start, slow at the end
      setCurrent(value * eased)
      if (progress < 1) frame = requestAnimationFrame(step)
    }
    frame = requestAnimationFrame(step)
    return () => cancelAnimationFrame(frame)
  }, [inView, value, duration])

  return <span ref={ref}>{format(current)}</span>
}