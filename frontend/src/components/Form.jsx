// Small building blocks used by the Login and Register forms.

// A label with an input under it. Any extra props (name, type, value...) go to the input.
export function Field({ label, ...inputProps }) {
  return (
    <label className="mb-4 block">
      <span className="mb-1 block text-sm font-medium">{label}</span>
      <input
        className="w-full rounded-lg border border-border bg-surface px-4 py-2 outline-none focus:border-primary"
        required
        {...inputProps}
      />
    </label>
  )
}

export function SubmitButton({ loading, children }) {
  return (
    <button
      type="submit"
      disabled={loading}
      className="w-full rounded-lg bg-primary px-5 py-3 font-semibold text-white transition hover:opacity-90 disabled:opacity-60"
    >
      {children}
    </button>
  )
}

// Shows a red message, or nothing if there is no error
export function ErrorText({ error }) {
  if (!error) return null
  return <p className="mb-4 rounded-lg bg-surface px-4 py-2 text-sm text-primary">{error}</p>
}