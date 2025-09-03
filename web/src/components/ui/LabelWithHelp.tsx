import React from 'react'

// Lightweight accessible tooltip (no external deps)
function Tooltip({ children }: { children: React.ReactNode }) {
  return (
    <div role="tooltip" className="z-50 rounded bg-neutral-800 text-neutral-100 border border-neutral-700 px-2 py-1 shadow-lg" >
      {children}
    </div>
  )
}

export function LabelWithHelp({
  label,
  short,
  help,
  id,
}: { label: string; short?: string; help?: string; id?: string }) {
  const [open, setOpen] = React.useState(false)
  return (
    <div className="mb-1">
      <div className="flex items-center gap-1">
        <label htmlFor={id} className="text-sm font-medium text-neutral-200">{label}</label>
        {help && (
          <div className="relative">
            <button
              type="button"
              aria-label={`Help for ${label}`}
              onMouseEnter={() => setOpen(true)}
              onMouseLeave={() => setOpen(false)}
              onFocus={() => setOpen(true)}
              onBlur={() => setOpen(false)}
              className="text-neutral-400 hover:text-neutral-200 focus:outline-none focus:ring-1 focus:ring-neutral-600 rounded"
            >
              {/* Info icon (inline SVG to avoid deps) */}
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="16" x2="12" y2="12"></line>
                <line x1="12" y1="8" x2="12.01" y2="8"></line>
              </svg>
            </button>
            {open && (
              <div className="absolute left-4 top-5 w-max max-w-[280px] text-xs leading-snug">
                <Tooltip>{help}</Tooltip>
              </div>
            )}
          </div>
        )}
      </div>
      {short && <p id={`${id}-help`} className="text-[11px] text-neutral-400">{short}</p>}
    </div>
  )
}


