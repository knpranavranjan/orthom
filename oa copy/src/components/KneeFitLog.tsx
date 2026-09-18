import { useEffect, useRef } from 'react';

/**
 * Compact event console — mirrors the KneeFit firmware's serial output so the
 * operator can see the command round-trip (→ 'C', ← Calibration complete, …).
 * Always visible; auto-scrolls to the newest line.
 */
export default function KneeFitLog({ lines }: { lines: string[] }) {
  const boxRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = boxRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines]);

  if (lines.length === 0) return null;
  return (
    <div className="kf-console" ref={boxRef} role="log" aria-live="polite">
      {lines.map((l, i) => (
        <div key={i} className="kf-console-line">{l}</div>
      ))}
    </div>
  );
}
