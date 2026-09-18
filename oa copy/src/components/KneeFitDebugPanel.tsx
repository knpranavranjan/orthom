import { useState } from 'react';
import type { KneeFitSnapshot } from '../services/kneefit/types';

/**
 * Dev-only diagnostics. Renders nothing in a production build
 * (`import.meta.env.DEV` is false), so it never clutters the TFT kiosk.
 * Collapsed by default even in dev.
 */
export default function KneeFitDebugPanel({ snapshot }: { snapshot: KneeFitSnapshot }) {
  const [open, setOpen] = useState(false);
  if (!import.meta.env.DEV) return null;

  const { connection: c } = snapshot;
  const age = snapshot.lastMessageAt ? `${Math.round((Date.now() - snapshot.lastMessageAt) / 100) / 10}s ago` : '—';

  return (
    <div className="kf-debug">
      <button className="kf-debug-toggle" onClick={() => setOpen((v) => !v)}>
        {open ? '▾' : '▸'} debug
      </button>
      {open && (
        <pre className="kf-debug-body">
{`state        ${snapshot.session}
conn         ${c.state}${c.uuidVariant ? ` (${c.uuidVariant})` : ''}
device       ${c.deviceName ?? '—'}
error        ${c.error ?? '—'}
testType     ${snapshot.testType ?? '—'}
sessionId    ${snapshot.sessionId}
baseline     ${snapshot.baseline ? `${snapshot.baseline.angle.toFixed(1)}° q${snapshot.baseline.quality} ${snapshot.baseline.deviceConfirmed ? 'device' : 'frontend'}` : '—'}
elapsed      ${snapshot.elapsed.toFixed(1)}s
last msg     ${age}
live         ${JSON.stringify(snapshot.live)}
metrics      ${JSON.stringify(snapshot.metrics)}
lastChunk    ${JSON.stringify((snapshot.lastChunk ?? '').slice(0, 300))}
lastRaw      ${(snapshot.lastRaw ?? '').slice(0, 240)}`}
        </pre>
      )}
    </div>
  );
}
