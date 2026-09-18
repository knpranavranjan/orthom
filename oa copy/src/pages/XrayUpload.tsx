import { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import AppShell from '../components/AppShell';
import XrayReport from '../components/XrayReport';
import { useSession } from '../store/session';
import { getInference, XrayRejectedError } from '../services/inference';
import { db, uid } from '../db';

type Phase = 'idle' | 'running' | 'done' | 'error' | 'rejected';

export default function XrayUpload({ onNext, onBack }: { onNext: () => void; onBack: () => void }) {
  const { t } = useTranslation();
  const s = useSession();
  const fileRef = useRef<HTMLInputElement>(null);
  const [phase, setPhase] = useState<Phase>(s.kl ? 'done' : 'idle');
  const [rejectMsg, setRejectMsg] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const runId = useRef(0);

  async function accept(f: File | undefined) {
    if (!f) return;
    const mine = ++runId.current;
    setRejectMsg(null);
    setErrorMsg(null);
    setPhase('running');
    const url = URL.createObjectURL(f);
    try {
      const res = await getInference().grade(f);
      if (runId.current !== mine) { URL.revokeObjectURL(url); return; }
      s.setXray(f, url);
      if (s.encounterId) {
        await db.xrays.put({ id: uid('xr'), encounterId: s.encounterId, blob: f, capturedAt: Date.now() });
      }
      s.setKL(res);
      setPhase('done');
    } catch (err) {
      if (runId.current !== mine) { URL.revokeObjectURL(url); return; }
      if (err instanceof XrayRejectedError) {
        URL.revokeObjectURL(url);
        s.setXray(null, null);
        s.setKL(null);
        setRejectMsg(err.userMessage);
        setPhase('rejected');
        return;
      }
      console.error('x-ray grading failed', err);
      s.setXray(f, url);
      if (s.encounterId) {
        await db.xrays.put({ id: uid('xr'), encounterId: s.encounterId, blob: f, capturedAt: Date.now() });
      }
      // Was previously silent: phase='error' had no message channel to the
      // UI at all, so a connectivity failure (e.g. VITE_XRAY_API pointing
      // at a backend that isn't reachable from this device) looked
      // identical to "nothing uploaded yet" — a real user hit exactly this
      // after deploying the frontend without deploying the API it talks to.
      setErrorMsg(t('xr.gradingError', { detail: err instanceof Error ? err.message : String(err) }));
      setPhase('error');
    }
  }

  return (
    <AppShell
      title={t('xr.title')}
      step="8 / 10"
      footer={
        <>
          <button className="btn grow0" onClick={onBack}>{t('nav.back')}</button>
          <button className="btn" onClick={() => fileRef.current?.click()}>{t('xr.upload')}</button>
          <button className="btn pri grow0" disabled={phase === 'rejected'} onClick={onNext}>
            {t('nav.next')}
          </button>
        </>
      }
    >
      <XrayReport
        layout="images"
        imageUrl={s.xrayUrl}
        heatmapUrl={s.kl?.heatmap ?? null}
        kl={s.kl}
        patient={s.patient}
        scanAt={s.xrayAt}
        pending={phase === 'running'}
        rejectMsg={phase === 'rejected' ? rejectMsg : null}
        errorMsg={phase === 'error' ? errorMsg : null}
      />

      <input ref={fileRef} type="file" accept="image/*" hidden
             onChange={(e) => void accept(e.target.files?.[0])} />
    </AppShell>
  );
}
