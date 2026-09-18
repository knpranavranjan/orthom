import { useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import AppShell from '../components/AppShell';
import MovementFigure from '../components/MovementFigure';
import KneeFitConnection from '../components/KneeFitConnection';
import KneeFitControls from '../components/KneeFitControls';
import KneeFitLiveMetrics from '../components/KneeFitLiveMetrics';
import KneeFitResult from '../components/KneeFitResult';
import KneeFitLog from '../components/KneeFitLog';
import KneeFitDebugPanel from '../components/KneeFitDebugPanel';
import GaitCapture from '../components/GaitCapture';
import { movementAt, type CaptureState } from '../movements';
import { useKneeFit } from '../hooks/useKneeFit';
import type { KneeFitSessionState, KneeFitSnapshot } from '../services/kneefit/types';
import { useSession } from '../store/session';
import { db, uid } from '../db';

interface Props { index: number; onNext: () => void; onBack: () => void }

/** KneeFit session state → the figure's animation state. */
function figureState(s: KneeFitSessionState, storedReview: boolean): CaptureState {
  if (storedReview) return 'review';
  switch (s) {
    case 'CALIBRATING': return 'calibrating';
    case 'CALIBRATED':
    case 'READY':
    case 'RETRYING': return 'ready';
    case 'RECORDING': return 'recording';
    case 'FINISHED':
    case 'ERROR': return 'review';
    default: return 'idle';
  }
}

export default function MovementCapture({ index, onNext, onBack }: Props) {
  const cfg = movementAt(index);
  const { t } = useTranslation();
  const kf = useKneeFit();
  const snap = kf.snapshot;

  const stored = useSession((s) => s.captures[cfg.id]);
  const setCapture = useSession((s) => s.setCapture);
  const clearCapture = useSession((s) => s.clearCapture);
  const encounterId = useSession((s) => s.encounterId);

  // Select this movement as the active KneeFit test whenever the screen changes.
  const { selectTest } = kf;
  useEffect(() => { selectTest(cfg.id); }, [cfg.id, selectTest]);

  // Persist the capture when the attempt finishes. Re-persists once if the
  // real result lands late (auto-finish fired first with no data), so the
  // stored capture is upgraded from empty → real metrics for the same attempt.
  const persisted = useRef<{ sessionId: number; keys: number }>({ sessionId: -1, keys: -1 });
  useEffect(() => {
    if (snap.session !== 'FINISHED' || snap.testType !== cfg.id) return;
    const keys = Object.keys(snap.metrics).length;
    const p = persisted.current;
    if (p.sessionId === snap.sessionId && keys <= p.keys) return;   // already stored, no upgrade
    persisted.current = { sessionId: snap.sessionId, keys };
    void persist(snap);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [snap.session, snap.sessionId, snap.testType, cfg.id, Object.keys(snap.metrics).length]);

  async function persist(s: KneeFitSnapshot) {
    const rec = {
      metrics: s.metrics,
      angles: s.crepitusAngles,
      quality: s.baseline?.quality ?? 0,
      samples: 0,
    };
    setCapture(cfg.id, rec);
    if (encounterId) {
      await db.captures.put({
        id: uid('cap'), encounterId, movementId: cfg.id,
        metrics: rec.metrics, angles: rec.angles,
        quality: rec.quality, createdAt: Date.now(),
      });
    }
  }

  function retryHere() {
    // Reset ONLY this movement session. History in db.captures is untouched.
    clearCapture(cfg.id);
    persisted.current = { sessionId: -1, keys: -1 };
    kf.retry();
  }

  // The KneeFit firmware only runs flexion (S1) and sit-to-stand (S2).
  const recordable = cfg.id === 'flexion' || cfg.id === 'sit_to_stand';

  // Gait is camera-based (see components/GaitCapture.tsx), not the KneeFit
  // BLE wearable — a completely different capture flow, so it renders its
  // own screen. This comes AFTER every hook above so hook call order stays
  // identical when Back/Next moves between mv0/mv1/mv2 (same component
  // instance, only the `index` prop changes — an early return before a
  // hook would violate the rules of hooks on that transition).
  if (cfg.id === 'gait') {
    return (
      <GaitCapture
        cfg={cfg}
        index={index}
        storedMetrics={stored?.metrics ?? null}
        onComplete={(metrics, frameCount) => {
          const rec = { metrics, angles: [] as number[], quality: 0, samples: frameCount };
          setCapture(cfg.id, rec);
          if (encounterId) {
            void db.captures.put({
              id: uid('cap'), encounterId, movementId: cfg.id,
              metrics: rec.metrics, angles: rec.angles,
              quality: rec.quality, createdAt: Date.now(),
            });
          }
        }}
        onRetry={() => clearCapture(cfg.id)}
        onBack={onBack}
        onNext={onNext}
      />
    );
  }

  const freshFinish = snap.session === 'FINISHED' && snap.testType === cfg.id;
  const midSession = snap.session === 'RECORDING' || snap.session === 'CALIBRATING';
  const storedReview = !!stored && !freshFinish && !midSession;

  // What the footer controls should reflect.
  const controlState: KneeFitSessionState = storedReview ? 'FINISHED' : snap.session;

  const liveAngle =
    cfg.id === 'flexion' && typeof snap.live['angle'] === 'number'
      ? snap.live['angle']
      : undefined;

  const reps = cfg.reps;
  const repsDone =
    cfg.id === 'sit_to_stand' ? Math.min(reps, Math.round(snap.metrics['repCount'] ?? 0)) : 0;

  // Snapshot to render in the metrics/result panel: the live session, or the
  // stored capture reconstituted into a snapshot shape for the review case.
  const viewSnap: KneeFitSnapshot = storedReview
    ? { ...snap, metrics: stored.metrics, crepitusAngles: stored.angles, live: {}, elapsed: 0 }
    : snap;

  const showMetrics =
    storedReview || freshFinish || snap.session === 'RECORDING' || snap.session === 'FINISHED';

  const statusKey = `kf.state.${storedReview ? 'SAVED' : snap.session}`;

  return (
    <AppShell
      title={t(`mv.${cfg.key}.title`)}
      step={`${index + 4} / 10`}
      footer={
        <KneeFitControls
          kf={kf}
          effectiveState={controlState}
          onBack={onBack}
          onNext={onNext}
          onRetry={retryHere}
          hasCapture={!!stored}
          recordable={recordable}
        />
      }
    >
      <div className="mv-kf">
        <KneeFitConnection kf={kf} />

        <div className="mv-kf-body">
          <div className="mv-kf-fig">
            <MovementFigure
              media={cfg.media}
              state={figureState(snap.session, storedReview)}
              angle={liveAngle}
            />
          </div>

          <div className="mv-kf-panel">
            {!showMetrics && (
              <div className="mv-kf-hint">
                <div className="instr">{t(`mv.${cfg.key}.instr`)}</div>
                {recordable ? (
                  <>
                    <div className="small muted">{t(`kf.hint.${snap.session}`, { defaultValue: t('kf.hint.CONNECTED') })}</div>
                    {snap.baseline && (
                      <div className="xs" style={{ textTransform: 'none', letterSpacing: 0 }}>
                        {t('kf.zeroSet')} · {t('cap.quality')} {snap.baseline.quality}%
                      </div>
                    )}
                  </>
                ) : (
                  <div className="small muted">{t('kf.notOnDevice')}</div>
                )}
              </div>
            )}

            {showMetrics && !storedReview && (
              (freshFinish || snap.session === 'FINISHED')
                ? <KneeFitResult cfg={cfg} snapshot={viewSnap} />
                : <KneeFitLiveMetrics cfg={cfg} snapshot={viewSnap} mode="live" />
            )}

            {storedReview && <KneeFitResult cfg={cfg} snapshot={viewSnap} />}
          </div>
        </div>

        {reps > 0 && snap.session === 'RECORDING' && (
          <div className="dots" style={{ padding: '0 4px' }}>
            {Array.from({ length: reps }, (_, i) => (
              <span key={i} className={i < repsDone ? 'dot on' : 'dot'} />
            ))}
          </div>
        )}

        <div className={snap.session === 'RECORDING' ? 'status rec' : 'status'}>
          {t(statusKey, { defaultValue: snap.session })}
        </div>

        <KneeFitLog lines={snap.log} />
        <KneeFitDebugPanel snapshot={snap} />
      </div>
    </AppShell>
  );
}
