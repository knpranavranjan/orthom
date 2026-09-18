import { useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { MovementConfig } from '../movements';
import AppShell from './AppShell';
import MovementFigure from './MovementFigure';
import ScoreRing from './ScoreRing';
import { PoseCapture, type KneeAngleFrame } from '../services/poseCapture';
import { containMap, drawSkeleton, drawHud } from '../services/poseDraw';
import { getGaitBackend, type GaitResult } from '../services/gaitAnalysis';

type GaitState = 'idle' | 'loading' | 'ready' | 'recording' | 'analyzing' | 'done' | 'error';

interface Props {
  cfg: MovementConfig;
  /** for the "N / 10" step header, matching MovementCapture's convention */
  index: number;
  /** flat metrics from a PRIOR session's capture (session store / Dexie) —
   *  survives navigating away and back, unlike the rich GaitResult below. */
  storedMetrics: Record<string, number> | null;
  onComplete: (metrics: Record<string, number>, frameCount: number) => void;
  onRetry: () => void;
  onBack: () => void;
  onNext: () => void;
}

/**
 * Camera-based gait capture: getUserMedia -> PoseCapture (in-browser
 * MediaPipe, see services/poseCapture.ts) -> a per-frame knee-angle buffer
 * -> gaitAnalysis.ts's backend on Stop. Self-contained: unlike
 * flexion/sit-to-stand this has no BLE device, so it owns its own state
 * machine and footer instead of using useKneeFit/KneeFitControls.
 *
 * The <video> and <canvas> are ALWAYS mounted (hidden via CSS, never
 * conditionally removed from the tree) — a real bug on first hardware test
 * turned out to be exactly the opposite: the stream was attached to
 * videoRef while the element only existed conditionally during 'ready', so
 * the ref was null at attach time and nothing was ever actually displayed,
 * even though the camera light came on and detection kept "running" (on a
 * blank video). Keeping the element permanently mounted is what guarantees
 * the ref exists before startCamera() ever runs.
 */
export default function GaitCapture({ cfg, index, storedMetrics, onComplete, onRetry, onBack, onNext }: Props) {
  const { t } = useTranslation();
  const [state, setState] = useState<GaitState>(storedMetrics ? 'done' : 'idle');
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [frameCount, setFrameCount] = useState(0);
  const [tracking, setTracking] = useState(false);
  /** only set after a fresh analyze() THIS session — a stored capture from
   *  a prior visit has the flat metrics (storedMetrics) but not this. */
  const [freshResult, setFreshResult] = useState<GaitResult | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const poseRef = useRef<PoseCapture | null>(null);
  const framesRef = useRef<KneeAngleFrame[]>([]);
  const rafRef = useRef<number | null>(null);
  const startedAtRef = useRef(0);
  const tickRef = useRef(0);
  /** read inside the rAF loop closure without needing to restart it on
   *  Start/Stop — the loop itself never changes, only whether it buffers. */
  const recordingRef = useRef(false);

  const showCamera = state === 'loading' || state === 'ready' || state === 'recording';

  const stopStream = useCallback(() => {
    if (rafRef.current !== null) { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
    recordingRef.current = false;
    streamRef.current?.getTracks().forEach((tr) => tr.stop());
    streamRef.current = null;
    poseRef.current?.dispose();
    poseRef.current = null;
    const ctx = canvasRef.current?.getContext('2d');
    if (ctx && canvasRef.current) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  }, []);

  useEffect(() => () => stopStream(), [stopStream]);

  /** One continuous loop from the moment the camera+model are ready until
   *  the stream stops: draws the live skeleton/HUD every frame (so the
   *  operator can see tracking working BEFORE pressing Start walk), and
   *  additionally buffers a KneeAngleFrame whenever recordingRef is true. */
  function runPreviewLoop() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    const pose = poseRef.current;
    if (!video || !canvas || !wrap || !pose) return;

    const loop = () => {
      const now = performance.now();
      try {
        if (canvas.width !== wrap.clientWidth || canvas.height !== wrap.clientHeight) {
          canvas.width = wrap.clientWidth;
          canvas.height = wrap.clientHeight;
        }
        const ctx = canvas.getContext('2d');
        const tSec = recordingRef.current ? (now - startedAtRef.current) / 1000 : 0;
        const frame = pose.detect(video, tSec, now);

        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          if (frame.joints && video.videoWidth > 0) {
            const toPx = containMap(video.videoWidth, video.videoHeight, canvas.width, canvas.height);
            drawSkeleton(ctx, frame.joints, toPx);
          }
          drawHud(ctx, canvas.width, canvas.height, frame.left_knee_angle, frame.right_knee_angle, frame.joints !== null);
        }

        if (recordingRef.current) {
          framesRef.current.push({
            time_sec: frame.time_sec,
            left_knee_angle: frame.left_knee_angle,
            right_knee_angle: frame.right_knee_angle,
          });
          if (now - tickRef.current > 250) {
            tickRef.current = now;
            setElapsed(tSec);
            setFrameCount(framesRef.current.length);
          }
        }
        setTracking(frame.joints !== null);
      } catch {
        // one bad frame must never stop the loop — see poseCapture.ts's own
        // try/catch for why (verified against a fake video device: an
        // uncaught throw here silently kills the rest of the walk).
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
  }

  async function startCamera() {
    setError(null);
    setState('loading');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
      streamRef.current = stream;
      // Safe now: the <video> element is ALWAYS mounted (see component
      // doc comment), so this ref is guaranteed to exist here.
      if (!videoRef.current) throw new Error('video element not mounted');
      videoRef.current.srcObject = stream;
      await videoRef.current.play();

      const pose = new PoseCapture();
      await pose.init();
      poseRef.current = pose;
      setState('ready');
      runPreviewLoop();
    } catch (e) {
      stopStream();
      setError(e instanceof Error ? e.message : String(e));
      setState('error');
    }
  }

  function startWalk() {
    framesRef.current = [];
    setFrameCount(0);
    setElapsed(0);
    startedAtRef.current = performance.now();
    tickRef.current = 0;
    recordingRef.current = true;
    setState('recording');
  }

  async function stopWalk() {
    recordingRef.current = false;
    const frames = framesRef.current;
    setState('analyzing');
    try {
      const fps = frames.length > 1 ? frames.length / (frames[frames.length - 1].time_sec - frames[0].time_sec) : 30;
      const gaitResult = await getGaitBackend().analyze(frames, Number.isFinite(fps) && fps > 0 ? fps : 30);
      setFreshResult(gaitResult);
      onComplete(gaitResult.metrics, frames.length);
      setState('done');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setState('error');
    } finally {
      stopStream();
    }
  }

  function retry() {
    stopStream();
    setError(null);
    setFreshResult(null);
    onRetry();
    setState('idle');
  }

  const footer = (
    <>
      <button className="btn grow0" onClick={onBack}>{t('nav.back')}</button>
      {state === 'idle' && <button className="btn pri" onClick={() => void startCamera()}>{t('gait.startCamera')}</button>}
      {state === 'loading' && <button className="btn" disabled>{t('gait.hint.loading')}</button>}
      {state === 'ready' && <button className="btn pri" onClick={startWalk}>{t('gait.startWalk')}</button>}
      {state === 'recording' && <button className="btn danger" onClick={() => void stopWalk()}>{t('cap.stop')}</button>}
      {state === 'analyzing' && <button className="btn" disabled>{t('gait.analyzing')}</button>}
      {state === 'done' && <button className="btn grow0" onClick={retry}>{t('cap.retry')}</button>}
      {state === 'error' && <button className="btn grow0" onClick={retry}>{t('cap.retry')}</button>}
      <button className={state === 'done' ? 'btn pri' : 'btn pri grow0'} onClick={onNext}>{t('nav.next')}</button>
    </>
  );

  return (
    <AppShell title={t(`mv.${cfg.key}.title`)} step={`${index + 4} / 10`} footer={footer}>
    <div className="mv-kf">
      <div className="mv-kf-body">
        <div className="mv-kf-fig">
          {/* Always mounted (see doc comment) — just hidden when not in use,
              never removed, so videoRef/canvasRef are never null when
              startCamera()/the preview loop need them. */}
          <div ref={wrapRef} className="gait-cam-wrap" style={{ display: showCamera ? 'block' : 'none' }}>
            <video ref={videoRef} className="gait-cam-video" muted playsInline aria-label="Camera preview" />
            <canvas ref={canvasRef} className="gait-cam-canvas" />
          </div>
          {!showCamera && <MovementFigure media={cfg.media} state={state === 'done' ? 'review' : 'idle'} />}
        </div>

        <div className="mv-kf-panel">
          {state === 'idle' && (
            <div className="mv-kf-hint">
              <div className="instr">{t(`mv.${cfg.key}.instr`)}</div>
              <div className="small muted">{t('gait.hint.idle')}</div>
            </div>
          )}

          {state === 'loading' && (
            <div className="mv-kf-hint">
              <div className="small muted">{t('gait.hint.loading')}</div>
            </div>
          )}

          {state === 'ready' && (
            <div className="mv-kf-hint">
              <div className="instr">{t(`mv.${cfg.key}.instr`)}</div>
              <div className="small muted">{t('gait.hint.ready')}</div>
            </div>
          )}

          {state === 'recording' && (
            <div className="col gap8">
              <div className="row gap10" style={{ alignItems: 'center' }}>
                <span className={tracking ? 'ledot ok' : 'ledot bad'} />
                <span className="small">{tracking ? t('gait.tracking') : t('gait.noPose')}</span>
              </div>
              <div className="xs muted">{elapsed.toFixed(1)}s · {frameCount} {t('gait.frames')}</div>
            </div>
          )}

          {state === 'analyzing' && (
            <div className="mv-kf-hint">
              <div className="small muted">{t('gait.analyzing')}</div>
            </div>
          )}

          {state === 'done' && (freshResult || storedMetrics) && (
            <div className="col gap8">
              <ScoreRing
                percent={freshResult?.overallRiskScore ?? storedMetrics?.overall_risk_score ?? 0}
                size={96} stroke={10}
              />
              {freshResult && <div className="small"><b>{freshResult.riskLevel}</b></div>}
              {freshResult && <div className="xs muted">{freshResult.affectedSide}</div>}
              {!freshResult && <div className="xs muted">{t('gait.storedNote')}</div>}
            </div>
          )}

          {state === 'error' && (
            <div className="mv-kf-hint">
              <div className="small" style={{ color: 'var(--signal)' }}>{t('gait.error')}</div>
              {error && <div className="xs muted">{error}</div>}
            </div>
          )}
        </div>
      </div>

      <div className="status">{t(`gait.state.${state}`)}</div>
    </div>
    </AppShell>
  );
}
