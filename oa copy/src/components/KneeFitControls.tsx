import { useTranslation } from 'react-i18next';
import type { UseKneeFit } from '../hooks/useKneeFit';
import type { KneeFitSessionState } from '../services/kneefit/types';

interface Props {
  kf: UseKneeFit;
  onBack: () => void;
  onNext: () => void;
  /** true once this movement has a stored capture */
  hasCapture: boolean;
  /** override the state the bar reflects (e.g. show a stored capture as FINISHED) */
  effectiveState?: KneeFitSessionState;
  /** page-level retry — clears the stored capture, then resets the session */
  onRetry?: () => void;
  /** false for movements the KneeFit firmware can't run (e.g. gait) — only
   *  Back / Next are offered so the operator can skip it */
  recordable?: boolean;
}

/**
 * State-driven action bar. Shows only the control valid for the current
 * session state, plus flow nav. Next is available in every state except the
 * two mid-action ones (CALIBRATING / RECORDING) so the operator can always
 * move on — skip a movement, or continue after a completed test.
 *
 *   DISCONNECTED        → Add device
 *   CONNECTED           → Calibrate
 *   CALIBRATING         → Hold still…            (no Next)
 *   CALIBRATED / READY  → Calibrate · Start
 *   RECORDING           → Stop                   (no Next)
 *   FINISHED            → Retake
 *   ERROR               → Reconnect · Retake
 */
export default function KneeFitControls({
  kf, onBack, onNext, hasCapture, effectiveState, onRetry, recordable = true,
}: Props) {
  const { t } = useTranslation();
  const st = effectiveState ?? kf.snapshot.session;
  const doRetry = onRetry ?? kf.retry;

  const Back = <button className="btn grow0" onClick={onBack}>{t('nav.back')}</button>;
  const Next = (
    <button className={hasCapture || !recordable ? 'btn pri' : 'btn pri grow0'} onClick={onNext}>
      {t('nav.next')}
    </button>
  );

  // Movement the device can't run → just let the operator move on.
  if (!recordable && st !== 'RECORDING' && st !== 'CALIBRATING') {
    return <>{Back}{Next}</>;
  }

  switch (st) {
    case 'CALIBRATING':
      return <>{Back}<button className="btn" disabled>{t('kf.holdStill')}</button></>;

    case 'RECORDING':
      return <>{Back}<button className="btn danger" onClick={() => void kf.finish()}>{t('kf.stop')}</button></>;

    case 'DISCONNECTED':
      return (
        <>
          {Back}
          <button className="btn pri" disabled={kf.availability !== 'ok'} onClick={() => void kf.addDevice()}>
            {t('kf.addDevice')}
          </button>
          {Next}
        </>
      );

    case 'CONNECTED':
      return (
        <>
          {Back}
          <button className="btn pri" onClick={() => void kf.calibrate()}>{t('cap.calibrate')}</button>
          {Next}
        </>
      );

    case 'CALIBRATED':
    case 'READY':
    case 'RETRYING':
      return (
        <>
          {Back}
          <button className="btn grow0" onClick={() => void kf.calibrate()}>{t('cap.calibrate')}</button>
          <button className="btn pri" onClick={() => void kf.start()}>{t('cap.start')}</button>
          {Next}
        </>
      );

    case 'FINISHED':
      return (
        <>
          {Back}
          <button className="btn" onClick={doRetry}>{t('cap.retry')}</button>
          {Next}
        </>
      );

    case 'ERROR':
      return (
        <>
          {Back}
          <button className="btn pri" onClick={() => void kf.reconnect()}>{t('kf.reconnect')}</button>
          <button className="btn grow0" onClick={doRetry}>{t('cap.retry')}</button>
          {Next}
        </>
      );

    default:
      return <>{Back}{Next}</>;
  }
}
