import { useTranslation } from 'react-i18next';
import type { UseKneeFit } from '../hooks/useKneeFit';

/**
 * KneeFit link status + the Add Device / Disconnect / Reconnect control.
 * One line on the 3.5" panel: dot · label · button.
 */
export default function KneeFitConnection({ kf }: { kf: UseKneeFit }) {
  const { t } = useTranslation();
  const { connection } = kf.snapshot;
  const s = connection.state;

  const dot = s === 'connected' ? 'ok' : s === 'connecting' ? 'busy' : s === 'error' ? 'bad' : 'mut';
  const label =
    s === 'connected' ? t('kf.connected', { name: connection.deviceName ?? 'KneeFit' }) :
    s === 'connecting' ? t('kf.connecting') :
    s === 'error' ? (connection.error ?? t('kf.error')) :
    kf.availability === 'insecure-context' ? t('kf.insecure') :
    kf.availability === 'no-api' ? t('kf.unsupported') :
    t('kf.disconnected');

  return (
    <div className="kf-conn">
      <span className={`ledot ${dot}`} />
      <span className="kf-conn-label">{label}</span>
      {s === 'connected' ? (
        <button className="btn grow0 kf-btn-sm" onClick={kf.disconnect}>{t('kf.disconnect')}</button>
      ) : s === 'error' ? (
        <button className="btn pri grow0 kf-btn-sm" onClick={() => void kf.reconnect()}>
          {t('kf.reconnect')}
        </button>
      ) : (
        <button
          className="btn pri grow0 kf-btn-sm"
          disabled={kf.availability !== 'ok' || s === 'connecting'}
          onClick={() => void kf.addDevice()}
        >
          {s === 'connecting' ? t('kf.connecting') : t('kf.addDevice')}
        </button>
      )}
    </div>
  );
}
