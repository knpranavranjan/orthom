import { useTranslation } from 'react-i18next';
import AppShell from '../components/AppShell';
import { useKneeFit } from '../hooks/useKneeFit';

export default function Home({ onStart, onRecords }: { onStart: () => void; onRecords: () => void }) {
  const { t } = useTranslation();
  const kf = useKneeFit();
  const conn = kf.snapshot.connection;

  const dot =
    conn.state === 'connected' ? 'ok' :
    conn.state === 'connecting' ? 'busy' :
    conn.state === 'error' ? 'bad' : 'mut';

  const label =
    conn.state === 'connected' ? t('kf.connected', { name: conn.deviceName ?? 'KneeFit' }) :
    conn.state === 'connecting' ? t('kf.connecting') :
    kf.availability === 'insecure-context' ? t('kf.insecure') :
    kf.availability === 'no-api' ? t('kf.unsupported') :
    conn.state === 'error' ? (conn.error ?? t('kf.error')) :
    t('kf.disconnected');

  const canConnect = kf.availability === 'ok' && conn.state !== 'connecting';

  return (
    <AppShell
      title={t('app.title')}
      footer={
        <>
          <button className="btn" onClick={onRecords}>{t('home.records')}</button>
          <button className="btn pri" onClick={onStart}>{t('home.start')}</button>
        </>
      }
    >
      <div className="pad col gap8">
        <div>
          <div className="h1">{t('home.title')}</div>
          <div className="small muted">{t('home.body')}</div>
        </div>

        {/* KneeFit link. Sits on the home screen because pairing is a
            once-per-session setup step, not a per-patient one. */}
        <div className="card row gap8" style={{ alignItems: 'center' }}>
          <span className={`ledot ${dot}`} />
          <span className="col grow" style={{ gap: 1, minWidth: 0 }}>
            <span className="small" style={{ fontWeight: 600 }}>{t('kf.title')}</span>
            <span className="xs" style={{ textTransform: 'none', letterSpacing: 0, fontWeight: 400 }}>
              {label}
            </span>
          </span>
          {conn.state === 'connected' ? (
            <button className="btn grow0" onClick={kf.disconnect}>{t('kf.disconnect')}</button>
          ) : conn.state === 'error' ? (
            <button className="btn pri grow0" disabled={!canConnect} onClick={() => void kf.reconnect()}>
              {t('kf.reconnect')}
            </button>
          ) : (
            <button
              className="btn pri grow0"
              disabled={!canConnect}
              onClick={() => void kf.addDevice()}
            >
              {conn.state === 'connecting' ? t('kf.connecting') : t('kf.addDevice')}
            </button>
          )}
        </div>

        <div className="spacer" />
        <div className="banner">{t('disclaimer')}</div>
      </div>
    </AppShell>
  );
}
