import { useTranslation } from 'react-i18next';
import type { MovementConfig } from '../movements';
import type { KneeFitSnapshot } from '../services/kneefit/types';
import KneeFitLiveMetrics from './KneeFitLiveMetrics';

/**
 * Frozen result for a finished KneeFit attempt. The numbers are the device's
 * own; the wording stays "measured movement parameters", never a diagnosis.
 */
export default function KneeFitResult({
  cfg, snapshot,
}: { cfg: MovementConfig; snapshot: KneeFitSnapshot }) {
  const { t } = useTranslation();
  const hasData = Object.keys(snapshot.metrics).length > 0;

  return (
    <div className="kf-result">
      <div className="kf-result-head">✓ {t('kf.complete')}</div>
      {!hasData && <div className="small muted">{t('kf.noResult')}</div>}
      <KneeFitLiveMetrics cfg={cfg} snapshot={snapshot} mode="final" />
      <div className="xs muted kf-note">{t('kf.measuredNote')}</div>
    </div>
  );
}
