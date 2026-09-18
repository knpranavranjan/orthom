import { useTranslation } from 'react-i18next';
import type { MovementConfig } from '../movements';
import { formatMetric, primarySpec } from '../metrics';
import { deviceDurationMs } from '../services/kneefit/protocol';
import type { KneeFitSnapshot } from '../services/kneefit/types';

interface Props {
  cfg: MovementConfig;
  snapshot: KneeFitSnapshot;
  /** 'live' during RECORDING, 'final' after FINISH */
  mode: 'live' | 'final';
}

function mmss(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

/**
 * KneeFit metrics for the current attempt — real values only.
 *
 * The big number is the movement's primary field (ROM for flexion,
 * repetitions for sit-to-stand). While RECORDING it prefers the live frame
 * value; otherwise it shows the device's own result metric. Nothing is
 * fabricated: a field with no value shows "—".
 */
export default function KneeFitLiveMetrics({ cfg, snapshot, mode }: Props) {
  const { t } = useTranslation();
  const primary = primarySpec(cfg);
  const { live, metrics, crepitusAngles, elapsed } = snapshot;

  const liveMetric = cfg.live.metric;
  const headlineValue =
    mode === 'live' && typeof live[liveMetric] === 'number'
      ? live[liveMetric]
      : primary && typeof metrics[primary.key] === 'number'
        ? metrics[primary.key]
        : null;
  const headlineUnit = mode === 'live' && typeof live[liveMetric] === 'number'
    ? cfg.live.unit
    : primary?.unit ?? '';
  const headlineLabel = mode === 'live' && typeof live[liveMetric] === 'number'
    ? t(`m.${liveMetric}`, { defaultValue: liveMetric })
    : primary ? t(`m.${primary.key}`, { defaultValue: primary.key }) : '';

  // Show every field the rig reports, verbatim — no gating, no interpretation.
  const rows = cfg.metrics
    .filter((m) => m.tier !== 'primary' && m.key !== 'crepitusCount')
    .filter((m) => typeof metrics[m.key] === 'number');

  const crepitusCount = metrics['crepitusCount'];

  return (
    <div className="kf-metrics">
      <div className="kf-head">
        <span className="kf-test">{t(`mv.${cfg.key}.title`)}</span>
        {(mode === 'live' || elapsed > 0) && (
          <span className={mode === 'live' ? 'kf-timer live' : 'kf-timer'}>
            {mode === 'live' && <span className="kf-rec-dot" />}
            {mmss(elapsed)}
            {mode === 'live' && ` / ${mmss(deviceDurationMs(cfg.id) / 1000)}`}
          </span>
        )}
      </div>

      <div className="kf-primary">
        <span className="kf-primary-num">
          {headlineValue === null ? '—' : headlineValue.toFixed(primary?.decimals ?? 1)}
          {headlineValue !== null && <span className="kf-primary-unit">{headlineUnit}</span>}
        </span>
        <span className="kf-primary-label">{headlineLabel}</span>
      </div>

      {rows.length > 0 && (
        <div className="kf-grid">
          {rows.map((m) => (
            <div className="kf-row" key={m.key}>
              <span className="kf-row-k">{t(`m.${m.key}`, { defaultValue: m.key })}</span>
              <span className="kf-row-v">{formatMetric(m, metrics[m.key])}</span>
            </div>
          ))}
        </div>
      )}

      {typeof crepitusCount === 'number' && (
        <div className="kf-row kf-crep">
          <span className="kf-row-k">{t('m.crepitusCount')}</span>
          {crepitusAngles.length > 0 ? (
            <span className="kf-chips">
              {crepitusAngles.slice(0, 8).map((a, i) => (
                <span className="chip" key={i}>{a.toFixed(0)}°</span>
              ))}
              {crepitusAngles.length > 8 && <span className="chip">+{crepitusAngles.length - 8}</span>}
            </span>
          ) : (
            <span className="kf-row-v">{crepitusCount}</span>
          )}
        </div>
      )}
    </div>
  );
}
