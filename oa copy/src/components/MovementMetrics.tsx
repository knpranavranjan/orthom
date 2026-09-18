import { useTranslation } from 'react-i18next';
import type { MovementConfig } from '../movements';
import {
  formatMetric, primarySpec, readingImplausible, specsByTier,
  timingsUnavailable, visibleSecondary,
} from '../metrics';

interface Props {
  cfg: MovementConfig;
  metrics: Record<string, number>;
  angles: number[];
  quality: number;
  /** hide the engineering fields (used in the compact record view) */
  compact?: boolean;
}

/** One movement's results, rendered identically wherever they appear. */
export default function MovementMetrics({ cfg, metrics, angles, quality, compact }: Props) {
  const { t } = useTranslation();
  const primary = primarySpec(cfg);
  const gated = timingsUnavailable(cfg, metrics);
  const implausible = readingImplausible(cfg, metrics);
  const secondary = visibleSecondary(cfg, metrics);
  const qa = specsByTier(cfg, 'qa').filter((m) => typeof metrics[m.key] === 'number');
  const crepitus = metrics['crepitusCount'];
  const hasData = Object.keys(metrics).length > 0;

  return (
    <div className="card" style={{ marginBottom: 6 }}>
      <div className="row" style={{ justifyContent: 'space-between', marginBottom: 4 }}>
        <span className="h2">{t(`mv.${cfg.key}.title`)}</span>
        <span className="xs">{t('cap.quality')} {quality}%</span>
      </div>

      {!hasData && <div className="small muted">{t('sum.noResult')}</div>}

      {primary && typeof metrics[primary.key] === 'number' && (
        <div className="row" style={{ alignItems: 'baseline', gap: 8, marginBottom: 5 }}>
          <span className="bignum">{formatMetric(primary, metrics[primary.key])}</span>
          <span className="xs">{t(`m.${primary.key}`, { defaultValue: primary.key })}</span>
        </div>
      )}

      {gated && <div className="banner" style={{ marginBottom: 5 }}>{t('sum.noReps')}</div>}
      {implausible && <div className="banner" style={{ marginBottom: 5 }}>{t('sum.implausible')}</div>}

      {secondary.length > 0 && (
        <div className="mgrid">
          {secondary.map((m) => (
            <div className="mrow" key={m.key}>
              <span className="muted">{t(`m.${m.key}`, { defaultValue: m.key })}</span>
              <span className="num">{formatMetric(m, metrics[m.key])}</span>
            </div>
          ))}
        </div>
      )}

      {typeof crepitus === 'number' && (
        <div className="crep">
          <span className="muted small">{t('m.crepitusCount')}</span>
          {angles.length > 0 ? (
            <span className="row gap4" style={{ flexWrap: 'wrap', justifyContent: 'flex-end' }}>
              {/* Crepitus at specific angles is the diagnostic signal; a bare
                  count throws away where in the arc it happened. */}
              {angles.map((a, i) => (
                <span className="chip" key={i}>{a.toFixed(0)}°</span>
              ))}
            </span>
          ) : (
            <span className="num muted">{t('sum.noCrepitus')}</span>
          )}
        </div>
      )}

      {!compact && qa.length > 0 && (
        <div className="qaline">
          {qa.map((m) => (
            <span key={m.key}>
              {t(`m.${m.key}`, { defaultValue: m.key })} {formatMetric(m, metrics[m.key])}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
