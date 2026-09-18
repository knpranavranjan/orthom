import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import AppShell from '../components/AppShell';
import ScoreRing from '../components/ScoreRing';
import MovementMetrics from '../components/MovementMetrics';
import { MOVEMENTS } from '../movements';
import { useSession } from '../store/session';
import { computeFunctionalScore } from '../services/functionalScore';

export default function SensorSummary({ onNext, onBack }: { onNext: () => void; onBack: () => void }) {
  const { t } = useTranslation();
  const captures = useSession((s) => s.captures);
  const any = Object.keys(captures).length > 0;

  const functional = useMemo(() => computeFunctionalScore({
    flexion: captures.flexion?.metrics,
    sit_to_stand: captures.sit_to_stand?.metrics,
    gait: captures.gait?.metrics,
  }), [captures]);

  return (
    <AppShell
      title={t('sum.title')}
      step="7 / 10"
      footer={
        <>
          <button className="btn grow0" onClick={onBack}>{t('nav.back')}</button>
          <button className="btn pri" onClick={onNext}>{t('nav.next')}</button>
        </>
      }
    >
      <div className="pad col gap10">
        {!any && <div className="empty">{t('sum.noData')}</div>}

        {functional && (
          <div className="score-card">
            <ScoreRing percent={functional.percent} />
            <div className="col gap8" style={{ flex: '1 1 auto', minWidth: 0 }}>
              <div className="xs">{t('func.score')}</div>
              <div className="row gap10" style={{ alignItems: 'center' }}>
                <span className="score-num">{functional.percent}%</span>
                <span className={`pill ${functional.band === 'low' ? 'ok' : functional.band === 'moderate' ? 'warn' : 'hi'}`}>
                  {t(`risk.${functional.band}`)}
                </span>
              </div>
              <div className="stat-row">
                {functional.flexion && (
                  <span>{t('mv.flexion.title')}: <b>{functional.flexion.percent}%</b></span>
                )}
                {functional.sitToStand && (
                  <span>{t('mv.sit_to_stand.title')}: <b>{functional.sitToStand.percent}%</b></span>
                )}
                {functional.gait && (
                  <span>{t('mv.gait.title')}: <b>{functional.gait.percent}%</b></span>
                )}
              </div>
            </div>
          </div>
        )}

        {MOVEMENTS.map((m) => {
          const c = captures[m.id];
          if (!c) return null;
          return (
            <MovementMetrics
              key={m.id}
              cfg={m}
              metrics={c.metrics}
              angles={c.angles}
              quality={c.quality}
            />
          );
        })}
      </div>
    </AppShell>
  );
}
