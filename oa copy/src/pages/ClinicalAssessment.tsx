import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import AppShell from '../components/AppShell';
import ScoreRing from '../components/ScoreRing';
import { Segmented, Check } from '../components/Controls';
import { useSession } from '../store/session';
import { db, uid, type ClinicalAssessment as ClinicalAssessmentRec, type DurationBand } from '../db';
import { WOMAC_ITEMS, SEVERITY_LABELS, type WomacSubscale } from '../data/womac';
import { computeWomacScores, computeBmi, computeClinicalScore } from '../services/clinicalScore';

const DURATIONS: { value: DurationBand; key: string }[] = [
  { value: '<3m', key: 'clinical.dur.lt3m' },
  { value: '3-6m', key: 'clinical.dur.3to6m' },
  { value: '6-12m', key: 'clinical.dur.6to12m' },
  { value: '>1y', key: 'clinical.dur.gt1y' },
];

const SEVERITY = SEVERITY_LABELS.map((l, i) => ({ value: String(i), label: l }));

const SECTIONS: { subscale: WomacSubscale; key: string }[] = [
  { subscale: 'pain', key: 'clinical.section.pain' },
  { subscale: 'stiffness', key: 'clinical.section.stiffness' },
  { subscale: 'function', key: 'clinical.section.function' },
];

type History = { priorInjury: boolean; priorSurgery: boolean; priorMskProblem: boolean; none: boolean };
const emptyHistory: History = { priorInjury: false, priorSurgery: false, priorMskProblem: false, none: false };

/**
 * The clinical/WOMAC intake, as one scrollable page rather than one item per
 * screen: history + duration + all 24 WOMAC items, grouped by subscale, with
 * a live-computed clinical score revealed once every item is answered.
 */
export default function ClinicalAssessment({ onNext, onBack }: { onNext: () => void; onBack: () => void }) {
  const { t } = useTranslation();
  const draft = useSession((s) => s.draft);
  const encounterId = useSession((s) => s.encounterId);
  const setClinical = useSession((s) => s.setClinical);

  const [history, setHistory] = useState<History>(emptyHistory);
  const [duration, setDuration] = useState<DurationBand>('<3m');
  const [womac, setWomac] = useState<(number | null)[]>(() => Array(WOMAC_ITEMS.length).fill(null));

  const answered = womac.filter((v) => v !== null).length;
  const complete = answered === WOMAC_ITEMS.length;
  const bmi = useMemo(() => computeBmi(draft.heightCm, draft.weightKg), [draft.heightCm, draft.weightKg]);

  // A specific history flag turning on means "not none"; "none" turning on
  // clears the specific flags — the two are mutually exclusive, explicit
  // answers rather than an implied "none" from three unchecked boxes.
  function toggleFlag(key: 'priorInjury' | 'priorSurgery' | 'priorMskProblem', v: boolean) {
    setHistory((h) => ({ ...h, [key]: v, none: v ? false : h.none }));
  }
  function toggleNone(v: boolean) {
    setHistory(() => (v ? { ...emptyHistory, none: true } : { ...emptyHistory }));
  }

  const result = useMemo(() => {
    if (!complete) return null;
    const scores = computeWomacScores(womac as number[]);
    const { score, band } = computeClinicalScore({
      age: draft.age, bmi, history, durationBand: duration, womac: scores,
    });
    return { scores, score, band };
  }, [complete, womac, draft.age, bmi, history, duration]);

  async function submit() {
    if (!result || !encounterId) return;
    const rec: ClinicalAssessmentRec = {
      id: uid('cli'), encounterId,
      history, durationBand: duration,
      womacItems: womac as number[],
      painScore: result.scores.painScore,
      stiffnessScore: result.scores.stiffnessScore,
      functionScore: result.scores.functionScore,
      womacTotal: result.scores.womacTotal,
      bmi,
      clinicalScore: result.score,
      clinicalBand: result.band,
      instrumentVersion: 'womac-unlicensed-prototype',
      completedAt: Date.now(),
    };
    await db.clinicalAssessments.put(rec);
    setClinical(rec);
    onNext();
  }

  return (
    <AppShell
      title={t('clinical.title')}
      step="3 / 10"
      footer={
        <>
          <button className="btn grow0" onClick={onBack}>{t('nav.back')}</button>
          <button className="btn pri" disabled={!complete} onClick={() => void submit()}>
            {t('nav.next')}
          </button>
        </>
      }
    >
      <div className="pad col gap8">
        <div className="banner">{t('clinical.notValidated')}</div>

        <div>
          <div className="xs">{t('clinical.history')}</div>
          <div className="col gap4" style={{ marginTop: 4 }}>
            <Check on={history.none} label={t('clinical.noneOfAbove')}
                   onChange={toggleNone} />
            <Check on={history.priorInjury} label={t('clinical.priorInjury')}
                   onChange={(v) => toggleFlag('priorInjury', v)} />
            <Check on={history.priorSurgery} label={t('clinical.priorSurgery')}
                   onChange={(v) => toggleFlag('priorSurgery', v)} />
            <Check on={history.priorMskProblem} label={t('clinical.priorMsk')}
                   onChange={(v) => toggleFlag('priorMskProblem', v)} />
          </div>
        </div>

        <div>
          <div className="xs">{t('clinical.duration')}</div>
          <div style={{ marginTop: 4 }}>
            <Segmented
              value={duration}
              options={DURATIONS.map((d) => ({ value: d.value, label: t(d.key) }))}
              onChange={setDuration}
            />
          </div>
        </div>

        <div>
          <div className="h2">{t('clinical.womacTitle')}</div>
          <div className="small muted">{t('clinical.womacSubtitle')}</div>
        </div>

        {SECTIONS.map((sec) => (
          <div key={sec.subscale}>
            <div className="xs">{t(sec.key)}</div>
            <div className="col gap6" style={{ marginTop: 4 }}>
              {WOMAC_ITEMS.map((item, i) => item.subscale === sec.subscale && (
                <div key={item.id} className="card col gap4">
                  <div className="small" style={{ fontWeight: 600 }}>{item.text}</div>
                  <Segmented
                    value={womac[i] === null ? '' : String(womac[i])}
                    options={SEVERITY}
                    onChange={(v) => setWomac((w) => { const n = [...w]; n[i] = Number(v); return n; })}
                  />
                </div>
              ))}
            </div>
          </div>
        ))}

        {result ? (
          <div className="score-card">
            <ScoreRing percent={result.score} />
            <div className="col gap8" style={{ flex: '1 1 auto', minWidth: 0 }}>
              <div className="xs">{t('clinical.score')}</div>
              <div className="row gap10" style={{ alignItems: 'center' }}>
                <span className="score-num">{result.score}%</span>
                <span className={`pill ${result.band === 'low' ? 'ok' : result.band === 'moderate' ? 'warn' : 'hi'}`}>
                  {t(`risk.${result.band}`)}
                </span>
              </div>
              <div className="stat-row">
                <span>{t('clinical.totalQuestions')}: <b>{WOMAC_ITEMS.length}</b></span>
                <span>{t('clinical.attended')}: <b>{answered}</b></span>
              </div>
              <div className="stat-row">
                <span>{t('clinical.pain')} <b>{result.scores.painScore}/20</b></span>
                <span>{t('clinical.stiffness')} <b>{result.scores.stiffnessScore}/8</b></span>
                <span>{t('clinical.function')} <b>{result.scores.functionScore}/68</b></span>
                {bmi !== null && <span>BMI <b>{bmi}</b></span>}
              </div>
            </div>
          </div>
        ) : (
          <div className="small muted center">
            {t('clinical.incomplete', { n: answered, total: WOMAC_ITEMS.length })}
          </div>
        )}
      </div>
    </AppShell>
  );
}
