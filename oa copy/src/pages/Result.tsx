import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import AppShell from '../components/AppShell';
import ScoreRing from '../components/ScoreRing';
import { computeFunctionalScore } from '../services/functionalScore';
import { computeOverallRisk } from '../services/overallRisk';
import {
  buildReportHtml, downloadHtml, blobToDataUrl, type ReportInput,
} from '../services/report';
import { allStrings } from '../i18n';
import { useSession } from '../store/session';
import { db, uid } from '../db';

export default function Result({ onDone, onBack }: { onDone: () => void; onBack: () => void }) {
  const { t, i18n } = useTranslation();
  const s = useSession();
  const [saved, setSaved] = useState(false);
  const [busy, setBusy] = useState(false);

  const metrics = useMemo(() => {
    const out: Record<string, Record<string, number>> = {};
    for (const [k, v] of Object.entries(s.captures)) out[k] = v.metrics;
    return out;
  }, [s.captures]);

  const functional = useMemo(() => computeFunctionalScore({
    flexion: s.captures.flexion?.metrics,
    sit_to_stand: s.captures.sit_to_stand?.metrics,
    gait: s.captures.gait?.metrics,
  }), [s.captures]);

  const overall = useMemo(() => computeOverallRisk({
    clinicalScore: s.clinical?.clinicalScore ?? null,
    functionalPercent: functional?.percent ?? null,
    kl: s.kl,
  }), [s.clinical, functional, s.kl]);

  // Fallback only for the edge case of reaching this page with literally
  // nothing captured (shouldn't happen — clinical intake is mandatory
  // before the flow gets here) so save()/buildInput() always have a band.
  const band = overall?.band ?? 'low';

  async function save() {
    if (!s.encounterId || !s.patient) return;
    await db.reports.put({
      id: uid('rep'), encounterId: s.encounterId, patientId: s.patient.id,
      riskBand: band, klGrade: s.kl?.grade ?? -1, klDist: s.kl?.dist ?? [],
      manualScore: 0, createdAt: Date.now(),
      heatmap: s.kl?.heatmap ?? null,
      overallPercent: overall?.percent ?? null,
    });
    await db.encounters.update(s.encounterId, { completedAt: Date.now() });
    setSaved(true);
  }

  async function buildInput(): Promise<ReportInput | null> {
    if (!s.patient) return null;
    const quality: Record<string, number> = {};
    const angles: Record<string, number[]> = {};
    for (const [k, v] of Object.entries(s.captures)) {
      quality[k] = v.quality;
      angles[k] = v.angles;
    }
    return {
      patient: s.patient,
      encounterId: s.encounterId ?? '',
      locale: i18n.language,
      metrics, quality, angles,
      klGrade: s.kl?.grade ?? null,
      klDist: s.kl?.dist ?? null,
      riskBand: band,
      overallPercent: overall?.percent ?? null,
      xrayDataUrl: s.xrayBlob ? await blobToDataUrl(s.xrayBlob) : null,
      heatmapDataUrl: s.kl?.heatmap ?? null,
      clinical: s.clinical,
      strings: allStrings(),
    };
  }

  const stem = () =>
    `OA-${s.patient?.id ?? 'unknown'}-${new Date().toISOString().slice(0, 10)}`;

  async function savePdf() {
    const input = await buildInput();
    if (!input) return;
    setBusy(true);
    try {
      // ~600 kB of PDF libraries, fetched only when this button is pressed.
      const { downloadPdf } = await import('../services/pdf');
      await downloadPdf(input, `${stem()}.pdf`);
    } finally { setBusy(false); }
  }

  async function saveHtml() {
    const input = await buildInput();
    if (!input) return;
    downloadHtml(`${stem()}.html`, buildReportHtml(input));
  }

  return (
    <AppShell
      title={t('res.title')}
      step="10 / 10"
      footer={
        <>
          <button className="btn grow0" onClick={onBack}>{t('nav.back')}</button>
          <button className="btn pri" disabled={busy} onClick={() => void savePdf()}>
            {busy ? t('res.preparing') : t('res.downloadPdf')}
          </button>
          <button className="btn grow0" disabled={busy} onClick={() => void saveHtml()}>
            {t('res.downloadHtml')}
          </button>
          <button className="btn" disabled={saved} onClick={() => void save()}>
            {saved ? t('res.saved') : t('res.save')}
          </button>
          {saved && <button className="btn pri grow0" onClick={onDone}>{t('nav.done')}</button>}
        </>
      }
    >
      <div className="grow" style={{ overflowY: 'auto', padding: '8px 12px' }}>
        {overall && (
          <div className="score-card" style={{ marginBottom: 10 }}>
            <ScoreRing percent={overall.percent} />
            <div className="col gap8" style={{ flex: '1 1 auto', minWidth: 0 }}>
              <div className="xs">{t('res.overall')}</div>
              <div className="row gap10" style={{ alignItems: 'center' }}>
                <span className="score-num">{overall.percent}%</span>
                <span className={`pill ${band === 'low' ? 'ok' : band === 'moderate' ? 'warn' : 'hi'}`}>
                  {t(`risk.${band}`)}
                </span>
              </div>
              <div className="stat-row">
                <span>
                  {t('clinical.score')}:{' '}
                  <b>{overall.parts.clinical != null ? `${Math.round(overall.parts.clinical)}%` : t('res.notCaptured')}</b>
                </span>
                <span>
                  {t('func.score')}:{' '}
                  <b>{overall.parts.functional != null ? `${Math.round(overall.parts.functional)}%` : t('res.notCaptured')}</b>
                </span>
                <span>
                  {t('res.xrayScore')}:{' '}
                  <b>{overall.parts.xray != null ? `${Math.round(overall.parts.xray)}%` : t('res.notCaptured')}</b>
                </span>
              </div>
              <div className="xs muted" style={{ textTransform: 'none', letterSpacing: 0, fontWeight: 400 }}>
                {t('res.overallNote')}
              </div>
            </div>
          </div>
        )}

        <div className="row gap8" style={{ marginBottom: 6 }}>
          {s.patient && <span className="xs">{s.patient.id} · {s.patient.age}</span>}
          {s.kl && <span className="xs">{t('kl.grade')} {s.kl.grade}</span>}
        </div>

        <div className="small" style={{ marginBottom: 6 }}>{t(`risk.${band}.body`)}</div>

        <div className="xs">{t('guide.title')}</div>
        <div className="small muted" style={{ marginBottom: 6 }}>{t('guide.body')}</div>

        <div className="banner">{t('disclaimer')}</div>
      </div>
    </AppShell>
  );
}
