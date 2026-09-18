import { useEffect, useState } from 'react';
import { useLiveQuery } from 'dexie-react-hooks';
import { useTranslation } from 'react-i18next';
import AppShell from '../components/AppShell';
import KLDistribution from '../components/KLDistribution';
import { MOVEMENTS } from '../movements';
import { db } from '../db';
import { allStrings } from '../i18n';
import { blobToDataUrl, buildReportHtml, downloadHtml, type ReportInput } from '../services/report';
import MovementMetrics from '../components/MovementMetrics';

/** Everything recorded for one screening, rebuilt from the database. */
export default function PatientDetail({ reportId, onBack }: { reportId: string; onBack: () => void }) {
  const { t, i18n } = useTranslation();
  const [xrayUrl, setXrayUrl] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const data = useLiveQuery(async () => {
    const report = await db.reports.get(reportId);
    if (!report) return null;
    const patient = await db.patients.get(report.patientId);
    const encounter = await db.encounters.get(report.encounterId);
    const captures = await db.captures.where('encounterId').equals(report.encounterId).toArray();
    const xray = await db.xrays.where('encounterId').equals(report.encounterId).first();
    const clinical = await db.clinicalAssessments.where('encounterId').equals(report.encounterId).first();
    return { report, patient, encounter, captures, xray, clinical };
  }, [reportId]);

  useEffect(() => {
    if (!data?.xray) { setXrayUrl(null); return; }
    const url = URL.createObjectURL(data.xray.blob);
    setXrayUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [data?.xray?.id]);

  async function buildInput(): Promise<ReportInput | null> {
    if (!data?.report || !data.patient) return null;
    const metrics: Record<string, Record<string, number>> = {};
    const quality: Record<string, number> = {};
    const angles: Record<string, number[]> = {};
    for (const c of data.captures) {
      metrics[c.movementId] = c.metrics;
      quality[c.movementId] = c.quality;
      angles[c.movementId] = c.angles ?? [];
    }
    return {
      patient: data.patient,
      encounterId: data.report.encounterId,
      locale: data.encounter?.locale ?? i18n.language,
      metrics, quality, angles,
      klGrade: data.report.klGrade >= 0 ? data.report.klGrade : null,
      klDist: data.report.klDist.length ? data.report.klDist : null,
      riskBand: data.report.riskBand,
      overallPercent: data.report.overallPercent ?? null,
      xrayDataUrl: data.xray ? await blobToDataUrl(data.xray.blob) : null,
      heatmapDataUrl: data.report.heatmap ?? null,
      clinical: data.clinical ?? null,
      strings: allStrings(),
    };
  }

  const stem = () =>
    `OA-${data?.patient?.id ?? 'unknown'}-${new Date(data?.report?.createdAt ?? Date.now())
      .toISOString().slice(0, 10)}`;

  async function savePdf() {
    const input = await buildInput();
    if (!input) return;
    setBusy(true);
    try {
      const { downloadPdf } = await import('../services/pdf');
      await downloadPdf(input, `${stem()}.pdf`);
    } finally { setBusy(false); }
  }

  async function saveHtml() {
    const input = await buildInput();
    if (input) downloadHtml(`${stem()}.html`, buildReportHtml(input));
  }

  if (data === undefined) {
    return <AppShell title={t('det.title')}><div className="empty">…</div></AppShell>;
  }
  if (data === null) {
    return (
      <AppShell
        title={t('det.title')}
        footer={<button className="btn pri" onClick={onBack}>{t('nav.back')}</button>}
      >
        <div className="empty">{t('det.missing')}</div>
      </AppShell>
    );
  }

  const { report, patient, captures, clinical } = data;
  const band = report.riskBand;
  const byMovement = new Map(captures.map((c) => [c.movementId, c]));

  return (
    <AppShell
      title={patient?.name || patient?.id || t('det.title')}
      footer={
        <>
          <button className="btn grow0" onClick={onBack}>{t('nav.back')}</button>
          <button className="btn pri" disabled={busy} onClick={() => void savePdf()}>
            {busy ? t('res.preparing') : t('res.downloadPdf')}
          </button>
          <button className="btn grow0" disabled={busy} onClick={() => void saveHtml()}>
            {t('res.downloadHtml')}
          </button>
        </>
      }
    >
      <div className="grow" style={{ overflowY: 'auto', padding: '8px 12px' }}>
        {/* identity */}
        <div className="row gap8" style={{ marginBottom: 4 }}>
          <span className={`pill ${band === 'low' ? 'ok' : band === 'moderate' ? 'warn' : 'hi'}`}
                style={{ fontSize: 12, padding: '4px 10px' }}>
            {t(`risk.${band}`)}
          </span>
          {report.overallPercent != null && <span className="xs">{t('res.overall')} {report.overallPercent}%</span>}
          {report.klGrade >= 0 && <span className="xs">{t('kl.grade')} {report.klGrade}</span>}
        </div>

        <div className="small muted" style={{ marginBottom: 8 }}>
          {patient?.id}
          {patient ? ` · ${patient.age} · ${t(`sex.${patient.sex}`)}` : ''}
          {' · '}{new Date(report.createdAt).toLocaleString()}
        </div>

        {/* clinical */}
        {clinical && (
          <>
            <div className="xs" style={{ marginBottom: 4 }}>{t('report.clinical')}</div>
            <div className="qaline" style={{ marginBottom: 8, border: 'none', paddingTop: 0 }}>
              <span>{t('clinical.score')} {clinical.clinicalScore}%</span>
              <span>{t('clinical.pain')} {clinical.painScore}/20</span>
              <span>{t('clinical.stiffness')} {clinical.stiffnessScore}/8</span>
              <span>{t('clinical.function')} {clinical.functionScore}/68</span>
              {clinical.bmi !== null && <span>BMI {clinical.bmi}</span>}
            </div>
          </>
        )}

        {/* movements */}
        <div className="xs" style={{ marginBottom: 4 }}>{t('report.movements')}</div>
        {captures.length === 0 && (
          <div className="small muted" style={{ marginBottom: 8 }}>{t('report.noCaptures')}</div>
        )}
        {MOVEMENTS.map((m) => {
          const c = byMovement.get(m.id);
          if (!c) return null;
          return (
            <MovementMetrics
              key={m.id}
              cfg={m}
              metrics={c.metrics}
              angles={c.angles ?? []}
              quality={c.quality}
              compact
            />
          );
        })}

        {/* radiograph */}
        <div className="xs" style={{ margin: '8px 0 4px' }}>{t('report.radiograph')}</div>
        {!xrayUrl && <div className="small muted">{t('report.noXray')}</div>}
        {xrayUrl && (
          <div className="xr-wrap" style={{ height: 104, marginBottom: 6 }}>
            <div className="xr-box">
              <div className="xr-img"><img src={xrayUrl} alt="" /></div>
              <div className="xr-cap">{t('xr.original')}</div>
            </div>
            {report.heatmap && (
              <div className="xr-box">
                <div className="xr-img">
                  <img src={xrayUrl} alt="" />
                  <img className="overlay" src={report.heatmap} alt=""
                       style={{ opacity: 0.65, objectFit: 'contain' }} />
                </div>
                <div className="xr-cap">{t('xr.heatmap')}</div>
              </div>
            )}
          </div>
        )}

        {report.klDist.length > 0 && (
          <>
            <div className="xs" style={{ margin: '8px 0 4px' }}>{t('kl.title')}</div>
            <KLDistribution dist={report.klDist} />
          </>
        )}

        <div className="small muted" style={{ margin: '8px 0 6px' }}>{t(`risk.${band}.body`)}</div>
        <div className="banner">{t('disclaimer')}</div>
      </div>
    </AppShell>
  );
}
