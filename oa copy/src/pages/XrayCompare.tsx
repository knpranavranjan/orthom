import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import AppShell from '../components/AppShell';
import XrayReport from '../components/XrayReport';
import { computeFunctionalScore } from '../services/functionalScore';
import { computeOverallRisk } from '../services/overallRisk';
import { buildReportInput } from '../services/report';
import { allStrings } from '../i18n';
import { useSession } from '../store/session';

export default function XrayCompare({ onNext, onBack }: { onNext: () => void; onBack: () => void }) {
  const { t, i18n } = useTranslation();
  const s = useSession();
  const [pdfBusy, setPdfBusy] = useState(false);

  // Same 3-way combination as the final Result screen (computeOverallRisk),
  // so a PDF exported from here and one exported from Result never disagree.
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

  async function exportPdf() {
    const input = await buildReportInput({
      patient: s.patient, encounterId: s.encounterId, locale: i18n.language,
      captures: s.captures, klGrade: s.kl?.grade ?? null, klDist: s.kl?.dist ?? null,
      riskBand: overall?.band ?? 'low', overallPercent: overall?.percent ?? null,
      xrayBlob: s.xrayBlob, heatmapDataUrl: s.kl?.heatmap ?? null,
      clinical: s.clinical,
      strings: allStrings(),
    });
    if (!input) return;
    setPdfBusy(true);
    try {
      const { downloadPdf } = await import('../services/pdf');
      const stem = `OA-${s.patient?.id ?? 'unknown'}-${new Date().toISOString().slice(0, 10)}`;
      await downloadPdf(input, `${stem}.pdf`);
    } finally {
      setPdfBusy(false);
    }
  }

  return (
    <AppShell
      title={t('xr.heatmap')}
      step="9 / 10"
      footer={
        <>
          <button className="btn grow0" onClick={onBack}>{t('nav.back')}</button>
          <button className="btn" disabled={!s.kl || pdfBusy} onClick={() => void exportPdf()}>
            {pdfBusy ? t('res.preparing') : t('xr.exportPdf')}
          </button>
          <button className="btn pri" onClick={onNext}>{t('nav.next')}</button>
        </>
      }
    >
      {!s.xrayUrl
        ? <div className="empty">{t('xr.needImage')}</div>
        : <XrayReport
            imageUrl={s.xrayUrl}
            heatmapUrl={s.kl?.heatmap ?? null}
            kl={s.kl}
            patient={s.patient}
            scanAt={s.xrayAt}
          />}
    </AppShell>
  );
}
