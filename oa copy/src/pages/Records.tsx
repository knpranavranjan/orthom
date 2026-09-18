import { useLiveQuery } from 'dexie-react-hooks';
import { useTranslation } from 'react-i18next';
import AppShell from '../components/AppShell';
import { db } from '../db';

export default function Records({ onBack, onOpen }: { onBack: () => void; onOpen: (reportId: string) => void }) {
  const { t } = useTranslation();

  const rows = useLiveQuery(async () => {
    const reports = await db.reports.orderBy('createdAt').reverse().limit(60).toArray();
    const patients = await db.patients.bulkGet(reports.map((r) => r.patientId));
    const encounters = await db.encounters.bulkGet(reports.map((r) => r.encounterId));
    return reports.map((r, i) => ({
      report: r,
      patient: patients[i],
      synced: encounters[i]?.synced === 1,
    }));
  }, []);

  return (
    <AppShell
      title={t('rec.title')}
      footer={<button className="btn pri" onClick={onBack}>{t('nav.home')}</button>}
    >
      <div className="grow" style={{ overflowY: 'auto', padding: '8px 12px' }}>
        {!rows?.length && <div className="empty">{t('rec.empty')}</div>}
        {!!rows?.length && <div className="xs" style={{ marginBottom: 6 }}>{t('rec.tapHint')}</div>}
        <div className="list">
          {rows?.map(({ report, patient, synced }) => (
            <button className="listrow" key={report.id} onClick={() => onOpen(report.id)}>
              <span className="col grow gap4" style={{ gap: 1 }}>
                <span style={{ fontWeight: 700 }}>
                  {patient?.name || patient?.id || '—'}
                </span>
                <span className="xs" style={{ textTransform: 'none', letterSpacing: 0 }}>
                  {patient?.id} · {new Date(report.createdAt).toLocaleDateString()}
                  {report.klGrade >= 0 ? ` · ${t('kl.grade')} ${report.klGrade}` : ''}
                </span>
              </span>
              <span className={`pill ${report.riskBand === 'low' ? 'ok' : report.riskBand === 'moderate' ? 'warn' : 'hi'}`}>
                {t(`risk.${report.riskBand}`)}
              </span>
              <span className={synced ? 'pill ok' : 'pill mut'}>
                {synced ? t('rec.synced') : t('rec.pending')}
              </span>
              <span className="chev" aria-hidden="true">›</span>
            </button>
          ))}
        </div>
      </div>
    </AppShell>
  );
}
