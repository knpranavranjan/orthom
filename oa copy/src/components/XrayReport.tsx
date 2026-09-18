import { useTranslation } from 'react-i18next';
import type { Patient } from '../db';
import type { KLResult } from '../services/inference';

interface Props {
  imageUrl: string | null;
  heatmapUrl: string | null;
  kl: KLResult | null;
  patient: Patient | null;
  scanAt: number | null;
  /** 'images' = two large tiles only (upload step);
   *  'split'  = tiles + the patient / result panel (review step). */
  layout?: 'images' | 'split';
  /** show a transient "analysing" state */
  pending?: boolean;
  rejectMsg?: string | null;
}

/** KL grade -> condition severity class (coloured pill, like the reference). */
function condClass(grade: number): string {
  if (grade <= 1) return 'ok';
  if (grade === 2) return 'warn';
  return 'hi';
}

/**
 * The X-ray page body — ORIGINAL · HEAT MAP (+ optional info panel), laid out
 * like the reference kiosk but in the app's own (light) theme.
 *
 *   layout="images" : the upload step — two large tiles, so the input and the
 *                     Grad-CAM output read clearly. One-line status underneath.
 *   layout="split"  : the review step — tiles + PATIENT INFO / CONDITION /
 *                     CONFIDENCE / OA bars on the right.
 */
export default function XrayReport({
  imageUrl, heatmapUrl, kl, patient, scanAt, layout = 'split', pending, rejectMsg,
}: Props) {
  const { t, i18n } = useTranslation();

  const scanStr = scanAt
    ? new Date(scanAt).toLocaleString(i18n.language, {
        day: '2-digit', month: 'short', year: '2-digit',
        hour: '2-digit', minute: '2-digit',
      })
    : '—';

  const pOA = kl?.oaScreen?.pOA;

  const tiles = (
    <div className="xr-tiles">
      <figure className="xr-tile">
        <div className={`xr-img ${rejectMsg ? 'xr-img-warn' : ''}`}>
          {rejectMsg
            ? <span className="xr-tile-msg">⚠ {rejectMsg}</span>
            : imageUrl
              ? <img src={imageUrl} alt={t('xr.original')} />
              : <span className="xr-tile-msg muted">{t('xr.none')}</span>}
        </div>
        <figcaption>{t('xr.original')}</figcaption>
      </figure>
      <figure className="xr-tile">
        <div className="xr-img">
          {imageUrl && !rejectMsg && <img src={imageUrl} alt="" />}
          {heatmapUrl && !rejectMsg && (
            <img className="overlay" src={heatmapUrl} alt="" style={{ objectFit: 'contain' }} />
          )}
          {(!imageUrl || rejectMsg) && <span className="xr-tile-msg muted">—</span>}
          {imageUrl && !heatmapUrl && !rejectMsg && pending && <span className="spin" aria-hidden />}
        </div>
        <figcaption>{t('xr.heatmap')}</figcaption>
      </figure>
    </div>
  );

  if (layout === 'images') {
    const status = (() => {
      if (pending) return { cls: 'muted', node: <><span className="spin" aria-hidden /> {t('xr.grading')}</> };
      if (rejectMsg) return { cls: 'bad', node: <>⚠ {rejectMsg}</> };
      if (kl) {
        const pct = kl.confidence != null ? ` · ${Math.round(kl.confidence * 100)}%` : '';
        return { cls: 'ok', node: <>{(kl.klName ?? `${t('kl.grade')} ${kl.grade}`) + pct} — {t('xr.gradedHint')}</> };
      }
      return { cls: 'muted', node: <>{t('xr.chooseHint')}</> };
    })();
    return (
      <div className="xr-report xr-report--images">
        {tiles}
        <div className={`xr-status ${status.cls}`}>{status.node}</div>
      </div>
    );
  }

  // review step — big images on top, a compact result strip beneath.
  const oaPct = pOA != null ? Math.round(pOA * 100) : null;
  return (
    <div className="xr-report xr-report--review">
      {tiles}

      <div className="xr-resultbar">
        <div className="xr-rb-meta">
          <b>{patient?.id ?? '—'}</b>
          <span>
            {patient ? `${patient.age} · ${t(`sex.${patient.sex}`)}` : '—'}
            {' · '}{t('xr.modalityVal')}{' · '}{scanStr}
          </span>
        </div>

        {pending && (
          <div className="xr-rb-state"><span className="spin" aria-hidden /> {t('xr.grading')}</div>
        )}
        {rejectMsg && !pending && <div className="xr-rb-state bad">⚠ {rejectMsg}</div>}

        {!pending && !rejectMsg && kl && (
          <div className="xr-rb-result">
            <span className={`xr-cond ${condClass(kl.grade)}`}>
              {kl.klName ?? `${t('kl.grade')} ${kl.grade}`}
            </span>
            <span className="xr-conf">
              {kl.confidence != null ? `${(kl.confidence * 100).toFixed(2)}%` : '—'}
              <em>{t('xr.confidence')}</em>
            </span>
            {oaPct != null && (
              <span className="xr-oa">
                <i><b className="hi" style={{ width: `${oaPct}%` }} /></i>
                <em>{t('xr.oa')} {oaPct}%</em>
              </span>
            )}
            {kl.abstain && <span className="xr-rb-abstain">⚠ {t('xr.abstain')}</span>}
          </div>
        )}

        {!pending && !rejectMsg && !kl && (
          <div className="xr-rb-state muted">{t('xr.chooseHint')}</div>
        )}
      </div>
    </div>
  );
}
