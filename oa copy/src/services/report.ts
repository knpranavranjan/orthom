/**
 * Summary export — PDF and standalone HTML from one source.
 *
 * Why the PDF is rasterised rather than typeset with jsPDF text calls:
 * jsPDF does no complex text shaping. Devanagari written through its text
 * API comes out with broken conjuncts and misplaced matras even after the
 * font is embedded — and you do not notice until a Hindi reader opens it.
 * Rendering the report in the browser and placing that raster into the PDF
 * keeps the shaping the browser already got right, in every language.
 */
import type { ClinicalAssessment, Patient, RiskBand } from '../db';
import { MOVEMENTS } from '../movements';
import { formatMetric as fmtSpec, timingsUnavailable, visibleSecondary, primarySpec, specsByTier } from '../metrics';
import { computeFunctionalScore } from './functionalScore';

export interface ReportInput {
  patient: Patient;
  encounterId: string;
  locale: string;
  metrics: Record<string, Record<string, number>>;
  /** movementId -> crepitus angles */
  angles?: Record<string, number[]>;
  quality: Record<string, number>;
  klGrade: number | null;
  klDist: number[] | null;
  riskBand: RiskBand;
  /** the 3-way combined score (clinical + functional + x-ray) shown on the Result screen */
  overallPercent: number | null;
  xrayDataUrl: string | null;
  heatmapDataUrl: string | null;
  clinical?: ClinicalAssessment | null;
  strings: Record<string, string>;
}

const esc = (s: string) =>
  s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c] as string));

const DURATION_KEY: Record<string, string> = {
  '<3m': 'clinical.dur.lt3m', '3-6m': 'clinical.dur.3to6m',
  '6-12m': 'clinical.dur.6to12m', '>1y': 'clinical.dur.gt1y',
};

/** Every rule is scoped to .oa-report so the same CSS serves the standalone
 *  file and the offscreen node we rasterise, with no leakage into the app. */
export const REPORT_CSS = `
.oa-report{font-family:system-ui,"Noto Sans","Noto Sans Devanagari",sans-serif;
  width:760px;margin:0 auto;padding:34px 30px;color:#12191b;line-height:1.55;
  background:#fff;box-sizing:border-box;font-size:14px}
.oa-report h1{font-size:23px;margin:0 0 6px;line-height:1.25}
.oa-report h2{font-size:15px;margin:24px 0 8px;border-bottom:1px solid #d5dcda;
  padding-bottom:5px;line-height:1.3}
.oa-report .sec{break-inside:avoid}
.oa-report .sec > h2:first-child{margin-top:20px}
.oa-report h3{font-size:13.5px;margin:15px 0 5px;display:flex;
  justify-content:space-between;align-items:baseline;line-height:1.35}
.oa-report .q{font-size:11.5px;font-weight:400;color:#66777a}
.oa-report table{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:6px}
.oa-report td{padding:5px 8px;border-bottom:1px solid #eceff0}
.oa-report td.n{text-align:right;font-variant-numeric:tabular-nums;
  font-weight:600;white-space:nowrap}
.oa-report .meta{font-size:13px;color:#3b4a4d;margin:0 0 3px}
.oa-report .band{display:inline-block;padding:5px 14px;border-radius:14px;
  font-weight:700;font-size:14px}
.oa-report .low{background:#e4f0ea;color:#2f6b4f}
.oa-report .moderate{background:#fbf1dd;color:#8a6a15}
.oa-report .high{background:#fbede4;color:#b4541f}
.oa-report figure{margin:0;flex:1}
.oa-report figure img{width:100%;max-height:230px;object-fit:contain;border-radius:4px;
  background:#0a0f11;display:block}
.oa-report figcaption{font-size:11px;color:#66777a;text-align:center;margin-top:5px;
  text-transform:uppercase;letter-spacing:.06em}
.oa-report .imgs{display:flex;gap:14px;margin:12px 0}
.oa-report .disc{margin-top:24px;padding:12px 14px;border:1px solid #b4541f;
  background:#fbede4;color:#b4541f;border-radius:4px;font-size:12.5px;font-weight:600}
.oa-report .muted{color:#66777a;font-size:13px}
.oa-report tr.qa td{color:#8b9a9c;font-size:11.5px}
`;

/** The report content, without the page chrome. */
export function reportBody(r: ReportInput): string {
  const S = r.strings;
  const rows = MOVEMENTS.map((cfg) => {
    const mm = r.metrics[cfg.id];
    if (!mm) return '';
    const angles = r.angles?.[cfg.id] ?? [];
    const primary = primarySpec(cfg);
    const gated = timingsUnavailable(cfg, mm);

    const lines: string[] = [];
    if (primary && typeof mm[primary.key] === 'number') {
      lines.push(`<tr><td><b>${esc(S['m.' + primary.key] ?? primary.key)}</b></td>
        <td class="n"><b>${fmtSpec(primary, mm[primary.key])}</b></td></tr>`);
    }
    for (const spec of visibleSecondary(cfg, mm)) {
      lines.push(`<tr><td>${esc(S['m.' + spec.key] ?? spec.key)}</td>
        <td class="n">${fmtSpec(spec, mm[spec.key])}</td></tr>`);
    }
    if (typeof mm['crepitusCount'] === 'number') {
      const at = angles.length
        ? angles.map((a) => `${a.toFixed(0)}\u00b0`).join(', ')
        : esc(S['sum.noCrepitus'] ?? 'None detected');
      lines.push(`<tr><td>${esc(S['m.crepitusCount'] ?? 'Crepitus')}</td>
        <td class="n">${mm['crepitusCount']}${angles.length ? ` &nbsp;(${at})` : ` — ${at}`}</td></tr>`);
    }
    for (const spec of specsByTier(cfg, 'qa')) {
      if (typeof mm[spec.key] !== 'number') continue;
      lines.push(`<tr class="qa"><td>${esc(S['m.' + spec.key] ?? spec.key)}</td>
        <td class="n">${fmtSpec(spec, mm[spec.key])}</td></tr>`);
    }

    return `<h3><span>${esc(S['mv.' + cfg.key + '.title'] ?? cfg.id)}</span>
      <span class="q">${esc(S['report.quality'] ?? 'Baseline')} ${r.quality[cfg.id] ?? '\u2014'}%</span></h3>
      ${gated ? `<p class="muted">${esc(S['sum.noReps'] ?? '')}</p>` : ''}
      <table>${lines.join('')}</table>`;
  }).join('');

  const kl = r.klDist
    ? `<table>${r.klDist
        .map((p, i) => `<tr><td>${esc(S['kl.grade'] ?? 'Grade')} ${i}</td>
          <td class="n">${(p * 100).toFixed(1)}%</td></tr>`)
        .join('')}</table>`
    : `<p class="muted">${esc(S['report.noXray'] ?? 'No radiograph provided.')}</p>`;

  const imgs = [
    r.xrayDataUrl
      ? `<figure><img src="${r.xrayDataUrl}" alt=""><figcaption>${esc(S['xr.original'] ?? 'Original')}</figcaption></figure>`
      : '',
    r.heatmapDataUrl
      ? `<figure><img src="${r.heatmapDataUrl}" alt=""><figcaption>${esc(S['xr.heatmap'] ?? 'Heat map')}</figcaption></figure>`
      : '',
  ].join('');

  return `
<h1>${esc(S['report.title'] ?? 'OA Screening Summary')}</h1>
<p class="meta"><b>${esc(S['reg.name'] ?? 'Name')}:</b> ${esc(r.patient.name || '—')} &nbsp;·&nbsp;
  <b>${esc(S['reg.id'] ?? 'ID')}:</b> ${esc(r.patient.id)}</p>
<p class="meta"><b>${esc(S['reg.age'] ?? 'Age')}:</b> ${r.patient.age} &nbsp;·&nbsp;
  <b>${esc(S['reg.sex'] ?? 'Sex')}:</b> ${esc(S['sex.' + r.patient.sex] ?? r.patient.sex)} &nbsp;·&nbsp;
  <b>${esc(S['report.date'] ?? 'Date')}:</b> ${new Date().toLocaleString()}</p>

<section class="sec">
<h2>${esc(S['report.outcome'] ?? 'Screening outcome')}</h2>
<p><span class="band ${r.riskBand}">${esc(S['risk.' + r.riskBand] ?? r.riskBand)}</span>
${r.overallPercent !== null ? ` <b>${r.overallPercent}%</b>` : ''}</p>
${r.overallPercent !== null
    ? `<p class="meta">${esc(S['res.overallNote'] ?? '')}</p>`
    : ''}
${r.klGrade !== null
    ? `<p class="meta">${esc(S['kl.grade'] ?? 'KL grade')}: <b>${r.klGrade}</b></p>`
    : ''}
<p class="muted">${esc(S['risk.' + r.riskBand + '.body'] ?? '')}</p>
</section>

<section class="sec">
<h2>${esc(S['report.clinical'] ?? 'Clinical findings')}</h2>
${r.clinical ? `
<p class="muted">${esc(S['clinical.notValidated'] ?? '')}</p>
<table>
<tr><td>${esc(S['clinical.score'] ?? 'Clinical score')}</td>
  <td class="n">${r.clinical.clinicalScore}% <span class="band ${r.clinical.clinicalBand}" style="font-size:11px;padding:2px 8px">${esc(S['risk.' + r.clinical.clinicalBand] ?? r.clinical.clinicalBand)}</span></td></tr>
<tr><td>${esc(S['clinical.pain'] ?? 'Pain')}</td><td class="n">${r.clinical.painScore} / 20</td></tr>
<tr><td>${esc(S['clinical.stiffness'] ?? 'Stiffness')}</td><td class="n">${r.clinical.stiffnessScore} / 8</td></tr>
<tr><td>${esc(S['clinical.function'] ?? 'Function')}</td><td class="n">${r.clinical.functionScore} / 68</td></tr>
${r.clinical.bmi !== null ? `<tr><td>BMI</td><td class="n">${r.clinical.bmi}</td></tr>` : ''}
<tr><td>${esc(S['clinical.duration'] ?? 'Duration')}</td><td class="n">${esc(S[DURATION_KEY[r.clinical.durationBand]] ?? r.clinical.durationBand)}</td></tr>
</table>` : `<p class="muted">${esc(S['report.noClinical'] ?? 'No clinical assessment recorded.')}</p>`}
</section>

<section class="sec">
<h2>${esc(S['report.movements'] ?? 'Movement assessment')}</h2>
${(() => {
  const fn = computeFunctionalScore({ flexion: r.metrics.flexion, sit_to_stand: r.metrics.sit_to_stand, gait: r.metrics.gait });
  return fn
    ? `<p class="meta"><b>${esc(S['func.score'] ?? 'Functional score')}:</b> ${fn.percent}%
       <span class="band ${fn.band}" style="font-size:11px;padding:2px 8px">${esc(S['risk.' + fn.band] ?? fn.band)}</span></p>`
    : '';
})()}
${rows || `<p class="muted">${esc(S['report.noCaptures'] ?? 'No movement captures recorded.')}</p>`}
</section>

<section class="sec">
<h2>${esc(S['report.radiograph'] ?? 'Radiograph')}</h2>
<div class="imgs">${imgs}</div>
${kl}
</section>

<section class="sec">
<h2>${esc(S['guide.title'] ?? 'Preventive guidance')}</h2>
<p class="muted">${esc(S['guide.body'] ?? '')}</p>
</section>

<div class="disc">${esc(S['disclaimer'] ?? 'Screening tool. Not a diagnostic device.')}</div>`;
}

export function buildReportHtml(r: ReportInput): string {
  return `<!doctype html>
<html lang="${r.locale}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>${esc(r.strings['report.title'] ?? 'OA Screening Summary')} — ${esc(r.patient.id)}</title>
<style>body{margin:0;background:#fff}${REPORT_CSS}
@media(max-width:800px){.oa-report{width:100%}}
@media print{.oa-report{width:100%;padding:0}}</style></head>
<body><div class="oa-report" lang="${r.locale}">${reportBody(r)}</div></body></html>`;
}

export function downloadHtml(filename: string, html: string): void {
  const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

export function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result as string);
    fr.onerror = reject;
    fr.readAsDataURL(blob);
  });
}

/**
 * Assemble a `ReportInput` from the current session — the single source used
 * by both the X-ray page's "Export PDF" and the final Result screen.
 */
export async function buildReportInput(args: {
  patient: Patient | null;
  encounterId: string | null;
  locale: string;
  captures: Record<string, { metrics: Record<string, number>; angles: number[]; quality: number }>;
  klGrade: number | null;
  klDist: number[] | null;
  riskBand: RiskBand;
  /** the 3-way combined score, computed by the caller via computeOverallRisk() */
  overallPercent?: number | null;
  xrayBlob: Blob | null;
  heatmapDataUrl: string | null;
  clinical?: ClinicalAssessment | null;
  strings: Record<string, string>;
}): Promise<ReportInput | null> {
  if (!args.patient) return null;
  const metrics: Record<string, Record<string, number>> = {};
  const quality: Record<string, number> = {};
  const angles: Record<string, number[]> = {};
  for (const [k, v] of Object.entries(args.captures)) {
    metrics[k] = v.metrics;
    quality[k] = v.quality;
    angles[k] = v.angles;
  }
  return {
    patient: args.patient,
    encounterId: args.encounterId ?? '',
    locale: args.locale,
    metrics, quality, angles,
    klGrade: args.klGrade,
    klDist: args.klDist,
    riskBand: args.riskBand,
    overallPercent: args.overallPercent ?? null,
    xrayDataUrl: args.xrayBlob ? await blobToDataUrl(args.xrayBlob) : null,
    heatmapDataUrl: args.heatmapDataUrl,
    clinical: args.clinical ?? null,
    strings: args.strings,
  };
}
