/**
 * PDF export. Split into its own module and loaded on demand — jsPDF plus
 * html2canvas are ~600 kB, and a kiosk should not pay that on first paint
 * for a button most sessions press once at the very end.
 *
 * The page is rasterised rather than typeset with jsPDF text calls because
 * jsPDF does no complex text shaping: Devanagari written through its text
 * API comes out with broken conjuncts and misplaced matras even after the
 * font is embedded. Rasterising keeps the shaping the browser already did.
 */
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import { REPORT_CSS, reportBody, type ReportInput } from './report';

export async function downloadPdf(r: ReportInput, filename: string): Promise<void> {
  const style = document.createElement('style');
  style.textContent = REPORT_CSS;

  const host = document.createElement('div');
  host.setAttribute('lang', r.locale);
  host.className = 'oa-report';
  host.innerHTML = reportBody(r);
  Object.assign(host.style, {
    position: 'fixed', left: '-10000px', top: '0',
    width: '760px', background: '#ffffff', zIndex: '-1',
  } as Partial<CSSStyleDeclaration>);

  document.head.appendChild(style);
  document.body.appendChild(host);

  try {
    // Images are data URIs, but html2canvas still needs them decoded first.
    await Promise.all(
      Array.from(host.querySelectorAll('img')).map((img) =>
        img.complete
          ? img.decode().catch(() => undefined)
          : new Promise<void>((res) => { img.onload = () => res(); img.onerror = () => res(); }),
      ),
    );
    await new Promise((res) => requestAnimationFrame(() => res(null)));

    const canvas = await html2canvas(host, {
      scale: 2, backgroundColor: '#ffffff', logging: false,
      windowWidth: 760, width: 760,
    });

    const pdf = new jsPDF({ unit: 'pt', format: 'a4', compress: true });
    const pw = pdf.internal.pageSize.getWidth();
    const ph = pdf.internal.pageSize.getHeight();

    // Scale factors: CSS px -> PDF pt, and CSS px -> raster px.
    const ptPerCss = pw / 760;
    const pxPerCss = canvas.width / 760;
    const pageCssH = ph / ptPerCss;

    // Break pages at element boundaries rather than slicing blindly, so a
    // page break never lands in the middle of a radiograph or a table.
    const kids = Array.from(host.children) as HTMLElement[];
    const totalCssH = host.scrollHeight;
    const breaks: number[] = [0];
    let pageTop = 0;
    for (const kid of kids) {
      const bottom = kid.offsetTop + kid.offsetHeight;
      if (bottom - pageTop > pageCssH) {
        // A single element taller than a page has to be sliced anyway.
        const top = kid.offsetTop > pageTop ? kid.offsetTop : pageTop + pageCssH;
        pageTop = Math.min(top, totalCssH);
        breaks.push(pageTop);
      }
    }
    breaks.push(totalCssH);

    const slice = document.createElement('canvas');
    const sctx = slice.getContext('2d');

    for (let i = 0; i < breaks.length - 1; i++) {
      const cssTop = breaks[i];
      const cssH = breaks[i + 1] - cssTop;
      if (cssH <= 1) continue;

      slice.width = canvas.width;
      slice.height = Math.round(cssH * pxPerCss);
      if (!sctx) break;
      sctx.fillStyle = '#ffffff';
      sctx.fillRect(0, 0, slice.width, slice.height);
      sctx.drawImage(
        canvas,
        0, Math.round(cssTop * pxPerCss), canvas.width, slice.height,
        0, 0, canvas.width, slice.height,
      );

      if (i > 0) pdf.addPage();
      pdf.addImage(
        slice.toDataURL('image/jpeg', 0.92), 'JPEG',
        0, 0, pw, cssH * ptPerCss, undefined, 'FAST',
      );
    }

    pdf.save(filename);
  } finally {
    host.remove();
    style.remove();
  }
}
