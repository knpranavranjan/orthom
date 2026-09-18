/**
 * X-ray grading boundary. One interface, swappable backends.
 *
 * `HttpBackend` talks to the AETHER-OA inference API (the OAF project's
 * `scripts/serve_api.py`): a knee radiograph is POSTed as multipart form-data,
 * the MobileNetV2 ONNX model grades it, a grad-free Class Activation Map is
 * rendered server-side, and the JSON response carries the KL distribution plus
 * the heatmap as a `data:` URL.
 *
 * Set `VITE_XRAY_API` (e.g. http://localhost:8000) to use it. With no such env
 * var the deterministic `FixtureBackend` runs so the flow is demoable offline.
 */

export interface KLResult {
  /** Kellgren-Lawrence grade 0..4 */
  grade: number;
  /** probability per grade, length 5, sums to ~1 (temperature-calibrated) */
  dist: number[];
  /** Grad-CAM style overlay as a data URL, or null if unavailable */
  heatmap: string | null;
  backend: string;

  // --- optional clinical detail, populated by the real model backend ---
  /** e.g. "KL2 - Minimal OA" */
  klName?: string;
  /** max calibrated probability (0..1) */
  confidence?: number;
  /** ordinal expectation Σ c·p_c */
  expectedGrade?: number;
  /** binary radiographic-OA screen (KL≥2) */
  oaScreen?: { label: string; pOA: number; pNoOA: number };
  /** 3-class band: Normal / Early / Advanced */
  threeClass?: { label: string; probs: Record<string, number> };
  /** true when confidence is below the model's validation-tuned threshold */
  abstain?: boolean;
  abstainThreshold?: number;
  /** grade-banded recommendation text */
  recommendation?: string;
  /** grade-typical radiographic findings */
  findings?: string[];
  /** one-line interpretation incl. where the model looked */
  interpretation?: string;
  /** server inference time, ms */
  latencyMs?: number;
  /** model id, e.g. "mobilenet_v2_oa" */
  model?: string;
}

export interface InferenceBackend {
  readonly name: string;
  grade(image: Blob): Promise<KLResult>;
}

/**
 * Thrown by `grade()` when the inference API's input-validation gate rejects
 * the upload as not-a-knee-X-ray. The model was NOT run; there is no grade or
 * heatmap. `userMessage` is safe to show to the operator as-is.
 */
export class XrayRejectedError extends Error {
  readonly userMessage: string;
  readonly reason: string;
  readonly detail: unknown;
  constructor(message: string, reason: string, detail?: unknown) {
    super(message);
    this.name = 'XrayRejectedError';
    this.userMessage = message;
    this.reason = reason;
    this.detail = detail;
  }
}

/**
 * Live model. POSTs the radiograph to the AETHER-OA FastAPI service and maps
 * its response onto `KLResult`. The heatmap comes back ready to drop into an
 * <img src>, so no client-side rendering.
 */
export class HttpBackend implements InferenceBackend {
  readonly name: string;
  private readonly base: string;

  constructor(baseUrl: string) {
    this.base = baseUrl.replace(/\/+$/, '');
    this.name = `api ${this.base}`;
  }

  async grade(image: Blob): Promise<KLResult> {
    const fd = new FormData();
    const filename = image instanceof File && image.name ? image.name : 'xray.png';
    fd.append('image', image, filename);
    fd.append('explain', 'true');

    const res = await fetch(`${this.base}/api/xray/grade`, { method: 'POST', body: fd });
    if (!res.ok) {
      const detail = await res.text().catch(() => '');
      throw new Error(`inference API ${res.status} ${res.statusText}${detail ? ` — ${detail}` : ''}`);
    }
    const j = await res.json();

    // Input-validation gate rejected the upload — no grade, no heatmap.
    if (j && j.accepted === false) {
      throw new XrayRejectedError(
        j.message || 'This does not look like a knee X-ray. Please upload a knee radiograph.',
        j.reason || 'not_xray',
        j.input_check,
      );
    }

    return {
      grade: j.grade,
      dist: j.dist,
      heatmap: j.heatmap ?? null,
      backend: this.name,
      klName: j.kl_name,
      confidence: j.confidence,
      expectedGrade: j.expected_grade,
      oaScreen: j.oa_screen
        ? { label: j.oa_screen.label, pOA: j.oa_screen.p_oa, pNoOA: j.oa_screen.p_no_oa }
        : undefined,
      threeClass: j.three_class,
      abstain: j.abstain,
      abstainThreshold: j.abstain_threshold,
      recommendation: j.recommendation,
      findings: j.findings,
      interpretation: j.interpretation,
      latencyMs: j.latency_ms,
      model: j.model,
    };
  }
}

/**
 * Deterministic stand-in. Produces a *borderline* distribution on purpose:
 * a 0.52 / 0.41 split between adjacent grades is the honest, interesting
 * case to demonstrate, and the one where a screening tool's advice matters.
 */
export class FixtureBackend implements InferenceBackend {
  readonly name = 'fixture';
  async grade(image: Blob): Promise<KLResult> {
    await new Promise((r) => setTimeout(r, 900));
    const seed = (image.size % 5);
    const top = 2 + (seed % 2);          // grade 2 or 3
    const dist = [0, 0, 0, 0, 0];
    dist[top] = 0.52;
    dist[top - 1] = 0.41;
    dist[top + 1 <= 4 ? top + 1 : 0] = 0.04;
    dist[0] += 0.02; dist[4] += 0.01;
    const sum = dist.reduce((a, b) => a + b, 0);
    const norm = dist.map((d) => d / sum);
    return {
      grade: top,
      dist: norm,
      heatmap: await syntheticHeatmap(image),
      backend: this.name,
      confidence: norm[top],
      oaScreen: { label: top >= 2 ? 'OA likely' : 'No / doubtful OA',
                  pOA: norm[2] + norm[3] + norm[4], pNoOA: norm[0] + norm[1] },
    };
  }
}

/**
 * Placeholder overlay so screen 8 is demonstrable before the model lands.
 * Draws a soft warm focus over the medial joint space region.
 * Replace wholesale with the real Grad-CAM output.
 *
 * BUG FIXED 2026-09-18: this used to set
 * `ctx.globalCompositeOperation = 'lighter'` (additive blending) before
 * filling the gradient. Additive blending clips to white on top of
 * already-bright pixels — and most of a real knee X-ray IS bright white —
 * so the overlay was nearly invisible on real radiographs: verified by
 * rendering it standalone and inspecting the output, which showed only a
 * faint pastel smudge, easily mistaken for "the heatmap tile is just
 * showing the original image again" (a real user report). Standard alpha
 * compositing (the canvas default, no globalCompositeOperation override)
 * blends the overlay color toward the base pixel rather than adding to it,
 * so it stays visible regardless of how bright the underlying image is.
 */
async function syntheticHeatmap(image: Blob): Promise<string | null> {
  try {
    const bmp = await createImageBitmap(image);
    const w = 320, h = Math.round((bmp.height / bmp.width) * 320) || 320;
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const ctx = c.getContext('2d');
    if (!ctx) return null;
    ctx.drawImage(bmp, 0, 0, w, h);
    const cx = w * 0.46, cy = h * 0.52, r = Math.min(w, h) * 0.32;
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
    g.addColorStop(0.0, 'rgba(200, 20, 20, 0.55)');
    g.addColorStop(0.25, 'rgba(230, 120, 20, 0.50)');
    g.addColorStop(0.5, 'rgba(230, 210, 20, 0.42)');
    g.addColorStop(0.75, 'rgba(40, 180, 120, 0.30)');
    g.addColorStop(1.0, 'rgba(30, 90, 200, 0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);
    bmp.close?.();
    return c.toDataURL('image/jpeg', 0.85);
  } catch {
    return null;
  }
}

const API_URL = (import.meta.env.VITE_XRAY_API ?? '').trim();

let backend: InferenceBackend = API_URL ? new HttpBackend(API_URL) : new FixtureBackend();
export function getInference(): InferenceBackend { return backend; }
export function setInference(b: InferenceBackend): void { backend = b; }
/** True when a real inference server is configured (vs the offline fixture). */
export const hasLiveInference = Boolean(API_URL);
